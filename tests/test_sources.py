import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_history_mcp.graph import HistoryGraphIndex
from agent_history_mcp.model import canonical_tool, command_signature, detect_user_signal
from agent_history_mcp.sources import resolve_sources

from fixtures import (
    all_adapters,
    write_claude_session,
    write_codex_session,
    write_cursor_store,
    write_gemini_session,
    write_opencode_db,
)

STEPS = [
    ("user", "Fix the failing pytest run in api"),
    ("assistant", "Running the tests first."),
    ("bash", "cd /work/api && pytest tests/test_api.py -q", "1 failed", 1),
    ("edit", "/work/api/src/api.py"),
    ("bash", "pytest tests/test_api.py -q", "3 passed", 0),
    ("todo", [("Fix api test", "completed"), ("Update docs", "pending")]),
    ("assistant", "Tests pass now; docs still need an update."),
]


def _write_all(root: Path, cwd: str = "/work/api") -> None:
    write_claude_session(root, "claude-1", cwd, STEPS, title="Fix api tests")
    write_codex_session(root, "rollout-2026-05-07-codex-1", cwd, STEPS)
    write_gemini_session(root, "gem-1", cwd, STEPS)
    write_opencode_db(root, [("oc-1", cwd, STEPS)])
    write_cursor_store(root, [("cur-1", cwd, STEPS)])


class SourceAdapterTests(unittest.TestCase):
    def test_every_adapter_emits_unified_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_all(root)
            for adapter in all_adapters(root):
                with self.subTest(source=adapter.name):
                    units = adapter.discover()
                    self.assertEqual(len(units), 1, adapter.name)
                    session = adapter.parse(units[0])
                    self.assertIsNotNone(session)
                    self.assertEqual(session["source"], adapter.name)
                    self.assertEqual(session["cwd"], "/work/api")
                    self.assertTrue(session["messages"])
                    self.assertGreater(session["input_tokens"], 0)

                    calls = [e for e in session["events"] if e.kind == "tool_call"]
                    shell = [e for e in calls if e.canon_tool == "shell"]
                    self.assertEqual([e.cmd_sig for e in shell], ["pytest", "pytest"])
                    edits = [e for e in calls if e.canon_tool == "edit"]
                    self.assertEqual(edits[0].paths, ["/work/api/src/api.py"])
                    plan = [e for e in calls if e.canon_tool == "plan"]
                    self.assertEqual(plan[-1].meta["todos"][1]["status"], "pending")

                    results = [e for e in session["events"] if e.kind == "tool_result" and e.canon_tool == "shell"]
                    self.assertEqual([e.is_error for e in results], [True, False])

    def test_graph_indexes_all_sources_incrementally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_all(root)
            index = HistoryGraphIndex(root / "graph.sqlite")
            try:
                first = index.refresh(adapters=all_adapters(root))
                self.assertEqual(first.sessions_indexed, 5)
                self.assertEqual(set(first.per_source), {"claude", "codex", "gemini", "opencode", "cursor"})
                again = index.refresh(adapters=all_adapters(root))
                self.assertEqual(again.files_unchanged, 5)
                self.assertEqual(again.sessions_indexed, 0)

                rows = index.session_rows(project="/work/api")
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(row["n_tool_calls"] >= 4 for row in rows))
                self.assertTrue(all(row["n_errors"] == 1 for row in rows))
                self.assertEqual(len(index.load_sessions(["gemini", "cursor"])), 2)

                hits = index.search("pytest tests/test_api.py", ["opencode"], max_results=3)
                self.assertTrue(hits)
            finally:
                index.close()

    def test_unknown_or_broken_stores_fail_soft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "opencode").mkdir()
            sqlite3.connect(root / "opencode" / "opencode.db").execute("CREATE TABLE other (x)").connection.close()
            gemini_chats = root / ".gemini" / "tmp" / "p" / "chats"
            gemini_chats.mkdir(parents=True)
            (gemini_chats / "session-bad.jsonl").write_text("not json\n{\"$set\": {}}\n", encoding="utf-8")
            index = HistoryGraphIndex(root / "graph.sqlite")
            try:
                stats = index.refresh(adapters=all_adapters(root))
                self.assertEqual(stats.sessions_indexed, 0)
                self.assertTrue(any("opencode" in d for d in stats.diagnostics))
            finally:
                index.close()

    def test_missing_source_is_not_treated_as_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_claude_session(root, "claude-1", "/work/api", STEPS)
            index = HistoryGraphIndex(root / "graph.sqlite")
            try:
                index.refresh(adapters=all_adapters(root))
                stats = index.refresh(adapters=[a for a in all_adapters(root) if a.name != "claude"])
                self.assertEqual(stats.files_removed, 0)
                self.assertEqual(len(index.session_rows()), 1)
            finally:
                index.close()


class ModelHelperTests(unittest.TestCase):
    def test_command_signature(self):
        self.assertEqual(command_signature("cd /x && git commit -m 'msg'"), "git commit")
        self.assertEqual(command_signature("FOO=1 sudo npm run build -- --prod"), "npm run build")
        self.assertEqual(command_signature("python -m pytest -q"), "python -m pytest")
        self.assertEqual(command_signature("/usr/bin/ls -la | head"), "ls")

    def test_canonical_tool(self):
        self.assertEqual(canonical_tool("Bash"), "shell")
        self.assertEqual(canonical_tool("apply_patch"), "edit")
        self.assertEqual(canonical_tool("mcp__github__create_issue"), "mcp:github")
        self.assertEqual(canonical_tool("TodoWrite"), "plan")

    def test_user_signals(self):
        self.assertEqual(detect_user_signal("No, don't use npm here", True), "correction")
        self.assertIsNone(detect_user_signal("No worries", False))
        self.assertEqual(detect_user_signal("Always run ruff before committing", False), "preference")
        self.assertEqual(detect_user_signal("[Request interrupted by user]", True), "interrupt")

    def test_resolve_sources(self):
        self.assertEqual(resolve_sources("all")[:2], ["claude", "codex"])
        self.assertEqual(resolve_sources(["codex", "bogus"]), ["codex"])
        self.assertEqual(resolve_sources("gemini,cursor"), ["gemini", "cursor"])


if __name__ == "__main__":
    unittest.main()
