import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_history_mcp.graph import HistoryGraphIndex
from agent_history_mcp.memory_sync import (
    BLOCK_BEGIN,
    claude_project_dir_name,
    discover_memory_files,
    list_agent_memory,
    suggest_rules,
    sync_memory,
)

from fixtures import all_adapters, write_claude_session, write_codex_session


class MemorySyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.home = root / "home"
        self.project = root / "repo"
        self.project.mkdir()
        (self.project / "CLAUDE.md").write_text(
            "# Project rules\n\n- Use pnpm, never npm, for installs.\n- Run ruff before committing.\n",
            encoding="utf-8",
        )
        (self.project / "AGENTS.md").write_text(
            "# Agents\n\nIntro paragraph that is not a rule\n\n- Run ruff before committing.\n",
            encoding="utf-8",
        )
        memory_dir = self.home / ".claude" / "projects" / claude_project_dir_name(str(self.project)) / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text("- [Feedback](feedback.md) — commit style\n", encoding="utf-8")
        (memory_dir / "feedback.md").write_text(
            "---\nname: commits\ntype: feedback\n---\n- User prefers small, focused commits.\n", encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_discovers_files_and_readers(self):
        files = {f.path.name: f for f in discover_memory_files(str(self.project), self.home)}
        self.assertIn("CLAUDE.md", files)
        self.assertEqual(files["AGENTS.md"].readers, ["codex", "cursor", "opencode"])
        self.assertIn("feedback.md", files)
        self.assertNotIn("MEMORY.md", files)
        self.assertEqual(files["AGENTS.md"].rules, ["Run ruff before committing."])

    def test_claude_import_of_agents_md_counts_as_reading_it(self):
        (self.project / "CLAUDE.md").write_text("@AGENTS.md\n- Use pnpm, never npm, for installs.\n", encoding="utf-8")
        files = {f.path.name: f for f in discover_memory_files(str(self.project), self.home)}
        self.assertIn("claude", files["AGENTS.md"].readers)

    def test_drift_listing(self):
        report = list_agent_memory(None, str(self.project), self.home)
        self.assertIn("Use pnpm, never npm", report)
        self.assertIn("missing for: codex, cursor, opencode", report)
        self.assertNotIn('"Run ruff before committing." — in', report)

    def test_sync_is_dry_run_then_idempotent_apply(self):
        agents_md = self.project / "AGENTS.md"
        original = agents_md.read_text(encoding="utf-8")
        preview = sync_memory(str(self.project), home=self.home)
        self.assertIn("+- Use pnpm, never npm, for installs.", preview)
        self.assertIn("Dry run only", preview)
        self.assertEqual(agents_md.read_text(encoding="utf-8"), original)
        self.assertNotIn("small, focused commits", preview)

        sync_memory(str(self.project), home=self.home, apply=True)
        updated = agents_md.read_text(encoding="utf-8")
        self.assertTrue(updated.startswith(original.rstrip("\n")))
        self.assertIn(BLOCK_BEGIN, updated)
        self.assertIn("- Use pnpm, never npm, for installs.", updated)
        self.assertEqual((self.project / "CLAUDE.md").read_text(encoding="utf-8").count("pnpm"), 1)

        again = sync_memory(str(self.project), home=self.home, apply=True)
        self.assertIn("up to date", again)
        self.assertEqual(agents_md.read_text(encoding="utf-8"), updated)

    def test_explicit_rules_and_targets(self):
        out = sync_memory(str(self.project), rules=["Never push to main directly."], targets=["gemini"], home=self.home, apply=True)
        self.assertIn("GEMINI.md", out)
        self.assertIn("Never push to main directly.", (self.project / "GEMINI.md").read_text(encoding="utf-8"))


class RuleMiningTests(unittest.TestCase):
    def test_repeated_corrections_and_rejections_become_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = str(root / "repo")
            Path(project).mkdir()
            (Path(project) / "CLAUDE.md").write_text("- Use pnpm instead of npm.\n", encoding="utf-8")
            write_claude_session(root, "c1", project, [
                ("user", "Install deps"), ("bash", "npm install", "ok", 0),
                ("user", "No, use pnpm instead of npm"),
                ("rejected", "git push origin main"),
            ], start="2026-05-07T10:00:00Z")
            write_codex_session(root, "rollout-2026-05-08-x", project, [
                ("user", "Add a dependency"), ("bash", "npm add left-pad", "ok", 0),
                ("user", "no - use pnpm not npm here"),
            ], start="2026-05-08T10:00:00Z")
            write_claude_session(root, "c2", project, [
                ("user", "Ship it"), ("rejected", "git push --force"),
            ], start="2026-05-09T10:00:00Z")

            index = HistoryGraphIndex(root / "graph.sqlite")
            try:
                index.refresh(adapters=all_adapters(root))
                candidates = suggest_rules(index, project, days_back=None, min_sessions=2, home=root / "home")
            finally:
                index.close()
            by_kind = {c.kind: c for c in candidates}
            self.assertIn("correction", by_kind)
            self.assertEqual(len(by_kind["correction"].sessions), 2)
            self.assertIn("pnpm", by_kind["correction"].rule.lower())
            self.assertTrue(by_kind["correction"].documented_in)
            self.assertEqual(by_kind["rejected"].rule, "Ask before running `git push`")


if __name__ == "__main__":
    unittest.main()
