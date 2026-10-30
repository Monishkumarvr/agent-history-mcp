import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_history_mcp import compaction

from fixtures import _append, write_claude_session, write_codex_session

LONG_ASK = "Keep the public API stable. " + "Background detail. " * 60 + "Never touch migrations/0007_billing.py."


class CompactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.project = str(self.root / "repo")
        Path(self.project).mkdir()
        self.transcript = write_claude_session(self.root, "sess-abc123", self.project, [
            ("user", LONG_ASK),
            ("bash", "pytest -q", "FAILED tests/test_api.py::test_refund - KeyError: 'currency'", 1),
            ("edit", f"{self.project}/api/refund.py"),
            ("todo", [("Fix refund currency", "completed"), ("Update changelog", "pending")]),
            ("assistant", "Refund now defaults the currency; changelog is next."),
        ])
        _append(self.transcript, {"type": "system", "subtype": "compact_boundary", "timestamp": "2026-05-07T10:30:00.000Z"})
        self.env = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "AGENT_HISTORY_HOME": str(self.root / "home"),
            "CLAUDE_PATH": str(self.root / ".claude"),
            "CODEX_PATH": str(self.root / ".codex"),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _hook(self, name: str, payload: dict, *extra: str) -> str:
        return subprocess.run(
            [sys.executable, "-m", "agent_history_mcp.cli", "hook", name, *extra],
            input=json.dumps(payload), capture_output=True, text=True, env=self.env, timeout=60, check=True,
        ).stdout

    def test_pre_compact_checkpoint_is_reinjected_after_compaction(self):
        out = self._hook("pre-compact", {"session_id": "sess-abc123", "transcript_path": str(self.transcript),
                                         "cwd": self.project, "trigger": "auto"})
        self.assertEqual(out.strip(), "")
        context = json.loads(self._hook("session-start", {"session_id": "sess-abc123", "source": "compact"}))
        text = context["hookSpecificOutput"]["additionalContext"]
        self.assertIn("Keep the public API stable", text)
        self.assertIn("Update changelog", text)
        self.assertIn("refund.py", text)
        self.assertIn("1 earlier compaction", text)
        self.assertIn('session_recall(session_id="sess-abc123"', text)

    def test_hooks_stay_silent_without_a_checkpoint(self):
        self.assertEqual(self._hook("pre-compact", {"session_id": "x"}).strip(), "")
        self.assertEqual(self._hook("session-start", {"session_id": "missing", "source": "compact"}).strip(), "")

    def test_recall_finds_dropped_text_and_returns_messages_in_full(self):
        path = compaction.find_transcript("sess-abc", "claude", self.root / ".claude")
        self.assertEqual(path, self.transcript)
        session = compaction.load_session(path, "claude")
        hits = compaction.recall(session, "0007_billing migrations")
        self.assertIn("[m0]", hits)
        self.assertIn("0007_billing.py", hits)
        self.assertIn("KeyError", compaction.recall(session, "currency KeyError"))
        full = compaction.recall(session, message=0)
        self.assertIn(LONG_ASK, full)
        self.assertIn("between 0 and", compaction.recall(session, message=99))
        self.assertIsNone(compaction.find_transcript("../*", "claude", self.root / ".claude"))

    def test_codex_session_checkpoint(self):
        uuid = "019a0000-aaaa-bbbb-cccc-000000000001"
        write_codex_session(self.root, f"rollout-2026-05-08T09-00-00-{uuid}", self.project,
                            [("user", "Port the exporter to parquet"), ("bash", "make test", "ok", 0)])
        path = compaction.find_transcript(uuid, "codex", self.root / ".codex")
        self.assertIsNotNone(path)
        with mock.patch.dict(os.environ, {"AGENT_HISTORY_HOME": self.env["AGENT_HISTORY_HOME"]}):
            compaction.write_checkpoint(path, uuid, "codex", "auto")
            self.assertIn("Port the exporter to parquet", compaction.read_checkpoint("codex", uuid))


if __name__ == "__main__":
    unittest.main()
