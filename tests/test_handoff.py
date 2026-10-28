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

from agent_history_mcp.graph import HistoryGraphIndex
from agent_history_mcp.handoff import handoff_brief, latest_checkpoint, save_checkpoint

from fixtures import all_adapters, write_claude_session, write_codex_session

CODEX_STEPS = [
    ("user", "Migrate the billing service to the new Redis client"),
    ("assistant", "Starting with the client wrapper."),
    ("edit", "{p}/billing/redis_client.py"),
    ("bash", "pytest tests/test_billing.py", "ConnectionError: redis timeout", 1),
    ("bash", "pytest tests/test_billing.py", "ConnectionError: redis timeout", 1),
    ("write", "{p}/billing/config.py"),
    ("todo", [("Wrap redis client", "completed"), ("Fix billing tests", "in_progress"), ("Update README", "pending")]),
    ("assistant", "Tests still time out against Redis; next step is the docker compose service."),
]


def _steps(project: str) -> list[tuple]:
    out = []
    for step in CODEX_STEPS:
        out.append(tuple(s.format(p=project) if isinstance(s, str) else s for s in step))
    return out


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.project = str(self.root / "repo")
        Path(self.project).mkdir()
        self.env = mock.patch.dict(os.environ, {"AGENT_HISTORY_HOME": str(self.root / "home")})
        self.env.start()
        write_codex_session(self.root, "rollout-2026-05-08-billing", self.project, _steps(self.project),
                            start="2026-05-08T09:00:00Z")
        write_claude_session(self.root, "claude-old", self.project,
                             [("user", "Set up the repo"), ("bash", "git init", "ok", 0)],
                             start="2026-05-01T09:00:00Z", title="Repo setup")
        self.index = HistoryGraphIndex(self.root / "graph.sqlite")
        self.index.refresh(adapters=all_adapters(self.root))

    def tearDown(self):
        self.index.close()
        self.env.stop()
        self.tmp.cleanup()

    def test_brief_reports_where_work_stopped(self):
        brief = handoff_brief(self.index, self.project, sessions=1, include_git=False)
        self.assertIn("codex", brief)
        self.assertIn("Migrate the billing service", brief)
        self.assertIn("- [ ] Fix billing tests", brief)
        self.assertIn("- [ ] Update README", brief)
        self.assertNotIn("Wrap redis client", brief)
        self.assertIn("`billing/redis_client.py`", brief)
        self.assertIn("Unresolved errors", brief)
        self.assertIn("failed 2×", brief)
        self.assertIn("Continue with the first open todo: Fix billing tests", brief)

    def test_exclude_source_and_session_count(self):
        brief = handoff_brief(self.index, self.project, sessions=2, exclude_source="codex", include_git=False)
        self.assertIn("Repo setup", brief)
        self.assertNotIn("billing", brief)
        other = handoff_brief(self.index, str(self.root / "elsewhere"), include_git=False)
        self.assertTrue(other.startswith("No indexed sessions"))

    def test_checkpoint_round_trip_and_apply(self):
        result = save_checkpoint(self.index, "Paused: need Redis running locally", self.project)
        self.assertIn("Checkpoint saved", result)
        self.assertFalse((Path(self.project) / ".agent-context" / "HANDOFF.md").exists())
        self.assertIsNotNone(latest_checkpoint(self.project))
        brief = handoff_brief(self.index, self.project, include_git=False)
        self.assertIn("Latest checkpoint", brief)
        self.assertIn("need Redis running locally", brief)

        save_checkpoint(self.index, "Second note", self.project, apply=True)
        handoff_file = Path(self.project) / ".agent-context" / "HANDOFF.md"
        self.assertTrue(handoff_file.exists())
        self.assertIn("Second note", handoff_file.read_text(encoding="utf-8"))


class SessionStartHookTests(unittest.TestCase):
    def test_hook_emits_additional_context_for_other_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = str(root / "repo")
            Path(project).mkdir()
            write_codex_session(root, "rollout-2026-05-08-billing", project, _steps(project))
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "AGENT_HISTORY_HOME": str(root / "home"),
                "AGENT_HISTORY_GRAPH_DB": str(root / "graph.sqlite"),
                "AGENT_HISTORY_SOURCES": "claude,codex",
                "CLAUDE_PATH": str(root / ".claude"),
                "CODEX_PATH": str(root / ".codex"),
            }
            out = subprocess.run(
                [sys.executable, "-m", "agent_history_mcp.cli", "hook", "session-start"],
                input=json.dumps({"cwd": project, "source": "startup"}),
                capture_output=True, text=True, env=env, timeout=60, check=True,
            )
            payload = json.loads(out.stdout)
            context = payload["hookSpecificOutput"]["additionalContext"]
            self.assertEqual(payload["hookSpecificOutput"]["hookEventName"], "SessionStart")
            self.assertIn("Fix billing tests", context)

            empty = subprocess.run(
                [sys.executable, "-m", "agent_history_mcp.cli", "hook", "session-start"],
                input=json.dumps({"cwd": str(root / "nothing-here")}),
                capture_output=True, text=True, env=env, timeout=60, check=True,
            )
            self.assertEqual(empty.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
