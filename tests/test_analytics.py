import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_history_mcp.analytics import analyze, format_analysis, render_html
from agent_history_mcp.graph import HistoryGraphIndex

from fixtures import all_adapters, write_claude_session, write_codex_session


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.project = "/work/shop"
        flaky = [
            ("user", "Fix docker compose startup for the shop api"),
            ("bash", "docker compose up -d", "error: port in use", 1),
            ("bash", "docker compose up -d", "error: port in use", 1),
            ("bash", "docker compose up -d", "error: port in use", 1),
            ("user", "No, don't kill other containers"),
            ("edit", "/work/shop/docker-compose.yml"),
            ("bash", "docker compose up -d", "started", 0),
        ]
        write_claude_session(root, "c1", self.project, flaky, start="2026-05-07T10:00:00Z", title="Docker startup")
        write_codex_session(root, "rollout-2026-05-08-a", self.project, flaky, start="2026-05-08T10:00:00Z")
        write_codex_session(root, "rollout-2026-05-09-b", self.project, flaky[:2], start="2026-05-09T10:00:00Z")
        write_claude_session(root, "c2", "/work/other", [("user", "hello"), ("assistant", "hi")], start="2026-05-09T12:00:00Z")
        self.index = HistoryGraphIndex(root / "graph.sqlite")
        self.index.refresh(adapters=all_adapters(root))

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_metrics_and_friction(self):
        result = analyze(self.index, None, self.project, days_back=None)
        self.assertEqual(len(result.sessions), 3)
        by_id = {s.session_id: s for s in result.sessions}
        claude = by_id["c1"]
        self.assertEqual(claude.retry_loops, ["docker compose"])
        self.assertEqual(claude.signals["correction"], 1)
        self.assertEqual(claude.edits, 1)
        self.assertAlmostEqual(claude.active_minutes, 6.0, places=1)
        self.assertEqual(result.commands["docker compose"], 9)
        self.assertEqual(result.command_failures["docker compose"], 7)

        text = format_analysis(result, "source")
        self.assertIn("## By source", text)
        self.assertIn("`docker compose` failed 7×", text)
        self.assertIn("retry loop on `docker compose`", text)

    def test_cross_agent_comparison_and_html(self):
        result = analyze(self.index, None, None, days_back=None)
        self.assertTrue(result.comparison)
        self.assertTrue(all(len(item["per_source"]) >= 2 for item in result.comparison))
        page = render_html(result, [("Skill candidates", [("Docker Workflow", "seen in 3 sessions")])])
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("Same topic, different agents", page)
        self.assertIn("Docker Workflow", page)
        self.assertIn("prefers-color-scheme:dark", page)

    def test_empty_scope(self):
        result = analyze(self.index, ["gemini"], None, days_back=None)
        self.assertIn("No sessions found", format_analysis(result))


if __name__ == "__main__":
    unittest.main()
