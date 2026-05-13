import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_history_mcp.graph import HistoryGraphIndex
from agent_history_mcp.skills import format_skill_candidates, suggest_skill_candidates

from test_graph import _append_jsonl, _codex_msg


def _write_codex_session(root: Path, day: str, name: str, messages: list[str]) -> None:
    session_dir = root / ".codex" / "sessions" / "2026" / "05" / day
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{name}.jsonl"
    for idx, text in enumerate(messages):
        role = "user" if idx % 2 == 0 else "assistant"
        _append_jsonl(path, _codex_msg(role, text, f"2026-05-{day}T10:{idx:02d}:00Z"))


class SkillSuggestionTests(unittest.TestCase):
    def test_suggests_repeated_actionable_skill_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_codex_session(
                root,
                "07",
                "rollout-azure-one",
                [
                    "Azure deployment failed for App Service. Run `az webapp log tail` and inspect 502 timeout.",
                    "Use az webapp config appsettings set and check Azure App Service startup logs.",
                ],
            )
            _write_codex_session(
                root,
                "08",
                "rollout-azure-two",
                [
                    "Debug Azure App Service deployment failure with `az webapp deployment list-publishing-profiles`.",
                    "Check appsettings and restart the webapp after fixing startup timeout.",
                ],
            )
            _write_codex_session(
                root,
                "09",
                "rollout-random-oneoff",
                [
                    "Explain a one-off color palette decision for a marketing page.",
                    "Use a quieter visual treatment for this one page.",
                ],
            )

            index = HistoryGraphIndex(root / "graph.sqlite")
            try:
                index.refresh(root / ".codex", None)
                candidates = suggest_skill_candidates(index, ["codex"], max_candidates=5, min_sessions=2)

                self.assertTrue(candidates)
                rendered = format_skill_candidates(candidates)
                self.assertIn("Confidence:", rendered)
                self.assertIn("Evidence:", rendered)
                self.assertTrue(
                    any(
                        "azure" in candidate.slug
                        or any("azure" in term.lower() for term in candidate.recurring_terms)
                        for candidate in candidates
                    )
                )
                self.assertFalse(any("palette" in candidate.slug for candidate in candidates))
            finally:
                index.close()

    def test_candidate_ids_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_codex_session(
                root,
                "07",
                "rollout-pytest-one",
                [
                    "Pytest fixture failure in tests/test_api.py. Run `pytest tests/test_api.py`.",
                    "Fix fixture scope and rerun pytest for the API tests.",
                ],
            )
            _write_codex_session(
                root,
                "08",
                "rollout-pytest-two",
                [
                    "Another pytest fixture issue in tests/test_api.py with ValueError: fixture not found.",
                    "Use pytest -k api and update the shared fixture.",
                ],
            )

            index = HistoryGraphIndex(root / "graph.sqlite")
            try:
                index.refresh(root / ".codex", None)
                first = suggest_skill_candidates(index, ["codex"], max_candidates=3, min_sessions=2)
                second = suggest_skill_candidates(index, ["codex"], max_candidates=3, min_sessions=2)
                self.assertEqual([c.candidate_id for c in first], [c.candidate_id for c in second])
            finally:
                index.close()

    def test_source_filter_limits_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_codex_session(
                root,
                "07",
                "rollout-docker-one",
                [
                    "Docker compose timeout. Run `docker compose logs api`.",
                    "Fix service healthcheck and rerun docker compose up.",
                ],
            )
            _write_codex_session(
                root,
                "08",
                "rollout-docker-two",
                [
                    "Docker compose startup failed in api service.",
                    "Inspect docker compose logs and update the healthcheck.",
                ],
            )

            index = HistoryGraphIndex(root / "graph.sqlite")
            try:
                index.refresh(root / ".codex", None)
                candidates = suggest_skill_candidates(index, ["claude"], max_candidates=3, min_sessions=1)
                self.assertEqual(candidates, [])
            finally:
                index.close()


if __name__ == "__main__":
    unittest.main()
