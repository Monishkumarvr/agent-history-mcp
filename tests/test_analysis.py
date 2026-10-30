import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_history_mcp.analysis import (
    analyze_session,
    analyze_sessions,
    load_sessions,
    suggest_instruction_updates,
)
from agent_history_mcp.graph import HistoryGraphIndex

from fixtures import all_adapters, write_claude_session, write_codex_session


def _session(source: str, name: str, messages: list[dict], events: list[dict] | None = None) -> dict:
    return {
        "session_id": name,
        "session_title": name,
        "session_date": "2026-05-10",
        "source": source,
        "messages": messages,
        "events": events or [],
    }


def _msg(role: str, text: str) -> dict:
    return {
        "role": role,
        "text": text,
        "ts": "2026-05-10 10:00",
        "session_id": "s",
        "session_title": "s",
        "session_date": "2026-05-10",
        "source": "codex",
    }


def _event(event_type: str, **kwargs) -> dict:
    base = {
        "event_type": event_type,
        "ts": "2026-05-10 10:00",
        "source": "codex",
        "session_id": "s",
        "role": "tool" if event_type.startswith("tool") else "assistant",
        "text": "",
        "tool_name": "",
        "command": "",
        "path": "",
        "success": None,
        "tokens": {},
    }
    base.update(kwargs)
    return base


class AnalysisTests(unittest.TestCase):
    def test_indexed_sessions_feed_scorecards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = [
                ("user", "Fix src/app.py"),
                ("read", "/work/app/src/app.py"),
                ("read", "/work/app/src/app.py"),
                ("read", "/work/app/src/app.py"),
                ("bash", "grep -r error src", "src/app.py: error handler", 0),
                ("bash", "pytest tests/test_app.py", "failed", 1),
                ("bash", "pytest tests/test_app.py", "failed", 1),
                ("assistant", "Done."),
            ]
            write_claude_session(root, "c1", "/work/app", steps)
            write_codex_session(root, "rollout-2026-05-08-a", "/work/app", steps, start="2026-05-08T10:00:00Z")
            index = HistoryGraphIndex(root / "graph.sqlite")
            try:
                index.refresh(adapters=all_adapters(root))
                sessions = load_sessions(index, ["claude", "codex"])
            finally:
                index.close()

        self.assertEqual({s["source"] for s in sessions}, {"claude", "codex"})
        for session in sessions:
            card = analyze_session(session)
            self.assertEqual(card.metrics["failed_commands"], 2, session["source"])
            self.assertTrue(any(f.label == "repeated_failed_commands" for f in card.findings))
        claude = next(s for s in sessions if s["source"] == "claude")
        self.assertTrue(any(f.label == "repeated_file_reads" for f in analyze_session(claude).findings))

    def test_repeated_file_reads_flag_context_inefficiency(self):
        events = [
            _event("tool_call", source="claude", tool_name="Read", path="src/app.py"),
            _event("tool_call", source="claude", tool_name="Read", path="src/app.py"),
            _event("tool_call", source="claude", tool_name="Read", path="src/app.py"),
        ]
        card = analyze_session(_session(
            "claude",
            "claude-repeat-read",
            [_msg("user", "Fix the auth handler in src/app.py"), _msg("assistant", "Done and verified.")],
            events,
        ))

        self.assertTrue(any(f.label == "repeated_file_reads" for f in card.findings))
        self.assertLess(card.context_efficiency, 92)

    def test_repeated_failed_codex_commands_flag_command_friction(self):
        events = [
            _event("tool_result", tool_name="shell_command", command="pytest tests/test_api.py", text="failed", success=False),
            _event("tool_result", tool_name="shell_command", command="pytest tests/test_api.py", text="failed again", success=False),
        ]
        card = analyze_session(_session(
            "codex",
            "codex-failed-pytest",
            [_msg("user", "Run the API tests"), _msg("assistant", "The command failed.")],
            events,
        ))

        self.assertTrue(any(f.label == "repeated_failed_commands" for f in card.findings))
        self.assertGreaterEqual(card.metrics["repeated_failed_commands"], 2)

    def test_vague_prompt_with_excessive_exploration_lowers_clarity(self):
        events = [_event("tool_call", tool_name="search", text=f"search {i}") for i in range(7)]
        card = analyze_session(_session(
            "codex",
            "vague-explore",
            [_msg("user", "Fix this"), _msg("assistant", "I searched widely.")],
            events,
        ))

        self.assertTrue(any(f.label == "vague_opener_excessive_exploration" for f in card.findings))
        self.assertLess(card.prompt_clarity, 80)

    def test_correction_loop_produces_rework_warning(self):
        card = analyze_session(_session(
            "claude",
            "correction-loop",
            [
                _msg("user", "Update the package name."),
                _msg("assistant", "Done."),
                _msg("user", "That is wrong, I asked for the module name."),
                _msg("assistant", "Updated."),
                _msg("user", "Still not what I asked, fix the imports too."),
            ],
        ))

        self.assertTrue(any(f.label == "correction_loop" for f in card.findings))
        self.assertLess(card.rework_rate, 90)

    def test_productive_session_scores_higher_than_noisy_session(self):
        productive = analyze_session(_session(
            "codex",
            "productive",
            [
                _msg("user", "Update src/app.py to handle ValueError and run `pytest tests/test_app.py`."),
                _msg("assistant", "Implemented and verified; tests pass."),
            ],
            [_event("tool_result", tool_name="shell_command", command="pytest tests/test_app.py", text="passed", success=True)],
        ))
        noisy = analyze_session(_session(
            "codex",
            "noisy",
            [
                _msg("user", "Fix this"),
                _msg("assistant", "I searched."),
                _msg("user", "Wrong, do not change the API."),
                _msg("assistant", "The command failed."),
            ],
            [
                *[_event("tool_call", tool_name="search", text=f"search {i}") for i in range(8)],
                _event("tool_result", tool_name="shell_command", command="pytest tests/test_app.py", text="failed", success=False),
                _event("tool_result", tool_name="shell_command", command="pytest tests/test_app.py", text="failed again", success=False),
            ],
        ))

        self.assertGreater(productive.overall_score, noisy.overall_score)

    def test_instruction_suggestions_are_evidence_backed(self):
        sessions = []
        for idx in range(2):
            sessions.append(_session(
                "codex",
                f"repeat-failed-{idx}",
                [_msg("user", "Run tests"), _msg("assistant", "Failed.")],
                [
                    _event("tool_result", tool_name="shell_command", command="pytest tests/test_api.py", text="failed", success=False),
                    _event("tool_result", tool_name="shell_command", command="pytest tests/test_api.py", text="failed again", success=False),
                ],
            ))
        cards = analyze_sessions(sessions, days_back=None, sources=["codex"])
        suggestions = suggest_instruction_updates(cards, targets=["AGENTS.md"])

        self.assertTrue(suggestions)
        self.assertEqual(suggestions[0].target, "AGENTS.md")
        self.assertTrue(suggestions[0].evidence)


if __name__ == "__main__":
    unittest.main()
