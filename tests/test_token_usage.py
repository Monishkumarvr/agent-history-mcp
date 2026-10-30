import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_history_mcp.graph import HistoryGraphIndex
from agent_history_mcp.sources.claude import parse_claude_session_file
from agent_history_mcp.sources.codex import parse_codex_session_file

from fixtures import _append, all_adapters

USAGE = {
    "input_tokens": 3, "cache_read_input_tokens": 20000, "cache_creation_input_tokens": 1500, "output_tokens": 400,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1500},
    "output_tokens_details": {"thinking_tokens": 120},
}


def _claude_records(path: Path) -> None:
    common = {"cwd": "/repo", "sessionId": "s1"}
    big = "x" * 5000
    _append(path, {**common, "type": "user", "timestamp": "2026-05-07T10:00:01.000Z",
                   "message": {"role": "user", "content": "Read the big file"}})
    # One API response logged twice (one record per content block): counted once.
    for block in ({"type": "text", "text": "Reading."}, {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/repo/a.py"}}):
        _append(path, {**common, "type": "assistant", "timestamp": "2026-05-07T10:00:05.000Z",
                       "message": {"id": "msg_a", "role": "assistant", "model": "claude-test", "content": [block], "usage": USAGE}})
    _append(path, {**common, "type": "user", "timestamp": "2026-05-07T10:00:06.000Z",
                   "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]}})
    _append(path, {**common, "type": "system", "subtype": "compact_boundary", "timestamp": "2026-05-07T10:05:00.000Z",
                   "compactMetadata": {"trigger": "auto", "preTokens": 210000, "postTokens": 22000, "durationMs": 70000}})
    _append(path, {**common, "type": "assistant", "timestamp": "2026-05-07T10:05:30.000Z",
                   "message": {"id": "msg_b", "role": "assistant", "model": "claude-test",
                               "content": [{"type": "text", "text": "Done."}],
                               "usage": {"input_tokens": 22000, "cache_creation_input_tokens": 900, "output_tokens": 10}}})


def _codex_records(path: Path) -> None:
    _append(path, {"timestamp": "2026-05-08T09:00:00.000Z", "type": "session_meta", "payload": {"id": "c1", "cwd": "/repo"}})
    _append(path, {"timestamp": "2026-05-08T09:00:00.000Z", "type": "turn_context", "payload": {"cwd": "/repo", "model": "gpt-test"}})
    _append(path, {"timestamp": "2026-05-08T09:00:01.000Z", "type": "response_item",
                   "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Port it"}]}})
    first = {"input_tokens": 12000, "cached_input_tokens": 9000, "output_tokens": 300, "reasoning_output_tokens": 50}
    for ts, last, total in (
        ("2026-05-08T09:00:05.000Z", first, {"input_tokens": 12000, "output_tokens": 300}),
        ("2026-05-08T09:00:06.000Z", first, {"input_tokens": 12000, "output_tokens": 300}),  # repeat, same call
        ("2026-05-08T09:00:09.000Z", {"input_tokens": 15000, "cached_input_tokens": 12000, "output_tokens": 100},
         {"input_tokens": 27000, "output_tokens": 400}),
    ):
        _append(path, {"timestamp": ts, "type": "event_msg",
                       "payload": {"type": "token_count", "info": {"last_token_usage": last, "total_token_usage": total}}})
    _append(path, {"timestamp": "2026-05-08T09:01:00.000Z", "type": "compacted", "payload": {"message": ""}})


class TokenUsageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.claude = self.root / ".claude" / "projects" / "-repo" / "s1.jsonl"
        self.codex = self.root / ".codex" / "sessions" / "2026" / "05" / "08" / "rollout-2026-05-08-c1.jsonl"
        _claude_records(self.claude)
        _codex_records(self.codex)

    def tearDown(self):
        self.tmp.cleanup()

    def test_claude_calls_output_size_and_compaction(self):
        session = parse_claude_session_file(self.claude)
        calls = session["calls"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], {
            "ts": "2026-05-07T10:00:05Z", "model": "claude-test", "sidechain": False, "input": 3,
            "cache_read": 20000, "cache_write_5m": 0, "cache_write_1h": 1500, "output": 400, "reasoning": 120,
        })
        self.assertEqual(calls[1]["cache_write_5m"], 900)  # no split recorded: the total counts as 5m
        result = next(e for e in session["events"] if e.kind == "tool_result")
        self.assertEqual(result.meta["out_chars"], 5000)
        self.assertFalse(result.is_error)  # a large output is not an error
        compact = next(e for e in session["events"] if e.meta.get("signal") == "compact")
        self.assertEqual(compact.meta, {"signal": "compact", "trigger": "auto", "pre_tokens": 210000,
                                        "post_tokens": 22000, "duration_ms": 70000})

    def test_codex_calls_dedupe_and_compaction_size(self):
        session = parse_codex_session_file(self.codex)
        calls = session["calls"]
        self.assertEqual([(c["input"], c["cache_read"], c["output"], c["reasoning"]) for c in calls],
                         [(3000, 9000, 300, 50), (3000, 12000, 100, 0)])
        self.assertEqual(calls[0]["model"], "gpt-test")
        compact = next(e for e in session["events"] if e.meta.get("signal") == "compact")
        self.assertEqual(compact.meta["pre_tokens"], 15000)

    def test_index_stores_calls(self):
        index = HistoryGraphIndex(self.root / "graph.sqlite")
        try:
            index.refresh(adapters=all_adapters(self.root))
            for source, count in (("claude", 2), ("codex", 2)):
                row = index.session_rows([source])[0]
                self.assertEqual(len(index.calls_for(row["session_key"])), count)
        finally:
            index.close()


if __name__ == "__main__":
    unittest.main()
