"""Tests for codex_mcp/parsers.py"""

import json
from pathlib import Path

import pytest

from codex_mcp.parsers import (
    _fmt_ts,
    _read_jsonl,
    parse_claude_sessions,
    parse_codex_sessions,
)


# ── _fmt_ts ───────────────────────────────────────────────────────────────────

class TestFmtTs:
    def test_valid_utc_timestamp(self):
        result = _fmt_ts("2026-03-10T14:22:00Z")
        # Should be a formatted date-time string, not the raw ISO string
        assert "2026-03-10" in result
        assert "Z" not in result

    def test_empty_string_returns_empty(self):
        assert _fmt_ts("") == ""

    def test_invalid_returns_first_16_chars(self):
        result = _fmt_ts("not-a-date-at-all")
        assert result == "not-a-date-at-al"   # first 16 chars


# ── _read_jsonl ───────────────────────────────────────────────────────────────

class TestReadJsonl:
    def test_nonexistent_path_returns_empty(self, tmp_path):
        result = _read_jsonl(tmp_path / "missing.jsonl")
        assert result == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert _read_jsonl(f) == []

    def test_mixed_valid_invalid_lines(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        f.write_text('{"a": 1}\nNOT JSON\n{"b": 2}\n')
        result = _read_jsonl(f)
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}


# ── parse_claude_sessions ─────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _claude_user_record(text: str, ts="2026-03-10T14:22:00Z"):
    return {
        "type": "user",
        "timestamp": ts,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _claude_assistant_record(text: str, ts="2026-03-10T14:23:00Z"):
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


class TestParseClaudeSessions:
    def _make_project_dir(self, tmp_path: Path) -> Path:
        proj = tmp_path / "projects" / "myproject"
        proj.mkdir(parents=True)
        return proj

    def test_basic_parses_user_and_assistant(self, tmp_path):
        proj = self._make_project_dir(tmp_path)
        _write_jsonl(proj / "session-abc.jsonl", [
            _claude_user_record("hello world"),
            _claude_assistant_record("hi there"),
        ])
        sessions = parse_claude_sessions(tmp_path)
        assert len(sessions) == 1
        msgs = sessions[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["text"] == "hello world"
        assert msgs[1]["role"] == "assistant"

    def test_skips_subagent_files(self, tmp_path):
        proj = self._make_project_dir(tmp_path)
        # main session
        _write_jsonl(proj / "session-main.jsonl", [
            _claude_user_record("main session msg"),
        ])
        # subagent file — should be ignored
        subdir = proj / "session-main" / "subagents"
        subdir.mkdir(parents=True)
        _write_jsonl(subdir / "agent-xyz.jsonl", [
            _claude_user_record("subagent msg"),
        ])
        sessions = parse_claude_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0]["messages"][0]["text"] == "main session msg"

    def test_extracts_ai_title(self, tmp_path):
        proj = self._make_project_dir(tmp_path)
        _write_jsonl(proj / "session-abc.jsonl", [
            {"type": "ai-title", "aiTitle": "Fix CUDA crash"},
            _claude_user_record("what is the fix?"),
        ])
        sessions = parse_claude_sessions(tmp_path)
        assert sessions[0]["session_title"] == "Fix CUDA crash"

    def test_filters_large_xml_messages(self, tmp_path):
        proj = self._make_project_dir(tmp_path)
        big_xml = "<system_context>" + "x" * 2100 + "</system_context>"
        _write_jsonl(proj / "session-abc.jsonl", [
            _claude_user_record("normal message"),
            _claude_assistant_record(big_xml),
        ])
        sessions = parse_claude_sessions(tmp_path)
        msgs = sessions[0]["messages"]
        # Only the normal message should survive
        assert len(msgs) == 1
        assert msgs[0]["text"] == "normal message"

    def test_missing_projects_dir_returns_empty(self, tmp_path):
        # tmp_path has no projects/ subdirectory
        sessions = parse_claude_sessions(tmp_path)
        assert sessions == []


# ── parse_codex_sessions ──────────────────────────────────────────────────────

def _codex_response_item(role: str, text: str, ts="2026-03-10T14:22:00Z"):
    return {
        "type": "response_item",
        "timestamp": ts,
        "payload": {
            "role": role,
            "content": text,
        },
    }


class TestParseCodexSessions:
    def _make_sessions_dir(self, tmp_path: Path, date="2026/03/10") -> Path:
        day_dir = tmp_path / "sessions" / Path(date)
        day_dir.mkdir(parents=True)
        return day_dir

    def test_basic_parses_user_and_assistant(self, tmp_path):
        day_dir = self._make_sessions_dir(tmp_path)
        _write_jsonl(day_dir / "rollout-abc.jsonl", [
            _codex_response_item("user",      "how to fix CUDA"),
            _codex_response_item("assistant", "remove that call"),
        ])
        sessions = parse_codex_sessions(tmp_path)
        assert len(sessions) == 1
        msgs = sessions[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_title_from_index(self, tmp_path):
        day_dir = self._make_sessions_dir(tmp_path)
        fname = "rollout-myid"
        _write_jsonl(day_dir / f"{fname}.jsonl", [
            _codex_response_item("user", "some message"),
        ])
        # write session index
        index_path = tmp_path / "session_index.jsonl"
        index_path.write_text(json.dumps({"id": fname, "thread_name": "My Codex Session"}) + "\n")

        sessions = parse_codex_sessions(tmp_path)
        assert sessions[0]["session_title"] == "My Codex Session"

    def test_skips_system_context_prefix(self, tmp_path):
        day_dir = self._make_sessions_dir(tmp_path)
        _write_jsonl(day_dir / "rollout-abc.jsonl", [
            _codex_response_item("user", "<environment_context>some env data</environment_context>"),
            _codex_response_item("user", "## My request for Codex:\nreal question here"),
        ])
        sessions = parse_codex_sessions(tmp_path)
        msgs = sessions[0]["messages"]
        # system context message is skipped, real question is cleaned up
        assert len(msgs) == 1
        assert msgs[0]["text"] == "real question here"

    def test_missing_sessions_dir_returns_empty(self, tmp_path):
        # tmp_path has no sessions/ subdirectory
        sessions = parse_codex_sessions(tmp_path)
        assert sessions == []
