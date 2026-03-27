"""Tests for codex_mcp/server.py MCP tools."""

import os
from pathlib import Path
from unittest import mock

import pytest

import codex_mcp.server as srv
from codex_mcp.search import build_fts_index


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session(session_id="s1", source="claude", n_msgs=4):
    msgs = []
    for i in range(n_msgs):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "text": f"msg {i}", "ts": "2026-01-01 10:00"})
    return {
        "session_id":    session_id,
        "session_title": f"Session {session_id}",
        "session_date":  "2026-01-01",
        "source":        source,
        "messages":      msgs,
    }


def patch_cache(monkeypatch, sessions):
    """Pre-populate the module-level _cache to avoid real filesystem access."""
    all_sessions = sessions
    smap = {s["session_id"]: s for s in all_sessions}
    conn = build_fts_index(all_sessions)
    codex   = [s for s in sessions if s["source"] == "codex"]
    claude  = [s for s in sessions if s["source"] == "claude"]
    monkeypatch.setattr(srv, "_cache", {
        "codex":       codex,
        "claude":      claude,
        "conn":        conn,
        "session_map": smap,
    })


# ── _resolve_path ─────────────────────────────────────────────────────────────

class TestResolvePath:
    def test_env_var_set_to_existing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEX_PATH", str(tmp_path))
        result = srv._resolve_path("CODEX_PATH", Path("/nonexistent"))
        assert result == tmp_path

    def test_env_var_not_set_uses_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CODEX_PATH", raising=False)
        result = srv._resolve_path("CODEX_PATH", tmp_path)
        assert result == tmp_path

    def test_path_not_existing_returns_none(self, monkeypatch):
        monkeypatch.delenv("CODEX_PATH", raising=False)
        result = srv._resolve_path("CODEX_PATH", Path("/does/not/exist/42"))
        assert result is None


# ── search_history ────────────────────────────────────────────────────────────

class TestSearchHistory:
    def test_invalid_source_returns_error(self, monkeypatch):
        patch_cache(monkeypatch, [])
        result = srv.search_history("query", sources=["invalid"])
        assert "Invalid sources" in result

    def test_no_sessions_returns_error(self, monkeypatch):
        patch_cache(monkeypatch, [])
        result = srv.search_history("query", sources=["claude"])
        assert "No sessions found" in result

    def test_valid_query_returns_qa_format(self, monkeypatch):
        sessions = [make_session(session_id="abc", source="claude", n_msgs=2)]
        sessions[0]["messages"][0]["text"] = "CUDA pipeline crash"
        sessions[0]["messages"][1]["text"] = "remove the bad call"
        patch_cache(monkeypatch, sessions)
        result = srv.search_history("CUDA", sources=["claude"])
        assert "[YOU ASKED]" in result or "[ANSWER]" in result


# ── list_sessions ─────────────────────────────────────────────────────────────

class TestListSessions:
    def test_all_source_lists_both(self, monkeypatch):
        sessions = [
            make_session(session_id="c1", source="codex"),
            make_session(session_id="c2", source="claude"),
        ]
        patch_cache(monkeypatch, sessions)
        result = srv.list_sessions(source="all")
        assert "codex" in result.lower()
        assert "claude" in result.lower()

    def test_limit_appends_more_message(self, monkeypatch):
        sessions = [make_session(session_id=str(i), source="claude") for i in range(5)]
        patch_cache(monkeypatch, sessions)
        result = srv.list_sessions(source="claude", limit=3)
        assert "more" in result

    def test_no_sessions_returns_not_found(self, monkeypatch):
        patch_cache(monkeypatch, [])
        result = srv.list_sessions(source="claude")
        assert "No sessions found" in result


# ── get_session ───────────────────────────────────────────────────────────────

class TestGetSession:
    def test_invalid_source_returns_error(self, monkeypatch):
        patch_cache(monkeypatch, [])
        result = srv.get_session("any-id", source="invalid")
        assert "must be" in result.lower()

    def test_session_not_found_returns_error(self, monkeypatch):
        patch_cache(monkeypatch, [make_session(session_id="real-id")])
        result = srv.get_session("wrong-id", source="claude")
        assert "not found" in result.lower()

    def test_valid_session_returns_transcript(self, monkeypatch):
        session = make_session(session_id="xyz", source="claude", n_msgs=4)
        patch_cache(monkeypatch, [session])
        result = srv.get_session("xyz", source="claude", max_messages=10)
        assert "Session xyz" in result
        assert "msg 0" in result
