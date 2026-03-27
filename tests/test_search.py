"""Tests for codex_mcp/search.py"""

import sqlite3
import sys
import types
import unittest.mock as mock

import pytest

from codex_mcp.search import (
    SearchHit,
    _build_qa_pair,
    build_fts_index,
    format_hits,
    format_session,
    search_fts,
    search_fuzzy,
    search_smart,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_msg(role, text, ts="2026-01-01 10:00"):
    return {"role": role, "text": text, "ts": ts}


def make_session(session_id="s1", source="claude", messages=None):
    if messages is None:
        messages = [
            make_msg("user",      "user question one"),
            make_msg("assistant", "assistant answer one"),
            make_msg("user",      "user question two"),
            make_msg("assistant", "assistant answer two"),
        ]
    return {
        "session_id":    session_id,
        "session_title": f"Session {session_id}",
        "session_date":  "2026-01-01",
        "source":        source,
        "messages":      messages,
    }


def make_hit(question_text="You asked this", answer_text="Here is the answer",
             question_ts="2026-01-01 10:00", answer_ts="2026-01-01 10:01"):
    return SearchHit(
        session_id="s1",
        session_title="Test Session",
        session_date="2026-01-01",
        source="claude",
        matched_role="user",
        question_text=question_text,
        question_ts=question_ts,
        answer_text=answer_text,
        answer_ts=answer_ts,
    )


# ── _build_qa_pair ────────────────────────────────────────────────────────────

class TestBuildQAPair:
    def test_user_match_has_answer(self):
        session = make_session(messages=[
            make_msg("user",      "my question"),
            make_msg("assistant", "my answer"),
        ])
        q, a = _build_qa_pair(session, 0)
        assert q["role"] == "user"
        assert q["text"] == "my question"
        assert a["role"] == "assistant"
        assert a["text"] == "my answer"

    def test_user_match_at_last_index_no_answer(self):
        session = make_session(messages=[
            make_msg("user", "lonely question"),
        ])
        q, a = _build_qa_pair(session, 0)
        assert q["role"] == "user"
        assert a is None

    def test_assistant_match_has_question(self):
        session = make_session(messages=[
            make_msg("user",      "what is it?"),
            make_msg("assistant", "it is this"),
        ])
        q, a = _build_qa_pair(session, 1)
        assert q["role"] == "user"
        assert a["text"] == "it is this"

    def test_assistant_at_start_no_question(self):
        session = make_session(messages=[
            make_msg("assistant", "unprompted answer"),
        ])
        q, a = _build_qa_pair(session, 0)
        assert q is None
        assert a["role"] == "assistant"

    def test_skips_wrong_role_finds_correct_one(self):
        # two user msgs then an assistant — matched is the second user msg
        session = make_session(messages=[
            make_msg("user",      "first user msg"),
            make_msg("user",      "second user msg"),   # match_idx=1
            make_msg("assistant", "finally an answer"),
        ])
        q, a = _build_qa_pair(session, 1)
        assert q["text"] == "second user msg"
        assert a["text"] == "finally an answer"

    def test_beyond_max_scan_returns_none(self):
        # assistant match, but the user msg is > 3 hops back
        msgs = [make_msg("user", "distant question")] \
             + [make_msg("assistant", f"filler {i}") for i in range(4)] \
             + [make_msg("assistant", "target assistant")]
        session = make_session(messages=msgs)
        q, a = _build_qa_pair(session, len(msgs) - 1)
        assert q is None   # 4 hops > MAX_SCAN=3
        assert a["text"] == "target assistant"


# ── build_fts_index ───────────────────────────────────────────────────────────

class TestBuildFtsIndex:
    def test_empty_sessions_returns_connection(self):
        conn = build_fts_index([])
        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)

    def test_indexed_messages_are_searchable(self):
        session = make_session(messages=[
            make_msg("user",      "CUDA illegal address"),
            make_msg("assistant", "remove unmap_nvds_buf_surface"),
        ])
        conn = build_fts_index([session])
        rows = conn.execute(
            "SELECT session_id, role FROM msg_fts WHERE msg_fts MATCH 'CUDA'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "s1"
        assert rows[0][1] == "user"

    def test_msg_idx_stored_correctly(self):
        session = make_session(messages=[
            make_msg("user",      "first"),
            make_msg("assistant", "second"),
            make_msg("user",      "third"),
        ])
        conn = build_fts_index([session])
        rows = conn.execute(
            "SELECT msg_idx FROM msg_fts WHERE msg_fts MATCH 'third'"
        ).fetchall()
        assert rows[0][0] == 2   # 0-based index


# ── search_fts ────────────────────────────────────────────────────────────────

class TestSearchFts:
    def _setup(self, sessions):
        conn = build_fts_index(sessions)
        session_map = {s["session_id"]: s for s in sessions}
        return conn, session_map

    def test_basic_hit(self):
        session = make_session(messages=[
            make_msg("user",      "CUDA pipeline crash"),
            make_msg("assistant", "here is the fix"),
        ])
        conn, smap = self._setup([session])
        hits = search_fts(conn, "CUDA", smap, ["claude"], max_results=5)
        assert len(hits) == 1
        assert hits[0].session_id == "s1"
        assert hits[0].question_text is not None
        assert hits[0].answer_text is not None

    def test_density_ranking(self):
        # session A: 4 of 5 messages contain "cuda" → high density
        session_a = make_session(session_id="a", messages=[
            make_msg("user",      "cuda error"),
            make_msg("assistant", "cuda fix"),
            make_msg("user",      "more cuda"),
            make_msg("assistant", "cuda confirmed"),
            make_msg("user",      "unrelated topic"),
        ])
        # session B: 1 of 20 messages contains "cuda" → low density
        msgs_b = [make_msg("user", f"msg {i}") for i in range(19)]
        msgs_b.append(make_msg("user", "cuda mentioned once"))
        session_b = make_session(session_id="b", messages=msgs_b)

        conn, smap = self._setup([session_a, session_b])
        hits = search_fts(conn, "cuda", smap, ["claude"], max_results=5)
        assert hits[0].session_id == "a"   # higher density ranked first

    def test_source_filtering(self):
        codex_session  = make_session(session_id="c1", source="codex")
        claude_session = make_session(session_id="c2", source="claude")
        # give both a matching word
        codex_session["messages"][0]["text"]  = "keyword in codex"
        claude_session["messages"][0]["text"] = "keyword in claude"

        conn, smap = self._setup([codex_session, claude_session])
        hits = search_fts(conn, "keyword", smap, ["claude"], max_results=5)
        assert all(h.source == "claude" for h in hits)
        assert len(hits) == 1

    def test_malformed_query_raises(self):
        conn, smap = self._setup([make_session()])
        with pytest.raises(sqlite3.OperationalError):
            search_fts(conn, '"unclosed', smap, ["claude"], max_results=5)

    def test_max_results_respected(self):
        sessions = [
            make_session(session_id=str(i), messages=[
                make_msg("user",      f"needle session {i}"),
                make_msg("assistant", "response"),
            ])
            for i in range(5)
        ]
        conn, smap = self._setup(sessions)
        hits = search_fts(conn, "needle", smap, ["claude"], max_results=3)
        assert len(hits) == 3


# ── search_fuzzy ──────────────────────────────────────────────────────────────

class TestSearchFuzzy:
    def test_no_rapidfuzz_returns_empty(self, monkeypatch):
        # Simulate rapidfuzz not installed
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def fake_import(name, *args, **kwargs):
            if name == "rapidfuzz.fuzz":
                raise ImportError("no rapidfuzz")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        # Remove cached module if already imported
        monkeypatch.delitem(sys.modules, "rapidfuzz.fuzz", raising=False)
        monkeypatch.delitem(sys.modules, "rapidfuzz",      raising=False)

        result = search_fuzzy([make_session()], "CUDA", max_results=5)
        assert result == []

    def test_finds_typo(self):
        pytest.importorskip("rapidfuzz")
        session = make_session(messages=[
            make_msg("user",      "CUDA illegal address error"),
            make_msg("assistant", "remove unmap_nvds"),
        ])
        hits = search_fuzzy([session], "CUUDA", max_results=5)
        assert len(hits) == 1

    def test_threshold_boundary(self):
        pytest.importorskip("rapidfuzz")
        from rapidfuzz.fuzz import partial_ratio

        session = make_session(messages=[
            make_msg("user", "abcdefg"),
        ])
        # Compute actual score so test is not brittle
        score = partial_ratio("xyz", "abcdefg")
        if score < 65:
            hits = search_fuzzy([session], "xyz", max_results=5)
            assert hits == []
        else:
            hits = search_fuzzy([session], "xyz", max_results=5)
            assert len(hits) == 1

    def test_deduplicates_to_one_hit_per_session(self):
        pytest.importorskip("rapidfuzz")
        session = make_session(messages=[
            make_msg("user",      "CUDA error one"),
            make_msg("assistant", "CUDA fix here"),
            make_msg("user",      "more CUDA questions"),
        ])
        hits = search_fuzzy([session], "CUDA", max_results=5)
        assert len(hits) == 1
        assert hits[0].session_id == "s1"


# ── search_smart ──────────────────────────────────────────────────────────────

class TestSearchSmart:
    def _sessions_and_map(self, sessions=None):
        if sessions is None:
            sessions = [make_session(messages=[
                make_msg("user",      "CUDA crash"),
                make_msg("assistant", "CUDA fix"),
            ])]
        smap = {s["session_id"]: s for s in sessions}
        return sessions, smap

    def test_uses_fts_when_conn_available(self):
        sessions, smap = self._sessions_and_map()
        conn = build_fts_index(sessions)
        hits = search_smart(conn, "CUDA", sessions, smap, ["claude"])
        assert len(hits) == 1

    def test_conn_none_falls_back_to_fuzzy(self):
        pytest.importorskip("rapidfuzz")
        sessions, smap = self._sessions_and_map()
        hits = search_smart(None, "CUDA", sessions, smap, ["claude"])
        assert len(hits) == 1

    def test_fts_operational_error_falls_back(self):
        pytest.importorskip("rapidfuzz")
        sessions, smap = self._sessions_and_map()
        conn = build_fts_index(sessions)
        # malformed query forces OperationalError → fuzzy fallback
        hits = search_smart(conn, '"unclosed', sessions, smap, ["claude"])
        # fuzzy may or may not find a match for '"unclosed' — just confirm no crash
        assert isinstance(hits, list)

    def test_fts_empty_result_falls_back_to_fuzzy(self):
        pytest.importorskip("rapidfuzz")
        sessions, smap = self._sessions_and_map()
        conn = build_fts_index(sessions)
        # "xyzzy" won't match in FTS → falls back to fuzzy (also won't match → empty)
        hits = search_smart(conn, "xyzzy", sessions, smap, ["claude"])
        assert hits == []


# ── format_hits ───────────────────────────────────────────────────────────────

class TestFormatHits:
    def test_empty_returns_no_results(self):
        out = format_hits([], "anything")
        assert "No results found" in out

    def test_full_qa_renders_both_blocks(self):
        out = format_hits([make_hit()], "test")
        assert "[YOU ASKED]" in out
        assert "[ANSWER]" in out
        assert "You asked this" in out
        assert "Here is the answer" in out

    def test_no_answer_renders_placeholder(self):
        out = format_hits([make_hit(answer_text=None, answer_ts=None)], "test")
        assert "[YOU ASKED]" in out
        assert "[ANSWER] (no response recorded)" in out

    def test_no_question_renders_only_answer(self):
        out = format_hits(
            [make_hit(question_text=None, question_ts=None)], "test"
        )
        assert "[YOU ASKED]" not in out
        assert "[ANSWER]" in out
        assert "Here is the answer" in out


# ── format_session ────────────────────────────────────────────────────────────

class TestFormatSession:
    def _session_with_n_msgs(self, n):
        msgs = [make_msg("user" if i % 2 == 0 else "assistant", f"msg {i}")
                for i in range(n)]
        return make_session(messages=msgs)

    def test_truncates_to_max_messages(self):
        session = self._session_with_n_msgs(40)
        out = format_session(session, max_messages=5)
        assert "showing first 5" in out
        assert "msg 5" not in out   # 6th message should not appear

    def test_no_truncation_label_when_under_limit(self):
        session = self._session_with_n_msgs(3)
        out = format_session(session, max_messages=30)
        assert "showing first" not in out

    def test_long_message_text_is_cut_off(self):
        long_text = "x" * 3000
        session = make_session(messages=[make_msg("user", long_text)])
        out = format_session(session, max_messages=30)
        # The 3000-char text should be truncated to 2000 in output
        assert "x" * 2001 not in out
        assert "x" * 2000 in out
