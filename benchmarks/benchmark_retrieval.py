"""
Benchmark local retrieval paths for agent-history-mcp.

The benchmark is intentionally local and reproducible:
- no network calls
- no model calls
- uses real Codex/Claude history paths unless overridden
- uses a temporary SQLite graph DB unless --db-path is provided
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_history_mcp.graph import HistoryGraphIndex  # noqa: E402
from agent_history_mcp.parsers import parse_claude_sessions, parse_codex_sessions  # noqa: E402
from agent_history_mcp.search import build_fts_index, search_smart  # noqa: E402
from agent_history_mcp.skills import suggest_skill_candidates  # noqa: E402


DEFAULT_QUERIES = [
    "history search graph",
    "git push",
    "permission denied",
    "pytest fixture",
    "redis migration timeout",
    "azure deployment",
    "CUDA illegal address",
]


@dataclass
class QueryResult:
    query: str
    fts_ms: float
    graph_ms: float
    fts_hits: int
    graph_hits: int
    graph_only_hits: int


def _ms(seconds: float) -> float:
    return seconds * 1000


def _now() -> float:
    return time.perf_counter()


def _load_sessions(codex_path: Path, claude_path: Path) -> list[dict]:
    sessions = []
    if codex_path.is_dir():
        sessions.extend(parse_codex_sessions(codex_path))
    if claude_path.is_dir():
        sessions.extend(parse_claude_sessions(claude_path))
    return sessions


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _session_ids(hits) -> set[tuple[str, str]]:
    return {(hit.source, hit.session_id) for hit in hits}


def run_benchmark(args: argparse.Namespace) -> str:
    codex_path = Path(args.codex_path).expanduser()
    claude_path = Path(args.claude_path).expanduser()
    queries = args.query or DEFAULT_QUERIES

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.db_path:
        db_path = Path(args.db_path).expanduser()
        if db_path.exists() and args.rebuild:
            db_path.unlink()
    else:
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "history_graph.sqlite"

    try:
        load_start = _now()
        sessions = _load_sessions(codex_path, claude_path)
        load_ms = _ms(_now() - load_start)

        messages = sum(len(session["messages"]) for session in sessions)
        chars = sum(len(msg["text"]) for session in sessions for msg in session["messages"])
        session_map = {session["session_id"]: session for session in sessions}

        fts_build_start = _now()
        conn = build_fts_index(sessions)
        fts_build_ms = _ms(_now() - fts_build_start)

        index = HistoryGraphIndex(db_path)
        try:
            graph_cold_start = _now()
            stats = index.refresh(codex_path if codex_path.is_dir() else None, claude_path if claude_path.is_dir() else None)
            graph_cold_ms = _ms(_now() - graph_cold_start)

            graph_warm_start = _now()
            warm_stats = index.refresh(codex_path if codex_path.is_dir() else None, claude_path if claude_path.is_dir() else None)
            graph_warm_refresh_ms = _ms(_now() - graph_warm_start)

            query_results: list[QueryResult] = []
            for query in queries:
                fts_times = []
                graph_times = []
                fts_hits = []
                graph_hits = []

                for _ in range(args.repeat):
                    fts_start = _now()
                    fts_hits = search_smart(conn, query, sessions, session_map, ["codex", "claude"], args.max_results)
                    fts_times.append(_ms(_now() - fts_start))

                    graph_start = _now()
                    graph_hits = index.search(query, ["codex", "claude"], args.max_results)
                    graph_times.append(_ms(_now() - graph_start))

                fts_ids = _session_ids(fts_hits)
                graph_ids = _session_ids(graph_hits)
                query_results.append(QueryResult(
                    query=query,
                    fts_ms=_mean(fts_times),
                    graph_ms=_mean(graph_times),
                    fts_hits=len(fts_ids),
                    graph_hits=len(graph_ids),
                    graph_only_hits=len(graph_ids - fts_ids),
                ))

            cold_rebuild_times = []
            for query in queries[: args.cold_queries]:
                start = _now()
                cold_sessions = _load_sessions(codex_path, claude_path)
                cold_map = {session["session_id"]: session for session in cold_sessions}
                cold_conn = build_fts_index(cold_sessions)
                search_smart(cold_conn, query, cold_sessions, cold_map, ["codex", "claude"], args.max_results)
                cold_rebuild_times.append(_ms(_now() - start))
                if cold_conn is not None:
                    cold_conn.close()

            skill_start = _now()
            candidates = suggest_skill_candidates(index, ["codex", "claude"], max_candidates=5, min_sessions=2)
            skill_ms = _ms(_now() - skill_start)

        finally:
            index.close()
            if conn is not None:
                conn.close()

        return _format_markdown(
            codex_path=codex_path,
            claude_path=claude_path,
            sessions=len(sessions),
            messages=messages,
            chars=chars,
            load_ms=load_ms,
            fts_build_ms=fts_build_ms,
            graph_cold_ms=graph_cold_ms,
            graph_warm_refresh_ms=graph_warm_refresh_ms,
            stats=stats.as_dict(),
            warm_stats=warm_stats.as_dict(),
            query_results=query_results,
            cold_rebuild_mean_ms=_mean(cold_rebuild_times),
            cold_rebuild_count=len(cold_rebuild_times),
            skill_ms=skill_ms,
            skill_candidates=len(candidates),
            repeat=args.repeat,
            max_results=args.max_results,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _format_markdown(
    *,
    codex_path: Path,
    claude_path: Path,
    sessions: int,
    messages: int,
    chars: int,
    load_ms: float,
    fts_build_ms: float,
    graph_cold_ms: float,
    graph_warm_refresh_ms: float,
    stats: dict,
    warm_stats: dict,
    query_results: list[QueryResult],
    cold_rebuild_mean_ms: float,
    cold_rebuild_count: int,
    skill_ms: float,
    skill_candidates: int,
    repeat: int,
    max_results: int,
) -> str:
    graph_only = sum(result.graph_only_hits for result in query_results)
    fts_total = sum(result.fts_hits for result in query_results)
    graph_total = sum(result.graph_hits for result in query_results)

    lines = [
        "## Local Benchmark Results",
        "",
        "Benchmarks are local-machine measurements, not universal claims. They use the saved Codex/Claude JSONL history on the test machine and a temporary graph database.",
        "",
        f"- Corpus: {sessions} sessions, {messages} parsed messages, {chars:,} message characters",
        f"- Codex path: `{codex_path}`",
        f"- Claude path: `{claude_path}`",
        f"- Query repeat count: {repeat}; max results per query: {max_results}",
        "",
        "| Operation | Mean / elapsed time | Notes |",
        "|-----------|---------------------|-------|",
        f"| Parse JSONL sessions | {load_ms:.1f} ms | Full parser pass over both history sources |",
        f"| Build in-memory FTS index | {fts_build_ms:.1f} ms | SQLite FTS5 over parsed messages |",
        f"| Cold graph index build | {graph_cold_ms:.1f} ms | Files seen: {stats['files_seen']}; sessions indexed: {stats['sessions_indexed']} |",
        f"| Warm graph refresh | {graph_warm_refresh_ms:.1f} ms | Unchanged files: {warm_stats['files_unchanged']}; no JSONL reparse |",
        f"| Raw JSONL + FTS rebuild + search | {cold_rebuild_mean_ms:.1f} ms | Mean over {cold_rebuild_count} representative cold queries |",
        f"| Skill suggestion pass | {skill_ms:.1f} ms | Returned {skill_candidates} candidates |",
        "",
        "| Query | FTS query ms | Graph query ms | FTS hits | Graph hits | Graph-only hits |",
        "|-------|--------------|----------------|----------|------------|-----------------|",
    ]

    for result in query_results:
        lines.append(
            f"| `{result.query}` | {result.fts_ms:.2f} | {result.graph_ms:.2f} | "
            f"{result.fts_hits} | {result.graph_hits} | {result.graph_only_hits} |"
        )

    lines.extend([
        "",
        f"Across these queries, FTS returned {fts_total} unique query-result sessions and graph search returned {graph_total}. "
        f"The graph layer added {graph_only} graph-only candidate sessions that keyword search did not return for the same query set.",
        "",
        "Interpretation:",
        "- Warm graph refresh is the main speed win for repeated MCP calls: unchanged history files are checked by metadata instead of reparsed.",
        "- Graph-only hits are retrieval expansion, not guaranteed correctness. They should be treated as additional candidates with evidence.",
        "- Exact numbers depend on disk speed, Python version, SQLite build, corpus size, and query mix.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark agent-history-mcp retrieval paths.")
    parser.add_argument("--codex-path", default=str(Path.home() / ".codex"))
    parser.add_argument("--claude-path", default=str(Path.home() / ".claude"))
    parser.add_argument("--db-path", default="")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--cold-queries", type=int, default=3)
    parser.add_argument("--query", action="append", help="Query to benchmark. Can be passed multiple times.")
    return parser.parse_args()


def main() -> None:
    print(run_benchmark(parse_args()))


if __name__ == "__main__":
    main()
