"""
`agent-history` command line: the same features as the MCP tools, for shells,
scripts, and agent hooks.

  agent-history                       run the MCP server (stdio)
  agent-history handoff [--project .] print a cross-agent handoff brief
  agent-history hook session-start    Claude Code SessionStart hook (reads JSON on stdin)
  agent-history report                write the HTML history report
  agent-history --help                list every command
"""

from __future__ import annotations

import argparse
import json
import sys


def _sources(value: str) -> str:
    return value or "all"


def _session_start_hook(args: argparse.Namespace) -> int:
    from . import handoff, server

    try:
        payload = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except ValueError:
        payload = {}
    cwd = payload.get("cwd") or args.project
    server._refresh_history_index()
    root = handoff.resolve_project(cwd)
    index = server._get_graph_index()
    if not index.session_rows(None, root, limit=1) and not handoff.latest_checkpoint(root):
        return 0
    brief = handoff.handoff_brief(
        index, root, sessions=args.sessions, exclude_source=args.exclude_source or None,
        compact=True, include_git=False,
    )
    if brief.startswith("No indexed sessions"):
        return 0
    context = (
        "Context from other coding agents' recent sessions in this repository "
        "(via agent-history-mcp; call handoff_brief or search_history for more):\n\n" + brief
    )
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-history", description="Cross-agent coding history: search, handoff, analytics, skills.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="run the MCP server over stdio (default)")

    p = sub.add_parser("refresh", help="refresh (or rebuild) the derived index")
    p.add_argument("--rebuild", action="store_true")

    p = sub.add_parser("search", help="search past sessions")
    p.add_argument("query")
    p.add_argument("--sources", default="all")
    p.add_argument("--project")
    p.add_argument("-n", "--max-results", type=int, default=5)

    p = sub.add_parser("sessions", help="list sessions")
    p.add_argument("--source", default="all")
    p.add_argument("--project")
    p.add_argument("-n", "--limit", type=int, default=30)

    p = sub.add_parser("handoff", help="print a handoff brief for a repository")
    p.add_argument("--project")
    p.add_argument("--sources", default="all")
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--exclude-source")

    p = sub.add_parser("checkpoint", help="save a handoff checkpoint note")
    p.add_argument("note")
    p.add_argument("--project")
    p.add_argument("--apply", action="store_true", help="also write .agent-context/HANDOFF.md in the repo")

    p = sub.add_parser("memory", help="show instruction/memory files per agent and their drift")
    p.add_argument("--project")

    p = sub.add_parser("rules", help="suggest rules from your repeated corrections")
    p.add_argument("--project", help="repository path; use --all-projects to search everywhere")
    p.add_argument("--all-projects", action="store_true")
    p.add_argument("--days-back", type=int, default=90)
    p.add_argument("--min-sessions", type=int, default=2)

    p = sub.add_parser("sync-memory", help="sync rules into CLAUDE.md / AGENTS.md / GEMINI.md (dry run by default)")
    p.add_argument("--rule", action="append", dest="rules", help="rule to add (repeatable); omit to sync drift")
    p.add_argument("--target", action="append", dest="targets", help="agent to update (repeatable)")
    p.add_argument("--project")
    p.add_argument("--scope", default="project", choices=["project", "user"])
    p.add_argument("--apply", action="store_true")

    p = sub.add_parser("analyze", help="session analytics")
    p.add_argument("--project")
    p.add_argument("--days-back", type=int, default=30)
    p.add_argument("--sources", default="all")
    p.add_argument("--group-by", default="project", choices=["project", "source", "day", "model"])

    p = sub.add_parser("report", help="write the HTML history report")
    p.add_argument("--project")
    p.add_argument("--days-back", type=int, default=30)
    p.add_argument("--sources", default="all")

    p = sub.add_parser("skills", help="suggest, draft, install and check skills")
    skills_sub = p.add_subparsers(dest="skills_cmd")
    s = skills_sub.add_parser("suggest")
    s.add_argument("--sources", default="all")
    s.add_argument("-n", "--max-candidates", type=int, default=5)
    s.add_argument("--min-sessions", type=int, default=2)
    s.add_argument("--days-back", type=int)
    s.add_argument("--project")
    s = skills_sub.add_parser("draft")
    s.add_argument("candidate_id")
    s = skills_sub.add_parser("install")
    s.add_argument("--file", help="reviewed SKILL.md to install")
    s.add_argument("--candidate-id")
    s.add_argument("--target", action="append", dest="targets")
    s.add_argument("--scope", default="user", choices=["user", "project"])
    s.add_argument("--project")
    s.add_argument("--apply", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    s = skills_sub.add_parser("health")
    s.add_argument("--days-back", type=int, default=60)
    s.add_argument("--project")

    p = sub.add_parser("hook", help="agent hook entry points")
    hook_sub = p.add_subparsers(dest="hook")
    h = hook_sub.add_parser("session-start", help="Claude Code SessionStart hook: inject other agents' recent context")
    h.add_argument("--project", help="fallback when the hook payload has no cwd")
    h.add_argument("--exclude-source", default="claude", help="agent whose own sessions to skip (default: claude)")
    h.add_argument("--sessions", type=int, default=1)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from . import server

    command = args.command or "serve"
    if command == "serve":
        server.run()
        return 0
    if command == "hook":
        if args.hook == "session-start":
            return _session_start_hook(args)
        parser.parse_args(["hook", "--help"])
        return 2

    if command == "refresh":
        out = server.refresh_history_index(rebuild=args.rebuild)
    elif command == "search":
        out = server.search_history(args.query, _sources(args.sources), args.max_results, args.project)
    elif command == "sessions":
        out = server.list_sessions(args.source, args.limit, args.project)
    elif command == "handoff":
        out = server.handoff_brief(args.project, _sources(args.sources), args.sessions, args.exclude_source)
    elif command == "checkpoint":
        out = server.save_checkpoint(args.note, args.project, args.apply)
    elif command == "memory":
        out = server.list_agent_memory(args.project)
    elif command == "rules":
        out = server.suggest_rules("" if args.all_projects else args.project, "all", args.days_back, args.min_sessions)
    elif command == "sync-memory":
        out = server.sync_memory(args.rules, args.targets, args.project, args.scope, args.apply)
    elif command == "analyze":
        out = server.analyze_sessions(args.project, args.days_back or None, _sources(args.sources), args.group_by)
    elif command == "report":
        out = server.history_report(args.project, args.days_back or None, _sources(args.sources))
    elif command == "skills":
        if args.skills_cmd == "suggest":
            out = server.suggest_skills(_sources(args.sources), args.max_candidates, args.min_sessions, args.days_back, True, args.project)
        elif args.skills_cmd == "draft":
            out = server.draft_skill(args.candidate_id)
        elif args.skills_cmd == "install":
            text = open(args.file, encoding="utf-8").read() if args.file else None
            out = server.install_skill(text, args.candidate_id, args.targets, args.scope, args.project, args.apply, args.overwrite)
        elif args.skills_cmd == "health":
            out = server.skill_health(args.days_back or None, args.project)
        else:
            parser.parse_args(["skills", "--help"])
            return 2
    else:
        parser.print_help()
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
