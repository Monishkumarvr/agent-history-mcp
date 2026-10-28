# Roadmap and landscape (October 2026)

This note records where coding-agent tooling is heading, what other projects already cover, what agent-history-mcp v0.4 adds, and what comes next.

## Where the agents are

| Agent | Memory and instructions | Skills | History on disk |
|-------|------------------------|--------|-----------------|
| Claude Code | `CLAUDE.md` hierarchy; auto memory in `~/.claude/projects/<proj>/memory/` (`MEMORY.md` index plus typed notes); `/insights` HTML report of recent sessions | Agent Skills in `~/.claude/skills`, `.claude/skills`, plugins; skills sync from claude.ai | JSONL per session with tool calls, cwd, branch, usage; subagent transcripts alongside |
| Codex CLI | Layered `AGENTS.md` (global + repo path); opt-in memories in `~/.codex/memories` | Agent Skills in `~/.agents/skills` / `.agents/skills` (older builds `~/.codex/skills`) | JSONL rollouts in `~/.codex/sessions/YYYY/MM/DD/`; `resume` and `fork` |
| Gemini CLI | `GEMINI.md` hierarchy | Agent Skills support | JSONL replay log per chat in `~/.gemini/tmp/<project>/chats/` |
| OpenCode | `AGENTS.md` | Agent Skills support | One SQLite DB (`session`, `message`, `part`, …) since early 2026 |
| Cursor | `.cursor/rules/*.mdc`, `AGENTS.md` | Agent Skills support | Undocumented SQLite KV (`cursorDiskKV`: `composerData:*`, `bubbleId:*`), can grow to tens of GB |

The trends this project builds on:

1. **Agent Skills is now a shared standard.** The `SKILL.md` format published at agentskills.io (Dec 2025) is supported by 40+ clients. In October 2026 it was reported to be moving to the Agentic AI Foundation. One skill file can serve every agent.
2. **Memory is per-agent and siloed.** Each tool added its own memory (Claude auto memory, Codex memories, Gemini `/memory`), and none of them reads the others'.
3. **MCP moved to a stateless core.** In spec 2026-07-28, sampling and roots are deprecated and elicitation and prompts remain. A server should stay deterministic and leave prose to the calling agent through prompts and tool output, rather than asking the client's model to write.
4. **Session analytics exists, but per agent.** Claude Code's `/insights` analyzes Claude sessions only.

## Adjacent projects

| Project | Focus | Gap this project fills |
|---------|-------|------------------------|
| [continues](https://github.com/yigitkonur/cli-continues) | Hand a session to another CLI (16 agents) | Ends at a one-off handoff: no index, search, analytics or skills |
| [agent-chorus](https://github.com/cote-star/agent-chorus) | Read/compare/hand off sessions, checkpoints, cross-agent messages (CLI) | Not MCP-native; no skill lifecycle or memory sync |
| [cross-session-mcp](https://github.com/reiarseni/cross-session-mcp) | MCP handoff; summarizes old transcripts with Gemini | Sends transcripts to a model; no workflow mining |
| [skill_miner](https://github.com/liaoandi/skill_miner) | LLM finds recurring workflows and drafts SKILL.md | LLM-based, one-shot; no overlap check or usage tracking after install |
| ThakiCloud "Chronicle" ([write-up](https://thakicloud.com/tech-blog/en/agentops/conversations-into-self-evolving-skills/)) | Deterministic counting, dedupe against existing skills, evolving skills | Claude-only pipeline, not a reusable tool |
| Claude Code `/insights` | Friction and suggestions from Claude sessions | Single agent; no cross-tool view |

## What v0.4 adds

- A single event model over five agents' histories: tool calls, exit codes, files, todos, user signals, tokens, and project.
- **Handoff with dead ends**: briefs include commands that failed repeatedly, so the next agent doesn't retry them. Checkpoints and a Claude Code SessionStart hook build on the same brief.
- **Instruction drift**: rules visible to one agent but not another, synced through a managed block. Claude reading `AGENTS.md` via `@AGENTS.md` imports is taken into account.
- **Corrections become rules**: repeated "no, do X" messages, stated preferences and rejected tool calls become rule candidates, each marked as already documented or not.
- **A closed skill loop**: mine workflows, check overlap, draft SKILL.md, install with opt-in writes, then measure usage, missed triggers, outcome proxies and drift.
- **Cross-agent comparison**: per-topic medians of turns, tools, errors and tokens per agent. This is a directional signal for which agent fits which work.

## Next

Ordered by value for effort.

1. **Cursor agent CLI store.** Parse `~/.cursor/chats/**/store.db` once the format is pinned with real samples. Add Copilot CLI, Amp and Kiro adapters behind the same interface.
2. **Packaging as plugins.** Ship a Claude Code plugin (MCP server + SessionStart hook + a bundled `weekly-review` skill) and the Codex equivalent, so setup is one command.
3. **Skill evaluation harness.** Replay the evidence sessions' first prompts against a skill/no-skill pair in a sandbox, and accept a skill only when it improves outcomes. This is the A/B gate that SkillMiner and Chronicle describe, applied to local history.
4. **Session-to-eval cases.** Turn sessions that ended in a fix (failing command, edits, then the same command passing) into regression cases for agents or skills.
5. **Optional local embeddings.** An opt-in sentence-embedding index (e.g. a small local ONNX model) for "similar problem" retrieval when terms differ, kept off by default to preserve the no-model guarantee.
6. **MCP Apps UI.** Serve the HTML report as an MCP App so clients that support it can render it inline with filters.
7. **Team mode.** Export redacted skill candidates, rules and workflow stats (not transcripts) to a shared registry, so a team can see which skills work across people.
8. **Routing advice.** Use cross-agent comparisons, once enough data exists, to suggest which agent or model to use for a task type, with confidence intervals instead of medians.
9. **Live cross-agent notes.** A lightweight `post_note` / `read_notes` channel per repository, so two agents running at once can coordinate without a central server.
