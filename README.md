# agent-history-mcp

Local memory, handoff, analytics and skill mining across your coding agents.

`agent-history-mcp` is a local MCP server and CLI that reads the session history of **Claude Code**, **OpenAI Codex CLI**, **Gemini CLI**, **OpenCode** and **Cursor**. It indexes everything into one local graph, then lets any of those agents:

- **search** what was solved before, in any agent
- **pick up where another agent stopped** in the same repository (handoff briefs and checkpoints)
- **keep instructions in sync** across CLAUDE.md, AGENTS.md, GEMINI.md and auto-memory
- **analyze sessions** for time, tokens, failing commands and friction
- **turn repeated workflows into skills**: mine, draft a `SKILL.md`, install it for every agent, then track whether it is actually used

Everything runs locally and deterministically. There are no network calls, no model calls, and no API keys. History files are read-only. The only writes outside `~/.agent-history-mcp` happen when you pass `apply=true`.

## Why

Claude Code, Codex, Gemini CLI, Cursor and OpenCode all keep their own transcripts, memories and instruction files, and none of them reads the others'. If you switch tools mid-task, the next agent starts cold. Fixes you made with one agent get rediscovered by another. Workflows you repeat every week never become reusable skills. Each tool now speaks the open [Agent Skills](https://agentskills.io) format and MCP, so one local layer can bridge all of them.

## Quick start

```bash
pip install git+https://github.com/monishkumarvr/agent-history-mcp.git
```

**Claude Code** (`~/.claude/.mcp.json` or `claude mcp add agent-history -- agent-history-mcp`):

```json
{
  "mcpServers": {
    "agent-history": { "command": "agent-history-mcp" }
  }
}
```

**Codex CLI** (`~/.codex/config.toml`):

```toml
[mcp_servers.agent-history]
command = "agent-history-mcp"
```

**Gemini CLI** (`~/.gemini/settings.json`), **Cursor** (`~/.cursor/mcp.json`) and **OpenCode** all accept the same command in their MCP server settings.

Then ask any agent:

```text
Where did the last agent stop in this repo?            -> handoff_brief
Did I solve a similar Redis timeout before?            -> search_history
What did I spend this week on, and where did I struggle? -> analyze_sessions / history_report
Which of my workflows should become skills?            -> suggest_skills -> draft_skill -> install_skill
Are my skills actually being used?                     -> skill_health
Which rules does Codex know that Claude doesn't?       -> list_agent_memory -> sync_memory
```

Clients that support MCP prompts expose `/handoff`, `/draft-skill` and `/weekly-review` as slash commands.

## Tools

| Area | Tool | What it does |
|------|------|--------------|
| Search | `search_history` | Hybrid FTS5 + fuzzy + graph search across every agent; optional `project` filter. |
| | `search_graph` | Relationship search over extracted commands, paths, APIs, errors and topics. |
| | `list_sessions`, `get_session` | Browse sessions; `include_tools=true` interleaves commands, edits and errors. |
| Context bridge | `handoff_brief` | Where the latest sessions in a repo stopped: goal, open todos, files touched, command outcomes, unresolved errors, dead ends, latest checkpoint, live git state. |
| | `save_checkpoint` | Save a note plus the brief for the next agent; `apply=true` also writes `.agent-context/HANDOFF.md`. |
| | `list_agent_memory` | Which instruction and memory files each agent reads, and the rules some agents are missing (drift). |
| | `suggest_rules` | Mines repeated corrections ("no, use pnpm"), stated preferences and rejected tool calls into rule candidates. |
| | `sync_memory` | Previews, then with `apply=true` writes, shared rules into a managed block in CLAUDE.md / AGENTS.md / GEMINI.md. |
| Analytics | `analyze_sessions` | Active time, tokens, tool mix, top and failing commands, edited files, friction, cross-agent comparison. |
| | `history_report` | Writes the same as a self-contained HTML report. |
| Skills | `suggest_skills` | Repeated multi-step workflows plus topic clusters, checked against installed skills (create / update / skip). |
| | `draft_skill` | Deterministic Agent Skills `SKILL.md` with real example commands, failure modes and evidence. |
| | `install_skill` | Installs to `~/.claude/skills` and `~/.agents/skills` (or project scope). Dry run unless `apply=true`; never overwrites unless `overwrite=true`. |
| | `skill_health` | Per-agent usage, missed triggers, outcome proxies and drift for installed skills, with a keep / revise / update / retire call. |
| Index | `refresh_history_index` | Incremental refresh or full rebuild of the derived index, with per-source counts and notes. |

## CLI

The `agent-history` command exposes the same features:

```bash
agent-history handoff --project .            # brief for this repo
agent-history checkpoint "Paused: waiting on API keys"
agent-history analyze --days-back 7 --group-by source
agent-history report                         # HTML report path
agent-history skills suggest
agent-history skills draft wf-1a2b3c4d5e6f > SKILL.md
agent-history skills install --file SKILL.md --apply
agent-history skills health
agent-history memory                         # instruction files and drift
agent-history rules --min-sessions 2
agent-history sync-memory                    # dry-run diff
agent-history                                # runs the MCP server
```

### Automatic handoff in Claude Code

To start every Claude Code session with what other agents last did in the repo, add a SessionStart hook to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [{ "type": "command", "command": "agent-history hook session-start" }] }
    ]
  }
}
```

The hook prints nothing when there is no other-agent history for the directory. For Codex, Gemini CLI, Cursor or OpenCode, add one line to AGENTS.md or GEMINI.md: *"At the start of a task, call the agent-history `handoff_brief` tool for this repository."*

## Supported sources

| Agent | Default location | Format | Notes |
|-------|------------------|--------|-------|
| Claude Code | `~/.claude/projects/` | JSONL | Tool calls/results, cwd, branch, model, tokens. Subagent transcripts are merged into their parent. |
| Codex CLI | `~/.codex/sessions/`, `~/.codex/archived_sessions/` | JSONL rollouts | Function/custom/local-shell calls, exit codes, token counts, thread names. |
| Gemini CLI | `~/.gemini/tmp/<project>/chats/` | JSONL replay log (legacy JSON) | Messages, tool calls, tokens. |
| OpenCode | `~/.local/share/opencode/opencode*.db` | SQLite, read-only | Messages, tool parts, tokens. |
| Cursor | `<Cursor User dir>/globalStorage/state.vscdb` | SQLite key-value, read-only | Agent/composer chats, tool calls, workspace folder. Undocumented format; best effort. The cursor-agent CLI store is detected but not parsed yet. |

Override locations with environment variables, and limit sources with `AGENT_HISTORY_SOURCES` (e.g. `claude,codex`):

```json
{
  "mcpServers": {
    "agent-history": {
      "command": "agent-history-mcp",
      "env": {
        "CLAUDE_PATH": "/custom/.claude",
        "CODEX_PATH": "/custom/.codex",
        "GEMINI_PATH": "/custom/.gemini",
        "OPENCODE_DB": "/custom/opencode",
        "CURSOR_PATH": "/custom/Cursor/User",
        "AGENT_HISTORY_HOME": "/custom/agent-history-data",
        "AGENT_HISTORY_GRAPH_DB": "/custom/history_graph.sqlite"
      }
    }
  }
}
```

Derived data (index, checkpoints, reports) lives in `~/.agent-history-mcp/` unless `AGENT_HISTORY_HOME` is set.

## How it works

```text
source adapters (claude, codex, gemini, opencode, cursor)
    -> one event model: messages, tool calls/results, commands, paths, exit codes,
       todos, skills, user signals (corrections, interrupts, rejections), cwd, tokens
    -> incremental SQLite index (unchanged units are skipped by change token)
       sessions · messages · events · graph nodes/edges · installed_skills
    -> search · handoff · memory sync · analytics · workflow mining · skill health
```

- **Workflow mining** turns each session into an action skeleton: shell commands normalized to `git commit`, `npm run build` and so on, edits by file type, MCP calls and skills, with read-only noise removed. It counts contiguous step sequences shared across sessions, drops patterns that appear almost everywhere, and keeps only the longest pattern that covers a given set of sessions. Patterns seen in several projects are marked `global`.
- **Skill overlap** compares a candidate's key terms with installed skills' names, descriptions and bodies, so you update an existing skill instead of adding a near-duplicate.
- **Skill health** detects use through the Skill tool, `/slash-commands` and reads of `…/skills/<name>/SKILL.md`. It finds missed triggers by matching the installed workflow's steps, or the skill's name, against later sessions.
- **Handoff** summarizes events. It does not summarize with a model, so a brief costs a few hundred tokens and is reproducible.
- **Memory sync** only edits the text between `<!-- agent-history:begin -->` and `<!-- agent-history:end -->`.

The index is rebuilt automatically when its schema version changes. You can also run `refresh_history_index(rebuild=true)` to rebuild it.

## Security and privacy

- History is read-only. Agent transcripts and databases are never modified; SQLite stores are opened in read-only mode.
- There are no network or model calls.
- Every tool output passes through secret redaction: API keys, GitHub, Slack and AWS tokens, JWTs, bearer tokens, private keys, `password=` pairs and URL credentials. Set `AGENT_HISTORY_REDACT=0` to disable it.
- Writes are opt-in. `install_skill`, `sync_memory` and `save_checkpoint(apply=true)` show a dry run or diff first and write only with `apply=true`. Project-scope writes land in your repository, so review them before committing.
- Credentials files such as `~/.codex/auth.json` are never read.
- The derived index contains message excerpts, so treat `~/.agent-history-mcp/` like your history directories.

## Limitations

- The Cursor and OpenCode formats are undocumented or change between versions. Their parsers degrade to skipping, with a note in `refresh_history_index`.
- Extraction is deterministic, so topics can be noisy. Skill candidates and rule candidates are suggestions to review.
- Cross-agent comparisons and skill outcome numbers are directional, not controlled experiments.
- The first index build over a large history can take a while. Later refreshes only parse changed units.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the landscape research and planned work.

## Benchmarks

Run the local benchmark:

```bash
python benchmarks/benchmark_retrieval.py
```

The benchmark compares:

- full JSONL parsing
- in-memory FTS index build and query
- cold graph index build
- warm graph refresh with unchanged files
- graph search query time
- graph-only candidate expansion versus FTS results
- skill suggestion time

### Local Benchmark Results (v0.3, Codex + Claude only)

These numbers were measured on one local Windows laptop against its saved Codex/Claude histories. They are useful as a directional signal, not a universal performance claim.

- Corpus: 43 sessions, 2,434 parsed messages, 2,741,934 message characters
- History files seen: 48
- Query repeat count: 5
- Max results per query: 5

| Operation | Mean / elapsed time | Notes |
|-----------|---------------------|-------|
| Parse JSONL sessions | 676.1 ms | Full parser pass over Codex and Claude history |
| Build in-memory FTS index | 28.5 ms | SQLite FTS5 over parsed messages |
| Cold graph index build | 15,958.6 ms | One-time derived graph build for 43 indexed sessions |
| Warm graph refresh | 20.7 ms | Unchanged files checked by metadata; no JSONL reparse |
| Raw JSONL + FTS rebuild + search | 1,324.1 ms | Mean over 3 representative cold queries |
| Skill suggestion pass | 7,376.1 ms | Returned 5 candidates |

| Query | FTS query ms | Graph query ms | FTS hits | Graph hits | Graph-only hits |
|-------|--------------|----------------|----------|------------|-----------------|
| `history search graph` | 0.53 | 3.23 | 5 | 5 | 4 |
| `git push` | 0.73 | 22.35 | 5 | 5 | 4 |
| `permission denied` | 0.35 | 12.06 | 5 | 5 | 1 |
| `pytest fixture` | 0.46 | 0.32 | 0 | 0 | 0 |
| `redis migration timeout` | 0.32 | 22.85 | 2 | 5 | 3 |
| `azure deployment` | 0.15 | 0.24 | 3 | 0 | 0 |
| `CUDA illegal address` | 0.31 | 18.13 | 4 | 5 | 4 |

Across these benchmark queries, FTS returned 24 query-result sessions and graph search returned 25. The graph layer added 16 graph-only candidate sessions that keyword search did not return for the same query set.

Interpretation:

- The speed win is not that graph query beats an already-hot FTS query. Hot FTS is extremely fast.
- The practical speed win is warm refresh: repeated MCP calls check unchanged files in about 20 ms instead of reparsing JSONL and rebuilding search state.
- The retrieval win is candidate expansion: graph search can surface related sessions through extracted relationships even when exact keywords differ.
- Graph-only hits are additional evidence-backed candidates, not guaranteed correct answers.

## Development

```bash
pip install -e .
python -m unittest discover -s tests
python -m py_compile src/agent_history_mcp/*.py src/agent_history_mcp/sources/*.py tests/*.py
python benchmarks/benchmark_retrieval.py
```

`tests/fixtures.py` writes realistic synthetic histories for all five agents. New features should be tested against more than one source.

## File structure

```text
src/agent_history_mcp/
  server.py        MCP tools and prompts
  cli.py           agent-history CLI and hooks
  model.py         unified session/event model, canonical tools, command signatures
  sources/         claude, codex, gemini, opencode, cursor adapters
  graph.py         incremental SQLite index and graph search
  search.py        FTS5 + fuzzy keyword search and formatting
  handoff.py       handoff briefs and checkpoints
  memory_sync.py   instruction/memory discovery, drift, rule mining, managed-block sync
  analytics.py     session analytics and HTML report
  workflows.py     workflow pattern mining
  skills.py        skill candidates, overlap, SKILL.md drafts, skill health
  skill_store.py   installed-skill discovery, usage detection, installation
  redact.py        secret redaction
tests/             unittest suites + multi-agent fixtures
benchmarks/        retrieval benchmark
docs/ROADMAP.md    research notes and next steps
```

## License

MIT
