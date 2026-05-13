# agent-history-mcp

Local memory search for Codex and Claude Code conversations.

`agent-history-mcp` is a local MCP server that lets coding agents search your past AI coding sessions across **OpenAI Codex CLI** and **Claude Code**. It keeps the history on your machine, builds a local graph index, and returns compact evidence instead of dumping full conversations into context.

## Why This Exists

Coding agents solve useful problems in one-off chats, then that knowledge disappears into JSONL history. This project turns those old sessions into a searchable local memory layer:

- find previous fixes, commands, APIs, errors, and file paths
- search Codex history from Claude Code, and Claude history from Codex-compatible MCP clients
- use graph search for "similar problem" retrieval when exact keywords differ
- surface repeated workflows that could become reusable skills

No chat history is uploaded. Codex and Claude history files are read-only; only the derived local index is written.

## Features

- **Hybrid search**: SQLite FTS5 keyword search plus fuzzy fallback.
- **Local graph search**: extracted relationships between sessions, messages, topics, commands, APIs, paths, and errors.
- **Skill suggestions**: evidence-backed ideas for reusable skills based on repeated chat patterns.
- **Incremental indexing**: changed JSONL files are parsed on demand; no background daemon required.
- **Privacy-first**: no network calls or model calls are used for indexing.

## Quick Start

### 1. Install

From GitHub:

```bash
pip install git+https://github.com/monishkumarvr/agent-history-mcp.git
```

From a local clone:

```bash
git clone https://github.com/monishkumarvr/agent-history-mcp.git
cd agent-history-mcp
pip install .
```

### 2. Register With Claude Code

Add this to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "agent-history": {
      "command": "python3",
      "args": ["-m", "agent_history_mcp"]
    }
  }
}
```

Claude Code starts the server automatically when needed.

### 3. Run Your First Query

Ask Claude Code:

```text
Search my history for CUDA illegal address
```

Other useful prompts:

```text
Use graph search for a similar Redis migration timeout
Did I solve a similar HMAC issue before?
Suggest skills I should create from my recent chats
List my recent Codex sessions
Get the full session where I fixed the GStreamer pipeline stall
```

### 4. Optional: Rebuild The Derived Index

```text
refresh_history_index(rebuild=true)
```

This deletes and recreates only the derived SQLite graph database. It never modifies Codex or Claude history files.

## Tools

| Tool | What it does |
|------|-------------|
| `search_history` | Hybrid search across past sessions. Uses keyword/fuzzy search plus graph relevance. |
| `search_graph` | Relationship-oriented graph search for related bugs, APIs, commands, files, and topics. |
| `suggest_skills` | Proposes evidence-backed reusable skill ideas from repeated chat patterns. |
| `list_sessions` | Lists sessions with titles, dates, sources, and message counts. |
| `get_session` | Retrieves a bounded portion of a specific session. |
| `refresh_history_index` | Manually refreshes or rebuilds the derived local graph index. |

## Example Outputs

### `search_history`

```text
Found 2 session(s) matching "CUDA illegal address":

-- Result 1 ------------------------------------------
Source : CODEX
Session: Fix CUDA kernel crash
Date   : 2026-05-07
ID     : rollout-2026-05-07...

[YOU ASKED]
Fix CUDA illegal address after kernel launch

[ANSWER]
Add synchronization around the kernel launch and rerun the focused pytest case.
```

### `search_graph`

```text
Found 1 graph result(s) matching "similar Redis migration timeout":

-- Graph Result 1 ------------------------------------
Source : CLAUDE
Session: Redis migration debugging
Why    : Matched extracted topics: topic:Redis, error:timeout
Related: command:docker compose logs api, api:redis.asyncio
```

### `suggest_skills`

```text
1. Azure Deployment Troubleshooting
   ID: skill-4f3a1b2c9e10
   Slug: azure-deployment-troubleshooting
   Confidence: 0.84
   When to use: azure deployment, app service logs, az webapp
   Evidence: 4 sessions, 2 source(s)
   Why: Repeated deployment/debug workflow with recurring commands and failure modes.
```

Skill suggestions are not generated skill files. They are ranked, evidence-backed ideas that you can review before creating an actual `SKILL.md`.

## Supported History Sources

| Source | Default location |
|--------|------------------|
| OpenAI Codex CLI | `~/.codex/sessions/` |
| Claude Code | `~/.claude/projects/` |

Override defaults with environment variables:

```json
{
  "mcpServers": {
    "agent-history": {
      "command": "python3",
      "args": ["-m", "agent_history_mcp"],
      "env": {
        "CODEX_PATH": "/custom/path/.codex",
        "CLAUDE_PATH": "/custom/path/.claude",
        "AGENT_HISTORY_GRAPH_DB": "/custom/path/history_graph.sqlite"
      }
    }
  }
}
```

If `AGENT_HISTORY_GRAPH_DB` is not set, the graph database is created at:

```text
~/.agent-history-mcp/history_graph.sqlite
```

## How It Works

```text
Claude Code or another MCP client
    calls an MCP tool
agent-history-mcp
    refreshes the local graph index for changed JSONL files
    parses Codex and Claude sessions into one message shape
    searches FTS5/fuzzy index
    searches persistent graph index
    returns concise excerpts, evidence, and graph explanations
```

The graph index extracts deterministic local entities:

- sessions
- messages and Q/A turns
- technical topics
- file paths
- commands
- package/API names
- error strings

It stores deterministic `EXTRACTED` relationships:

- session contains message
- message mentions topic/path/command/API/error
- question answered by assistant response
- topics co-occur in a Q/A pair
- sessions relate through shared extracted topics

## New Chat Updates

Every MCP tool call performs a lightweight refresh:

1. Discover Codex and Claude JSONL files.
2. Compare known files by path, size, and modified time.
3. Parse only new or changed files into the graph index.
4. Remove indexed rows for deleted history files.
5. Invalidate the in-memory keyword cache only when files changed.

New chats become searchable the next time Claude or Codex calls one of the MCP tools.

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

### Local Benchmark Results

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

## Security And Privacy

- History sources are read-only: `~/.codex` and `~/.claude` are never modified.
- Indexing is local-only: no network calls or model calls are used.
- Credentials are not read: `~/.codex/auth.json` is never accessed.
- The derived graph database can contain extracted terms and message excerpts for local search.

If past conversations contain secrets, search can surface them because it searches your local history.

## Limitations

- Deterministic graph extraction can be noisy, especially for broad or generic topics.
- Skill suggestions are candidates, not guaranteed complete skills.
- The first index build may take time on large histories.
- Search quality depends on the structure and content of your saved Codex and Claude JSONL files.

## Development

Run tests:

```bash
python -m unittest discover -s tests
```

Compile check:

```bash
python -m py_compile src/agent_history_mcp/*.py tests/*.py
```

Packaging check:

```bash
python -m pip install . --dry-run --no-deps
```

Benchmark check:

```bash
python benchmarks/benchmark_retrieval.py
```

## File Structure

```text
agent-history-mcp/
  pyproject.toml
  README.md
  LICENSE
  benchmarks/
    benchmark_retrieval.py
  src/
    agent_history_mcp/
      __init__.py
      __main__.py
      graph.py
      parsers.py
      search.py
      server.py
      skills.py
  tests/
    test_graph.py
    test_skills.py
```

## License

MIT
