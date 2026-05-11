# agent-history-mcp

An MCP server that lets Claude Code and Codex search past AI coding conversations across both **OpenAI Codex CLI** and **Claude Code**.

It now combines two local search layers:

- keyword search with SQLite FTS5 and fuzzy fallback
- a persistent local knowledge graph inspired by Graphify's "cache changed inputs, search relationships first" principle

No chat history is uploaded. Codex and Claude history files are read-only; only the derived local index is written.

## Tools

| Tool | What it does |
|------|-------------|
| `search_history` | Hybrid search across past sessions. Uses keyword/fuzzy search plus graph relevance. |
| `search_graph` | Relationship-oriented graph search for related bugs, APIs, commands, files, and topics. |
| `suggest_skills` | Propose reusable skill ideas from repeated chat patterns, with evidence and trigger phrases. |
| `list_sessions` | List sessions with titles, dates, sources, and message counts. |
| `get_session` | Retrieve a bounded portion of a specific session. |
| `refresh_history_index` | Manually refresh or rebuild the derived local graph index. |

Example prompts:

```text
Search my history for CUDA illegal address
Use graph search for a similar Redis migration timeout
Did I solve a similar HMAC issue before?
Suggest skills I should create from my recent chats
List my recent Codex sessions
Get the full session where I fixed the GStreamer pipeline stall
Refresh the history index
```

## Supported History Sources

| Source | Location |
|--------|----------|
| OpenAI Codex CLI | `~/.codex/sessions/` |
| Claude Code | `~/.claude/projects/` |

Override defaults with:

```json
{
  "mcpServers": {
    "agent-history": {
      "command": "python3",
      "args": ["-m", "codex_mcp"],
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

## Installation

### pip from GitHub

```bash
pip install git+https://github.com/monishkumarvr/agent-history-mcp.git
```

### pip from local clone

```bash
git clone https://github.com/monishkumarvr/agent-history-mcp.git
cd agent-history-mcp
pip install .
```

## Setup With Claude Code

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "agent-history": {
      "command": "python3",
      "args": ["-m", "codex_mcp"]
    }
  }
}
```

Claude Code starts the server automatically when needed.

## How Search Works

```text
Claude Code / Codex
    calls MCP tool
agent-history server
    refreshes graph index for changed JSONL files
    parses Codex + Claude sessions into one message shape
    searches FTS5/fuzzy index
    searches persistent graph index
    merges keyword and graph-ranked results
returns concise Q/A excerpts and graph evidence
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

## Skill Suggestions

`suggest_skills` analyzes the existing graph index and returns ranked skill ideas. It does not write `SKILL.md` files and does not call a model.

Each suggestion includes:

- proposed skill name and stable candidate ID
- slug
- confidence score
- trigger phrases
- recurring terms
- supporting session IDs and dates
- why the skill is worth making
- suggested boundaries to keep the skill narrow

Example:

```text
suggest_skills(max_candidates=5, min_sessions=2)
```

Optional filters:

```text
suggest_skills(sources=["codex"], days_back=30)
```

## New Chat Updates

Every MCP tool call performs a lightweight refresh:

1. Discover Codex and Claude JSONL files.
2. Compare known files by path, size, and modified time.
3. Parse only new or changed files into the graph index.
4. Remove indexed rows for deleted history files.
5. Invalidate the in-memory keyword cache only when files changed.

No background daemon is required. New chats become searchable the next time Claude or Codex calls one of the MCP tools.

Manual refresh:

```text
refresh_history_index()
```

Full derived-index rebuild:

```text
refresh_history_index(rebuild=true)
```

This deletes only the derived SQLite graph database, never Codex or Claude history.

## Security And Privacy

- Read-only history sources: `~/.codex` and `~/.claude` are never modified.
- Local-only indexing: no network calls or model calls are used for graph extraction.
- No credentials accessed: `~/.codex/auth.json` is never read.
- Derived index only: the graph database contains extracted terms and message excerpts for local search.

If past conversations contain secrets, search can still surface them because it searches your local history.

## Development

Run tests:

```bash
python -m unittest discover -s tests
```

Compile check:

```bash
python -m py_compile codex_mcp/parsers.py codex_mcp/search.py codex_mcp/graph.py codex_mcp/server.py
```

## File Structure

```text
codex_mcp/
  __init__.py
  __main__.py
  graph.py       # persistent local graph index and graph search
  parsers.py     # Codex + Claude JSONL parsers
  search.py      # keyword/fuzzy search and result formatting
  server.py      # FastMCP server and tools
  skills.py      # chat-derived reusable skill suggestions
tests/
  test_graph.py
  test_skills.py
```

## License

MIT
