# agent-history-mcp

An MCP server that lets Claude Code search your past AI coding conversations — across both **OpenAI Codex CLI** and **Claude Code** — so you can pull relevant context from previous sessions on demand.

## What it does

Exposes three tools to Claude via the [Model Context Protocol](https://modelcontextprotocol.io):

| Tool | What it does |
|------|-------------|
| `search_history` | Keyword search across all past sessions. Returns matching excerpts with surrounding context. |
| `list_sessions` | List all sessions with titles, dates, and message counts. |
| `get_session` | Retrieve the full conversation for a specific session. |

**Example prompts you can use in Claude:**
- *"Search my history for CUDA illegal address"*
- *"Did I solve a similar sync issue before? Search for HMAC"*
- *"List my recent Codex sessions"*
- *"Get the full session where I fixed the GStreamer pipeline stall"*

## Supported history sources

| Source | Location |
|--------|----------|
| OpenAI Codex CLI | `~/.codex/sessions/` |
| Claude Code | `~/.claude/projects/` |

Both are read-only. Nothing is uploaded anywhere — all search happens locally.

## Requirements

- Python 3.10+
- [OpenAI Codex CLI](https://github.com/openai/codex) and/or [Claude Code](https://claude.ai/code) with existing session history

## Installation

### Option 1 — pip from GitHub (recommended)

```bash
pip install git+https://github.com/monishkumarvr/agent-history-mcp.git
```

### Option 2 — pip from local clone

```bash
git clone https://github.com/monishkumarvr/agent-history-mcp.git
cd agent-history-mcp
pip install .
```

## Setup

### 1. Register with Claude Code

Add to `~/.claude/.mcp.json` (create the file if it doesn't exist):

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

That's it. Claude Code auto-launches the server when it starts.

### 2. Verify it's running

Start Claude Code, then ask:

```
list my recent sessions
```

If you see a table of sessions, it's working.

### 3. Override history paths (optional)

By default the server auto-detects `~/.codex` and `~/.claude`. Override with environment variables if your paths differ:

```json
{
  "mcpServers": {
    "agent-history": {
      "command": "python3",
      "args": ["-m", "codex_mcp"],
      "env": {
        "CODEX_PATH": "/custom/path/.codex",
        "CLAUDE_PATH": "/custom/path/.claude"
      }
    }
  }
}
```

## Usage

Once registered, the tools are available automatically in any Claude Code session. Claude will use them when relevant, or you can ask directly:

```
Search my history for "docker compose"
Search codex history for the redis migration
List my last 20 Claude sessions
Get session <id from list_sessions>
```

Use `sources` to narrow the search:
```
Search only Codex history for "pytest fixture"
Search only Claude history for "GStreamer pipeline"
```

## Security & Privacy

- **Read-only**: the server never writes to disk
- **Local-only**: no network calls; the MCP server is a subprocess communicating over stdin/stdout
- **No credentials accessed**: `~/.codex/auth.json` is never read
- **Conversations stay on your machine**: excerpts are passed to Claude in your active session only, same as pasting text

> **Note:** If you previously typed sensitive values (API keys, passwords, tokens) directly into a conversation, `search_history` could surface that text when Claude searches relevant terms. This is expected behaviour — it's your own history. Be mindful when sharing screen recordings or MCP logs.

## How it works

```
Claude Code
    ↓  calls MCP tool (JSON-RPC over stdin/stdout)
agent-history server (python3 -m codex_mcp)
    ↓  reads JSONL files
~/.codex/sessions/**/*.jsonl       (Codex CLI sessions)
~/.claude/projects/**/*.jsonl      (Claude Code sessions)
    ↓  keyword search, deduplicate by session
Returns excerpts + context window to Claude
```

Token-efficient by design: `search_history` returns ~500 tokens of excerpts rather than injecting full sessions (which can be 200k+ tokens each).

## File structure

```
codex_mcp/
├── __init__.py
├── __main__.py     # python -m codex_mcp entry point
├── server.py       # FastMCP server + 3 tools
├── parsers.py      # Codex + Claude JSONL parsers → unified format
└── search.py       # keyword search + result formatting
pyproject.toml
README.md
.gitignore
```

## Contributing

Issues and PRs welcome. The parsers are the most likely thing to need updates as Codex CLI and Claude Code evolve their session formats.

## License

MIT
