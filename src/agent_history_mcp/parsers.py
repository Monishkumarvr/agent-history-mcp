"""
Compatibility layer for the original Codex/Claude parser API.

The implementations now live in agent_history_mcp.sources; every parser emits
the unified session shape documented in agent_history_mcp.model:
  {session_id, session_title, session_date, source, messages: [...], events: [...], ...}
where each message has:
  {role, text, ts, session_id, session_title, session_date, source}
"""

from .sources import read_jsonl as _read_jsonl  # noqa: F401
from .sources.claude import (  # noqa: F401
    iter_claude_session_files,
    parse_claude_session_file,
    parse_claude_sessions,
)
from .sources.codex import (  # noqa: F401
    iter_codex_session_files,
    load_codex_title_index,
    parse_codex_session_file,
    parse_codex_sessions,
)
