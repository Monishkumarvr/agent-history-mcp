"""Entry point: python -m agent_history_mcp (runs the MCP server; pass a subcommand for the CLI)."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["serve"]))
