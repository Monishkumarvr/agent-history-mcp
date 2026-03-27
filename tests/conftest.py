"""
Pytest configuration: stub out the 'mcp' package so tests run
without needing the full mcp[cli] install.
"""

import sys
import types


def _install_mcp_stub():
    """Install minimal mcp stub modules before any test imports server.py."""
    if "mcp" in sys.modules:
        return

    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")

    class _FakeMCP:
        def __init__(self, name, instructions=""):
            self.name = name

        def tool(self):
            return lambda f: f

        def run(self):
            pass

    fastmcp_mod.FastMCP = _FakeMCP
    mcp_mod.server = server_mod
    server_mod.fastmcp = fastmcp_mod

    sys.modules["mcp"] = mcp_mod
    sys.modules["mcp.server"] = server_mod
    sys.modules["mcp.server.fastmcp"] = fastmcp_mod


_install_mcp_stub()
