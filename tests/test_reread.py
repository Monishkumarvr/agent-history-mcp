import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_history_mcp import reread


class RereadGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"AGENT_HISTORY_HOME": str(root / "home")})
        self.env.start()
        self.file = root / "a.py"
        self.file.write_text("x = 1\n", encoding="utf-8")

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def read(self, session="s1", agent="", **extra):
        payload = {"session_id": session, "tool_name": "Read", "cwd": self.tmp.name,
                   "tool_input": {"file_path": "a.py", **extra}}
        if agent:
            payload["agent_id"] = agent
        return reread.check(payload)

    def test_repeat_denied_once_then_allowed(self):
        self.assertEqual(self.read(), "")
        self.assertIn("already read a.py", self.read())
        self.assertEqual(self.read(), "")  # repeating the Read gets through
        self.assertIn("already read", self.read())

    def test_change_range_subagent_and_reset_allow(self):
        self.assertEqual(self.read(), "")
        self.assertEqual(self.read(offset=10, limit=5), "")    # different range
        self.assertEqual(self.read(agent="sub1"), "")          # subagent has its own context
        self.assertEqual(self.read(session="s2"), "")
        os.utime(self.file, ns=(1, 1))                          # file changed
        self.assertEqual(self.read(), "")
        reread.reset("s1")                                       # compaction
        self.assertEqual(self.read(), "")

    def test_switch(self):
        self.assertEqual(self.read(), "")
        self.assertEqual(reread.set_enabled(False), "re-read guard: off")
        self.assertEqual(self.read(), "")  # would be a repeat, but the guard is off
        self.assertEqual(reread.set_enabled(True), "re-read guard: on")
        self.assertIn("already read", self.read())

    def test_ignores_other_tools_and_missing_files(self):
        self.assertEqual(reread.check({"session_id": "s1", "tool_name": "Bash", "tool_input": {"command": "ls"}}), "")
        self.assertEqual(self.read(), "")
        self.file.unlink()
        self.assertEqual(self.read(), "")


if __name__ == "__main__":
    unittest.main()
