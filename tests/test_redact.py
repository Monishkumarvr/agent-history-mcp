import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_history_mcp.redact import redact


class RedactTests(unittest.TestCase):
    def test_masks_common_secrets(self):
        text = (
            "export OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456\n"
            "token ghp_abcdefghijklmnopqrstuvwxyz0123456789\n"
            "aws AKIAABCDEFGHIJKLMNOP\n"
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz.123\n"
            "password=hunter2secret\n"
            "postgres://admin:supersecret@db.local/app\n"
        )
        out = redact(text)
        for secret in ("sk-proj-abc", "ghp_abc", "AKIAABCD", "abcdefghijklmnopqrstuvwxyz.123", "hunter2secret", "supersecret"):
            self.assertNotIn(secret, out)
        self.assertIn("postgres://admin:[REDACTED]@db.local/app", out)

    def test_leaves_normal_text_alone(self):
        text = "Run pytest tests/test_api.py and check src/api.py"
        self.assertEqual(redact(text), text)

    def test_can_be_disabled(self):
        with mock.patch.dict(os.environ, {"AGENT_HISTORY_REDACT": "0"}):
            self.assertIn("ghp_", redact("ghp_abcdefghijklmnopqrstuvwxyz0123456789"))


if __name__ == "__main__":
    unittest.main()
