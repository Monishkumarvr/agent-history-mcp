import json
import tempfile
import time
import unittest
from pathlib import Path

from codex_mcp.graph import HistoryGraphIndex, extract_graph_terms


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _codex_msg(role: str, text: str, ts: str = "2026-05-07T10:00:00Z") -> dict:
    return {
        "type": "response_item",
        "timestamp": ts,
        "payload": {
            "role": role,
            "content": [{"type": "text", "text": text}],
        },
    }


class GraphExtractionTests(unittest.TestCase):
    def test_extracts_local_first_graph_terms(self):
        text = (
            "Fix CUDA illegal address in src/kernel.cu. "
            "Run `pytest tests/test_cuda.py`. "
            "ValueError: bad launch config from torch.cuda.synchronize"
        )

        terms = extract_graph_terms(text)
        labels = {(term.kind, term.norm) for term in terms}

        self.assertIn(("error", "illegal address in src/kernel"), labels)
        self.assertIn(("path", "src/kernel.cu"), labels)
        self.assertIn(("command", "pytest tests/test_cuda.py"), labels)
        self.assertIn(("api", "torch.cuda.synchronize"), labels)


class GraphRefreshTests(unittest.TestCase):
    def test_incremental_refresh_search_and_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / ".codex"
            session_dir = codex / "sessions" / "2026" / "05" / "07"
            session_dir.mkdir(parents=True)
            session_file = session_dir / "rollout-test-session.jsonl"
            db_path = root / "graph.sqlite"

            _append_jsonl(session_file, _codex_msg("user", "Fix CUDA illegal address after kernel launch"))
            _append_jsonl(
                session_file,
                _codex_msg(
                    "assistant",
                    "The fix was to add cudaDeviceSynchronize around src/kernel.cu and run pytest tests/test_cuda.py.",
                ),
            )

            index = HistoryGraphIndex(db_path)
            try:
                first = index.refresh(codex, None)
                self.assertEqual(first.files_added, 1)
                self.assertEqual(first.sessions_indexed, 1)

                unchanged = index.refresh(codex, None)
                self.assertEqual(unchanged.files_unchanged, 1)
                self.assertEqual(unchanged.sessions_indexed, 0)

                hits = index.search("CUDA kernel illegal address", ["codex"], max_results=3)
                self.assertTrue(hits)
                self.assertEqual(hits[0].session_id, "rollout-test-session")
                self.assertTrue(hits[0].matched_topics)

                time.sleep(0.01)
                _append_jsonl(
                    session_file,
                    _codex_msg("user", "Now handle Redis migration timeout", "2026-05-07T10:05:00Z"),
                )
                _append_jsonl(
                    session_file,
                    _codex_msg("assistant", "Use redis.asyncio and increase the migration timeout.", "2026-05-07T10:06:00Z"),
                )

                changed = index.refresh(codex, None)
                self.assertEqual(changed.files_changed, 1)
                self.assertEqual(changed.sessions_indexed, 1)
                redis_hits = index.search("redis migration timeout", ["codex"], max_results=3)
                self.assertTrue(redis_hits)

                session_file.unlink()
                removed = index.refresh(codex, None)
                self.assertEqual(removed.files_removed, 1)
                self.assertFalse(index.search("CUDA kernel illegal address", ["codex"], max_results=3))
            finally:
                index.close()


if __name__ == "__main__":
    unittest.main()
