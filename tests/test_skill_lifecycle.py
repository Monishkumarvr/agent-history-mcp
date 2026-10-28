import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_history_mcp.graph import HistoryGraphIndex
from agent_history_mcp.skill_store import discover_installed_skills, install_skill, validate_skill_md
from agent_history_mcp.skills import (
    draft_skill_md,
    find_candidate,
    format_skill_health,
    skill_health,
    suggest_skill_candidates,
)
from agent_history_mcp.workflows import mine_workflows

from fixtures import all_adapters, write_claude_session, write_codex_session, write_opencode_db


def release_flow(project: str) -> list[tuple]:
    return [
        ("user", "Release the package to PyPI"),
        ("read", f"{project}/pyproject.toml"),
        ("bash", "python -m build", "built", 0),
        ("bash", "twine check dist/*", "PASSED", 0),
        ("edit", f"{project}/CHANGELOG.md"),
        ("bash", "git tag v1.2.0", "", 0),
        ("bash", "twine upload dist/*", "HTTPError: 400 File already exists", 1),
        ("bash", "twine upload --skip-existing dist/*", "uploaded", 0),
    ]


class WorkflowMiningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_claude_session(self.root, "c1", "/w/a", release_flow("/w/a"), title="Release a")
        write_codex_session(self.root, "rollout-2026-05-08-x", "/w/b", release_flow("/w/b"), start="2026-05-08T10:00:00Z")
        write_opencode_db(self.root, [("o1", "/w/c", release_flow("/w/c"))])
        for i in range(8):
            write_claude_session(self.root, f"noise{i}", "/w/a", [
                ("user", f"tweak {i}"), ("bash", "pytest -q", "ok", 0), ("edit", "/w/a/x.py"), ("bash", "pytest -q", "ok", 0),
            ], start=f"2026-05-1{i}T10:00:00Z")
        self.index = HistoryGraphIndex(self.root / "graph.sqlite")
        self.index.refresh(adapters=all_adapters(self.root))

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_mines_cross_agent_closed_pattern(self):
        patterns = mine_workflows(self.index, min_sessions=2)
        self.assertTrue(patterns)
        top = patterns[0]
        self.assertEqual(top.steps, (
            "shell:python -m build", "shell:twine check", "edit:.md", "shell:git tag", "shell:twine upload",
        ))
        self.assertEqual(top.sources, {"claude", "codex", "opencode"})
        self.assertEqual(top.scope, "global")
        self.assertTrue(any("twine upload --skip-existing" in fix for _sig, _err, fix in top.failures))
        # Subsumed sub-sequences with the same sessions are not reported separately.
        self.assertFalse(any(p.steps == ("shell:python -m build", "shell:twine check", "edit:.md") for p in patterns))

    def test_ambient_patterns_are_dropped(self):
        patterns = mine_workflows(self.index, min_sessions=2)
        self.assertFalse(any(p.steps == ("shell:pytest", "edit:.py", "shell:pytest") for p in patterns))

    def test_candidates_draft_and_overlap(self):
        candidates = suggest_skill_candidates(self.index, None, 5, 2, include_workflows=True, installed=[])
        workflow = next(c for c in candidates if c.kind == "workflow")
        self.assertEqual(workflow.action, "create")
        self.assertIn("release", workflow.slug)

        draft = draft_skill_md(workflow)
        meta, errors = validate_skill_md(draft)
        self.assertEqual(errors, [])
        self.assertEqual(meta["name"], workflow.slug)
        self.assertIn("`twine upload --skip-existing dist/*`", draft)
        self.assertIn("## Known failure modes", draft)
        self.assertIs(find_candidate(self.index, workflow.candidate_id), workflow)

        home = self.root / "home"
        skill_dir = home / ".claude" / "skills" / "pypi-release"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: pypi-release\ndescription: Release a Python package to PyPI with build, twine check, "
            "changelog, git tag and twine upload.\n---\nSteps...\n",
            encoding="utf-8",
        )
        installed = discover_installed_skills(None, home)
        again = suggest_skill_candidates(self.index, None, 5, 2, include_workflows=True, installed=installed)
        workflow = next(c for c in again if c.kind == "workflow")
        self.assertIn(workflow.action, ("skip", "update"))
        self.assertEqual(workflow.overlap_with, "pypi-release")


class InstallAndHealthTests(unittest.TestCase):
    def test_install_is_dry_run_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            content = "---\nname: demo-skill\ndescription: Demo. Use when testing.\n---\n\n# Demo\n\n1. Do it.\n"
            preview = install_skill(None, content, home=home)
            self.assertIn("Dry run only", preview)
            self.assertFalse((home / ".claude" / "skills" / "demo-skill" / "SKILL.md").exists())

            install_skill(None, content, home=home, apply=True)
            self.assertTrue((home / ".claude" / "skills" / "demo-skill" / "SKILL.md").exists())
            self.assertTrue((home / ".agents" / "skills" / "demo-skill" / "SKILL.md").exists())

            changed = content.replace("Do it.", "Do it better.")
            refused = install_skill(None, changed, home=home, apply=True)
            self.assertIn("pass overwrite=true", refused)
            self.assertIn("Do it.", (home / ".claude" / "skills" / "demo-skill" / "SKILL.md").read_text(encoding="utf-8"))

            invalid = install_skill(None, "---\nname: Bad Name\n---\nbody", home=home)
            self.assertIn("not valid", invalid)

    def test_health_detects_usage_and_missed_triggers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            skill_dir = home / ".claude" / "skills" / "redis-migration"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: redis-migration\ndescription: Migrate Redis clients.\n---\nRun `kubectl rollout restart` after.\n",
                encoding="utf-8",
            )
            write_claude_session(root, "used", "/w/a", [
                ("user", "Do the redis migration for billing"), ("skill", "redis-migration"), ("bash", "pytest", "ok", 0),
            ])
            write_codex_session(root, "rollout-2026-05-09-missed", "/w/b", [
                ("user", "Start the redis migration for orders"), ("bash", "pytest", "fail", 1),
            ], start="2026-05-09T10:00:00Z")
            index = HistoryGraphIndex(root / "graph.sqlite")
            try:
                index.refresh(adapters=all_adapters(root))
                report = skill_health(index, discover_installed_skills(None, home), days_back=None)
            finally:
                index.close()
            health = report[0]
            self.assertEqual(health.by_source["claude"], 1)
            self.assertEqual([m["source"] for m in health.missed], ["codex"])
            self.assertIn("kubectl rollout", health.stale_commands)
            self.assertIn("revise description", health.recommendation)
            self.assertIn("missed triggers: 1", format_skill_health(report, None))


if __name__ == "__main__":
    unittest.main()
