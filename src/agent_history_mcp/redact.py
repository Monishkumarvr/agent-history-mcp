"""
Mask likely secrets before history text leaves the server.

Past sessions often contain pasted tokens and keys. Every MCP tool output passes
through redact(); set AGENT_HISTORY_REDACT=0 to disable.
"""

from __future__ import annotations

import os
import re

# Each pattern carries literals it cannot match without. A plain `in` check skips the regex
# scan for nearly every text; 11 full scans were the biggest cost of parsing large sessions.
# Literals for (?i) patterns are lowercase and checked against the lowercased text.
_PATTERNS: list[tuple[re.Pattern, str, tuple[str, ...]]] = [
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
        "[REDACTED:private-key]",
        ("-----BEGIN",),
    ),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED:aws-key]", ("AKIA", "ASIA")),
    (
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
        "[REDACTED:github-token]",
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
    ),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"), "[REDACTED:github-token]", ("github_pat_",)),
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"), "[REDACTED:api-key]", ("sk-",)),
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"), "[REDACTED:slack-token]", ("xox",)),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[REDACTED:google-key]", ("AIza",)),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "[REDACTED:jwt]",
        ("eyJ",),
    ),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{20,}"), r"\1 [REDACTED]", ("bearer",)),
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key|client[_-]?secret)"
            r"[\"']?\s*[:=]\s*[\"']?)([^\s\"',;]{6,})"
        ),
        r"\1[REDACTED]",
        ("key", "secret", "token", "passw", "pwd"),
    ),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^:/\s@]+:)([^@\s]{3,})@"), r"\1[REDACTED]@", ("://",)),
]


def redaction_enabled() -> bool:
    return os.environ.get("AGENT_HISTORY_REDACT", "1").strip().lower() not in ("0", "false", "no", "off")


def redact(text: str) -> str:
    if not text or not redaction_enabled():
        return text
    lowered = None
    for pattern, replacement, literals in _PATTERNS:
        haystack = text
        if pattern.flags & re.IGNORECASE:
            if lowered is None:
                lowered = text.lower()
            haystack = lowered
        if any(literal in haystack for literal in literals):
            text = pattern.sub(replacement, text)
            lowered = None
    return text
