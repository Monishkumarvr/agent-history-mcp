"""
Mask likely secrets before history text leaves the server.

Past sessions often contain pasted tokens and keys. Every MCP tool output passes
through redact(); set AGENT_HISTORY_REDACT=0 to disable.
"""

from __future__ import annotations

import os
import re

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED:private-key]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED:aws-key]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "[REDACTED:github-token]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"), "[REDACTED:github-token]"),
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"), "[REDACTED:api-key]"),
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"), "[REDACTED:slack-token]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[REDACTED:google-key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[REDACTED:jwt]"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{20,}"), r"\1 [REDACTED]"),
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key|client[_-]?secret)"
            r"[\"']?\s*[:=]\s*[\"']?)([^\s\"',;]{6,})"
        ),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^:/\s@]+:)([^@\s]{3,})@"), r"\1[REDACTED]@"),
]


def redaction_enabled() -> bool:
    return os.environ.get("AGENT_HISTORY_REDACT", "1").strip().lower() not in ("0", "false", "no", "off")


def redact(text: str) -> str:
    if not text or not redaction_enabled():
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
