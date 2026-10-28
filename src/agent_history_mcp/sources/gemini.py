"""
Gemini CLI chats: ~/.gemini/tmp/<project>/chats/session-*.jsonl (legacy *.json)

JSONL files are a replay log: a metadata record ({sessionId, projectHash,
startTime, ...}), message records keyed by "id" (type "user" | "gemini", with
content, toolCalls, tokens, model), and control records "$set", "$patch",
"$rewindTo". Legacy .json files hold one {sessionId, messages: [...]} object.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..model import SessionBuilder, output_text
from . import SourceAdapter, SourceUnit, file_token, read_jsonl


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", "") or "")
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("text") and not part.get("thought"):
            parts.append(part["text"])
    return "\n".join(parts)


def _replay(records: list[dict]) -> tuple[dict, list[dict]]:
    metadata: dict = {}
    messages: dict[str, dict] = {}
    for rec in records:
        if "$set" in rec and isinstance(rec["$set"], dict):
            updates = dict(rec["$set"])
            if isinstance(updates.get("messages"), list):
                messages = {str(m.get("id", i)): m for i, m in enumerate(updates.pop("messages")) if isinstance(m, dict)}
            metadata.update(updates)
        elif "$rewindTo" in rec:
            target = str(rec["$rewindTo"])
            if target in messages:
                keys = list(messages)
                for key in keys[keys.index(target):]:
                    messages.pop(key, None)
        elif "$patch" in rec and isinstance(rec["$patch"], dict):
            patch = rec["$patch"]
            for key in patch.get("removeIds", []) or []:
                messages.pop(str(key), None)
            for update in patch.get("updates", []) or []:
                if isinstance(update, dict) and str(update.get("id")) in messages:
                    messages[str(update["id"])].update({k: v for k, v in update.items() if k != "id"})
            if patch.get("id") and str(patch["id"]) in messages:
                messages[str(patch["id"])].update({k: v for k, v in patch.items() if k not in ("id", "updates", "removeIds", "orderIds")})
        elif isinstance(rec.get("id"), str) and "type" in rec:
            existing = messages.get(rec["id"])
            if existing is not None:
                existing.update(rec)
            else:
                messages[rec["id"]] = dict(rec)
        elif rec.get("sessionId") and "messages" in rec and isinstance(rec["messages"], list):
            metadata.update({k: v for k, v in rec.items() if k != "messages"})
            messages = {str(m.get("id", i)): m for i, m in enumerate(rec["messages"]) if isinstance(m, dict)}
        elif rec.get("sessionId"):
            metadata.update(rec)
    return metadata, list(messages.values())


def _project_dir(chats_dir: Path) -> str:
    marker = chats_dir.parent / ".project_root"
    try:
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


def parse_gemini_session_file(path: Path) -> dict | None:
    if path.suffix == ".json":
        try:
            records = [json.loads(path.read_text(encoding="utf-8", errors="replace"))]
        except (OSError, ValueError):
            return None
        records = [r for r in records if isinstance(r, dict)]
    else:
        records = read_jsonl(path)
    metadata, messages = _replay(records)
    if not messages:
        return None

    session_id = str(metadata.get("sessionId") or path.stem)
    builder = SessionBuilder("gemini", session_id, str(path))
    builder.title = str(metadata.get("summary") or "")
    directories = metadata.get("directories")
    builder.set_cwd(_project_dir(path.parent) or (directories[0] if isinstance(directories, list) and directories else ""))
    saw_tools = False

    for msg in messages:
        ts = msg.get("timestamp")
        msg_type = msg.get("type")
        text = _content_text(msg.get("content"))
        if msg_type == "user":
            builder.add_message("user", text, ts)
            if not builder.title and text:
                builder.title = text.strip().splitlines()[0][:60]
        elif msg_type in ("gemini", "model", "assistant"):
            builder.add_model(msg.get("model"))
            tokens = msg.get("tokens") or {}
            if isinstance(tokens, dict):
                builder.add_tokens(tokens.get("input"), tokens.get("output"))
            builder.add_message("assistant", text, ts)
            for call in msg.get("toolCalls") or []:
                if not isinstance(call, dict):
                    continue
                saw_tools = True
                call_id = str(call.get("id", ""))
                call_ts = call.get("timestamp") or ts
                builder.add_tool_call(call.get("name", ""), call.get("args") or {}, call_ts, call_id=call_id)
                if "result" in call or call.get("status"):
                    status = str(call.get("status", "")).lower()
                    builder.add_tool_result(
                        call_id,
                        output_text(call.get("result")),
                        call_ts,
                        is_error=True if status in ("error", "cancelled") else None,
                    )
        elif msg_type == "error":
            builder.add_signal("error", text, ts)

    builder.tools_available = saw_tools
    return builder.build()


class GeminiAdapter(SourceAdapter):
    name = "gemini"

    def discover(self) -> list[SourceUnit]:
        tmp = self.root / "tmp" if self.root else None
        if not tmp or not tmp.is_dir():
            return []
        units = []
        for chats_dir in tmp.glob("*/chats"):
            for path in sorted(chats_dir.iterdir(), reverse=True):
                if path.suffix not in (".jsonl", ".json") or not path.is_file():
                    continue
                # A migrated legacy .json has a .jsonl twin; index the replay log only.
                if path.suffix == ".json" and path.with_suffix(".jsonl").exists():
                    continue
                try:
                    units.append(SourceUnit(str(path.resolve()), self.name, file_token(path), path))
                except OSError:
                    continue
        return units

    def parse(self, unit: SourceUnit) -> dict | None:
        return parse_gemini_session_file(unit.locator)
