"""Small, dependency-free helpers for Ori's chat/tool response protocol."""

from __future__ import annotations

import json
import uuid
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Accept only a complete JSON object, optionally inside a JSON code fence."""
    candidate = text.strip()
    fence = chr(96) * 3
    if candidate.startswith(fence):
        lines = candidate.splitlines()
        if len(lines) < 3 or lines[-1].strip() != fence:
            return None
        fence_language = lines[0].strip()[3:].strip().lower()
        if fence_language not in {"", "json"}:
            return None
        candidate = "\n".join(lines[1:-1]).strip()

    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def normalize_tool_call(
    content: str,
    tools: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert a complete, allow-listed tool object into the runtime call shape."""
    if not tools:
        return content, []

    allowed: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            allowed[name] = function

    candidate = extract_json_object(content)
    if not candidate:
        return content, []

    tool_name = candidate.get("tool")
    if not isinstance(tool_name, str) or tool_name not in allowed:
        return content, []

    arguments = {
        key: value for key, value in candidate.items() if key != "tool"
    }

    if tool_name == "run_sandbox_command":
        command = arguments.get("command")
        args = arguments.get("args", [])
        if not isinstance(command, str) or not command:
            return content, []
        if not isinstance(args, list) or not all(
            isinstance(value, str) for value in args
        ):
            return content, []
        arguments = {"command": command, "args": args[:32]}

    elif tool_name == "write_workspace_file":
        path = arguments.get("path")
        file_content = arguments.get("content")
        if not isinstance(path, str) or not isinstance(file_content, str):
            return content, []
        arguments = {"path": path, "content": file_content}

    call_id = f"call_{uuid.uuid4().hex[:12]}"
    return "", [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, separators=(",", ":")),
            },
        }
    ]
