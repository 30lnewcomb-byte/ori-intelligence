"""Qwen Mentor interface for Ori.

The Mentor is deliberately an adapter, not Ori's identity or runtime model.
Set ORI_MENTOR_URL to an OpenAI-compatible chat endpoint when a Qwen Mentor
service is available. With no endpoint configured, this module stays offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class MentorResult:
    corrected_answer: str
    rationale: str = ""


class MentorUnavailable(RuntimeError):
    pass


class QwenMentor:
    def __init__(self, endpoint: str | None = None, model: str | None = None, timeout: int = 60):
        self.endpoint = endpoint or os.getenv("ORI_MENTOR_URL", "").strip()
        self.model = model or os.getenv("ORI_MENTOR_MODEL", "Qwen/Qwen3-0.6B")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint)

    def review(self, prompt: str, ori_answer: str, expected: str) -> MentorResult:
        if not self.endpoint:
            raise MentorUnavailable("ORI_MENTOR_URL is not configured")

        system = (
            "You are Ori's Mentor. Review the small Ori model's answer. "
            "Return JSON with corrected_answer and rationale. Keep Ori's identity "
            "separate from the Mentor and do not invent device actions."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"prompt": prompt, "ori_answer": ori_answer, "expected": expected})},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"]
        result = json.loads(content) if isinstance(content, str) else content
        return MentorResult(
            corrected_answer=str(result.get("corrected_answer", expected)),
            rationale=str(result.get("rationale", "")),
        )
