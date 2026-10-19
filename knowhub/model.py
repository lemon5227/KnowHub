"""Native provider Function Calling; credentials are read only, never copied."""

import json
import os
import threading
from pathlib import Path

import httpx


class ModelError(RuntimeError):
    pass


class Model:
    def __init__(self, *, credential_file=None, session_limit=40):
        config = {}
        filename = credential_file or os.environ.get("KNOWHUB_CREDENTIAL_FILE")
        if filename:
            raw = json.loads(Path(filename).expanduser().read_text())
            config = raw.get("model", {}).get("cloud_api", raw)
        self.key = os.environ.get("KNOWHUB_MODEL_API_KEY") or config.get("api_key")
        if not self.key:
            raise ModelError("Configure KNOWHUB_CREDENTIAL_FILE or KNOWHUB_MODEL_API_KEY")
        self.name = os.environ.get("KNOWHUB_MODEL") or config.get("model_name", "deepseek-flash")
        self.base_url = os.environ.get("KNOWHUB_BASE_URL") or config.get(
            "base_url", "https://api.deepseek.com"
        )
        self.session_limit = session_limit
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, messages, tools):
        with self.lock:
            if self.calls >= self.session_limit:
                raise ModelError("Model session request budget exhausted")
            self.calls += 1
        body = {"model": self.name, "messages": messages, "max_tokens": 1200, "temperature": 0}
        if "deepseek.com" in self.base_url:
            body["thinking"] = {"type": "disabled"}
        if tools:
            body["tools"] = tools
        try:
            response = httpx.post(
                self.base_url.rstrip("/") + "/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.key}"},
                timeout=45,
            )
            response.raise_for_status()
            data = response.json()
            # Hidden reasoning is neither fed into the public graph nor stored in runs.
            raw = data["choices"][0]["message"]
            message = {k: raw[k] for k in ("role", "content", "tool_calls") if k in raw}
            return {"message": message, "usage": data.get("usage", {})}
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"Model provider returned HTTP {exc.response.status_code}") from None
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            raise ModelError("Model request failed or returned an invalid response") from None
