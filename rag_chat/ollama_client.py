"""Ollama の HTTP API を標準ライブラリだけで呼ぶ。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, host: str):
        self.host = host.rstrip("/")

    def _post(self, path: str, payload: dict, timeout: float = 600):
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise OllamaError(f"Ollama {path} が失敗しました ({e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise OllamaError(f"Ollama ({self.host}) に接続できません: {e.reason}") from e

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=5) as r:
                return [m["name"] for m in json.load(r).get("models", [])]
        except (urllib.error.URLError, OSError) as e:
            raise OllamaError(f"Ollama ({self.host}) に接続できません: {e}") from e

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        with self._post("/api/embed", {"model": model, "input": texts}) as r:
            return json.load(r)["embeddings"]

    def capabilities(self, model: str) -> list[str]:
        with self._post("/api/show", {"model": model}, timeout=10) as r:
            return json.load(r).get("capabilities", [])

    def chat_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        """回答本文だけを流す。

        思考専用のモデル (qwen3:4b = Thinking-2507 版など) は think=false を無視して思考を本文に
        混ぜてくるため、思考に対応したモデルには think=true を送って thinking 欄に分離させ、
        本文 (content) だけを返す。思考する分だけ遅いので、既定は非思考モデルにしている。
        """
        think = "thinking" in self.capabilities(model)
        payload = {"model": model, "messages": messages, "stream": True, "think": think,
                   "options": {"temperature": 0}}
        with self._post("/api/chat", payload) as r:
            for line in r:
                if not line.strip():
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise OllamaError(data["error"])
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break


def has_model(installed: list[str], name: str) -> bool:
    """"bge-m3" と "bge-m3:latest" を同一視して判定する。"""
    want = name if ":" in name else f"{name}:latest"
    return any(m == name or m == want for m in installed)
