"""設定 (config.toml) と、UI で選んだフォルダ履歴の読み書き。"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"
INDEX_DIR = PROJECT_ROOT / ".rag_index"
STATE_PATH = INDEX_DIR / "state.json"


@dataclass
class Config:
    data_dir: str = "sample_data"
    ollama_host: str = "http://127.0.0.1:11434"
    embed_model: str = "bge-m3"
    chat_model: str = "qwen3:4b-instruct"
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 80
    bm25_weight: float = 0.3
    extra: dict = field(default_factory=dict)


def load_config(path: Path = CONFIG_PATH) -> Config:
    """config.toml を読む。無ければ既定値 (config.example.toml と同じ内容)。"""
    if not path.exists():
        return Config()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    ollama = raw.get("ollama", {})
    search = raw.get("search", {})
    d = Config()
    return Config(
        data_dir=raw.get("data_dir", d.data_dir),
        ollama_host=ollama.get("host", d.ollama_host).rstrip("/"),
        embed_model=ollama.get("embed_model", d.embed_model),
        chat_model=ollama.get("chat_model", d.chat_model),
        top_k=int(search.get("top_k", d.top_k)),
        chunk_size=int(search.get("chunk_size", d.chunk_size)),
        chunk_overlap=int(search.get("chunk_overlap", d.chunk_overlap)),
        bm25_weight=float(search.get("bm25_weight", d.bm25_weight)),
        extra=raw,
    )


def resolve_dir(p: str) -> Path:
    """相対パスはプロジェクトルート基準で解決する。"""
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_state() -> dict:
    """UI の状態 (選択中のフォルダ等)。無い・壊れているときは空。"""
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(**updates) -> None:
    state = load_state() | updates
    INDEX_DIR.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
