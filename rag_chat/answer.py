"""検索結果から、出典番号付きの短い回答を生成する。"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from collections.abc import Iterator

from .config import Config
from .index import Hit
from .ollama_client import Ollama

SYSTEM_PROMPT = """あなたは社内マニュアルの案内係です。
与えられた「資料」だけを根拠に、日本語で簡潔に答えてください。
- 資料はすべて読み、質問に当てはまる記述を探す（最初の資料とは限らない）
- 質問と資料で言葉が違っても、意味が当てはまれば答えてよい（例：「お金が戻る」→「振込」、「風邪」→「体調不良」）
- 根拠にした資料の番号を文末に [1] のように付ける
- 資料のどこにも関係する記述が無いときだけ「資料には見当たりません」とだけ答える（番号は付けない）
- 手順は箇条書きにする。前置きや繰り返しは書かない"""


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    blocks = []
    for i, h in enumerate(hits, start=1):
        where = f"{Path(h.folder).name}/{h.path} {h.location}".strip()
        blocks.append(f"[{i}] ({where})\n{h.text}")
    user = "資料:\n\n" + "\n\n".join(blocks) + f"\n\n質問: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def answer_stream(cfg: Config, question: str, hits: list[Hit]) -> Iterator[str]:
    if not hits:
        yield "資料が索引に入っていません。フォルダを選んで索引を作成してください。"
        return
    yield from Ollama(cfg.ollama_host).chat_stream(cfg.chat_model, build_messages(question, hits))


_CITE = re.compile(r"\[(\d+)\]")


def cited_sources(text: str, hits: list[dict]) -> list[tuple[int, dict]]:
    """回答が [n] で引用した資料だけを、番号順に返す (範囲外の番号は無視)。"""
    nums = sorted({int(n) for n in _CITE.findall(unicodedata.normalize("NFKC", text))})
    return [(n, hits[n - 1]) for n in nums if 1 <= n <= len(hits)]
