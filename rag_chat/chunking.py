"""セクションを検索単位 (チャンク) に分割する。"""

from __future__ import annotations

import re


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """段落の区切りを優先しつつ、おおよそ size 文字のチャンクに分ける。

    1 段落が size を超える場合は文字数で切り、前のチャンクと overlap 文字重ねる。
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paras:
        if len(p) <= size:
            pieces.append(p)
        else:
            step = max(1, size - overlap)
            pieces.extend(p[i : i + size] for i in range(0, len(p), step) if p[i : i + size].strip())

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 2 + len(piece) > size:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks
