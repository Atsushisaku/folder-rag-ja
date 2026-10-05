"""CLI: 索引の作成と質問 (UI を使わない確認用)。

    uv run python -m rag_chat index <フォルダ>
    uv run python -m rag_chat ask <フォルダ> "質問"
"""

from __future__ import annotations

import argparse
import sys
import time

from .answer import answer_stream
from .config import load_config, resolve_dir
from .index import FolderIndex, MultiIndex


def main() -> int:
    ap = argparse.ArgumentParser(prog="rag_chat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_idx = sub.add_parser("index", help="索引を作成・差分更新する")
    p_idx.add_argument("folder")
    p_ask = sub.add_parser("ask", help="質問する")
    p_ask.add_argument("folder")
    p_ask.add_argument("question")
    p_ask.add_argument("--no-llm", action="store_true", help="検索結果だけ表示する")
    args = ap.parse_args()

    cfg = load_config()
    folder = resolve_dir(args.folder)
    if not folder.is_dir():
        print(f"フォルダがありません: {folder}", file=sys.stderr)
        return 1
    idx = FolderIndex(folder, cfg)

    if args.cmd == "index":
        t = time.perf_counter()
        r = idx.update(lambda i, n, name: print(f"[{i}/{n}] {name}"))
        print(f"追加 {r.added} / 更新 {r.updated} / 削除 {r.removed} / 変更なし {r.unchanged}"
              f" / チャンク {r.chunks} ({time.perf_counter() - t:.1f}s)")
        for e in r.errors:
            print("エラー:", e, file=sys.stderr)
        return 1 if r.errors else 0

    t = time.perf_counter()
    hits = MultiIndex([idx], cfg).search(args.question)
    print(f"--- 検索結果 ({time.perf_counter() - t:.2f}s)")
    for i, h in enumerate(hits, start=1):
        print(f"[{i}] {h.path} {h.location}  score={h.score:.4f}")
    if args.no_llm:
        return 0
    print("--- 回答")
    t = time.perf_counter()
    first = None
    for piece in answer_stream(cfg, args.question, hits):
        first = first or time.perf_counter()
        print(piece, end="", flush=True)
    end = time.perf_counter()
    print(f"\n--- 最初の文字まで {(first or end) - t:.1f}s / 全体 {end - t:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
