"""フォルダの索引 (SQLite) の作成・差分更新と、ハイブリッド検索。

索引はプロジェクト内の .rag_index/ に置き、読み込むフォルダ (OneDrive 等) には何も書かない。
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunking import split_text
from .config import INDEX_DIR, Config
from .loaders import SUPPORTED_EXTS, load_file
from .ollama_client import Ollama

EMBED_BATCH = 16

Progress = Callable[[int, int, str], None]


@dataclass
class Hit:
    path: str  # フォルダからの相対パス
    location: str
    text: str
    score: float


@dataclass
class UpdateResult:
    added: int
    updated: int
    removed: int
    unchanged: int
    errors: list[str]
    chunks: int


def index_path_for(folder: Path) -> Path:
    key = hashlib.sha1(str(folder).lower().encode("utf-8")).hexdigest()[:12]
    return INDEX_DIR / f"{key}.sqlite"


def scan_files(folder: Path) -> list[Path]:
    """対応形式のファイルを再帰的に集める (隠しフォルダ・Office の一時ファイルは除く)。"""
    out = []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder)
        if any(part.startswith(".") for part in rel.parts) or p.name.startswith("~$"):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            out.append(p)
    return sorted(out)


@dataclass
class IndexedFolder:
    folder: str
    files: int
    chunks: int
    exists: bool


def list_indexed_folders() -> list[IndexedFolder]:
    """.rag_index/ にある索引から、読み込み済みのフォルダを一覧にする。"""
    out = []
    for db_path in sorted(INDEX_DIR.glob("*.sqlite")):
        try:
            con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
            try:
                folder = con.execute("SELECT value FROM meta WHERE key='folder'").fetchone()
                files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            finally:
                con.close()
        except sqlite3.Error:
            continue
        if folder:
            out.append(IndexedFolder(folder[0], files, chunks, Path(folder[0]).is_dir()))
    return sorted(out, key=lambda f: f.folder.lower())


# プロセス全体で 1 フォルダにつき 1 つの FolderIndex (SQLite 接続) を共有する。
# ブラウザのタブ (セッション) ごとに接続を持つと、閉じたタブの接続が残って Windows では
# 索引ファイルを削除できなくなるため。
_open: dict[str, FolderIndex] = {}
_open_lock = threading.Lock()


def open_index(folder: Path, cfg: Config) -> FolderIndex:
    with _open_lock:
        key = str(folder)
        if key not in _open:
            _open[key] = FolderIndex(folder, cfg)
        return _open[key]


def delete_index(folder: Path) -> None:
    """フォルダの索引を削除する (元のファイルには触れない)。"""
    with _open_lock:
        idx = _open.pop(str(folder), None)
        if idx is not None:
            idx.close()
    p = index_path_for(folder)
    for f in (p, p.with_name(p.name + "-journal")):
        f.unlink(missing_ok=True)


class FolderIndex:
    def __init__(self, folder: Path, cfg: Config):
        self.folder = folder
        self.cfg = cfg
        self.ollama = Ollama(cfg.ollama_host)
        INDEX_DIR.mkdir(exist_ok=True)
        self.db = sqlite3.connect(index_path_for(folder), check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER);
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, path TEXT, location TEXT, text TEXT, emb BLOB);
            CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
            """
        )
        self._reset_if_settings_changed()
        self._bm25: BM25 | None = None
        self._rows: list[tuple[str, str, str]] = []
        self._mat: np.ndarray | None = None

    # ---------- 索引の作成・更新 ----------

    def _settings_key(self) -> str:
        c = self.cfg
        return f"{c.embed_model}|{c.chunk_size}|{c.chunk_overlap}"

    def _reset_if_settings_changed(self) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
        if row and row[0] != self._settings_key():
            self.db.executescript("DELETE FROM files; DELETE FROM chunks;")
        self.db.execute(
            "INSERT OR REPLACE INTO meta VALUES ('settings', ?), ('folder', ?)",
            (self._settings_key(), str(self.folder)),
        )
        self.db.commit()

    def update(self, progress: Progress | None = None) -> UpdateResult:
        """追加・変更・削除されたファイルだけ索引を作り直す。

        ファイルは 1 つ終わるごとに確定 (commit) する。progress から例外を投げれば途中で止められ、
        それまでに終えたファイルは索引に残り、次回の update は残りだけを処理する。
        """
        self._bm25 = None  # 途中で止まっても、次の検索では索引を読み直す
        files = scan_files(self.folder)
        known = {p: (m, s) for p, m, s in self.db.execute("SELECT path, mtime, size FROM files")}
        current = {}
        for f in files:
            st = f.stat()
            current[f.relative_to(self.folder).as_posix()] = (f, st.st_mtime, st.st_size)

        removed = [p for p in known if p not in current]
        todo = [(rel, *v) for rel, v in current.items() if known.get(rel) != (v[1], v[2])]
        added = updated = 0
        errors: list[str] = []

        for p in removed:
            self.db.execute("DELETE FROM chunks WHERE path=?", (p,))
            self.db.execute("DELETE FROM files WHERE path=?", (p,))
        self.db.commit()

        for i, (rel, f, mtime, size) in enumerate(todo, start=1):
            if progress:
                progress(i, len(todo), rel)
            try:
                chunks = [
                    (sec.location, c)
                    for sec in load_file(f)
                    for c in split_text(sec.text, self.cfg.chunk_size, self.cfg.chunk_overlap)
                ]
                embs = self._embed_chunks(
                    rel, chunks,
                    (lambda k, m, i=i, rel=rel: progress(i, len(todo), f"{rel} ({k}/{m})")) if progress else None,
                )
            except Exception as e:  # 1 ファイルの失敗で全体を止めない
                errors.append(f"{rel}: {e}")
                continue
            self.db.execute("DELETE FROM chunks WHERE path=?", (rel,))
            self.db.executemany(
                "INSERT INTO chunks (path, location, text, emb) VALUES (?, ?, ?, ?)",
                [(rel, loc, text, emb.tobytes()) for (loc, text), emb in zip(chunks, embs)],
            )
            self.db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?)", (rel, mtime, size))
            self.db.commit()
            if rel in known:
                updated += 1
            else:
                added += 1
        self.db.commit()
        self._bm25 = None  # 次の検索で読み直す

        return UpdateResult(
            added=added,
            updated=updated,
            removed=len(removed),
            unchanged=len(current) - len(todo),
            errors=errors,
            chunks=self.chunk_count(),
        )

    def _embed_chunks(self, rel: str, chunks: list[tuple[str, str]],
                      on_batch: Callable[[int, int], None] | None = None) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        n_batches = (len(chunks) + EMBED_BATCH - 1) // EMBED_BATCH
        for i in range(0, len(chunks), EMBED_BATCH):
            if on_batch and n_batches > 1:
                on_batch(i // EMBED_BATCH + 1, n_batches)
            batch = [_with_header(rel, loc, text) for loc, text in chunks[i : i + EMBED_BATCH]]
            for v in self.ollama.embed(self.cfg.embed_model, batch):
                a = np.asarray(v, dtype=np.float32)
                out.append(a / (np.linalg.norm(a) or 1.0))
        return out

    def close(self) -> None:
        self.db.close()

    def chunk_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def file_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    # ---------- 検索 ----------

    def _load(self) -> None:
        rows = self.db.execute("SELECT path, location, text, emb FROM chunks ORDER BY id").fetchall()
        self._rows = [(p, loc, t) for p, loc, t, _ in rows]
        self._mat = np.vstack([np.frombuffer(e, dtype=np.float32) for *_, e in rows]) if rows else None
        self._bm25 = BM25([_with_header(p, loc, t) for p, loc, t in self._rows])

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        """埋め込み検索と BM25 のスコアを正規化して重み付きで足し合わせる。

        順位だけを使う RRF も試したが、文書が少ないと BM25 の偶然の一致 (「レシート」の「ート」が
        「スマートフォン」に当たる等) が上位に入り、埋め込み検索の 1 位を押し下げてしまった。
        スコアの差 (確信の強さ) を残すため、ここではスコアの加重和にしている。
        """
        top_k = top_k or self.cfg.top_k
        if self._bm25 is None:
            self._load()
        if not self._rows:
            return []

        q = np.asarray(self.ollama.embed(self.cfg.embed_model, [query])[0], dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        fused = (1 - self.cfg.bm25_weight) * _minmax(self._mat @ q)
        bm = np.asarray(self._bm25.scores(query))
        if bm.max() > 0:
            fused += self.cfg.bm25_weight * bm / bm.max()
        best = np.argsort(-fused)[:top_k]
        return [Hit(*self._rows[i], score=float(fused[i])) for i in best]


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.ones_like(a)


def _with_header(rel: str, location: str, text: str) -> str:
    """ファイル名と見出しも検索対象に含める (見出しの語で引けるように)。"""
    title = Path(rel).stem
    head = f"{title} / {location}" if location else title
    return f"{head}\n{text}"
