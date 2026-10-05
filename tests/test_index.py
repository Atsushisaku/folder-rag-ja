from pathlib import Path

import pytest

import rag_chat.index as index_mod
from rag_chat.config import Config
from rag_chat.index import FolderIndex, delete_index, list_indexed_folders


class FakeOllama:
    def embed(self, model, texts):
        return [[1.0, float(len(t) % 7), 0.5] for t in texts]


class Stop(BaseException):
    """Streamlit の打ち切り (BaseException) の代わり。"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(index_mod, "INDEX_DIR", tmp_path / "idx")
    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        (data / f"doc{i}.md").write_text(f"# 見出し{i}\n本文{i}", encoding="utf-8")
    return data


def make_index(folder: Path) -> FolderIndex:
    idx = FolderIndex(folder, Config())
    idx.ollama = FakeOllama()
    return idx


def test_list_and_delete(env: Path):
    idx = make_index(env)
    idx.update()
    [f] = list_indexed_folders()
    assert (f.folder, f.files, f.exists) == (str(env), 3, True)
    idx.close()
    delete_index(env)
    assert list_indexed_folders() == []
    assert (env / "doc0.md").exists()  # 元のファイルは残る


def test_stop_midway_keeps_finished_files_and_resumes(env: Path):
    idx = make_index(env)

    def stop_at_third(i, n, name):
        if i == 3:
            raise Stop

    with pytest.raises(Stop):
        idx.update(stop_at_third)
    assert idx.file_count() == 2  # 終わった 2 ファイルは確定済み

    r = idx.update()
    assert (r.added, r.unchanged, idx.file_count()) == (1, 2, 3)


def test_missing_folder_is_flagged(env: Path, tmp_path: Path):
    other = tmp_path / "gone"
    other.mkdir()
    make_index(other).close()
    other.rmdir()
    flags = {f.folder: f.exists for f in list_indexed_folders()}
    assert flags[str(other)] is False


def test_delete_closes_shared_connection(env: Path):
    idx = index_mod.open_index(env, Config())
    assert index_mod.open_index(env, Config()) is idx  # 共有される
    idx.ollama = FakeOllama()
    idx.update()
    delete_index(env)  # 開いたままの接続を閉じてから消す (Windows でもロックで失敗しない)
    assert not index_mod.index_path_for(env).exists()


class TopicOllama:
    """「経費」を含む文と質問を同じ向きに埋め込む簡易版。"""

    def embed(self, model, texts):
        return [[1.0, 0.0] if "経費" in t else [0.0, 1.0] for t in texts]


def test_multi_index_searches_across_folders(env: Path, tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "keihi.md").write_text("# 経費\n経費は20日締め", encoding="utf-8")
    a, b = make_index(env), make_index(other)
    a.ollama = b.ollama = TopicOllama()
    a.update()
    b.update()

    multi = index_mod.MultiIndex([a, b], Config())
    multi.ollama = TopicOllama()
    hits = multi.search("経費の締め日", top_k=2)
    assert hits[0].folder == str(other) and hits[0].path == "keihi.md"
    assert {h.folder for h in multi.search("本文", top_k=4)} == {str(env), str(other)}

    # 片方の索引が変わったら読み直す
    (env / "doc0.md").write_text("# 経費の補足\n経費の申請", encoding="utf-8")
    a.update()
    assert any(h.folder == str(env) for h in multi.search("経費", top_k=2))
