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
