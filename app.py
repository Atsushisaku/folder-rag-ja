"""Streamlit UI。起動: uv run streamlit run app.py"""

from __future__ import annotations

import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import streamlit as st

from rag_chat.answer import answer_stream, cited_sources
from rag_chat.config import load_config, load_state, resolve_dir, save_state
from rag_chat.index import FolderIndex, delete_index, list_indexed_folders, open_index
from rag_chat.ollama_client import Ollama, OllamaError, has_model

st.set_page_config(page_title="ローカル RAG チャット", page_icon="📚", layout="wide")
cfg = load_config()
ss = st.session_state

PICK_FOLDER = (
    "import tkinter as tk; from tkinter import filedialog; r = tk.Tk(); r.withdraw();"
    "r.attributes('-topmost', True); print(filedialog.askdirectory(title='読み込むフォルダを選択'))"
)


def pick_folder() -> str:
    """OS のフォルダ選択ダイアログを別プロセスで開く (Streamlit のスレッドで tkinter を使わないため)。"""
    try:
        out = subprocess.run([sys.executable, "-c", PICK_FOLDER], capture_output=True, text=True, timeout=600)
        return out.stdout.strip()
    except Exception:
        return ""


def check_ollama() -> list[str]:
    try:
        installed = Ollama(cfg.ollama_host).list_models()
    except OllamaError as e:
        return [f"{e}  \nOllama を起動してください。"]
    return [f"モデル `{m}` がありません。`ollama pull {m}` を実行してください。"
            for m in (cfg.embed_model, cfg.chat_model) if not has_model(installed, m)]


# ---------- 状態 ----------
# ss["folder"]     : 選択中のフォルダ (絶対パスの文字列)。None なら未選択
# ss["checked"]    : このセッションで差分更新を済ませた (または中止した) フォルダ
# ss["notice"]     : フォルダごとの直近の更新結果 (メッセージ, 一覧の下にも出すか)
# ss["added"]      : このセッションで追加したフォルダ。索引がまだ無くても選べるのはこれだけ
#                    (別のタブで索引を削除したフォルダを、こちらのタブが勝手に作り直さないため)

ss.setdefault("checked", set())
ss.setdefault("added", set())
ss.setdefault("notice", {})
ss.setdefault("messages", [])
if "folder" not in ss:
    state = load_state()
    if "folder" in state:
        ss["folder"] = state["folder"]
    else:  # 初回起動だけ config の data_dir (既定はサンプル) を開く
        d = resolve_dir(cfg.data_dir)
        ss["folder"] = str(d) if d.is_dir() else None
        if ss["folder"]:
            ss["added"].add(ss["folder"])


def select_folder(folder: str | None) -> None:
    if folder != ss.get("folder"):
        ss["messages"] = []
    ss["folder"] = folder
    save_state(folder=folder)


def add_folder(folder: str) -> None:
    ss["added"].add(folder)
    select_folder(folder)


def get_index(folder: str) -> FolderIndex:
    return open_index(Path(folder), cfg)


def remove_folder(folder: str) -> None:
    try:
        delete_index(Path(folder))
    except PermissionError:
        ss["notice"][folder] = ("削除できませんでした。索引の更新中なら、終わるか中止してから再度お試しください。", True)
        return
    ss["checked"].discard(folder)
    ss["added"].discard(folder)
    ss["notice"].pop(folder, None)
    rest = [f.folder for f in list_indexed_folders()]
    select_folder(rest[0] if rest else None)


def request_cancel() -> None:
    # ボタンを押すと Streamlit は実行中のスクリプトを次の st 呼び出しで打ち切って再実行する。
    # この関数はその再実行の最初に呼ばれるので、中止されたことだけ記録する。
    ss["cancelled"] = ss.get("updating")


def run_update(folder: str) -> None:
    idx = get_index(folder)
    slot = st.empty()  # 終わったら進捗バーと中止ボタンをまとめて消すための置き場
    box = slot.container()
    bar = box.progress(0.0, text="索引を確認中…")
    box.button("中止", key="cancel_update", on_click=request_cancel, use_container_width=True)
    ss["updating"] = folder
    t = time.perf_counter()
    r = idx.update(lambda i, n, name: bar.progress(min(i / n, 1.0), text=f"[{i}/{n}] {name}"))
    ss["updating"] = None
    slot.empty()
    ss["checked"].add(folder)
    changed = r.added + r.updated + r.removed
    msg = (f"追加 {r.added} / 更新 {r.updated} / 削除 {r.removed} ({time.perf_counter() - t:.1f}s)"
           if changed else "変更はありません")
    if r.errors:
        msg += "  \n" + "  \n".join(f"⚠ {e}" for e in r.errors)
    ss["notice"][folder] = (msg, bool(r.errors))


# 中止ボタンで打ち切られた直後の再実行
if cancelled := ss.pop("cancelled", None):
    ss["updating"] = None
    ss["checked"].add(cancelled)  # 自動で再開しない
    ss["notice"][cancelled] = ("中止しました。ここまでに読み込んだファイルは検索できます。続きは「索引編集」の「索引を更新」で再開します。", True)
elif ss.get("updating"):
    # 中止ボタン以外 (フォルダの切り替え等) で打ち切られた場合。次に開いたときに続きから更新する
    ss["checked"].discard(ss["updating"])
    ss["updating"] = None


# ---------- サイドバー: 参照フォルダ ----------

def request_update(folder: str) -> None:
    ss["force_update"] = folder


def request_delete(folder: str) -> None:
    remove_folder(folder)
    ss["confirm_delete"] = False  # 次に開いたときにチェックが残らないように


def folder_labels(paths: list[str]) -> dict[str, str]:
    """一覧に出す名前。同じ名前のフォルダが複数あるときは親フォルダ名も付けて区別する。"""
    names = [Path(p).name or p for p in paths]
    return {
        p: (f"{Path(p).parent.name}/{n}" if names.count(n) > 1 else n)
        for p, n in zip(paths, names)
    }


with st.sidebar:
    st.header("参照フォルダ")

    folders = {f.folder: f for f in list_indexed_folders()}
    paths = list(folders)
    current = ss["folder"]
    if current and current not in folders:
        if current in ss["added"] and Path(current).is_dir():
            paths.append(current)  # 追加したばかりで索引がまだ無いもの
        else:  # 別のタブで索引が削除された
            select_folder(paths[0] if paths else None)
            current = ss["folder"]

    if paths:
        labels = folder_labels(paths)

        def caption(p: str) -> str:
            info = folders.get(p)
            if info is None:
                return "未読み込み"
            return f"{info.files} ファイル" + ("" if info.exists else " · ⚠ フォルダが見つかりません")

        chosen = st.radio(
            "参照フォルダ", paths, index=paths.index(current) if current in paths else 0,
            format_func=labels.get, captions=[caption(p) for p in paths], label_visibility="collapsed",
        )
        if chosen != current:
            select_folder(chosen)
            st.rerun()
    else:
        st.info("「参照…」で読み込むフォルダを追加してください")

    folder = ss["folder"]
    problems = check_ollama()
    idx = get_index(folder) if folder and Path(folder).is_dir() and not problems else None

    c1, c2 = st.columns(2)
    if c1.button("参照…", use_container_width=True, help="フォルダを選んで追加します"):
        picked = pick_folder()
        if picked:
            add_folder(str(Path(picked).resolve()))
            st.rerun()
    with c2.popover("索引編集", use_container_width=True, disabled=folder is None):
        if folder:
            st.markdown(f"**{Path(folder).name or folder}**")
            st.caption(folder)
            if idx is not None:
                st.caption(f"{idx.file_count()} ファイル / {idx.chunk_count()} チャンク")
            if notice := ss["notice"].get(folder):
                st.caption(f"前回: {notice[0]}")
            st.button("索引を更新", on_click=request_update, args=(folder,), disabled=idx is None,
                      use_container_width=True, help="追加・変更・削除されたファイルだけ読み直します")
            st.divider()
            confirm = st.checkbox("この索引を削除する（元のファイルは消えません）", key="confirm_delete")
            st.button("削除", type="primary", on_click=request_delete, args=(folder,), disabled=not confirm,
                      use_container_width=True)

    for msg in problems:
        st.warning(msg)
    if folder and not Path(folder).is_dir():
        st.warning("フォルダが見つかりません（移動・削除された可能性があります）。不要なら「索引編集」から削除してください。")

    # 「索引を更新」が押されたとき、またはこのセッションで初めて開いたときに差分更新する (1 回の実行で 1 回だけ)
    if idx is not None and (ss.pop("force_update", None) == folder or folder not in ss["checked"]):
        run_update(folder)
    if folder and (notice := ss["notice"].get(folder)) and notice[1]:
        st.caption(notice[0])  # 中止・エラーなど、気づいてほしいものだけ一覧の下にも出す

    st.divider()
    st.caption(f"生成: `{cfg.chat_model}` / 埋め込み: `{cfg.embed_model}`")


# ---------- メイン: チャット ----------

st.markdown(
    """
    <style>
    /* ボットの回答を枠で囲む (ユーザーの入力と見分けやすくする) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 0.75rem;
        padding: 1rem;
    }
    /* 待ち時間の「...」アニメーション */
    .rag-waiting { color: rgba(128, 128, 128, 0.9); font-size: 0.9rem; }
    .rag-waiting::after {
        content: "";
        display: inline-block;
        width: 1.5em;
        text-align: left;
        animation: rag-dots 1.2s steps(1, end) infinite;
    }
    @keyframes rag-dots {
        0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75% { content: "..."; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 ローカル RAG チャット")
if folder:
    st.caption(f"対象: {folder}")

STOPPED_NOTE = "（生成を停止しました）"


def waiting(ph, label: str) -> None:
    ph.markdown(f'<span class="rag-waiting">{label}</span>', unsafe_allow_html=True)


def show_references(text: str, hits: list[dict]) -> None:
    """回答が [n] で引用した資料だけを 1 行で示す (引用が無ければ何も出さない)。"""
    refs = cited_sources(text, hits)
    if refs:
        st.caption("参照: " + " / ".join(f"[{n}] {h['path']} {h['location']}".strip() for n, h in refs))


# 前回の実行が生成の途中で打ち切られた (停止ボタン、生成中に次の質問を送った等)。
# Streamlit はボタン操作などで実行中のスクリプトを打ち切って再実行するので、
# 途中までの回答を ss["streaming"] に残しておき、ここで履歴に移す。
if (partial := ss.pop("streaming", None)) is not None:
    body = partial["text"] + "\n\n" if partial["text"] else ""
    ss["messages"].append({"role": "assistant", "content": body + f"*{STOPPED_NOTE}*", "hits": partial["hits"]})

for m in ss["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_references(m["content"], m["hits"])

question = st.chat_input("マニュアルについて質問してください", disabled=idx is None)
if question and idx is not None:
    ss["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer_ph, status_ph, stop_ph = st.empty(), st.empty(), st.empty()
        # 押されると再実行が起き、上の「打ち切られた」処理で履歴に残る (ボタン自体は何もしない)
        stop_ph.button("⏹ 生成を停止", key="stop_generation")
        streaming = ss["streaming"] = {"text": "", "hits": []}
        try:
            waiting(status_ph, "資料を検索中")
            hits = idx.search(question)
            streaming["hits"] = [h.__dict__ for h in hits]
            waiting(status_ph, "回答を作成中")
            # closing: 打ち切られたときも Ollama への接続を確実に閉じ、生成を止める
            with closing(answer_stream(cfg, question, hits)) as pieces:
                for piece in pieces:
                    streaming["text"] += piece
                    answer_ph.markdown(streaming["text"])
            text = streaming["text"]
        except OllamaError as e:
            answer_ph.error(str(e))
            text = f"エラー: {e}"
        status_ph.empty()
        stop_ph.empty()
        show_references(text, streaming["hits"])
    ss.pop("streaming", None)
    ss["messages"].append({"role": "assistant", "content": text, "hits": streaming["hits"]})
