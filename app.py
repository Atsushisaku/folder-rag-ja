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
from rag_chat.index import FolderIndex, MultiIndex, delete_index, list_indexed_folders, open_index
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
# 参照フォルダ = .rag_index/ に索引があるフォルダ。チェックを外したものは state.json の "excluded" に残す
# ss["checked"] : このセッションで差分更新を済ませた (または中止した) フォルダ
# ss["notice"]  : フォルダごとの直近の更新結果 (メッセージ, 一覧の下にも出すか = 中止・エラー)

ss.setdefault("checked", set())
ss.setdefault("notice", {})
ss.setdefault("messages", [])

if not load_state().get("initialized"):
    # 初回起動だけ config の data_dir (既定はサンプル) を参照フォルダに入れておく
    d = resolve_dir(cfg.data_dir)
    if d.is_dir() and not list_indexed_folders():
        open_index(d, cfg)
    save_state(initialized=True)


def get_index(folder: str) -> FolderIndex:
    return open_index(Path(folder), cfg)


def excluded() -> set[str]:
    return set(load_state().get("excluded", []))


def set_enabled(folder: str) -> None:
    ex = excluded()
    if ss[f"use::{folder}"]:
        ex.discard(folder)
    else:
        ex.add(folder)
    save_state(excluded=sorted(ex))


def add_folder(folder: str) -> None:
    get_index(folder)  # 索引ファイルを作ると一覧に載る。中身は次の差分更新で読み込む
    ss["checked"].discard(folder)
    ss[f"use::{folder}"] = True
    save_state(excluded=sorted(excluded() - {folder}))


def remove_folder(folder: str) -> None:
    try:
        delete_index(Path(folder))
    except PermissionError:
        ss["notice"][folder] = ("削除できませんでした。索引の更新中なら、終わるか中止してから再度お試しください。", True)
        return
    ss["checked"].discard(folder)
    ss["notice"].pop(folder, None)
    ss.pop(f"use::{folder}", None)
    save_state(excluded=sorted(excluded() - {folder}))


def request_update() -> None:
    ss["force_update"] = True


def request_delete(folder: str) -> None:
    ss["delete_target"] = folder


@st.dialog("索引を削除")
def confirm_delete(folder: str, label: str) -> None:
    st.markdown(f"**{label}** の索引を削除します。元のファイルは削除されません。")
    st.caption(folder)
    c1, c2 = st.columns(2)
    if c1.button("削除する", type="primary", use_container_width=True):
        remove_folder(folder)
        st.rerun()
    if c2.button("キャンセル", use_container_width=True):
        st.rerun()


def request_cancel() -> None:
    # ボタンを押すと Streamlit は実行中のスクリプトを次の st 呼び出しで打ち切って再実行する。
    # この関数はその再実行の最初に呼ばれるので、中止されたことだけ記録する。
    ss["cancelled"] = ss.get("updating")


def run_updates(targets: list[str], labels: dict[str, str]) -> list[str]:
    """targets の差分更新を順に行い、変更があったフォルダの結果を返す。

    進捗バーと中止ボタンは終わったらまとめて消す。
    """
    changes = []
    slot = st.empty()
    box = slot.container()
    bar = box.progress(0.0, text="索引を確認中…")
    box.button("中止", key="cancel_update", on_click=request_cancel, use_container_width=True)
    for k, folder in enumerate(targets):
        ss["updating"] = targets[k:]  # 中止されたら、残りのフォルダも自動では再開しない
        name = labels.get(folder, folder)
        t = time.perf_counter()
        r = get_index(folder).update(
            lambda i, n, f: bar.progress(min(i / n, 1.0), text=f"{name}: [{i}/{n}] {f}"))
        ss["checked"].add(folder)
        changed = r.added + r.updated + r.removed
        msg = (f"追加 {r.added} / 更新 {r.updated} / 削除 {r.removed} ({time.perf_counter() - t:.1f}s)"
               if changed else "変更はありません")
        if r.errors:
            msg = f"{name}: {msg}  \n" + "  \n".join(f"⚠ {e}" for e in r.errors)
        ss["notice"][folder] = (msg, bool(r.errors))
        if changed:
            changes.append(f"{name}: {msg}")
    ss["updating"] = None
    slot.empty()
    return changes


# 中止ボタンで打ち切られた直後の再実行
if cancelled := ss.pop("cancelled", None):
    ss["updating"] = None
    ss["checked"].update(cancelled)
    ss["notice"][cancelled[0]] = (
        "中止しました。ここまでに読み込んだファイルは検索できます。続きは「更新」で再開します。", True)
elif ss.get("updating"):
    # 中止ボタン以外 (チェックの付け外し等) で打ち切られた場合。次の実行で続きから更新する
    ss["checked"].difference_update(ss["updating"])
    ss["updating"] = None


def folder_labels(paths: list[str]) -> dict[str, str]:
    """一覧に出す名前。同じ名前のフォルダが複数あるときは親フォルダ名も付けて区別する。"""
    names = [Path(p).name or p for p in paths]
    return {
        p: (f"{Path(p).parent.name}/{n}" if names.count(n) > 1 else n)
        for p, n in zip(paths, names)
    }


# ---------- サイドバー: 参照フォルダ ----------

with st.sidebar:
    st.header("参照フォルダ")
    st.caption("チェックしたフォルダをまとめて検索します")

    folders = {f.folder: f for f in list_indexed_folders()}
    labels = folder_labels(list(folders))
    ex = excluded()
    for p, info in folders.items():
        ss.setdefault(f"use::{p}", p not in ex)
        status = f"{info.files} ファイル" + ("" if info.exists else " · ⚠ フォルダが見つかりません")
        c_check, c_del = st.columns([0.85, 0.15], vertical_alignment="center")
        c_check.checkbox(f"{labels[p]} :gray[{status}]", key=f"use::{p}", on_change=set_enabled, args=(p,),
                         help=p)
        c_del.button("✕", key=f"del::{p}", on_click=request_delete, args=(p,), type="tertiary",
                     help="索引を削除（元のファイルは消えません）")
    if not folders:
        st.info("「追加」で読み込むフォルダを追加してください")
    enabled = [p for p in folders if ss.get(f"use::{p}")]
    if (target := ss.pop("delete_target", None)) in folders:
        confirm_delete(target, labels[target])

    problems = check_ollama()

    c1, c2 = st.columns(2)
    if c1.button("追加", use_container_width=True, help="読み込むフォルダを選んで追加します"):
        picked = pick_folder()
        if picked:
            add_folder(str(Path(picked).resolve()))
            st.rerun()
    c2.button("更新", on_click=request_update, use_container_width=True,
              disabled=bool(problems) or not any(folders[p].exists for p in enabled),
              help="チェックしたフォルダの追加・変更・削除されたファイルだけ読み直します")

    for msg in problems:
        st.warning(msg)
    missing = [labels[p] for p in enabled if not folders[p].exists]
    if missing:
        st.warning(f"フォルダが見つかりません: {'、'.join(missing)}。索引は残っているので検索には使います"
                   "（移動・削除した場合は ✕ で索引を削除してください）。")

    # 「更新」が押されたとき、またはこのセッションでまだ確認していないフォルダがあるとき、チェック済みを差分更新する
    if not problems:
        forced = ss.pop("force_update", False)
        targets = [p for p in enabled if folders[p].exists and (forced or p not in ss["checked"])]
        if targets:
            changes = run_updates(targets, labels)
            if forced or changes:
                ss["toast"] = "  \n".join(changes) if changes else "変更はありません"
            st.rerun()  # 一覧のファイル数を最新にする
    if msg := ss.pop("toast", None):
        st.toast(msg)
    for p in folders:
        if (notice := ss["notice"].get(p)) and notice[1]:
            st.caption(notice[0])  # 中止・エラーなど、気づいてほしいものだけ一覧の下にも出す

    st.divider()
    st.caption(f"生成: `{cfg.chat_model}` / 埋め込み: `{cfg.embed_model}`")

search_index = None
if enabled and not problems:
    # 対象が変わらない限り使い回す (読み込んだチャンクと BM25 を毎回作り直さないため)。
    # 索引を削除して作り直すと FolderIndex も別物になるので、オブジェクトの同一性で判定する
    indexes = [get_index(p) for p in enabled]
    if ss.get("search_key") != [id(i) for i in indexes]:
        ss["search_index"] = MultiIndex(indexes, cfg)
        ss["search_key"] = [id(i) for i in indexes]
    search_index = ss["search_index"]


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
if folders:
    st.caption("検索対象: " + ("、".join(labels[p] for p in enabled) if enabled
                              else "なし（参照フォルダにチェックを入れてください）"))

STOPPED_NOTE = "（生成を停止しました）"


def waiting(ph, label: str) -> None:
    ph.markdown(f'<span class="rag-waiting">{label}</span>', unsafe_allow_html=True)


def show_references(text: str, hits: list[dict]) -> None:
    """回答が [n] で引用した資料だけを 1 行で示す (引用が無ければ何も出さない)。"""
    refs = cited_sources(text, hits)
    if refs:
        st.caption("参照: " + " / ".join(
            f"[{n}] {Path(h['folder']).name}/{h['path']} {h['location']}".strip() for n, h in refs))


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

question = st.chat_input("マニュアルについて質問してください", disabled=search_index is None)
if question and search_index is not None:
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
            hits = search_index.search(question)
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
