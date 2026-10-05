"""日本語向けの簡易 BM25。

形態素解析器を入れない代わりに、日本語は文字 2-gram、英数字は単語単位でトークン化する。
ひらがなは助詞・活用語尾ばかりで、少量の文書では無関係な一致 (「とき」「どう」) のほうが
多くなるため捨てる。ひらがなで区切った残り (漢字・カタカナの連なり) から 2-gram を作り、
1 文字だけの連なり (「立て替え」の「立」「替」など) はその 1 文字をトークンにする。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9_]+|[^\sa-z0-9_\W]+|[^\s\w]", re.UNICODE)
_WORD = re.compile(r"[a-z0-9_]+")
_HIRAGANA = re.compile(r"[\u3040-\u309f]+")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for m in _TOKEN.finditer(text):
        s = m.group(0)
        if _WORD.fullmatch(s):
            tokens.append(s)
        else:
            for run in _HIRAGANA.split(s):
                if len(run) == 1:
                    if run.isalnum():
                        tokens.append(run)
                else:
                    tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = set(tokenize(query))
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out
