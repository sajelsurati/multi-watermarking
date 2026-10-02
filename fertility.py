"""Tokenizer fertility: how many tokens each model's tokenizer uses per word, per language.

Text: the same FLORES passages used as prompts in the main experiment.

Words: whitespace-separated, with leading/trailing punctuation removed; Chinese is
segmented with jieba, Japanese with MeCab (fugashi + unidic-lite). Punctuation-only
segments are not words.

Tokens per word are counted in running text: each passage is tokenized whole and
every token is assigned to the word(s) whose characters it overlaps (via the fast
tokenizer's offset mapping). A token spanning two words counts for both; a token
covering only whitespace counts for neither.

Reported per (model, language):
  fertility_mean  mean tokens per word (Rust et al. 2021, "How Good is Your Tokenizer?")
  frac_split      share of words split into >= 2 tokens ("proportion of continued words")
  plus median, 90th percentile and the word count.

CPU only; needs network access to download tokenizers (Llama is gated: HF_TOKEN).
Usage: python fertility.py   -> results/fertility.csv and a printed table
"""

import argparse
import bisect
import csv
import re
import statistics
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer

from data import LANGS, build_prompts
from models import MODELS


def _is_punct(c: str) -> bool:
    return c.isspace() or unicodedata.category(c).startswith("P")


def _trim(text: str, s: int, e: int):
    while s < e and _is_punct(text[s]):
        s += 1
    while e > s and _is_punct(text[e - 1]):
        e -= 1
    return (s, e) if s < e else None


def _spans_from_segments(text: str, segments) -> list[tuple[int, int]]:
    """Locate segmenter outputs in the text (segmenters return surfaces in order)."""
    spans, cursor = [], 0
    for seg in segments:
        if not seg.strip():
            continue
        s = text.find(seg, cursor)
        if s < 0:
            continue
        cursor = s + len(seg)
        spans.append((s, cursor))
    return spans


class WordSegmenter:
    def __init__(self):
        self._jieba = None
        self._mecab = None

    def spans(self, text: str, lang: str) -> list[tuple[int, int]]:
        if lang == "zh":
            if self._jieba is None:
                import jieba
                jieba.setLogLevel(60)
                self._jieba = jieba
            raw = [(s, e) for _, s, e in self._jieba.tokenize(text)]
        elif lang == "ja":
            if self._mecab is None:
                import fugashi
                self._mecab = fugashi.Tagger()
            raw = _spans_from_segments(text, [w.surface for w in self._mecab(text)])
        else:
            raw = [m.span() for m in re.finditer(r"\S+", text)]
        return [t for s, e in raw if (t := _trim(text, s, e))]


def tokens_per_word(text: str, word_spans, tok) -> list[int]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [s for s, _ in word_spans]
    counts = [0] * len(word_spans)
    for ts, te in enc["offset_mapping"]:
        if te <= ts:
            continue
        k = bisect.bisect_right(starts, te - 1) - 1  # last word starting before the token ends
        while k >= 0 and word_spans[k][1] > ts:
            counts[k] += 1
            k -= 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("results/fertility.csv"))
    args = ap.parse_args()

    prompts = build_prompts(args.n_prompts)
    seg = WordSegmenter()
    words = {lang: [] for lang in LANGS}  # lang -> [(passage, spans)]
    for r in prompts:
        words[r["lang"]].append((r["passage"], seg.spans(r["passage"], r["lang"])))

    rows = []
    for key, (hf_id, _) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(hf_id)
        assert tok.is_fast, f"{hf_id}: offset mapping needs a fast tokenizer"
        for lang in LANGS:
            counts = [c for text, spans in words[lang] for c in tokens_per_word(text, spans, tok)]
            rows.append({
                "model": key, "hf_id": hf_id, "lang": lang, "n_words": len(counts),
                "fertility_mean": round(statistics.mean(counts), 3),
                "fertility_median": statistics.median(counts),
                "fertility_p90": statistics.quantiles(counts, n=10)[-1],
                "frac_split": round(sum(c >= 2 for c in counts) / len(counts), 3),
            })
        print(f"done {key}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    models = list(MODELS)
    print("\nMean tokens per word (fraction of words split into >=2 tokens)")
    print(f"{'lang':6}" + "".join(f"{m:>22}" for m in models))
    for lang in LANGS:
        cells = {r["model"]: r for r in rows if r["lang"] == lang}
        print(f"{lang:6}" + "".join(
            f"{cells[m]['fertility_mean']:>14.2f} ({cells[m]['frac_split']:.2f})" for m in models))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
