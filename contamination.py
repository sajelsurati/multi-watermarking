"""Contamination test (a): greedy completion of FLORES articles vs. unseen control text.

For each document (sentence list), the first ceil(n/2) sentences are the prefix
and the rest is the target, truncated to --max-target-tokens. The model greedily
generates as many tokens as the target has, and we score:
  chrf      sacrebleu chrF of generation vs. target (character-level, so it
            works for zh/ja without word segmentation)
  lcp_frac  fraction of target tokens reproduced verbatim before the first mismatch
  exact     every target token reproduced

FLORES documents: the same 200 articles used in the main experiment, all their
devtest sentences. Control documents: data/control/<lang>.jsonl, one JSON object
per line: {"doc_id": str, "sentences": [str, ...], "date": "YYYY-MM-DD", "source": str}.
Control text must postdate every model's release (see models.py). Each control
document is cut to a sentence count drawn from the FLORES article lengths so the
two sets have the same length distribution.

All models (instruct included) see raw text with no chat template: this tests
memorization of the text itself.

Usage: python contamination.py --model llama-base
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import LANGS, articles, join_sentences, select_urls
from models import MODELS

chrf_metric = CHRF()


def flores_docs(n_prompts: int, seed: int) -> list[dict]:
    en = articles("en")
    urls = select_urls(en, n_prompts, seed)
    docs = []
    for lang in LANGS:
        arts = en if lang == "en" else articles(lang)
        for url in urls:
            docs.append({"source": "flores", "lang": lang, "doc_id": url,
                         "sentences": [t for _, t in arts[url]]})
    return docs


def control_docs(control_dir: Path, lengths: list[int]) -> list[dict]:
    docs = []
    for lang in LANGS:
        path = control_dir / f"{lang}.jsonl"
        if not path.exists():
            print(f"WARNING: no control file for {lang} ({path})")
            continue
        for j, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            d = json.loads(line)
            n = lengths[j % len(lengths)]
            docs.append({"source": "control", "lang": lang, "doc_id": d["doc_id"],
                         "sentences": d["sentences"][:n], "date": d.get("date")})
    return docs


def make_item(doc: dict, tok, max_target_tokens: int):
    sents = doc["sentences"]
    if len(sents) < 2:
        return None
    k = math.ceil(len(sents) / 2)
    prefix = join_sentences(sents[:k], doc["lang"])
    full = join_sentences(sents, doc["lang"])
    prefix_ids = tok(prefix)["input_ids"]
    full_ids = tok(full)["input_ids"]
    # The target is whatever follows the prefix in the joint tokenization, so the
    # tokens the model must reproduce are the ones it would see in running text.
    if full_ids[: len(prefix_ids)] != prefix_ids:
        prefix_ids = prefix_ids[:-1]  # boundary merge: last prefix token absorbed into the target
        if full_ids[: len(prefix_ids)] != prefix_ids:
            return None
    target_ids = full_ids[len(prefix_ids):][:max_target_tokens]
    if not target_ids:
        return None
    return {**doc, "n_sentences": len(sents), "prefix_ids": prefix_ids, "target_ids": target_ids}


@torch.no_grad()
def complete(model, tok, items: list[dict], batch_size: int):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prefix_ids"]))
    for start in range(0, len(order), batch_size):
        batch = [items[i] for i in order[start:start + batch_size]]
        enc = tok.pad({"input_ids": [b["prefix_ids"] for b in batch]}, return_tensors="pt").to(model.device)
        out = model.generate(
            **enc,
            do_sample=False, temperature=None, top_p=None, top_k=None,
            max_new_tokens=max(len(b["target_ids"]) for b in batch),
            pad_token_id=tok.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:].tolist()
        for b, g in zip(batch, gen):
            g = g[: len(b["target_ids"])]
            lcp = next((i for i, (x, y) in enumerate(zip(g, b["target_ids"])) if x != y), len(g))
            hyp = tok.decode(g, skip_special_tokens=True).strip()
            ref = tok.decode(b["target_ids"], skip_special_tokens=True).strip()
            b.update(generation=hyp, target=ref,
                     chrf=chrf_metric.sentence_score(hyp, [ref]).score,
                     lcp_frac=lcp / len(b["target_ids"]),
                     exact=lcp == len(b["target_ids"]))
        print(f"  {min(start + batch_size, len(order))}/{len(order)}", flush=True)


def summarize(items: list[dict]) -> list[dict]:
    rows = []
    for lang in LANGS:
        for source in ("flores", "control"):
            xs = [i for i in items if i["lang"] == lang and i["source"] == source]
            if not xs:
                continue
            chrf = [x["chrf"] for x in xs]
            rows.append({
                "lang": lang, "source": source, "n": len(xs),
                "chrf_mean": round(statistics.mean(chrf), 2),
                "chrf_median": round(statistics.median(chrf), 2),
                "frac_chrf_ge_80": round(sum(c >= 80 for c in chrf) / len(xs), 3),
                "lcp_frac_mean": round(statistics.mean(x["lcp_frac"] for x in xs), 3),
                "frac_exact": round(sum(x["exact"] for x in xs) / len(xs), 3),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-dir", type=Path, default=Path("data/control"))
    ap.add_argument("--max-target-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("results/contamination"))
    args = ap.parse_args()

    hf_id, _ = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(hf_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    flores = flores_docs(args.n_prompts, args.seed)
    lengths = [len(d["sentences"]) for d in flores if d["lang"] == "en"]
    docs = flores + control_docs(args.control_dir, lengths)
    items = [it for d in docs if (it := make_item(d, tok, args.max_target_tokens))]
    print(f"{args.model}: {len(items)} items ({len(docs) - len(items)} skipped)", flush=True)

    complete(model, tok, items, args.batch_size)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"{args.model}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            rec = {k: v for k, v in it.items() if k not in ("prefix_ids", "target_ids")}
            f.write(json.dumps({**rec, "model": args.model, "hf_id": hf_id}, ensure_ascii=False) + "\n")
    rows = summarize(items)
    with open(args.out / f"{args.model}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
