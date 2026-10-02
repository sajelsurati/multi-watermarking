"""Main experiment: generate with and without the soft watermark, logging per-token statistics.

One invocation = one model x one language, looping over all conditions and both
stopping modes. Writes one Parquet file per (condition, stop mode) and skips
files that already exist, so a failed job can simply be resubmitted.

Conditions: "baseline" (no watermark, with shadow logging for every (gamma, delta)
in the grid) and one condition per (gamma, delta).
Stop modes: "forced" (EOS blocked, exactly --max-new-tokens tokens) and
"natural" (EOS allowed; statistics are kept up to and including the EOS step).

Seeds: every batch is seeded from (seed, language, batch index) and batches are
identical across conditions and stop modes, so prompt i shares its sampling
randomness everywhere (once the watermarked and baseline texts diverge, later
draws are no longer comparable token-by-token).

Usage: python generate.py --model llama-base --lang hi
"""

import argparse
import copy
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from data import LANGS, build_prompts, format_prompt
from models import MODELS
from watermark import SoftWatermarkProcessor, WatermarkConfig

GAMMAS = (0.25, 0.5)
DELTAS = (1.0, 2.0, 5.0)
GRID = [(g, d) for g in GAMMAS for d in DELTAS]
STOP_MODES = ("forced", "natural")


def tag(gamma: float, delta: float) -> str:
    return f"g{gamma}_d{delta:g}"


def conditions() -> dict[str, tuple[bool, list[tuple[float, float]]]]:
    """name -> (watermark applied?, [(gamma, delta) to log])."""
    conds = {"baseline": (False, GRID)}
    for g, d in GRID:
        conds[tag(g, d)] = (True, [(g, d)])
    return conds


def sampling_config(model, max_new_tokens: int, forced: bool):
    """Pure multinomial sampling at T=1 regardless of the model's shipped defaults
    (e.g. Qwen-Instruct ships repetition_penalty=1.05, Llama-Instruct T=0.6/top_p=0.9).
    The watermark processor must see the model's untransformed logits."""
    gc = copy.deepcopy(model.generation_config)
    gc.update(
        do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=None, typical_p=1.0,
        repetition_penalty=1.0, no_repeat_ngram_size=0, num_beams=1,
        max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens if forced else 0,
    )
    return gc


def eos_ids(model) -> set[int]:
    e = model.generation_config.eos_token_id
    return set(e if isinstance(e, list) else [e])


def run(model, tok, prompts, instruct, cond, stop_mode, args) -> pa.Table:
    applied, logged = conditions()[cond]
    vocab = model.config.vocab_size
    procs = [SoftWatermarkProcessor(WatermarkConfig(g, d, vocab), apply=applied) for g, d in logged]
    gc = sampling_config(model, args.max_new_tokens, stop_mode == "forced")
    gc.pad_token_id = tok.pad_token_id
    stops = eos_ids(model)
    lang_idx = list(LANGS).index(prompts[0]["lang"])

    rows = []
    for b, start in enumerate(range(0, len(prompts), args.batch_size)):
        batch = prompts[start:start + args.batch_size]
        texts = [format_prompt(r, tok, instruct) for r in batch]
        # Chat templates already contain BOS; raw passages need the tokenizer to add it.
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=not instruct).to(model.device)
        for p in procs:
            p.reset()
        seed = args.seed * 1_000_000 + lang_idx * 1_000 + b
        torch.manual_seed(seed)
        out = model.generate(**enc, generation_config=gc, logits_processor=LogitsProcessorList(procs))
        gen = out[:, enc["input_ids"].shape[1]:]
        stats = {tag(*lg): p.finalize(out) for lg, p in zip(logged, procs)}

        for i, r in enumerate(batch):
            ids = gen[i].tolist()
            stop_at = next((t for t, x in enumerate(ids) if x in stops), None)
            n = len(ids) if (stop_mode == "forced" or stop_at is None) else stop_at + 1
            row = {
                "model": args.model, "lang": r["lang"], "prompt_idx": r["prompt_idx"],
                "condition": cond, "stop_mode": stop_mode, "seed": seed,
                "prompt_len": int(enc["attention_mask"][i].sum()),
                "prev_token": int(enc["input_ids"][i, -1]),
                "n_tokens": n, "ended_eos": stop_at is not None and stop_mode == "natural",
                "token_ids": ids[:n],
                "text": tok.decode(ids[:n], skip_special_tokens=True),
            }
            for t, st in stats.items():
                for k, v in st.items():
                    row[f"{t}.{k}"] = v[i, :n].float().tolist() if k != "sampled_green" else v[i, :n].tolist()
            rows.append(row)
        print(f"    batch {b}: {len(batch)} prompts", flush=True)
    return pa.Table.from_pylist(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--lang", required=True, choices=LANGS)
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of conditions (default: all)")
    ap.add_argument("--stop-modes", nargs="*", default=list(STOP_MODES), choices=STOP_MODES)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/generations"))
    args = ap.parse_args()

    hf_id, instruct = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(hf_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    prompts = [r for r in build_prompts(args.n_prompts) if r["lang"] == args.lang]
    out_dir = args.out / args.model / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions or conditions():
        for stop_mode in args.stop_modes:
            path = out_dir / f"{cond}__{stop_mode}.parquet"
            if path.exists():
                print(f"skip {path} (exists)")
                continue
            t0 = time.time()
            print(f"{args.model} {args.lang} {cond} {stop_mode}", flush=True)
            table = run(model, tok, prompts, instruct, cond, stop_mode, args)
            tmp = path.with_suffix(".tmp")
            pq.write_table(table, tmp, compression="zstd")
            tmp.rename(path)
            print(f"  wrote {path} ({table.num_rows} rows, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
