"""Analysis of the main experiment, with languages ordered by tokenizer fertility.

Language order: mean tokens per word (results/fertility.csv), averaged over the
models, lowest first. The same order is used in every panel.

Metrics, per (model, language, stop mode, gamma, delta):
  kl_mean            mean per-token KL(q || p) between the watermarked (q) and
                     original (p) next-token distributions, along the watermarked
                     generations; averaged per generation, then over generations
  z_mean             mean detection z-score over the full generation
  tpr_at_1pct        share of watermarked generations with z > 2.326, the one-sided
                     1% threshold under the null z ~ N(0, 1)
  fpr_at_1pct        share of unwatermarked (baseline) generations above that same
                     threshold, scored with this gamma's green lists; should be ~0.01
                     and checks the N(0, 1) assumption per language
  tokens_to_detect   median number of tokens until the running z first exceeds 4
                     (the paper's threshold); inf if most generations never reach it
                     (plotted as a gap in the line)
  frac_detected      share of generations that reach z > 4 at all
In natural-stop runs the EOS token is not scored for detection.

Outputs (results/analysis/): summary.csv and, per stop mode,
  kl_<stop>.png, z_<stop>.png, tpr_<stop>.png, tokens_to_detect_<stop>.png

Usage: python analyze.py
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from generate import DELTAS, GAMMAS, STOP_MODES, tag
from models import MODELS

Z_1PCT = 2.326
Z_DETECT = 4.0

# Ordinal blue ramp for delta (validated: monotone lightness, single hue,
# light end >= 2:1 on the surface); markers add a non-color channel.
DELTA_STYLE = {
    1.0: {"color": "#6da7ec", "marker": "o"},
    2.0: {"color": "#2a78d6", "marker": "s"},
    5.0: {"color": "#104281", "marker": "^"},
}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}


# ---------- per-generation scoring ----------

def running_z(green: np.ndarray, gamma: float) -> np.ndarray:
    T = np.arange(1, len(green) + 1)
    return (np.cumsum(green) - gamma * T) / np.sqrt(T * gamma * (1 - gamma))


def score_generation(green, ended_eos: bool, gamma: float) -> dict:
    g = np.asarray(green, dtype=float)
    if ended_eos:
        g = g[:-1]
    if len(g) == 0:
        return {"z": np.nan, "tokens_to_detect": np.inf}
    z = running_z(g, gamma)
    hit = np.flatnonzero(z > Z_DETECT)
    return {"z": z[-1], "tokens_to_detect": hit[0] + 1 if len(hit) else np.inf}


def load_scores(gen_dir: Path) -> pd.DataFrame:
    """One row per generation: watermarked runs scored with their own (gamma, delta),
    baseline runs scored with each gamma's green lists (the null distribution)."""
    rows = []
    for path in sorted(gen_dir.glob("*/*/*.parquet")):
        model, lang = path.parent.parent.name, path.parent.name
        cond, stop = path.stem.split("__")
        if cond == "baseline":
            # Shadow green hits depend only on gamma; read one delta per gamma.
            for gamma in GAMMAS:
                col = f"{tag(gamma, DELTAS[0])}.sampled_green"
                t = pq.read_table(path, columns=["prompt_idx", "ended_eos", col]).to_pydict()
                for i in range(len(t["prompt_idx"])):
                    s = score_generation(t[col][i], t["ended_eos"][i], gamma)
                    rows.append({"model": model, "lang": lang, "stop": stop, "watermarked": False,
                                 "gamma": gamma, "delta": np.nan, "kl": np.nan, **s})
        else:
            gamma, delta = (float(x[1:]) for x in cond.split("_"))
            t = pq.read_table(path, columns=["prompt_idx", "ended_eos", f"{cond}.sampled_green",
                                             f"{cond}.kl_q_p"]).to_pydict()
            for i in range(len(t["prompt_idx"])):
                s = score_generation(t[f"{cond}.sampled_green"][i], t["ended_eos"][i], gamma)
                rows.append({"model": model, "lang": lang, "stop": stop, "watermarked": True,
                             "gamma": gamma, "delta": delta,
                             "kl": float(np.mean(t[f"{cond}.kl_q_p"][i])), **s})
    return pd.DataFrame(rows)


# ---------- aggregation ----------

def _se(x: pd.Series) -> float:
    return x.std(ddof=1) / math.sqrt(x.count()) if x.count() > 1 else np.nan


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    null = scores[~scores.watermarked]
    fpr = (null.assign(above=null.z > Z_1PCT)
               .groupby(["model", "lang", "stop", "gamma"]).above.mean()
               .rename("fpr_at_1pct"))
    wm = scores[scores.watermarked]
    agg = wm.groupby(["model", "lang", "stop", "gamma", "delta"]).agg(
        n=("z", "count"),
        kl_mean=("kl", "mean"), kl_se=("kl", _se),
        z_mean=("z", "mean"), z_se=("z", _se),
        tpr_at_1pct=("z", lambda z: (z > Z_1PCT).mean()),
        tokens_to_detect=("tokens_to_detect", "median"),
        frac_detected=("tokens_to_detect", lambda t: np.isfinite(t).mean()),
    ).reset_index()
    return agg.merge(fpr.reset_index(), on=["model", "lang", "stop", "gamma"], how="left")


def language_order(fertility_csv: Path) -> pd.Series:
    """Mean tokens per word per language, averaged over models, sorted ascending."""
    f = pd.read_csv(fertility_csv)
    return f.groupby("lang").fertility_mean.mean().sort_values()


# ---------- plotting ----------

def plot_metric(summary, order, stop, metric, err, ylabel, title, path, ylim=None, ref=None):
    models = [m for m in MODELS if m in set(summary.model)]
    langs = list(order.index)
    x = np.arange(len(langs))
    fig, axes = plt.subplots(len(models), len(GAMMAS), figsize=(4.2 * len(GAMMAS), 2.3 * len(models)),
                             sharex=True, sharey=True, squeeze=False, facecolor=INK["surface"])
    for r, model in enumerate(models):
        for c, gamma in enumerate(GAMMAS):
            ax = axes[r, c]
            ax.set_facecolor(INK["surface"])
            sub = summary[(summary.model == model) & (summary.stop == stop) & (summary.gamma == gamma)]
            for delta in DELTAS:
                d = sub[sub.delta == delta].set_index("lang").reindex(langs)
                y = d[metric].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
                st = DELTA_STYLE[delta]
                ax.plot(x, y, color=st["color"], marker=st["marker"], markersize=5, linewidth=1.5,
                        label=f"δ = {delta:g}")
                if err:
                    ax.errorbar(x, y, yerr=1.96 * d[err].to_numpy(dtype=float), fmt="none",
                                ecolor=st["color"], elinewidth=1, capsize=0)
            if ref is not None:
                ax.axhline(ref, color=INK["muted"], linewidth=1, linestyle="--")
            ax.set_title(f"{model}   γ = {gamma:g}", fontsize=9, color=INK["secondary"], loc="left")
            ax.grid(axis="y", color=INK["grid"], linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK["axis"])
            ax.tick_params(colors=INK["muted"], labelsize=8)
            if ylim:
                ax.set_ylim(*ylim)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=INK["secondary"])
    for ax in axes[-1]:
        ax.set_xticks(x, [f"{l}\n{order[l]:.2f}" for l in langs])
    fig.supxlabel("language  (mean tokens per word across tokenizers →)", fontsize=9, color=INK["secondary"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(DELTAS), frameon=False, fontsize=9)
    fig.suptitle(f"{title}  ({stop} stopping)", x=0.01, ha="left", fontsize=11, color=INK["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=INK["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=Path, default=Path("results/generations"))
    ap.add_argument("--fertility", type=Path, default=Path("results/fertility.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis"))
    args = ap.parse_args()

    order = language_order(args.fertility)
    print("language order (mean tokens/word):", ", ".join(f"{l} {v:.2f}" for l, v in order.items()))

    scores = load_scores(args.generations)
    summary = summarize(scores)
    summary["lang_rank"] = summary.lang.map({l: i for i, l in enumerate(order.index)})
    summary["fertility"] = summary.lang.map(order)
    summary = summary.sort_values(["stop", "model", "gamma", "delta", "lang_rank"])
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index=False)

    for stop in STOP_MODES:
        if stop not in set(summary.stop):
            continue
        plot_metric(summary, order, stop, "kl_mean", "kl_se", "KL(q‖p) per token (nats)",
                    "Distribution shift from the watermark", args.out / f"kl_{stop}.png")
        plot_metric(summary, order, stop, "z_mean", "z_se", "mean z-score",
                    "Detection z-score", args.out / f"z_{stop}.png", ref=Z_DETECT)
        plot_metric(summary, order, stop, "tpr_at_1pct", None, "detected at 1% FPR",
                    "True-positive rate at 1% false-positive rate", args.out / f"tpr_{stop}.png",
                    ylim=(0, 1.02))
        plot_metric(summary, order, stop, "tokens_to_detect", None, "median tokens to z > 4",
                    "Tokens needed for detection", args.out / f"tokens_to_detect_{stop}.png",
                    ylim=(0, 210))
        print(f"wrote plots for {stop} stopping")
    print(f"wrote {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
