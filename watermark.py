"""Soft watermark from Kirchenbauer et al. 2023 (arXiv:2301.10226), with per-step logging.

Generation: at each step, seed an RNG with hash_key * prev_token, take a random
permutation of the vocabulary, call the first gamma*|V| ids "green", and add
delta to their logits.

Logging: for every generated position we record quantities computed from the
*unwatermarked* distribution p and the watermarked distribution q:
  entropy        H(p)
  green_mass     G = sum_{i in green} p_i
  kl_q_p         KL(q || p) = delta * Q_G - log Z      (Q_G = e^delta G / Z)
  kl_p_q         KL(p || q) = log Z - delta * G
  spike_entropy  S(p, z) = sum_i p_i / (1 + z p_i),  z = (1-gamma)(e^delta-1) / (1+(e^delta-1)gamma)
where Z = 1 + (e^delta - 1) G. Spike entropy is the quantity the paper's
detectability bound (Thm 4.2) is stated in.

With apply=False the processor leaves logits unchanged but still logs the
statistics for its (gamma, delta) — "shadow" logging, so unwatermarked baseline
runs get the counterfactual green mass, KL and green hits for every setting.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import LogitsProcessor

DEFAULT_HASH_KEY = 15485863  # the prime used in the paper's reference code


@dataclass(frozen=True)
class WatermarkConfig:
    gamma: float
    delta: float
    vocab_size: int  # logits dimension (model.config.vocab_size), NOT len(tokenizer)
    hash_key: int = DEFAULT_HASH_KEY


class GreenList:
    """Device-independent green-list lookup: lists are generated on CPU and cached.

    Use GreenList.shared(cfg) to reuse one cache across processors with the same
    gamma/vocab/key (each cached mask is |V| bytes, ~150 KB for Qwen).
    """

    _instances: dict = {}

    @classmethod
    def shared(cls, cfg: WatermarkConfig) -> "GreenList":
        key = (cfg.gamma, cfg.vocab_size, cfg.hash_key)
        if key not in cls._instances:
            cls._instances[key] = cls(cfg)
        return cls._instances[key]

    def __init__(self, cfg: WatermarkConfig, cache_size: int = 4096):
        self.cfg = cfg
        self.green_size = int(cfg.gamma * cfg.vocab_size)
        self._mask = lru_cache(maxsize=cache_size)(self._compute_mask)

    def _compute_mask(self, prev_token: int) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.cfg.hash_key * prev_token)
        perm = torch.randperm(self.cfg.vocab_size, generator=g)
        mask = torch.zeros(self.cfg.vocab_size, dtype=torch.bool)
        mask[perm[: self.green_size]] = True
        return mask

    def mask(self, prev_token: int) -> torch.Tensor:
        return self._mask(int(prev_token))

    def batch_mask(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """(B,) previous tokens -> (B, V) bool green mask on the same device."""
        rows = [self.mask(t) for t in prev_tokens.tolist()]
        return torch.stack(rows).to(prev_tokens.device)


class SoftWatermarkProcessor(LogitsProcessor):
    """Adds delta to green-list logits and records per-step distribution statistics.

    Must be the only logits transform in the pipeline (temperature 1.0, no
    top-k/top-p, no repetition penalty) for the logged p to be the model's
    true next-token distribution.
    """

    STATS = ("entropy", "green_mass", "kl_q_p", "kl_p_q", "spike_entropy", "sampled_green")

    def __init__(self, cfg: WatermarkConfig, apply: bool = True, log_stats: bool = True):
        self.cfg = cfg
        self.greens = GreenList.shared(cfg)
        self.apply = apply
        self.log_stats = log_stats
        self.reset()
        d, g = cfg.delta, cfg.gamma
        self._z = (1 - g) * (math.exp(d) - 1) / (1 + (math.exp(d) - 1) * g)

    def reset(self):
        """Call before each generate() call."""
        self._steps = {k: [] for k in self.STATS}
        self._last_green = None  # green mask of the previous step, to score the sampled token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[-1] != self.cfg.vocab_size:
            raise ValueError(f"logits dim {scores.shape[-1]} != cfg.vocab_size {self.cfg.vocab_size}")

        # Was the token sampled at the previous step green? (input_ids[:, -1] is that token.)
        if self.log_stats and self._last_green is not None:
            sampled = input_ids[:, -1:]
            self._steps["sampled_green"].append(self._last_green.gather(1, sampled).squeeze(1).cpu())

        green = self.greens.batch_mask(input_ids[:, -1])
        if self.log_stats:
            self._log(scores, green)
            self._last_green = green
        if not self.apply:
            return scores
        return scores + self.cfg.delta * green.to(scores.dtype)

    @torch.no_grad()
    def _log(self, scores: torch.Tensor, green: torch.Tensor):
        logp = torch.log_softmax(scores.float(), dim=-1)
        p = logp.exp()
        d = self.cfg.delta
        G = (p * green).sum(-1)
        Z = 1 + math.expm1(d) * G
        logZ = torch.log(Z)
        QG = math.exp(d) * G / Z
        stats = {
            "entropy": -(p * logp).nan_to_num().sum(-1),
            "green_mass": G,
            "kl_q_p": d * QG - logZ,
            "kl_p_q": logZ - d * G,
            "spike_entropy": (p / (1 + self._z * p)).sum(-1),
        }
        for k, v in stats.items():
            self._steps[k].append(v.cpu())

    def finalize(self, sequences: torch.LongTensor) -> dict:
        """Pass generate()'s output sequences; returns {stat: (B, T) tensor}.

        The last sampled token is never seen by __call__, so its green status
        is recovered here from the returned sequences.
        """
        if self._last_green is not None:
            last = sequences[:, -1:].to(self._last_green.device)
            self._steps["sampled_green"].append(self._last_green.gather(1, last).squeeze(1).cpu())
        out = {k: torch.stack(v, dim=1) for k, v in self._steps.items() if v}
        self.reset()
        return out


def detect(token_ids, prev_token: int, greens: GreenList, ignore_mask=None) -> dict:
    """Score one generated sequence.

    token_ids:   generated ids (prompt excluded), 1-D sequence.
    prev_token:  last prompt token (context for the first generated token).
    ignore_mask: optional bool per position to exclude (e.g. padding after EOS).

    Returns green count, T, z-score, and the running z-score at every prefix
    length (for detectability-vs-length curves).
    """
    ids = torch.as_tensor(token_ids).tolist()
    prevs = [prev_token] + ids[:-1]
    keep = [True] * len(ids) if ignore_mask is None else [not m for m in torch.as_tensor(ignore_mask).tolist()]
    hits = [bool(greens.mask(pv)[t]) for pv, t, k in zip(prevs, ids, keep) if k]

    g = greens.cfg.gamma
    running, count = [], 0
    for T, h in enumerate(hits, start=1):
        count += h
        running.append((count - g * T) / math.sqrt(T * g * (1 - g)))
    return {
        "green": count,
        "T": len(hits),
        "z": running[-1] if running else float("nan"),
        "z_by_length": running,
    }
