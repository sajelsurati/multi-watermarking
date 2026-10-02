import math

import torch

from watermark import GreenList, SoftWatermarkProcessor, WatermarkConfig, detect

V = 1000


def test_green_list_size_and_determinism():
    gl = GreenList(WatermarkConfig(gamma=0.25, delta=2.0, vocab_size=V))
    assert gl.mask(7).sum() == 250
    assert torch.equal(gl.mask(7), GreenList(gl.cfg).mask(7))
    assert not torch.equal(gl.mask(7), gl.mask(8))


def test_logged_stats_match_brute_force():
    torch.manual_seed(0)
    cfg = WatermarkConfig(gamma=0.25, delta=2.0, vocab_size=V)
    proc = SoftWatermarkProcessor(cfg)
    ids = torch.randint(0, V, (4, 5))
    scores = torch.randn(4, V) * 3
    out = proc(ids, scores)

    p = scores.softmax(-1)
    q = out.softmax(-1)
    green = proc.greens.batch_mask(ids[:, -1])
    torch.testing.assert_close(proc._steps["green_mass"][0], (p * green).sum(-1))
    torch.testing.assert_close(proc._steps["kl_q_p"][0], (q * (q / p).log()).sum(-1), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(proc._steps["kl_p_q"][0], (p * (p / q).log()).sum(-1), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(proc._steps["entropy"][0], -(p * p.log()).sum(-1))


def test_delta_zero_is_identity():
    proc = SoftWatermarkProcessor(WatermarkConfig(gamma=0.25, delta=0.0, vocab_size=V))
    scores = torch.randn(2, V)
    assert torch.equal(proc(torch.randint(0, V, (2, 3)), scores), scores)
    assert torch.allclose(proc._steps["kl_q_p"][0], torch.zeros(2))


def _sample(proc, T, B=8):
    """Generate with uniform base logits through the processor, like HF generate."""
    torch.manual_seed(1)
    ids = torch.randint(0, V, (B, 1))
    for _ in range(T):
        logits = proc(ids, torch.zeros(B, V))
        nxt = torch.multinomial(logits.softmax(-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


def test_detection_null_and_watermarked():
    T = 200
    null = SoftWatermarkProcessor(WatermarkConfig(gamma=0.25, delta=0.0, vocab_size=V))
    strong = SoftWatermarkProcessor(WatermarkConfig(gamma=0.25, delta=10.0, vocab_size=V))
    greens = GreenList(strong.cfg)  # same gamma/key -> same lists as the null run's detector

    z_null = [detect(s[1:], s[0].item(), greens)["z"] for s in _sample(null, T)]
    z_wm = [detect(s[1:], s[0].item(), greens)["z"] for s in _sample(strong, T)]
    assert abs(sum(z_null) / len(z_null)) < 1.5
    # Uniform logits + delta=10 -> nearly every token green -> z ~ sqrt(T * (1-g)/g) ~ 24.5
    assert min(z_wm) > 20


def test_sampled_green_matches_detector():
    cfg = WatermarkConfig(gamma=0.5, delta=2.0, vocab_size=V)
    proc = SoftWatermarkProcessor(cfg)
    seqs = _sample(proc, 50)
    stats = proc.finalize(seqs)
    assert stats["sampled_green"].shape == (8, 50)
    for row, s in zip(stats["sampled_green"], seqs):
        assert row.sum().item() == detect(s[1:], s[0].item(), proc.greens)["green"]


def test_shadow_processor_logs_but_does_not_modify():
    proc = SoftWatermarkProcessor(WatermarkConfig(gamma=0.25, delta=2.0, vocab_size=V), apply=False)
    scores = torch.randn(2, V)
    assert torch.equal(proc(torch.randint(0, V, (2, 3)), scores), scores)
    assert (proc._steps["kl_q_p"][0] > 0).all()  # counterfactual KL is still logged
