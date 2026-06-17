"""arcstore.torch.observability: PerfTracker CPU path, EMA correctness, StageTimer."""
from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from arcstore.torch import EMA, PerfTracker, StageTimer  # noqa: E402


def _full_step(perf: PerfTracker, batch_size: int = 2) -> dict:
    perf.step_start()
    perf.fetch_start()
    time.sleep(0.001)
    perf.fetch_end()
    perf.record_bytes(100)
    with perf.mark("h2d"):
        pass
    with perf.mark("compute"):
        with perf.mark("fwd"):
            pass
        with perf.mark("bwd"):
            pass
        with perf.mark("opt"):
            pass
    return perf.step_end(batch_size=batch_size)


def test_perf_tracker_cpu_full_run():
    perf = PerfTracker(track_io=True, warmup_steps=2)
    for _ in range(5):
        out = _full_step(perf)
        assert out["step_time_s"] > 0
        assert out["samples_per_sec"] > 0
        assert "fetch_time_s" in out

    summary = perf.summary()
    assert summary["total_steps"] == 5
    assert summary["total_samples"] == 10
    assert len(perf._pending) == 0  # everything drained
    series = perf.raw_series()
    assert len(series["step_times_s"]) == 5
    assert sum(series["bytes_per_step"]) == 500


def test_perf_tracker_minimal_run():
    perf = PerfTracker(track_io=False, track_compute_breakdown=False)
    perf.step_start()
    perf.step_end(batch_size=2)
    summary = perf.summary()
    assert summary["total_steps"] == 1
    assert summary["total_samples"] == 2


def test_ema_update_matches_reference():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for name, p in model.named_parameters():
        expected = reference[name].lerp(p, 1.0 - 0.9)
        assert torch.equal(ema.shadow[name], expected), name


def test_ema_apply_and_restore():
    model = nn.Linear(4, 2)
    ema = EMA(model, decay=0.0)  # shadow == params at init
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.apply_shadow(model)
    assert torch.equal(model.weight, original)
    ema.restore(model)
    assert torch.equal(model.weight, torch.full_like(model.weight, 5.0))


def test_stage_timer_accumulate_and_summary():
    timer = StageTimer()
    with timer.stage("init"):
        time.sleep(0.001)
    with timer.stage("train"):
        time.sleep(0.001)
    s = timer.summary()
    assert s["init"] > 0
    assert "total_main_s" in s
    assert "train" in timer.format_table()
