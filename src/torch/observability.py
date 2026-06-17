"""Training observability: EMA, GPU memory tracking, performance instrumentation.

Ported from ``arc_toolkit.observability``. All torch-only, no storage
dependency — these complement the IO layer for end-to-end training support.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import torch
import torch.nn as nn

__all__ = ["EMA", "PerfTracker", "StageTimer", "get_gpu_memory_stats"]


class StageTimer:
    """Records wall-clock for coarse training-lifecycle stages in one run."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Accumulate the wall-clock time of the wrapped block under ``name``."""
        t = time.perf_counter()
        try:
            yield
        finally:
            if self.enabled:
                self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t)

    def record(self, name: str, seconds: float) -> None:
        """Manually accumulate ``seconds`` under ``name``."""
        if self.enabled:
            self.stages[name] = self.stages.get(name, 0.0) + float(seconds)

    def total_s(self) -> float:
        """Wall-clock seconds since this timer was constructed."""
        return time.perf_counter() - self._t0

    def summary(self) -> dict[str, float]:
        """All stage totals plus ``total_main_s`` (overall wall-clock)."""
        out = dict(self.stages)
        out["total_main_s"] = self.total_s()
        return out

    def format_table(self) -> str:
        """Human-readable stage table for logging."""
        s = self.summary()
        rows = "\n".join(f"  {k:28s} {v:8.2f}s" for k, v in s.items())
        return "Stage profile (end-to-end, this run):\n" + rows


_HAS_FOREACH = hasattr(torch, "_foreach_lerp_")


class EMA:
    """Exponential moving average of model parameters.

    ``update()`` uses grouped ``torch._foreach_lerp_`` (one kernel per
    device/dtype group instead of one per parameter). ``device="cpu"`` keeps the
    shadow on CPU (saves GPU memory; each update then pays one bulk D2H copy).
    ``include_buffers=True`` additionally tracks buffers (copied, not averaged).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        *,
        device: torch.device | str | None = None,
        include_buffers: bool = False,
    ):
        self.decay = decay
        self.include_buffers = include_buffers
        self._device = torch.device(device) if device is not None else None
        self.shadow: dict[str, torch.Tensor] = {}
        self._param_names: set[str] = set()
        self._buffer_names: set[str] = set()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = self._clone(p)
                self._param_names.add(name)
        if include_buffers:
            for name, b in model.named_buffers():
                self.shadow[name] = self._clone(b)
                self._buffer_names.add(name)
        self.backup: dict[str, torch.Tensor] = {}
        self._cache_key: int | None = None
        self._lerp_groups: list[tuple[list, list, list | None]] = []
        self._copy_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _clone(self, t: torch.Tensor) -> torch.Tensor:
        c = t.detach().clone()
        return c.to(self._device) if self._device is not None else c

    def _build_cache(self, model: nn.Module) -> None:
        groups: dict[tuple, tuple[list, list]] = {}
        for name, p in model.named_parameters():
            if name in self._param_names:
                sh = self.shadow[name]
                shadows, srcs = groups.setdefault((p.device, p.dtype, sh.device), ([], []))
                shadows.append(sh)
                srcs.append(p)
        self._lerp_groups = []
        for (p_dev, p_dtype, s_dev), (shadows, srcs) in groups.items():
            if p_dev == s_dev:
                self._lerp_groups.append((shadows, srcs, None))
            else:
                pin = s_dev.type == "cpu" and p_dev.type == "cuda"
                staging = [
                    torch.empty(p.shape, dtype=p_dtype, device=s_dev, pin_memory=pin) for p in srcs
                ]
                self._lerp_groups.append((shadows, srcs, staging))
        self._copy_pairs = [
            (self.shadow[name], b)
            for name, b in model.named_buffers()
            if name in self._buffer_names
        ]
        self._cache_key = id(model)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend current parameters into the shadow: ``shadow = lerp(shadow, p, 1-decay)``."""
        if self._cache_key != id(model):
            self._build_cache(model)
        weight = 1.0 - self.decay
        needs_sync = False
        for _shadows, srcs, staging in self._lerp_groups:
            if staging is not None:
                for st, p in zip(staging, srcs):
                    st.copy_(p, non_blocking=True)
                needs_sync = needs_sync or srcs[0].device.type == "cuda"
        if needs_sync:
            torch.cuda.synchronize()
        for shadows, srcs, staging in self._lerp_groups:
            srcs = staging if staging is not None else srcs
            if _HAS_FOREACH:
                torch._foreach_lerp_(shadows, srcs, weight)
            else:
                for sh, p in zip(shadows, srcs):
                    sh.lerp_(p, weight)
        for sh, b in self._copy_pairs:
            sh.copy_(b)

    def apply_shadow(self, model: nn.Module) -> None:
        """Swap shadow weights into the model (back up the originals first)."""
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore the original weights saved by :meth:`apply_shadow`."""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Flat ``{name: tensor}`` of the shadow weights."""
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Replace the shadow, keeping each entry on its current device."""
        new: dict[str, torch.Tensor] = {}
        for name, value in state.items():
            ref = self.shadow.get(name)
            dev = ref.device if ref is not None else (self._device or value.device)
            new[name] = value.detach().clone().to(dev)
        self.shadow = new
        self._cache_key = None

    def __len__(self) -> int:
        return len(self.shadow)


def get_gpu_memory_stats(device: torch.device | int | None = None) -> dict[str, float]:
    """Current/peak CUDA memory in GiB ({} when CUDA is unavailable)."""
    if not torch.cuda.is_available():
        return {}
    if device is None:
        device = torch.cuda.current_device()
    return {
        "allocated_GiB": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_GiB": torch.cuda.memory_reserved(device) / (1024**3),
        "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / (1024**3),
    }


def _percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = (len(sorted_xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


class _CpuClock:
    """perf_counter marks; always immediately readable."""

    def mark(self) -> float:
        return time.perf_counter()

    def ready(self, mark: float) -> bool:
        return True

    def synchronize(self, mark: float) -> None:
        pass

    def elapsed_s(self, start: float, end: float) -> float:
        return end - start

    def recycle(self, *marks) -> None:
        pass


class _CudaClock:
    """CUDA-event marks drawn from a free-list pool, read out lazily."""

    def __init__(self) -> None:
        self._free: list[torch.cuda.Event] = []

    def mark(self) -> torch.cuda.Event:
        event = self._free.pop() if self._free else torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def ready(self, mark: torch.cuda.Event) -> bool:
        return mark.query()

    def synchronize(self, mark: torch.cuda.Event) -> None:
        mark.synchronize()

    def elapsed_s(self, start: torch.cuda.Event, end: torch.cuda.Event) -> float:
        return start.elapsed_time(end) / 1000.0

    def recycle(self, *marks) -> None:
        self._free.extend(m for m in marks if m is not None)


# (record key, series attribute, gating flag attribute) for paired marks.
_MARK_SPECS = (
    ("h2d", "_h2d_times", "track_io"),
    ("compute", "_compute_times", "track_io"),
    ("fwd", "_fwd_times", "track_compute_breakdown"),
    ("bwd", "_bwd_times", "track_compute_breakdown"),
    ("opt", "_opt_times", "track_compute_breakdown"),
)


class PerfTracker:
    """Track training throughput and optional IO/compute timing.

    On CUDA, timing uses event pairs that are read out *lazily*: ``step_end``
    never synchronizes the device — completed steps are drained via
    ``event.query()`` and the reported ``step_time_s`` is the most recent
    *completed* step (it can lag the submitted step by the in-flight depth,
    typically 1-2). ``summary()`` and ``raw_series()`` drain exactly. On CPU,
    ``perf_counter`` is used and there is no lag.
    """

    def __init__(
        self,
        window_size: int = 50,
        warmup_steps: int = 10,
        track_io: bool = False,
        track_compute_breakdown: bool = True,
        max_pending: int = 128,
    ):
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.track_io = track_io
        self.track_compute_breakdown = track_compute_breakdown
        self.max_pending = max_pending

        self._clock = _CudaClock() if torch.cuda.is_available() else _CpuClock()
        self._cur: dict | None = None
        self._pending: deque[dict] = deque()
        self._fetch_t0: float = 0.0

        self._step_times: deque[float] = deque(maxlen=window_size)
        self._all_step_times: list[float] = []
        self._fetch_times: list[float] = []
        self._h2d_times: list[float] = []
        self._compute_times: list[float] = []
        self._fwd_times: list[float] = []
        self._bwd_times: list[float] = []
        self._opt_times: list[float] = []
        self._bytes_per_step: list[int] = []

        self.total_steps: int = 0
        self.total_samples: int = 0
        self._last_batch_size: int = 0
        self._global_start: float = time.time()
        self._steady_start_time: float | None = None

    # ------------------------------------------------------------------ marks
    def step_start(self) -> None:
        """Mark the beginning of a training step."""
        self._cur = {"start": self._clock.mark()}

    def step_end(self, batch_size: int) -> dict[str, float]:
        """Mark the end of a step; return rolling stats for the latest completed step."""
        rec = self._cur
        assert rec is not None, "step_end() called without step_start()"
        rec["end"] = self._clock.mark()
        self._cur = None
        self._pending.append(rec)

        self.total_steps += 1
        self.total_samples += batch_size
        self._last_batch_size = batch_size
        if self._steady_start_time is None and self.total_steps > self.warmup_steps:
            self._steady_start_time = time.time()

        if len(self._pending) >= self.max_pending:
            self._clock.synchronize(self._pending[0]["end"])
        self._drain()
        if not self._all_step_times:
            # Nothing completed yet (only the first step or two) — force it through.
            self._clock.synchronize(self._pending[0]["end"])
            self._drain()
        return self._latest_stats()

    def fetch_start(self) -> None:
        """Mark the start of the dataloader fetch (CPU wall-clock)."""
        if self.track_io:
            self._fetch_t0 = time.perf_counter()

    def fetch_end(self) -> None:
        """Mark the end of the dataloader fetch."""
        if self.track_io and self._cur is not None:
            self._cur["fetch_s"] = time.perf_counter() - self._fetch_t0

    def record_bytes(self, n: int) -> None:
        """Record bytes read for the current step (feeds read-bandwidth stats)."""
        if self.track_io and self._cur is not None:
            self._cur["nbytes"] = int(n)

    @contextmanager
    def mark(self, key: Literal["h2d", "compute", "fwd", "bwd", "opt"]) -> Iterator[None]:
        """Time a named sub-step of the current step.

        ``key`` in {``h2d``, ``compute``} is gated by ``track_io``; {``fwd``,
        ``bwd``, ``opt``} by ``track_compute_breakdown``. Use as a context
        manager: ``with perf.mark("fwd"): out = model(...)``.
        """
        flag = self.track_io if key in ("h2d", "compute") else self.track_compute_breakdown
        if not flag or self._cur is None:
            yield
            return
        self._cur[f"{key}_start"] = self._clock.mark()
        try:
            yield
        finally:
            self._cur[f"{key}_end"] = self._clock.mark()

    # ------------------------------------------------------------- completion
    def _drain(self) -> None:
        while self._pending and self._clock.ready(self._pending[0]["end"]):
            self._finalize(self._pending.popleft())

    def _drain_all(self) -> None:
        if self._pending:
            self._clock.synchronize(self._pending[-1]["end"])
            self._drain()

    def _finalize(self, rec: dict) -> None:
        clk = self._clock
        elapsed = clk.elapsed_s(rec["start"], rec["end"])
        self._step_times.append(elapsed)
        self._all_step_times.append(elapsed)
        recycle = [rec["start"], rec["end"]]
        if self.track_io:
            self._fetch_times.append(rec.get("fetch_s", 0.0))
            if rec.get("nbytes") is not None:
                self._bytes_per_step.append(rec["nbytes"])
        for key, series_attr, flag_attr in _MARK_SPECS:
            if not getattr(self, flag_attr):
                continue
            start, end = rec.get(f"{key}_start"), rec.get(f"{key}_end")
            series: list[float] = getattr(self, series_attr)
            series.append(
                clk.elapsed_s(start, end) if start is not None and end is not None else 0.0
            )
            recycle += [start, end]
        clk.recycle(*recycle)

    def _latest_stats(self) -> dict[str, float]:
        elapsed = self._all_step_times[-1]
        avg = sum(self._step_times) / len(self._step_times)
        bs = self._last_batch_size
        out: dict[str, float] = {
            "step_time_s": elapsed,
            "avg_step_time_s": avg,
            "samples_per_sec": bs / elapsed if elapsed > 0 else 0.0,
            "avg_samples_per_sec": bs / avg if avg > 0 else 0.0,
        }
        if self.track_io and self._fetch_times:
            out["fetch_time_s"] = self._fetch_times[-1]
            fetch_window = self._fetch_times[-self.window_size :]
            step_window = list(self._step_times)[-len(fetch_window) :]
            out["stall_pct"] = 100.0 * sum(fetch_window) / max(sum(step_window), 1e-9)
            if self._bytes_per_step:
                bw_window_bytes = sum(self._bytes_per_step[-self.window_size :])
                bw_window_sec = sum(step_window)
                if bw_window_sec > 0:
                    out["read_bw_gibps"] = bw_window_bytes / (1024**3) / bw_window_sec
        return out

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict[str, float]:
        """Exact end-of-run aggregates (drains all in-flight timing first)."""
        self._drain_all()
        wall_time = time.time() - self._global_start
        result: dict[str, float] = {
            "total_steps": self.total_steps,
            "total_samples": self.total_samples,
            "wall_time_s": wall_time,
            "overall_samples_per_sec": self.total_samples / wall_time if wall_time > 0 else 0,
        }

        if self._all_step_times:
            result["avg_step_time_s"] = sum(self._all_step_times) / len(self._all_step_times)

        steady = self._all_step_times[self.warmup_steps :]
        if steady:
            avg = sum(steady) / len(steady)
            result["steady_avg_step_time_s"] = avg
            if avg > 0 and self._last_batch_size > 0:
                result["steady_samples_per_sec"] = self._last_batch_size / avg

        if self.track_compute_breakdown:

            def _steady_mean(xs: list[float]) -> float:
                tail = xs[self.warmup_steps :] or xs
                return sum(tail) / len(tail) if tail else 0.0

            if self._fwd_times:
                result["steady_forward_s"] = _steady_mean(self._fwd_times)
                result["steady_backward_s"] = _steady_mean(self._bwd_times)
                result["steady_optimizer_s"] = _steady_mean(self._opt_times)

        if not self.track_io:
            return result

        def _steady_or_full(xs: list) -> list:
            tail = xs[self.warmup_steps :]
            return tail if tail else xs

        fetch_steady = _steady_or_full(self._fetch_times)
        h2d_steady = _steady_or_full(self._h2d_times)
        compute_steady = _steady_or_full(self._compute_times)
        bytes_steady = _steady_or_full(self._bytes_per_step)
        step_steady = _steady_or_full(self._all_step_times)

        if fetch_steady:
            s = sorted(fetch_steady)
            result["mean_fetch_time_s"] = sum(s) / len(s)
            result["p50_fetch_time_s"] = _percentile(s, 0.50)
            result["p99_fetch_time_s"] = _percentile(s, 0.99)

        if h2d_steady:
            s = sorted(h2d_steady)
            result["mean_h2d_time_s"] = sum(s) / len(s)
            result["p99_h2d_time_s"] = _percentile(s, 0.99)

        if compute_steady:
            result["mean_compute_time_s"] = sum(compute_steady) / len(compute_steady)

        step_steady_sum = sum(step_steady) if step_steady else 0.0
        if step_steady_sum > 0 and fetch_steady:
            result["dataloader_stall_pct"] = 100.0 * sum(fetch_steady) / step_steady_sum

        if bytes_steady:
            total_bytes = sum(bytes_steady)
            result["bytes_read_total_gib"] = total_bytes / (1024**3)
            steady_wall = (
                time.time() - self._steady_start_time
                if self._steady_start_time is not None
                else step_steady_sum
            )
            if steady_wall > 0:
                result["per_rank_read_bw_gibps"] = total_bytes / (1024**3) / steady_wall

        return result

    def raw_series(self) -> dict[str, list[float] | list[int]]:
        """Per-step time series (drains all in-flight timing first)."""
        self._drain_all()
        return {
            "step_times_s": list(self._all_step_times),
            "fetch_times_s": list(self._fetch_times),
            "h2d_times_s": list(self._h2d_times),
            "compute_times_s": list(self._compute_times),
            "bytes_per_step": list(self._bytes_per_step),
        }
