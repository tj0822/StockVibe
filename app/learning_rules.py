from __future__ import annotations

MINIMUM_SAMPLE_COUNT = 20
MAX_WEIGHT_ADJUSTMENT = 0.03
MIN_RECENT_HIT_RATE_20D = 0.45
MAX_DRAWDOWN_WORSENING = 3.0


def has_minimum_samples(count: int) -> bool:
    return int(count or 0) >= MINIMUM_SAMPLE_COUNT


def allows_auto_update(hit_rate_20d: float | None, avg_drawdown_20d: float | None, baseline_drawdown_20d: float | None = None) -> bool:
    hit_rate = float(hit_rate_20d or 0.0)
    drawdown = float(avg_drawdown_20d or 0.0)
    baseline = float(baseline_drawdown_20d) if baseline_drawdown_20d is not None else None

    if hit_rate < MIN_RECENT_HIT_RATE_20D:
        return False
    if baseline is not None and (drawdown - baseline) > MAX_DRAWDOWN_WORSENING:
        return False
    return True
