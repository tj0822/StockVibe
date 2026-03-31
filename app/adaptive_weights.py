from __future__ import annotations

import json
import os

import pandas as pd

from app.learning_rules import MAX_WEIGHT_ADJUSTMENT, MINIMUM_SAMPLE_COUNT


DEFAULT_WEIGHTS = {
    "sector_power": 0.17,
    "financial_score": 0.17,
    "momentum_score": 0.15,
    "signal_strength": 0.15,
    "strategy_fit": 0.14,
    "money_flow_score": 0.10,
    "sector_prediction_score": 0.12,
}


def load_adaptive_weights(path: str = "data/adaptive_weights.json") -> dict:
    if not os.path.exists(path):
        return DEFAULT_WEIGHTS.copy()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        weights = DEFAULT_WEIGHTS.copy()
        if isinstance(raw, dict):
            for key in DEFAULT_WEIGHTS:
                if key in raw:
                    weights[key] = float(raw[key])
        return _normalize_weights(weights)
    except Exception:
        return DEFAULT_WEIGHTS.copy()


def save_adaptive_weights(weights: dict, path: str = "data/adaptive_weights.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_normalize_weights(weights), fh, ensure_ascii=False, indent=2)


def _normalize_weights(weights: dict) -> dict:
    base = {key: float(weights.get(key, DEFAULT_WEIGHTS[key])) for key in DEFAULT_WEIGHTS}
    total = sum(max(0.0, value) for value in base.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    normalized = {key: max(0.0, value) / total for key, value in base.items()}
    diff = 1.0 - sum(normalized.values())
    first_key = next(iter(normalized))
    normalized[first_key] += diff
    return normalized


def suggest_adaptive_weights(
    feature_perf_df: pd.DataFrame,
    previous_weights: dict = None,
) -> dict:
    """
    Suggest new weights based on feature performance.
    Apply smoothing and max delta constraints.
    """
    previous = _normalize_weights(previous_weights or DEFAULT_WEIGHTS)
    if feature_perf_df is None or feature_perf_df.empty:
        return previous

    perf = feature_perf_df.copy()
    if "feature" not in perf.columns or "count" not in perf.columns:
        return previous

    sample_count = int(pd.to_numeric(perf["count"], errors="coerce").fillna(0).max())
    if sample_count < MINIMUM_SAMPLE_COUNT:
        return previous

    perf = perf[perf["feature"].isin(DEFAULT_WEIGHTS.keys())].copy()
    if perf.empty:
        return previous

    perf["correlation_20d"] = pd.to_numeric(perf.get("correlation_20d"), errors="coerce").fillna(0.0)
    perf["spread_20d"] = pd.to_numeric(perf.get("spread_20d"), errors="coerce").fillna(0.0)
    perf["strength"] = (perf["correlation_20d"].clip(lower=0.0) * 0.7) + (perf["spread_20d"].clip(lower=0.0) * 0.3 / 10.0)

    strength_map = {row["feature"]: float(row["strength"]) for _, row in perf.iterrows()}
    total_strength = sum(strength_map.values())
    if total_strength <= 0:
        return previous

    target = {}
    for key in DEFAULT_WEIGHTS:
        target[key] = strength_map.get(key, 0.0) / total_strength if total_strength > 0 else DEFAULT_WEIGHTS[key]

    alpha = 0.35
    smoothed = {key: ((1.0 - alpha) * previous[key]) + (alpha * target.get(key, previous[key])) for key in DEFAULT_WEIGHTS}

    constrained = {}
    for key in DEFAULT_WEIGHTS:
        delta = smoothed[key] - previous[key]
        delta = max(-MAX_WEIGHT_ADJUSTMENT, min(MAX_WEIGHT_ADJUSTMENT, delta))
        constrained[key] = previous[key] + delta

    return _normalize_weights(constrained)
