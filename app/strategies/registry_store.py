from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

from app.strategies.registry import list_strategies

REGISTRY_PATH = os.path.join("data", "strategy_registry.json")


DEFAULT_VALIDATION_THRESHOLDS = {
    "overall": {
        "min_cagr": 0.05,
        "max_mdd": 0.25,
        "min_win_rate": 0.52,
        "min_trades": 30,
    },
    "by_regime": {
        "BULL": {"min_cagr": 0.08},
        "BEAR": {"max_mdd": 0.20},
        "SIDEWAYS": {"min_win_rate": 0.50},
        "HIGH_VOL": {"max_mdd": 0.18},
    },
}


def _merge_validation_thresholds(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = raw or {}
    out = deepcopy(DEFAULT_VALIDATION_THRESHOLDS)

    if "overall" not in raw and any(k in raw for k in ["min_cagr", "max_mdd", "min_win_rate", "min_trades"]):
        out["overall"].update(raw)
        return out

    out["overall"].update(raw.get("overall", {}))

    raw_by_regime = raw.get("by_regime", {})
    for regime in ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]:
        out["by_regime"][regime].update(raw_by_regime.get(regime, {}))

    return out


def _ensure_registry_dir() -> None:
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)


def _default_state() -> Dict[str, Any]:
    return {
        "enabled": True,
        "validated": False,
        "in_production": False,
        "last_validation": None,
        "validation_thresholds": deepcopy(DEFAULT_VALIDATION_THRESHOLDS),
        "notes": "",
    }


def _normalize_registry(raw: Dict[str, Any]) -> Dict[str, Any]:
    specs = list_strategies()
    out: Dict[str, Any] = {}

    for spec in specs:
        state = deepcopy(_default_state())
        state.update(raw.get(spec.strategy_id, {}) if isinstance(raw, dict) else {})
        state["validation_thresholds"] = _merge_validation_thresholds(state.get("validation_thresholds"))
        out[spec.strategy_id] = state

    return out


def load_registry() -> Dict[str, Any]:
    _ensure_registry_dir()

    raw: Dict[str, Any] = {}
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, dict):
                    raw = loaded
        except Exception:
            raw = {}

    registry = _normalize_registry(raw)
    save_registry(registry)
    return registry


def save_registry(registry: Dict[str, Any]) -> None:
    _ensure_registry_dir()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, ensure_ascii=False, indent=2)


def set_enabled(strategy_id: str, enabled: bool) -> Dict[str, Any]:
    registry = load_registry()
    if strategy_id not in registry:
        raise KeyError(f"Unknown strategy_id={strategy_id}")
    registry[strategy_id]["enabled"] = bool(enabled)
    save_registry(registry)
    return registry


def set_validated(strategy_id: str, validated: bool) -> Dict[str, Any]:
    registry = load_registry()
    if strategy_id not in registry:
        raise KeyError(f"Unknown strategy_id={strategy_id}")
    registry[strategy_id]["validated"] = bool(validated)
    if not validated:
        registry[strategy_id]["in_production"] = False
    save_registry(registry)
    return registry


def promote_to_production(strategy_id: str) -> Dict[str, Any]:
    registry = load_registry()
    if strategy_id not in registry:
        raise KeyError(f"Unknown strategy_id={strategy_id}")
    if not registry[strategy_id].get("validated", False):
        raise ValueError("Only validated strategies can be promoted to production")
    registry[strategy_id]["in_production"] = True
    save_registry(registry)
    return registry


def demote_from_production(strategy_id: str) -> Dict[str, Any]:
    registry = load_registry()
    if strategy_id not in registry:
        raise KeyError(f"Unknown strategy_id={strategy_id}")
    registry[strategy_id]["in_production"] = False
    save_registry(registry)
    return registry


def update_validation_result(strategy_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    registry = load_registry()
    if strategy_id not in registry:
        raise KeyError(f"Unknown strategy_id={strategy_id}")

    state = registry[strategy_id]
    passed = bool(payload.get("passed", payload.get("validated", False)))
    state["validated"] = passed

    thresholds_payload = payload.get("thresholds")
    if thresholds_payload:
        state["validation_thresholds"] = _merge_validation_thresholds(thresholds_payload)

    state["last_validation"] = {
        "run_ts": payload.get("run_ts") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "universe": payload.get("universe"),
        "date_range": payload.get("date_range"),
        "overall": payload.get("overall", payload.get("metrics", {})),
        "by_regime": payload.get("by_regime", {}),
        "thresholds": _merge_validation_thresholds(payload.get("thresholds", state.get("validation_thresholds"))),
        "passed": passed,
    }
    if not state["validated"]:
        state["in_production"] = False

    save_registry(registry)
    return registry


def get_production_strategy_ids() -> list[str]:
    registry = load_registry()
    return [sid for sid, state in registry.items() if state.get("in_production", False)]


def get_strategy_thresholds(strategy_id: str) -> dict:
    registry = load_registry()
    if strategy_id not in registry:
        raise KeyError(f"Unknown strategy_id={strategy_id}")
    return _merge_validation_thresholds(registry[strategy_id].get("validation_thresholds"))
