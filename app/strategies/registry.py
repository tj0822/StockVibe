from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List

from .base import Strategy, StrategySpec


def _discover_strategy_instances() -> Dict[str, Strategy]:
    strategies: Dict[str, Strategy] = {}

    impl_pkg = importlib.import_module("app.strategies.impl")
    for module_info in pkgutil.iter_modules(impl_pkg.__path__):
        if module_info.ispkg:
            continue
        module = importlib.import_module(f"app.strategies.impl.{module_info.name}")

        instance = getattr(module, "STRATEGY", None)
        if instance is None and hasattr(module, "get_strategy"):
            instance = module.get_strategy()

        if instance is None:
            continue

        if not isinstance(instance, Strategy):
            continue

        strategy_id = instance.spec.strategy_id
        if strategy_id:
            strategies[strategy_id] = instance

    return strategies


def list_strategies() -> List[StrategySpec]:
    strategies = _discover_strategy_instances()
    specs = [strategy.spec for strategy in strategies.values()]
    return sorted(specs, key=lambda s: (s.name.lower(), s.strategy_id))


def get_strategy(strategy_id: str) -> Strategy:
    strategies = _discover_strategy_instances()
    if strategy_id not in strategies:
        available = ", ".join(sorted(strategies.keys()))
        raise KeyError(f"Unknown strategy_id='{strategy_id}'. available=[{available}]")
    return strategies[strategy_id]
