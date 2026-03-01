from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    name: str
    version: str
    description: str
    supports_sell: bool
    default_params: Dict[str, Any]


class Strategy:
    spec: StrategySpec

    def generate_signals(
        self,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Return signals_df with columns:
        - date
        - code
        - signal ("BUY"|"SELL")
        - confidence (0~100)
        - features_json (dict serialized)
        """
        raise NotImplementedError
