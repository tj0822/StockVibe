from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd

from app.signals import build_signals

from app.strategies.base import Strategy, StrategySpec


class TurnoverSpikeStrategy(Strategy):
    spec = StrategySpec(
        strategy_id="turnover_spike",
        name="Turnover Spike",
        version="1.0.0",
        description="거래대금 급등 + 캔들 방향 기반 BUY/SELL 시그널",
        supports_sell=True,
        default_params={
            "turnover_window": 10,
            "turnover_multiplier": 3.0,
            "combine_mode": "ANY",
            "enabled_algos": ["Turnover Spike"],
            "top_n": 10,
        },
    )

    def generate_signals(
        self,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        if price_df is None or price_df.empty:
            return pd.DataFrame(columns=["date", "code", "signal", "confidence", "features_json"])

        merged_params = {**self.spec.default_params, **(params or {})}

        signals = build_signals(
            price_df,
            int(merged_params.get("turnover_window", 10)),
            float(merged_params.get("turnover_multiplier", 3.0)),
            20,
            5.0,
            20,
            2.0,
            20,
            2.0,
            list(merged_params.get("enabled_algos", ["Turnover Spike"])),
            str(merged_params.get("combine_mode", "ANY")),
        )

        if signals.empty:
            return pd.DataFrame(columns=["date", "code", "signal", "confidence", "features_json"])

        out = signals[signals["signal"].isin(["BUY", "SELL"])].copy()
        if out.empty:
            return pd.DataFrame(columns=["date", "code", "signal", "confidence", "features_json"])

        if "spike_ratio" in out.columns:
            ratio = pd.to_numeric(out["spike_ratio"], errors="coerce").fillna(1.0)
            out["confidence"] = (50 + (ratio - 1.0) * 20).clip(0, 100)
        else:
            out["confidence"] = 50.0

        def _row_features(row: pd.Series) -> str:
            payload = {
                "spike_ratio": float(row.get("spike_ratio", 0.0)) if pd.notna(row.get("spike_ratio", None)) else None,
                "avg_turnover": float(row.get("avg_turnover", 0.0)) if pd.notna(row.get("avg_turnover", None)) else None,
                "turnover": float(row.get("turnover", 0.0)) if pd.notna(row.get("turnover", None)) else None,
                "candle": row.get("candle", None),
                "strategy_version": self.spec.version,
            }
            return json.dumps(payload, ensure_ascii=False)

        out["features_json"] = out.apply(_row_features, axis=1)
        out["strategy_id"] = self.spec.strategy_id
        out["strategy_name"] = self.spec.name
        out["strategy_version"] = self.spec.version

        keep_cols = [
            "date",
            "code",
            "signal",
            "confidence",
            "features_json",
            "strategy_id",
            "strategy_name",
            "strategy_version",
        ]

        optional_cols = ["spike_ratio", "open", "close", "high", "low", "volume"]
        for col in optional_cols:
            if col in out.columns:
                keep_cols.append(col)

        out = out[keep_cols].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["code"] = out["code"].astype(str)
        out = out.dropna(subset=["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)

        return out


STRATEGY = TurnoverSpikeStrategy()


def get_strategy() -> TurnoverSpikeStrategy:
    return STRATEGY
