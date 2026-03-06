from __future__ import annotations

import numpy as np
import pandas as pd


def classify_market_regime(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: KOSPI index df with date, close
    Output: same df plus:
    - return_20d
    - vol_20d
    - regime
    """
    if index_df is None or index_df.empty:
        return pd.DataFrame(columns=["date", "close", "return_20d", "vol_20d", "regime"])

    out = index_df.copy()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")

    # Support either `close` or `index` input naming.
    if "close" in out.columns:
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
    elif "index" in out.columns:
        out["close"] = pd.to_numeric(out["index"], errors="coerce")
    else:
        out["close"] = np.nan

    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=["date", "close", "return_20d", "vol_20d", "regime"])

    daily_ret = out["close"].pct_change()
    out["return_20d"] = out["close"].pct_change(20)
    out["vol_20d"] = daily_ret.rolling(20, min_periods=20).std() * np.sqrt(252)

    vol_non_null = out["vol_20d"].dropna()
    # Dynamic volatility threshold keeps behavior robust across market regimes.
    vol_threshold = float(vol_non_null.quantile(0.7)) if not vol_non_null.empty else 0.0

    out["regime"] = "SIDEWAYS"

    bull_mask = out["return_20d"] > 0.05
    bear_mask = out["return_20d"] < -0.05
    high_vol_mask = (
        out["return_20d"].abs().le(0.05)
        & out["vol_20d"].gt(vol_threshold)
    )

    out.loc[bull_mask, "regime"] = "BULL"
    out.loc[bear_mask, "regime"] = "BEAR"
    out.loc[high_vol_mask & ~bull_mask & ~bear_mask, "regime"] = "HIGH_VOL"

    return out
