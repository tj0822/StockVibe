from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_minmax_to_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return pd.Series(0.0, index=series.index, dtype=float)

    min_val = float(values.min())
    max_val = float(values.max())

    if np.isclose(max_val, min_val):
        # Avoid divide-by-zero: keep a neutral level when all sectors are tied.
        out = pd.Series(50.0, index=series.index, dtype=float)
        out[values.isna()] = 0.0
        return out

    norm = (values - min_val) / (max_val - min_val)
    return (norm * 100.0).clip(0.0, 100.0)


def compute_sector_power(stock_master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns dataframe with columns:
    - sector
    - stock_count
    - avg_final_score
    - avg_momentum_score
    - buy_count
    - sell_count
    - avg_return_1m
    - sector_power
    - sector_rank
    - sector_state
    """
    output_columns = [
        "sector",
        "stock_count",
        "avg_final_score",
        "avg_momentum_score",
        "buy_count",
        "sell_count",
        "avg_return_1m",
        "sector_power",
        "sector_rank",
        "sector_state",
    ]

    if stock_master_df is None or stock_master_df.empty:
        return pd.DataFrame(columns=output_columns)

    df = stock_master_df.copy()
    df["sector"] = df.get("sector", pd.Series(index=df.index, dtype=object)).astype(str).str.strip()
    df.loc[df["sector"].isin(["", "None", "nan"]), "sector"] = "미분류"

    df["final_score"] = pd.to_numeric(df.get("final_score"), errors="coerce")
    df["momentum_score"] = pd.to_numeric(df.get("momentum_score"), errors="coerce")
    df["return_1m"] = pd.to_numeric(df.get("return_1m"), errors="coerce")
    signals = df.get("signal", pd.Series(index=df.index, dtype=object)).astype(str)

    grouped = (
        df.groupby("sector", as_index=False)
        .agg(
            stock_count=("code", "count"),
            avg_final_score=("final_score", "mean"),
            avg_momentum_score=("momentum_score", "mean"),
            avg_return_1m=("return_1m", "mean"),
        )
        .copy()
    )

    buy_count = signals.eq("BUY").groupby(df["sector"]).sum().rename("buy_count")
    sell_count = signals.eq("SELL").groupby(df["sector"]).sum().rename("sell_count")

    grouped = grouped.merge(buy_count, on="sector", how="left")
    grouped = grouped.merge(sell_count, on="sector", how="left")

    grouped["buy_count"] = pd.to_numeric(grouped["buy_count"], errors="coerce").fillna(0).astype(int)
    grouped["sell_count"] = pd.to_numeric(grouped["sell_count"], errors="coerce").fillna(0).astype(int)

    score_norm = _safe_minmax_to_100(grouped["avg_final_score"]) * 0.4
    momentum_norm = _safe_minmax_to_100(grouped["avg_momentum_score"]) * 0.2
    buy_norm = _safe_minmax_to_100(grouped["buy_count"]) * 0.2
    return_norm = _safe_minmax_to_100(grouped["avg_return_1m"]) * 0.2

    grouped["sector_power"] = (score_norm + momentum_norm + buy_norm + return_norm).clip(0.0, 100.0)
    grouped["sector_rank"] = grouped["sector_power"].rank(method="dense", ascending=False).astype(int)

    grouped["sector_state"] = "WEAK"
    grouped.loc[grouped["sector_rank"] <= 5, "sector_state"] = "LEADING"
    grouped.loc[(grouped["sector_rank"] >= 6) & (grouped["sector_rank"] <= 10), "sector_state"] = "STRONG"
    grouped.loc[(grouped["sector_rank"] >= 11) & (grouped["sector_rank"] <= 20), "sector_state"] = "ROTATION"

    grouped = grouped.sort_values(["sector_rank", "sector"], ascending=[True, True]).reset_index(drop=True)

    # Keep numeric fields consistent for downstream rendering.
    grouped["avg_final_score"] = grouped["avg_final_score"].fillna(0.0).astype(float)
    grouped["avg_momentum_score"] = grouped["avg_momentum_score"].fillna(0.0).astype(float)
    grouped["avg_return_1m"] = grouped["avg_return_1m"].fillna(0.0).astype(float)
    grouped["sector_power"] = grouped["sector_power"].fillna(0.0).astype(float)

    return grouped[output_columns]
