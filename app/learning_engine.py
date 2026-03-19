from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _aggregate_performance(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "count", "avg_return_5d", "avg_return_20d", "hit_rate_20d", "avg_drawdown_20d"])

    work = df.copy()
    work["return_5d"] = _safe_numeric(work.get("return_5d", pd.Series(index=work.index)))
    work["return_20d"] = _safe_numeric(work.get("return_20d", pd.Series(index=work.index)))
    work["hit_20d"] = _safe_numeric(work.get("hit_20d", pd.Series(index=work.index)))
    work["max_drawdown_20d"] = _safe_numeric(work.get("max_drawdown_20d", pd.Series(index=work.index)))

    summary = (
        work.groupby(group_col, dropna=False)
        .agg(
            count=(group_col, "size"),
            avg_return_5d=("return_5d", "mean"),
            avg_return_20d=("return_20d", "mean"),
            hit_rate_20d=("hit_20d", "mean"),
            avg_drawdown_20d=("max_drawdown_20d", "mean"),
        )
        .reset_index()
    )
    summary["hit_rate_20d"] = summary["hit_rate_20d"] * 100.0
    return summary.sort_values(["avg_return_20d", "hit_rate_20d"], ascending=[False, False]).reset_index(drop=True)


def evaluate_strategy_performance(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df is None or decision_df.empty:
        return pd.DataFrame(columns=["strategy", "count", "avg_return_5d", "avg_return_20d", "hit_rate_20d", "avg_drawdown_20d"])

    work = decision_df.copy()
    strategy_col = None
    for candidate in ["triggered_strategy", "recommended_strategy", "strategy"]:
        if candidate in work.columns:
            strategy_col = candidate
            break
    if strategy_col is None:
        return pd.DataFrame(columns=["strategy", "count", "avg_return_5d", "avg_return_20d", "hit_rate_20d", "avg_drawdown_20d"])

    work[strategy_col] = work[strategy_col].fillna("").astype(str)
    work["strategy"] = work[strategy_col].str.split(",")
    work = work.explode("strategy")
    work["strategy"] = work["strategy"].fillna("").astype(str).str.strip()
    work = work[work["strategy"] != ""]
    return _aggregate_performance(work, "strategy")


def evaluate_sector_performance(decision_df: pd.DataFrame) -> pd.DataFrame:
    return _aggregate_performance(decision_df, "sector")


def evaluate_regime_performance(decision_df: pd.DataFrame) -> pd.DataFrame:
    regime_col = "market_regime" if decision_df is not None and "market_regime" in decision_df.columns else "market_bias"
    return _aggregate_performance(decision_df, regime_col)


def evaluate_feature_importance(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df is None or decision_df.empty or "return_20d" not in decision_df.columns:
        return pd.DataFrame(columns=["feature", "count", "correlation_20d", "top_bucket_return_20d", "bottom_bucket_return_20d", "spread_20d", "bucket_summary"])

    feature_cols = [
        "sector_power",
        "financial_score",
        "momentum_score",
        "signal_strength",
        "strategy_fit",
        "money_flow_score",
        "sector_prediction_score",
        "investment_score",
        "confidence",
    ]
    work = decision_df.copy()
    work["return_20d"] = _safe_numeric(work["return_20d"])
    work = work.dropna(subset=["return_20d"])

    rows: list[dict] = []
    for feature in feature_cols:
        if feature not in work.columns:
            continue
        series = _safe_numeric(work[feature])
        frame = pd.DataFrame({"feature_value": series, "return_20d": work["return_20d"]}).dropna()
        if frame.empty:
            continue

        corr = float(frame["feature_value"].corr(frame["return_20d"])) if len(frame) >= 2 else 0.0

        try:
            frame["bucket"] = pd.qcut(frame["feature_value"], q=min(4, frame["feature_value"].nunique()), duplicates="drop")
            bucket_summary = frame.groupby("bucket", observed=False)["return_20d"].mean().reset_index()
            bucket_summary["bucket"] = bucket_summary["bucket"].astype(str)
        except Exception:
            bucket_summary = pd.DataFrame({"bucket": ["all"], "return_20d": [frame["return_20d"].mean()]})

        top_bucket = float(bucket_summary["return_20d"].max()) if not bucket_summary.empty else np.nan
        bottom_bucket = float(bucket_summary["return_20d"].min()) if not bucket_summary.empty else np.nan
        spread = top_bucket - bottom_bucket if np.isfinite(top_bucket) and np.isfinite(bottom_bucket) else np.nan

        rows.append(
            {
                "feature": feature,
                "count": int(len(frame)),
                "correlation_20d": round(corr, 4) if np.isfinite(corr) else 0.0,
                "top_bucket_return_20d": round(top_bucket, 4) if np.isfinite(top_bucket) else np.nan,
                "bottom_bucket_return_20d": round(bottom_bucket, 4) if np.isfinite(bottom_bucket) else np.nan,
                "spread_20d": round(spread, 4) if np.isfinite(spread) else np.nan,
                "bucket_summary": bucket_summary.to_dict(orient="records"),
            }
        )

    return pd.DataFrame(rows).sort_values(["correlation_20d", "spread_20d"], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame(columns=["feature", "count", "correlation_20d", "top_bucket_return_20d", "bottom_bucket_return_20d", "spread_20d", "bucket_summary"])
