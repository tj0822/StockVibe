from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _scale_to_0_100(series: pd.Series, low: float, high: float) -> pd.Series:
    if high <= low:
        return pd.Series([50.0] * len(series), index=series.index)
    scaled = ((series - low) / (high - low)) * 100.0
    return scaled.clip(0.0, 100.0)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_market_temperature(stock_master_df: pd.DataFrame) -> Dict:
    if stock_master_df is None or stock_master_df.empty:
        return {"temperature_score": 50.0, "market_state": "🌤 Neutral"}

    df = stock_master_df.copy()
    df["return_1m"] = _to_num(df.get("return_1m", pd.Series(dtype=float)))
    df["momentum_score"] = _to_num(df.get("momentum_score", pd.Series(dtype=float)))
    df["sector_power"] = _to_num(df.get("sector_power", pd.Series(dtype=float)))

    avg_return_raw = float(df["return_1m"].dropna().mean()) if df["return_1m"].notna().any() else 0.0
    avg_return_scaled = float(_scale_to_0_100(pd.Series([avg_return_raw]), low=-15.0, high=15.0).iloc[0])

    avg_momentum = float(df["momentum_score"].dropna().mean()) if df["momentum_score"].notna().any() else 50.0
    avg_momentum = max(0.0, min(100.0, avg_momentum))

    if "signal" in df.columns and len(df) > 0:
        buy_ratio = float((df["signal"] == "BUY").sum() / len(df)) * 100.0
    else:
        buy_ratio = 0.0

    avg_sector_power = float(df["sector_power"].dropna().mean()) if df["sector_power"].notna().any() else 50.0
    avg_sector_power = max(0.0, min(100.0, avg_sector_power))

    temperature_score = (
        0.3 * avg_return_scaled
        + 0.3 * avg_momentum
        + 0.2 * buy_ratio
        + 0.2 * avg_sector_power
    )
    temperature_score = float(max(0.0, min(100.0, temperature_score)))

    if temperature_score >= 80:
        state = "🔥 Overheated"
    elif temperature_score >= 60:
        state = "🌞 Strong Uptrend"
    elif temperature_score >= 40:
        state = "🌤 Neutral"
    elif temperature_score >= 20:
        state = "🌧 Weak"
    else:
        state = "❄ Risk-off"

    return {
        "temperature_score": temperature_score,
        "market_state": state,
        "avg_return_1m_scaled": avg_return_scaled,
        "avg_momentum_score": avg_momentum,
        "buy_signal_ratio": buy_ratio,
        "avg_sector_power": avg_sector_power,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def build_sector_heatmap(stock_master_df: pd.DataFrame) -> pd.DataFrame:
    if stock_master_df is None or stock_master_df.empty:
        return pd.DataFrame(columns=["sector", "sector_power", "avg_score", "avg_return", "stock_count"])

    df = stock_master_df.copy()
    df["final_score_adjusted"] = _to_num(df.get("final_score_adjusted", pd.Series(dtype=float)))
    df["return_1m"] = _to_num(df.get("return_1m", pd.Series(dtype=float)))
    df["momentum_score"] = _to_num(df.get("momentum_score", pd.Series(dtype=float)))
    df["sector_power"] = _to_num(df.get("sector_power", pd.Series(dtype=float)))

    out = (
        df.dropna(subset=["sector"])
        .groupby("sector", as_index=False)
        .agg(
            sector_power=("sector_power", "mean"),
            avg_score=("final_score_adjusted", "mean"),
            avg_return=("return_1m", "mean"),
            avg_momentum=("momentum_score", "mean"),
            stock_count=("code", "nunique"),
        )
        .sort_values("sector_power", ascending=False)
        .reset_index(drop=True)
    )

    return out[["sector", "sector_power", "avg_score", "avg_return", "stock_count", "avg_momentum"]]


@st.cache_data(ttl=3600, show_spinner=False)
def compute_sector_rotation_strength(sector_rank_df: pd.DataFrame) -> Dict:
    empty_ret = {
        "rotation_strength": 0.0,
        "rotation_level": "Stable",
        "top_rising_sectors": pd.DataFrame(columns=["sector", "previous_rank", "current_rank", "rank_change"]),
        "top_falling_sectors": pd.DataFrame(columns=["sector", "previous_rank", "current_rank", "rank_change"]),
    }
    if sector_rank_df is None or sector_rank_df.empty:
        return empty_ret

    df = sector_rank_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = _to_num(df.get("rank", pd.Series(dtype=float)))
    df = df.dropna(subset=["date", "sector", "rank"])
    if df.empty:
        return empty_ret

    dates = sorted(df["date"].unique().tolist())
    if len(dates) < 2:
        return empty_ret

    latest_date = dates[-1]
    prev_date = dates[-2]

    latest = df[df["date"] == latest_date][["sector", "rank"]].rename(columns={"rank": "current_rank"})
    prev = df[df["date"] == prev_date][["sector", "rank"]].rename(columns={"rank": "previous_rank"})
    merged = latest.merge(prev, on="sector", how="inner")
    if merged.empty:
        return empty_ret

    merged["rank_change"] = merged["previous_rank"] - merged["current_rank"]

    rotation_strength = float(merged["rank_change"].abs().mean())
    if rotation_strength > 5:
        level = "Strong sector rotation"
    elif rotation_strength >= 2:
        level = "Moderate"
    else:
        level = "Stable"

    top_rising = merged.sort_values("rank_change", ascending=False).head(5).copy()
    top_falling = merged.sort_values("rank_change", ascending=True).head(5).copy()

    return {
        "rotation_strength": rotation_strength,
        "rotation_level": level,
        "top_rising_sectors": top_rising[["sector", "previous_rank", "current_rank", "rank_change"]],
        "top_falling_sectors": top_falling[["sector", "previous_rank", "current_rank", "rank_change"]],
    }


@st.cache_data(ttl=3600, show_spinner=False)
def compute_market_concentration(marketcap_rank_df: pd.DataFrame) -> Dict:
    empty_ret = {
        "largest_sector": "-",
        "largest_sector_share": 0.0,
        "concentration_level": "Diversified",
        "sector_distribution": pd.DataFrame(columns=["sector", "count", "share_pct"]),
    }
    if marketcap_rank_df is None or marketcap_rank_df.empty:
        return empty_ret

    df = marketcap_rank_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = _to_num(df.get("rank", pd.Series(dtype=float)))
    df = df.dropna(subset=["date", "rank"])
    if df.empty:
        return empty_ret

    latest = df[df["date"] == df["date"].max()].sort_values("rank").head(10).copy()
    if latest.empty:
        return empty_ret

    dist = latest["sector"].fillna("Unknown").value_counts().rename_axis("sector").reset_index(name="count")
    total = max(int(dist["count"].sum()), 1)
    dist["share_pct"] = (dist["count"] / total) * 100.0
    dist = dist.sort_values(["share_pct", "sector"], ascending=[False, True]).reset_index(drop=True)

    largest_sector = str(dist.iloc[0]["sector"])
    largest_share = float(dist.iloc[0]["share_pct"])

    if largest_share > 50:
        level = "Highly concentrated"
    elif largest_share >= 30:
        level = "Moderate"
    else:
        level = "Diversified"

    return {
        "largest_sector": largest_sector,
        "largest_sector_share": largest_share,
        "concentration_level": level,
        "sector_distribution": dist,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_leaders(marketcap_rank_df: pd.DataFrame, stock_master_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rank", "code", "name", "sector", "market_cap", "final_score_adjusted", "signal", "sector_state"]

    if marketcap_rank_df is None or marketcap_rank_df.empty:
        return pd.DataFrame(columns=cols)

    mc = marketcap_rank_df.copy()
    mc["date"] = pd.to_datetime(mc["date"], errors="coerce")
    mc["rank"] = _to_num(mc.get("rank", pd.Series(dtype=float)))
    mc = mc.dropna(subset=["date", "rank"]).copy()
    if mc.empty:
        return pd.DataFrame(columns=cols)

    latest = mc[mc["date"] == mc["date"].max()].sort_values("rank").head(10).copy()

    sm = stock_master_df.copy() if stock_master_df is not None else pd.DataFrame()
    if not sm.empty:
        enrich_cols = [c for c in ["code", "final_score_adjusted", "signal", "sector_state"] if c in sm.columns]
        sm = sm[enrich_cols].copy()
        if "code" in sm.columns:
            sm["code"] = sm["code"].astype(str).str.zfill(6)
        latest["code"] = latest["code"].astype(str).str.zfill(6)
        latest = latest.merge(sm, on="code", how="left")

    for c in cols:
        if c not in latest.columns:
            latest[c] = pd.NA

    latest["final_score_adjusted"] = _to_num(latest["final_score_adjusted"])
    return latest[cols]
