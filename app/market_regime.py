from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from app.market_regime_rules import (
    classify_primary_regime,
    classify_secondary_regime,
    classify_volatility_regime,
    compute_regime_confidence,
    generate_regime_interpretation,
    map_regime_to_strategy_guidance,
)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _extract_index_series(kospi_index_df: pd.DataFrame) -> pd.DataFrame:
    if kospi_index_df is None or kospi_index_df.empty:
        return pd.DataFrame(columns=["date", "close"])

    df = kospi_index_df.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

    if "index" in df.columns:
        df["close"] = _to_num(df["index"])
    elif "close" in df.columns:
        df["close"] = _to_num(df["close"])
    else:
        return pd.DataFrame(columns=["date", "close"])

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df[["date", "close"]]


@st.cache_data(ttl=3600, show_spinner=False)
def compute_market_regime_features(
    stock_master_df: pd.DataFrame,
    kospi_index_df: pd.DataFrame,
    sector_rank_df: pd.DataFrame,
    marketcap_rank_df: pd.DataFrame,
) -> dict:
    idx = _extract_index_series(kospi_index_df)

    kospi_return_20d = 0.0
    kospi_return_60d = 0.0
    kospi_vol_20d = 0.0
    as_of_date = datetime.now().strftime("%Y-%m-%d")

    if not idx.empty:
        idx["ret_1d"] = idx["close"].pct_change()
        idx["ret_20d"] = idx["close"].pct_change(20)
        idx["ret_60d"] = idx["close"].pct_change(60)
        idx["vol_20d"] = idx["ret_1d"].rolling(20).std()

        latest = idx.iloc[-1]
        kospi_return_20d = float(latest.get("ret_20d", 0.0) or 0.0)
        kospi_return_60d = float(latest.get("ret_60d", 0.0) or 0.0)
        kospi_vol_20d = float(latest.get("vol_20d", 0.0) or 0.0)
        as_of_date = pd.to_datetime(latest["date"]).strftime("%Y-%m-%d")

    sm = stock_master_df.copy() if stock_master_df is not None else pd.DataFrame()

    above_ma20_ratio = 0.5
    above_ma60_ratio = 0.5
    buy_sell_ratio = 1.0
    leading_sector_count = 0

    if not sm.empty:
        if "close" in sm.columns and "ma20" in sm.columns:
            above_ma20_ratio = float((_to_num(sm["close"]) > _to_num(sm["ma20"])).mean())
        else:
            above_ma20_ratio = float((_to_num(sm.get("momentum_score", pd.Series(dtype=float))) >= 55).mean())

        if "close" in sm.columns and "ma60" in sm.columns:
            above_ma60_ratio = float((_to_num(sm["close"]) > _to_num(sm["ma60"])).mean())
        else:
            above_ma60_ratio = float((_to_num(sm.get("momentum_score", pd.Series(dtype=float))) >= 65).mean())

        if "signal" in sm.columns:
            buy_cnt = int((sm["signal"] == "BUY").sum())
            sell_cnt = int((sm["signal"] == "SELL").sum())
            buy_sell_ratio = float((buy_cnt + 1) / (sell_cnt + 1))

        if "sector_state" in sm.columns and "sector" in sm.columns:
            sec = sm[["sector", "sector_state"]].dropna(subset=["sector"]).drop_duplicates(subset=["sector"], keep="last")
            leading_sector_count = int((sec["sector_state"] == "LEADING").sum())

    rotation_strength = 0.0
    sr = sector_rank_df.copy() if sector_rank_df is not None else pd.DataFrame()
    if not sr.empty and {"date", "sector", "rank"}.issubset(sr.columns):
        sr["date"] = pd.to_datetime(sr["date"], errors="coerce")
        sr["rank"] = _to_num(sr["rank"])
        sr = sr.dropna(subset=["date", "sector", "rank"])

        dates = sorted(sr["date"].unique().tolist())
        if len(dates) >= 2:
            latest_date = dates[-1]
            prev_date = dates[-2]
            latest = sr[sr["date"] == latest_date][["sector", "rank"]].rename(columns={"rank": "current_rank"})
            prev = sr[sr["date"] == prev_date][["sector", "rank"]].rename(columns={"rank": "previous_rank"})
            merged = latest.merge(prev, on="sector", how="inner")
            if not merged.empty:
                merged["rank_change"] = merged["previous_rank"] - merged["current_rank"]
                rotation_strength = float(merged["rank_change"].abs().mean())

    market_temperature = 50.0

    top_sector_concentration = 0.0
    mc = marketcap_rank_df.copy() if marketcap_rank_df is not None else pd.DataFrame()
    if not mc.empty and {"date", "sector", "rank"}.issubset(mc.columns):
        mc["date"] = pd.to_datetime(mc["date"], errors="coerce")
        mc["rank"] = _to_num(mc["rank"])
        mc = mc.dropna(subset=["date", "rank"])
        if not mc.empty:
            latest = mc[mc["date"] == mc["date"].max()].sort_values("rank").head(10)
            if not latest.empty:
                counts = latest["sector"].fillna("Unknown").value_counts()
                top_sector_concentration = float(counts.iloc[0] / len(latest))

    features = {
        "kospi_return_20d": round(kospi_return_20d, 4),
        "kospi_return_60d": round(kospi_return_60d, 4),
        "kospi_vol_20d": round(kospi_vol_20d, 4),
        "above_ma20_ratio": round(above_ma20_ratio, 4),
        "above_ma60_ratio": round(above_ma60_ratio, 4),
        "buy_sell_ratio": round(buy_sell_ratio, 4),
        "leading_sector_count": int(leading_sector_count),
        "rotation_strength": round(rotation_strength, 4),
        "market_temperature": round(market_temperature, 2),
        "top_sector_concentration": round(top_sector_concentration, 4),
        "vol_threshold": 0.02,
        "as_of_date": as_of_date,
    }
    return features


@st.cache_data(ttl=3600, show_spinner=False)
def run_market_regime_engine(
    stock_master_df: pd.DataFrame,
    kospi_index_df: pd.DataFrame,
    sector_rank_df: pd.DataFrame,
    marketcap_rank_df: pd.DataFrame,
) -> dict:
    features = compute_market_regime_features(
        stock_master_df=stock_master_df,
        kospi_index_df=kospi_index_df,
        sector_rank_df=sector_rank_df,
        marketcap_rank_df=marketcap_rank_df,
    )

    primary_regime = classify_primary_regime(features)
    secondary_regime = classify_secondary_regime(features)
    volatility_regime = classify_volatility_regime(features)

    regimes = {
        "primary_regime": primary_regime,
        "secondary_regime": secondary_regime,
        "volatility_regime": volatility_regime,
    }

    confidence = compute_regime_confidence(features, regimes)
    interpretation = generate_regime_interpretation(regimes, features)
    strategy_guidance = map_regime_to_strategy_guidance(regimes)

    return {
        "as_of_date": features.get("as_of_date", datetime.now().strftime("%Y-%m-%d")),
        "primary_regime": primary_regime,
        "secondary_regime": secondary_regime,
        "volatility_regime": volatility_regime,
        "confidence": confidence,
        "features": features,
        "interpretation": interpretation,
        "strategy_guidance": strategy_guidance,
    }


def save_market_regime_snapshot(result: dict, path: str = "data/market_regime_history.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    history = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                history = raw
        except Exception:
            history = []

    as_of_date = str(result.get("as_of_date", datetime.now().strftime("%Y-%m-%d")))

    filtered = [row for row in history if str(row.get("as_of_date", "")) != as_of_date]
    filtered.append(result)
    filtered = sorted(filtered, key=lambda x: str(x.get("as_of_date", "")))

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(filtered, fh, ensure_ascii=False, indent=2)


def render_regime_summary(result: dict) -> None:
    st.markdown("#### 🧠 Market Regime")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Primary", result.get("primary_regime", "-"))
    with c2:
        st.metric("Secondary", result.get("secondary_regime", "-"))
    with c3:
        st.metric("Volatility", result.get("volatility_regime", "-"))
    with c4:
        st.metric("Confidence", f"{float(result.get('confidence', 0.0)):.1f}")

    interp = result.get("interpretation", {}) if isinstance(result, dict) else {}
    summary = str(interp.get("summary", "-"))
    risk_note = str(interp.get("risk_note", "-"))
    st.info(summary)
    st.caption(f"리스크 노트: {risk_note}")

    guidance = result.get("strategy_guidance", {}) if isinstance(result, dict) else {}
    rec = guidance.get("recommended", []) if isinstance(guidance, dict) else []
    avoid = guidance.get("avoid", []) if isinstance(guidance, dict) else []

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**추천 전략**")
        if rec:
            st.markdown("- " + "\n- ".join([str(x) for x in rec]))
        else:
            st.caption("없음")
    with g2:
        st.markdown("**비추천 전략**")
        if avoid:
            st.markdown("- " + "\n- ".join([str(x) for x in avoid]))
        else:
            st.caption("없음")
