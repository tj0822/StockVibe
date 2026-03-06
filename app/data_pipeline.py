import json
import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from financial_utils import StockScoringEngine

from .data import load_finance_data, load_kospi_index, load_kospi_list, load_stock_data
from .sector_context import compute_sector_power
from .signals import build_signals
from .strategies.registry import get_strategy
from .strategies.registry_store import get_production_strategy_ids


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _load_sector_map_df(data_dir: str) -> pd.DataFrame:
    pkl_path = os.path.join(data_dir, "kospi_sector_info.pkl")
    json_path = os.path.join(data_dir, "kospi_sector_info.json")

    sector_df = pd.DataFrame(columns=["code", "name", "sector"])
    if os.path.exists(pkl_path):
        try:
            loaded = pd.read_pickle(pkl_path)
            if isinstance(loaded, pd.DataFrame):
                sector_df = loaded.copy()
        except Exception:
            pass
    elif os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                sector_df = pd.DataFrame(raw)
        except Exception:
            pass

    if sector_df.empty:
        return pd.DataFrame(columns=["code", "sector"])

    sector_df = sector_df.copy()
    if "code" not in sector_df.columns:
        return pd.DataFrame(columns=["code", "sector"])

    if "sector" not in sector_df.columns:
        sector_df["sector"] = None

    sector_df["code"] = sector_df["code"].astype(str).str.zfill(6)
    sector_df["sector"] = sector_df["sector"].astype(str).str.strip()
    sector_df.loc[sector_df["sector"] == "", "sector"] = None
    sector_df = sector_df.drop_duplicates(subset=["code"], keep="last")
    return sector_df[["code", "sector"]]


def _build_primary_signal_latest(price_df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    signals = pd.DataFrame()

    try:
        production_ids = get_production_strategy_ids()
        if production_ids:
            primary_id = sorted(production_ids)[0]
            strategy = get_strategy(primary_id)
            params = dict(strategy.spec.default_params)

            volume_cols = [c for c in ["date", "code", "volume"] if c in price_df.columns]
            volume_df = price_df[volume_cols].copy() if volume_cols else pd.DataFrame()
            market_df = load_kospi_index(data_dir)

            signals = strategy.generate_signals(
                price_df=price_df,
                volume_df=volume_df,
                market_df=market_df,
                params=params,
            )
    except Exception:
        signals = pd.DataFrame()

    if signals is None or signals.empty:
        fallback = build_signals(
            price_df,
            10,
            3.0,
            20,
            5.0,
            20,
            2.0,
            20,
            2.0,
            ["Turnover Spike"],
            "ANY",
        )
        signals = fallback[["date", "code", "signal"]].copy() if not fallback.empty else pd.DataFrame()

    if signals is None or signals.empty:
        return pd.DataFrame(columns=["code", "signal"])

    sig = signals.copy()
    if "date" in sig.columns:
        sig["date"] = pd.to_datetime(sig["date"], errors="coerce")
    else:
        sig["date"] = pd.NaT

    sig["code"] = sig["code"].astype(str).str.zfill(6)
    sig = sig[sig.get("signal", pd.Series(dtype=str)).isin(["BUY", "SELL"])].copy()
    if sig.empty:
        return pd.DataFrame(columns=["code", "signal"])

    sig = sig.dropna(subset=["date"]).sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)
    return sig[["code", "signal"]]


def _compute_base_score_from_finance(latest_df: pd.DataFrame, finance_df: pd.DataFrame) -> pd.Series:
    engine = StockScoringEngine()

    if finance_df is None or finance_df.empty:
        return pd.Series(50.0, index=latest_df.index, dtype=float)

    fdf = finance_df.copy()
    fdf["code"] = fdf["code"].astype(str).str.zfill(6)
    if "date" in fdf.columns:
        fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
        fdf = fdf.dropna(subset=["date"]).sort_values(["code", "date"]) 
    latest_fin = fdf.groupby("code", as_index=False).tail(1)

    fin_map = latest_fin.set_index("code").to_dict(orient="index")

    scores: list[float] = []
    for _, row in latest_df.iterrows():
        code = str(row["code"]).zfill(6)
        fin = fin_map.get(code)
        if not fin:
            scores.append(50.0)
            continue

        financial_dict: dict[str, Any] = {}
        for key in ["roe", "debt_ratio", "operating_margin", "free_cash_flow"]:
            val = fin.get(key)
            if val is not None and pd.notna(val):
                financial_dict[key] = float(val)

        per = fin.get("per")
        pbr = fin.get("pbr")
        current_per = float(per) if per is not None and pd.notna(per) else None
        current_pbr = float(pbr) if pbr is not None and pd.notna(pbr) else None

        if (not financial_dict) and (current_per is None or current_pbr is None):
            scores.append(50.0)
            continue

        result = engine.calculate_final_score(
            financial_dict=financial_dict if financial_dict else None,
            current_per=current_per,
            industry_per=current_per if current_per is not None else None,
            current_pbr=current_pbr,
            industry_pbr=current_pbr if current_pbr is not None else None,
            financial_history=None,
            price_df=None,
            industry_tailwind_score=None,
            verbose=False,
        )
        score = float(result.get("final_score", 50)) if isinstance(result, dict) else 50.0
        scores.append(max(0.0, min(100.0, score)))

    return pd.Series(scores, index=latest_df.index, dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def build_stock_master_df(data_dir: str = "data") -> pd.DataFrame:
    """
    Returns stock_master_df with required columns:
    - code
    - name
    - sector
    - close
    - final_score
    - final_score_adjusted
    - momentum_score
    - signal
    - return_1m
    - sector_power
    - sector_rank
    - sector_state
    """
    price_df = load_stock_data(data_dir).copy()
    if price_df.empty:
        return pd.DataFrame(
            columns=[
                "code",
                "name",
                "sector",
                "close",
                "final_score",
                "final_score_adjusted",
                "momentum_score",
                "signal",
                "return_1m",
                "sector_power",
                "sector_rank",
                "sector_state",
            ]
        )

    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    price_df["code"] = price_df["code"].astype(str).str.zfill(6)
    price_df = price_df.dropna(subset=["date", "code", "close"]).sort_values(["code", "date"]).copy()

    price_df["return_1m"] = price_df.groupby("code")["close"].pct_change(20)

    if "ma20" not in price_df.columns:
        price_df["ma20"] = price_df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    if "ma60" not in price_df.columns:
        price_df["ma60"] = price_df.groupby("code")["close"].transform(lambda x: x.rolling(60).mean())
    if "rsi" not in price_df.columns:
        price_df["rsi"] = price_df.groupby("code")["close"].transform(_compute_rsi)

    rsi = pd.to_numeric(price_df["rsi"], errors="coerce")
    rsi_score = np.select(
        [
            rsi > 70,
            (rsi > 55) & (rsi <= 70),
            (rsi >= 45) & (rsi <= 55),
            (rsi > 30) & (rsi < 45),
            rsi <= 30,
        ],
        [100, 70, 50, 30, 10],
        default=50,
    )

    ma_score = (
        np.where(pd.to_numeric(price_df["close"], errors="coerce") > pd.to_numeric(price_df["ma20"], errors="coerce"), 20, 0)
        + np.where(pd.to_numeric(price_df["close"], errors="coerce") > pd.to_numeric(price_df["ma60"], errors="coerce"), 20, 0)
    )

    momentum_score = (rsi_score * 0.6) + ma_score
    price_df["momentum_score"] = pd.Series(momentum_score, index=price_df.index).clip(0, 100).astype(float)

    names_df = load_kospi_list(data_dir).copy()
    names_df["code"] = names_df["code"].astype(str).str.zfill(6)
    names_df = names_df.sort_values(["code"]).drop_duplicates(subset=["code"], keep="last")

    latest_df = (
        price_df.sort_values(["code", "date"])
        .groupby("code", as_index=False)
        .tail(1)
        .copy()
    )

    latest_df = latest_df.merge(names_df[["code", "name"]], on="code", how="left", suffixes=("", "_name"))
    if "name_name" in latest_df.columns:
        latest_df["name"] = latest_df["name"].fillna(latest_df["name_name"])

    finance_df = load_finance_data(data_dir)
    latest_df["base_score"] = _compute_base_score_from_finance(latest_df, finance_df)

    latest_df["final_score"] = (
        (latest_df["base_score"] * 0.7) + (latest_df["momentum_score"] * 0.3)
    ).clip(0, 100).astype(float)

    signal_df = _build_primary_signal_latest(price_df, data_dir)
    if not signal_df.empty:
        signal_df = signal_df.sort_values(["code"]).drop_duplicates(subset=["code"], keep="last")
    latest_df = latest_df.merge(signal_df, on="code", how="left")
    latest_df["signal"] = latest_df["signal"].where(latest_df["signal"].isin(["BUY", "SELL"]), None)

    sector_map_df = _load_sector_map_df(data_dir)
    latest_df = latest_df.merge(sector_map_df, on="code", how="left")

    stock_master_df = latest_df[[
        "code",
        "name",
        "sector",
        "close",
        "final_score",
        "momentum_score",
        "signal",
        "return_1m",
    ]].copy()

    sector_summary_df = compute_sector_power(stock_master_df)
    if not sector_summary_df.empty:
        stock_master_df = stock_master_df.merge(
            sector_summary_df,
            on="sector",
            how="left",
        )
    else:
        stock_master_df["sector_power"] = np.nan
        stock_master_df["sector_rank"] = np.nan
        stock_master_df["sector_state"] = None

    sector_bonus = {
        "LEADING": 8,
        "STRONG": 4,
        "ROTATION": 0,
        "WEAK": -4,
    }
    stock_master_df["final_score_adjusted"] = (
        pd.to_numeric(stock_master_df["final_score"], errors="coerce").fillna(0.0)
        + stock_master_df["sector_state"].map(sector_bonus).fillna(0.0)
    ).clip(0, 100)

    if "sector_rank" in stock_master_df.columns:
        stock_master_df["sector_rank"] = pd.to_numeric(stock_master_df["sector_rank"], errors="coerce").astype("Int64")

    stock_master_df = (
        stock_master_df
        .sort_values(["code", "final_score_adjusted", "final_score", "momentum_score"], ascending=[True, False, False, False])
        .drop_duplicates(subset=["code"], keep="first")
        .reset_index(drop=True)
    )
    return stock_master_df
