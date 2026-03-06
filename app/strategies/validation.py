from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from app.data import load_kospi_index, load_kospi_list, load_stock_data
from app.strategies.market_regime import classify_market_regime
from app.strategies.registry import get_strategy


DEFAULT_THRESHOLDS = {
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


REGIME_ORDER = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]


def _normalize_thresholds(raw_thresholds: dict | None) -> Dict[str, Any]:
    raw_thresholds = raw_thresholds or {}

    # Backward compatibility: accept flat dict schema.
    if "overall" not in raw_thresholds and any(
        key in raw_thresholds for key in ["min_cagr", "max_mdd", "min_win_rate", "min_trades"]
    ):
        overall = {**DEFAULT_THRESHOLDS["overall"], **raw_thresholds}
        by_regime = deepcopy(DEFAULT_THRESHOLDS["by_regime"])
        return {"overall": overall, "by_regime": by_regime}

    overall = {**DEFAULT_THRESHOLDS["overall"], **raw_thresholds.get("overall", {})}

    by_regime = deepcopy(DEFAULT_THRESHOLDS["by_regime"])
    user_by_regime = raw_thresholds.get("by_regime", {})
    for regime in REGIME_ORDER:
        by_regime[regime] = {**by_regime.get(regime, {}), **user_by_regime.get(regime, {})}

    return {
        "overall": overall,
        "by_regime": by_regime,
    }


def _compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
    if equity_df is None or equity_df.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 1.0,
            "trades_count": 0,
            "avg_return": 0.0,
            "profit_loss_ratio": 0.0,
            "sharpe": 0.0,
            "avg_holding_days": 0.0,
        }

    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
    eq = eq.dropna(subset=["date", "equity"]).sort_values("date")

    if eq.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 1.0,
            "trades_count": 0,
            "avg_return": 0.0,
            "profit_loss_ratio": 0.0,
            "sharpe": 0.0,
            "avg_holding_days": 0.0,
        }

    initial_equity = float(eq["equity"].iloc[0]) if len(eq) else 0.0
    final_equity = float(eq["equity"].iloc[-1]) if len(eq) else 0.0
    total_return = ((final_equity / initial_equity) - 1.0) if initial_equity > 0 else 0.0

    years = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days / 365.25, 1e-9)
    cagr = ((final_equity / initial_equity) ** (1 / years) - 1.0) if initial_equity > 0 else 0.0

    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax.replace(0, np.nan)
    max_drawdown = abs(float(drawdown.min())) if not drawdown.empty else 1.0

    daily_ret = eq["equity"].pct_change().dropna()
    sharpe = 0.0
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))

    sell_trades = trades_df[trades_df["action"] == "SELL"].copy() if not trades_df.empty else pd.DataFrame()
    if not sell_trades.empty and "return_pct" in sell_trades.columns:
        valid = sell_trades.dropna(subset=["return_pct"]).copy()
        trades_count = int(len(valid))
        win_rate = float((valid["return_pct"] > 0).mean()) if trades_count > 0 else 0.0
        avg_return = float(valid["return_pct"].mean()) if trades_count > 0 else 0.0
        avg_win = float(valid.loc[valid["return_pct"] > 0, "return_pct"].mean()) if (valid["return_pct"] > 0).any() else 0.0
        avg_loss = float(valid.loc[valid["return_pct"] < 0, "return_pct"].mean()) if (valid["return_pct"] < 0).any() else 0.0
        profit_loss_ratio = float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    else:
        trades_count = 0
        win_rate = 0.0
        avg_return = 0.0
        profit_loss_ratio = 0.0

    avg_holding_days = _compute_avg_holding_days(trades_df if trades_df is not None else pd.DataFrame())

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
        "trades_count": int(trades_count),
        "avg_return": float(avg_return),
        "profit_loss_ratio": float(profit_loss_ratio),
        "sharpe": float(sharpe),
        "avg_holding_days": float(avg_holding_days),
    }


def _passes_overall(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> bool:
    return (
        float(metrics.get("cagr", 0.0)) >= float(thresholds.get("min_cagr", 0.0))
        and float(metrics.get("max_drawdown", 1.0)) <= float(thresholds.get("max_mdd", 1.0))
        and float(metrics.get("win_rate", 0.0)) >= float(thresholds.get("min_win_rate", 0.0))
        and int(metrics.get("trades_count", 0)) >= int(thresholds.get("min_trades", 0))
    )


def _passes_regime(metrics: Dict[str, Any], regime_thresholds: Dict[str, Any]) -> bool:
    if not regime_thresholds:
        return True

    checks = []
    if "min_cagr" in regime_thresholds:
        checks.append(float(metrics.get("cagr", 0.0)) >= float(regime_thresholds["min_cagr"]))
    if "max_mdd" in regime_thresholds:
        checks.append(float(metrics.get("max_drawdown", 1.0)) <= float(regime_thresholds["max_mdd"]))
    if "min_win_rate" in regime_thresholds:
        checks.append(float(metrics.get("win_rate", 0.0)) >= float(regime_thresholds["min_win_rate"]))
    if "min_trades" in regime_thresholds:
        checks.append(int(metrics.get("trades_count", 0)) >= int(regime_thresholds["min_trades"]))

    return all(checks) if checks else True


def _compute_avg_holding_days(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or "action" not in trades_df.columns:
        return 0.0

    tdf = trades_df.copy()
    tdf["date"] = pd.to_datetime(tdf["date"], errors="coerce")
    tdf = tdf.dropna(subset=["date", "code", "action"]).sort_values(["code", "date"]) 

    holding_days = []
    last_buy_date: Dict[str, pd.Timestamp] = {}

    for _, row in tdf.iterrows():
        code = str(row["code"])
        action = str(row["action"]).upper()
        dt = row["date"]
        if action == "BUY":
            last_buy_date[code] = dt
        elif action == "SELL" and code in last_buy_date:
            holding_days.append((dt - last_buy_date[code]).days)
            del last_buy_date[code]

    if not holding_days:
        return 0.0
    return float(np.mean(holding_days))


@st.cache_data(ttl=1800, show_spinner=False)
def _validate_strategy_cached(
    strategy_id: str,
    params_json: str,
    universe: str,
    start_date: str,
    end_date: str,
    thresholds_json: str,
) -> Dict[str, Any]:
    params = json.loads(params_json) if params_json else {}
    thresholds = _normalize_thresholds(json.loads(thresholds_json) if thresholds_json else {})

    strategy = get_strategy(strategy_id)

    price_df = load_stock_data("data")
    kospi_list = load_kospi_list("data")
    kospi_index = load_kospi_index("data")

    if price_df.empty:
        return {
            "validated": False,
            "passed": False,
            "reason": "price_df empty",
            "overall": {},
            "by_regime": {k: {} for k in REGIME_ORDER},
            "thresholds": thresholds,
        }

    work = price_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]) 

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    work = work[(work["date"] >= start_dt) & (work["date"] <= end_dt)]

    if universe == "KOSPI200" and not kospi_list.empty:
        codes = kospi_list["code"].astype(str).str.zfill(6).drop_duplicates().head(200).tolist()
        work_codes = work["code"].astype(str).str.zfill(6)
        work = work[work_codes.isin(codes)]

    if work.empty:
        return {
            "validated": False,
            "passed": False,
            "reason": "filtered universe/date data empty",
            "overall": {},
            "by_regime": {k: {} for k in REGIME_ORDER},
            "thresholds": thresholds,
        }

    signals_df = strategy.generate_signals(
        price_df=work,
        volume_df=work[["date", "code", "volume"]].copy() if "volume" in work.columns else pd.DataFrame(),
        market_df=kospi_index,
        params=params,
    )

    if signals_df.empty:
        return {
            "validated": False,
            "passed": False,
            "reason": "no signals generated",
            "overall": {
                "trades_count": 0,
            },
            "by_regime": {k: {} for k in REGIME_ORDER},
            "thresholds": thresholds,
        }

    from app.ui import run_turnover_strategy_backtest

    equity_df, trades_df = run_turnover_strategy_backtest(
        price_df=work,
        signal_df=signals_df,
        kospi_index=kospi_index,
        start_date=start_dt,
        top_n=int(params.get("top_n", 2)),
        initial_cash=float(params.get("initial_cash", 50_000_000)),
        max_daily_buys=int(params.get("max_daily_buys", 2)),
        buy_unit=float(params.get("buy_unit", 2_000_000)),
        add_buy_threshold_pct=float(params.get("add_buy_threshold_pct", -7.0)),
    )

    overall_metrics = _compute_metrics(equity_df, trades_df)

    regime_df = classify_market_regime(kospi_index)
    by_regime: Dict[str, Dict[str, Any]] = {}
    regime_checks: list[bool] = []

    if not regime_df.empty and not equity_df.empty:
        eq = equity_df.copy()
        eq["date"] = pd.to_datetime(eq["date"], errors="coerce").dt.normalize()
        eq = eq.dropna(subset=["date"]).copy()

        regime_map = regime_df[["date", "regime"]].copy()
        regime_map["date"] = pd.to_datetime(regime_map["date"], errors="coerce").dt.normalize()
        regime_map = regime_map.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")

        eq = eq.merge(regime_map, on="date", how="left")
        if trades_df is not None and not trades_df.empty:
            tdf = trades_df.copy()
            tdf["date"] = pd.to_datetime(tdf["date"], errors="coerce").dt.normalize()
            tdf = tdf.dropna(subset=["date"]).merge(regime_map, on="date", how="left")
        else:
            tdf = pd.DataFrame(columns=["date", "action", "return_pct", "code", "regime"])

        for regime in REGIME_ORDER:
            eq_reg = eq[eq["regime"] == regime].copy()
            tdf_reg = tdf[tdf["regime"] == regime].copy()
            regime_metrics = _compute_metrics(eq_reg, tdf_reg)
            by_regime[regime] = regime_metrics
            regime_checks.append(_passes_regime(regime_metrics, thresholds.get("by_regime", {}).get(regime, {})))
    else:
        for regime in REGIME_ORDER:
            by_regime[regime] = _compute_metrics(pd.DataFrame(), pd.DataFrame())
            regime_checks.append(_passes_regime(by_regime[regime], thresholds.get("by_regime", {}).get(regime, {})))

    overall_passed = _passes_overall(overall_metrics, thresholds.get("overall", {}))
    passed = bool(overall_passed and all(regime_checks))

    return {
        "strategy_id": strategy_id,
        "strategy_version": strategy.spec.version,
        "validated": bool(passed),
        "passed": bool(passed),
        "run_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "universe": universe,
        "date_range": f"{start_date} ~ {end_date}",
        "overall": overall_metrics,
        "metrics": overall_metrics,  # backward compatibility
        "by_regime": by_regime,
        "thresholds": thresholds,
    }


def validate_strategy(
    strategy_id: str,
    params: dict,
    universe: str,
    start_date: str,
    end_date: str,
    thresholds: dict,
) -> Dict[str, Any]:
    params_json = json.dumps(params or {}, sort_keys=True, default=str)
    thresholds_json = json.dumps(thresholds or {}, sort_keys=True, default=str)

    cache_key = {
        "strategy_id": strategy_id,
        "params": params_json,
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "thresholds": thresholds_json,
    }
    _ = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()

    return _validate_strategy_cached(
        strategy_id=strategy_id,
        params_json=params_json,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        thresholds_json=thresholds_json,
    )
