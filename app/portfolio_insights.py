from __future__ import annotations

from typing import Any, Dict

import pandas as pd


DEFAULT_PORTFOLIO_THRESHOLDS = {
    "concentration_warning_pct": 35.0,
    "concentration_critical_pct": 50.0,
    "compliant_score_threshold": 70.0,
    "watch_score_threshold": 60.0,
    "weak_sector_warning_pct": 30.0,
}


def normalize_portfolio_thresholds(raw: Dict[str, Any] | None) -> Dict[str, float]:
    raw = raw or {}
    out = dict(DEFAULT_PORTFOLIO_THRESHOLDS)
    for key in out.keys():
        if key in raw:
            try:
                out[key] = float(raw[key])
            except Exception:
                pass
    return out


def classify_strategy_compliance(signal: Any, adjusted_score: Any, thresholds: Dict[str, float] | None = None) -> str:
    t = normalize_portfolio_thresholds(thresholds)
    score = float(pd.to_numeric(adjusted_score, errors="coerce") if pd.notna(adjusted_score) else 0.0)
    signal_str = str(signal).upper() if signal is not None else ""

    if signal_str == "SELL" or score < t["watch_score_threshold"]:
        return "NON_COMPLIANT"

    if signal_str == "BUY" and score >= t["compliant_score_threshold"]:
        return "COMPLIANT"

    if score >= t["watch_score_threshold"]:
        return "WATCH"

    return "NON_COMPLIANT"


def merge_holdings_with_context(
    holding_summary_df: pd.DataFrame,
    stock_master_df: pd.DataFrame,
    thresholds: Dict[str, float] | None = None,
) -> pd.DataFrame:
    if holding_summary_df is None or holding_summary_df.empty:
        return pd.DataFrame()

    out = holding_summary_df.copy()
    out["종목코드"] = out["종목코드"].astype(str).str.zfill(6)

    if stock_master_df is not None and not stock_master_df.empty and "code" in stock_master_df.columns:
        keep = [
            "code",
            "name",
            "sector",
            "sector_power",
            "sector_state",
            "final_score_adjusted",
            "signal",
        ]
        keep = [c for c in keep if c in stock_master_df.columns]
        if keep:
            ctx = stock_master_df[keep].copy()
            ctx["code"] = ctx["code"].astype(str).str.zfill(6)
            ctx = ctx.sort_values("code").drop_duplicates(subset=["code"], keep="last")
            ctx = ctx.rename(columns={"code": "종목코드", "name": "컨텍스트종목명"})
            out = out.merge(ctx, on="종목코드", how="left")

    out["sector"] = out.get("sector", pd.Series(index=out.index, dtype=object)).fillna("미분류")
    out["sector_state"] = out.get("sector_state", pd.Series(index=out.index, dtype=object)).fillna("UNKNOWN")
    out["signal"] = out.get("signal", pd.Series(index=out.index, dtype=object)).fillna("")
    out["final_score_adjusted"] = pd.to_numeric(out.get("final_score_adjusted"), errors="coerce").fillna(0.0)
    out["sector_power"] = pd.to_numeric(out.get("sector_power"), errors="coerce")

    price = pd.to_numeric(out.get("현재가격"), errors="coerce")
    qty = pd.to_numeric(out.get("보유수량"), errors="coerce").fillna(0.0)
    out["market_value"] = (price.fillna(0.0) * qty).fillna(0.0)

    out["strategy_compliance"] = out.apply(
        lambda row: classify_strategy_compliance(row.get("signal"), row.get("final_score_adjusted"), thresholds),
        axis=1,
    )

    return out


def compute_sector_exposure(holdings_df: pd.DataFrame) -> pd.DataFrame:
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(columns=["sector", "market_value", "sector_weight", "sector_state", "sector_power_mean"])

    def _first_non_null(series: pd.Series):
        valid = series.dropna()
        return valid.iloc[0] if not valid.empty else None

    sec = (
        holdings_df.groupby("sector", as_index=False)
        .agg(
            market_value=("market_value", "sum"),
            sector_state=("sector_state", _first_non_null),
            sector_power_mean=("sector_power", "mean"),
        )
        .sort_values("market_value", ascending=False)
        .reset_index(drop=True)
    )

    total = float(sec["market_value"].sum())
    sec["sector_weight"] = (sec["market_value"] / total * 100.0).clip(0, 100) if total > 0 else 0.0
    return sec


def build_rebalance_suggestions(
    holdings_df: pd.DataFrame,
    stock_master_df: pd.DataFrame,
    sector_exposure_df: pd.DataFrame,
    thresholds: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Return suggestions with:
    - code
    - name
    - suggestion_type ("REDUCE", "REVIEW", "HOLD", "ADD_CANDIDATE")
    - reason
    """
    t = normalize_portfolio_thresholds(thresholds)

    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(columns=["code", "name", "suggestion_type", "reason"])

    out_rows = []
    sec_map = {}
    if sector_exposure_df is not None and not sector_exposure_df.empty:
        sec_map = sector_exposure_df.set_index("sector")["sector_weight"].to_dict()

    for _, row in holdings_df.iterrows():
        code = str(row.get("종목코드", "")).zfill(6)
        name = row.get("종목명", "")
        sector = row.get("sector", "미분류")
        sector_weight = float(sec_map.get(sector, 0.0))
        state = str(row.get("sector_state", "UNKNOWN"))
        compliance = str(row.get("strategy_compliance", "WATCH"))
        score = float(pd.to_numeric(row.get("final_score_adjusted"), errors="coerce") or 0.0)

        if sector_weight > t["concentration_warning_pct"] and compliance == "NON_COMPLIANT":
            out_rows.append(
                {
                    "code": code,
                    "name": name,
                    "suggestion_type": "REDUCE",
                    "reason": f"과대집중 섹터({sector_weight:.1f}%) + NON_COMPLIANT",
                }
            )
            continue

        if state == "WEAK" and score < t["watch_score_threshold"]:
            out_rows.append(
                {
                    "code": code,
                    "name": name,
                    "suggestion_type": "REVIEW",
                    "reason": "WEAK 섹터 + 낮은 점수",
                }
            )
            continue

        if compliance == "COMPLIANT" and state == "LEADING":
            out_rows.append(
                {
                    "code": code,
                    "name": name,
                    "suggestion_type": "HOLD",
                    "reason": "LEADING 섹터 + 전략 적합",
                }
            )

    held_codes = set(holdings_df["종목코드"].astype(str).str.zfill(6).tolist())
    if stock_master_df is not None and not stock_master_df.empty:
        candidates = stock_master_df.copy()
        candidates["code"] = candidates["code"].astype(str).str.zfill(6)
        candidates = candidates[
            candidates["signal"].astype(str).eq("BUY")
            & candidates["sector_state"].astype(str).eq("LEADING")
            & (~candidates["code"].isin(held_codes))
        ].copy()

        sort_cols = [c for c in ["final_score_adjusted", "sector_power"] if c in candidates.columns]
        if sort_cols:
            candidates = candidates.sort_values(sort_cols, ascending=False)

        for _, row in candidates.head(5).iterrows():
            out_rows.append(
                {
                    "code": str(row.get("code", "")),
                    "name": row.get("name", ""),
                    "suggestion_type": "ADD_CANDIDATE",
                    "reason": "LEADING 섹터 BUY 시그널 신규 후보",
                }
            )

    if not out_rows:
        return pd.DataFrame(columns=["code", "name", "suggestion_type", "reason"])

    priority = {"REDUCE": 0, "REVIEW": 1, "HOLD": 2, "ADD_CANDIDATE": 3}
    out = pd.DataFrame(out_rows)
    out["_priority"] = out["suggestion_type"].map(priority).fillna(9)
    out = out.sort_values(["_priority", "code"]).drop(columns=["_priority"]).reset_index(drop=True)
    return out


def build_trade_sector_summary(
    sell_trades: list[dict],
    stock_master_df: pd.DataFrame,
    thresholds: Dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not sell_trades:
        empty_cols = ["sector", "realized_pnl", "win_rate", "trades"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame()

    trades_df = pd.DataFrame(sell_trades).copy()
    trades_df["code"] = trades_df["code"].astype(str).str.zfill(6)
    trades_df["pnl"] = pd.to_numeric(trades_df.get("pnl"), errors="coerce").fillna(0.0)
    trades_df["return_pct"] = pd.to_numeric(trades_df.get("return_pct"), errors="coerce").fillna(0.0)

    if stock_master_df is not None and not stock_master_df.empty and "code" in stock_master_df.columns:
        keep = ["code", "sector", "sector_state", "signal", "final_score_adjusted"]
        keep = [c for c in keep if c in stock_master_df.columns]
        ctx = stock_master_df[keep].copy()
        ctx["code"] = ctx["code"].astype(str).str.zfill(6)
        ctx = ctx.sort_values("code").drop_duplicates(subset=["code"], keep="last")
        trades_df = trades_df.merge(ctx, on="code", how="left")

    trades_df["sector"] = trades_df.get("sector", pd.Series(index=trades_df.index, dtype=object)).fillna("미분류")
    trades_df["compliance_now"] = trades_df.apply(
        lambda row: classify_strategy_compliance(row.get("signal"), row.get("final_score_adjusted"), thresholds),
        axis=1,
    )
    trades_df["is_win"] = (trades_df["pnl"] > 0).astype(int)

    summary = (
        trades_df.groupby("sector", as_index=False)
        .agg(
            realized_pnl=("pnl", "sum"),
            win_rate=("is_win", "mean"),
            trades=("code", "count"),
        )
        .sort_values("realized_pnl", ascending=False)
        .reset_index(drop=True)
    )
    summary["win_rate"] = summary["win_rate"] * 100.0

    compliance_summary = (
        trades_df.groupby("compliance_now", as_index=False)
        .agg(trades=("code", "count"), realized_pnl=("pnl", "sum"))
        .sort_values("trades", ascending=False)
        .reset_index(drop=True)
    )

    return summary, compliance_summary
