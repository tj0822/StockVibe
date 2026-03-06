from __future__ import annotations

from typing import Dict

import pandas as pd


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _latest_by_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    if out.empty:
        return pd.DataFrame(columns=out.columns)
    return out[out[date_col] == out[date_col].max()].copy()


def get_current_market_snapshot(stock_master_df: pd.DataFrame, marketcap_rank_df: pd.DataFrame) -> Dict:
    """
    Return:
    - strongest_sector
    - leading_sector_count
    - most_represented_marketcap_sector
    - buy_count_leading_sectors
    """
    strongest_sector = "-"
    leading_sector_count = 0
    buy_count_leading = 0

    sm = stock_master_df.copy() if stock_master_df is not None else pd.DataFrame()
    if not sm.empty and "sector" in sm.columns:
        sec_cols = [c for c in ["sector", "sector_power", "sector_rank", "sector_state"] if c in sm.columns]
        sec_df = sm[sec_cols].dropna(subset=["sector"]).drop_duplicates(subset=["sector"], keep="last").copy()

        if not sec_df.empty:
            if "sector_rank" in sec_df.columns:
                sec_df["sector_rank"] = _safe_num(sec_df["sector_rank"])
                pick = sec_df.sort_values("sector_rank", ascending=True).iloc[0]
            else:
                sec_df["sector_power"] = _safe_num(sec_df.get("sector_power", pd.Series(dtype=float)))
                pick = sec_df.sort_values("sector_power", ascending=False).iloc[0]
            strongest_sector = str(pick.get("sector", "-"))

            if "sector_state" in sec_df.columns:
                leading_sector_count = int((sec_df["sector_state"] == "LEADING").sum())

        if "signal" in sm.columns and "sector_state" in sm.columns:
            buy_count_leading = int(((sm["sector_state"] == "LEADING") & (sm["signal"] == "BUY")).sum())

    mc_latest = _latest_by_date(marketcap_rank_df, "date")
    most_rep_sector = "-"
    if not mc_latest.empty and "sector" in mc_latest.columns:
        rep = (
            mc_latest.groupby("sector", as_index=False)
            .agg(count=("code", "nunique"))
            .sort_values(["count", "sector"], ascending=[False, True])
        )
        if not rep.empty:
            most_rep_sector = f"{rep.iloc[0]['sector']} ({int(rep.iloc[0]['count'])})"

    return {
        "strongest_sector": strongest_sector,
        "leading_sector_count": leading_sector_count,
        "most_represented_marketcap_sector": most_rep_sector,
        "buy_count_leading_sectors": buy_count_leading,
    }


def summarize_sector_rank_changes(sector_rank_df: pd.DataFrame) -> Dict:
    """
    Return latest risers/fallers vs previous date.
    """
    if sector_rank_df is None or sector_rank_df.empty:
        return {
            "latest_date": None,
            "prev_date": None,
            "risers": [],
            "fallers": [],
            "change_df": pd.DataFrame(columns=["sector", "latest_rank", "prev_rank", "rank_change"]),
            "rotation_intensity": 0.0,
        }

    df = sector_rank_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = _safe_num(df["rank"])
    df = df.dropna(subset=["date", "sector", "rank"]).copy()

    dates = sorted(df["date"].unique().tolist())
    if len(dates) < 2:
        latest = dates[-1] if dates else None
        return {
            "latest_date": latest,
            "prev_date": None,
            "risers": [],
            "fallers": [],
            "change_df": pd.DataFrame(columns=["sector", "latest_rank", "prev_rank", "rank_change"]),
            "rotation_intensity": 0.0,
        }

    latest_date = dates[-1]
    prev_date = dates[-2]

    latest_df = df[df["date"] == latest_date][["sector", "rank"]].rename(columns={"rank": "latest_rank"})
    prev_df = df[df["date"] == prev_date][["sector", "rank"]].rename(columns={"rank": "prev_rank"})

    merged = latest_df.merge(prev_df, on="sector", how="outer")
    merged["latest_rank"] = _safe_num(merged["latest_rank"])
    merged["prev_rank"] = _safe_num(merged["prev_rank"])
    merged["rank_change"] = merged["latest_rank"] - merged["prev_rank"]

    risers_df = (
        merged.dropna(subset=["latest_rank", "prev_rank", "rank_change"])
        .sort_values("rank_change", ascending=True)
        .head(5)
    )
    fallers_df = (
        merged.dropna(subset=["latest_rank", "prev_rank", "rank_change"])
        .sort_values("rank_change", ascending=False)
        .head(5)
    )

    rotation_intensity = float(merged["rank_change"].dropna().abs().mean()) if merged["rank_change"].notna().any() else 0.0

    return {
        "latest_date": latest_date,
        "prev_date": prev_date,
        "risers": risers_df.to_dict(orient="records"),
        "fallers": fallers_df.to_dict(orient="records"),
        "change_df": merged.sort_values("latest_rank", ascending=True),
        "rotation_intensity": rotation_intensity,
    }


def build_market_structure_watchlists(stock_master_df: pd.DataFrame, sector_rank_df: pd.DataFrame) -> Dict:
    """
    Returns:
    - leaders_df
    - rising_df
    - warning_df
    """
    sm = stock_master_df.copy() if stock_master_df is not None else pd.DataFrame()
    if sm.empty:
        cols = ["code", "name", "sector", "final_score_adjusted", "signal"]
        return {
            "leaders_df": pd.DataFrame(columns=cols),
            "rising_df": pd.DataFrame(columns=cols),
            "warning_df": pd.DataFrame(columns=cols),
        }

    for col in ["final_score_adjusted", "sector_rank", "sector_power", "return_1m"]:
        if col in sm.columns:
            sm[col] = _safe_num(sm[col])

    leaders_df = sm[
        (sm.get("sector_state", "") == "LEADING")
        & (sm.get("signal", "") == "BUY")
        & (sm.get("final_score_adjusted", 0) >= 75)
    ].copy()

    sector_change = summarize_sector_rank_changes(sector_rank_df).get("change_df", pd.DataFrame())
    rising_sectors = []
    if not sector_change.empty:
        # rank_change < 0 means rank improved (e.g. 8 -> 3)
        rising_sectors = (
            sector_change.dropna(subset=["rank_change"])
            .sort_values("rank_change", ascending=True)
            .query("rank_change <= -2")["sector"]
            .head(8)
            .tolist()
        )

    rising_df = sm[
        sm.get("sector", "").isin(rising_sectors)
        & (sm.get("final_score_adjusted", 0) >= 65)
    ].copy()

    warning_df = sm[
        (sm.get("sector_state", "") == "WEAK")
        & ((sm.get("signal", "") == "SELL") | (sm.get("final_score_adjusted", 0) < 60))
    ].copy()

    sort_cols = [c for c in ["final_score_adjusted", "sector_power", "return_1m"] if c in sm.columns]
    if sort_cols:
        leaders_df = leaders_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        rising_df = rising_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        warning_df = warning_df.sort_values(sort_cols, ascending=[True, False, False][: len(sort_cols)])

    cols = [c for c in ["code", "name", "sector", "final_score_adjusted", "signal", "sector_state", "sector_rank"] if c in sm.columns]
    return {
        "leaders_df": leaders_df[cols].head(20),
        "rising_df": rising_df[cols].head(20),
        "warning_df": warning_df[cols].head(20),
    }


def build_topn_sector_composition(rank_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    For latest date:
    - sector
    - count_in_top_n
    - avg_rank
    """
    latest_df = _latest_by_date(rank_history_df, "date")
    if latest_df.empty:
        return pd.DataFrame(columns=["sector", "count_in_top_n", "avg_rank"])

    latest_df["rank"] = _safe_num(latest_df["rank"])
    latest_df["sector"] = latest_df.get("sector", "Unknown").fillna("Unknown")

    return (
        latest_df.groupby("sector", as_index=False)
        .agg(count_in_top_n=("code", "nunique"), avg_rank=("rank", "mean"))
        .sort_values(["count_in_top_n", "avg_rank"], ascending=[False, True])
        .reset_index(drop=True)
    )
