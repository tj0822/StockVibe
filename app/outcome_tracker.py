from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _normalize_code(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return text.zfill(6) if text.isdigit() else text


def _normalize_date(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return datetime.now().strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")


def _build_decision_id(decision_row: dict) -> str:
    parts = [
        _normalize_code(decision_row.get("code")),
        _normalize_date(decision_row.get("decision_date")),
        str(decision_row.get("source", "decision_orchestrator")),
        str(decision_row.get("signal", "BUY")),
        str(decision_row.get("triggered_strategy", "")),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]


def log_decision(decision_row: dict, path="data/decision_history.json"):
    """Append a new decision record."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = dict(decision_row or {})
    row["code"] = _normalize_code(row.get("code"))
    row["decision_date"] = _normalize_date(row.get("decision_date"))
    row.setdefault("source", "decision_orchestrator")
    row.setdefault("signal", "BUY")
    row.setdefault("logged_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    row.setdefault("decision_id", _build_decision_id(row))

    history: list[dict] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                history = raw
        except Exception:
            history = []

    existing_ids = {str(item.get("decision_id", "")) for item in history if isinstance(item, dict)}
    if row["decision_id"] not in existing_ids:
        history.append(row)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)


def load_decision_history(path="data/decision_history.json") -> pd.DataFrame:
    """Load decision history."""
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            return pd.DataFrame()
        df = pd.DataFrame(raw)
        if df.empty:
            return df
        if "code" in df.columns:
            df["code"] = df["code"].map(_normalize_code)
        if "decision_date" in df.columns:
            df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def update_decision_outcomes(
    decision_df: pd.DataFrame,
    price_df: pd.DataFrame,
    holding_periods=(5, 20, 60),
) -> pd.DataFrame:
    """
    Add:
    - return_5d
    - return_20d
    - return_60d
    - hit_5d
    - hit_20d
    - hit_60d
    - max_drawdown_20d
    """
    if decision_df is None or decision_df.empty:
        return pd.DataFrame()

    out = decision_df.copy()
    if "code" not in out.columns or "decision_date" not in out.columns:
        return out

    out["code"] = out["code"].map(_normalize_code)
    out["decision_date"] = pd.to_datetime(out["decision_date"], errors="coerce").dt.normalize()

    px = price_df.copy() if price_df is not None else pd.DataFrame()
    if px.empty or not {"code", "date", "close"}.issubset(px.columns):
        return out

    px["code"] = px["code"].map(_normalize_code)
    px["date"] = pd.to_datetime(px["date"], errors="coerce").dt.normalize()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["code", "date", "close"]).sort_values(["code", "date"]).reset_index(drop=True)
    if px.empty:
        return out

    for period in holding_periods:
        out[f"return_{int(period)}d"] = np.nan
        out[f"hit_{int(period)}d"] = np.nan
    out["max_drawdown_20d"] = np.nan

    code_groups = {code: df.reset_index(drop=True) for code, df in px.groupby("code")}

    for idx, row in out.iterrows():
        code = str(row.get("code", ""))
        decision_date = row.get("decision_date")
        signal = str(row.get("signal", "BUY")).upper()

        if not code or pd.isna(decision_date) or code not in code_groups:
            continue

        series = code_groups[code]
        base_candidates = series[series["date"] >= decision_date]
        if base_candidates.empty:
            continue

        base_idx = int(base_candidates.index[0])
        base_close = float(series.loc[base_idx, "close"])
        if not np.isfinite(base_close) or base_close <= 0:
            continue

        for period in holding_periods:
            forward_idx = base_idx + int(period)
            if forward_idx >= len(series):
                continue
            forward_close = float(series.loc[forward_idx, "close"])
            ret = ((forward_close / base_close) - 1.0) * 100.0
            out.at[idx, f"return_{int(period)}d"] = ret
            if signal == "SELL":
                out.at[idx, f"hit_{int(period)}d"] = 1 if ret < 0 else 0
            else:
                out.at[idx, f"hit_{int(period)}d"] = 1 if ret > 0 else 0

        dd_window_end = min(base_idx + 20, len(series) - 1)
        if dd_window_end > base_idx:
            min_close = float(series.loc[base_idx:dd_window_end, "close"].min())
            drawdown = ((min_close / base_close) - 1.0) * 100.0
            out.at[idx, "max_drawdown_20d"] = drawdown

    return out
