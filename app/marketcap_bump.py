import hashlib
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


DEFAULT_PALETTE = (
    qualitative.Safe
    + qualitative.Bold
    + qualitative.Set3
    + qualitative.D3
)


def _ensure_marketcap_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "code", "name", "market_cap"]
    if df is None or df.empty:
        return pd.DataFrame(columns=required)

    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = pd.NA

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["name"] = out["name"].astype(str)
    out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
    out = out.dropna(subset=["date", "code", "market_cap"])
    return out[required]


def build_marketcap_rank_history(
    marketcap_df: pd.DataFrame,
    sector_map_df: pd.DataFrame,
    freq: str = "W",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Input columns in marketcap_df:
    - date
    - code
    - name
    - market_cap

    Output columns:
    - date
    - code
    - name
    - sector
    - market_cap
    - rank
    """
    base = _ensure_marketcap_columns(marketcap_df)
    if base.empty:
        return pd.DataFrame(columns=["date", "code", "name", "sector", "market_cap", "rank"])

    safe_freq = "M" if str(freq).upper().startswith("M") else "W"
    safe_top_n = max(int(top_n), 1)

    base = base.sort_values(["code", "date"]).copy()
    # Per period bucket, keep the latest available record for each stock.
    base["bucket"] = base["date"].dt.to_period(safe_freq)
    snap = base.groupby(["bucket", "code"], as_index=False).tail(1).copy()

    snap["date"] = snap["bucket"].dt.to_timestamp(how="end").dt.normalize()
    snap = snap.drop(columns=["bucket"])

    ranked = (
        snap.sort_values(["date", "market_cap"], ascending=[True, False])
        .groupby("date", as_index=False, group_keys=False)
        .head(safe_top_n)
        .copy()
    )
    ranked["rank"] = ranked.groupby("date").cumcount() + 1

    if sector_map_df is None or sector_map_df.empty:
        ranked["sector"] = "Unknown"
    else:
        sm = sector_map_df.copy()
        if "code" not in sm.columns:
            sm["code"] = pd.NA
        if "sector" not in sm.columns:
            sm["sector"] = "Unknown"
        sm["code"] = sm["code"].astype(str).str.zfill(6)
        sm["sector"] = sm["sector"].astype(str).str.strip().replace("", "Unknown")
        sm = sm[["code", "sector"]].drop_duplicates(subset=["code"], keep="last")
        ranked = ranked.merge(sm, on="code", how="left")
        ranked["sector"] = ranked["sector"].fillna("Unknown").replace("", "Unknown")

    ranked = ranked[["date", "code", "name", "sector", "market_cap", "rank"]]
    return ranked.sort_values(["date", "rank", "code"]).reset_index(drop=True)


def _deterministic_color(key: str, palette: List[str]) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % max(len(palette), 1)
    return palette[idx]


def get_stock_color_map(codes: List[str]) -> Dict[str, str]:
    """Return deterministic color per stock code."""
    uniq = sorted({str(c) for c in codes if pd.notna(c)})
    return {code: _deterministic_color(f"stock:{code}", DEFAULT_PALETTE) for code in uniq}


def get_sector_color_map(sectors: List[str]) -> Dict[str, str]:
    """Return deterministic color per sector."""
    uniq = sorted({str(s) for s in sectors if pd.notna(s)})
    return {sector: _deterministic_color(f"sector:{sector}", DEFAULT_PALETTE) for sector in uniq}


def render_marketcap_bump_chart(rank_history_df: pd.DataFrame, color_mode: str = "stock") -> go.Figure:
    """
    color_mode: 'stock' or 'sector'
    Returns a Plotly figure.
    """
    fig = go.Figure()

    if rank_history_df is None or rank_history_df.empty:
        fig.update_layout(
            template="plotly_dark",
            height=560,
            xaxis_title="기간",
            yaxis_title="랭킹 (1=상위)",
            annotations=[
                dict(
                    text="표시할 시가총액 랭킹 데이터가 없습니다.",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
        )
        return fig

    df = rank_history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.dropna(subset=["date", "code", "rank", "market_cap"]).copy()

    if df.empty:
        return render_marketcap_bump_chart(pd.DataFrame(), color_mode=color_mode)

    stock_colors = get_stock_color_map(df["code"].tolist())
    sector_colors = get_sector_color_map(df["sector"].fillna("Unknown").tolist())

    latest_date = df["date"].max()
    latest_top_n = int((df["date"] == latest_date).sum())
    show_legend = latest_top_n <= 15

    date_axis = sorted(df["date"].dropna().unique().tolist())

    for code, group_df in df.groupby("code"):
        g = group_df.sort_values("date").set_index("date").reindex(date_axis)
        source_row = group_df.sort_values("date").iloc[-1]
        label = f"{source_row.get('name', code)} ({code})"
        sector = str(source_row.get("sector", "Unknown"))

        if str(color_mode).lower() == "sector":
            line_color = sector_colors.get(sector, "#BBBBBB")
        else:
            line_color = stock_colors.get(str(code), "#BBBBBB")

        custom_data = pd.DataFrame(
            {
                "name": g.get("name", pd.Series(index=g.index, dtype=object)).fillna(source_row.get("name", "")),
                "code": str(code),
                "sector": g.get("sector", pd.Series(index=g.index, dtype=object)).fillna(sector),
                "market_cap": g.get("market_cap", pd.Series(index=g.index, dtype=float)),
                "date": g.index,
            }
        )

        fig.add_trace(
            go.Scatter(
                x=g.index,
                y=g["rank"],
                mode="lines+markers",
                name=label,
                marker=dict(size=7),
                line=dict(width=2.0, color=line_color),
                connectgaps=False,
                opacity=0.85,
                customdata=custom_data[["name", "code", "sector", "market_cap", "date"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "코드: %{customdata[1]}<br>"
                    "섹터: %{customdata[2]}<br>"
                    "랭킹: %{y:.0f}<br>"
                    "시가총액: %{customdata[3]:,.0f}<br>"
                    "기준일: %{customdata[4]|%Y-%m-%d}<extra></extra>"
                ),
                showlegend=show_legend,
            )
        )

    max_rank = max(int(df["rank"].max()), 1)
    fig.update_layout(
        height=620,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="기간",
        yaxis_title="랭킹 (1=상위)",
        legend_title="종목",
        hovermode="closest",
    )
    fig.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=list(range(1, max_rank + 1)),
    )
    return fig


def summarize_rank_changes(rank_history_df: pd.DataFrame) -> dict:
    """
    Compare latest date vs previous date bucket.
    Return:
    - latest_date
    - prev_date
    - new_entries: list[dict]
    - exits: list[dict]
    """
    if rank_history_df is None or rank_history_df.empty:
        return {
            "latest_date": None,
            "prev_date": None,
            "new_entries": [],
            "exits": [],
        }

    df = rank_history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "code"])
    dates = sorted(df["date"].unique().tolist())

    if len(dates) < 2:
        return {
            "latest_date": dates[-1] if dates else None,
            "prev_date": None,
            "new_entries": [],
            "exits": [],
        }

    latest_date = dates[-1]
    prev_date = dates[-2]

    latest_df = df[df["date"] == latest_date].copy()
    prev_df = df[df["date"] == prev_date].copy()

    latest_codes = set(latest_df["code"].astype(str))
    prev_codes = set(prev_df["code"].astype(str))

    new_codes = latest_codes - prev_codes
    exit_codes = prev_codes - latest_codes

    latest_df["code"] = latest_df["code"].astype(str)
    prev_df["code"] = prev_df["code"].astype(str)

    new_entries = (
        latest_df[latest_df["code"].isin(new_codes)]
        .sort_values("rank")
        [["code", "name", "sector", "rank"]]
        .rename(columns={"rank": "current_rank"})
        .to_dict(orient="records")
    )
    exits = (
        prev_df[prev_df["code"].isin(exit_codes)]
        .sort_values("rank")
        [["code", "name", "sector", "rank"]]
        .rename(columns={"rank": "previous_rank"})
        .to_dict(orient="records")
    )

    return {
        "latest_date": latest_date,
        "prev_date": prev_date,
        "new_entries": new_entries,
        "exits": exits,
    }
