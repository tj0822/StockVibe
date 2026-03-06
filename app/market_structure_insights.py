from __future__ import annotations

from typing import List, Dict

import pandas as pd
import streamlit as st


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out.dropna(subset=[col])


def _safe_to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


@st.cache_data(ttl=3600, show_spinner=False)
def generate_market_structure_insights(
    stock_master_df: pd.DataFrame,
    sector_rank_df: pd.DataFrame,
    marketcap_rank_df: pd.DataFrame,
) -> List[Dict[str, str]]:
    insights: List[Dict[str, str]] = []

    sm = stock_master_df.copy() if stock_master_df is not None else _empty_df([])
    sr = sector_rank_df.copy() if sector_rank_df is not None else _empty_df(["date", "sector", "rank"])
    mc = marketcap_rank_df.copy() if marketcap_rank_df is not None else _empty_df(["date", "code", "name", "sector", "rank", "market_cap"])

    # Rule 1: Strongest sector
    strongest_sector = None
    if not sm.empty and "sector" in sm.columns and "sector_rank" in sm.columns:
        sec = (
            sm[["sector", "sector_rank"]]
            .dropna(subset=["sector"])
            .drop_duplicates(subset=["sector"], keep="last")
            .copy()
        )
        sec["sector_rank"] = _safe_to_num(sec["sector_rank"])
        sec = sec.dropna(subset=["sector_rank"]).sort_values("sector_rank", ascending=True)
        if not sec.empty:
            strongest_sector = str(sec.iloc[0]["sector"])

    if strongest_sector is None and not sr.empty:
        sr = _safe_to_datetime(sr, "date")
        if not sr.empty:
            latest_sr = sr[sr["date"] == sr["date"].max()].copy()
            latest_sr["rank"] = _safe_to_num(latest_sr["rank"])
            latest_sr = latest_sr.dropna(subset=["rank"]).sort_values("rank", ascending=True)
            if not latest_sr.empty:
                strongest_sector = str(latest_sr.iloc[0]["sector"])

    if strongest_sector:
        insights.append(
            {
                "title": "시장 주도 섹터",
                "message": f"현재 {strongest_sector} 섹터가 시장에서 가장 강한 자금 흐름을 보이고 있습니다.",
                "level": "positive",
            }
        )

    # Rule 2: Sector concentration in Top10 market-cap leaders
    if not mc.empty and {"date", "sector"}.issubset(mc.columns):
        mc = _safe_to_datetime(mc, "date")
        if not mc.empty:
            latest_mc = mc[mc["date"] == mc["date"].max()].copy()
            latest_mc["rank"] = _safe_to_num(latest_mc.get("rank", pd.Series(dtype=float)))
            latest_mc = latest_mc.dropna(subset=["rank"]).sort_values("rank", ascending=True).head(10)

            if not latest_mc.empty:
                sector_counts = latest_mc["sector"].fillna("Unknown").value_counts()
                top_sector = str(sector_counts.index[0])
                top_ratio = float(sector_counts.iloc[0] / max(len(latest_mc), 1))
                if top_ratio >= 0.40:
                    insights.append(
                        {
                            "title": "시장 리더십 편중 경고",
                            "message": (
                                f"현재 시총 상위 종목이 {top_sector} 섹터에 집중되어 있습니다. "
                                "시장 리더십이 특정 산업에 편중된 상태입니다."
                            ),
                            "level": "warning",
                        }
                    )
                else:
                    insights.append(
                        {
                            "title": "시장 리더십 분산",
                            "message": "시총 상위 종목이 여러 섹터에 분산되어 있습니다. 시장 리더십이 비교적 넓게 퍼져 있습니다.",
                            "level": "info",
                        }
                    )

    # Rule 3: Sector rotation from latest vs previous rank
    if not sr.empty and {"date", "sector", "rank"}.issubset(sr.columns):
        sr = _safe_to_datetime(sr, "date")
        sr["rank"] = _safe_to_num(sr["rank"])
        sr = sr.dropna(subset=["rank", "sector"])

        dates = sorted(sr["date"].unique().tolist())
        if len(dates) >= 2:
            latest_date = dates[-1]
            prev_date = dates[-2]
            latest_df = sr[sr["date"] == latest_date][["sector", "rank"]].rename(columns={"rank": "latest_rank"})
            prev_df = sr[sr["date"] == prev_date][["sector", "rank"]].rename(columns={"rank": "prev_rank"})
            diff = latest_df.merge(prev_df, on="sector", how="inner")
            diff["improvement"] = diff["prev_rank"] - diff["latest_rank"]
            diff["drop"] = diff["latest_rank"] - diff["prev_rank"]

            risers = diff[diff["improvement"] >= 5].sort_values("improvement", ascending=False)
            for _, row in risers.head(2).iterrows():
                sector = str(row["sector"])
                insights.append(
                    {
                        "title": "섹터 로테이션",
                        "message": f"{sector} 섹터가 최근 빠르게 상승하며 시장의 새로운 관심을 받고 있습니다.",
                        "level": "positive",
                    }
                )

            fallers = diff[diff["drop"] >= 5].sort_values("drop", ascending=False)
            for _, row in fallers.head(2).iterrows():
                sector = str(row["sector"])
                insights.append(
                    {
                        "title": "섹터 약화 경고",
                        "message": f"{sector} 섹터의 수급 순위가 최근 크게 하락했습니다. 단기적인 자금 이탈 가능성이 있습니다.",
                        "level": "warning",
                    }
                )

    # Rule 4: Market leader stability (Top3)
    if not mc.empty and {"date", "name", "rank"}.issubset(mc.columns):
        mc = _safe_to_datetime(mc, "date")
        mc["rank"] = _safe_to_num(mc["rank"])
        mc = mc.dropna(subset=["rank", "name"])
        dates = sorted(mc["date"].unique().tolist())
        if len(dates) >= 2:
            latest_date = dates[-1]
            prev_date = dates[-2]
            latest_top3 = mc[mc["date"] == latest_date].sort_values("rank").head(3)
            prev_top3 = mc[mc["date"] == prev_date].sort_values("rank").head(3)

            latest_names = set(latest_top3["name"].astype(str).tolist())
            prev_names = set(prev_top3["name"].astype(str).tolist())

            if latest_names and latest_names == prev_names:
                insights.append(
                    {
                        "title": "시장 리더십 안정성",
                        "message": "상위 시총 종목 구성이 안정적으로 유지되고 있습니다. 현재 시장 리더십 구조가 비교적 안정적인 상태입니다.",
                        "level": "info",
                    }
                )
            else:
                new_entrants = [n for n in latest_top3["name"].astype(str).tolist() if n not in prev_names]
                if new_entrants:
                    insights.append(
                        {
                            "title": "시총 리더 교체",
                            "message": f"{new_entrants[0]} 종목이 시총 상위권에 새롭게 진입했습니다. 시장 리더십 변화가 나타나고 있습니다.",
                            "level": "positive",
                        }
                    )

    # Rule 5: Weak sector warning (using current analyzed universe as portfolio proxy)
    if not sm.empty and {"sector_state"}.issubset(sm.columns):
        weak_ratio = float((sm["sector_state"] == "WEAK").mean())
        weak_count = int((sm["sector_state"] == "WEAK").sum())
        if weak_ratio >= 0.35 and weak_count >= 5:
            insights.append(
                {
                    "title": "약세 섹터 비중 경고",
                    "message": "현재 약세 섹터 비중이 높은 상태입니다. 방어적인 포트폴리오 관리가 필요할 수 있습니다.",
                    "level": "warning",
                }
            )

    if not insights:
        insights.append(
            {
                "title": "시장구조 요약",
                "message": "현재 데이터에서 뚜렷한 구조 변화 신호는 제한적입니다. 추세를 계속 관찰하세요.",
                "level": "info",
            }
        )

    return insights


def render_market_insight_cards(insights: list[dict]) -> None:
    if not insights:
        return

    style = """
    <style>
    .ms-insight-card {
        border-radius: 12px;
        padding: 12px 14px;
        border: 1px solid #2e2e2e;
        background: rgba(20, 20, 20, 0.55);
        min-height: 120px;
    }
    .ms-insight-title {
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .ms-insight-msg {
        font-size: 13px;
        line-height: 1.45;
        color: #d7d7d7;
        word-break: keep-all;
    }
    .ms-pos { border-left: 5px solid #22c55e; }
    .ms-info { border-left: 5px solid #3b82f6; }
    .ms-warn { border-left: 5px solid #ef4444; }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

    icon_map = {"positive": "🟢", "info": "🔵", "warning": "🔴"}
    class_map = {"positive": "ms-pos", "info": "ms-info", "warning": "ms-warn"}

    per_row = 4
    for i in range(0, len(insights), per_row):
        row = insights[i : i + per_row]
        cols = st.columns(len(row))
        for col, insight in zip(cols, row):
            title = str(insight.get("title", "인사이트"))
            message = str(insight.get("message", ""))
            level = str(insight.get("level", "info"))
            icon = icon_map.get(level, "🔵")
            css_class = class_map.get(level, "ms-info")

            with col:
                st.markdown(
                    (
                        f"<div class='ms-insight-card {css_class}'>"
                        f"<div class='ms-insight-title'>{icon} {title}</div>"
                        f"<div class='ms-insight-msg'>{message}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
