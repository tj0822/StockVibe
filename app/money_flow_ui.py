from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data import load_kospi_list, load_stock_data
from app.data_pipeline import build_stock_master_df
from app.money_flow_engine import MoneyFlowEngine
from app.ui_utils import add_naver_link_column


@st.cache_data(ttl=3600, show_spinner=False)
def _load_money_flow_input(data_dir: str = "data") -> pd.DataFrame:
    stock_df = load_stock_data(data_dir)
    if stock_df is None or stock_df.empty:
        return pd.DataFrame()

    work = stock_df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)

    names = load_kospi_list(data_dir)
    if names is not None and not names.empty:
        nm = names[["code", "name"]].copy()
        nm["code"] = nm["code"].astype(str).str.zfill(6)
        work = work.merge(nm, on="code", how="left")

    stock_master_df = build_stock_master_df(data_dir)
    if stock_master_df is not None and not stock_master_df.empty and "code" in stock_master_df.columns:
        sm = stock_master_df[["code", "sector"]].copy()
        sm["code"] = sm["code"].astype(str).str.zfill(6)
        sm = sm.drop_duplicates(subset=["code"], keep="last")
        work = work.merge(sm, on="code", how="left")

    return work


def render_money_flow_page() -> None:
    data_dir = "data"
    st.title("💰 Money Flow")
    st.caption("섹터/종목 단위 자금 유입 강도를 추적해 자금 회전과 유입 상위 종목을 확인합니다.")

    flow_input = _load_money_flow_input(data_dir)
    if flow_input is None or flow_input.empty:
        st.warning("Money Flow 분석을 위한 데이터가 없습니다.")
        return

    engine = MoneyFlowEngine()
    stock_flow_df = engine.compute_stock_money_flow(flow_input)
    stock_flow_df = engine.compute_flow_score(stock_flow_df)

    sector_flow_df = engine.compute_sector_money_flow(stock_flow_df)
    sector_flow_df = engine.compute_flow_score(sector_flow_df)
    sector_rank_df = engine.rank_flow_sectors(sector_flow_df)

    top_stocks_df = engine.top_flow_stocks(stock_flow_df)

    st.markdown("### Sector Flow Ranking")
    if sector_rank_df.empty:
        st.info("섹터 자금흐름 데이터가 없습니다.")
    else:
        st.dataframe(
            sector_rank_df[["rank", "sector", "sector_money_flow", "sector_flow_change", "flow_score"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "sector": st.column_config.TextColumn("Sector"),
                "sector_money_flow": st.column_config.NumberColumn("Sector Money Flow", format="%.0f"),
                "sector_flow_change": st.column_config.NumberColumn("Sector Flow Change(%)", format="%.2f"),
                "flow_score": st.column_config.NumberColumn("Flow Score", format="%.2f"),
            },
        )

    st.divider()
    st.markdown("### Sector Flow Heatmap")
    if sector_rank_df.empty:
        st.caption("표시할 히트맵 데이터가 없습니다.")
    else:
        heat = go.Figure(
            go.Heatmap(
                z=[sector_rank_df["flow_score"].tolist()],
                x=sector_rank_df["sector"].tolist(),
                y=["Flow Score"],
                colorscale="YlGnBu",
                zmin=0,
                zmax=100,
                colorbar=dict(title="Score"),
            )
        )
        heat.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(heat, use_container_width=True)

    st.divider()
    st.markdown("### Top Money Flow Stocks")
    if top_stocks_df.empty:
        st.info("유입 상위 종목 데이터가 없습니다.")
    else:
        top_stocks_df = add_naver_link_column(top_stocks_df, code_col="code", link_col="naver_stock_url")
        st.dataframe(
            top_stocks_df[["rank", "stock", "sector", "money_flow", "flow_change", "flow_momentum", "flow_score", "naver_stock_url"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "stock": st.column_config.TextColumn("Stock"),
                "sector": st.column_config.TextColumn("Sector"),
                "money_flow": st.column_config.NumberColumn("Money Flow", format="%.0f"),
                "flow_change": st.column_config.NumberColumn("Flow Change(%)", format="%.2f"),
                "flow_momentum": st.column_config.NumberColumn("Flow Momentum", format="%.2f"),
                "flow_score": st.column_config.NumberColumn("Flow Score", format="%.2f"),
                "naver_stock_url": st.column_config.LinkColumn("네이버 증권", display_text="바로가기"),
            },
        )

    st.divider()
    st.markdown("### Rotation Map")
    if sector_rank_df.empty:
        st.caption("로테이션 맵 데이터가 없습니다.")
    else:
        # x: sector_flow_change, y: flow_score, bubble: sector_money_flow
        bubble_size = pd.to_numeric(sector_rank_df.get("sector_money_flow"), errors="coerce").fillna(0.0)
        if bubble_size.max() > 0:
            bubble_size = 12 + 48 * (bubble_size / bubble_size.max())
        else:
            bubble_size = pd.Series([14] * len(sector_rank_df), index=sector_rank_df.index)

        scatter = go.Figure(
            go.Scatter(
                x=sector_rank_df["sector_flow_change"],
                y=sector_rank_df["flow_score"],
                mode="markers+text",
                text=sector_rank_df["sector"],
                textposition="top center",
                marker=dict(
                    size=bubble_size,
                    color=sector_rank_df["flow_score"],
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(title="Flow Score"),
                    opacity=0.8,
                ),
            )
        )
        scatter.update_layout(
            xaxis_title="Sector Flow Change (%)",
            yaxis_title="Flow Score",
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(scatter, use_container_width=True)
