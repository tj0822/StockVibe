from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data import load_kospi_index, load_marketcap_history
from app.data_pipeline import build_stock_master_df
from app.market_regime import (
    render_regime_summary,
    run_market_regime_engine,
    save_market_regime_snapshot,
)
from app.market_structure_ui import _load_sector_rank_history_cached
from app.marketcap_bump import build_marketcap_rank_history
from app.market_thermometer import (
    build_sector_heatmap,
    compute_market_concentration,
    compute_market_temperature,
    compute_sector_rotation_strength,
    get_market_leaders,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_stock_master_cached(data_dir: str = "data") -> pd.DataFrame:
    return build_stock_master_df(data_dir)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_marketcap_rank_cached(data_dir: str = "data") -> pd.DataFrame:
    stock_master_df = build_stock_master_df(data_dir)
    sector_map = stock_master_df[["code", "sector"]].copy() if not stock_master_df.empty else pd.DataFrame(columns=["code", "sector"])
    marketcap_df = load_marketcap_history(data_dir)
    return build_marketcap_rank_history(
        marketcap_df=marketcap_df,
        sector_map_df=sector_map,
        freq="W",
        top_n=10,
    )


def _render_temperature_gauge(score: float, state: str) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"suffix": " / 100", "font": {"size": 30}},
            title={"text": state, "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#f59e0b"},
                "steps": [
                    {"range": [0, 20], "color": "#1e3a8a"},
                    {"range": [20, 40], "color": "#0f766e"},
                    {"range": [40, 60], "color": "#334155"},
                    {"range": [60, 80], "color": "#15803d"},
                    {"range": [80, 100], "color": "#b91c1c"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_market_thermometer() -> None:
    data_dir = "data"

    st.title("🌡 시장 체온계")
    st.caption("시장 온도, 섹터 열지도, 로테이션, 리더십 집중도를 한 화면에서 빠르게 확인합니다.")

    stock_master_df = _load_stock_master_cached(data_dir)
    kospi_index_df = load_kospi_index(data_dir)
    sector_rank_df = _load_sector_rank_history_cached(data_dir)
    marketcap_rank_df = _load_marketcap_rank_cached(data_dir)

    if stock_master_df is None or stock_master_df.empty:
        st.warning("시장 체온계를 표시할 데이터가 없습니다.")
        return

    regime_result = run_market_regime_engine(
        stock_master_df=stock_master_df,
        kospi_index_df=kospi_index_df,
        sector_rank_df=sector_rank_df,
        marketcap_rank_df=marketcap_rank_df,
    )
    save_market_regime_snapshot(regime_result)
    render_regime_summary(regime_result)
    st.divider()

    # Section 1
    st.markdown("#### 🌡 Market Temperature")
    temp = compute_market_temperature(stock_master_df)

    t1, t2 = st.columns([2, 1])
    with t1:
        _render_temperature_gauge(temp.get("temperature_score", 50.0), temp.get("market_state", "🌤 Neutral"))
    with t2:
        st.metric("Market State", temp.get("market_state", "🌤 Neutral"))
        st.metric("Temperature Score", f"{float(temp.get('temperature_score', 50.0)):.1f}")
        st.caption("시장 상태 해석: Overheated / Strong Uptrend / Neutral / Weak / Risk-off")

    st.divider()

    # Section 2
    st.markdown("#### 🔥 Sector Heatmap")
    heatmap_df = build_sector_heatmap(stock_master_df)
    if heatmap_df.empty:
        st.info("섹터 열지도 데이터가 없습니다.")
    else:
        strongest = heatmap_df.sort_values("sector_power", ascending=False).iloc[0]
        weakest = heatmap_df.sort_values("sector_power", ascending=True).iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Strongest Sector", str(strongest["sector"]), f"Power {float(strongest['sector_power']):.1f}")
        with c2:
            st.metric("Weakest Sector", str(weakest["sector"]), f"Power {float(weakest['sector_power']):.1f}")

        fig_heat = go.Figure(
            go.Heatmap(
                z=[heatmap_df["sector_power"].tolist()],
                x=heatmap_df["sector"].tolist(),
                y=["sector_power"],
                colorscale="YlOrRd",
                zmin=0,
                zmax=100,
                colorbar=dict(title="Power"),
            )
        )
        fig_heat.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.dataframe(
            heatmap_df[["sector", "sector_power", "avg_score", "avg_return", "stock_count"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "sector": st.column_config.TextColumn("섹터"),
                "sector_power": st.column_config.NumberColumn("Sector Power", format="%.1f"),
                "avg_score": st.column_config.NumberColumn("Avg Score", format="%.1f"),
                "avg_return": st.column_config.NumberColumn("Avg Return(1M,%)", format="%.2f"),
                "stock_count": st.column_config.NumberColumn("Stock Count", format="%d"),
            },
        )

    st.divider()

    # Section 3
    st.markdown("#### 🔄 Sector Rotation")
    rotation = compute_sector_rotation_strength(sector_rank_df)
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Rotation Strength", f"{float(rotation.get('rotation_strength', 0.0)):.2f}")
    with r2:
        st.metric("Interpretation", rotation.get("rotation_level", "Stable"))

    rr1, rr2 = st.columns(2)
    with rr1:
        st.markdown("**Top Rising Sectors**")
        rising_df = rotation.get("top_rising_sectors", pd.DataFrame())
        if rising_df is None or rising_df.empty:
            st.caption("없음")
        else:
            st.dataframe(rising_df, hide_index=True, use_container_width=True)
    with rr2:
        st.markdown("**Top Falling Sectors**")
        falling_df = rotation.get("top_falling_sectors", pd.DataFrame())
        if falling_df is None or falling_df.empty:
            st.caption("없음")
        else:
            st.dataframe(falling_df, hide_index=True, use_container_width=True)

    st.divider()

    # Section 4
    st.markdown("#### 👑 Market Leadership")
    concentration = compute_market_concentration(marketcap_rank_df)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Dominant Sector", concentration.get("largest_sector", "-"))
        st.metric("Concentration", f"{float(concentration.get('largest_sector_share', 0.0)):.1f}%")
        st.metric("State", concentration.get("concentration_level", "Diversified"))
    with c2:
        dist_df = concentration.get("sector_distribution", pd.DataFrame())
        if dist_df is not None and not dist_df.empty:
            pie = go.Figure(
                go.Pie(
                    labels=dist_df["sector"],
                    values=dist_df["count"],
                    hole=0.45,
                )
            )
            pie.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(pie, use_container_width=True)

    dominant_sector = concentration.get("largest_sector", "-")
    dominant_share = float(concentration.get("largest_sector_share", 0.0))
    st.caption(f"{dominant_sector} 섹터가 시총 상위 종목의 {dominant_share:.1f}%를 차지하고 있습니다.")

    st.divider()

    # Section 5
    st.markdown("#### 🏆 Current Market Leaders")
    leaders_df = get_market_leaders(marketcap_rank_df, stock_master_df)
    if leaders_df.empty:
        st.info("시총 리더 테이블 데이터가 없습니다.")
    else:
        st.dataframe(
            leaders_df[["rank", "name", "sector", "sector_state", "signal", "final_score_adjusted"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "name": st.column_config.TextColumn("종목명"),
                "sector": st.column_config.TextColumn("섹터"),
                "sector_state": st.column_config.TextColumn("섹터 상태"),
                "signal": st.column_config.TextColumn("Signal"),
                "final_score_adjusted": st.column_config.NumberColumn("Adjusted Score", format="%.1f"),
            },
        )
