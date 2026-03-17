from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data import load_kospi_list, load_stock_data
from app.data_pipeline import build_stock_master_df
from app.market_structure_ui import _load_sector_rank_history_cached
from app.money_flow_engine import MoneyFlowEngine
from app.sector_prediction_engine import SectorPredictionEngine


@st.cache_data(ttl=3600, show_spinner=False)
def _load_prediction_input(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock_df = load_stock_data(data_dir)
    stock_master_df = build_stock_master_df(data_dir)
    sector_rank_df = _load_sector_rank_history_cached(data_dir)

    if stock_df is None:
        stock_df = pd.DataFrame()
    if stock_master_df is None:
        stock_master_df = pd.DataFrame()
    if sector_rank_df is None:
        sector_rank_df = pd.DataFrame()

    return stock_df, stock_master_df, sector_rank_df


@st.cache_data(ttl=3600, show_spinner=False)
def _run_sector_prediction_backtest_cached(
    stock_df: pd.DataFrame,
    sector_rank_df: pd.DataFrame,
    sector_map_df: pd.DataFrame,
    horizons: tuple[int, ...],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = SectorPredictionEngine()
    return engine.backtest_prediction_accuracy(
        stock_df=stock_df,
        sector_rank_df=sector_rank_df,
        sector_map_df=sector_map_df,
        horizon_months=horizons,
        top_n=top_n,
    )


def render_sector_prediction_page() -> None:
    data_dir = "data"
    st.title("🔮 Sector Prediction")
    st.caption("자금흐름/순위변화/리더강도를 결합해 향후 자금 유입 가능성이 높은 섹터를 예측합니다.")

    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        horizon_labels = st.multiselect(
            "백테스트 기간",
            ["3개월", "6개월", "1년"],
            default=["3개월", "6개월", "1년"],
            help="과거 월말 예측 기준으로 이후 해당 기간 동안 섹터 예측이 얼마나 적중했는지 평가합니다.",
        )
    with ctrl2:
        backtest_top_n = st.number_input(
            "예측 섹터 수",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="각 과거 시점마다 상위 몇 개 섹터를 예측 대상으로 볼지 설정합니다.",
        )

    stock_df, stock_master_df, sector_rank_df = _load_prediction_input(data_dir)
    if stock_df.empty or stock_master_df.empty:
        st.warning("Sector Prediction 분석에 필요한 데이터가 부족합니다.")
        return

    # Build stock flow input with sector and name
    flow_input = stock_df.copy()
    flow_input["code"] = flow_input["code"].astype(str).str.zfill(6)

    sector_map = stock_master_df[["code", "sector"]].copy() if "code" in stock_master_df.columns else pd.DataFrame(columns=["code", "sector"])
    if not sector_map.empty:
        sector_map["code"] = sector_map["code"].astype(str).str.zfill(6)
        sector_map = sector_map.drop_duplicates(subset=["code"], keep="last")
        flow_input = flow_input.merge(sector_map, on="code", how="left")

    names_df = load_kospi_list(data_dir)
    if names_df is not None and not names_df.empty:
        nm = names_df[["code", "name"]].copy()
        nm["code"] = nm["code"].astype(str).str.zfill(6)
        flow_input = flow_input.merge(nm, on="code", how="left")

    money_engine = MoneyFlowEngine()
    stock_flow_df = money_engine.compute_stock_money_flow(flow_input)
    stock_flow_df = money_engine.compute_flow_score(stock_flow_df)
    sector_flow_df = money_engine.compute_sector_money_flow(stock_flow_df)
    sector_flow_df = money_engine.compute_flow_score(sector_flow_df)

    # enrich stock_master with money_flow_score for leader detection
    sm_work = stock_master_df.copy()
    if "code" in sm_work.columns and stock_flow_df is not None and not stock_flow_df.empty:
        sm_work["code"] = sm_work["code"].astype(str).str.zfill(6)
        fm = stock_flow_df[["code", "flow_score"]].copy()
        fm["code"] = fm["code"].astype(str).str.zfill(6)
        sm_work = sm_work.merge(fm, on="code", how="left")
        sm_work["money_flow_score"] = pd.to_numeric(sm_work.get("flow_score"), errors="coerce").fillna(50.0)

    pred_engine = SectorPredictionEngine()
    features_df = pred_engine.compute_sector_features(sector_flow_df, sector_rank_df, sm_work)
    scored_df = pred_engine.compute_prediction_score(features_df)
    classified_df = pred_engine.classify_sector_state(scored_df)
    emerging_df = pred_engine.predict_future_sectors(classified_df)
    leaders_df = pred_engine.detect_leader_stocks(sm_work)

    st.markdown("### Emerging Sectors")
    if emerging_df.empty:
        st.info("현재 Emerging/Leading 섹터가 감지되지 않았습니다.")
    else:
        st.dataframe(
            emerging_df[["sector", "prediction_score", "sector_state_pred", "flow_momentum", "rank_change", "leader_strength"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "sector": st.column_config.TextColumn("Sector"),
                "prediction_score": st.column_config.NumberColumn("Prediction Score", format="%.2f"),
                "sector_state_pred": st.column_config.TextColumn("State"),
                "flow_momentum": st.column_config.NumberColumn("Flow Momentum", format="%.2f"),
                "rank_change": st.column_config.NumberColumn("Rank Change", format="%.2f"),
                "leader_strength": st.column_config.NumberColumn("Leader Strength", format="%.2f"),
            },
        )

    st.divider()
    st.markdown("### Sector Rotation Radar")
    if classified_df.empty:
        st.caption("레이더 차트 데이터가 없습니다.")
    else:
        radar_target = classified_df.sort_values("prediction_score", ascending=False).head(5).copy()
        radar = go.Figure()
        categories = ["flow_momentum", "rank_change", "volume_spike", "leader_strength", "prediction_score"]
        for _, row in radar_target.iterrows():
            values = [
                float(pd.to_numeric(row.get("flow_momentum", 0.0), errors="coerce")),
                float(pd.to_numeric(row.get("rank_change", 0.0), errors="coerce") * 10.0),
                float(pd.to_numeric(row.get("volume_spike", 0.0), errors="coerce")),
                float(pd.to_numeric(row.get("leader_strength", 0.0), errors="coerce")),
                float(pd.to_numeric(row.get("prediction_score", 0.0), errors="coerce")),
            ]
            values = [max(0.0, min(100.0, v)) for v in values]
            radar.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=str(row.get("sector", "-")),
                )
            )

        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=480,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(radar, use_container_width=True)

    st.divider()
    st.markdown("### Leader Stocks")
    if leaders_df.empty:
        st.info("조건을 만족하는 리더 종목이 없습니다.")
    else:
        st.dataframe(
            leaders_df[["sector", "stock", "momentum_score", "money_flow_score", "leader_strength"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "sector": st.column_config.TextColumn("Sector"),
                "stock": st.column_config.TextColumn("Leader Stock"),
                "momentum_score": st.column_config.NumberColumn("Momentum", format="%.1f"),
                "money_flow_score": st.column_config.NumberColumn("Money Flow", format="%.1f"),
                "leader_strength": st.column_config.NumberColumn("Leader Strength", format="%.1f"),
            },
        )

    st.divider()
    st.markdown("### Prediction Heatmap")
    if classified_df.empty:
        st.caption("히트맵 데이터가 없습니다.")
    else:
        heat_df = classified_df.sort_values("prediction_score", ascending=False)
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=[heat_df["prediction_score"].tolist()],
                x=heat_df["sector"].tolist(),
                y=["prediction_score"],
                zmin=0,
                zmax=100,
                colorscale="YlOrRd",
                colorbar=dict(title="Score"),
            )
        )
        fig_heat.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.markdown("### ⏪ Prediction Backtest")
    horizon_map = {"3개월": 3, "6개월": 6, "1년": 12}
    selected_horizons = tuple(horizon_map[label] for label in horizon_labels if label in horizon_map)

    if not selected_horizons:
        st.info("백테스트 기간을 1개 이상 선택하세요.")
        return

    sector_map_df = stock_master_df[[c for c in ["code", "name", "sector"] if c in stock_master_df.columns]].copy()
    with st.spinner("Sector Prediction 백테스트 계산 중..."):
        summary_df, detail_df = _run_sector_prediction_backtest_cached(
            stock_df=stock_df,
            sector_rank_df=sector_rank_df,
            sector_map_df=sector_map_df,
            horizons=selected_horizons,
            top_n=int(backtest_top_n),
        )

    if summary_df.empty:
        st.info("백테스트 가능한 과거 구간이 부족합니다.")
        return

    summary_order = {"3M": 3, "6M": 6, "12M": 12}
    summary_df["_order"] = summary_df["horizon"].map(summary_order).fillna(999)
    summary_df = summary_df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    metric_cols = st.columns(len(summary_df))
    for idx, row in summary_df.iterrows():
        with metric_cols[idx]:
            st.metric(
                f"{row['horizon']} 적중률",
                f"{float(row['hit_rate']):.1f}%",
                delta=f"초과수익 {float(row['avg_predicted_return'] - row['avg_benchmark_return']):+.2f}%p",
            )

    st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "horizon": st.column_config.TextColumn("기간"),
            "samples": st.column_config.NumberColumn("샘플 수", format="%d"),
            "hit_rate": st.column_config.NumberColumn("적중률(%)", format="%.1f"),
            "avg_predicted_return": st.column_config.NumberColumn("예측 섹터 평균 수익률(%)", format="%.2f"),
            "avg_benchmark_return": st.column_config.NumberColumn("벤치마크 평균 수익률(%)", format="%.2f"),
        },
    )

    if detail_df is not None and not detail_df.empty:
        detail_df = detail_df.copy()
        detail_df["hit_label"] = detail_df["hit"].map({True: "HIT", False: "MISS"})
        st.caption("적중 기준: 예측한 섹터의 미래 수익률이 동일 시점 전체 섹터 중앙값을 상회하면 HIT")
        st.dataframe(
            detail_df[["as_of_date", "future_date", "horizon", "sector", "prediction_score", "future_return", "benchmark_return", "hit_label"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "as_of_date": st.column_config.DateColumn("예측 기준일"),
                "future_date": st.column_config.DateColumn("평가 종료일"),
                "horizon": st.column_config.TextColumn("기간"),
                "sector": st.column_config.TextColumn("섹터"),
                "prediction_score": st.column_config.NumberColumn("Prediction Score", format="%.2f"),
                "future_return": st.column_config.NumberColumn("미래 수익률(%)", format="%.2f"),
                "benchmark_return": st.column_config.NumberColumn("벤치마크(%)", format="%.2f"),
                "hit_label": st.column_config.TextColumn("결과"),
            },
        )
