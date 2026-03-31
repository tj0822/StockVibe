from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data import load_stock_data
from app.data_pipeline import build_stock_master_df
from app.market_structure_ui import _load_sector_rank_history_cached
from app.money_flow_engine import MoneyFlowEngine
from app.sector_prediction_ai import SectorPredictionAI
from app.ui_utils import add_naver_link_column


@st.cache_data(ttl=3600, show_spinner=False)
def _load_future_sector_inputs(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock_df = load_stock_data(data_dir)
    stock_master_df = build_stock_master_df(data_dir)
    sector_rank_df = _load_sector_rank_history_cached(data_dir)
    return stock_df if stock_df is not None else pd.DataFrame(), stock_master_df if stock_master_df is not None else pd.DataFrame(), sector_rank_df if sector_rank_df is not None else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_news_proxy_df(data_dir: str, stock_master_df: pd.DataFrame) -> pd.DataFrame:
    path = os.path.join(data_dir, "trading_input_log.json")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "name", "title", "sentiment_score", "sector", "source"])

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return pd.DataFrame(columns=["date", "name", "title", "sentiment_score", "sector", "source"])

    rows: list[dict] = []
    if isinstance(raw, list):
        for entry in raw:
            parsed_trades = entry.get("parsed_trades", []) if isinstance(entry, dict) else []
            for trade in parsed_trades:
                if not isinstance(trade, dict):
                    continue
                action = str(trade.get("action", "")).upper()
                name = str(trade.get("name", "")).strip()
                if not name:
                    continue
                sentiment_score = 0.6 if action == "BUY" else (-0.6 if action == "SELL" else 0.0)
                title = f"{name} {action} activity"
                rows.append(
                    {
                        "date": trade.get("date"),
                        "name": name,
                        "title": title,
                        "sentiment_score": sentiment_score,
                        "source": "trade_log_proxy",
                    }
                )

    news_df = pd.DataFrame(rows)
    if news_df.empty:
        return pd.DataFrame(columns=["date", "name", "title", "sentiment_score", "sector", "source"])

    if stock_master_df is not None and not stock_master_df.empty and {"name", "sector"}.issubset(set(stock_master_df.columns)):
        sector_map = stock_master_df[["name", "sector"]].drop_duplicates(subset=["name"], keep="last")
        news_df = news_df.merge(sector_map, on="name", how="left")
    else:
        news_df["sector"] = None

    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
    return news_df


def render_future_sector_prediction_page() -> None:
    data_dir = "data"
    st.title("🔮 Future Sector Prediction")
    st.caption("자금흐름, 순위 변화, 섹터 파워, 뉴스 감성/량을 결합해 미래 주도 섹터를 예측합니다.")
    st.caption("뉴스 원천이 없을 경우 로컬 거래 이벤트 로그를 감성 프록시로 사용합니다.")

    stock_df, stock_master_df, sector_rank_df = _load_future_sector_inputs(data_dir)
    if stock_df.empty or stock_master_df.empty:
        st.warning("Future Sector Prediction 분석에 필요한 데이터가 부족합니다.")
        return

    flow_input = stock_df.copy()
    flow_input["code"] = flow_input["code"].astype(str).str.zfill(6)
    sector_map = stock_master_df[["code", "name", "sector"]].copy() if {"code", "name", "sector"}.issubset(set(stock_master_df.columns)) else pd.DataFrame(columns=["code", "name", "sector"])
    if not sector_map.empty:
        sector_map["code"] = sector_map["code"].astype(str).str.zfill(6)
        sector_map = sector_map.drop_duplicates(subset=["code"], keep="last")
        flow_input = flow_input.merge(sector_map, on="code", how="left")

    money_engine = MoneyFlowEngine()
    stock_flow_df = money_engine.compute_stock_money_flow(flow_input)
    stock_flow_df = money_engine.compute_flow_score(stock_flow_df)
    sector_flow_df = money_engine.compute_sector_money_flow(stock_flow_df)
    sector_flow_df = money_engine.compute_flow_score(sector_flow_df)

    if "code" in stock_master_df.columns and not stock_flow_df.empty:
        flow_merge = stock_flow_df[["code", "flow_score"]].copy()
        flow_merge["code"] = flow_merge["code"].astype(str).str.zfill(6)
        stock_master_df = stock_master_df.copy()
        stock_master_df["code"] = stock_master_df["code"].astype(str).str.zfill(6)
        stock_master_df = stock_master_df.merge(flow_merge, on="code", how="left")
        stock_master_df["money_flow_score"] = pd.to_numeric(stock_master_df.get("flow_score"), errors="coerce").fillna(50.0)

    news_df = _load_news_proxy_df(data_dir, stock_master_df)

    ai_engine = SectorPredictionAI()
    feature_df = ai_engine.build_sector_feature_table(
        sector_flow_df=sector_flow_df,
        sector_rank_df=sector_rank_df,
        news_df=news_df,
        stock_master_df=stock_master_df,
    )
    scored_df = ai_engine.compute_prediction_score(feature_df)
    classified_df = ai_engine.classify_sector_state(scored_df)
    emerging_df = ai_engine.predict_future_sectors(classified_df)
    leaders_df = ai_engine.predict_leader_stocks(stock_master_df)

    st.markdown("### Emerging Sectors")
    if emerging_df.empty:
        st.info("예측 가능한 섹터가 없습니다.")
    else:
        st.dataframe(
            emerging_df[["sector", "prediction_score", "sector_state_ai", "sector_power", "flow_momentum", "news_sentiment", "news_volume"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "sector": st.column_config.TextColumn("Sector"),
                "prediction_score": st.column_config.NumberColumn("Prediction Score", format="%.2f"),
                "sector_state_ai": st.column_config.TextColumn("State"),
                "sector_power": st.column_config.NumberColumn("Sector Power", format="%.1f"),
                "flow_momentum": st.column_config.NumberColumn("Flow Momentum", format="%.2f"),
                "news_sentiment": st.column_config.NumberColumn("News Sentiment", format="%.1f"),
                "news_volume": st.column_config.NumberColumn("News Volume", format="%.0f"),
            },
        )

    st.divider()
    st.markdown("### Sector Prediction Heatmap")
    if classified_df.empty:
        st.caption("표시할 히트맵 데이터가 없습니다.")
    else:
        heat_df = classified_df.sort_values("prediction_score", ascending=False)
        heatmap = go.Figure(
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
        heatmap.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(heatmap, use_container_width=True)

    st.divider()
    st.markdown("### Leader Stocks")
    if leaders_df.empty:
        st.info("조건을 만족하는 리더 종목이 없습니다.")
    else:
        leaders_df = add_naver_link_column(leaders_df, code_col="code", link_col="naver_stock_url")
        st.dataframe(
            leaders_df[["sector", "stock", "momentum_score", "money_flow_score", "financial_score", "leader_strength", "naver_stock_url"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "sector": st.column_config.TextColumn("Sector"),
                "stock": st.column_config.TextColumn("Leader Stock"),
                "momentum_score": st.column_config.NumberColumn("Momentum", format="%.1f"),
                "money_flow_score": st.column_config.NumberColumn("Money Flow", format="%.1f"),
                "financial_score": st.column_config.NumberColumn("Financial", format="%.1f"),
                "leader_strength": st.column_config.NumberColumn("Leader Strength", format="%.1f"),
                "naver_stock_url": st.column_config.LinkColumn("네이버 증권", display_text="바로가기"),
            },
        )

    st.divider()
    st.markdown("### Sector Rotation Radar")
    if classified_df.empty:
        st.caption("레이더 차트 데이터가 없습니다.")
    else:
        radar_target = classified_df.sort_values("prediction_score", ascending=False).head(5).copy()
        radar = go.Figure()
        categories = ["flow_momentum", "rank_change", "leader_strength", "sector_power", "news_sentiment", "prediction_score"]
        for _, row in radar_target.iterrows():
            values = [
                float(pd.to_numeric(row.get("flow_momentum", 0.0), errors="coerce")),
                float(pd.to_numeric(row.get("rank_change", 0.0), errors="coerce") * 10.0),
                float(pd.to_numeric(row.get("leader_strength", 0.0), errors="coerce")),
                float(pd.to_numeric(row.get("sector_power", 0.0), errors="coerce")),
                float(pd.to_numeric(row.get("news_sentiment", 0.0), errors="coerce")),
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
            height=500,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(radar, use_container_width=True)
