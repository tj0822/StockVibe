from __future__ import annotations

import pandas as pd
import streamlit as st

from app.adaptive_weights import DEFAULT_WEIGHTS, load_adaptive_weights, save_adaptive_weights, suggest_adaptive_weights
from app.data import load_stock_data
from app.learning_engine import (
    evaluate_feature_importance,
    evaluate_regime_performance,
    evaluate_sector_performance,
    evaluate_strategy_performance,
)
from app.learning_rules import MINIMUM_SAMPLE_COUNT, allows_auto_update
from app.outcome_tracker import load_decision_history, update_decision_outcomes
from app.ui_components import render_kpi_row


def render_learning_page():
    st.title("🧠 Learning Engine")
    st.caption("과거 투자 판단의 성과를 추적하고, 어떤 특징이 잘 작동했는지 분석합니다.")

    decision_df = load_decision_history()
    if decision_df.empty:
        st.info("저장된 투자 판단 이력이 없습니다.")
        return

    price_df = load_stock_data("data")
    learning_df = update_decision_outcomes(decision_df, price_df)

    strategy_perf = evaluate_strategy_performance(learning_df)
    sector_perf = evaluate_sector_performance(learning_df)
    regime_perf = evaluate_regime_performance(learning_df)
    feature_perf = evaluate_feature_importance(learning_df)

    valid_20d = learning_df[pd.to_numeric(learning_df.get("return_20d"), errors="coerce").notna()].copy()
    decision_count = int(len(learning_df))
    hit_rate_20d = float(pd.to_numeric(valid_20d.get("hit_20d"), errors="coerce").mean() * 100.0) if not valid_20d.empty else 0.0
    avg_return_20d = float(pd.to_numeric(valid_20d.get("return_20d"), errors="coerce").mean()) if not valid_20d.empty else 0.0
    avg_drawdown_20d = float(pd.to_numeric(valid_20d.get("max_drawdown_20d"), errors="coerce").mean()) if not valid_20d.empty else 0.0

    render_kpi_row([
        {"label": "Decision Count", "value": f"{decision_count:,}"},
        {"label": "Hit Rate 20D", "value": f"{hit_rate_20d:.1f}%"},
        {"label": "Avg Return 20D", "value": f"{avg_return_20d:+.2f}%"},
        {"label": "Avg Drawdown 20D", "value": f"{avg_drawdown_20d:.2f}%"},
    ])

    current_weights = load_adaptive_weights()
    suggested_weights = suggest_adaptive_weights(feature_perf, previous_weights=current_weights)
    safe_to_auto = allows_auto_update(hit_rate_20d / 100.0, avg_drawdown_20d, baseline_drawdown_20d=avg_drawdown_20d)

    st.caption(f"학습 최소 표본 수 기준: {MINIMUM_SAMPLE_COUNT} | 자동 업데이트 허용 상태: {'YES' if safe_to_auto else 'NO'}")

    tabs = st.tabs([
        "📊 Summary",
        "🧩 Strategy",
        "🏭 Sector",
        "🌐 Regime",
        "🧪 Features",
        "⚖️ Weights",
    ])

    with tabs[0]:
        st.dataframe(learning_df.sort_values("decision_date", ascending=False), hide_index=True, use_container_width=True)

    with tabs[1]:
        if strategy_perf.empty:
            st.info("전략 성과 데이터가 없습니다.")
        else:
            st.dataframe(strategy_perf, hide_index=True, use_container_width=True)

    with tabs[2]:
        if sector_perf.empty:
            st.info("섹터 성과 데이터가 없습니다.")
        else:
            st.dataframe(sector_perf, hide_index=True, use_container_width=True)

    with tabs[3]:
        if regime_perf.empty:
            st.info("시장 국면 성과 데이터가 없습니다.")
        else:
            st.dataframe(regime_perf, hide_index=True, use_container_width=True)

    with tabs[4]:
        if feature_perf.empty:
            st.info("특징 중요도 데이터가 없습니다.")
        else:
            display = feature_perf.copy()
            display["bucket_summary"] = display["bucket_summary"].apply(lambda x: str(x))
            st.dataframe(display, hide_index=True, use_container_width=True)

    with tabs[5]:
        weights_df = pd.DataFrame(
            {
                "feature": list(DEFAULT_WEIGHTS.keys()),
                "current_weight": [current_weights.get(key, DEFAULT_WEIGHTS[key]) for key in DEFAULT_WEIGHTS],
                "suggested_weight": [suggested_weights.get(key, DEFAULT_WEIGHTS[key]) for key in DEFAULT_WEIGHTS],
                "delta": [suggested_weights.get(key, DEFAULT_WEIGHTS[key]) - current_weights.get(key, DEFAULT_WEIGHTS[key]) for key in DEFAULT_WEIGHTS],
            }
        )
        st.dataframe(weights_df, hide_index=True, use_container_width=True)

        if st.button("제안 가중치 저장", use_container_width=True):
            save_adaptive_weights(suggested_weights)
            st.success("제안 가중치를 data/adaptive_weights.json에 저장했습니다.")
