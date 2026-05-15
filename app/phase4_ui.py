"""
Phase 4 통합 페이지 - 모든 신규 기능을 담은 메뉴 시스템
"""
import streamlit as st
import pandas as pd
import os
import traceback
from datetime import datetime
import plotly.graph_objects as go

# Phase 4 모듈 import
from app.settings import UserSettings
from app.data_pipeline import build_stock_master_df
from app.strategies.registry import list_strategies
from app.strategies.registry_store import (
    demote_from_production,
    load_registry,
    promote_to_production,
    set_enabled,
)
from app.ui_components import render_kpi_row, render_header
from app.ui_utils import make_naver_stock_url

# 기존 모듈
from crawling_kospi import CrawlingKospi
from app.data import load_stock_data, get_data_status
from app.ui import run_app

# ===== 유틸리티 함수들 =====

def convert_to_bool(value):
    """문자열이나 다른 타입의 값을 boolean으로 변환"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)

# ===== 캐싱 함수들 (성능 최적화) =====

@st.cache_data(ttl=3600)  # 1시간 캐싱
def load_kospi_name_map():
    """종목명 매핑 로드 (캐싱)"""
    import os
    kospi_list_file = "data/kospi_list.pkl"
    name_map = {}
    
    if os.path.exists(kospi_list_file):
        try:
            kospi_list_df = pd.read_pickle(kospi_list_file)
            # 최신 종목명 사용
            name_map = kospi_list_df.drop_duplicates(subset=['code'], keep='last')[['code', 'name']].set_index('code')['name'].to_dict()
        except Exception as e:
            pass
    
    return name_map

# =================================

def _build_default_analysis_signals(df: pd.DataFrame, kospi: pd.DataFrame) -> pd.DataFrame:
    """종목/섹터 분석 페이지 공통 기본 시그널 생성"""
    from app.signals import build_signals

    signals = build_signals(
        df,
        10,  # turnover_window
        3.0,  # turnover_multiplier
        20,
        5.0,
        20,
        2.0,
        20,
        2.0,
        ["Turnover Spike"],
        "ANY",
    )
    return signals.merge(kospi, on="code", how="left")

def _get_default_analysis_params() -> dict:
    """종목/섹터 분석 페이지 공통 기본 파라미터"""
    return {
        "data_dir": "data",
        "turnover_window": 10,
        "turnover_multiplier": 3.0,
    }

def render_sidebar_menu() -> tuple[str, str]:
    """사이드바 메뉴"""
    st.sidebar.title("📊 StockVibe Pro")
    st.sidebar.markdown("---")
    
    selected = st.sidebar.radio("메뉴", ["🏠 메인 대시보드", "📈 종목분석", "⚙️ 설정"])

    page_map = {
        "🏠 메인 대시보드": "main",
        "📈 종목분석": "analysis",
        "⚙️ 설정": "settings",
    }
    selected_page = page_map[selected]

    main_tabs = ["📊 시그널", "🎯 시뮬레이션", "⚙️ 최적화", "🔄 데이터"]
    selected_main_tab = st.session_state.get("active_tab", "📊 시그널")

    if selected_page == "main":
        st.sidebar.caption("메인 대시보드 탭")
        selected_main_tab = st.sidebar.radio(
            "",
            main_tabs,
            index=main_tabs.index(selected_main_tab) if selected_main_tab in main_tabs else 0,
            key="main_dashboard_tab",
            label_visibility="collapsed",
        )
        st.session_state.active_tab = selected_main_tab
    
    st.sidebar.markdown("---")
    
    return selected_page, selected_main_tab

def page_settings():
    """설정 페이지"""
    st.title("⚙️ 설정")
    
    settings_mgr = UserSettings()
    current_settings = settings_mgr.load_settings()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 화면 설정", "📊 기본값 설정", "🔧 고급 설정", "🧠 전략 관리"])
    
    with tab1:
        st.subheader("화면 설정")
        
        theme = st.selectbox("테마", ["light", "dark", "blue"], 
                            index=["light", "dark", "blue"].index(current_settings.get('theme', 'light')))
        
        # chart_height 값을 정수로 변환
        chart_height_value = current_settings.get('chart_height', 600)
        if isinstance(chart_height_value, str):
            try:
                chart_height_value = int(chart_height_value)
            except (ValueError, TypeError):
                chart_height_value = 600
        
        chart_height = st.slider("차트 높이", 400, 800, chart_height_value, 50)
        
        # boolean 값 제대로 변환
        show_volume_value = convert_to_bool(current_settings.get('show_volume', True))
        show_volume = st.checkbox("거래량 표시", value=show_volume_value)
        
        if st.button("저장"):
            settings_mgr.set('theme', theme)
            settings_mgr.set('chart_height', chart_height)
            settings_mgr.set('show_volume', show_volume)
            st.success("설정이 저장되었습니다!")
    
    with tab2:
        st.subheader("기본값 설정")
        
        default_period = st.selectbox("기본 조회 기간", 
                                     ["1mo", "3mo", "6mo", "1y", "3y"],
                                     index=["1mo", "3mo", "6mo", "1y", "3y"].index(current_settings.get('default_period', '1y')))
        
        # boolean 값 제대로 변환
        auto_expand_value = convert_to_bool(current_settings.get('auto_expand_top', True))
        auto_expand = st.checkbox("1위 종목 자동 펼치기", 
                                 value=auto_expand_value)

        st.markdown("---")
        st.markdown("#### 포트폴리오 정책 임계값")
        concentration_warning_pct = st.number_input(
            "집중 경고 비중(%)",
            min_value=10.0,
            max_value=90.0,
            value=float(current_settings.get('concentration_warning_pct', 35.0)),
            step=1.0,
        )
        concentration_critical_pct = st.number_input(
            "집중 위험 비중(%)",
            min_value=10.0,
            max_value=95.0,
            value=float(current_settings.get('concentration_critical_pct', 50.0)),
            step=1.0,
        )
        compliant_score_threshold = st.number_input(
            "COMPLIANT 점수 기준",
            min_value=0.0,
            max_value=100.0,
            value=float(current_settings.get('compliant_score_threshold', 70.0)),
            step=1.0,
        )
        watch_score_threshold = st.number_input(
            "WATCH 점수 기준",
            min_value=0.0,
            max_value=100.0,
            value=float(current_settings.get('watch_score_threshold', 60.0)),
            step=1.0,
        )
        if st.button("저장", key="save_defaults"):
            settings_mgr.set('default_period', default_period)
            settings_mgr.set('auto_expand_top', auto_expand)
            settings_mgr.set('concentration_warning_pct', concentration_warning_pct)
            settings_mgr.set('concentration_critical_pct', concentration_critical_pct)
            settings_mgr.set('compliant_score_threshold', compliant_score_threshold)
            settings_mgr.set('watch_score_threshold', watch_score_threshold)
            st.success("설정이 저장되었습니다!")
    
    with tab3:
        st.subheader("고급 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 기본값으로 초기화"):
                settings_mgr.reset_to_default()
                st.success("설정이 초기화되었습니다!")
                st.rerun()
        
        with col2:
            st.json(current_settings)

    with tab4:
        st.subheader("전략 관리")
        st.caption("Enabled/Validated/Production 상태를 관리합니다. Production 승격은 검증 통과 전략만 가능합니다.")

        strategy_specs = list_strategies()
        registry = load_registry()

        if not strategy_specs:
            st.info("등록된 전략이 없습니다. app/strategies/impl를 확인하세요.")
        else:
            for spec in strategy_specs:
                state = registry.get(spec.strategy_id, {})
                enabled = bool(state.get("enabled", True))
                validated = bool(state.get("validated", False))
                in_production = bool(state.get("in_production", False))

                with st.container(border=True):
                    st.markdown(f"**{spec.name}** · `{spec.strategy_id}` · v{spec.version}")
                    st.caption(spec.description)

                    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])
                    with c1:
                        enabled_new = st.toggle("Enabled", value=enabled, key=f"strategy_enabled_{spec.strategy_id}")
                        if enabled_new != enabled:
                            set_enabled(spec.strategy_id, enabled_new)
                            st.rerun()
                    with c2:
                        st.metric("Validated", "✅" if validated else "❌")
                    with c3:
                        st.metric("Production", "✅" if in_production else "❌")
                    with c4:
                        if in_production:
                            if st.button("Demote", key=f"demote_{spec.strategy_id}", use_container_width=True):
                                demote_from_production(spec.strategy_id)
                                st.rerun()
                        else:
                            if st.button(
                                "Promote",
                                key=f"promote_{spec.strategy_id}",
                                disabled=not validated,
                                use_container_width=True,
                            ):
                                try:
                                    promote_to_production(spec.strategy_id)
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"승격 실패: {exc}")

                    last_validation = state.get("last_validation") if isinstance(state, dict) else None
                    if isinstance(last_validation, dict) and last_validation:
                        run_ts = last_validation.get("run_ts", "-")
                        universe = last_validation.get("universe", "-")
                        date_range = last_validation.get("date_range", "-")
                        st.caption(f"Last validation: {run_ts} · {universe} · {date_range}")
                        overall = last_validation.get("overall", last_validation.get("metrics", {}))
                        if isinstance(overall, dict) and overall:
                            m1, m2, m3, m4 = st.columns(4)
                            with m1:
                                st.metric("CAGR", f"{overall.get('cagr', 0.0):.2%}")
                            with m2:
                                st.metric("MDD", f"{overall.get('max_drawdown', 0.0):.2%}")
                            with m3:
                                st.metric("WinRate", f"{overall.get('win_rate', 0.0):.2%}")
                            with m4:
                                st.metric("Trades", int(overall.get('trades_count', 0)))

                        by_regime = last_validation.get("by_regime", {}) if isinstance(last_validation, dict) else {}
                        if isinstance(by_regime, dict) and by_regime:
                            strong_bull = False
                            weak_bear = False
                            bull_metrics = by_regime.get("BULL", {}) if isinstance(by_regime.get("BULL", {}), dict) else {}
                            bear_metrics = by_regime.get("BEAR", {}) if isinstance(by_regime.get("BEAR", {}), dict) else {}

                            if bull_metrics:
                                strong_bull = (
                                    float(bull_metrics.get("cagr", 0.0)) > 0
                                    and float(bull_metrics.get("win_rate", 0.0)) >= 0.5
                                )
                            if bear_metrics:
                                weak_bear = (
                                    float(bear_metrics.get("cagr", 0.0)) < 0
                                    or float(bear_metrics.get("max_drawdown", 0.0)) > 0.20
                                )

                            badge_col1, badge_col2 = st.columns(2)
                            with badge_col1:
                                if strong_bull:
                                    st.success("Strong in BULL")
                                else:
                                    st.info("Neutral in BULL")
                            with badge_col2:
                                if weak_bear:
                                    st.error("Weak in BEAR")
                                else:
                                    st.success("Resilient in BEAR")

                            with st.expander("Regime Validation Summary", expanded=False):
                                regime_rows = []
                                for regime in ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]:
                                    rm = by_regime.get(regime, {}) if isinstance(by_regime.get(regime, {}), dict) else {}
                                    regime_rows.append(
                                        {
                                            "regime": regime,
                                            "cagr": float(rm.get("cagr", 0.0)),
                                            "max_drawdown": float(rm.get("max_drawdown", 0.0)),
                                            "win_rate": float(rm.get("win_rate", 0.0)),
                                            "trades_count": int(rm.get("trades_count", 0)),
                                            "avg_return": float(rm.get("avg_return", 0.0)),
                                            "profit_loss_ratio": float(rm.get("profit_loss_ratio", 0.0)),
                                        }
                                    )
                                regime_df = pd.DataFrame(regime_rows)
                                st.dataframe(
                                    regime_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "cagr": st.column_config.NumberColumn("CAGR", format="%.2f%%"),
                                        "max_drawdown": st.column_config.NumberColumn("MDD", format="%.2f%%"),
                                        "win_rate": st.column_config.NumberColumn("WinRate", format="%.2f%%"),
                                        "avg_return": st.column_config.NumberColumn("AvgReturn", format="%.2f"),
                                        "profit_loss_ratio": st.column_config.NumberColumn("P/L", format="%.2f"),
                                    },
                                )

def run_phase4_app():
    """Phase 4 앱 실행"""
    if "last_run_ts" not in st.session_state:
        st.session_state.last_run_ts = "-"
    if "ai_cache_hit" not in st.session_state:
        st.session_state.ai_cache_hit = "UNKNOWN"
    
    # 사이드바 메뉴
    page, selected_main_tab = render_sidebar_menu()
    
    if 'selected_menu' in st.session_state:
        del st.session_state.selected_menu

    if page != "main":
        selected_main_tab = st.session_state.get("active_tab", "📊 시그널")

    page_name_map = {
        "main": f"메인 대시보드 · {selected_main_tab}",
        "analysis": "종목분석",
        "settings": "설정",
    }

    if page != "main":
        data_status = get_data_status("data")
        cache_state = st.session_state.get("ai_cache_hit", "UNKNOWN")
        render_header(
            page_name_map.get(page, "StockVibe"),
            [
                {"label": f"Price {data_status.get('price_ts', '-')}", "type": "info"},
                {"label": f"Finance {data_status.get('finance_ts', '-')}", "type": "info"},
                {"label": f"News {data_status.get('news_ts', '-')}", "type": "info"},
                {"label": f"Cache {cache_state}", "type": "success" if cache_state == "HIT" else "warning"},
                {"label": f"Last Run {st.session_state.get('last_run_ts', '-')}", "type": "success"},
            ],
        )
    
    # 페이지 라우팅
    if page == "settings":
        page_settings()
    elif page == "analysis":
        from app.ui import render_stock_analysis_page
        from app.data import load_stock_data, load_kospi_list, load_finance_data

        with st.spinner("📊 데이터 로딩 중..."):
            df = load_stock_data("data")
            kospi = load_kospi_list("data")
            finance_df = load_finance_data("data")

        with st.spinner("🔍 시그널 분석 중..."):
            signals = _build_default_analysis_signals(df, kospi)

        params = _get_default_analysis_params()
        render_stock_analysis_page(df, signals, finance_df, params)
    else:
        run_app(selected_main_tab)
