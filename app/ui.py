import json
import hashlib
import textwrap
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from crawling_kospi import CrawlingKospi
from naver_news_crawler import NaverNewsCrawler
from kakao_message import KakaoMessageSender
from optimizer import BacktestOptimizer, get_period_dates
from sentiment_analyzer import SentimentAnalyzer
from stock_ontology import build_stock_ontology, StockOntology

from .data import DATA_DIR_DEFAULT, load_kospi_index, load_kospi_list, load_stock_data, load_finance_data
from .signals import build_signals
from .strategies.registry import get_strategy, list_strategies
from .strategies.registry_store import get_production_strategy_ids, load_registry
from .strategies.validation import validate_strategy
from .strategies.registry_store import update_validation_result
from .ui_components import render_action_row, render_progress_steps


@st.cache_data(ttl=3600, show_spinner=False)  # 1시간 캐시
def get_cached_ai_prediction(code: str, price_df_hash: int, finance_df_hash: int, news_data_str: str, data_refresh_time: str):
    """AI 예측 캐싱 (데이터 갱신 시 자동 무효화)"""
    from stock_ontology import build_stock_ontology
    import json
    
    # news_data 복원
    news_data = json.loads(news_data_str) if news_data_str else []
    
    # 온톨로지 구축 (캐시됨)
    # Note: price_df와 finance_df는 해시로만 받고 실제로는 전역에서 사용
    # 실제 구현 시에는 별도 처리 필요
    return None  # placeholder


def get_news_data_for_stock(code: str):
    """주식 뉴스 데이터 가져오기 (캐싱)"""
    crawler = NaverNewsCrawler()
    news_df = crawler.get_recent_news(code)
    
    news_data = []
    if not news_df.empty:
        if "링크" in news_df.columns:
            news_df = news_df.rename(columns={"링크": "news_url"})
        
        news_rows = news_df.to_dict(orient="records")
        
        for news_row in news_rows:
            news_data.append({
                'title': news_row.get('제목', ''),
                'body': '',
                'date': news_row.get('날짜', ''),
                'source': news_row.get('출처', '')
            })
    
    return news_data


@st.cache_data(ttl=3600, show_spinner=False)
def load_kospi_stocks_cached() -> dict:
    try:
        crawler = CrawlingKospi()
        return crawler.GetKospi200()
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def load_portfolio_codes_cached() -> set:
    from app.portfolio import PortfolioManager
    portfolio_mgr = PortfolioManager()
    portfolio = portfolio_mgr.load_portfolio()
    return set(portfolio.keys())


def render_sidebar(current_tab: str = "시그널") -> dict:
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 최적 파라미터가 적용되었는지 확인
        if 'optimal_params_applied' in st.session_state and st.session_state.optimal_params_applied:
            st.success("✨ 최적 파라미터 적용됨")
            if st.button("🔄 기본값으로 초기화", key="reset_params"):
                st.session_state.optimal_params_applied = False
                if 'applied_turnover_window' in st.session_state:
                    del st.session_state.applied_turnover_window
                if 'applied_turnover_multiplier' in st.session_state:
                    del st.session_state.applied_turnover_multiplier
                if 'applied_buy_unit' in st.session_state:
                    del st.session_state.applied_buy_unit
                st.rerun()
        
        # 적용된 파라미터 값 사용 (있으면)
        default_window = st.session_state.get('applied_turnover_window', 10)
        default_multiplier = st.session_state.get('applied_turnover_multiplier', 3.0)
        
        with st.expander("📊 거래량 분석", expanded=True):
            turnover_window = st.number_input(
                "분석 기간", 
                min_value=5, max_value=120, value=default_window, step=1, 
                key="turnover_window",
                help="직전 n거래일 평균 거래량 계산"
            )
            turnover_multiplier = st.number_input(
                "급등 기준", 
                min_value=1.0, max_value=20.0, value=default_multiplier, step=0.1, 
                key="turnover_multiplier",
                help="평균 대비 몇 배 이상 급등"
            )
            top_n = st.number_input(
                "표시 종목 수", 
                min_value=1, max_value=200, value=10, step=1,
                help="거래량 급등 상위 N개 종목"
            )
            
            st.markdown("---")

        initial_cash = float(st.session_state.get("sim_initial_cash", 50_000_000))
        max_daily_buys = int(st.session_state.get("sim_max_daily_buys", 2))
        buy_unit = float(st.session_state.get("sim_buy_unit", 2_000_000))

    return {
        "data_dir": DATA_DIR_DEFAULT,
        "turnover_window": int(turnover_window),
        "turnover_multiplier": float(turnover_multiplier),
        "signal_filter": ["BUY", "SELL"],
        "top_n": int(top_n),
        "initial_cash": float(initial_cash),
        "max_daily_buys": int(max_daily_buys),
        "buy_unit": float(buy_unit),
        "add_buy_threshold_pct": -7.0,  # 기본값 (시뮬레이션 탭에서 동적 조정)
        "use_finance_filter": False,
        "per_max": None,
        "pbr_max": None,
        "dvr_min": None,
    }


def apply_signal_filters(signals: pd.DataFrame, signal_filter: list[str]) -> pd.DataFrame:
    if signal_filter:
        return signals[signals["signal"].isin(signal_filter)]
    return signals


@st.cache_data(ttl=3600, show_spinner=False)
def generate_strategy_signals_cached(
    strategy_id: str,
    strategy_version: str,
    price_df: pd.DataFrame,
    market_df: pd.DataFrame,
    params_json: str,
) -> pd.DataFrame:
    strategy = get_strategy(strategy_id)
    params = json.loads(params_json) if params_json else {}

    volume_cols = [c for c in ["date", "code", "volume"] if c in price_df.columns]
    volume_df = price_df[volume_cols].copy() if volume_cols else pd.DataFrame()

    signals = strategy.generate_signals(
        price_df=price_df,
        volume_df=volume_df,
        market_df=market_df,
        params=params,
    )
    if signals is None or signals.empty:
        return pd.DataFrame(columns=["date", "code", "signal", "confidence", "features_json", "strategy_id", "strategy_name", "strategy_version"])

    signals = signals.copy()
    if "strategy_id" not in signals.columns:
        signals["strategy_id"] = strategy.spec.strategy_id
    if "strategy_name" not in signals.columns:
        signals["strategy_name"] = strategy.spec.name
    if "strategy_version" not in signals.columns:
        signals["strategy_version"] = strategy.spec.version
    if "confidence" not in signals.columns:
        signals["confidence"] = 50.0
    if "features_json" not in signals.columns:
        signals["features_json"] = "{}"

    return signals


def _get_runtime_strategy_params(strategy_id: str, params: dict, rolling_days: int | None = None, volume_threshold: float | None = None) -> dict:
    strategy = get_strategy(strategy_id)
    strategy_params = dict(strategy.spec.default_params)

    if strategy_id == "turnover_spike":
        strategy_params.update(
            {
                "turnover_window": int(rolling_days if rolling_days is not None else params.get("turnover_window", 10)),
                "turnover_multiplier": float(volume_threshold if volume_threshold is not None else params.get("turnover_multiplier", 3.0)),
                "combine_mode": "ANY",
                "enabled_algos": ["Turnover Spike"],
                "top_n": int(params.get("top_n", 10)),
            }
        )

    return strategy_params


def select_date(signals: pd.DataFrame) -> pd.Timestamp | None:
    available_dates = signals["date"].dropna().sort_values().unique()
    if len(available_dates) == 0:
        st.warning("조건에 맞는 시그널이 없습니다.")
        return None

    latest_date = available_dates[-1]
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_date = st.date_input(
            "📅 기준 날짜",
            value=latest_date.date(),
            min_value=available_dates[0].date(),
            max_value=latest_date.date(),
        )
    with col2:
        st.metric("최신 데이터", latest_date.strftime("%Y-%m-%d"))
    return pd.to_datetime(selected_date)


def build_latest_table(signals: pd.DataFrame, selected_date: pd.Timestamp, top_n: int, finance_df: pd.DataFrame = None, price_df: pd.DataFrame = None) -> tuple[pd.DataFrame, list[str]]:
    latest = signals[(signals["date"] == selected_date) & (signals["signal"] != "")].copy()
    latest = latest.head(int(top_n))

    # 재무 데이터 병합
    if finance_df is not None and not finance_df.empty:
        # 가장 최근 재무 데이터 가져오기 (날짜가 선택 날짜 이전)
        finance_latest = finance_df[finance_df['date'] <= selected_date].copy()
        finance_latest = finance_latest.sort_values('date').groupby('code').tail(1)
        
        # 필요한 재무 컬럼만 선택
        finance_cols = ['code', 'per', 'pbr', 'eps', 'bps', 'dvr', 'foreigner_ratio']
        finance_latest = finance_latest[[col for col in finance_cols if col in finance_latest.columns]]
        
        # 병합
        latest = latest.merge(finance_latest, on='code', how='left')

    # 다음날 종가 및 등락률 계산
    if price_df is not None and not price_df.empty:
        # 다음 거래일 찾기
        next_date = price_df[price_df['date'] > selected_date]['date'].min()
        
        if pd.notna(next_date):
            # 다음날 데이터 가져오기
            next_day_data = price_df[price_df['date'] == next_date][['code', 'close']].copy()
            next_day_data = next_day_data.rename(columns={'close': 'next_close'})
            
            # 병합
            latest = latest.merge(next_day_data, on='code', how='left')
            
            # 등락률 계산 (당일 종가 대비)
            if 'next_close' in latest.columns and 'close' in latest.columns:
                latest['change_rate'] = ((latest['next_close'] - latest['close']) / latest['close'] * 100).round(2)

    latest["date"] = latest["date"].dt.date
    if "spike_rank" in latest.columns:
        latest["spike_rank"] = latest["spike_rank"].round(0).astype("Int64")
    if "name" in latest.columns and "code" in latest.columns:
        latest["name"] = latest.apply(
            lambda row: (
                f'<a href="https://finance.naver.com/item/news.naver?code={row["code"]}" '
                f'target="_blank">{row["name"]}</a>'
                if pd.notna(row["name"]) and pd.notna(row["code"]) else row["name"]
            ),
            axis=1,
        )

    cols = [
        "date",
        "code",
        "name",
        "open",
        "close",
        "high",
        "low",
        "volume",
    ]
    
    # 다음날 종가와 등락률 추가
    if 'next_close' in latest.columns:
        cols.append("next_close")
    if 'change_rate' in latest.columns:
        cols.append("change_rate")
    
    cols.append("signal")
    
    if "spike_ratio" in latest.columns:
        cols.insert(-1, "spike_ratio")
    
    # 재무 데이터는 별도 표시를 위해 컬럼에서 제외
    
    return latest, cols


def render_table(latest: pd.DataFrame, cols: list[str]) -> None:
    # 포트폴리오 정보 로드
    from app.portfolio import PortfolioManager
    portfolio_mgr = PortfolioManager()
    portfolio = portfolio_mgr.load_portfolio()
    
    # 포트폴리오에서 코드와 종목명 모두 수집
    portfolio_codes = set()
    portfolio_names = set()
    for key, value in portfolio.items():
        portfolio_codes.add(key)  # 키가 코드 또는 종목명
        if 'name' in value:
            portfolio_names.add(value['name'])  # 종목명도 추가
    
    # 포트폴리오 보유 여부 확인 함수
    def is_in_portfolio(row):
        code = row.get('code', '')
        name = row.get('name', '')
        # HTML 태그 제거
        import re
        clean_name = re.sub(r'<[^>]+>', '', str(name))
        
        # 코드 또는 종목명으로 확인
        return '💼' if (code in portfolio_codes or 
                       code in portfolio_names or 
                       clean_name in portfolio_codes or 
                       clean_name in portfolio_names) else ''
    
    # 포트폴리오 보유 여부 컬럼 추가
    latest_copy = latest.copy()
    latest_copy['보유'] = latest_copy.apply(is_in_portfolio, axis=1)
    
    # cols에 '보유' 컬럼 추가 (code 다음에)
    if 'code' in cols:
        code_idx = cols.index('code')
        cols_with_portfolio = cols[:code_idx+1] + ['보유'] + cols[code_idx+1:]
    else:
        cols_with_portfolio = ['보유'] + cols
    
    def color_signal(val: str) -> str:
        if val == "BUY":
            return "color: red; font-weight: 600;"
        if val == "SELL":
            return "color: blue; font-weight: 600;"
        return ""

    format_map = {
        "open": "{:,.0f}",
        "close": "{:,.0f}",
        "high": "{:,.0f}",
        "low": "{:,.0f}",
        "volume": "{:,.0f}",
        "next_close": "{:,.0f}",
        "change_rate": "{:+.2f}%",
        "per": "{:.2f}",
        "pbr": "{:.2f}",
        "dvr": "{:.2f}%",
        "foreigner_ratio": "{:.2f}%",
    }
    if "spike_ratio" in latest_copy.columns:
        format_map["spike_ratio"] = "{:.0%}"

    # 등락률에 색상 적용
    def color_change_rate(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: red; font-weight: 600;"
        elif val < 0:
            return "color: blue; font-weight: 600;"
        return ""

    styled = latest_copy[cols_with_portfolio].style.applymap(color_signal, subset=["signal"]).format(format_map)
    
    if "change_rate" in latest_copy.columns:
        styled = styled.applymap(color_change_rate, subset=["change_rate"])
    
    st.markdown(styled.hide(axis="index").to_html(escape=False), unsafe_allow_html=True)


def render_table_with_finance(
    latest: pd.DataFrame,
    cols: list[str],
    finance_df: pd.DataFrame,
    price_df: pd.DataFrame = None,
    focus_code: str | None = None,
) -> None:
    """재무정보를 포함하여 각 종목별로 개별 펼치기로 표시"""
    
    # 포트폴리오 정보 로드
    from app.portfolio import PortfolioManager
    portfolio_mgr = PortfolioManager()
    portfolio = portfolio_mgr.load_portfolio()
    
    # 포트폴리오에서 코드와 종목명 모두 수집
    portfolio_codes = set()
    portfolio_names = set()
    for key, value in portfolio.items():
        portfolio_codes.add(key)  # 키가 코드 또는 종목명
        if 'name' in value:
            portfolio_names.add(value['name'])  # 종목명도 추가
    
    # 재무정보 컬럼 확인
    finance_cols_available = [col for col in ['per', 'pbr', 'eps', 'bps', 'dvr', 'foreigner_ratio'] 
                              if col in latest.columns]
    has_finance = len(finance_cols_available) > 0
    
    for idx, row in latest.iterrows():
        # HTML 태그 제거한 종목명
        import re
        clean_name = re.sub(r'<[^>]+>', '', str(row.get('name', '')))
        code = row.get('code', '')
        signal = row.get('signal', '')
        
        # 포트폴리오 보유 여부 확인 (코드 또는 종목명으로 확인)
        is_in_portfolio = (code in portfolio_codes or 
                          code in portfolio_names or 
                          clean_name in portfolio_codes or 
                          clean_name in portfolio_names)
        portfolio_badge = "💼 " if is_in_portfolio else ""
        
        # 신호 색상
        signal_color = "🔴" if signal == "BUY" else "🔵" if signal == "SELL" else "⚪"
        
        # 등락률 색상
        change_rate = row.get('change_rate', None)
        if pd.notna(change_rate):
            if change_rate > 0:
                change_emoji = "📈"
                change_text = f"+{change_rate:.2f}%"
                change_color = "red"
            elif change_rate < 0:
                change_emoji = "📉"
                change_text = f"{change_rate:.2f}%"
                change_color = "blue"
            else:
                change_emoji = "➖"
                change_text = "0.00%"
                change_color = "gray"
        else:
            change_emoji = ""
            change_text = "-"
            change_color = "gray"
        
        # 제목 구성
        if pd.notna(change_rate):
            title = f"{portfolio_badge}{signal_color} **{clean_name}** ({code}) | {signal} | 다음날 :{change_color}[{change_text}] {change_emoji}"
        else:
            title = f"{portfolio_badge}{signal_color} **{clean_name}** ({code}) | {signal}"
        
        # 최상위 종목(idx=0) 또는 선택 종목은 자동으로 펼치기
        if focus_code is not None:
            focus_code = str(focus_code).strip()
        # auto_expand를 명시적으로 boolean으로 변환
        auto_expand = bool((idx == 0) or (focus_code and str(code) == focus_code))
        
        with st.expander(title, expanded=auto_expand):
            # 탭으로 정보 구조화
            tab1, tab2, tab3 = st.tabs(["📊 개요", "📈 차트", "📰 뉴스"])
            
            # === 탭 1: 개요 ===
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 📊 가격 정보")
                    price_data = {
                        "시가": f"{row.get('open', 0):,.0f}원",
                        "종가": f"{row.get('close', 0):,.0f}원",
                        "고가": f"{row.get('high', 0):,.0f}원",
                        "저가": f"{row.get('low', 0):,.0f}원",
                        "거래량": f"{row.get('volume', 0):,.0f}",
                    }
                    
                    if pd.notna(row.get('next_close')):
                        price_data["다음날종가"] = f"{row.get('next_close', 0):,.0f}원"
                    
                    if 'spike_ratio' in row and pd.notna(row['spike_ratio']):
                        price_data["거래량급증률"] = f"{row['spike_ratio']:.0%}"
                    
                    for key, value in price_data.items():
                        st.text(f"{key}: {value}")
                
                with col2:
                    if has_finance:
                        st.markdown("##### 💼 재무 정보")
                        finance_data = {}
                        
                        if 'per' in row and pd.notna(row['per']):
                            finance_data["PER"] = f"{row['per']:.2f}"
                        if 'pbr' in row and pd.notna(row['pbr']):
                            finance_data["PBR"] = f"{row['pbr']:.2f}"
                        if 'eps' in row and pd.notna(row['eps']):
                            finance_data["EPS"] = f"{row['eps']:,.0f}원"
                        if 'bps' in row and pd.notna(row['bps']):
                            finance_data["BPS"] = f"{row['bps']:,.0f}원"
                        if 'dvr' in row and pd.notna(row['dvr']):
                            finance_data["배당수익률"] = f"{row['dvr']:.2f}%"
                        if 'foreigner_ratio' in row and pd.notna(row['foreigner_ratio']):
                            finance_data["외국인보유율"] = f"{row['foreigner_ratio']:.2f}%"
                        
                        if finance_data:
                            for key, value in finance_data.items():
                                st.text(f"{key}: {value}")
                        else:
                            st.info("재무 정보가 없습니다.")
                    else:
                        st.markdown("##### 💼 포트폴리오")
                        if st.button(f"💼 {clean_name} 포트폴리오 추가하기", key=f"add_portfolio_{code}_{idx}"):
                            st.session_state.selected_menu = "💼 포트폴리오"
                            st.session_state.portfolio_add_stock = {'code': code, 'name': clean_name}
                            st.rerun()
                
                # 날짜 정보
                if 'date' in row and pd.notna(row['date']):
                    st.caption(f"📅 시그널 발생일: {row['date']}")
                
                # 빠른 액션 버튼
                st.divider()
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    st.markdown(f"[🔗 네이버금융](https://finance.naver.com/item/main.naver?code={code})")
                with col_btn2:
                    st.markdown(f"[📊 증권정보](https://finance.naver.com/item/coinfo.naver?code={code})")
                with col_btn3:
                    st.markdown(f"[📈 투자자별](https://finance.naver.com/item/frgn.naver?code={code})")
            
            # === 탭 2: 차트 ===
            with tab2:
                # 주가 차트 (기존 코드 이동)
                if price_df is not None and not price_df.empty and code:
                    stock_prices = price_df[price_df['code'] == code].copy()
                    if not stock_prices.empty:
                        stock_prices = stock_prices.sort_values('date')
                        
                        # 기간 선택
                        price_period_col1, price_period_col2 = st.columns([1, 3])
                        with price_period_col1:
                            chart_period = st.selectbox(
                                "기간",
                                ["최근 1개월", "최근 3개월", "최근 6개월", "최근 1년", "최근 3년", "전체"],
                                index=2,  # 기본값: 6개월
                                key=f"chart_period_{code}",
                                label_visibility="collapsed"
                            )
                        
                        # 주가 데이터 필터링
                        filtered_prices = stock_prices.copy()
                        if chart_period == "최근 1개월":
                            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=1)
                            filtered_prices = filtered_prices[filtered_prices['date'] >= cutoff_date]
                        elif chart_period == "최근 3개월":
                            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=3)
                            filtered_prices = filtered_prices[filtered_prices['date'] >= cutoff_date]
                        elif chart_period == "최근 6개월":
                            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=6)
                            filtered_prices = filtered_prices[filtered_prices['date'] >= cutoff_date]
                        elif chart_period == "최근 1년":
                            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=1)
                            filtered_prices = filtered_prices[filtered_prices['date'] >= cutoff_date]
                        elif chart_period == "최근 3년":
                            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=3)
                            filtered_prices = filtered_prices[filtered_prices['date'] >= cutoff_date]
                        
                        # 재무 데이터 필터링 (같은 기간)
                        filtered_finance = None
                        if not finance_df.empty and code:
                            stock_finance = finance_df[finance_df['code'] == code].copy()
                            if not stock_finance.empty:
                                stock_finance = stock_finance.sort_values('date')
                                if chart_period != "전체":
                                    stock_finance = stock_finance[stock_finance['date'] >= cutoff_date]
                                if len(stock_finance) > 0:
                                    filtered_finance = stock_finance
                        
                        if len(filtered_prices) > 0:
                            # 탭으로 구분
                            tab1, tab2, tab3 = st.tabs(["💹 주가/거래량", "📊 밸류에이션", "💰 수익성"])
                            
                            with tab1:
                                # Plotly를 사용한 캔들스틱 + 거래량 차트 (이중축)
                                fig = make_subplots(
                                    rows=2, cols=1,
                                    shared_xaxes=True,
                                    vertical_spacing=0.03,
                                    row_heights=[0.7, 0.3],
                                    subplot_titles=('주가', '거래량')
                                )
                                
                                # 캔들스틱 차트
                                fig.add_trace(
                                    go.Candlestick(
                                        x=filtered_prices['date'],
                                        open=filtered_prices['open'],
                                        high=filtered_prices['high'],
                                        low=filtered_prices['low'],
                                        close=filtered_prices['close'],
                                        name='주가',
                                        increasing_line_color='red',
                                        decreasing_line_color='blue'
                                    ),
                                    row=1, col=1
                                )
                                
                                # 거래량 바 차트
                                colors = ['red' if close >= open else 'blue' 
                                         for close, open in zip(filtered_prices['close'], filtered_prices['open'])]
                                
                                fig.add_trace(
                                    go.Bar(
                                        x=filtered_prices['date'],
                                        y=filtered_prices['volume'],
                                        name='거래량',
                                        marker_color=colors,
                                        showlegend=False
                                    ),
                                    row=2, col=1
                                )
                                
                                # 레이아웃 설정
                                fig.update_layout(
                                    height=600,
                                    xaxis_rangeslider_visible=False,
                                    hovermode='x unified',
                                    template='plotly_white',
                                    margin=dict(l=0, r=0, t=40, b=0)
                                )
                                
                                # y축 레이블
                                fig.update_yaxes(title_text="가격 (원)", row=1, col=1)
                                fig.update_yaxes(title_text="거래량", row=2, col=1)
                                
                                # x축 설정
                                fig.update_xaxes(
                                    rangebreaks=[
                                        dict(bounds=["sat", "mon"])  # 주말 제거
                                    ]
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 통계 정보
                                st.divider()
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                with col_stat1:
                                    st.metric("최고가", f"{filtered_prices['high'].max():,.0f}원")
                                with col_stat2:
                                    st.metric("최저가", f"{filtered_prices['low'].min():,.0f}원")
                                with col_stat3:
                                    price_change = ((filtered_prices['close'].iloc[-1] - filtered_prices['close'].iloc[0]) / filtered_prices['close'].iloc[0] * 100)
                                    st.metric("기간 수익률", f"{price_change:+.2f}%")
                                with col_stat4:
                                    avg_volume = filtered_prices['volume'].mean()
                                    st.metric("평균 거래량", f"{avg_volume:,.0f}")
                            
                            with tab2:
                                # 밸류에이션 지표 (PER, PBR)
                                if filtered_finance is not None:
                                    chart_data = filtered_finance.set_index('date')[['per', 'pbr']].dropna(how='all')
                                    if not chart_data.empty:
                                        st.line_chart(chart_data, height=300)
                                        
                                        # 재무 통계
                                        col_f1, col_f2 = st.columns(2)
                                        with col_f1:
                                            if 'per' in chart_data.columns:
                                                st.metric("평균 PER", f"{chart_data['per'].mean():.2f}")
                                        with col_f2:
                                            if 'pbr' in chart_data.columns:
                                                st.metric("평균 PBR", f"{chart_data['pbr'].mean():.2f}")
                                    else:
                                        st.info("밸류에이션 데이터가 없습니다.")
                                else:
                                    st.info("재무 데이터가 없습니다.")
                            
                            with tab3:
                                # 수익성 지표 (EPS, BPS, 배당수익률)
                                if filtered_finance is not None:
                                    chart_cols = []
                                    if 'eps' in filtered_finance.columns:
                                        chart_cols.append('eps')
                                    if 'bps' in filtered_finance.columns:
                                        chart_cols.append('bps')
                                    if 'dvr' in filtered_finance.columns:
                                        chart_cols.append('dvr')
                                    
                                    if chart_cols:
                                        chart_data = filtered_finance.set_index('date')[chart_cols].dropna(how='all')
                                        if not chart_data.empty:
                                            st.line_chart(chart_data, height=300)
                                            
                                            # 수익성 통계
                                            cols = st.columns(len(chart_cols))
                                            for idx, col in enumerate(chart_cols):
                                                with cols[idx]:
                                                    if col == 'eps':
                                                        st.metric("평균 EPS", f"{chart_data['eps'].mean():,.0f}원")
                                                    elif col == 'bps':
                                                        st.metric("평균 BPS", f"{chart_data['bps'].mean():,.0f}원")
                                                    elif col == 'dvr':
                                                        st.metric("평균 배당수익률", f"{chart_data['dvr'].mean():.2f}%")
                                        else:
                                            st.info("수익성 데이터가 없습니다.")
                                    else:
                                        st.info("수익성 데이터가 없습니다.")
                                else:
                                    st.info("재무 데이터가 없습니다.")
                        else:
                            st.info(f"{chart_period} 데이터가 없습니다.")
                else:
                    st.info("주가 데이터가 없습니다.")
            
                # === 탭 3: 뉴스 ===
            with tab3:
                # 뉴스 섹션 (기존 코드 이동)
                with st.spinner("뉴스 불러오는 중..."):
                    from naver_news_crawler import NaverNewsCrawler
                    crawler = NaverNewsCrawler()
                    news_df = crawler.get_recent_news(code)
                
                if news_df.empty:
                    st.info("해당 종목의 최근 뉴스가 없습니다.")
                else:
                    if "링크" in news_df.columns:
                        news_df = news_df.rename(columns={"링크": "news_url"})
                    
                    news_rows = news_df.to_dict(orient="records")
                    urls = [row.get("news_url", "") for row in news_rows]
                    bodies: dict[str, str] = {}
                    
                    if urls:
                        max_workers = min(8, len(urls))
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            for url, body in zip(urls, executor.map(crawler.get_news_body, urls)):
                                bodies[url] = body
                    
                    # HTML 카드 방식으로 뉴스 표시
                    for idx, row in enumerate(news_rows):
                        title = row.get("제목", "(제목 없음)")
                        source = row.get("출처", "")
                        date = row.get("날짜", "")
                        news_url = row.get("news_url", "")
                        body = bodies.get(news_url, "") if news_url else ""
                        
                        # HTML 카드 생성
                        card_html = f"""
                        <div style="
                            border: 1px solid #e0e0e0;
                            border-radius: 8px;
                            padding: 16px;
                            margin-bottom: 16px;
                            background-color: #f9f9f9;
                        ">
                            <div style="
                                display: flex;
                                justify-content: space-between;
                                align-items: flex-start;
                                margin-bottom: 8px;
                            ">
                                <h4 style="margin: 0; color: #1f1f1f; flex: 1;">
                                    {idx + 1}. {title}
                                </h4>
                            </div>
                            <div style="
                                display: flex;
                                gap: 12px;
                                margin-bottom: 12px;
                                font-size: 0.9em;
                                color: #666;
                            ">
                                <span>📰 {source}</span>
                                <span>📅 {date}</span>
                                <a href="{news_url}" target="_blank" style="color: #0066cc; text-decoration: none;">
                                    🔗 원문보기
                                </a>
                            </div>
                            <div style="
                                color: #333;
                                line-height: 1.6;
                                white-space: pre-wrap;
                                word-wrap: break-word;
                            ">
                                {body if body else "본문을 가져올 수 없습니다."}
                            </div>
                        </div>
                        """
                        
                        st.markdown(card_html, unsafe_allow_html=True)


def build_forward_return_series_by_stock(
    price_df: pd.DataFrame,
    signal_df: pd.DataFrame,
) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()

    data = price_df[["code", "date", "close"]].copy()
    data = data.sort_values(["code", "date"])

    base = signal_df.copy()
    base = base.rename(columns={"date": "signal_date", "close": "base_close"})

    merged = base.merge(data, on="code", how="left", suffixes=("", "_future"))
    merged = merged[merged["date"] >= merged["signal_date"]]
    merged["return_pct"] = (merged["close"] / merged["base_close"] - 1) * 100

    merged.loc[merged["signal"] == "SELL", "return_pct"] = -merged.loc[merged["signal"] == "SELL", "return_pct"]
    merged = merged.dropna(subset=["date", "return_pct"])

    def _build_label(row: pd.Series) -> str:
        name = row.get("name")
        code = row.get("code")
        base_label = f"{name} ({code})" if pd.notna(name) else str(code)
        return base_label

    merged["label"] = merged.apply(_build_label, axis=1)
    merged = merged.rename(columns={"date": "target_date"})
    return merged[["target_date", "return_pct", "label", "signal"]].rename(
        columns={"target_date": "date"}
    )


def render_backtest_curve(
    equity_df: pd.DataFrame,
    kospi_index: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> None:
    if equity_df.empty:
        st.info("수익률 곡선을 만들 데이터가 없습니다.")
        return

    st.subheader("추천 시점 이후 수익률 추이")

    def _series_to_tv(data: pd.DataFrame, value_col: str) -> list[dict]:
        if data.empty:
            return []
        temp = data.copy()
        temp = temp.dropna(subset=["date", value_col]).sort_values("date")
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp = temp.dropna(subset=["date"])
        return [
            {"time": d.strftime("%Y-%m-%d"), "value": float(v)}
            for d, v in zip(temp["date"], temp[value_col])
        ]

    eq = equity_df.copy()
    eq = eq[eq["date"] >= selected_date].copy()
    if eq.empty:
        st.info("선택한 날짜 이후 데이터가 없습니다.")
        return
    base_equity = eq["equity"].iloc[0]
    eq["return_pct"] = (eq["equity"] / base_equity - 1) * 100

    tv_series = []
    data_points = _series_to_tv(eq, "return_pct")
    if data_points:
        tv_series.append(
            {
                "name": "전략 수익률",
                "color": "#E74C3C",
                "lineWidth": 2,
                "data": data_points,
            }
        )

    if not kospi_index.empty:
        kospi = kospi_index.copy()
        kospi = kospi[kospi["date"] <= selected_date].sort_values("date")
        if not kospi.empty:
            base_index = kospi["index"].iloc[-1]
            kospi_curve = kospi_index[kospi_index["date"].isin(eq["date"])].copy()
            if not kospi_curve.empty:
                kospi_curve["return_pct"] = (kospi_curve["index"] / base_index - 1) * 100
                data_points = _series_to_tv(kospi_curve, "return_pct")
                if data_points:
                    tv_series.append(
                        {
                            "name": "KOSPI 수익률",
                            "color": "#7F8C8D",
                            "lineWidth": 2,
                            "lineStyle": 2,
                            "data": data_points,
                        }
                    )

    if not tv_series:
        st.info("표시할 차트 데이터가 없습니다.")
        return

    tv_data = json.dumps(tv_series, ensure_ascii=False)
    chart_html = f"""
    <div id="tv_chart" style="width:100%; height:420px;"></div>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script>
      const chart = LightweightCharts.createChart(document.getElementById('tv_chart'), {{
        layout: {{ textColor: '#333', background: {{ type: 'solid', color: '#ffffff' }} }},
        rightPriceScale: {{ borderColor: '#ccc' }},
        timeScale: {{ borderColor: '#ccc', timeVisible: false, secondsVisible: false }},
        grid: {{ vertLines: {{ color: '#f0f0f0' }}, horzLines: {{ color: '#f0f0f0' }} }},
      }});

      const seriesList = {tv_data};
      seriesList.forEach(s => {{
        const line = chart.addLineSeries({{
          color: s.color,
          lineWidth: s.lineWidth || 2,
          lineStyle: s.lineStyle || 0,
        }});
        line.setData(s.data);
      }});

      chart.timeScale().fitContent();
    </script>
    """
    components.html(chart_html, height=440, scrolling=False)


def run_turnover_strategy_backtest(
    price_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    kospi_index: pd.DataFrame,
    start_date: pd.Timestamp,
    top_n: int = 2,
    initial_cash: float = 0.0,
    max_daily_buys: int = 2,
    buy_unit: float = 2_000_000,
    kospi_bullish_only: bool = False,
    kospi_bullish_lookback_months: int = 6,
    add_buy_threshold_pct: float = -7.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    open_pivot: pd.DataFrame | None = None,
    close_pivot: pd.DataFrame | None = None,
    kospi_bullish_dates: set | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if open_pivot is None or close_pivot is None:
        # 데이터 전처리 및 최적화
        price_data = price_df[["date", "code", "open", "close"]].copy()
        price_data["date"] = pd.to_datetime(price_data["date"], errors="coerce").dt.normalize()
        price_data = price_data.dropna(subset=["date", "code", "open", "close"])
        
        # 중복 제거 (같은 날짜, 같은 종목의 경우 마지막 값 사용)
        price_data = price_data.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last")

        # 빠른 조회를 위한 pivot 테이블 생성
        open_pivot = price_data.pivot(index="date", columns="code", values="open")
        close_pivot = price_data.pivot(index="date", columns="code", values="close")
    
    signal_df = signal_df.copy()
    signal_df["date"] = pd.to_datetime(signal_df["date"], errors="coerce").dt.normalize()

    # 매수 시그널 사전 정리
    base_buy = signal_df[signal_df["signal"] == "BUY"]
    if "spike_ratio" in base_buy.columns:
        buy_candidates = base_buy.sort_values(["date", "spike_ratio"], ascending=[True, False])
    elif "momentum" in base_buy.columns:
        buy_candidates = base_buy.sort_values(["date", "momentum"], ascending=[True, False])
    else:
        buy_candidates = base_buy.sort_values(["date"], ascending=[True])
    buy_by_date = buy_candidates.groupby("date")["code"].apply(lambda s: list(s.head(top_n))).to_dict()

    # 매도 시그널 세트 생성
    sell_signals = signal_df[signal_df["signal"] == "SELL"][["date", "code"]].copy()
    sell_by_date = {}
    for _, row in sell_signals.iterrows():
        date = row["date"]
        code = row["code"]
        if date not in sell_by_date:
            sell_by_date[date] = set()
        sell_by_date[date].add(code)

    start_date = pd.to_datetime(start_date, errors="coerce").normalize()
    dates = [d for d in open_pivot.index if d >= start_date]
    
    # 코스피 상승기간 날짜 세트 생성 (n개월 추세 수익률 기준)
    if kospi_bullish_dates is None:
        kospi_bullish_dates = set()
        if kospi_bullish_only and not kospi_index.empty:
            ki = kospi_index.copy()
            ki["date"] = pd.to_datetime(ki["date"], errors="coerce").dt.normalize()
            ki = ki.dropna(subset=["date", "index"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
            if not ki.empty and len(ki) >= 2:
                step_days = ki["date"].diff().dt.days.median()
                step_days = int(step_days) if pd.notna(step_days) and step_days > 0 else 1
                lookback_points = max(1, int(round((int(kospi_bullish_lookback_months) * 30) / step_days)))
                ki["trend_return"] = ki["index"].pct_change(periods=lookback_points)
                ki["change"] = ki["index"].diff()
                ki["is_bullish"] = ki["trend_return"] >= 0
                ki["is_bullish"] = ki["is_bullish"].where(ki["trend_return"].notna(), ki["change"] >= 0)
                kospi_bullish_dates = set(ki[ki["is_bullish"]]["date"].tolist())
    
    positions: dict[str, dict] = {}
    cash = float(initial_cash)
    equity_records = []
    trades = []
    
    # 대기 주문
    pending_buys: dict[pd.Timestamp, list] = {}
    pending_sells: dict[pd.Timestamp, set] = {}
    pending_stop_loss: dict[pd.Timestamp, set] = {}  # 손절 대기

    # 이름 매핑
    name_map = {}
    if "name" in signal_df.columns:
        name_map = signal_df[["code", "name"]].dropna().drop_duplicates().set_index("code")["name"].to_dict()

    def _buy(code: str, date: pd.Timestamp, price: float, max_amount: float, step: int) -> None:
        nonlocal cash
        if cash <= 0 or price <= 0:
            return
        spend = min(cash, max_amount)
        effective_price = price * (1 + slippage_rate)
        max_cash_per_share = effective_price * (1 + fee_rate)
        shares = int(spend // max_cash_per_share) if max_cash_per_share > 0 else 0
        if shares == 0:
            return
        gross_amount = shares * effective_price
        fee_amount = gross_amount * fee_rate
        actual_amount = gross_amount + fee_amount
        if code in positions:
            pos = positions[code]
            total_cost = pos["avg_cost"] * pos["shares"] + actual_amount
            total_shares = pos["shares"] + shares
            pos.update({
                "shares": total_shares,
                "avg_cost": total_cost / total_shares if total_shares > 0 else 0,
                "step": step,
            })
        else:
            positions[code] = {"shares": shares, "avg_cost": actual_amount / shares, "step": step}
        cash -= actual_amount
        trades.append({
            "date": date,
            "code": code,
            "name": name_map.get(code, ""),
            "action": "BUY",
            "amount": actual_amount,
            "price": effective_price,
            "shares": shares,
            "step": step,
            "return_pct": None,
        })

    def _sell(code: str, date: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal cash
        if code not in positions:
            return
        pos = positions.pop(code)
        effective_price = price * (1 - slippage_rate)
        proceeds = pos["shares"] * effective_price
        fee_amount = proceeds * fee_rate
        tax_amount = proceeds * sell_tax_rate
        net_proceeds = proceeds - fee_amount - tax_amount
        cash += net_proceeds
        pnl = net_proceeds - (pos["avg_cost"] * pos["shares"])
        trades.append({
            "date": date,
            "code": code,
            "name": name_map.get(code, ""),
            "action": "SELL",
            "amount": net_proceeds,
            "price": effective_price,
            "shares": pos["shares"],
            "pnl": pnl,
            "buy_price": pos["avg_cost"],
            "return_pct": ((net_proceeds / (pos["avg_cost"] * pos["shares"])) - 1) * 100 if pos["avg_cost"] > 0 else 0,
            "reason": reason,
        })

    for i, date in enumerate(dates):
        # 대기 매수 실행
        if date in pending_buys:
            for code, max_amount, step in pending_buys[date]:
                try:
                    open_price = open_pivot.loc[date, code]
                    if pd.notna(open_price):
                        _buy(code, date, open_price, max_amount, step)
                except (KeyError, IndexError):
                    pass
            del pending_buys[date]
        
        # 대기 손절 실행 (시그널 매도보다 우선)
        if date in pending_stop_loss:
            for code in pending_stop_loss[date]:
                try:
                    open_price = open_pivot.loc[date, code]
                    if pd.notna(open_price):
                        _sell(code, date, open_price, "STOP_LOSS")
                except (KeyError, IndexError):
                    pass
            del pending_stop_loss[date]
        
        # 대기 매도 실행
        if date in pending_sells:
            for code in pending_sells[date]:
                try:
                    open_price = open_pivot.loc[date, code]
                    if pd.notna(open_price):
                        _sell(code, date, open_price, "SELL_SIGNAL")
                except (KeyError, IndexError):
                    pass
            del pending_sells[date]

        # 포지션 평가 및 손절/추가매수 체크
        next_date = dates[i + 1] if i + 1 < len(dates) else None
        
        for code in list(positions.keys()):
            try:
                close_price = close_pivot.loc[date, code]
                if pd.isna(close_price):
                    continue
            except (KeyError, IndexError):
                continue
            
            pos = positions[code]
            ret = (close_price - pos["avg_cost"]) / pos["avg_cost"] if pos["avg_cost"] > 0 else 0
            
            if ret <= (add_buy_threshold_pct / 100.0):
                is_kospi_bullish = (date in kospi_bullish_dates) if kospi_bullish_only else True
                if pos["step"] == 0:
                    # step 0: 첫 번째 추가 매수 (다음날 시작가로)
                    if next_date and is_kospi_bullish:
                        if next_date not in pending_buys:
                            pending_buys[next_date] = []
                        pending_buys[next_date].append((code, buy_unit, 1))
                elif pos["step"] == 1:
                    # step 1: 두 번째 추가 매수 (다음날 시작가로)
                    if next_date and is_kospi_bullish:
                        if next_date not in pending_buys:
                            pending_buys[next_date] = []
                        pending_buys[next_date].append((code, buy_unit * 2, 2))
                else:
                    # step 2 이상: 즉시 손절 (평균매수가의 -7% 가격)
                    stop_price = pos["avg_cost"] * 0.93
                    _sell(code, date, stop_price, "STOP_LOSS")

        # 시그널 처리
        if next_date:
            # 매수 시그널
            if date in buy_by_date:
                daily_buy_count = 0
                is_kospi_bullish_for_buy = (date in kospi_bullish_dates) if kospi_bullish_only else True
                for code in buy_by_date[date]:
                    if code not in positions and daily_buy_count < max_daily_buys and is_kospi_bullish_for_buy:
                        if next_date not in pending_buys:
                            pending_buys[next_date] = []
                        pending_buys[next_date].append((code, buy_unit, 0))
                        daily_buy_count += 1
            
            # 매도 시그널
            if date in sell_by_date:
                for code in sell_by_date[date]:
                    if code in positions:
                        # 손실 상태(-5% 이하)인 경우 SELL 시그널 무시 (손절 로직 우선)
                        try:
                            close_price = close_pivot.loc[date, code]
                            if pd.notna(close_price):
                                pos = positions[code]
                                ret = (close_price - pos["avg_cost"]) / pos["avg_cost"] if pos["avg_cost"] > 0 else 0
                                # -5% 이하면 손절 로직이 이미 처리했으므로 SELL 시그널 무시
                                if ret <= -0.05:
                                    continue
                        except (KeyError, IndexError):
                            pass
                        
                        if next_date not in pending_sells:
                            pending_sells[next_date] = set()
                        pending_sells[next_date].add(code)

        # 자산 평가
        market_value = 0
        for code, pos in positions.items():
            try:
                close_price = close_pivot.loc[date, code]
                if pd.notna(close_price):
                    market_value += pos["shares"] * close_price
            except (KeyError, IndexError):
                pass
        
        equity_records.append({
            "date": date,
            "cash": cash,
            "market_value": market_value,
            "equity": cash + market_value,
            "positions": len(positions),
        })

    return pd.DataFrame(equity_records), pd.DataFrame(trades)


def render_news_page(signals: pd.DataFrame, selected_date: pd.Timestamp, selected_stock: str | None) -> None:
    st.divider()
    st.subheader("📰 종목별 뉴스")

    available = (
        signals[signals["date"] == selected_date][["code", "name"]]
        .dropna()
        .drop_duplicates()
        .sort_values("name")
    )
    if available.empty:
        st.info("선택한 날짜에 표시할 종목이 없습니다.")
        return

    options = available.apply(lambda r: f"{r['name']} ({r['code']})", axis=1).tolist()
    code_map = dict(zip(options, available["code"]))

    if selected_stock in code_map:
        default_index = options.index(selected_stock)
    else:
        default_index = 0

    selected = st.selectbox("종목 선택", options, index=default_index)
    stock_code = code_map[selected]

    with st.spinner("뉴스 불러오는 중..."):
        crawler = NaverNewsCrawler()
        news_df = crawler.get_recent_news(stock_code)

    if news_df.empty:
        st.info("해당 종목의 최근 뉴스가 없습니다.")
        return

    if "링크" in news_df.columns:
        news_df = news_df.rename(columns={"링크": "news_url"})

    news_rows = news_df.to_dict(orient="records")
    urls = [row.get("news_url", "") for row in news_rows]
    bodies: dict[str, str] = {}
    if urls:
        max_workers = min(8, len(urls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for url, body in zip(urls, executor.map(crawler.get_news_body, urls)):
                bodies[url] = body

    for row in news_rows:
        title = row.get("제목", "(제목 없음)")
        source = row.get("출처", "")
        date = row.get("날짜", "")
        news_url = row.get("news_url", "")

        label = f"{title}"
        if source or date:
            label = f"{label} · {source} {date}".strip()

        with st.expander(label):
            if news_url:
                st.markdown(f"[원문 보기]({news_url})")
            body = bodies.get(news_url, "") if news_url else ""
            st.write(body if body else "본문을 가져올 수 없습니다.")


def render_kakao_section(signals_df: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    """카카오톡 전송 섹션 렌더링"""
    # 세션 상태 초기화
    if "kakao_sender" not in st.session_state:
        st.session_state.kakao_sender = KakaoMessageSender()
    
    kakao = st.session_state.kakao_sender
    
    # 포트폴리오 로드
    from app.portfolio import PortfolioManager
    portfolio_mgr = PortfolioManager()
    portfolio = portfolio_mgr.load_portfolio()
    
    # URL 쿼리 파라미터에서 code 자동 추출
    query_params = st.query_params
    if "code" in query_params and not kakao.is_authenticated():
        auth_code = query_params["code"]
        with st.spinner("카카오톡 인증 중..."):
            success, message = kakao.get_token(auth_code)
            if success:
                st.success(message)
                # URL 파라미터 제거
                st.query_params.clear()
                st.rerun()
            else:
                st.error(message)
                st.query_params.clear()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 인증 상태 표시
        if kakao.is_authenticated():
            st.success("✅ 카카오톡 연동 완료")
        else:
            st.warning("⚠️ 카카오톡 연동 필요")
    
    with col2:
        # 인증 버튼
        if not kakao.is_authenticated():
            if st.button("🔐 카카오톡 연동하기", use_container_width=True):
                if not kakao.rest_api_key:
                    st.error("⚠️ REST API 키가 설정되지 않았습니다. kakao_config.json 파일을 확인하세요.")
                else:
                    try:
                        auth_url = kakao.get_auth_url()
                        st.info("아래 버튼을 클릭하여 카카오 인증을 진행하세요. 인증 완료 후 자동으로 연동됩니다.")
                        st.link_button("🔐 카카오 인증하기", auth_url, use_container_width=True)
                        st.caption("💡 인증 후 자동으로 이 페이지로 돌아옵니다.")
                    except ValueError as e:
                        st.error(str(e))
        else:
            # 메시지 전송 버튼
            if st.button("📤 카카오톡으로 전송", use_container_width=True):
                with st.spinner("전송 중..."):
                    success, message = kakao.send_message(signals_df, str(selected_date.date()), portfolio)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)




def render_stock_analysis_page(
    price_df: pd.DataFrame,
    signals: pd.DataFrame,
    finance_df: pd.DataFrame,
    params: dict,
) -> None:
    """종목별 분석 페이지: 차트상에 BUY/SELL 신호를 표시"""
    st.title("📈 종목분석")
    st.caption("개별 종목의 주가 추이와 매매 신호를 분석합니다")

    # KOSPI 200/포트폴리오 로드 (캐시)
    kospi_dict = load_kospi_stocks_cached()  # {종목코드: 종목명}
    
    if not kospi_dict:
        st.error("KOSPI 200 종목 데이터를 불러올 수 없습니다.")
        return
    
    portfolio_codes = load_portfolio_codes_cached()  # 보유 종목 코드
    
    # 최신 신호 발생 종목 추출
    latest_signals = signals[signals['signal'].isin(['BUY', 'SELL'])].copy()
    latest_signals = latest_signals.sort_values('date', ascending=False)
    signal_codes = latest_signals['code'].unique().tolist()
    
    # 종목 선택 (UI 개선: 한 줄에 모두 배치)
    st.subheader("📊 종목 분석")
    
    # 보유 종목 분리 및 통합 리스트 생성
    held_stocks = []
    unheld_stocks = []
    code_to_name = {}
    
    # 신호 발생 종목 먼저 처리
    for code in signal_codes:
        if code in kospi_dict:
            name = kospi_dict[code]
            if code in portfolio_codes:
                display_name = f"💼 {name} ({code})"  # 보유 + 신호
                held_stocks.append(display_name)
            else:
                display_name = f"⭐ {name} ({code})"  # 신호만
                unheld_stocks.append(display_name)
            code_to_name[display_name] = code
    
    # 신호 없는 종목 처리
    for code, name in sorted(kospi_dict.items()):
        if code not in signal_codes:
            if code in portfolio_codes:
                display_name = f"💼 {name} ({code})"  # 보유만
                held_stocks.append(display_name)
            else:
                display_name = f"{name} ({code})"  # 일반
                unheld_stocks.append(display_name)
            code_to_name[display_name] = code
    
    # 통합 리스트 생성 (보유 종목 먼저, 신호 있는 것 먼저)
    all_stocks = held_stocks + unheld_stocks
    
    if not all_stocks:
        st.error("분석 가능한 종목이 없습니다.")
        return
    
    # 선택 UI
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 기본값: 첫 번째 보유 종목, 없으면 신호 발생 종목, 없으면 첫 번째 종목
        default_index = 0
        if held_stocks:
            default_index = 0
        elif unheld_stocks:
            default_index = len(held_stocks)
        
        selected_stock = st.selectbox(
            "종목",
            all_stocks,
            index=default_index,
            label_visibility="collapsed",
            key="stock_select"
        )
    
    with col2:
        st.caption("기간은 아래에서 시작/종료일로 선택")
    
    if selected_stock not in code_to_name:
        st.error("종목을 선택해주세요.")
        return
    
    selected_code = code_to_name[selected_stock]
    selected_name = kospi_dict.get(selected_code, selected_code)
    
    # 주가 데이터 필터링
    stock_prices = price_df[price_df['code'] == selected_code].copy()
    stock_prices['date'] = pd.to_datetime(stock_prices['date'], errors='coerce')
    stock_prices = stock_prices.dropna(subset=['date']).sort_values('date')

    
    if stock_prices.empty:
        st.warning(f"⚠️ {selected_name}({selected_code})의 주가 데이터가 없습니다.")
        return

    # 날짜 범위 선택 (3년 초과 가능)
    data_start = pd.to_datetime(stock_prices['date']).min()
    data_end = pd.to_datetime(stock_prices['date']).max()

    # 기본 시작일: 최근 1년(데이터가 더 짧으면 데이터 시작일)
    default_start = max(data_start, data_end - pd.DateOffset(days=365))
    start_key = "stock_analysis_start_date"
    end_key = "stock_analysis_end_date"

    # 세션 값 클램프 (선택 종목 데이터 범위 벗어나는 경우 대비)
    if start_key not in st.session_state:
        st.session_state[start_key] = default_start.date()
    else:
        current_start = st.session_state[start_key]
        if current_start < data_start.date():
            st.session_state[start_key] = data_start.date()
        elif current_start > data_end.date():
            st.session_state[start_key] = data_end.date()

    if end_key not in st.session_state:
        st.session_state[end_key] = data_end.date()
    else:
        current_end = st.session_state[end_key]
        if current_end < data_start.date():
            st.session_state[end_key] = data_start.date()
        elif current_end > data_end.date():
            st.session_state[end_key] = data_end.date()

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        selected_start_date = st.date_input(
            "시작일자",
            min_value=data_start.date(),
            max_value=data_end.date(),
            key=start_key,
            help="분석 시작 날짜를 선택하세요",
        )
    with date_col2:
        selected_end_date = st.date_input(
            "종료일자",
            min_value=data_start.date(),
            max_value=data_end.date(),
            key=end_key,
            help="분석 종료 날짜를 선택하세요",
        )

    if selected_start_date > selected_end_date:
        st.error("⚠️ 시작일자는 종료일자보다 빠르거나 같아야 합니다.")
        return

    st.caption(f"선택 구간: {selected_start_date} ~ {selected_end_date} (데이터 범위: {data_start.date()} ~ {data_end.date()})")
    
    # 구간 필터링
    selected_start_ts = pd.to_datetime(selected_start_date)
    selected_end_ts = pd.to_datetime(selected_end_date)
    filtered_prices = stock_prices[
        (stock_prices['date'] >= selected_start_ts) &
        (stock_prices['date'] <= selected_end_ts)
    ].copy()
    
    if filtered_prices.empty:
        st.warning(f"⚠️ 선택한 구간({selected_start_date} ~ {selected_end_date})의 데이터가 없습니다.")
        return

    # 렌더링/구간 판정 설정
    perf_col1, perf_col2, perf_col3 = st.columns([1, 1, 2])
    with perf_col1:
        fast_plot_mode = st.toggle(
            "고속 차트 모드",
            value=True,
            key="stock_analysis_fast_plot_mode",
            help="장기 구간에서 주간 리샘플링으로 차트 렌더링 속도를 높입니다",
        )
    with perf_col2:
        regime_window_label = st.selectbox(
            "상승/하락 기준",
            options=["3개월", "6개월", "9개월", "12개월"],
            index=1,
            key="stock_analysis_regime_window_months",
            help="코스피 지수의 n개월 추세 수익률(+)이면 상승 구간, (-)이면 하락 구간으로 판정합니다",
        )
    with perf_col3:
        st.caption("고속 모드 ON: 캔들/코스피는 주간 기준으로 단순화되어 렌더링됩니다")
    regime_window_months = int(regime_window_label.replace("개월", ""))

    plot_prices = filtered_prices
    if fast_plot_mode and len(filtered_prices) > 520:
        plot_prices = (
            filtered_prices
            .set_index('date')
            .resample('W-FRI')
            .agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            })
            .dropna(subset=['open', 'high', 'low', 'close'])
            .reset_index()
        )
    
    # 해당 종목의 신호 추출
    stock_signals = signals[signals['code'] == selected_code].copy()
    if not stock_signals.empty:
        stock_signals['date'] = pd.to_datetime(stock_signals['date'], errors='coerce')
        stock_signals = stock_signals.dropna(subset=['date'])
        stock_signals = stock_signals[
            (stock_signals['date'] >= selected_start_ts) &
            (stock_signals['date'] <= selected_end_ts)
        ].sort_values('date')

    # 신호 기반 매매 수익률 계산
    # 규칙: 최초 매수 후 추가 매수 금지, SELL로 청산된 뒤에만 재매수 허용
    signal_return_pct = 0.0
    signal_trade_count = 0
    signal_mode_label = "신호 부족"
    signal_return_valid = False
    executed_buy_dates = []
    executed_buy_prices = []
    executed_sell_dates = []
    executed_sell_prices = []
    if not stock_signals.empty and not filtered_prices.empty:
        signal_points_for_return = stock_signals[
            stock_signals['signal'].isin(['BUY', 'SELL'])
        ][['date', 'signal']].copy()
        signal_points_for_return['signal_priority'] = signal_points_for_return['signal'].map({'SELL': 0, 'BUY': 1}).fillna(2)
        signal_points_for_return = signal_points_for_return.sort_values(['date', 'signal_priority'])

        if not signal_points_for_return.empty:
            price_for_return = filtered_prices[['date', 'close']].sort_values('date').copy()
            signal_points_for_return = pd.merge_asof(
                signal_points_for_return,
                price_for_return,
                on='date',
                direction='nearest',
            )
            signal_points_for_return = signal_points_for_return.dropna(subset=['close'])

            cash = 1.0
            shares = 0.0
            has_position = False
            can_buy = True

            for _, row in signal_points_for_return.iterrows():
                px = float(row['close'])
                if px <= 0:
                    continue

                if row['signal'] == 'BUY' and can_buy and not has_position:
                    shares = cash / px
                    cash = 0.0
                    has_position = True
                    can_buy = False
                    executed_buy_dates.append(row['date'])
                    executed_buy_prices.append(px)
                elif row['signal'] == 'SELL' and has_position:
                    cash = shares * px
                    shares = 0.0
                    has_position = False
                    can_buy = True
                    signal_trade_count += 1
                    executed_sell_dates.append(row['date'])
                    executed_sell_prices.append(px)

            final_close = float(filtered_prices['close'].iloc[-1])
            final_equity = cash if not has_position else shares * final_close
            signal_return_pct = (final_equity - 1.0) * 100
            signal_return_valid = True

            if signal_trade_count > 0:
                signal_mode_label = f"완료 {signal_trade_count}건"
            elif has_position:
                signal_mode_label = "보유중(미청산)"
            else:
                signal_mode_label = "거래 없음"

    # 코스피 지수 로드 및 동일 기간 필터링
    kospi_index_df = load_kospi_index(params.get("data_dir", "data"))
    if not kospi_index_df.empty:
        kospi_index_df = kospi_index_df.copy()
        kospi_index_df["date"] = pd.to_datetime(kospi_index_df["date"], errors="coerce")
        kospi_index_df = kospi_index_df.dropna(subset=["date", "index"]).sort_values("date")
        chart_start = selected_start_ts
        chart_end = selected_end_ts
        kospi_filtered = kospi_index_df[
            (kospi_index_df["date"] >= chart_start) & (kospi_index_df["date"] <= chart_end)
        ].copy()

        if fast_plot_mode and len(kospi_filtered) > 520:
            kospi_plot = (
                kospi_filtered
                .set_index("date")
                .resample("W-FRI")
                .last()
                .dropna()
                .reset_index()
            )
        else:
            kospi_plot = kospi_filtered
    else:
        kospi_filtered = pd.DataFrame()
        kospi_plot = pd.DataFrame()

    market_label = "데이터 없음"
    market_emoji = "⚪"
    kospi_change_pct = 0.0
    if not kospi_filtered.empty and len(kospi_filtered) >= 2:
        kospi_start = float(kospi_filtered["index"].iloc[0])
        kospi_end = float(kospi_filtered["index"].iloc[-1])
        if kospi_start > 0:
            kospi_change_pct = ((kospi_end - kospi_start) / kospi_start) * 100
        if kospi_change_pct >= 0:
            market_label = "상승장"
            market_emoji = "📈"
        else:
            market_label = "하락장"
            market_emoji = "📉"

    # 코스피 상승/하락 구간 계산 (연속 구간)
    kospi_regime_segments = []
    if not kospi_filtered.empty and len(kospi_filtered) >= 2:
        regime_df = kospi_filtered[["date", "index"]].copy().sort_values("date")
        # 장기간 조회 시 segment 과다 생성으로 렌더링이 느려지는 문제 완화
        if len(regime_df) > 520:
            regime_df = (
                regime_df.set_index("date")
                .resample("2W-FRI")
                .last()
                .dropna()
                .reset_index()
            )
        elif len(regime_df) > 260:
            regime_df = (
                regime_df.set_index("date")
                .resample("W-FRI")
                .last()
                .dropna()
                .reset_index()
            )

        # n개월 기준 추세 수익률로 상승/하락 구간 판정
        step_days = regime_df["date"].diff().dt.days.median()
        step_days = int(step_days) if pd.notna(step_days) and step_days > 0 else 1
        lookback_points = max(1, int(round((regime_window_months * 30) / step_days)))

        regime_df["trend_return"] = regime_df["index"].pct_change(periods=lookback_points)
        regime_df["change"] = regime_df["index"].diff()
        regime_df["is_up"] = regime_df["trend_return"] >= 0
        # 초기 구간(lookback 이전)은 단기 변화 방향으로 보정
        regime_df["is_up"] = regime_df["is_up"].where(regime_df["trend_return"].notna(), regime_df["change"] >= 0)
        first_regime = bool(regime_df["is_up"].iloc[1]) if len(regime_df) > 1 else True
        regime_df.iloc[0, regime_df.columns.get_loc("is_up")] = first_regime

        segment_start = regime_df["date"].iloc[0]
        current_regime = bool(regime_df["is_up"].iloc[0])

        for i in range(1, len(regime_df)):
            point_regime = bool(regime_df["is_up"].iloc[i])
            if point_regime != current_regime:
                segment_end = regime_df["date"].iloc[i]
                kospi_regime_segments.append((segment_start, segment_end, current_regime))
                segment_start = regime_df["date"].iloc[i]
                current_regime = point_regime

        kospi_regime_segments.append((segment_start, regime_df["date"].iloc[-1], current_regime))
    
    # =====  캔들스틱 차트 + 신호 표시 =====
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('주가(코스피 비교)', '거래량'),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    
    # 캔들스틱 차트
    fig.add_trace(
        go.Candlestick(
            x=plot_prices['date'],
            open=plot_prices['open'],
            high=plot_prices['high'],
            low=plot_prices['low'],
            close=plot_prices['close'],
            name='주가',
            increasing_line_color='red',
            decreasing_line_color='blue'
        ),
        row=1, col=1,
        secondary_y=False,
    )

    # 코스피 지수 라인(보조축)
    if not kospi_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=kospi_plot["date"],
                y=kospi_plot["index"],
                mode="lines",
                name="KOSPI 지수",
                line=dict(color="#6b7280", width=2, dash="dot"),
                opacity=0.9,
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    # 코스피 상승/하락 구간 배경 표시 (상승=빨강, 하락=파랑)
    show_regime_background = (not fast_plot_mode) or (len(kospi_regime_segments) <= 120)
    if show_regime_background:
        for segment_start, segment_end, is_up in kospi_regime_segments:
            fig.add_vrect(
                x0=segment_start,
                x1=segment_end,
                fillcolor="rgba(255, 0, 0, 0.08)" if is_up else "rgba(0, 0, 255, 0.08)",
                opacity=1.0,
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )
    
    # 거래량 바 차트
    colors = ['red' if close >= open else 'blue' 
             for close, open in zip(plot_prices['close'], plot_prices['open'])]
    
    fig.add_trace(
        go.Bar(
            x=plot_prices['date'],
            y=plot_prices['volume'],
            name='거래량',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # BUY/SELL 신호 표시 (스캐터 포인트)
    buy_signals = stock_signals[stock_signals['signal'] == 'BUY']
    sell_signals = stock_signals[stock_signals['signal'] == 'SELL']

    signal_points = stock_signals[stock_signals['signal'].isin(['BUY', 'SELL'])][['date', 'signal']].copy()
    if not signal_points.empty:
        signal_points = signal_points.sort_values('date')
        price_for_match = filtered_prices[['date', 'high', 'low']].sort_values('date')
        signal_points = pd.merge_asof(
            signal_points,
            price_for_match,
            on='date',
            direction='nearest',
        )
    else:
        signal_points = pd.DataFrame(columns=['date', 'signal', 'high', 'low'])
    
    if not buy_signals.empty:
        buy_points = signal_points[signal_points['signal'] == 'BUY'].copy()
        buy_points = buy_points.dropna(subset=['high'])
        buy_prices = (buy_points['high'] * 1.02).tolist()
        
        if buy_prices and not buy_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_points['date'],
                    y=buy_prices,
                    mode='markers',
                    name='BUY',
                    marker=dict(
                        size=12,
                        color='green',
                        symbol='triangle-up',
                        line=dict(width=2, color='darkgreen')
                    ),
                    text=[f"매수 신호<br>{v:.0f}원" for v in buy_prices],
                    hoverinfo='text'
                ),
                row=1, col=1,
                secondary_y=False,
            )
    
    if not sell_signals.empty:
        sell_points = signal_points[signal_points['signal'] == 'SELL'].copy()
        sell_points = sell_points.dropna(subset=['low'])
        sell_prices = (sell_points['low'] * 0.98).tolist()
        
        if sell_prices and not sell_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_points['date'],
                    y=sell_prices,
                    mode='markers',
                    name='SELL',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='triangle-down',
                        line=dict(width=2, color='darkred')
                    ),
                    text=[f"매도 신호<br>{v:.0f}원" for v in sell_prices],
                    hoverinfo='text'
                ),
                row=1, col=1,
                secondary_y=False,
            )

    # 실제 체결 매수/매도 지점 표시 (시그널 규칙 반영)
    if executed_buy_dates and executed_buy_prices:
        fig.add_trace(
            go.Scatter(
                x=executed_buy_dates,
                y=executed_buy_prices,
                mode='markers',
                name='실거래 매수',
                marker=dict(
                    size=14,
                    color='#16a34a',
                    symbol='star',
                    line=dict(width=1, color='black')
                ),
                text=[f"실거래 매수<br>{v:.0f}원" for v in executed_buy_prices],
                hoverinfo='text'
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    if executed_sell_dates and executed_sell_prices:
        fig.add_trace(
            go.Scatter(
                x=executed_sell_dates,
                y=executed_sell_prices,
                mode='markers',
                name='실거래 매도',
                marker=dict(
                    size=14,
                    color='#dc2626',
                    symbol='x',
                    line=dict(width=1, color='black')
                ),
                text=[f"실거래 매도<br>{v:.0f}원" for v in executed_sell_prices],
                hoverinfo='text'
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    
    # 레이아웃 설정
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"{selected_name} ({selected_code}) - {selected_start_date} ~ {selected_end_date}",
    )
    
    # y축 레이블
    fig.update_yaxes(title_text="종목 가격 (원)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="KOSPI 지수", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    
    # x축 설정
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # 주말 제거
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"시장 국면: {market_emoji} {market_label} (코스피 {kospi_change_pct:+.2f}%)")
    if show_regime_background:
        st.caption(f"배경 구간: 🔴 코스피 상승 구간 / 🔵 코스피 하락 구간 (판정 기준: {regime_window_label})")
    else:
        st.caption("배경 구간은 고속 모드에서 자동 생략됨 (렌더링 속도 개선)")
    
    # =====  신호 통계 및 상세 정보 =====
    st.divider()
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("현재가", f"{filtered_prices['close'].iloc[-1]:,.0f}원")
    with col2:
        price_change = filtered_prices['close'].iloc[-1] - filtered_prices['close'].iloc[0]
        price_change_pct = (price_change / filtered_prices['close'].iloc[0] * 100)
        st.metric("기간 수익률", f"{price_change_pct:+.2f}%")
    with col3:
        st.metric("최고가", f"{filtered_prices['high'].max():,.0f}원")
    with col4:
        st.metric("최저가", f"{filtered_prices['low'].min():,.0f}원")
    with col5:
        st.metric("코스피 기간수익률", f"{kospi_change_pct:+.2f}%", delta=market_label)
    with col6:
        if signal_return_valid:
            signal_delta = f"vs 종목 {signal_return_pct - price_change_pct:+.2f}%p"
            st.metric("시그널 매매 수익률", f"{signal_return_pct:+.2f}%", delta=signal_delta)
        else:
            st.metric("시그널 매매 수익률", "N/A", delta=signal_mode_label)
    
    # 신호 통계
    st.subheader("📊 신호 통계")
    
    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🟢 **BUY 신호**: {buy_count}회")
        if not buy_signals.empty:
            st.caption("발생 날짜:")
            for _, row in buy_signals.sort_values('date', ascending=False).head(5).iterrows():
                st.text(f"  • {row['date'].date()}")
    
    with col2:
        st.info(f"🔴 **SELL 신호**: {sell_count}회")
        if not sell_signals.empty:
            st.caption("발생 날짜:")
            for _, row in sell_signals.sort_values('date', ascending=False).head(5).iterrows():
                st.text(f"  • {row['date'].date()}")
    
    # 재무정보 (있으면)
    if not finance_df.empty:
        stock_finance = finance_df[finance_df['code'] == selected_code].copy()
        if not stock_finance.empty:
            stock_finance = stock_finance.sort_values('date')
            latest_finance = stock_finance.iloc[-1]
            
            st.divider()
            st.subheader("💼 재무정보")
            
            fin_col1, fin_col2, fin_col3, fin_col4 = st.columns(4)
            
            with fin_col1:
                if pd.notna(latest_finance.get('per')):
                    st.metric("PER", f"{latest_finance['per']:.2f}")
            with fin_col2:
                if pd.notna(latest_finance.get('pbr')):
                    st.metric("PBR", f"{latest_finance['pbr']:.2f}")
            with fin_col3:
                if pd.notna(latest_finance.get('eps')):
                    st.metric("EPS", f"{latest_finance['eps']:,.0f}원")
            with fin_col4:
                if pd.notna(latest_finance.get('dvr')):
                    st.metric("배당수익률", f"{latest_finance['dvr']:.2f}%")


def render_kospi_crawling_page() -> None:
    st.subheader("🔄 KOSPI 데이터 업데이트")
    st.caption("최신 주가 데이터를 수집합니다")
    st.write("KOSPI 200 종목 리스트와 가격/재무 데이터를 갱신합니다.")

    if st.button("KOSPI 크롤링 실행"):
        with st.spinner("크롤링 중... 잠시만 기다려 주세요."):
            try:
                crawler = CrawlingKospi()
                crawler.crawling()
                # 캐시 클리어하여 최신 데이터 로드
                st.cache_data.clear()
                # 데이터 갱신 시간 저장 (AI 예측 캐시 무효화용)
                import datetime
                st.session_state['data_refreshed_at'] = datetime.datetime.now()
                st.success("크롤링이 완료되었습니다. 데이터가 갱신되었습니다.")
                st.info("시그널 탭으로 이동하여 갱신된 데이터를 확인하세요.")
            except Exception as exc:
                st.error(f"크롤링 실패: {exc}")


def render_optimizer_page(df: pd.DataFrame, kospi_index: pd.DataFrame, params: dict) -> None:
    """최적화 탭 UX 리팩토링: 탐색공간 미리보기 + Top N 비교 + 적용/내보내기"""
    st.subheader("⚙️ 최적화")
    st.caption("탐색 공간을 먼저 확인하고, 실행 후 Top N 후보를 비교·적용합니다")

    from optimizer import get_gpu_info, get_available_years

    # session scaffolding
    if "opt_grid" not in st.session_state:
        st.session_state["opt_grid"] = {}
    if "opt_constraints" not in st.session_state:
        st.session_state["opt_constraints"] = {}
    if "opt_results_df" not in st.session_state:
        st.session_state["opt_results_df"] = pd.DataFrame()
    if "opt_best" not in st.session_state:
        st.session_state["opt_best"] = {}
    if "opt_running" not in st.session_state:
        st.session_state["opt_running"] = False
    if "opt_cancel" not in st.session_state:
        st.session_state["opt_cancel"] = False
    if "opt_last_hash" not in st.session_state:
        st.session_state["opt_last_hash"] = ""
    if "opt_last_run_ts" not in st.session_state:
        st.session_state["opt_last_run_ts"] = "-"

    gpu_info = get_gpu_info()
    info_cols = st.columns(3)
    with info_cols[0]:
        st.metric("사용 가능 CPU 코어", gpu_info["num_cpus"])
    with info_cols[1]:
        processing_mode = "🚀 GPU 가속" if gpu_info["cuda_available"] else "⚙️ CPU 멀티프로세싱"
        st.metric("처리 모드", processing_mode)
    with info_cols[2]:
        st.metric("마지막 실행", st.session_state.get("opt_last_run_ts", "-"))

    available_year_ranges = get_available_years(df)
    if not available_year_ranges:
        st.error("데이터에서 사용 가능한 연도를 찾을 수 없습니다.")
        return

    available_years = sorted(available_year_ranges.keys(), reverse=True)
    all_start_dates = [available_year_ranges[year][0] for year in available_years]
    all_end_dates = [available_year_ranges[year][1] for year in available_years]
    global_start_date = min(all_start_dates)
    global_end_date = max(all_end_dates)

    def _build_float_range(min_v: float, max_v: float, step_v: float) -> list[float]:
        if step_v <= 0 or min_v > max_v:
            return []
        result = []
        cursor = float(min_v)
        while cursor <= float(max_v) + 1e-9:
            result.append(round(cursor, 4))
            cursor += float(step_v)
        return sorted(list(dict.fromkeys(result)))

    def _build_int_range(min_v: int, max_v: int, step_v: int) -> list[int]:
        if step_v <= 0 or min_v > max_v:
            return []
        return list(range(int(min_v), int(max_v) + 1, int(step_v)))

    # [1] Search Space Builder
    block1 = st.container(border=True)
    with block1:
        st.markdown("#### [1] Search Space Builder")
        left, right = st.columns(2)

        with left:
            st.markdown("**기본 설정**")
            initial_cash = st.number_input(
                "초기 자산 (원)",
                min_value=10_000_000,
                max_value=1_000_000_000,
                value=int(st.session_state.get("opt_initial_cash", 50_000_000)),
                step=10_000_000,
                format="%d",
                key="opt_initial_cash",
            )

            optimization_metric = st.selectbox(
                "최적화 기준",
                options=["total_return", "sharpe_ratio", "excess_return"],
                format_func=lambda x: {
                    "total_return": "총 수익률",
                    "sharpe_ratio": "샤프 비율",
                    "excess_return": "초과 수익률",
                }[x],
                key="opt_metric",
            )

            search_mode = st.selectbox(
                "탐색 방식",
                options=["random", "grid"],
                index=0,
                format_func=lambda x: "🚀 랜덤 서치" if x == "random" else "그리드 서치",
                key="opt_search_mode",
            )
            sample_count = st.number_input(
                "랜덤 샘플 수",
                min_value=50,
                max_value=5000,
                value=int(st.session_state.get("opt_sample_count", 150)),
                step=50,
                disabled=(search_mode != "random"),
                key="opt_sample_count",
            )
            random_seed = st.number_input(
                "랜덤 시드",
                min_value=0,
                max_value=99999,
                value=int(st.session_state.get("opt_random_seed", 42)),
                step=1,
                disabled=(search_mode != "random"),
                key="opt_random_seed",
            )

        with right:
            st.markdown("**파라미터 범위**")
            c1, c2, c3 = st.columns(3)
            with c1:
                turnover_window_min = st.number_input("turnover_window 최소", 5, 120, 5, 1, key="opt_tw_min")
            with c2:
                turnover_window_max = st.number_input("turnover_window 최대", 5, 120, 20, 1, key="opt_tw_max")
            with c3:
                turnover_window_step = st.number_input("turnover_window 간격", 1, 20, 5, 1, key="opt_tw_step")

            c4, c5, c6 = st.columns(3)
            with c4:
                turnover_multiplier_min = st.number_input("turnover_multiplier 최소", 1.0, 20.0, 1.5, 0.1, key="opt_tm_min")
            with c5:
                turnover_multiplier_max = st.number_input("turnover_multiplier 최대", 1.0, 20.0, 3.0, 0.1, key="opt_tm_max")
            with c6:
                turnover_multiplier_step = st.number_input("turnover_multiplier 간격", 0.1, 5.0, 0.5, 0.1, key="opt_tm_step")

            c7, c8, c9 = st.columns(3)
            with c7:
                add_buy_threshold_min = st.number_input("extra_buy_threshold 최소", -30.0, -1.0, -10.0, 0.5, key="opt_abt_min")
            with c8:
                add_buy_threshold_max = st.number_input("extra_buy_threshold 최대", -30.0, -1.0, -3.0, 0.5, key="opt_abt_max")
            with c9:
                add_buy_threshold_step = st.number_input("extra_buy_threshold 간격", 0.5, 10.0, 1.0, 0.5, key="opt_abt_step")

            c10, c11, c12 = st.columns(3)
            with c10:
                max_daily_buys_min = st.number_input("max_daily_buys 최소", 1, 10, 1, 1, key="opt_mdb_min")
            with c11:
                max_daily_buys_max = st.number_input("max_daily_buys 최대", 1, 10, 3, 1, key="opt_mdb_max")
            with c12:
                max_daily_buys_step = st.number_input("max_daily_buys 간격", 1, 5, 1, 1, key="opt_mdb_step")

            c13, c14, c15 = st.columns(3)
            with c13:
                buy_unit_min = st.number_input("buy_unit(만원) 최소", 50, 5000, 100, 50, key="opt_bu_min")
            with c14:
                buy_unit_max = st.number_input("buy_unit(만원) 최대", 50, 5000, 300, 50, key="opt_bu_max")
            with c15:
                buy_unit_step = st.number_input("buy_unit(만원) 간격", 50, 1000, 100, 50, key="opt_bu_step")

            st.markdown("**Constraints (선택)**")
            cs1, cs2, cs3 = st.columns(3)
            with cs1:
                max_trades_per_month = st.number_input("max_trades_per_month", min_value=0, max_value=1000, value=0, step=5, key="opt_c_max_trades")
            with cs2:
                max_drawdown_limit = st.number_input("max_drawdown_limit(%)", min_value=-100.0, max_value=0.0, value=0.0, step=1.0, key="opt_c_mdd")
            with cs3:
                min_win_rate = st.number_input("min_win_rate(%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="opt_c_win")

        opt_grid = {
            "max_daily_buys": _build_int_range(max_daily_buys_min, max_daily_buys_max, max_daily_buys_step),
            "rolling_days": _build_int_range(turnover_window_min, turnover_window_max, turnover_window_step),
            "volume_threshold": _build_float_range(turnover_multiplier_min, turnover_multiplier_max, turnover_multiplier_step),
            "add_buy_threshold_pct": _build_float_range(add_buy_threshold_min, add_buy_threshold_max, add_buy_threshold_step),
            "buy_unit": [int(v) * 10_000 for v in _build_int_range(buy_unit_min, buy_unit_max, buy_unit_step)],
        }
        opt_constraints = {
            "max_trades_per_month": int(max_trades_per_month) if int(max_trades_per_month) > 0 else None,
            "max_drawdown_limit": float(max_drawdown_limit) if float(max_drawdown_limit) != 0 else None,
            "min_win_rate": float(min_win_rate) if float(min_win_rate) != 0 else None,
        }
        st.session_state["opt_grid"] = opt_grid
        st.session_state["opt_constraints"] = opt_constraints

    # [2] Search Space Preview
    block2 = st.container(border=True)
    with block2:
        st.markdown("#### [2] Search Space Preview")
        grid_counts = {k: len(v) for k, v in opt_grid.items() if isinstance(v, list)}
        total_combinations = 1
        for count in grid_counts.values():
            total_combinations *= max(count, 0)

        active_constraints = sum(1 for v in opt_constraints.values() if v is not None)
        test_count = total_combinations if search_mode == "grid" else min(int(sample_count), total_combinations)
        estimate_seconds = max(1, int(test_count * 0.35))

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("조합 수", f"{total_combinations:,}")
        with k2:
            st.metric("예상 소요시간", f"약 {estimate_seconds}s")
        with k3:
            st.metric("선택 제약 조건 수", f"{active_constraints}개")

        range_rows = []
        for param_name, values in opt_grid.items():
            if not values:
                continue
            range_rows.append(
                {
                    "param": param_name,
                    "count": len(values),
                    "min": values[0],
                    "max": values[-1],
                    "step": (values[1] - values[0]) if len(values) > 1 else 0,
                }
            )
        range_df = pd.DataFrame(range_rows)
        st.dataframe(range_df, use_container_width=True, hide_index=True)

        if len(opt_grid.get("rolling_days", [])) > 0 and len(opt_grid.get("volume_threshold", [])) > 0:
            preview_points = []
            for rd in opt_grid["rolling_days"]:
                for vt in opt_grid["volume_threshold"]:
                    preview_points.append({"rolling_days": rd, "volume_threshold": vt})
                    if len(preview_points) >= 2000:
                        break
                if len(preview_points) >= 2000:
                    break
            if preview_points:
                st.caption("grid preview: rolling_days × volume_threshold")
                preview_df = pd.DataFrame(preview_points)
                st.scatter_chart(preview_df, x="rolling_days", y="volume_threshold")

    # hash guard
    universe_size = int(df["code"].astype(str).nunique()) if "code" in df.columns else 0
    hash_payload = {
        "grid": opt_grid,
        "constraints": opt_constraints,
        "start": str(global_start_date.date()),
        "end": str(global_end_date.date()),
        "universe": universe_size,
        "search_mode": search_mode,
        "sample_count": int(sample_count),
        "random_seed": int(random_seed),
        "metric": optimization_metric,
    }
    run_hash = hashlib.md5(str(hash_payload).encode()).hexdigest()

    # [3] Execution Control
    block3 = st.container(border=True)
    with block3:
        st.markdown("#### [3] Execution Control")
        too_large = total_combinations > 2000
        large_confirm = True
        if too_large:
            st.warning(f"⚠️ 조합 수가 큽니다: {total_combinations:,}")
            large_confirm = st.checkbox("조합 수가 많아 시간이 오래 걸릴 수 있습니다. 실행합니다.", value=False, key="opt_large_confirm")

        force_rerun = st.checkbox("캐시 무시하고 다시 실행", value=False, key="opt_force_rerun")
        actions = [
            {"key": "run", "label": "최적화 시작", "kind": "primary", "disabled": st.session_state["opt_running"]},
            {"key": "reset", "label": "초기화", "kind": "secondary"},
        ]
        if st.session_state["opt_running"]:
            actions.append({"key": "cancel", "label": "중단", "kind": "secondary"})

        clicked = render_action_row(actions, key_prefix="opt_action_")
        run_clicked = clicked.get("run", False)
        reset_clicked = clicked.get("reset", False)
        cancel_clicked = clicked.get("cancel", False)

        if reset_clicked:
            st.session_state["opt_results_df"] = pd.DataFrame()
            st.session_state["opt_best"] = {}
            st.session_state["opt_cancel"] = False
            st.session_state["opt_running"] = False
            st.success("초기화 완료")

        if cancel_clicked:
            st.session_state["opt_cancel"] = True
            st.warning("취소 요청됨(다음 실행부터 반영)")

        can_use_cache = (
            st.session_state.get("opt_last_hash") == run_hash
            and not st.session_state.get("opt_results_df", pd.DataFrame()).empty
        )

        if run_clicked and too_large and not large_confirm:
            st.error("대규모 실행 확인 체크가 필요합니다.")
            run_clicked = False

        if run_clicked and can_use_cache and not force_rerun:
            st.info("동일 조건 결과가 있어 재실행을 생략합니다.")
            run_clicked = False

    # execute
    if run_clicked:
        st.session_state["opt_running"] = True
        st.session_state["opt_cancel"] = False

        progress_bar = st.progress(0)
        status = st.empty()
        status.info("[1/2] 조합 생성")
        progress_bar.progress(10)
        time_start = datetime.now()

        param_ranges = {
            "max_daily_buys": opt_grid["max_daily_buys"],
            "rolling_days": opt_grid["rolling_days"],
            "volume_threshold": opt_grid["volume_threshold"],
            "add_buy_threshold_pct": opt_grid["add_buy_threshold_pct"],
            "buy_unit": opt_grid["buy_unit"],
        }

        if any(len(v) == 0 for v in param_ranges.values()):
            st.error("파라미터 범위가 비어 있습니다. 최소/최대/간격을 확인하세요.")
            st.session_state["opt_running"] = False
            return

        sample_size_to_use = int(sample_count)
        if search_mode == "random":
            sample_size_to_use = min(sample_size_to_use, total_combinations)
            if sample_size_to_use <= 0:
                st.error("랜덤 샘플 수가 0입니다.")
                st.session_state["opt_running"] = False
                return

        status.info("[2/2] 백테스트 평가 중")
        progress_bar.progress(20)

        def update_progress(current, total, combo_params):
            total = max(int(total), 1)
            current = max(0, int(current))
            progress = min(max(current / total, 0.0), 1.0)
            progress_bar.progress(20 + int(progress * 80))
            if st.session_state.get("opt_cancel", False):
                status.warning("취소 요청됨(진행 중 작업 완료 후 반영)")
                return

            elapsed = (datetime.now() - time_start).total_seconds()
            if progress > 0.03:
                remain = max((elapsed / progress) - elapsed, 0)
                remain_txt = f"{remain/60:.1f}분" if remain > 60 else f"{int(remain)}초"
            else:
                remain_txt = "계산 중"
            status.info(
                f"진행: {current}/{total} ({int(progress*100)}%) · 예상 남은 시간: {remain_txt}"
            )

        optimizer = BacktestOptimizer(df, kospi_index)
        try:
            results_df = optimizer.optimize_parameters(
                start_date=global_start_date,
                end_date=global_end_date,
                param_ranges=param_ranges,
                initial_cash=float(initial_cash),
                progress_callback=update_progress,
                search_mode=search_mode,
                sample_size=sample_size_to_use,
                random_seed=int(random_seed),
            )

            progress_bar.progress(100)
            st.session_state["opt_running"] = False

            if results_df is None or results_df.empty:
                st.warning("⚠️ 최적화 결과가 없습니다.")
            else:
                results_df = results_df.copy()
                if optimization_metric in results_df.columns:
                    results_df = results_df.sort_values(optimization_metric, ascending=False).reset_index(drop=True)
                results_df.insert(0, "rank", range(1, len(results_df) + 1))

                st.session_state["opt_results_df"] = results_df
                best_row = results_df.iloc[0].to_dict()
                st.session_state["opt_best"] = best_row
                st.session_state["opt_last_hash"] = run_hash
                st.session_state["opt_last_run_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success(f"✅ 최적화 완료 · {len(results_df)}개 후보 평가")
        except Exception as e:
            st.session_state["opt_running"] = False
            st.error("⚠️ 최적화 중 오류 발생")
            with st.expander("🔍 오류 상세 정보"):
                import traceback
                st.code(f"Error: {str(e)}", language="text")
                st.code(traceback.format_exc(), language="python")

    # [4] Results
    block4 = st.container(border=True)
    with block4:
        st.markdown("#### [4] Results")
        results_df = st.session_state.get("opt_results_df", pd.DataFrame())
        if results_df is None or results_df.empty:
            st.info("최적화 실행 후 결과가 여기에 표시됩니다.")
            return

        metric_alias = {
            "total_return": "CAGR/total_return",
            "win_rate": "win_rate",
            "mdd": "max_drawdown",
            "max_drawdown": "max_drawdown",
            "total_trades": "trades_count",
        }

        best_row = st.session_state.get("opt_best", {})
        st.markdown("##### Best Params")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("objective", f"{best_row.get(st.session_state.get('opt_metric', 'total_return'), '-')}")
        with b2:
            st.metric("rolling_days", f"{best_row.get('rolling_days', '-')}")
        with b3:
            st.metric("volume_threshold", f"{best_row.get('volume_threshold', '-')}")

        top_n = st.selectbox("Top N", options=[5, 10, 20], index=1, key="opt_top_n")
        top_df = results_df.head(int(top_n)).copy()

        show_cols = [
            "rank",
            st.session_state.get("opt_metric", "total_return"),
            "total_return",
            "win_rate",
            "mdd",
            "max_drawdown",
            "total_trades",
            "rolling_days",
            "volume_threshold",
            "add_buy_threshold_pct",
            "max_daily_buys",
            "buy_unit",
        ]
        show_cols = [c for c in show_cols if c in top_df.columns]
        if "buy_unit" in top_df.columns:
            top_df["buy_unit(만원)"] = (top_df["buy_unit"] // 10_000).astype(int)
            show_cols = [c for c in show_cols if c != "buy_unit"] + ["buy_unit(만원)"]

        st.dataframe(top_df[show_cols], use_container_width=True, hide_index=True)

        st.markdown("##### Compare (최대 3개)")
        compare_labels = [f"#{int(r['rank'])}" for _, r in top_df.iterrows()]
        selected_labels = st.multiselect("비교 후보", options=compare_labels, default=compare_labels[: min(2, len(compare_labels))])
        selected_ranks = [int(lbl.replace("#", "")) for lbl in selected_labels[:3]]
        selected_rows = top_df[top_df["rank"].isin(selected_ranks)]
        if not selected_rows.empty:
            cols = st.columns(len(selected_rows))
            for col, (_, row) in zip(cols, selected_rows.iterrows()):
                with col:
                    st.markdown(f"**#{int(row['rank'])}**")
                    for key in ["total_return", "sharpe_ratio", "excess_return", "win_rate", "mdd", "rolling_days", "volume_threshold", "add_buy_threshold_pct", "max_daily_buys"]:
                        if key in row:
                            st.caption(f"{key}: {row[key]}")

        st.markdown("##### Apply / Export")
        apply_cols = st.columns(min(len(top_df), 3)) if len(top_df) > 0 else []
        for idx, (_, row) in enumerate(top_df.head(len(apply_cols)).iterrows()):
            with apply_cols[idx]:
                if st.button(f"Apply #{int(row['rank'])}", key=f"opt_apply_{int(row['rank'])}", use_container_width=True):
                    if "rolling_days" in row:
                        st.session_state["backtest_rolling_days"] = int(row["rolling_days"])
                    if "volume_threshold" in row:
                        st.session_state["backtest_volume_threshold"] = float(row["volume_threshold"])
                    if "add_buy_threshold_pct" in row:
                        st.session_state["backtest_loss_threshold"] = float(row["add_buy_threshold_pct"])
                    if "max_daily_buys" in row:
                        st.session_state["sim_max_daily_buys"] = int(row["max_daily_buys"])
                    if "buy_unit" in row:
                        st.session_state["sim_buy_unit"] = float(row["buy_unit"])
                    st.success("🎯 시뮬레이션 탭에 파라미터가 적용되었습니다. 실행 버튼을 눌러 확인하세요.")

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                label="Export Top N as CSV",
                data=top_df.to_csv(index=False),
                file_name=f"optimization_top_{top_n}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_col2:
            best_json = json.dumps(best_row, ensure_ascii=False, indent=2, default=str)
            st.download_button(
                label="Export Best Params as JSON",
                data=best_json,
                file_name="optimization_best_params.json",
                mime="application/json",
                use_container_width=True,
            )


def run_app(current_tab: str = "📊 시그널") -> None:
    # 페이지 설정은 streamlit_app.py에서 수행
    st.title("📈 StockVibe")
    st.caption("거래량 급등 기반 스마트 투자 시그널")

    # 상위 사이드바에서 선택한 탭 사용
    available_tabs = ["📊 시그널", "🎯 시뮬레이션", "⚙️ 최적화", "🔄 데이터"]
    if current_tab not in available_tabs:
        current_tab = "📊 시그널"

    st.session_state.active_tab = current_tab

    if "selected_stock" not in st.session_state:
        st.session_state.selected_stock = None

    # 현재 탭에 따라 사이드바 렌더링
    params = render_sidebar(current_tab)

    if current_tab == "📊 시그널":
        # 데이터 로딩 (시그널 탭 전용)
        with st.spinner("📊 데이터 로딩 중..."):
            df = load_stock_data(params["data_dir"])
            kospi = load_kospi_list(params["data_dir"])
            kospi_index = load_kospi_index(params["data_dir"])
            finance_df = load_finance_data(params["data_dir"])

        # 시그널 생성 (운영 전략만)
        with st.spinner("🔍 시그널 분석 중..."):
            registry = load_registry()
            production_ids = get_production_strategy_ids()

            fallback_mode = False
            if not production_ids:
                fallback_mode = True
                st.warning("No validated strategy in production. Please validate one in Simulation.")
                production_ids = ["turnover_spike"]

            all_signals = []
            active_strategy_labels = []
            last_validation_labels = []

            for strategy_id in production_ids:
                try:
                    strategy = get_strategy(strategy_id)
                except Exception:
                    continue

                strategy_params = _get_runtime_strategy_params(strategy_id, params)
                sig = generate_strategy_signals_cached(
                    strategy_id=strategy.spec.strategy_id,
                    strategy_version=strategy.spec.version,
                    price_df=df,
                    market_df=kospi_index,
                    params_json=json.dumps(strategy_params, sort_keys=True, default=str),
                )
                if not sig.empty:
                    all_signals.append(sig)

                active_strategy_labels.append(f"{strategy.spec.name}({strategy.spec.version})")
                state = registry.get(strategy_id, {})
                lv = state.get("last_validation") if isinstance(state, dict) else None
                if isinstance(lv, dict) and lv.get("run_ts"):
                    last_validation_labels.append(f"{strategy.spec.strategy_id}: {lv.get('run_ts')}")

            if all_signals:
                signals = pd.concat(all_signals, ignore_index=True)
            else:
                signals = pd.DataFrame(columns=["date", "code", "signal", "confidence", "features_json", "strategy_id", "strategy_name", "strategy_version"])

            if not signals.empty:
                signals = signals.merge(kospi, on="code", how="left")
                if "spike_ratio" in signals.columns:
                    signals = signals.sort_values(["date", "spike_ratio"], ascending=[False, False])
                else:
                    signals = signals.sort_values(["date"], ascending=[False])
                signals = apply_signal_filters(signals, params["signal_filter"])

            if active_strategy_labels:
                mode_label = "(Fallback)" if fallback_mode else "(Production)"
                st.caption(f"검증된 전략만 반영 {mode_label}: " + ", ".join(active_strategy_labels))
            if last_validation_labels:
                st.caption("Last validation: " + " | ".join(last_validation_labels))

        if not signals.empty and "strategy_id" in signals.columns:
            strategy_filter_options = ["All"] + sorted(signals["strategy_id"].dropna().astype(str).unique().tolist())
            selected_strategy_filter = st.selectbox("전략 필터", strategy_filter_options, index=0, key="signal_strategy_filter")
            if selected_strategy_filter != "All":
                signals = signals[signals["strategy_id"] == selected_strategy_filter]
            strategy_counts = signals["strategy_id"].value_counts().to_dict()
            if strategy_counts:
                st.caption("전략별 시그널 수: " + ", ".join([f"{k}:{v}" for k, v in strategy_counts.items()]))

        selected_date = select_date(signals)
        if selected_date is None:
            st.warning("⚠️ 조건에 맞는 시그널이 없습니다.")
            return

        st.divider()

        # 코스피 양봉 여부 확인
        is_kospi_bullish = False
        kospi_change_pct = 0.0
        if not kospi_index.empty:
            ki = kospi_index.copy()
            ki["date"] = pd.to_datetime(ki["date"], errors="coerce").dt.normalize()
            ki = ki.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            
            # 선택된 날짜의 코스피 데이터 찾기
            selected_normalized = pd.to_datetime(selected_date).normalize()
            today_mask = ki["date"] == selected_normalized
            
            if today_mask.any():
                today_pos = ki[today_mask].index[0]  # reset_index 후라 정수 인덱스
                if today_pos > 0:
                    today_val = ki.loc[today_pos, "index"]
                    prev_val = ki.loc[today_pos - 1, "index"]
                    is_kospi_bullish = today_val > prev_val
                    kospi_change_pct = ((today_val - prev_val) / prev_val * 100) if prev_val > 0 else 0
        
        focus_code = st.query_params.get("focus", "")
        if isinstance(focus_code, list):
            focus_code = focus_code[0] if focus_code else ""
        focus_code = str(focus_code).strip()
        session_focus = str(st.session_state.get("focus_code", "")).strip()
        if session_focus:
            focus_code = session_focus

        selected_signals = signals[(signals["date"] == selected_date) & (signals["signal"] != "")].copy()
        buy_count = int((selected_signals["signal"] == "BUY").sum())
        sell_count = int((selected_signals["signal"] == "SELL").sum())
        total_count = int(len(selected_signals))

        buy_alpha = min(0.35, 0.08 + (buy_count * 0.02))
        sell_alpha = min(0.35, 0.08 + (sell_count * 0.02))
        buy_border = min(0.5, buy_alpha + 0.15)
        sell_border = min(0.5, sell_alpha + 0.15)

        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
        :root {
            --sv-bg: #f7f5ef;
            --sv-card: #ffffff;
            --sv-ink: #1f1f1f;
            --sv-muted: #6b6b6b;
            --sv-accent: #0ea5a8;
            --sv-buy: #e11d48;
            --sv-sell: #2563eb;
            --sv-border: #e7e2d7;
            --sv-buy-alpha: __BUY_ALPHA__;
            --sv-sell-alpha: __SELL_ALPHA__;
            --sv-buy-border: __BUY_BORDER__;
            --sv-sell-border: __SELL_BORDER__;
        }
        .sv-dashboard {
            background: linear-gradient(135deg, #faf6ee 0%, #eef7f6 100%);
            border: 1px solid var(--sv-border);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 10px 24px rgba(22, 22, 22, 0.06);
        }
        .sv-grid {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        .sv-card {
            background: var(--sv-card);
            border: 1px solid var(--sv-border);
            border-radius: 14px;
            padding: 12px 14px;
            min-width: 160px;
            flex: 1;
        }
        .sv-kicker {
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--sv-muted);
            margin-bottom: 6px;
        }
        .sv-title {
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            font-weight: 700;
            font-size: 18px;
            color: var(--sv-ink);
            margin: 0;
        }
        .sv-sub {
            font-size: 13px;
            color: var(--sv-muted);
        }
        .sv-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid transparent;
            margin-right: 6px;
            margin-bottom: 6px;
            text-decoration: none;
        }
        .sv-pill.buy { background: rgba(225, 29, 72, var(--sv-buy-alpha)); color: var(--sv-buy); border-color: rgba(225, 29, 72, var(--sv-buy-border)); }
        .sv-pill.sell { background: rgba(37, 99, 235, var(--sv-sell-alpha)); color: var(--sv-sell); border-color: rgba(37, 99, 235, var(--sv-sell-border)); }
        .sv-pill.neutral { background: rgba(14, 165, 168, 0.08); color: var(--sv-accent); border-color: rgba(14, 165, 168, 0.25); }
        .sv-chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
        .sv-highlight {
            background: #fff8ef;
            border: 1px solid #f0dcc4;
            border-radius: 16px;
            padding: 14px 16px;
        }
        .sv-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }
        .sv-signal-card {
            background: var(--sv-card);
            border: 1px solid var(--sv-border);
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 8px 18px rgba(18, 18, 18, 0.04);
        }
        .sv-signal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .sv-tag {
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 999px;
            border: 1px solid transparent;
        }
        .sv-tag.buy { background: rgba(225, 29, 72, var(--sv-buy-alpha)); color: var(--sv-buy); border-color: rgba(225, 29, 72, var(--sv-buy-border)); }
        .sv-tag.sell { background: rgba(37, 99, 235, var(--sv-sell-alpha)); color: var(--sv-sell); border-color: rgba(37, 99, 235, var(--sv-sell-border)); }
        .sv-tag.neutral { background: rgba(14, 165, 168, 0.08); color: var(--sv-accent); border-color: rgba(14, 165, 168, 0.25); }
        .sv-card-footer {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 12px;
            color: var(--sv-muted);
        }
        @media (max-width: 640px) {
            .sv-card { min-width: 140px; }
        }
        </style>
        """
        css = css.replace("__BUY_ALPHA__", f"{buy_alpha:.2f}")
        css = css.replace("__SELL_ALPHA__", f"{sell_alpha:.2f}")
        css = css.replace("__BUY_BORDER__", f"{buy_border:.2f}")
        css = css.replace("__SELL_BORDER__", f"{sell_border:.2f}")
        css = textwrap.dedent(css).strip()
        st.markdown(css, unsafe_allow_html=True)

        mood_label = "양봉" if is_kospi_bullish else "음봉"
        mood_emoji = "📈" if is_kospi_bullish else "📉"

        dashboard_html = textwrap.dedent(
            f"""
            <div class="sv-dashboard">
                <div class="sv-grid">
                    <div class="sv-card">
                        <div class="sv-kicker">Date</div>
                        <div class="sv-title">{selected_date.date()}</div>
                        <div class="sv-sub">기준일</div>
                    </div>
                    <div class="sv-card">
                        <div class="sv-kicker">KOSPI Index</div>
                        <div class="sv-title">{mood_emoji} {mood_label}</div>
                        <div class="sv-sub">{kospi_change_pct:+.2f}%</div>
                    </div>
                    <div class="sv-card">
                        <div class="sv-kicker">Signals</div>
                        <div class="sv-title">{total_count}개</div>
                        <div class="sv-sub">BUY {buy_count} · SELL {sell_count}</div>
                    </div>
                    <div class="sv-card">
                        <div class="sv-kicker">Top Picks</div>
                        <div class="sv-title">상위 {params['top_n']}개</div>
                        <div class="sv-sub">표시 종목 수</div>
                    </div>
                </div>
            </div>
            """
        ).strip()
        st.markdown(dashboard_html, unsafe_allow_html=True)

        if "pending_view_mode" in st.session_state:
            st.session_state.signal_view_mode = st.session_state.pop("pending_view_mode")
        elif focus_code:
            st.session_state.signal_view_mode = "📋 상세"
        elif "signal_view_mode" not in st.session_state:
            st.session_state.signal_view_mode = "📊 테이블"

        view_mode = st.radio(
            "표시 방식",
            ["📊 테이블", "📋 상세"],
            horizontal=True,
            label_visibility="collapsed",
            key="signal_view_mode",
        )

        st.divider()
        st.subheader(f"📊 결과 · {selected_date.date()} 투자 시그널")
        st.caption(f"현재 보기: {view_mode} · 조건에 맞는 종목만 표시됩니다.")
        
        latest, cols = build_latest_table(signals, selected_date, params["top_n"], finance_df, df)

        if latest.empty:
            st.info("표시할 시그널 결과가 없습니다.")
            st.caption("분석 기간, 급등 기준, 표시 종목 수를 조정하면 결과가 나타날 수 있습니다.")
        else:
            import html
            import re

            buy_pick = latest[latest["signal"] == "BUY"].head(1)
            st.subheader("✅ 매수 후보")
            if buy_pick.empty:
                st.caption("상위 표시 종목 중 매수 시그널이 없습니다.")
            else:
                row = buy_pick.iloc[0]
                raw_name = row.get("name", "")
                clean_name = re.sub(r"<[^>]+>", "", str(raw_name))
                code = row.get("code", "")
                spike_ratio = row.get("spike_ratio", None)
                badge = "급등 비율 정보 없음"
                if pd.notna(spike_ratio):
                    badge = f"급등 비율 {spike_ratio:.0%}"
                buy_pick_html = textwrap.dedent(
                    f"""
                    <div class="sv-highlight">
                        <div class="sv-kicker">Buy Pick</div>
                        <div class="sv-title">{html.escape(clean_name)} ({html.escape(str(code))})</div>
                        <div class="sv-sub">{badge}</div>
                    </div>
                    """
                ).strip()
                st.markdown(buy_pick_html, unsafe_allow_html=True)

            chip_items = []
            for _, row in latest.iterrows():
                raw_name = row.get("name", "")
                clean_name = re.sub(r"<[^>]+>", "", str(raw_name))
                code = row.get("code", "")
                signal = row.get("signal", "")
                chip_items.append({
                    "label": f"{clean_name} ({code}) · {signal}",
                    "code": str(code),
                })
            if chip_items:
                st.markdown("#### 🔖 상위 시그널")
                chips_per_row = 6
                for start in range(0, len(chip_items), chips_per_row):
                    row_items = chip_items[start:start + chips_per_row]
                    chip_cols = st.columns(len(row_items))
                    for chip_col, item in zip(chip_cols, row_items):
                        with chip_col:
                            if st.button(item["label"], key=f"chip_{item['code']}_{start}"):
                                st.session_state.focus_code = item["code"]
                                st.session_state.pending_view_mode = "📋 상세"
                                st.rerun()
                st.caption("칩을 클릭하면 해당 종목 상세가 즉시 펼쳐집니다.")

            if view_mode == "📊 테이블":
                st.caption(f"조회 결과: 총 {len(latest)}개 종목")
                render_table(latest, cols)
            else:
                st.caption(f"조회 결과: 총 {len(latest)}개 종목")
                render_table_with_finance(latest, cols, finance_df, df, focus_code=focus_code)

        st.divider()
        st.subheader("📤 액션")
        st.caption(f"공유 기준일: {selected_date.date()}")
        if latest.empty:
            st.caption("전송 가능한 시그널이 없어 공유 액션이 비활성 상태입니다.")
        else:
            render_kakao_section(latest, selected_date)

    elif current_tab == "🎯 시뮬레이션":
        st.subheader("🎯 백테스트 시뮬레이션")
        if "simulation_mode" not in st.session_state:
            st.session_state["simulation_mode"] = "📊 기본 시뮬레이션"
        if "sim_running" not in st.session_state:
            st.session_state["sim_running"] = False
        if "sim_stop_requested" not in st.session_state:
            st.session_state["sim_stop_requested"] = False
        if "recent_runs" not in st.session_state:
            st.session_state["recent_runs"] = []
        if "simulation_results" not in st.session_state:
            st.session_state["simulation_results"] = {}
        if "simulation_last_param_hash" not in st.session_state:
            st.session_state["simulation_last_param_hash"] = ""

        # 최근 실행 불러오기 값을 위젯 생성 전에 적용 (Streamlit widget key 직접 수정 오류 방지)
        if "sim_load_pending" in st.session_state:
            slot = st.session_state.pop("sim_load_pending") or {}
            st.session_state["simulation_mode"] = slot.get("mode", st.session_state.get("simulation_mode", "📊 기본 시뮬레이션"))
            if slot.get("strategy_id"):
                st.session_state["sim_strategy_id"] = slot["strategy_id"]

            st.session_state["sim_initial_cash"] = float(slot.get("initial_cash", st.session_state.get("sim_initial_cash", params["initial_cash"])))
            st.session_state["sim_buy_unit"] = float(slot.get("buy_unit", st.session_state.get("sim_buy_unit", params["buy_unit"])))
            st.session_state["sim_max_daily_buys"] = int(slot.get("max_daily_buys", st.session_state.get("sim_max_daily_buys", params["max_daily_buys"])))

            st.session_state["backtest_rolling_days"] = int(slot.get("window", st.session_state.get("backtest_rolling_days", params["turnover_window"])))
            st.session_state["backtest_volume_threshold"] = float(slot.get("multiplier", st.session_state.get("backtest_volume_threshold", params["turnover_multiplier"])))
            st.session_state["backtest_loss_threshold"] = float(slot.get("loss", st.session_state.get("backtest_loss_threshold", params["add_buy_threshold_pct"])))

            st.session_state["sim_kospi_bullish_only"] = bool(slot.get("kospi_bullish_only", st.session_state.get("sim_kospi_bullish_only", False)))
            months_loaded = int(slot.get("kospi_bullish_lookback_months", 6))
            st.session_state["sim_kospi_bullish_month_label"] = f"{months_loaded}개월"

            if slot.get("start") and slot.get("start") != "-":
                st.session_state["sim_start_date"] = pd.to_datetime(slot["start"]).date()
            if slot.get("end") and slot.get("end") != "-":
                st.session_state["sim_end_date"] = pd.to_datetime(slot["end"]).date()

        mode_desc = {
            "📊 기본 시뮬레이션": "기본: 전체 신호 기반 단일 백테스트",
            "🔬 종목별 일괄 테스트": "일괄: 다수 종목 성과 비교",
            "📅 연도별 성과 분석": "연도별: 전략 일관성 검증",
        }

        mode_box = st.container(border=True)
        with mode_box:
            st.markdown("#### [1] Mode Selector")
            simulation_mode = st.radio(
                "분석 방식을 선택하세요",
                options=["📊 기본 시뮬레이션", "🔬 종목별 일괄 테스트", "📅 연도별 성과 분석"],
                index=["📊 기본 시뮬레이션", "🔬 종목별 일괄 테스트", "📅 연도별 성과 분석"].index(st.session_state.get("simulation_mode", "📊 기본 시뮬레이션")),
                horizontal=True,
                key="sim_mode_select",
            )
            st.session_state["simulation_mode"] = simulation_mode
            st.caption(mode_desc.get(simulation_mode, ""))

        param_box = st.container(border=True)
        with param_box:
            st.markdown("#### [2] Parameter Block")
            strategy_specs = list_strategies()
            strategy_registry = load_registry()
            enabled_specs = [s for s in strategy_specs if strategy_registry.get(s.strategy_id, {}).get("enabled", True)]
            select_specs = enabled_specs if enabled_specs else strategy_specs

            if not select_specs:
                st.error("전략이 등록되어 있지 않습니다. app/strategies/impl를 확인하세요.")
                return

            selected_strategy_id = st.session_state.get("sim_strategy_id", select_specs[0].strategy_id)
            strategy_ids = [s.strategy_id for s in select_specs]
            if selected_strategy_id not in strategy_ids:
                selected_strategy_id = strategy_ids[0]

            sim_strategy = st.selectbox(
                "전략 선택 (실험)",
                options=strategy_ids,
                index=strategy_ids.index(selected_strategy_id),
                format_func=lambda sid: f"{get_strategy(sid).spec.name} · {sid} · v{get_strategy(sid).spec.version}",
                key="sim_strategy_id",
            )
            selected_strategy_id = sim_strategy

            cap1, cap2, cap3 = st.columns(3)
            with cap1:
                initial_cash = st.number_input(
                    "초기 자산",
                    min_value=0,
                    max_value=1_000_000_000,
                    value=int(st.session_state.get("sim_initial_cash", params["initial_cash"])),
                    step=5_000_000,
                    format="%d",
                    key="sim_initial_cash_main",
                )
            with cap2:
                buy_unit = st.number_input(
                    "매수 금액 단위 (만원)",
                    min_value=50,
                    max_value=1000,
                    value=int(st.session_state.get("sim_buy_unit", params["buy_unit"]) // 10_000),
                    step=50,
                    key="sim_buy_unit_main",
                ) * 10_000
            with cap3:
                max_daily_buys = st.number_input(
                    "일일 매수 한도",
                    min_value=1,
                    max_value=10,
                    value=int(st.session_state.get("sim_max_daily_buys", params["max_daily_buys"])),
                    step=1,
                    key="sim_max_daily_buys_main",
                )

            st.session_state["sim_initial_cash"] = float(initial_cash)
            st.session_state["sim_buy_unit"] = float(buy_unit)
            st.session_state["sim_max_daily_buys"] = int(max_daily_buys)

            params["initial_cash"] = float(initial_cash)
            params["buy_unit"] = float(buy_unit)
            params["max_daily_buys"] = int(max_daily_buys)

            if simulation_mode != "📅 연도별 성과 분석":
                date_col1, date_col2 = st.columns(2)
                from datetime import timedelta
                today = datetime.now().date()
                one_year_ago = today - timedelta(days=365)
                with date_col1:
                    start_date = st.date_input("시작일자", value=one_year_ago, key="sim_start_date")
                with date_col2:
                    end_date = st.date_input("종료일자", value=today, key="sim_end_date")
                if start_date >= end_date:
                    st.error("⚠️ 시작일자는 종료일자보다 빨라야 합니다")
                    return
                days_diff = (end_date - start_date).days
            else:
                start_date = None
                end_date = None
                days_diff = 365

            roll_col, vol_col, loss_col = st.columns(3)
            with roll_col:
                rolling_days = st.slider("분석 기간 (일)", 5, 60, int(params["turnover_window"]), 1, key="backtest_rolling_days")
            with vol_col:
                volume_threshold = st.slider("급등 기준 (배수)", 1.0, 10.0, float(params["turnover_multiplier"]), 0.1, key="backtest_volume_threshold")
            with loss_col:
                loss_threshold = st.slider("추가매수 손실 임계값 (%)", -30.0, -1.0, float(params["add_buy_threshold_pct"]), 0.5, key="backtest_loss_threshold")

            kospi_gate_col1, kospi_gate_col2 = st.columns(2)
            with kospi_gate_col1:
                kospi_bullish_only = st.toggle(
                    "코스피 상승기간일 때만 매수",
                    value=bool(st.session_state.get("sim_kospi_bullish_only", False)),
                    key="sim_kospi_bullish_only",
                    help="ON이면 코스피 n개월 추세 수익률이 양(+)인 기간에만 BUY 진입합니다",
                )
            with kospi_gate_col2:
                kospi_bullish_month_label = st.selectbox(
                    "상승기간 기준",
                    options=["3개월", "6개월", "9개월", "12개월"],
                    index=["3개월", "6개월", "9개월", "12개월"].index(
                        st.session_state.get("sim_kospi_bullish_month_label", "6개월")
                    ) if st.session_state.get("sim_kospi_bullish_month_label", "6개월") in ["3개월", "6개월", "9개월", "12개월"] else 1,
                    key="sim_kospi_bullish_month_label",
                    disabled=not kospi_bullish_only,
                )
            kospi_bullish_lookback_months = int(str(kospi_bullish_month_label).replace("개월", ""))

        df_status = load_stock_data(params["data_dir"])
        df_status["date"] = pd.to_datetime(df_status["date"])
        total_stocks = int(df_status["code"].nunique()) if "code" in df_status.columns else 0
        coverage_start = df_status["date"].min().date() if not df_status.empty else "-"
        coverage_end = df_status["date"].max().date() if not df_status.empty else "-"
        estimated_days = max(int(days_diff), 1)
        estimated_seconds = (total_stocks * estimated_days) * 0.0008
        workload = total_stocks * estimated_days

        current_run = {
            "mode": simulation_mode,
            "strategy_id": selected_strategy_id,
            "strategy_version": get_strategy(selected_strategy_id).spec.version,
            "start": str(start_date) if start_date else "-",
            "end": str(end_date) if end_date else "-",
            "window": int(rolling_days),
            "multiplier": float(volume_threshold),
            "loss": float(loss_threshold),
            "kospi_bullish_only": bool(kospi_bullish_only),
            "kospi_bullish_lookback_months": int(kospi_bullish_lookback_months),
            "initial_cash": float(params["initial_cash"]),
            "buy_unit": float(params["buy_unit"]),
            "max_daily_buys": int(params["max_daily_buys"]),
        }
        param_hash = hashlib.md5(str(current_run).encode()).hexdigest()

        exec_box = st.container(border=True)
        with exec_box:
            st.markdown("#### [3] Execution Control Block")
            st.caption(f"예상 소요시간: 약 {estimated_seconds:.1f}초 | 데이터 커버리지: {coverage_start} ~ {coverage_end}")
            if workload > 100000:
                st.warning(f"⚠️ 대규모 실행 감지: workload={workload:,}")

            actions = [
                {"key": "run", "label": "실행", "kind": "primary", "disabled": st.session_state["sim_running"]},
                {"key": "reset", "label": "초기화", "kind": "secondary"},
            ]
            if st.session_state["sim_running"]:
                actions.append({"key": "stop", "label": "중단", "kind": "secondary"})

            clicked = render_action_row(actions, key_prefix="sim_action_")
            run_simulation = clicked.get("run", False)
            reset_params = clicked.get("reset", False)
            stop_requested = clicked.get("stop", False)

            st.markdown("##### 최근 실행 설정")
            if st.session_state["recent_runs"]:
                recent_df = pd.DataFrame(st.session_state["recent_runs"])
                st.dataframe(recent_df, use_container_width=True, hide_index=True)
                load_cols = st.columns(len(st.session_state["recent_runs"]))
                for idx, slot in enumerate(st.session_state["recent_runs"]):
                    with load_cols[idx]:
                        if st.button(f"Load {idx + 1}", key=f"sim_load_{idx}", use_container_width=True):
                            st.session_state["sim_load_pending"] = dict(slot)
                            st.rerun()

        if reset_params:
            st.session_state.clear()
            st.rerun()
        if stop_requested:
            st.session_state["sim_stop_requested"] = True
            st.warning("중단 요청이 접수되었습니다.")
        if not run_simulation:
            st.info("⚙️ 파라미터를 설정하고 실행 버튼을 눌러주세요.")
            return

        st.session_state["sim_running"] = True
        st.session_state["sim_stop_requested"] = False

        recent_runs = [current_run] + [r for r in st.session_state["recent_runs"] if r != current_run]
        st.session_state["recent_runs"] = recent_runs[:3]

        use_cached_result = (
            st.session_state.get("simulation_last_param_hash") == param_hash
            and param_hash in st.session_state.get("simulation_results", {})
        )

        progress_placeholder = st.empty()
        with progress_placeholder.container():
            render_progress_steps(["데이터 로드", "시그널 생성", "백테스트 실행"], 0, 0.1)

        result_box = st.container(border=True)
        with result_box:
            st.markdown("#### [4] Results Block")
        
        # 데이터 로딩
        with st.spinner("📊 데이터 로딩 중..."):
            df = load_stock_data(params["data_dir"])
            kospi = load_kospi_list(params["data_dir"])
            kospi_index = load_kospi_index(params["data_dir"])
            
            if simulation_mode != "📅 연도별 성과 분석":
                # 선택된 날짜 범위로 전체 데이터 필터링
                df['date'] = pd.to_datetime(df['date'])
                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)
                df_filtered = df[(df['date'] >= start_date_dt) & (df['date'] <= end_date_dt)]
                
                if df_filtered.empty:
                    st.error(f"⚠️ 선택한 기간({start_date} ~ {end_date})에 데이터가 없습니다.")
                    return
            else:
                # 연도별 성과분석은 전체 historical data 사용
                df['date'] = pd.to_datetime(df['date'])
                df_filtered = df  # 전체 데이터 사용
        
        st.divider()
        
        # 모드별 시뮬레이션 실행
        if simulation_mode == "📊 기본 시뮬레이션":
            st.subheader(f"📊 백테스트 결과 ({start_date} ~ {end_date})")
            st.caption(f"분석 기간: {(end_date - start_date).days}일 동안 시뮬레이션")

            strategy_for_sim = get_strategy(selected_strategy_id)
            strategy_params_for_sim = _get_runtime_strategy_params(
                selected_strategy_id,
                params,
                rolling_days=int(rolling_days),
                volume_threshold=float(volume_threshold),
            )
            strategy_params_for_sim.update(
                {
                    "initial_cash": float(params["initial_cash"]),
                    "buy_unit": float(params["buy_unit"]),
                    "max_daily_buys": int(params["max_daily_buys"]),
                    "add_buy_threshold_pct": float(loss_threshold),
                }
            )
            st.caption(f"실험 전략: {strategy_for_sim.spec.name} (v{strategy_for_sim.spec.version})")

            cached_payload = st.session_state["simulation_results"].get(param_hash) if use_cached_result else None
            if cached_payload and cached_payload.get("mode") == simulation_mode:
                equity_df = cached_payload.get("equity_df", pd.DataFrame())
                trades_df = cached_payload.get("trades_df", pd.DataFrame())
                strategy_signals = cached_payload.get("strategy_signals", pd.DataFrame())
                st.info("🧠 동일 파라미터 결과를 캐시에서 불러왔습니다.")
            else:
                with st.spinner("🔍 시그널 분석 중..."):
                    signals = generate_strategy_signals_cached(
                        strategy_id=strategy_for_sim.spec.strategy_id,
                        strategy_version=strategy_for_sim.spec.version,
                        price_df=df_filtered,
                        market_df=kospi_index,
                        params_json=json.dumps(strategy_params_for_sim, sort_keys=True, default=str),
                    )
                    signals = signals.merge(kospi, on="code", how="left")

                    if "spike_ratio" in signals.columns:
                        signals = signals.sort_values(["date", "spike_ratio"], ascending=[False, False])
                    else:
                        signals = signals.sort_values(["date"], ascending=[False])
                    signals = apply_signal_filters(signals, params["signal_filter"])

                strategy_signals = generate_strategy_signals_cached(
                    strategy_id=strategy_for_sim.spec.strategy_id,
                    strategy_version=strategy_for_sim.spec.version,
                    price_df=df_filtered,
                    market_df=kospi_index,
                    params_json=json.dumps(strategy_params_for_sim, sort_keys=True, default=str),
                )
                strategy_signals = strategy_signals.merge(kospi, on="code", how="left")

                equity_df, trades_df = run_turnover_strategy_backtest(
                    df_filtered,
                    strategy_signals,
                    kospi_index,
                    start_date_dt,
                    top_n=2,
                    initial_cash=params["initial_cash"],
                    max_daily_buys=params["max_daily_buys"],
                    buy_unit=params["buy_unit"],
                    kospi_bullish_only=bool(kospi_bullish_only),
                    kospi_bullish_lookback_months=int(kospi_bullish_lookback_months),
                    add_buy_threshold_pct=float(loss_threshold),
                )
                st.session_state["simulation_results"][param_hash] = {
                    "mode": simulation_mode,
                    "equity_df": equity_df,
                    "trades_df": trades_df,
                    "strategy_signals": strategy_signals,
                }
                st.session_state["simulation_last_param_hash"] = param_hash

            validation_col1, validation_col2 = st.columns([2, 1])
            with validation_col1:
                st.caption("검증 게이트: 시뮬레이션에서 전략 성능 검증 후 설정에서 Production 승격")
            with validation_col2:
                if st.button("✅ Validate 전략", key="validate_selected_strategy", use_container_width=True):
                    validation_result = validate_strategy(
                        strategy_id=selected_strategy_id,
                        params=strategy_params_for_sim,
                        universe="All KOSPI",
                        start_date=str(start_date),
                        end_date=str(end_date),
                        thresholds={
                            "min_cagr": 0.05,
                            "max_mdd": 0.25,
                            "min_win_rate": 0.52,
                            "min_trades": 30,
                        },
                    )
                    update_validation_result(selected_strategy_id, validation_result)
                    if validation_result.get("validated"):
                        st.success("전략 검증 통과: Settings에서 Production 승격 가능합니다.")
                    else:
                        st.warning("전략 검증 미통과: 기준치 충족 후 다시 검증하세요.")
                    st.json(validation_result)
            render_backtest_curve(equity_df, kospi_index, start_date_dt)
            if not equity_df.empty:
                equity_df = equity_df.copy()
                for col in ["cash", "market_value", "equity"]:
                    equity_df[col] = equity_df[col].fillna(0).astype(float).floordiv(1).astype(int)
                st.line_chart(equity_df.set_index("date")["equity"])
                st.dataframe(
                    equity_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "date": st.column_config.DateColumn("날짜"),
                        "cash": st.column_config.NumberColumn("현금", format="%,.0f"),
                        "market_value": st.column_config.NumberColumn("평가금액", format="%,.0f"),
                        "equity": st.column_config.NumberColumn("총자산", format="%,.0f"),
                        "positions": st.column_config.NumberColumn("보유 종목수"),
                    },
                )
            if not trades_df.empty:
                st.divider()
                st.subheader("💼 거래 내역")
            trades_df = trades_df.copy()
            
            # 승률 계산 (매도 거래만 대상)
            sell_trades = trades_df[trades_df["action"] == "SELL"].copy()
            if not sell_trades.empty and "return_pct" in sell_trades.columns:
                sell_trades_valid = sell_trades.dropna(subset=["return_pct"])
                if not sell_trades_valid.empty:
                    total_trades = len(sell_trades_valid)
                    winning_trades = len(sell_trades_valid[sell_trades_valid["return_pct"] > 0])
                    losing_trades = len(sell_trades_valid[sell_trades_valid["return_pct"] < 0])
                    breakeven_trades = len(sell_trades_valid[sell_trades_valid["return_pct"] == 0])
                    
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    avg_return = sell_trades_valid["return_pct"].mean()
                    avg_win = sell_trades_valid[sell_trades_valid["return_pct"] > 0]["return_pct"].mean() if winning_trades > 0 else 0
                    avg_loss = sell_trades_valid[sell_trades_valid["return_pct"] < 0]["return_pct"].mean() if losing_trades > 0 else 0
                    
                    # 손익비 (평균 수익 / 평균 손실 절대값)
                    profit_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0
                    
                    st.markdown("#### 📊 백테스트 승률 통계")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 거래 수", f"{total_trades}건")
                    with col2:
                        win_delta = f"+{winning_trades}승 / -{losing_trades}패"
                        st.metric("승률", f"{win_rate:.1f}%", delta=win_delta)
                    with col3:
                        st.metric("평균 수익률", f"{avg_return:.2f}%")
                    with col4:
                        st.metric("손익비", f"{profit_loss_ratio:.2f}")
                    
                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        st.metric("평균 수익", f"+{avg_win:.2f}%" if avg_win > 0 else "N/A")
                    with col6:
                        st.metric("평균 손실", f"{avg_loss:.2f}%" if avg_loss < 0 else "N/A")
                    with col7:
                        # 총 실현손익
                        if "pnl" in sell_trades_valid.columns:
                            total_pnl = sell_trades_valid["pnl"].sum()
                            st.metric("총 실현손익", f"{total_pnl:,.0f}원")
                    with col8:
                        # 초기자산 대비 총자산 변동
                        if not equity_df.empty:
                            initial_equity = params["initial_cash"]
                            final_equity = equity_df["equity"].iloc[-1]
                            equity_change = final_equity - initial_equity
                            equity_change_pct = (equity_change / initial_equity * 100) if initial_equity > 0 else 0
                            st.metric("자산 변동", f"{equity_change:+,.0f}원", delta=f"{equity_change_pct:+.2f}%")
                    
                    st.markdown("---")
            
            # equity_df와 merge하여 총자산 정보 추가
            if not equity_df.empty:
                equity_summary = equity_df[["date", "equity"]].copy()
                trades_df = trades_df.merge(equity_summary, on="date", how="left")
            
            trades_df["return_pct_display"] = trades_df["return_pct"].apply(
                lambda v: f"{v:.2f}%" if pd.notna(v) else ""
            )
            st.dataframe(
                trades_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("날짜"),
                    "amount": st.column_config.NumberColumn("금액", format="%,.0f"),
                    "price": st.column_config.NumberColumn("가격", format="%,.0f"),
                    "buy_price": st.column_config.NumberColumn("매수가", format="%,.0f"),
                    "shares": st.column_config.NumberColumn("수량", format="%,.0f"),
                    "step": st.column_config.NumberColumn("매수단계"),
                    "pnl": st.column_config.NumberColumn("손익", format="%,.0f"),
                    "return_pct_display": st.column_config.TextColumn("수익률(%)"),
                    "reason": st.column_config.TextColumn("매도사유"),
                    "equity": st.column_config.NumberColumn("총자산", format="%,.0f"),
                },
            )
        
        elif simulation_mode == "🔬 종목별 일괄 테스트":
            st.subheader("🔬 종목별 알고리즘 테스트")
            
            # 테스트할 종목 선택
            with st.expander("📌 테스트 종목 선택", expanded=True):
                test_mode = st.radio(
                    "테스트 방식",
                    options=["전체 KOSPI 종목", "상위 N개 종목", "선택적 종목"],
                    horizontal=True
                )
                
                test_stock_codes = []
    
                # KOSPI 목록 로드
                temp_kospi = kospi  # 이미 로드됨
                
                if test_mode == "전체 KOSPI 종목":
                    test_stock_codes = [str(code) for code in (list(temp_kospi.keys()) if isinstance(temp_kospi, dict) else temp_kospi['code'].unique().tolist())]
                    st.success(f"✅ {len(test_stock_codes)}개 종목 테스트 예정")
                    st.caption(f"샘플: {', '.join(test_stock_codes[:5])}")
                
                elif test_mode == "상위 N개 종목":
                    top_n = st.slider("상위 N개", min_value=5, max_value=50, value=20, step=5)
                    if isinstance(temp_kospi, dict):
                        test_stock_codes = [str(code) for code in list(temp_kospi.keys())[:top_n]]
                    else:
                        test_stock_codes = [str(code) for code in temp_kospi['code'].head(top_n).tolist()]
                    st.success(f"✅ 상위 {top_n}개 종목 테스트")
                    st.caption(f"샘플: {', '.join(test_stock_codes[:5])}")
                
                else:  # 선택적 종목
                    if isinstance(temp_kospi, dict):
                        stock_labels = [f"{code} - {name}" for code, name in temp_kospi.items()]
                    else:
                        stock_labels = [f"{str(code)} - {name}" for code, name in zip(temp_kospi.get('code', []), temp_kospi.get('name', []))]
                    
                    selected_labels = st.multiselect(
                        "테스트할 종목 선택",
                        options=stock_labels,
                        max_selections=30
                    )
                    test_stock_codes = [label.split(" - ")[0].strip() for label in selected_labels]
                    st.success(f"✅ {len(test_stock_codes)}개 종목 선택됨")
                
                # 선택된 종목과 df_filtered의 데이터 교집합 확인
                if test_stock_codes:
                    available_stocks = df_filtered['code'].astype(str).unique().tolist()
                    overlapping = [code for code in test_stock_codes if code in available_stocks]
                    st.info(f"📊 데이터 검증: {len(overlapping)}/{len(test_stock_codes)} 종목에 데이터 있음")
                    
                    if len(overlapping) < len(test_stock_codes):
                        missing = [code for code in test_stock_codes if code not in available_stocks]
                        st.warning(f"⚠️ 데이터 없는 종목: {', '.join(missing[:5])}")
                    
                    test_stock_codes = overlapping  # 데이터가 있는 종목만 테스트
            
            st.divider()

            if len(test_stock_codes) == 0:
                st.info("📌 테스트할 종목을 선택해주세요.")
                return
            
            st.divider()
            
            st.subheader("🔬 종목별 알고리즘 성과 분석")
            st.caption(f"기간: {start_date} ~ {end_date} | 테스트 종목: {len(test_stock_codes)}개")
            
            # 일괄 백테스트 실행
            from app.backtest_analyzer import run_batch_backtest

            cached_payload = st.session_state["simulation_results"].get(param_hash) if use_cached_result else None
            if cached_payload and cached_payload.get("mode") == simulation_mode:
                batch_results = cached_payload.get("batch_results", pd.DataFrame())
                st.info("🧠 동일 파라미터 결과를 캐시에서 불러왔습니다.")
            else:
                with st.spinner("🔬 종목별 테스트 진행 중..."):
                    batch_results = run_batch_backtest(
                        df_filtered,
                        test_stock_codes,
                        int(rolling_days),
                        float(volume_threshold),
                        float(loss_threshold),
                        start_date,
                        end_date,
                        build_signals,
                        run_turnover_strategy_backtest,
                        kospi_index,
                        kospi_list=kospi,
                        initial_cash=params["initial_cash"],
                        max_daily_buys=params["max_daily_buys"],
                        buy_unit=params["buy_unit"],
                        kospi_bullish_only=bool(kospi_bullish_only),
                        kospi_bullish_lookback_months=int(kospi_bullish_lookback_months),
                    )
                st.session_state["simulation_results"][param_hash] = {
                    "mode": simulation_mode,
                    "batch_results": batch_results,
                }
                st.session_state["simulation_last_param_hash"] = param_hash
            
            if not batch_results.empty:
                st.success(f"✅ {len(batch_results)}개 종목 분석 완료")
                
                st.divider()
                
                # 테스트 설정 요약
                st.subheader("⚙️ 테스트 설정")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                with summary_col1:
                    st.metric("📅 분석 기간", f"{(end_date - start_date).days}일")
                with summary_col2:
                    st.metric("📊 급등 기준", f"{volume_threshold:.1f}배")
                with summary_col3:
                    st.metric("💰 초기 자본금", f"{params['initial_cash']:,.0f}원")
                with summary_col4:
                    st.metric("🔴 손절 임계값", f"{loss_threshold:.1f}%")
                
                st.divider()
                st.subheader("📊 종목별 성과 비교")
                
                # 필터링 옵션
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_trades = st.slider("최소 거래수", min_value=0, max_value=20, value=3)
                
                with col2:
                    sort_by = st.selectbox(
                        "정렬 기준",
                        options=["total_return_pct", "win_rate", "profit_loss_ratio"],
                        format_func=lambda x: {
                            "total_return_pct": "📈 총 수익률",
                            "win_rate": "🎯 승률",
                            "profit_loss_ratio": "💰 손익비"
                        }[x]
                    )
                
                with col3:
                    top_n = st.slider("상위 N개 표시", min_value=5, max_value=50, value=20)
                
                # 결과 필터링 및 정렬
                filtered_results = batch_results[batch_results['total_trades'] >= min_trades].copy()
                
                if not filtered_results.empty:
                    if sort_by in filtered_results.columns:
                        filtered_results = filtered_results.sort_values(sort_by, ascending=False)
                    
                    top_results = filtered_results.head(top_n)
                    
                    # 상위 3개 강조 표시
                    st.markdown("### 🏆 Top 3 성과 종목")
                    
                    top_3 = top_results.head(3)
                    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                        medal = ["🥇", "🥈", "🥉"][idx - 1]
                        
                        col_medal, col_info = st.columns([0.5, 4.5])
                        
                        with col_medal:
                            st.metric("", f"{medal} #{idx}")
                        
                        with col_info:
                            # 수익률에 따라 색상 결정
                            return_color = "🟢" if row['total_return_pct'] >= 0 else "🔴"
                            
                            col_name, col_return, col_trades, col_win = st.columns(4)
                            
                            with col_name:
                                st.write(f"**{row['name']}** ({row['code']})")
                            
                            with col_return:
                                st.write(f"{return_color} 수익률: **{row['total_return_pct']:+.2f}%**")
                            
                            with col_trades:
                                st.write(f"📊 매매: **{int(row['total_trades'])}회** ({int(row['win_trades'])}승/{int(row['lose_trades'])}패)")
                            
                            with col_win:
                                st.write(f"🎯 승률: **{row['win_rate']:.1f}%** | 손익비: **{row['profit_loss_ratio']:.2f}**")
                    
                    st.divider()
                    
                    # 상세 결과 표 (컬럼 선택)
                    st.markdown("### 📋 상세 결과")
                    
                    # 메트릭 설명
                    with st.expander("📖 지표 설명", expanded=False):
                        col_exp1, col_exp2 = st.columns(2)
                        
                        with col_exp1:
                            st.write("**총 수익률 (total_return_pct)**")
                            st.caption("초기 자본금 대비 최종 자산의 총 수익률\n예: 초기 5,000만원 → 최종 5,126만원 → +2.52%")
                            st.write("")
                            st.write("**승률 (win_rate)**")
                            st.caption("매도한 거래 중 수익을 본 거래의 비율\n예: 10회 매도 중 7회 수익 → 70%")
                            st.write("")
                            st.write("**최대낙폭 (max_drawdown)**")
                            st.caption("시뮬레이션 기간 중 최고점 대비 최저점의 낙폭")
                        
                        with col_exp2:
                            st.write("**평균 수익률 (avg_return)**")
                            st.caption("**개별 거래(매수가격 대비)의 평균 수익률**\n예: 매수 10,000원 → 평균 10,050원 매도 → +0.5%\n주의: 전체 자산 대비 수익률이 아닙니다")
                            st.write("")
                            st.write("**손익비 (profit_loss_ratio)**")
                            st.caption("평균 수익 / 평균 손실의 비율\n예: 평균수익 +2% / 평균손실 -1% → 2.0")
                            st.write("")
                            st.write("**거래수 (total_trades)**")
                            st.caption("시뮬레이션 기간 동안의 총 매매 횟수\n승거래 + 패배래 = 총 거래")
                    
                    display_cols = ['code', 'name', 'total_return_pct', 'total_trades', 'win_trades', 
                                   'lose_trades', 'win_rate', 'avg_return', 'profit_loss_ratio', 'max_drawdown']
                    available_cols = [col for col in display_cols if col in top_results.columns]
                    
                    # 데이터 포맷팅
                    display_df = top_results[available_cols].copy()
                    
                    # 백분율과 소수점 포맷팅
                    if 'total_return_pct' in display_df.columns:
                        display_df['total_return_pct'] = display_df['total_return_pct'].apply(lambda x: f"{x:+.2f}%")
                    if 'win_rate' in display_df.columns:
                        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1f}%")
                    if 'avg_return' in display_df.columns:
                        display_df['avg_return'] = display_df['avg_return'].apply(lambda x: f"{x:+.2f}%")
                    if 'max_drawdown' in display_df.columns:
                        display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                    if 'profit_loss_ratio' in display_df.columns:
                        display_df['profit_loss_ratio'] = display_df['profit_loss_ratio'].apply(lambda x: f"{x:.2f}")
                    
                    # 정수 포맷팅
                    if 'total_trades' in display_df.columns:
                        display_df['total_trades'] = display_df['total_trades'].astype(int)
                    if 'win_trades' in display_df.columns:
                        display_df['win_trades'] = display_df['win_trades'].astype(int)
                    if 'lose_trades' in display_df.columns:
                        display_df['lose_trades'] = display_df['lose_trades'].astype(int)
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    st.divider()
                    st.subheader("📈 성과 분포")
                    
                    # 통계 정보
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    with stats_col1:
                        avg_return = filtered_results['total_return_pct'].mean()
                        return_color = "🟢" if avg_return >= 0 else "🔴"
                        st.metric(f"{return_color} 평균 수익률", f"{avg_return:+.2f}%")
                    
                    with stats_col2:
                        avg_win_rate = filtered_results['win_rate'].mean()
                        st.metric("🎯 평균 승률", f"{avg_win_rate:.1f}%")
                    
                    with stats_col3:
                        avg_ratio = filtered_results['profit_loss_ratio'].mean()
                        st.metric("💰 평균 손익비", f"{avg_ratio:.2f}")
                    
                    with stats_col4:
                        avg_dd = filtered_results['max_drawdown'].mean()
                        st.metric("📉 평균 최대낙폭", f"{avg_dd:.2f}%")
                    
                    # 성과 차트
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 상위 15개 종목 수익률 분포")
                        chart_data = filtered_results.set_index('name')['total_return_pct'].head(15)
                        st.bar_chart(chart_data)
                    
                    with col2:
                        st.markdown("#### 📊 상위 15개 종목 승률 분포")
                        chart_data = filtered_results.set_index('name')['win_rate'].head(15)
                        st.bar_chart(chart_data)
                    
                    st.divider()
                    st.subheader("🏆 추천 종목 (조건 필터링)")
                    
                    # 추천 조건을 expander에 넣음
                    with st.expander("🔧 필터 조건 설정", expanded=True):
                        rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
                        
                        with rec_col1:
                            min_return = st.number_input("최소 수익률 (%)", value=5.0, step=0.5)
                        
                        with rec_col2:
                            min_win_rate = st.number_input("최소 승률 (%)", value=40.0, step=5.0)
                        
                        with rec_col3:
                            min_ratio = st.number_input("최소 손익비", value=0.5, step=0.1)
                        
                        with rec_col4:
                            max_drawdown = st.number_input("최대 낙폭 (%)", value=-20.0, step=-5.0)
                    
                    # 조건 적용
                    recommendations = filtered_results[
                        (filtered_results['total_return_pct'] >= min_return) &
                        (filtered_results['win_rate'] >= min_win_rate) &
                        (filtered_results['profit_loss_ratio'] >= min_ratio) &
                        (filtered_results['max_drawdown'] >= max_drawdown)
                    ].copy()
                    
                    if not recommendations.empty:
                        st.success(f"✅ 추천 종목: {len(recommendations)}개 (필터 조건 만족)")
                        
                        # 추천 종목 상세 정보
                        rec_df = recommendations[['code', 'name', 'total_return_pct', 'total_trades', 'win_rate', 'profit_loss_ratio', 'max_drawdown']].copy()
                        
                        rec_df['total_return_pct'] = rec_df['total_return_pct'].apply(lambda x: f"{x:+.2f}%")
                        rec_df['win_rate'] = rec_df['win_rate'].apply(lambda x: f"{x:.1f}%")
                        rec_df['profit_loss_ratio'] = rec_df['profit_loss_ratio'].apply(lambda x: f"{x:.2f}")
                        rec_df['max_drawdown'] = rec_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                        rec_df['total_trades'] = rec_df['total_trades'].astype(int)
                        
                        st.dataframe(
                            rec_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.warning("⚠️ 설정한 조건에 맞는 종목이 없습니다. 필터 조건을 완화해보세요.")

                else:
                    st.warning(f"⚠️ 최소 거래수 {min_trades}개 이상인 종목이 없습니다.")
            else:
                st.error("❌ 테스트할 데이터가 부족합니다.")
        
        elif simulation_mode == "📅 연도별 성과 분석":
            st.subheader("📅 연도별 알고리즘 성과 분석")
            st.caption("시황 변화에 무관하게 일관되게 성과를 내는 종목 찾기")
            
            # 분석 종목 선택
            with st.expander("📌 분석 종목 선택", expanded=True):
                st.caption("단일 종목 또는 모든 종목에 대해 2010년부터 현재까지 연도별로 분석합니다")
                
                # 분석 범위 선택
                analysis_scope = st.radio(
                    "분석 범위",
                    options=["📌 단일 종목", "📊 모든 종목"],
                    horizontal=True,
                    help="단일: 선택한 종목만 | 모든 종목: 가격 데이터가 있는 모든 종목"
                )
                
                # KOSPI 목록에서 종목 선택
                if not isinstance(kospi, pd.DataFrame):
                    st.error("❌ KOSPI 목록을 불러올 수 없습니다.")
                    return
                
                if analysis_scope == "📌 단일 종목":
                    stock_options = [f"{code} - {name}" for code, name in zip(kospi['code'], kospi['name'])]
                    selected_stock = st.selectbox(
                        "분석할 종목",
                        options=stock_options,
                        key="yearly_stock_select"
                    )
                    selected_code = selected_stock.split(" - ")[0].strip()
                    selected_name = selected_stock.split(" - ")[1].strip()
                    selected_codes = [selected_code]
                else:  # 모든 종목
                    st.info("📊 가격 데이터가 있는 모든 종목을 분석합니다")
                    selected_codes = kospi['code'].astype(str).tolist()
                    st.write(f"분석할 종목 수: **{len(selected_codes)}개**")
                
                # 연도 범위 선택
                col_start_year, col_end_year = st.columns(2)
                with col_start_year:
                    start_year = st.number_input("시작 연도", min_value=2010, max_value=2026, value=2010)
                with col_end_year:
                    end_year = st.number_input("종료 연도", min_value=2010, max_value=2026, value=2026)
                
                if start_year > end_year:
                    st.error("❌ 시작 연도가 종료 연도보다 클 수 없습니다.")
                    return
            
            st.divider()
            
            # 연도별 백테스트 실행
            cached_payload = st.session_state["simulation_results"].get(param_hash) if use_cached_result else None
            if cached_payload and cached_payload.get("mode") == simulation_mode:
                yearly_results = cached_payload.get("yearly_results", pd.DataFrame())
                consistency_summary = cached_payload.get("consistency_summary", pd.DataFrame())
                st.info("🧠 동일 파라미터 결과를 캐시에서 불러왔습니다.")
            else:
                if analysis_scope == "📌 단일 종목":
                    from app.yearly_backtest import run_yearly_backtest, analyze_consistency
                    
                    with st.spinner(f"📅 {selected_name} 연도별 분석 진행 중..."):
                        yearly_results = run_yearly_backtest(
                            df,
                            selected_code,
                            selected_name,
                            int(rolling_days),
                            float(volume_threshold),
                            float(loss_threshold),
                            build_signals,
                            run_turnover_strategy_backtest,
                            kospi_index,
                            initial_cash=params["initial_cash"],
                            max_daily_buys=params["max_daily_buys"],
                            buy_unit=params["buy_unit"],
                            start_year=int(start_year),
                            end_year=int(end_year),
                        )
                    consistency_summary = pd.DataFrame()
                else:  # 모든 종목
                    from app.yearly_backtest import run_yearly_backtest_batch, summarize_all_stocks_consistency
                    
                    with st.spinner(f"📅 {len(selected_codes)}개 종목 연도별 분석 진행 중... (시간이 걸릴 수 있습니다)"):
                        yearly_results = run_yearly_backtest_batch(
                            df,
                            selected_codes,
                            int(rolling_days),
                            float(volume_threshold),
                            float(loss_threshold),
                            build_signals,
                            run_turnover_strategy_backtest,
                            kospi_index,
                            kospi_list=kospi,
                            initial_cash=params["initial_cash"],
                            max_daily_buys=params["max_daily_buys"],
                            buy_unit=params["buy_unit"],
                            start_year=int(start_year),
                            end_year=int(end_year),
                        )
                    consistency_summary = summarize_all_stocks_consistency(yearly_results) if not yearly_results.empty else pd.DataFrame()

                st.session_state["simulation_results"][param_hash] = {
                    "mode": simulation_mode,
                    "yearly_results": yearly_results,
                    "consistency_summary": consistency_summary,
                }
                st.session_state["simulation_last_param_hash"] = param_hash
            
            if not yearly_results.empty:
                if analysis_scope == "📌 단일 종목":
                    # 단일 종목 결과 표시
                    st.success(f"✅ {len(yearly_results)}개 연도 분석 완료")
                    
                    st.divider()
                    
                    # 일관성 분석
                    from app.yearly_backtest import analyze_consistency
                    consistency = analyze_consistency(yearly_results)
                    
                    st.subheader("📊 성과 일관성 분석")
                    
                    cons_col1, cons_col2, cons_col3, cons_col4 = st.columns(4)
                    
                    with cons_col1:
                        pos_years = consistency.get('positive_years', 0)
                        total = consistency.get('total_years', 0)
                        success_rate = (pos_years / total * 100) if total > 0 else 0
                        st.metric("✅ 수익 연도", f"{pos_years}/{total}년 ({success_rate:.0f}%)")
                    
                    with cons_col2:
                        avg_ret = consistency.get('avg_return_pct', 0)
                        std_ret = consistency.get('std_return_pct', 0)
                        color = "🟢" if avg_ret >= 0 else "🔴"
                        st.metric(f"{color} 평균 연수익률", f"{avg_ret:+.2f}%", f"표준편차: {std_ret:.2f}%")
                    
                    with cons_col3:
                        min_ret = consistency.get('min_return_pct', 0)
                        max_ret = consistency.get('max_return_pct', 0)
                        st.metric("📈 수익률 범위", f"{min_ret:+.2f}% ~ {max_ret:+.2f}%")
                    
                    with cons_col4:
                        win_rate_avg = consistency.get('win_rate_avg', 0)
                        win_rate_std = consistency.get('win_rate_std', 0)
                        st.metric("🎯 평균 승률", f"{win_rate_avg:.1f}%", f"표준편차: {win_rate_std:.1f}%")
                
                else:  # 모든 종목
                    # 모든 종목 결과 표시
                    st.success(f"✅ {yearly_results['code'].nunique()}개 종목의 {len(yearly_results)}개 연도별 분석 완료")
                    
                    st.divider()
                    st.subheader("🏆 종목별 성과 순위")
                    
                    # 종목별 일관성 요약
                    ranking_cols = ['code', 'name', 'test_years', 'positive_years', 'success_rate', 'avg_return_pct', 'std_return_pct', 'avg_win_rate']
                    available_ranking_cols = [col for col in ranking_cols if col in consistency_summary.columns]
                    
                    ranking_df = consistency_summary[available_ranking_cols].copy()
                    
                    # 포맷팅
                    if 'success_rate' in ranking_df.columns:
                        ranking_df['success_rate'] = ranking_df['success_rate'].apply(lambda x: f"{x:.0f}%")
                    if 'avg_return_pct' in ranking_df.columns:
                        ranking_df['avg_return_pct'] = ranking_df['avg_return_pct'].apply(lambda x: f"{x:+.2f}%")
                    if 'std_return_pct' in ranking_df.columns:
                        ranking_df['std_return_pct'] = ranking_df['std_return_pct'].apply(lambda x: f"{x:.2f}%")
                    if 'avg_win_rate' in ranking_df.columns:
                        ranking_df['avg_win_rate'] = ranking_df['avg_win_rate'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(
                        ranking_df.head(30),
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    # CSV 다운로드
                    csv_data = consistency_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 종목별 성과 순위 다운로드 (CSV)",
                        data=csv_data,
                        file_name="yearly_performance_ranking.csv",
                        mime="text/csv",
                        key="download_ranking"
                    )
                
                st.divider()
                
                if analysis_scope == "📌 단일 종목":
                    # 연도별 상세 결과 (단일 종목에만 표시)
                    st.subheader("📋 연도별 상세 결과")
                    
                    display_cols = ['year', 'total_return_pct', 'total_trades', 'win_trades', 'lose_trades', 
                                   'win_rate', 'avg_return', 'profit_loss_ratio', 'max_drawdown']
                    available_cols = [col for col in display_cols if col in yearly_results.columns]
                    
                    display_df = yearly_results[available_cols].copy()
                    
                    # 포맷팅
                    if 'total_return_pct' in display_df.columns:
                        display_df['total_return_pct'] = display_df['total_return_pct'].apply(lambda x: f"{x:+.2f}%")
                    if 'win_rate' in display_df.columns:
                        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1f}%")
                    if 'avg_return' in display_df.columns:
                        display_df['avg_return'] = display_df['avg_return'].apply(lambda x: f"{x:+.2f}%")
                    if 'max_drawdown' in display_df.columns:
                        display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                    if 'profit_loss_ratio' in display_df.columns:
                        display_df['profit_loss_ratio'] = display_df['profit_loss_ratio'].apply(lambda x: f"{x:.2f}")
                    if 'total_trades' in display_df.columns:
                        display_df['total_trades'] = display_df['total_trades'].astype(int)
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    st.divider()
                    
                    # 연도별 수익률 차트
                    st.subheader("📈 연도별 수익률 추이")
                    
                    chart_df = yearly_results[['year', 'total_return_pct']].set_index('year')
                    st.bar_chart(chart_df)
                    
                    st.subheader("📊 연도별 승률 추이")
                    
                    chart_df2 = yearly_results[['year', 'win_rate']].set_index('year')
                    st.bar_chart(chart_df2)
            else:
                st.error("❌ 분석할 수 있는 데이터가 없습니다.")

        st.session_state["sim_running"] = False

    elif current_tab == "⚙️ 최적화":
        # 데이터 로딩 (최적화 탭 전용)
        with st.spinner("📊 데이터 로딩 중..."):
            df = load_stock_data(params["data_dir"])
            kospi_index = load_kospi_index(params["data_dir"])

        render_optimizer_page(df, kospi_index, params)
    
    else:
        render_kospi_crawling_page()
