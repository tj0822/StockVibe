import json
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

        # 백테스트 설정은 시뮬레이션 탭에서만 표시
        initial_cash = 50_000_000
        max_daily_buys = 2
        
        # 매수 금액 단위 기본값 (최적 파라미터가 있으면 그것을 사용)
        default_buy_unit_won = st.session_state.get('applied_buy_unit', 2_000_000)
        default_buy_unit_man = int(default_buy_unit_won // 10_000)
        buy_unit = 2_000_000
        
        if current_tab == "🎯 시뮬레이션":
            with st.expander("💰 백테스트 설정", expanded=True):
                initial_cash = st.number_input(
                    "초기 자산", 
                    min_value=0, max_value=1_000_000_000, 
                    value=50_000_000, step=5_000_000,
                    format="%d",
                    help="백테스트 시작 자산 (원)"
                )
                buy_unit = st.number_input(
                    "매수 금액 단위 (만원)",
                    min_value=50, max_value=1000, value=default_buy_unit_man, step=100,
                    help="1회 매수 시 투자 금액 (만원 단위). 예: 100 = 100만원"
                ) * 10_000  # 만원 -> 원 변환
                max_daily_buys = st.number_input(
                    "일일 매수 한도", 
                    min_value=1, max_value=10, value=2, step=1, 
                    help="하루에 매수할 수 있는 최대 종목 수"
                )

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
    
    # 코스피 양봉 날짜 세트 생성 (전일 대비 상승한 날)
    if kospi_bullish_dates is None:
        kospi_bullish_dates = set()
        if not kospi_index.empty:
            ki = kospi_index.copy()
            ki["date"] = pd.to_datetime(ki["date"], errors="coerce").dt.normalize()
            ki = ki.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            ki["prev_index"] = ki["index"].shift(1)
            ki["is_bullish"] = ki["index"] > ki["prev_index"]
            kospi_bullish_dates = set(ki[ki["is_bullish"]]["date"].values)
    
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
                is_kospi_bullish = (date in kospi_bullish_dates) if kospi_bullish_dates else True
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
                for code in buy_by_date[date]:
                    if code not in positions and daily_buy_count < max_daily_buys:
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
    
    # KOSPI 200 종목 로드
    from crawling_kospi import CrawlingKospi
    from app.portfolio import PortfolioManager
    
    @st.cache_data(ttl=3600)
    def load_kospi_stocks():
        try:
            crawler = CrawlingKospi()
            kospi_dict = crawler.GetKospi200()
            return kospi_dict
        except:
            return {}
    
    kospi_dict = load_kospi_stocks()  # {종목코드: 종목명}
    
    if not kospi_dict:
        st.error("KOSPI 200 종목 데이터를 불러올 수 없습니다.")
        return
    
    # 포트폴리오 로드
    portfolio_mgr = PortfolioManager()
    portfolio = portfolio_mgr.load_portfolio()
    portfolio_codes = set(portfolio.keys())  # 보유 종목 코드
    
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
    
    # 기간 선택 (기본값: 6개월)
    period_map = {
        "1개월": 30,
        "3개월": 90,
        "6개월": 180,
        "1년": 365,
        "3년": 1095,
    }
    
    # 선택 UI를 한 줄에 배치
    col1, col2, col3 = st.columns([2, 1, 1])
    
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
        selected_period = st.selectbox(
            "기간",
            list(period_map.keys()),
            index=2,  # 기본값: 6개월
            label_visibility="collapsed",
            key="period_select"
        )
    
    with col3:
        st.write("")  # 간격 조정
    
    if selected_stock not in code_to_name:
        st.error("종목을 선택해주세요.")
        return
    
    selected_code = code_to_name[selected_stock]
    selected_name = kospi_dict.get(selected_code, selected_code)
    chart_period = selected_period
    
    # 주가 데이터 필터링
    stock_prices = price_df[price_df['code'] == selected_code].copy()
    stock_prices = stock_prices.sort_values('date')

    
    if stock_prices.empty:
        st.warning(f"⚠️ {selected_name}({selected_code})의 주가 데이터가 없습니다.")
        return
    
    # 기간 필터링
    days = period_map[chart_period]
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(days=days)
    filtered_prices = stock_prices[stock_prices['date'] >= cutoff_date].copy()
    
    if filtered_prices.empty:
        st.warning(f"⚠️ {chart_period} 기간의 데이터가 없습니다.")
        return
    
    # 해당 종목의 신호 추출
    stock_signals = signals[signals['code'] == selected_code].copy()
    stock_signals = stock_signals[stock_signals['date'] >= cutoff_date]
    
    # =====  캔들스틱 차트 + 신호 표시 =====
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
    
    # BUY/SELL 신호 표시 (스캐터 포인트)
    buy_signals = stock_signals[stock_signals['signal'] == 'BUY']
    sell_signals = stock_signals[stock_signals['signal'] == 'SELL']
    
    if not buy_signals.empty:
        # 신호 발생 날짜의 종가 찾기
        buy_prices = []
        for _, signal_row in buy_signals.iterrows():
            signal_date = signal_row['date']
            price_on_date = filtered_prices[filtered_prices['date'] == signal_date]
            if not price_on_date.empty:
                buy_prices.append(price_on_date['high'].iloc[0] * 1.02)  # 고가 위에 약간 위에 표시
            else:
                # 날짜를 못 찾으면 가장 가까운 날짜 사용
                closest = filtered_prices.iloc[(filtered_prices['date'] - signal_date).abs().argsort()[:1]]
                if not closest.empty:
                    buy_prices.append(closest['high'].iloc[0] * 1.02)
        
        if buy_prices:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['date'],
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
                row=1, col=1
            )
    
    if not sell_signals.empty:
        # 신호 발생 날짜의 종가 찾기
        sell_prices = []
        for _, signal_row in sell_signals.iterrows():
            signal_date = signal_row['date']
            price_on_date = filtered_prices[filtered_prices['date'] == signal_date]
            if not price_on_date.empty:
                sell_prices.append(price_on_date['low'].iloc[0] * 0.98)  # 저가 아래에 약간 아래에 표시
            else:
                # 날짜를 못 찾으면 가장 가까운 날짜 사용
                closest = filtered_prices.iloc[(filtered_prices['date'] - signal_date).abs().argsort()[:1]]
                if not closest.empty:
                    sell_prices.append(closest['low'].iloc[0] * 0.98)
        
        if sell_prices:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['date'],
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
                row=1, col=1
            )
    
    # 레이아웃 설정
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"{selected_name} ({selected_code}) - {chart_period} 차트",
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
    
    # =====  신호 통계 및 상세 정보 =====
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
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
    """파라미터 최적화 페이지 렌더링 - Unified 접근법"""
    st.subheader("🎯 파라미터 최적화")
    st.caption("📊 전체 기간(2022-2024)에 대해 최적의 범용 투자 전략 파라미터를 찾습니다")
    st.write("모든 거래일에 대해 파라미터 조합을 테스트하여 가장 우수한 범용 파라미터를 찾습니다.")
    
    # GPU/CPU 정보 표시
    from optimizer import get_gpu_info, get_available_years
    gpu_info = get_gpu_info()
    
    info_cols = st.columns(3)
    with info_cols[0]:
        st.metric("사용 가능 CPU 코어", gpu_info['num_cpus'])
    with info_cols[1]:
        processing_mode = "🚀 GPU 가속" if gpu_info['cuda_available'] else "⚙️ CPU 멀티프로세싱"
        st.metric("처리 모드", processing_mode, delta="최고 성능" if gpu_info['cuda_available'] else "표준")
    with info_cols[2]:
        st.metric("병렬처리 방식", "multiprocessing.Pool", delta="Windows 호환")
    
    # 설정 영역을 컬럼으로 나누기
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 최적화 기간")
        
        # 데이터에서 이용 가능한 연도 추출
        available_year_ranges = get_available_years(df)
        
        if not available_year_ranges:
            st.error("데이터에서 사용 가능한 연도를 찾을 수 없습니다.")
            return
        
        available_years = sorted(available_year_ranges.keys(), reverse=True)
        
        # 전체 데이터 범위 계산
        all_start_dates = [available_year_ranges[year][0] for year in available_years]
        all_end_dates = [available_year_ranges[year][1] for year in available_years]
        global_start_date = min(all_start_dates)
        global_end_date = max(all_end_dates)
        
        # 데이터 범위 정보 표시
        with st.expander("📊 데이터 범위 정보"):
            st.info(f"📈 **전체 최적화 기간**: {global_start_date.date()} ~ {global_end_date.date()}")
            range_info = []
            for year in sorted(available_years, reverse=True):
                start_date, end_date = available_year_ranges[year]
                range_info.append(f"**{year}년**: {start_date.date()} ~ {end_date.date()}")
            st.write("\n".join(range_info))
        
        st.success(f"✅ 최적화 기간: {global_start_date.date()} ~ {global_end_date.date()}")
        st.caption("💡 팁: 전체 기간 최적화는 연도별 최적화보다 3배 빠릅니다!")
        
        initial_cash = st.number_input(
            "초기 자산 (원)",
            min_value=10_000_000,
            max_value=1_000_000_000,
            value=50_000_000,
            step=10_000_000,
            format="%d"
        )
        
        st.divider()
    
    with col2:
        st.markdown("#### ⚙️ 파라미터 범위 설정")
        
        st.markdown("**일일 최대 매수 종목 수**")
        col_a, col_b = st.columns(2)
        with col_a:
            max_daily_buys_min = st.number_input("최소", min_value=1, max_value=5, value=1, key="mdb_min")
        with col_b:
            max_daily_buys_max = st.number_input("최대", min_value=1, max_value=5, value=3, key="mdb_max")
        
        st.markdown("**직전 거래일 평균** (⚡5-20일 권장)")
        col_c, col_d = st.columns(2)
        with col_c:
            rolling_days_options = st.multiselect(
                "테스트할 값 선택",
                options=list(range(5, 31, 5)),
                default=[5, 10, 15, 20],
                key="rolling_options",
                help="5, 10, 15, 20만 권장"
            )
        
        st.markdown("**평균 대비 배수** (⚡1.5-3.0 권장)")
        col_e, col_f = st.columns(2)
        with col_e:
            volume_threshold_options = st.multiselect(
                "테스트할 값 선택",
                options=[x / 2 for x in range(3, 21)],
                default=[1.5, 2.0, 2.5, 3.0],
                key="vol_options",
                help="1.5, 2.0, 2.5, 3.0만 권장"
            )

        st.markdown("**추가매수 손실 임계값(%)** (⚡-7% 권장)")
        add_buy_threshold_options = st.multiselect(
            "테스트할 값 선택",
            options=[-float(x) for x in range(1, 11)],
            default=[-7.0],
            key="add_buy_threshold_options",
            help="⚡-7.0만 권장 (가장 안정적 & 매우 빠름)"
        )
        
        st.markdown("**매수 금액 단위 (만원)**")
        buy_unit_options = st.multiselect(
            "테스트할 값 선택",
            options=[100, 150, 200, 250, 300],
            default=[200],
            key="buy_unit_options",
            help="100=100만원, 200=200만원, 300=300만원"
        )
        
    
    # 최적화 기준 선택
    st.markdown("#### 🎯 최적화 기준")
    optimization_metric = st.selectbox(
        "어떤 지표를 기준으로 최적화할까요?",
        options=["total_return", "sharpe_ratio", "excess_return"],
        format_func=lambda x: {
            "total_return": "총 수익률",
            "sharpe_ratio": "샤프 비율 (위험 대비 수익)",
            "excess_return": "초과 수익률 (KOSPI 대비)"
        }[x]
    )

    st.markdown("#### 🔎 탐색 방식 (속도 최적화)")
    search_mode = st.selectbox(
        "최적화 방식",
        options=["random", "grid"],
        index=0,
        format_func=lambda x: "🚀 랜덤 서치 (빠름!)" if x == "random" else "그리드 서치 (느림)",
        help="랜덤: 빠르고 충분히 정확 | 그리드: 느리지만 완전한 탐색"
    )
    sample_count = st.number_input(
        "랜덤 샘플 수",
        min_value=50,
        max_value=5000,
        value=150,
        step=50,
        help="⚡팁: 150-300개 추천 (빠름), 1000개 이상은 매우 느림",
        disabled=(search_mode != "random")
    )
    random_seed = st.number_input(
        "랜덤 시드",
        min_value=0,
        max_value=99999,
        value=42,
        step=1,
        help="같은 시드면 동일한 샘플을 사용합니다",
        disabled=(search_mode != "random")
    )
    
    # 최적화 실행 버튼
    st.divider()
    
    # ⚡ 속도 최적화 팁
    with st.expander("⚡ 속도 향상 팁", expanded=False):
        st.markdown("""
        ### 🚀 빠른 최적화를 위한 권장 설정:
        
        **현재 설정 (권장):**
        - 🎯 탐색 방식: **랜덤 서치** (기본값)
        - 📊 샘플 수: **150개** (기본값)
        - 📈 매개변수 범위:
          - 직전 거래일: **5, 10, 15, 20** 만 선택 (기본값)
          - 평균 배수: **1.5, 2.0, 2.5, 3.0** 만 선택 (기본값)
          - 손실 임계값: **-7.0** 만 선택 (기본값)
        
        **성능 비교:**
        | 설정 | 예상 시간 | 품질 |
        |------|---------|------|
        | 권장 설정 | ⚡ 1-2분 | ✅ 충분함 |
        | 일반 설정 | 🟡 5-10분 | ✅ 동일 |
        | 전체 탐색 | 🐢 30분+ | ✅ 조금더 정밀함 |
        
        **⚡ 최적화 기술:**
        1. **랜덤 서치**: 그리드보다 3-5배 빠름
        2. **파라미터 축소**: 범위를 절반만 사용해도 충분함
        3. **병렬 처리**: CPU 코어 수 활용 (자동 최적화됨)
        4. **신호 사전 계산**: 이미 메인 프로세스에서 계산됨
        """)
    
    if st.button("🚀 범용 파라미터 최적화 실행", type="primary", use_container_width=True):
        # 파라미터 범위 검증
        if not rolling_days_options:
            st.error("직전 거래일 평균 값을 최소 1개 이상 선택해주세요.")
            return
        
        if not volume_threshold_options:
            st.error("평균 대비 배수 값을 최소 1개 이상 선택해주세요.")
            return

        if not add_buy_threshold_options:
            st.error("추가매수 손실 임계값을 최소 1개 이상 선택해주세요.")
            return
        
        if not buy_unit_options:
            st.error("매수 금액 단위를 최소 1개 이상 선택해주세요.")
            return
        
        if max_daily_buys_min > max_daily_buys_max:
            st.error("일일 최대 매수 종목 수의 최소값이 최대값보다 클 수 없습니다.")
            return
        
        # 파라미터 범위 구성 (만원 -> 원 변환)
        param_ranges = {
            'max_daily_buys': list(range(max_daily_buys_min, max_daily_buys_max + 1)),
            'rolling_days': sorted(rolling_days_options),
            'volume_threshold': sorted(volume_threshold_options),
            'add_buy_threshold_pct': sorted(add_buy_threshold_options),
            'buy_unit': [int(b) * 10_000 for b in sorted(buy_unit_options)]  # 만원 -> 원
        }
        
        # 총 조합 수 계산
        total_combinations = (
            len(param_ranges['max_daily_buys']) * 
            len(param_ranges['rolling_days']) * 
            len(param_ranges['volume_threshold']) *
            len(param_ranges['add_buy_threshold_pct']) *
            len(param_ranges['buy_unit'])
        )

        if search_mode == "random":
            sample_count = min(int(sample_count), total_combinations)
            if sample_count <= 0:
                st.error("랜덤 샘플 수가 0입니다. 파라미터 범위를 확인하세요.")
                return
        
        # ===== 전체 기간 최적화 (Unified Approach) =====
        st.markdown("---")
        
        # 예상 시간 계산
        total_combos = (
            len(param_ranges['max_daily_buys']) * 
            len(param_ranges['rolling_days']) * 
            len(param_ranges['volume_threshold']) *
            len(param_ranges['add_buy_threshold_pct']) *
            len(param_ranges['buy_unit'])
        )
        
        if search_mode == "random":
            test_count = min(int(sample_count), total_combos)
            estimate_time = max(1, test_count // 150)  # 분당 약 150개 처리
        else:
            test_count = total_combos
            estimate_time = max(2, test_count // 100)  # 분당 약 100개 처리
        
        st.info(f"📊 **전체 기간 최적화 중...**\n예상 시간: **{estimate_time}분** | 테스트 조합: {test_count}개 | 데이터 범위: {global_start_date.date()} ~ {global_end_date.date()}")
        
        # 프로그레스 바 설정
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_start = datetime.now()
        
        def update_progress(current, total, params):
            import time
            elapsed = (datetime.now() - time_start).total_seconds()
            progress = current / total
            progress_bar.progress(progress)
            
            # 예상 남은 시간 계산
            if progress > 0.05:  # 최소 5% 이상 진행했을 때만 계산
                estimated_total = elapsed / progress
                estimated_remaining = estimated_total - elapsed
                time_str = f"{int(estimated_remaining)}초"
                if estimated_remaining > 60:
                    time_str = f"{estimated_remaining/60:.1f}분"
            else:
                time_str = "계산 중..."
            
            buy_unit_man = params.get('buy_unit', 2_000_000) // 10_000  # 원 -> 만원
            status_text.text(
                f"진행: {current}/{total} ({int(progress*100)}%) | "
                f"남은 시간: {time_str} | "
                f"매수: {params.get('max_daily_buys')}개, "
                f"평균: {params.get('rolling_days')}일, "
                f"배수: {params.get('volume_threshold')}"
            )
        
        # 최적화 실행 (전체 기간, 한 번만!)
        optimizer = BacktestOptimizer(df, kospi_index)
        
        try:
            results_df = optimizer.optimize_parameters(
                start_date=global_start_date,
                end_date=global_end_date,
                param_ranges=param_ranges,
                initial_cash=initial_cash,
                progress_callback=update_progress,
                search_mode=search_mode,
                sample_size=sample_count,
                random_seed=random_seed,
            )
            
            progress_bar.progress(1.0)
            
            if results_df.empty:
                st.warning(f"⚠️ 최적화 결과가 없습니다.")
                
                # 디버깅 정보 표시
                with st.expander("🔍 진단 정보"):
                    st.write(f"""
                    **데이터 범위**: {global_start_date.date()} ~ {global_end_date.date()}
                    **파라미터 범위**: 
                    - max_daily_buys: {param_ranges['max_daily_buys']}
                    - rolling_days: {param_ranges['rolling_days']}
                    - volume_threshold: {param_ranges['volume_threshold']}
                    - add_buy_threshold_pct: {param_ranges['add_buy_threshold_pct']}
                    
                    **가능한 원인**:
                    1. 선택한 기간에 거래 데이터가 부족합니다
                    2. 모든 백테스트 조합에서 오류가 발생했습니다 (터미널 로그 확인)
                    3. 시그널이 생성되지 않았습니다
                    
                    **해결 방법**:
                    - 파라미터 범위를 줄여보세요
                    - 더 많은 데이터를 로드해보세요
                    - 터미널 출력을 확인하세요
                    """)
                
            else:
                # 최적 파라미터 찾기
                optimal_params = optimizer.get_optimal_params(results_df, optimization_metric)
                
                # 최적화 완료 메시지
                st.success(f"✅ 최적화 완료! {len(results_df)}개 조합 테스트 완료")
                
                # ===== 최적 범용 파라미터 표시 =====
                st.markdown("### 🏆 최적 범용 파라미터")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("일일 최대 매수", f"{optimal_params['max_daily_buys']}개")
                    st.metric("직전 거래일 평균", f"{optimal_params['rolling_days']}일")
                
                with col2:
                    st.metric("평균 대비 배수", f"{optimal_params['volume_threshold']:.2f}배")
                    st.metric("추가매수 손절", f"{optimal_params['add_buy_threshold_pct']:.1f}%")
                
                with col3:
                    buy_unit_man = optimal_params.get('buy_unit', 2_000_000) // 10_000
                    st.metric("매수 금액 단위", f"{int(buy_unit_man)}만원")
                    metric_name = {
                        "total_return": "총 수익률",
                        "sharpe_ratio": "샤프 비율",
                        "excess_return": "초과 수익률"
                    }[optimization_metric]
                    metric_value = optimal_params[f'best_{optimization_metric}']
                    if optimization_metric in ["total_return", "excess_return"]:
                        st.metric(f"🎯 {metric_name}", f"{metric_value:.2f}%")
                    else:
                        st.metric(f"🎯 {metric_name}", f"{metric_value:.4f}")
                
                # ===== 성과 지표 =====
                st.markdown("### 📊 성과 지표")
                
                metric_cols = st.columns(5)
                
                with metric_cols[0]:
                    st.metric("총 수익률", f"{optimal_params['best_total_return']:.2f}%")
                
                with metric_cols[1]:
                    st.metric("KOSPI 수익률", f"{optimal_params['best_kospi_return']:.2f}%")
                
                with metric_cols[2]:
                    st.metric("초과 수익률", f"{optimal_params['best_excess_return']:.2f}%")
                
                with metric_cols[3]:
                    st.metric("샤프 비율", f"{optimal_params['best_sharpe_ratio']:.4f}")
                
                with metric_cols[4]:
                    st.metric("최대 낙폭", f"{optimal_params['best_mdd']:.2f}%")
                
                # ===== 거래 통계 =====
                st.markdown("### 💹 거래 통계")
                
                trade_cols = st.columns(3)
                
                with trade_cols[0]:
                    st.metric("총 거래 횟수", f"{optimal_params['total_trades']}회")
                
                with trade_cols[1]:
                    st.metric("승률", f"{optimal_params['win_rate']:.2f}%")
                
                with trade_cols[2]:
                    st.metric("기간", f"{(global_end_date - global_start_date).days}일")
                
                # ===== 결과 상세 정보 =====
                with st.expander("📈 상세 결과 (모든 테스트 조합)"):
                    # 상위 10개 결과
                    top_10 = results_df.nlargest(10, optimization_metric)
                    display_cols = ['max_daily_buys', 'rolling_days', 'volume_threshold', 
                                   'add_buy_threshold_pct', 'total_return', 'sharpe_ratio', 
                                   'excess_return', 'total_trades', 'win_rate']
                    
                    # buy_unit는 원 단위이므로 만원으로 변환해서 표시
                    display_df = top_10[display_cols].copy()
                    if 'buy_unit' in top_10.columns:
                        display_df.insert(4, '매수금액(만원)', top_10['buy_unit'] // 10_000)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # 다운로드 버튼
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 전체 결과 CSV 다운로드",
                        data=csv_data,
                        file_name=f"optimization_results_{global_start_date.strftime('%Y%m%d')}_{global_end_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                # ===== 최적 파라미터 저장 제안 =====
                st.markdown("### 💾 최적 파라미터 저장")
                
                save_col1, save_col2 = st.columns(2)
                
                with save_col1:
                    if st.button("💾 최적 파라미터를 설정으로 저장", use_container_width=True, type="primary"):
                        from app.settings import UserSettings
                        settings_mgr = UserSettings()
                        
                        settings_mgr.set('optimal_max_daily_buys', optimal_params['max_daily_buys'])
                        settings_mgr.set('optimal_rolling_days', optimal_params['rolling_days'])
                        settings_mgr.set('optimal_volume_threshold', optimal_params['volume_threshold'])
                        settings_mgr.set('optimal_add_buy_threshold_pct', optimal_params['add_buy_threshold_pct'])
                        settings_mgr.set('optimal_buy_unit', optimal_params.get('buy_unit', 2_000_000))
                        
                        st.success("✅ 최적 파라미터가 설정으로 저장되었습니다!")
                
                with save_col2:
                    # 최적 파라미터를 JSON으로 표시
                    optimal_json = {
                        'max_daily_buys': int(optimal_params['max_daily_buys']),
                        'rolling_days': int(optimal_params['rolling_days']),
                        'volume_threshold': float(optimal_params['volume_threshold']),
                        'add_buy_threshold_pct': float(optimal_params['add_buy_threshold_pct']),
                        'buy_unit': int(optimal_params.get('buy_unit', 2_000_000)),
                        'optimization_date': global_end_date.strftime('%Y-%m-%d'),
                        'test_period': f"{global_start_date.strftime('%Y-%m-%d')} ~ {global_end_date.strftime('%Y-%m-%d')}"
                    }
                    
                    json_str = json.dumps(optimal_json, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📥 파라미터 JSON 다운로드",
                        data=json_str,
                        file_name=f"optimal_params_{global_end_date.strftime('%Y%m%d')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"⚠️ 최적화 중 오류 발생")
            
            # 상세 에러 정보 표시
            with st.expander("🔍 오류 상세 정보"):
                import traceback
                st.code(f"Error: {str(e)}", language="text")
                st.code(traceback.format_exc(), language="python")


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

        # 시그널 생성 (시그널 탭 전용)
        with st.spinner("🔍 시그널 분석 중..."):
            signals = build_signals(
                df,
                params["turnover_window"],
                params["turnover_multiplier"],
                20,
                5.0,
                20,
                2.0,
                20,
                2.0,
                ["Turnover Spike"],
                "ANY",
            )
            signals = signals.merge(kospi, on="code", how="left")

            if "spike_ratio" in signals.columns:
                signals = signals.sort_values(["date", "spike_ratio"], ascending=[False, False])
            else:
                signals = signals.sort_values(["date"], ascending=[False])
            signals = apply_signal_filters(signals, params["signal_filter"])

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
        
        # 파라미터 조정 섹션
        with st.expander("⚙️ 거래량 분석 파라미터 설정", expanded=True):
            st.caption("파라미터를 조정하고 백테스트를 실행하세요")
            
            col_roll, col_vol, col_add = st.columns(3)
            
            with col_roll:
                rolling_days = st.slider(
                    "분석 기간 (일)",
                    min_value=5, max_value=60, 
                    value=params["turnover_window"], 
                    step=1,
                    help="거래량 평균을 계산하는 기간",
                    key="backtest_rolling_days"
                )
            
            with col_vol:
                volume_threshold = st.slider(
                    "급등 기준 (배수)",
                    min_value=1.0, max_value=10.0, 
                    value=params["turnover_multiplier"], 
                    step=0.1,
                    help="평균 거래량 대비 배수",
                    key="backtest_volume_threshold"
                )
            
            with col_add:
                loss_threshold = st.slider(
                    "추가매수 손실 임계값 (%)",
                    min_value=-30.0, max_value=-1.0, 
                    value=params["add_buy_threshold_pct"], 
                    step=0.5,
                    help="이 수익률 이하일 때 추가매수 고려",
                    key="backtest_loss_threshold"
                )
            
            # 현재 설정 표시
            st.markdown("---")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("📅 분석 기간", f"{int(rolling_days)}일")
            with col_info2:
                st.metric("📈 급등 기준", f"{volume_threshold:.1f}배")
            with col_info3:
                st.metric("🔴 손절 기준", f"{loss_threshold:.1f}%")
        
        # 데이터 로딩 (시뮬레이션 탭 전용)
        with st.spinner("📊 데이터 로딩 중..."):
            df = load_stock_data(params["data_dir"])
            kospi = load_kospi_list(params["data_dir"])
            kospi_index = load_kospi_index(params["data_dir"])

        # 시그널 생성 (시뮬레이션 탭 전용) - 조정된 파라미터 사용
        with st.spinner("🔍 시그널 분석 중..."):
            signals = build_signals(
                df,
                int(rolling_days),
                float(volume_threshold),
                20,
                5.0,
                20,
                2.0,
                20,
                2.0,
                ["Turnover Spike"],
                "ANY",
            )
            signals = signals.merge(kospi, on="code", how="left")

            if "spike_ratio" in signals.columns:
                signals = signals.sort_values(["date", "spike_ratio"], ascending=[False, False])
            else:
                signals = signals.sort_values(["date"], ascending=[False])
            signals = apply_signal_filters(signals, params["signal_filter"])

        selected_date = select_date(signals)
        if selected_date is None:
            st.warning("⚠️ 조건에 맞는 시그널이 없습니다.")
            return

        st.divider()
        st.subheader("📊 백테스트 결과")
        strategy_signals = build_signals(
            df,
            int(rolling_days),
            float(volume_threshold),
            20,
            5.0,
            20,
            2.0,
            20,
            2.0,
            ["Turnover Spike"],
            "ANY",
        )
        strategy_signals = strategy_signals.merge(kospi, on="code", how="left")

        equity_df, trades_df = run_turnover_strategy_backtest(
            df,
            strategy_signals,
            kospi_index,
            selected_date,
            top_n=2,
            initial_cash=params["initial_cash"],
            max_daily_buys=params["max_daily_buys"],
            buy_unit=params["buy_unit"],
            add_buy_threshold_pct=float(loss_threshold),
        )
        render_backtest_curve(equity_df, kospi_index, selected_date)
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

    elif current_tab == "⚙️ 최적화":
        # 데이터 로딩 (최적화 탭 전용)
        with st.spinner("📊 데이터 로딩 중..."):
            df = load_stock_data(params["data_dir"])
            kospi_index = load_kospi_index(params["data_dir"])

        render_optimizer_page(df, kospi_index, params)
    
    else:
        render_kospi_crawling_page()
