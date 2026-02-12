import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from crawling_kospi import CrawlingKospi
from naver_news_crawler import NaverNewsCrawler
from kakao_message import KakaoMessageSender
from optimizer import BacktestOptimizer, get_period_dates
from sentiment_analyzer import SentimentAnalyzer
from stock_ontology import build_stock_ontology, StockOntology

from .data import DATA_DIR_DEFAULT, load_kospi_index, load_kospi_list, load_stock_data, load_finance_data
from .signals import build_signals
import datetime


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
                if 'applied_kospi_bullish' in st.session_state:
                    del st.session_state.applied_kospi_bullish
                st.rerun()
        
        # 적용된 파라미터 값 사용 (있으면)
        default_window = st.session_state.get('applied_turnover_window', 30)
        default_multiplier = st.session_state.get('applied_turnover_multiplier', 2.0)
        default_kospi_bullish = st.session_state.get('applied_kospi_bullish', True)
        
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
            kospi_bullish_only = st.toggle(
                "📈 코스피 양봉일 때만 매수",
                value=default_kospi_bullish,
                help="켜면: 코스피 지수가 전일 대비 상승(양봉)인 날에만 BUY 시그널 표시",
                key="kospi_bullish_signal"
            )

        # 백테스트 설정은 시뮬레이션 탭에서만 표시
        initial_cash = 50_000_000
        max_daily_buys = 2
        
        # 매수 금액 단위 기본값
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
                    min_value=50, max_value=1000, value=200, step=100,
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
        "kospi_bullish_only": bool(kospi_bullish_only),
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


def render_table_with_finance(latest: pd.DataFrame, cols: list[str], finance_df: pd.DataFrame, price_df: pd.DataFrame = None) -> None:
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
        
        # 최상위 종목(idx=0)은 자동으로 펼치기
        auto_expand = (idx == 0)
        
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    kospi_bullish_dates = set()
    if kospi_bullish_only and not kospi_index.empty:
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
        shares = int(spend // price)
        if shares == 0:
            return
        actual_amount = shares * price
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
            positions[code] = {"shares": shares, "avg_cost": price, "step": step}
        cash -= actual_amount
        trades.append({
            "date": date,
            "code": code,
            "name": name_map.get(code, ""),
            "action": "BUY",
            "amount": actual_amount,
            "price": price,
            "shares": shares,
            "step": step,
            "return_pct": None,
        })

    def _sell(code: str, date: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal cash
        if code not in positions:
            return
        pos = positions.pop(code)
        proceeds = pos["shares"] * price
        cash += proceeds
        pnl = (price - pos["avg_cost"]) * pos["shares"]
        trades.append({
            "date": date,
            "code": code,
            "name": name_map.get(code, ""),
            "action": "SELL",
            "amount": proceeds,
            "price": price,
            "shares": pos["shares"],
            "pnl": pnl,
            "buy_price": pos["avg_cost"],
            "return_pct": ((price / pos["avg_cost"]) - 1) * 100 if pos["avg_cost"] > 0 else 0,
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
            
            if ret <= -0.05:
                if pos["step"] == 0:
                    # step 0: 첫 번째 추가 매수 (다음날 시작가로)
                    if next_date:
                        if next_date not in pending_buys:
                            pending_buys[next_date] = []
                        pending_buys[next_date].append((code, buy_unit, 1))
                elif pos["step"] == 1:
                    # step 1: 두 번째 추가 매수 (다음날 시작가로)
                    if next_date:
                        if next_date not in pending_buys:
                            pending_buys[next_date] = []
                        pending_buys[next_date].append((code, buy_unit * 2, 2))
                else:
                    # step 2 이상: 즉시 손절 (평균매수가의 -5% 가격)
                    stop_price = pos["avg_cost"] * 0.95
                    _sell(code, date, stop_price, "STOP_LOSS")

        # 시그널 처리
        if next_date:
            # 매수 시그널 (코스피 양봉 조건 체크)
            if date in buy_by_date:
                # 코스피 양봉 조건이 켜져 있으면, 해당 날짜가 양봉인지 확인
                can_buy = True
                if kospi_bullish_only and kospi_bullish_dates:
                    can_buy = (date in kospi_bullish_dates)
                
                if can_buy:
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
    """파라미터 최적화 페이지 렌더링"""
    st.subheader("🎯 파라미터 최적화")
    st.caption("최적의 투자 전략 파라미터를 찾습니다")
    st.write("다양한 파라미터 조합을 테스트하여 최적의 전략을 찾습니다.")
    
    # 설정 영역을 컬럼으로 나누기
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 투자 기간 설정")
        period = st.selectbox(
            "백테스트 기간",
            options=["1Y", "3Y", "5Y"],
            index=1,
            format_func=lambda x: {"1Y": "1년", "3Y": "3년", "5Y": "5년"}[x]
        )
        
        # 데이터의 최대 날짜 가져오기
        max_date = pd.to_datetime(df['date'].max()).normalize()
        start_date, end_date = get_period_dates(period, max_date)
        
        st.info(f"테스트 기간: {start_date.date()} ~ {end_date.date()}")
        
        initial_cash = st.number_input(
            "초기 자산 (원)",
            min_value=10_000_000,
            max_value=1_000_000_000,
            value=50_000_000,
            step=10_000_000,
            format="%d"
        )
    
    with col2:
        st.markdown("#### ⚙️ 파라미터 범위 설정")
        
        st.markdown("**일일 최대 매수 종목 수**")
        col_a, col_b = st.columns(2)
        with col_a:
            max_daily_buys_min = st.number_input("최소", min_value=1, max_value=5, value=1, key="mdb_min")
        with col_b:
            max_daily_buys_max = st.number_input("최대", min_value=1, max_value=5, value=3, key="mdb_max")
        
        st.markdown("**직전 거래일 평균**")
        col_c, col_d = st.columns(2)
        with col_c:
            rolling_days_options = st.multiselect(
                "테스트할 값 선택",
                options=[10, 15, 20, 30, 40, 60],
                default=[10, 20, 30],
                key="rolling_options"
            )
        
        st.markdown("**평균 대비 배수**")
        col_e, col_f = st.columns(2)
        with col_e:
            volume_threshold_options = st.multiselect(
                "테스트할 값 선택",
                options=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                default=[1.5, 2.0, 2.5, 3.0],
                key="vol_options"
            )
        
        st.markdown("**코스피 양봉 조건**")
        kospi_bullish_options = st.multiselect(
            "ON/OFF 테스트",
            options=[False, True],
            default=[False, True],
            format_func=lambda x: "ON (양봉일때만 매수)" if x else "OFF (항상 매수)",
            key="kospi_bullish_options"
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
    
    # 최적화 실행 버튼
    st.divider()
    
    if st.button("🚀 최적화 실행", type="primary", use_container_width=True):
        # 파라미터 범위 검증
        if not rolling_days_options:
            st.error("직전 거래일 평균 값을 최소 1개 이상 선택해주세요.")
            return
        
        if not volume_threshold_options:
            st.error("평균 대비 배수 값을 최소 1개 이상 선택해주세요.")
            return
        
        if not kospi_bullish_options:
            st.error("코스피 양봉 조건을 최소 1개 이상 선택해주세요.")
            return
        
        if max_daily_buys_min > max_daily_buys_max:
            st.error("일일 최대 매수 종목 수의 최소값이 최대값보다 클 수 없습니다.")
            return
        
        # 파라미터 범위 구성
        param_ranges = {
            'max_daily_buys': list(range(max_daily_buys_min, max_daily_buys_max + 1)),
            'rolling_days': sorted(rolling_days_options),
            'volume_threshold': sorted(volume_threshold_options),
            'kospi_bullish_only': kospi_bullish_options
        }
        
        # 총 조합 수 계산
        total_combinations = (
            len(param_ranges['max_daily_buys']) * 
            len(param_ranges['rolling_days']) * 
            len(param_ranges['volume_threshold']) *
            len(param_ranges['kospi_bullish_only'])
        )
        
        st.info(f"총 {total_combinations}개의 파라미터 조합을 테스트합니다. 시간이 다소 걸릴 수 있습니다.")
        
        # 프로그레스 바 설정
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, params):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(
                f"진행중: {current}/{total} "
                f"(매수종목: {params.get('max_daily_buys')}, "
                f"평균일: {params.get('rolling_days')}, "
                f"배수: {params.get('volume_threshold')})"
            )
        
        # 최적화 실행
        optimizer = BacktestOptimizer(df, kospi_index)
        
        try:
            results_df = optimizer.optimize_parameters(
                start_date=start_date,
                end_date=end_date,
                param_ranges=param_ranges,
                initial_cash=initial_cash,
                progress_callback=update_progress
            )
            
            progress_bar.progress(1.0)
            status_text.text(f"완료! {len(results_df)}개 결과 분석 중...")
            
            # 디버깅: 결과 확인
            st.write(f"📊 디버깅 정보: 총 {total_combinations}개 조합 중 {len(results_df)}개 성공")
            
            if results_df.empty:
                st.error("⚠️ 최적화 결과가 없습니다.")
                st.info("""
                가능한 원인:
                1. 선택한 기간에 거래 데이터가 부족합니다
                2. 모든 백테스트에서 오류가 발생했습니다
                3. 시그널이 생성되지 않았습니다
                
                해결 방법:
                - 더 긴 기간(3년 또는 5년)을 선택해보세요
                - 파라미터 범위를 조정해보세요
                - 터미널 출력을 확인하여 상세 오류를 확인하세요
                """)
                return
            
            # 최적 파라미터 찾기
            optimal_params = optimizer.get_optimal_params(results_df, optimization_metric)
            
            # 결과 표시
            st.success("✅ 최적화 완료!")
            
            # 최적 파라미터 표시
            st.markdown("### 🏆 최적 파라미터")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("일일 최대 매수 종목", f"{optimal_params['max_daily_buys']}개")
            with col2:
                st.metric("직전 거래일 평균", f"{optimal_params['rolling_days']}일")
            with col3:
                st.metric("평균 대비 배수", f"{optimal_params['volume_threshold']}배")
            with col4:
                kospi_bullish_val = optimal_params.get('kospi_bullish_only', False)
                st.metric("코스피 양봉", "ON" if kospi_bullish_val else "OFF")
            with col5:
                metric_name = {
                    "total_return": "총 수익률",
                    "sharpe_ratio": "샤프 비율",
                    "excess_return": "초과 수익률"
                }[optimization_metric]
                metric_value = optimal_params[f'best_{optimization_metric}']
                if optimization_metric in ["total_return", "excess_return"]:
                    st.metric(metric_name, f"{metric_value:.2f}%")
                else:
                    st.metric(metric_name, f"{metric_value:.3f}")
            
            # 파라미터 적용 버튼
            st.markdown("")
            if st.button("🎯 이 파라미터로 시그널 설정 적용", type="primary", use_container_width=True):
                # session_state에 최적 파라미터 저장
                st.session_state.applied_turnover_window = optimal_params['rolling_days']
                st.session_state.applied_turnover_multiplier = optimal_params['volume_threshold']
                st.session_state.applied_kospi_bullish = optimal_params.get('kospi_bullish_only', False)
                st.session_state.optimal_params_applied = True
                st.success("✅ 파라미터가 적용되었습니다! '📊 시그널' 탭에서 확인하세요.")
                st.balloons()
            
            # 상위 결과 표시
            st.markdown("### 📈 상위 10개 결과")
            top_results = results_df.nlargest(10, optimization_metric).copy()
            
            # 컬럼명 한글화
            display_df = top_results[[
                'max_daily_buys', 'rolling_days', 'volume_threshold',
                'total_return', 'kospi_return', 'excess_return',
                'sharpe_ratio', 'mdd', 'win_rate', 'total_trades'
            ]].copy()
            
            display_df.columns = [
                '일일매수', '평균일', '배수',
                '수익률(%)', 'KOSPI(%)', '초과수익(%)',
                '샤프비율', 'MDD(%)', '승률(%)', '거래수'
            ]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    '일일매수': st.column_config.NumberColumn(format="%d"),
                    '평균일': st.column_config.NumberColumn(format="%d"),
                    '배수': st.column_config.NumberColumn(format="%.1f"),
                    '코스피양봉': st.column_config.TextColumn(),
                    '수익률(%)': st.column_config.NumberColumn(format="%.2f"),
                    'KOSPI(%)': st.column_config.NumberColumn(format="%.2f"),
                    '초과수익(%)': st.column_config.NumberColumn(format="%.2f"),
                    '샤프비율': st.column_config.NumberColumn(format="%.3f"),
                    'MDD(%)': st.column_config.NumberColumn(format="%.2f"),
                    '승률(%)': st.column_config.NumberColumn(format="%.2f"),
                    '거래수': st.column_config.NumberColumn(format="%d"),
                }
            )
            
            # 전체 결과 다운로드
            st.markdown("### 💾 전체 결과 다운로드")
            csv = results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f"optimization_results_{period}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # 시각화
            st.markdown("### 📊 파라미터별 성과 분석")
            
            # 각 파라미터별 평균 수익률
            tab1, tab2, tab3, tab4 = st.tabs(["일일 매수 종목수", "평균 거래일", "평균 대비 배수", "코스피 양봉 조건"])
            
            with tab1:
                avg_by_buys = results_df.groupby('max_daily_buys')[optimization_metric].mean().reset_index()
                st.bar_chart(avg_by_buys.set_index('max_daily_buys'))
            
            with tab2:
                avg_by_days = results_df.groupby('rolling_days')[optimization_metric].mean().reset_index()
                st.bar_chart(avg_by_days.set_index('rolling_days'))
            
            with tab3:
                avg_by_threshold = results_df.groupby('volume_threshold')[optimization_metric].mean().reset_index()
                st.bar_chart(avg_by_threshold.set_index('volume_threshold'))
            
            with tab4:
                avg_by_kospi = results_df.groupby('kospi_bullish_only')[optimization_metric].mean().reset_index()
                avg_by_kospi['kospi_bullish_only'] = avg_by_kospi['kospi_bullish_only'].apply(lambda x: 'ON (양봉만)' if x else 'OFF (항상)')
                avg_by_kospi = avg_by_kospi.set_index('kospi_bullish_only')
                st.bar_chart(avg_by_kospi)
            
        except Exception as e:
            st.error(f"최적화 중 오류가 발생했습니다: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def run_app(current_tab: str = "📊 시그널") -> None:
    # 페이지 설정은 streamlit_app.py에서 수행
    st.title("📈 StockVibe")
    st.caption("거래량 급등 기반 스마트 투자 시그널")

    # 상위 사이드바에서 선택한 탭 사용
    available_tabs = ["📊 시그널", "🎯 시뮬레이션", "⚙️ 최적화", " 데이터"]
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
        
        st.subheader("🧭 필터")
        filter_col1, filter_col2, filter_col3 = st.columns([2, 1.5, 1.2])
        with filter_col1:
            st.caption(f"기준일: {selected_date.date()}")
            st.caption(f"표시 종목 수: 상위 {params['top_n']}개")
            st.caption(f"코스피 양봉 조건: {'ON' if params['kospi_bullish_only'] else 'OFF'}")
        with filter_col2:
            st.caption("결과 표시 방식")
            view_mode = st.radio(
                "표시 방식",
                ["📊 테이블", "📋 상세"],
                horizontal=True,
                label_visibility="collapsed",
                key="signal_view_mode",
            )
        with filter_col3:
            if is_kospi_bullish:
                st.success(f"📈 양봉 ({kospi_change_pct:+.2f}%)")
            else:
                st.error(f"📉 음봉 ({kospi_change_pct:+.2f}%)")
        
        # 코스피 양봉 조건이 켜져 있고 음봉일 때 경고
        if params["kospi_bullish_only"] and not is_kospi_bullish:
            st.warning("⚠️ 코스피가 음봉이므로 BUY 시그널이 필터링됩니다. (양봉 조건 ON)")
            # BUY 시그널 필터링
            signals = signals[signals["signal"] != "BUY"]

        st.divider()
        st.subheader(f"📊 결과 · {selected_date.date()} 투자 시그널")
        st.caption(f"현재 보기: {view_mode} · 조건에 맞는 종목만 표시됩니다.")
        
        latest, cols = build_latest_table(signals, selected_date, params["top_n"], finance_df, df)

        if latest.empty:
            st.info("표시할 시그널 결과가 없습니다.")
            st.caption("분석 기간, 급등 기준, 표시 종목 수를 조정하면 결과가 나타날 수 있습니다.")
        elif view_mode == "📊 테이블":
            st.caption(f"조회 결과: 총 {len(latest)}개 종목")
            render_table(latest, cols)
        else:
            st.caption(f"조회 결과: 총 {len(latest)}개 종목")
            render_table_with_finance(latest, cols, finance_df, df)

        st.divider()
        st.subheader("📤 액션")
        st.caption(f"공유 기준일: {selected_date.date()}")
        if latest.empty:
            st.caption("전송 가능한 시그널이 없어 공유 액션이 비활성 상태입니다.")
        else:
            render_kakao_section(latest, selected_date)

    elif current_tab == "🎯 시뮬레이션":
        # 데이터 로딩 (시뮬레이션 탭 전용)
        with st.spinner("📊 데이터 로딩 중..."):
            df = load_stock_data(params["data_dir"])
            kospi = load_kospi_list(params["data_dir"])
            kospi_index = load_kospi_index(params["data_dir"])

        # 시그널 생성 (시뮬레이션 탭 전용)
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
        st.subheader("📊 백테스트 결과")
        strategy_signals = build_signals(
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
            kospi_bullish_only=params["kospi_bullish_only"],
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
