"""
Phase 4 통합 페이지 - 모든 신규 기능을 담은 메뉴 시스템
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Phase 4 모듈 import
from app.portfolio import PortfolioManager
from app.backtesting import BacktestingEngine
from app.comparison import ComparisonAnalyzer, SectorAnalyzer
from app.export import DataExporter, ReportGenerator, ChartExporter
from app.advanced_charts import CandlePatternRecognizer, CorrelationAnalyzer, VolumeAnalyzer
from app.settings import UserSettings, ThemeManager, DisplaySettings

# 기존 모듈
from crawling_kospi import CrawlingKospi
from app.data import load_stock_data


def render_sidebar_menu():
    """사이드바 메뉴"""
    st.sidebar.title("📊 StockVibe Pro")
    st.sidebar.markdown("---")
    
    # 페이지 선택
    pages = {
        "🏠 메인 대시보드": "main",
        "💼 포트폴리오": "portfolio",
        "📈 백테스팅": "backtesting",
        "🌐 섹터 분석": "sector",
        "⚙️ 설정": "settings"
    }
    
    selected = st.sidebar.radio("메뉴", list(pages.keys()))
    
    st.sidebar.markdown("---")
    
    # 빠른 액션
    st.sidebar.subheader("⚡ 빠른 액션")
    if st.sidebar.button("🔄 데이터 새로고침"):
        st.session_state.data_refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()
    
    if st.sidebar.button("📊 전체 리포트 생성"):
        st.session_state.show_full_report = True
    
    return pages[selected]


def page_portfolio():
    """포트폴리오 관리 페이지"""
    st.title("💼 포트폴리오 관리")
    
    portfolio_mgr = PortfolioManager()
    
    # KOSPI 200 데이터 로드 함수 (캐싱)
    @st.cache_data(ttl=3600)
    def load_kospi_stocks():
        try:
            crawler = CrawlingKospi()
            kospi_dict = crawler.GetKospi200()
            return kospi_dict
        except:
            return {}
    
    @st.cache_data(ttl=3600)
    def get_kospi_df():
        try:
            crawler = CrawlingKospi()
            df = crawler.get_all_kospi_data()
            if not df.empty:
                # 종목명(종목코드) 형식으로 변환
                df['display_name'] = df['종목명'] + ' (' + df['종목코드'] + ')'
            return df
        except:
            return pd.DataFrame()
    
    # 탭 구성
    tab1, tab2 = st.tabs(["📊 보유 종목", "➕ 종목 추가"])
    
    with tab1:
        st.subheader("보유 종목 현황")
        
        # 메인 대시보드에서 종목 하이라이트 요청이 있는 경우
        highlight_stock = st.session_state.get('portfolio_highlight_stock', None)
        if highlight_stock:
            st.info(f"💡 선택한 종목: **{highlight_stock}** - 아래 목록에서 확인하세요")
            # 한번 표시 후 제거
            st.session_state.portfolio_highlight_stock = None
        
        # 손절 기준 설정
        col_setting1, col_setting2 = st.columns([3, 1])
        with col_setting1:
            st.write("")
        with col_setting2:
            stop_loss_rate = st.number_input(
                "손절 기준 (%)", 
                min_value=-50.0, 
                max_value=0.0, 
                value=-5.0, 
                step=0.5,
                help="손실이 이 비율에 도달하면 손절 알림을 표시합니다",
                key="stop_loss_rate"
            )
        
        portfolio = portfolio_mgr.load_portfolio()
        
        if not portfolio:
            st.info("보유 종목이 없습니다. '종목 추가' 탭에서 추가해주세요.")
        else:
            # 현재가 가져오기
            current_prices = {}
            price_histories = {}
            crawler = CrawlingKospi()
            
            # KOSPI 200 종목 목록 로드 (종목명->종목코드 매핑용)
            kospi_stocks = load_kospi_stocks()
            name_to_code = {name: code for code, name in kospi_stocks.items()}
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            for idx, code in enumerate(portfolio.keys()):
                progress_text.text(f"데이터 수집 중... ({idx+1}/{len(portfolio)})")
                progress_bar.progress((idx + 1) / len(portfolio))
                
                # 종목코드 확인 (종목명인 경우 종목코드로 변환)
                actual_code = code
                if code in name_to_code:
                    actual_code = name_to_code[code]
                elif not code.isdigit() or len(code) != 6:
                    # 종목코드가 아닌 경우 (직접 입력한 종목명)
                    # 매입단가를 현재가로 사용
                    current_prices[code] = portfolio[code]['avg_price']
                    continue
                
                try:
                    # GetCurrentPrice로 현재가 빠르게 가져오기
                    price_data = crawler.GetCurrentPrice(actual_code)
                    
                    if price_data and len(price_data) > 0:
                        current_prices[code] = price_data[0][3]  # pClose (종가)
                        # 가격 히스토리는 기존 방식 유지 (매도 타이밍 분석용)
                        try:
                            df = load_stock_data(actual_code, period="1mo")
                            if not df.empty:
                                price_histories[code] = df
                        except:
                            pass  # 히스토리 데이터 실패해도 현재가는 유지
                    else:
                        # GetCurrentPrice 실패 시 yfinance로 현재가 조회 시도
                        try:
                            df = load_stock_data(actual_code, period="1d")
                            if not df.empty:
                                current_prices[code] = df['Close'].iloc[-1]
                                # 한달 데이터도 가져오기
                                df_month = load_stock_data(actual_code, period="1mo")
                                if not df_month.empty:
                                    price_histories[code] = df_month
                            else:
                                current_prices[code] = 0
                                st.warning(f"⚠️ {portfolio[code]['name']} ({actual_code}) 현재가 조회 실패")
                        except:
                            current_prices[code] = 0
                            st.warning(f"⚠️ {portfolio[code]['name']} ({actual_code}) 현재가 조회 실패")
                except Exception as e:
                    # 오류 발생 시 yfinance로 재시도
                    try:
                        df = load_stock_data(actual_code, period="1d")
                        if not df.empty:
                            current_prices[code] = df['Close'].iloc[-1]
                        else:
                            current_prices[code] = 0
                            st.warning(f"⚠️ {portfolio[code]['name']} ({actual_code}) 현재가 조회 실패: {str(e)}")
                    except:
                        current_prices[code] = 0
                        st.warning(f"⚠️ {portfolio[code]['name']} ({actual_code}) 현재가 조회 실패: {str(e)}")
            
            progress_text.empty()
            progress_bar.empty()
            
            # 포트폴리오 테이블
            portfolio_df = portfolio_mgr.calculate_portfolio_value(current_prices)
            
            if not portfolio_df.empty:
                # 실시간 새로고침 버튼과 마지막 업데이트 시간
                col_refresh1, col_refresh2, col_refresh3 = st.columns([2, 1, 1])
                with col_refresh1:
                    if 'last_refresh' not in st.session_state:
                        st.session_state.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.info(f"📅 마지막 업데이트: {st.session_state.last_refresh}")
                
                with col_refresh2:
                    if st.button("🔄 실시간 새로고침", use_container_width=True):
                        st.session_state.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.rerun()
                
                with col_refresh3:
                    # 자동 새로고침 설정
                    auto_refresh = st.checkbox("자동 새로고침 (30초)", value=False)
                
                if auto_refresh:
                    import time
                    time.sleep(30)
                    st.session_state.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.rerun()
                
                st.markdown("---")
                
                # 요약 통계
                summary = portfolio_mgr.get_portfolio_summary(current_prices)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("총 평가금액 (세후)", f"{summary['total_value']:,.0f}원")
                with col2:
                    profit_color = "normal" if summary['total_profit'] >= 0 else "inverse"
                    st.metric("총 평가손익", f"{summary['total_profit']:,.0f}원",
                             delta=f"{summary['total_profit_rate']:.2f}%")
                with col3:
                    st.metric("총 매입금액", f"{summary['total_purchase']:,.0f}원")
                with col4:
                    st.metric("예상 세금", f"{summary.get('total_tax', 0):,.0f}원",
                             help="매도 시 부과되는 증권거래세 0.23%")
                with col5:
                    st.metric("보유 종목 수", f"{summary['num_stocks']}개")
                
                st.markdown("---")
                
                # 매도 타이밍 알림
                st.subheader("🔔 매도 타이밍 알림")
                sell_signals = portfolio_mgr.get_all_sell_signals(current_prices, price_histories, stop_loss_rate)
                
                if sell_signals:
                    for signal in sell_signals:
                        if signal['recommendation'] == "즉시 손절":
                            alert_type = "error"
                        elif signal['recommendation'] == "수익 실현":
                            alert_type = "success"
                        else:
                            alert_type = "warning"
                        
                        with st.container():
                            if alert_type == "error":
                                st.error(f"**{signal['name']} ({signal['code']})**")
                            elif alert_type == "success":
                                st.success(f"**{signal['name']} ({signal['code']})**")
                            else:
                                st.warning(f"**{signal['name']} ({signal['code']})**")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("수익률", f"{signal['profit_rate']:.2f}%")
                            with col2:
                                st.metric("수익금액", f"{signal['profit_amount']:,.0f}원")
                            with col3:
                                st.metric("보유일", f"{signal['hold_days']}일")
                            with col4:
                                st.metric("권장", signal['recommendation'])
                            
                            for sig in signal['signals']:
                                st.write(sig)
                else:
                    st.info("현재 매도 신호가 있는 종목이 없습니다. 👍")
                
                st.markdown("---")
                
                # 종목별 상세
                st.subheader("📊 종목별 상세 현황")
                
                for _, row in portfolio_df.iterrows():
                    with st.expander(f"{row['종목명']} ({row['종목코드']}) - 수익률: {row['수익률']:.2f}%"):
                        code = row['종목코드']
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("보유수량", f"{row['보유수량']:,}주")
                        with col2:
                            st.metric("평균단가", f"{row['평균단가']:,.0f}원")
                        with col3:
                            st.metric("현재가", f"{row['현재가']:,.0f}원")
                        with col4:
                            st.metric("평가금액", f"{row['평가금액']:,.0f}원")
                        with col5:
                            st.metric("평가손익", f"{row['평가손익']:,.0f}원",
                                    delta=f"{row['수익률']:.2f}%")
                        
                        st.markdown("---")
                        
                        # 수정 및 삭제 버튼 (항상 표시)
                        st.markdown("**종목 관리**")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            if st.button("✏️ 수정", key=f"edit_{code}", use_container_width=True):
                                st.session_state[f'editing_{code}'] = True
                                st.rerun()
                        
                        with col_b:
                            if st.button("🗑️ 삭제", key=f"delete_{code}", type="secondary", use_container_width=True):
                                result = portfolio_mgr.remove_from_portfolio(code)
                                if result:
                                    st.success(f"{row['종목명']} 삭제 완료!")
                                    st.rerun()
                                else:
                                    st.error(f"삭제 실패: {code}를 찾을 수 없습니다.")
                        
                        # 수정 모드
                        if st.session_state.get(f'editing_{code}', False):
                            st.markdown("---")
                            st.markdown("**📝 종목 정보 수정**")
                            
                            edit_col1, edit_col2 = st.columns(2)
                            with edit_col1:
                                new_quantity = st.number_input(
                                    "보유 수량 (주)", 
                                    min_value=1, 
                                    value=int(row['보유수량']),
                                    key=f"edit_qty_{code}"
                                )
                            with edit_col2:
                                new_avg_price = st.number_input(
                                    "평균 매입단가 (원)", 
                                    min_value=0, 
                                    value=int(row['평균단가']),
                                    step=1000,
                                    key=f"edit_price_{code}"
                                )
                            
                            save_col1, save_col2 = st.columns(2)
                            with save_col1:
                                if st.button("💾 저장", key=f"save_{code}", type="primary", use_container_width=True):
                                    # 현재 포트폴리오 정보 다시 로드
                                    current_portfolio = portfolio_mgr.load_portfolio()
                                    purchase_date = current_portfolio[code].get('purchase_date', datetime.now().strftime("%Y-%m-%d"))
                                    
                                    # 기존 종목 제거
                                    portfolio_mgr.remove_from_portfolio(code)
                                    # 새로운 정보로 추가
                                    portfolio_mgr.add_to_portfolio(
                                        code, 
                                        row['종목명'], 
                                        new_quantity, 
                                        new_avg_price,
                                        purchase_date
                                    )
                                    st.session_state[f'editing_{code}'] = False
                                    st.success("수정 완료!")
                                    st.rerun()
                            
                            with save_col2:
                                if st.button("❌ 취소", key=f"cancel_{code}", use_container_width=True):
                                    st.session_state[f'editing_{code}'] = False
                                    st.rerun()
                        
                        # 매도 타이밍 분석 (데이터가 있는 경우만)
                        if code in current_prices and code in price_histories:
                            st.markdown("---")
                            st.markdown("**매도 타이밍 분석**")
                            
                            analysis = portfolio_mgr.analyze_sell_timing(
                                code, current_prices[code], price_histories[code], stop_loss_rate
                            )
                            
                            st.write(f"보유 기간: {analysis['hold_days']}일")
                            st.write(f"권장 행동: **{analysis['recommendation']}**")
                            
                            for sig in analysis['signals']:
                                st.write(f"- {sig}")
                
                st.markdown("---")
                
                # 요약 테이블
                st.subheader("📋 요약 테이블")
                
                # 수익률에 따른 색상 적용 함수
                def color_negative_red(val):
                    if isinstance(val, (int, float)):
                        color = 'red' if val < 0 else 'green' if val > 0 else 'gray'
                        return f'color: {color}'
                    return ''
                
                # 세금 관련 컬럼 포함 여부 확인
                format_dict = {
                    '평균단가': '{:,.0f}',
                    '현재가': '{:,.0f}',
                    '매입금액': '{:,.0f}',
                    '평가금액': '{:,.0f}',
                    '평가손익': '{:,.0f}',
                    '수익률': '{:.2f}'
                }
                
                if '예상세금' in portfolio_df.columns:
                    format_dict['예상세금'] = '{:,.0f}'
                if '평가금액(세전)' in portfolio_df.columns:
                    format_dict['평가금액(세전)'] = '{:,.0f}'
                
                st.dataframe(
                    portfolio_df.style.format(format_dict).applymap(color_negative_red, subset=['수익률']),
                    use_container_width=True
                )
                
                # 원형 차트
                fig = go.Figure(data=[go.Pie(
                    labels=portfolio_df['종목명'],
                    values=portfolio_df['평가금액'],
                    hole=0.4,
                    textinfo='label+percent',
                    textposition='auto'
                )])
                fig.update_layout(title="포트폴리오 비중", height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("보유 종목 추가")
        
        # 현재 포트폴리오 로드 (중복 체크용)
        current_portfolio = portfolio_mgr.load_portfolio()
        
        # 메인 대시보드에서 종목 추가 요청이 있는 경우
        add_stock = st.session_state.get('portfolio_add_stock', None)
        if add_stock:
            st.info(f"💡 선택한 종목: **{add_stock['name']}** ({add_stock['code']}) - 아래에서 수량과 매입단가를 입력하세요")
        
        # KOSPI 200 종목 데이터 로드
        kospi_dict = load_kospi_stocks()  # {종목코드: 종목명}
        
        st.markdown("#### 📝 종목 선택 및 보유 정보 입력")
        
        # 초기화 - 선택된 종목 정보를 session_state에 저장
        if 'selected_stock_info' not in st.session_state:
            st.session_state.selected_stock_info = {
                'code': None,
                'name': None,
                'price': 0
            }
        
        selected_code = st.session_state.selected_stock_info['code']
        selected_name = st.session_state.selected_stock_info['name']
        selected_price = st.session_state.selected_stock_info['price']
        
        if kospi_dict:
            st.markdown("**종목 선택**")
            
            # 메인 대시보드에서 전달받은 종목이 있으면 자동 선택할 종목명 찾기
            auto_select_code = None
            if add_stock and add_stock.get('code'):
                auto_select_code = add_stock['code']
            
            # 종목명 목록 생성 (종목코드: 종목명)
            code_to_name = kospi_dict  # {code: name}
            name_to_code = {v: k for k, v in kospi_dict.items()}  # {name: code}
            
            # 드롭다운 옵션: 종목명 (종목코드) 형식
            stock_display_options = [""] + [f"{name} ({code})" for code, name in sorted(kospi_dict.items(), key=lambda x: x[1])]
            
            # 자동 선택 인덱스 찾기
            default_index = 0
            if auto_select_code and auto_select_code in kospi_dict:
                display_name = f"{kospi_dict[auto_select_code]} ({auto_select_code})"
                if display_name in stock_display_options:
                    default_index = stock_display_options.index(display_name)
            
            selected_display = st.selectbox(
                "KOSPI 200 종목 선택",
                options=stock_display_options,
                index=default_index,
                key="stock_name_selectbox",
                help="종목을 선택하면 자동으로 정보가 입력됩니다"
            )
            
            # 선택된 종목 처리
            if selected_display and selected_display != "":
                # 종목코드와 종목명 추출
                parts = selected_display.split(" (")
                temp_name = parts[0]
                temp_code = parts[1].rstrip(")")
                
                # session_state에 저장
                st.session_state.selected_stock_info = {
                    'code': temp_code,
                    'name': temp_name,
                    'price': 0  # 현재가는 추후 조회
                }
                
                # 메인 대시보드에서 전달받은 정보 제거
                if 'portfolio_add_stock' in st.session_state:
                    del st.session_state.portfolio_add_stock
                
                selected_code = temp_code
                selected_name = temp_name
                
                # 중복 체크
                is_duplicate = (selected_code in current_portfolio or 
                               selected_name in current_portfolio)
                
                if is_duplicate:
                    st.error(f"⚠️ **{selected_name} ({selected_code})**은(는) 이미 포트폴리오에 있습니다!")
                    st.info("💡 기존 종목의 수량을 변경하려면 '보유 종목' 탭에서 수정 버튼을 사용하세요.")
                else:
                    # 선택된 종목 정보 표시
                    st.success(f"✅ **{selected_name} ({selected_code})** 선택됨")
                    st.info("👇 아래에서 보유 수량과 매입단가를 입력하세요")
            
            st.markdown("---")
        else:
            st.error("⚠️ KOSPI 200 종목 데이터를 불러올 수 없습니다.")
        
        # 보유 정보 입력 폼
        st.markdown("**보유 정보 입력**")
        
        # 선택된 종목 정보 표시
        if selected_code and selected_name:
            st.info(f"📌 선택된 종목: **{selected_name} ({selected_code})**")
            code = selected_code
            name = selected_name
        else:
            st.warning("⚠️ 위에서 먼저 종목을 선택해주세요!")
            code = None
            name = None
        
        col1, col2 = st.columns(2)
        
        with col1:
            quantity = st.number_input("보유 수량 (주)", min_value=1, value=1, step=1, key="quantity_input")
        
        with col2:
            avg_price = st.number_input("평균 매입단가 (원)", 
                                      min_value=0, 
                                      value=0, 
                                      step=1000, 
                                      key="avg_price_input",
                                      help="예: 70000")
        
        st.markdown("")
        
        if st.button("➕ 포트폴리오에 추가", type="primary", use_container_width=True):
            if code and name and avg_price > 0:
                # 중복 체크
                if code in current_portfolio or name in current_portfolio:
                    st.error(f"⚠️ **{name} ({code})**은(는) 이미 포트폴리오에 있습니다!")
                    st.info("💡 기존 종목의 수량을 변경하려면 '보유 종목' 탭에서 수정 버튼을 사용하세요.")
                else:
                    # 매수일은 자동으로 오늘 날짜로 설정
                    purchase_date = datetime.now().strftime("%Y-%m-%d")
                    
                    portfolio_mgr.add_to_portfolio(
                        code, name, quantity, avg_price, 
                        purchase_date
                    )
                    st.success(f"✅ {name} ({code}) {quantity}주를 포트폴리오에 추가했습니다!")
                    
                    # 메인 대시보드에서 전달받은 종목 정보 제거
                    if 'portfolio_add_stock' in st.session_state:
                        del st.session_state.portfolio_add_stock
                    
                    # 선택 정보 초기화
                    st.session_state.selected_stock_info = {
                        'code': None,
                        'name': None,
                        'price': 0
                    }
                    
                    st.balloons()
                    import time
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("⚠️ 모든 항목을 올바르게 입력해주세요.")
        
        # 도움말
        with st.expander("💡 사용 팁"):
            st.markdown("""
            **종목 선택:**
            - KOSPI 200 종목 목록에서 선택하세요
            - 종목명순으로 정렬되어 있어 찾기 쉽습니다
            - 선택 시 종목코드와 현재가가 자동으로 입력됩니다
            
            **중복 방지:**
            - 이미 보유 중인 종목은 추가할 수 없습니다
            - 기존 종목의 수량을 변경하려면 '보유 종목' 탭에서 수정 버튼을 사용하세요
            
            **필수 입력 항목:**
            - 종목: 콤보박스에서 선택
            - 보유 수량: 보유 중인 주식 수
            - 평균 매입단가: 실제 매수한 평균 가격 (현재가가 자동 입력됨)
            - 매수일은 자동으로 오늘 날짜로 설정됩니다
            
            **평균 매입단가 계산 예시:**
            - 여러 번 매수한 경우 가중평균을 계산하세요
            - 예: 10주@70,000원 + 5주@80,000원 = 평균 73,333원
            - 계산: (10×70,000 + 5×80,000) ÷ 15 = 73,333원
            """)


def page_backtesting():
    """백테스팅 페이지"""
    st.title("📈 백테스팅 & 정확도 검증")
    
    backtest_engine = BacktestingEngine()
    
    tab1, tab2, tab3 = st.tabs(["🎯 전략 백테스팅", "📊 AI 예측 정확도", "📈 성과 비교"])
    
    with tab1:
        st.subheader("기술적 지표 전략 백테스팅")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            code = st.text_input("종목코드", value="005930")
            period = st.selectbox("기간", ["1y", "2y", "3y", "5y"])
            strategy = st.selectbox("전략", ["골든크로스", "RSI(30/70)", "RSI(20/80)"])
        
        if st.button("백테스트 실행"):
            with st.spinner("백테스팅 중..."):
                try:
                    df = load_stock_data(code, period=period)
                    
                    if strategy == "골든크로스":
                        result = backtest_engine.backtest_technical_strategy(df, 'golden_cross')
                    elif strategy == "RSI(30/70)":
                        result = backtest_engine.backtest_rsi_strategy(df, 30, 70)
                    else:
                        result = backtest_engine.backtest_rsi_strategy(df, 20, 80)
                    
                    # 결과 표시
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 수익률", f"{result['total_return']:.2f}%")
                    with col2:
                        st.metric("승률", f"{result['win_rate']:.1f}%")
                    with col3:
                        st.metric("거래 횟수", result['num_trades'])
                    with col4:
                        st.metric("평균 수익", f"{result.get('avg_profit', 0):.2f}%")
                    
                    # 거래 상세
                    if result.get('signals'):
                        st.markdown("#### 거래 내역")
                        signals_df = pd.DataFrame(result['signals'])
                        st.dataframe(signals_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"백테스팅 오류: {e}")
    
    with tab2:
        st.subheader("AI 예측 정확도 추적")
        
        code = st.text_input("종목코드", value="005930", key="ai_accuracy_code")
        
        if st.button("정확도 확인"):
            try:
                df = load_stock_data(code, period="1mo")
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    result = backtest_engine.verify_predictions(code, current_price)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("전체 예측", f"{result['total']}회")
                    with col2:
                        st.metric("검증 완료", f"{result['verified']}회")
                    with col3:
                        st.metric("정확한 예측", f"{result['correct']}회")
                    with col4:
                        st.metric("정확도", f"{result['accuracy']:.1f}%")
                    
                    # 추이 그래프
                    trend_df = backtest_engine.get_prediction_accuracy_trend(code)
                    if not trend_df.empty:
                        st.markdown("#### 예측 이력")
                        st.dataframe(trend_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"오류: {e}")
    
    with tab3:
        st.subheader("전략 성과 비교")
        
        code = st.text_input("종목코드", value="005930", key="strategy_compare_code")
        
        if st.button("비교 분석"):
            with st.spinner("분석 중..."):
                try:
                    df = load_stock_data(code, period="2y")
                    comparison_df = backtest_engine.compare_strategies(df)
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"오류: {e}")


def page_alerts():
    """알림 관리 페이지"""
    st.title("🔔 알림 관리")
    
    alert_mgr = AlertManager()
    
    tab1, tab2, tab3 = st.tabs(["➕ 알림 추가", "🔔 활성 알림", "✅ 발동 이력"])
    
    with tab1:
        st.subheader("새 알림 설정")
        
        alert_type = st.radio("알림 유형", ["가격 알림", "등락률 알림", "뉴스 키워드 알림"])
        
        code = st.text_input("종목코드")
        name = st.text_input("종목명")
        
        if alert_type == "가격 알림":
            col1, col2 = st.columns(2)
            with col1:
                direction = st.selectbox("방향", ["이상", "이하"])
            with col2:
                target_price = st.number_input("목표가", min_value=0, value=100000)
            
            current_price = st.number_input("현재가", min_value=0, value=100000)
            
            if st.button("알림 추가"):
                alert_type_code = 'above' if direction == "이상" else 'below'
                alert_mgr.add_price_alert(code, name, alert_type_code, target_price, current_price)
                st.success("알림이 추가되었습니다!")
        
        elif alert_type == "등락률 알림":
            col1, col2 = st.columns(2)
            with col1:
                change_type = st.selectbox("유형", ["급등", "급락"])
            with col2:
                threshold = st.number_input("기준 (%)", min_value=0.0, value=5.0)
            
            if st.button("알림 추가"):
                change_code = 'surge' if change_type == "급등" else 'plunge'
                alert_mgr.add_change_alert(code, name, change_code, threshold)
                st.success("알림이 추가되었습니다!")
        
        else:  # 뉴스 키워드
            keywords_text = st.text_input("키워드 (쉼표로 구분)", "실적,배당,인수")
            keywords = [k.strip() for k in keywords_text.split(',')]
            
            if st.button("알림 추가"):
                alert_mgr.add_news_alert(code, name, keywords)
                st.success("알림이 추가되었습니다!")
    
    with tab2:
        st.subheader("활성 알림 목록")
        
        active_alerts = alert_mgr.get_active_alerts()
        
        if not active_alerts:
            st.info("활성 알림이 없습니다.")
        else:
            for alert in active_alerts:
                with st.expander(f"{alert['name']} ({alert['code']}) - {alert['type']}"):
                    st.json(alert)
    
    with tab3:
        st.info("탭 콘텐츠")


def page_comparison():
    """종목 비교 페이지"""
    st.title("🔍 종목 비교 분석")
    
    st.subheader("비교할 종목 선택")
    
    codes_text = st.text_input("종목코드 (쉼표로 구분)", "005930,000660,035420")
    codes = [c.strip() for c in codes_text.split(',')]
    
    if st.button("비교 분석"):
        with st.spinner("데이터 수집 중..."):
            stock_data = {}
            stock_prices = {}
            
            for code in codes:
                try:
                    df = load_stock_data(code, period="1y")
                    if not df.empty:
                        stock_prices[code] = df
                        
                        current_price = df['Close'].iloc[-1]
                        change = ((current_price / df['Close'].iloc[-5]) - 1) * 100
                        
                        stock_data[code] = {
                            'name': code,  # 실제로는 종목명 조회 필요
                            'price': current_price,
                            'change': change,
                            'volume': df['Volume'].iloc[-1],
                            'ai_score': 75  # 실제로는 AI 점수 계산 필요
                        }
                except:
                    pass
            
            if stock_data:
                # 비교 테이블
                analyzer = ComparisonAnalyzer()
                comparison_df = analyzer.compare_stocks(stock_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # 비교 차트
                st.markdown("### 가격 추이 비교")
                fig = analyzer.create_comparison_chart(stock_prices, normalize=True)
                st.plotly_chart(fig, use_container_width=True)


def page_sector():
    """섹터 분석 페이지"""
    st.title("🌐 섹터 분석")
    
    if st.button("섹터 분석 실행"):
        with st.spinner("KOSPI 200 데이터 수집 중..."):
            try:
                crawler = CrawlingKospi()
                kospi_df = crawler.get_all_kospi_data()
                
                if not kospi_df.empty:
                    analyzer = SectorAnalyzer()
                    
                    # 섹터별 수익률
                    sector_perf = analyzer.get_sector_performance(kospi_df)
                    
                    st.markdown("### 섹터별 등락률")
                    st.dataframe(sector_perf, use_container_width=True)
                    
                    # 히트맵
                    fig = analyzer.create_sector_heatmap(sector_perf)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 섹터별 상위 종목
                    st.markdown("### 섹터별 상위 종목")
                    top_stocks = analyzer.get_top_stocks_by_sector(kospi_df, top_n=3)
                    
                    for sector, df in top_stocks.items():
                        with st.expander(f"📊 {sector}"):
                            st.dataframe(df, use_container_width=True)
            
            except Exception as e:
                st.error(f"오류: {e}")


def page_advanced_charts():
    """고급 차트 페이지"""
    st.title("📊 고급 차트 & 패턴 인식")
    
    code = st.text_input("종목코드", value="005930")
    period = st.selectbox("기간", ["1mo", "3mo", "6mo", "1y"])
    
    if st.button("차트 생성"):
        with st.spinner("차트 생성 중..."):
            try:
                df = load_stock_data(code, period=period)
                
                if not df.empty:
                    # 거래량 분석
                    st.markdown("### 📊 가격-거래량 차트")
                    volume_fig = VolumeAnalyzer.create_price_volume_chart(df)
                    st.plotly_chart(volume_fig, use_container_width=True)
                    
                    # 거래량 추세
                    volume_trend = VolumeAnalyzer.analyze_volume_trend(df)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("최근 평균 거래량", f"{volume_trend.get('recent_avg', 0):,.0f}")
                    with col2:
                        st.metric("전체 평균 거래량", f"{volume_trend.get('overall_avg', 0):,.0f}")
                    with col3:
                        st.metric("거래량 추세", volume_trend.get('trend', '보통'))
                    
                    # 캔들 패턴 인식
                    st.markdown("### 🕯️ 캔들 패턴 분석")
                    pattern_recognizer = CandlePatternRecognizer()
                    
                    doji = pattern_recognizer.detect_doji(df)
                    hammer = pattern_recognizer.detect_hammer(df)
                    star = pattern_recognizer.detect_shooting_star(df)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("도지 패턴", len(doji))
                    with col2:
                        st.metric("망치형 패턴", len(hammer))
                    with col3:
                        st.metric("유성형 패턴", len(star))
            
            except Exception as e:
                st.error(f"오류: {e}")


def page_export():
    """데이터 내보내기 페이지"""
    st.title("📥 데이터 내보내기")
    
    tab1, tab2 = st.tabs(["📊 데이터 내보내기", "📄 리포트 생성"])
    
    with tab1:
        st.subheader("KOSPI 200 데이터 내보내기")
        
        if st.button("데이터 수집 및 내보내기"):
            with st.spinner("데이터 수집 중..."):
                try:
                    crawler = CrawlingKospi()
                    kospi_df = crawler.get_all_kospi_data()
                    
                    if not kospi_df.empty:
                        exporter = DataExporter()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV 다운로드
                            csv_data = exporter.create_download_link(kospi_df, format='csv')
                            st.download_button(
                                label="📥 CSV 다운로드",
                                data=csv_data,
                                file_name=f"kospi200_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Excel 다운로드
                            excel_data = exporter.create_download_link(kospi_df, format='excel')
                            st.download_button(
                                label="📥 Excel 다운로드",
                                data=excel_data,
                                file_name=f"kospi200_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        st.success("데이터 내보내기 준비 완료!")
                        st.dataframe(kospi_df.head(10), use_container_width=True)
                
                except Exception as e:
                    st.error(f"오류: {e}")
    
    with tab2:
        st.subheader("종목 분석 리포트 생성")
        
        code = st.text_input("종목코드", value="005930")
        name = st.text_input("종목명", value="삼성전자")
        
        if st.button("리포트 생성"):
            # 샘플 데이터
            analysis_data = {
                'current_price': 70000,
                'change_rate': 2.5,
                'volume': 15000000,
                'market_cap': '400조원',
                'ai_score': 85,
                'recommendation': '매수',
                'reasoning': 'AI 분석 결과 긍정적인 시그널이 감지되었습니다.',
                'rsi': 55,
                'macd': '양호',
                'ma5': 69500,
                'ma20': 68000,
                'ma60': 67000,
                'per': 15.2,
                'pbr': 1.8,
                'roe': 12.5,
                'operating_margin': 15.3,
                'recent_news': '최근 실적 발표 예정'
            }
            
            report = ReportGenerator.generate_stock_report(code, name, analysis_data)
            
            st.markdown(report)
            
            # 다운로드
            st.download_button(
                label="📥 Markdown 다운로드",
                data=report,
                file_name=f"{name}_{code}_report.md",
                mime="text/markdown"
            )


def page_settings():
    """설정 페이지"""
    st.title("⚙️ 설정")
    
    settings_mgr = UserSettings()
    current_settings = settings_mgr.load_settings()
    
    tab1, tab2, tab3 = st.tabs(["🎨 화면 설정", "📊 기본값 설정", "🔧 고급 설정"])
    
    with tab1:
        st.subheader("화면 설정")
        
        theme = st.selectbox("테마", ["light", "dark", "blue"], 
                            index=["light", "dark", "blue"].index(current_settings.get('theme', 'light')))
        
        chart_height = st.slider("차트 높이", 400, 800, current_settings.get('chart_height', 600), 50)
        
        show_volume = st.checkbox("거래량 표시", value=current_settings.get('show_volume', True))
        
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
        
        auto_expand = st.checkbox("1위 종목 자동 펼치기", 
                                 value=current_settings.get('auto_expand_top', True))
        
        if st.button("저장", key="save_defaults"):
            settings_mgr.set('default_period', default_period)
            settings_mgr.set('auto_expand_top', auto_expand)
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


# 메인 실행 함수
def run_phase4_app():
    """Phase 4 앱 실행"""
    
    # 사이드바 메뉴
    page = render_sidebar_menu()
    
    # 메인 대시보드에서 포트폴리오 페이지로 이동 요청이 있는 경우
    if 'selected_menu' in st.session_state:
        # 메뉴명을 페이지 키로 변환
        menu_to_page = {
            "🏠 메인 대시보드": "main",
            "💼 포트폴리오": "portfolio",
            "📈 백테스팅": "backtesting",
            "🔔 알림 관리": "alerts",
            "🌐 섹터 분석": "sector",
            "⚙️ 설정": "settings"
        }
        selected_menu = st.session_state.selected_menu
        if selected_menu in menu_to_page:
            page = menu_to_page[selected_menu]
            # 세션 상태 제거
            del st.session_state.selected_menu
    
    # 페이지 라우팅
    if page == "portfolio":
        page_portfolio()
    elif page == "backtesting":
        page_backtesting()
    elif page == "sector":
        page_sector()
    elif page == "settings":
        page_settings()
    else:
        # 기존 메인 대시보드
        from app.ui import run_app
        run_app()
