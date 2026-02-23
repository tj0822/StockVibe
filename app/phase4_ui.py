"""
Phase 4 통합 페이지 - 모든 신규 기능을 담은 메뉴 시스템
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Phase 4 모듈 import
from app.portfolio import PortfolioManager
from app.comparison import ComparisonAnalyzer, SectorAnalyzer
from app.export import DataExporter, ReportGenerator, ChartExporter
from app.advanced_charts import CandlePatternRecognizer, CorrelationAnalyzer, VolumeAnalyzer
from app.settings import UserSettings, ThemeManager, DisplaySettings

# 기존 모듈
from crawling_kospi import CrawlingKospi
from app.data import load_stock_data
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
def load_kospi_data():
    """저장된 KOSPI 200 주가 데이터 로드 (캐싱)"""
    import os
    try:
        crawler = CrawlingKospi()
        kospi_df = crawler.get_all_kospi_data()
        return kospi_df
    except Exception as e:
        st.warning(f"KOSPI 데이터 로드 오류: {e}")
        return pd.DataFrame()


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


@st.cache_data(ttl=3600)  # 1시간 캐싱
def prepare_sector_analysis_data(kospi_df: pd.DataFrame, name_map: dict) -> pd.DataFrame:
    """섹터 분석용 데이터 전처리 (캐싱)"""
    if kospi_df.empty:
        return pd.DataFrame()
    
    processed_df = kospi_df.copy()
    
    # 필요한 컬럼 추가/변환
    if 'code' in processed_df.columns:
        processed_df.rename(columns={'code': '종목코드'}, inplace=True)
    
    # 최신 날짜의 데이터만 사용
    if 'date' in processed_df.columns:
        latest_date = pd.to_datetime(processed_df['date']).max()
        processed_df = processed_df[pd.to_datetime(processed_df['date']) == latest_date]
    
    # 수익률 계산
    if '등락률' not in processed_df.columns:
        processed_df['등락률'] = ((processed_df['close'] - processed_df['open']) / processed_df['open'] * 100).fillna(0)
    
    # 종목명 추가
    processed_df['종목명'] = processed_df['종목코드'].apply(lambda x: name_map.get(x, x))
    processed_df['현재가'] = processed_df['close']
    processed_df['AI점수'] = 50  # 기본값
    
    return processed_df


# =================================


def render_sidebar_menu() -> tuple[str, str]:
    """사이드바 메뉴"""
    st.sidebar.title("📊 StockVibe Pro")
    st.sidebar.markdown("---")
    
    # 페이지 선택
    pages = {
        "🏠 메인 대시보드": "main",
        "📈 종목분석": "analysis",
        "💼 포트폴리오": "portfolio",
        "⚙️ 설정": "settings"
    }
    
    selected = st.sidebar.radio("메뉴", list(pages.keys()))
    selected_page = pages[selected]

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


def page_portfolio():
    """포트폴리오 관리 페이지 - 개선 버전"""
    st.title("💼 포트폴리오 관리")
    
    portfolio_mgr = PortfolioManager()
    
    # 세션 상태 초기화
    if 'selected_portfolio_code' not in st.session_state:
        st.session_state.selected_portfolio_code = None
    
    # KOSPI 200 데이터 로드 함수 (캐싱)
    @st.cache_data(ttl=3600)
    def load_kospi_stocks():
        try:
            crawler = CrawlingKospi()
            kospi_dict = crawler.GetKospi200()
            return kospi_dict
        except:
            return {}
    
    # 주가 데이터 로드 (캐싱)
    @st.cache_data(ttl=1800)
    def load_stock_prices():
        try:
            from app.data import load_stock_data
            return load_stock_data("data")
        except:
            return pd.DataFrame()
    
    # 재무 데이터 로드 (캐싱)
    @st.cache_data(ttl=1800)
    def load_finance_info():
        try:
            from app.data import load_finance_data
            return load_finance_data("data")
        except:
            return pd.DataFrame()
    
    # 뉴스 로드 (캐싱)
    @st.cache_data(ttl=1800)
    def load_portfolio_news(stock_code):
        try:
            from naver_news_crawler import NaverNewsCrawler
            crawler = NaverNewsCrawler()
            return crawler.get_recent_news(stock_code, max_news=5)
        except Exception as e:
            return pd.DataFrame()
    
    # 데이터 로드
    kospi_dict = load_kospi_stocks()
    price_df = load_stock_prices()
    finance_df = load_finance_info()
    portfolio = portfolio_mgr.load_portfolio()
    
    if not portfolio:
        st.info("보유 종목이 없습니다. 아래에서 종목을 추가해주세요.")
        # 종목 추가 섹션
        st.subheader("➕ 종목 추가")
        if kospi_dict:
            available_stocks = {code: name for code, name in kospi_dict.items()}
            stock_options = ["선택하세요..."] + [f"{name} ({code})" 
                            for code, name in sorted(available_stocks.items(), key=lambda x: x[1])]
            
            selected = st.selectbox("종목 선택", options=stock_options, label_visibility="collapsed")
            
            if selected and selected != "선택하세요...":
                parts = selected.split(" (")
                stock_name = parts[0]
                stock_code = parts[1].rstrip(")")
                
                if st.button("추가", type="primary", use_container_width=True):
                    portfolio_mgr.add_to_portfolio(stock_code, stock_name, 1, 0)
                    st.success(f"✅ {stock_name} 추가됨")
                    st.rerun()
    else:
        # 3열 레이아웃
        col_left, col_middle, col_right = st.columns([1.2, 1.5, 1.3])
        
        # ===== 왼쪽: 보유종목 목록 =====
        with col_left:
            st.subheader("📊 보유 종목")
            
            # 영문/한글 구분 함수
            def is_english_stock(name):
                if not name:
                    return False
                return ord(name[0]) < 128
            
            # 종목명으로 정렬
            sorted_portfolio = sorted(
                portfolio.items(),
                key=lambda x: (not is_english_stock(x[1]['name']), x[1]['name'])
            )
            
            # 종목 선택 버튼들
            for code, info in sorted_portfolio:
                is_selected = code == st.session_state.selected_portfolio_code
                btn_color = "🟢" if is_selected else "⚪"
                
                col_select, col_del = st.columns([4, 1])
                with col_select:
                    if st.button(f"{btn_color} {info['name']}\n({code})", 
                                key=f"select_{code}",
                                use_container_width=True):
                        st.session_state.selected_portfolio_code = code
                        st.rerun()
                
                with col_del:
                    if st.button("🗑️", key=f"del_{code}", help="삭제"):
                        portfolio_mgr.remove_from_portfolio(code)
                        st.session_state.selected_portfolio_code = None
                        st.rerun()
            
            st.markdown("---")
            st.caption(f"총 {len(portfolio)}개 종목")
            
            # 종목 추가
            st.subheader("➕ 추가")
            if kospi_dict:
                current_codes = set(portfolio_mgr.load_portfolio().keys())
                available_stocks = {code: name for code, name in kospi_dict.items() 
                                   if code not in current_codes}
                
                if not available_stocks:
                    st.success("모든 KOSPI 200 보유 중!")
                else:
                    stock_options = ["선택..."] + [f"{name} ({code})" 
                                    for code, name in sorted(available_stocks.items(), key=lambda x: x[1])]
                    
                    selected = st.selectbox("종목", options=stock_options, 
                                           label_visibility="collapsed", key="add_stock")
                    
                    if selected and selected != "선택...":
                        parts = selected.split(" (")
                        stock_name = parts[0]
                        stock_code = parts[1].rstrip(")")
                        
                        if st.button("✅ 추가", use_container_width=True, type="primary"):
                            portfolio_mgr.add_to_portfolio(stock_code, stock_name, 1, 0)
                            st.success(f"{stock_name} 추가됨")
                            st.rerun()
        
        # ===== 중간: 주가흐름 차트 =====
        with col_middle:
            if st.session_state.selected_portfolio_code:
                selected_code = st.session_state.selected_portfolio_code
                selected_info = portfolio.get(selected_code, {})
                
                st.subheader(f"📈 {selected_info.get('name', selected_code)}")
                
                # 선택된 종목의 최근 30일 주가 데이터
                if not price_df.empty:
                    stock_data = price_df[price_df['code'] == selected_code].copy()
                    stock_data = stock_data.tail(30).sort_values('date')
                    
                    if not stock_data.empty:
                        # 간단한 라인 차트 (Plotly)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=stock_data['date'],
                            y=stock_data['close'],
                            mode='lines+markers',
                            name='종가',
                            line=dict(color='#1f77b4', width=2),
                            marker=dict(size=4)
                        ))
                        fig.update_layout(
                            title=f"최근 30일 주가추이",
                            xaxis_title="날짜",
                            yaxis_title="가격 (₩)",
                            height=300,
                            hovermode='x unified',
                            margin=dict(l=50, r=20, t=40, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 주가 통계
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("현재가", f"₩{stock_data['close'].iloc[-1]:,.0f}")
                            st.metric("30일 고가", f"₩{stock_data['high'].max():,.0f}" if 'high' in stock_data.columns else "N/A")
                        
                        with col_stat2:
                            change_pct = ((stock_data['close'].iloc[-1] - stock_data['close'].iloc[0]) / stock_data['close'].iloc[0] * 100) if len(stock_data) > 1 else 0
                            st.metric("30일 등락률", f"{change_pct:+.2f}%")
                            st.metric("30일 저가", f"₩{stock_data['low'].min():,.0f}" if 'low' in stock_data.columns else "N/A")
                    else:
                        st.warning("주가 데이터가 없습니다.")
                else:
                    st.warning("주가 데이터를 불러올 수 없습니다.")
            else:
                st.info("왼쪽에서 종목을 선택하면 차트가 표시됩니다.")
        
        # ===== 오른쪽: 상세정보 & 뉴스 =====
        with col_right:
            if st.session_state.selected_portfolio_code:
                selected_code = st.session_state.selected_portfolio_code
                selected_info = portfolio.get(selected_code, {})
                
                st.subheader("📊 상세 정보")
                
                # 재무정보 표시
                if not finance_df.empty:
                    finance_data = finance_df[finance_df['code'] == selected_code]
                    
                    if not finance_data.empty:
                        latest_finance = finance_data.sort_values('date').iloc[-1]
                        
                        st.markdown("**주요 지표**")
                        col_f1, col_f2 = st.columns(2)
                        
                        with col_f1:
                            if pd.notna(latest_finance.get('per')):
                                st.metric("PER", f"{latest_finance['per']:.2f}")
                            if pd.notna(latest_finance.get('eps')):
                                st.metric("EPS", f"₩{latest_finance['eps']:,.0f}")
                        
                        with col_f2:
                            if pd.notna(latest_finance.get('pbr')):
                                st.metric("PBR", f"{latest_finance['pbr']:.2f}")
                            if pd.notna(latest_finance.get('bps')):
                                st.metric("BPS", f"₩{latest_finance['bps']:,.0f}")
                        
                        # 추가 정보
                        col_f3, col_f4 = st.columns(2)
                        with col_f3:
                            if pd.notna(latest_finance.get('dvr')):
                                st.metric("배당률", f"{latest_finance['dvr']:.2f}%")
                        
                        with col_f4:
                            if 'foreigner_ratio' in latest_finance and pd.notna(latest_finance['foreigner_ratio']):
                                st.metric("외국인 보유율", f"{latest_finance['foreigner_ratio']:.1f}%")
                        
                        st.markdown("---")
                
                # 뉴스 섹션
                st.markdown("**📰 최근 뉴스**")
                
                news_df = load_portfolio_news(selected_code)
                
                if not news_df.empty:
                    from naver_news_crawler import NaverNewsCrawler
                    crawler = NaverNewsCrawler()
                    
                    for idx, news in news_df.head(5).iterrows():
                        # NaverNewsCrawler의 컬럼명: '제목', '출처', '날짜', '링크'
                        news_title = news.get('제목', '제목 없음')
                        news_date = news.get('날짜', '')
                        news_source = news.get('출처', '')
                        news_link = news.get('링크', '')
                        
                        # Expander로 뉴스 제목을 클릭하면 본문 표시
                        with st.expander(f"📰 {news_title[:50]}... ({news_source})"):
                            st.caption(f"📅 {news_date}")
                            
                            if news_link and news_link.startswith('http'):
                                with st.spinner("📄 뉴스 본문 로딩 중..."):
                                    try:
                                        news_body = crawler.get_news_body(news_link)
                                        if news_body:
                                            st.markdown(f"""{news_body}""")
                                        else:
                                            st.info("본문을 가져올 수 없습니다.")
                                    except Exception as e:
                                        st.error(f"본문 로드 실패: {str(e)}")
                            else:
                                st.info("기사 링크를 찾을 수 없습니다.")
                else:
                    st.info("최근 뉴스가 없습니다.")
            else:
                st.info("왼쪽에서 종목을 선택하면\n상세 정보가 표시됩니다.")


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
    
    st.markdown("""
    KOSPI 200 섹터별 성과 분석 및 트렌드 파악
    """)
    
    refresh_data = st.button("📊 섹터 분석 실행", use_container_width=True)
    
    if refresh_data:
        with st.spinner("섹터 분석 데이터 처리 중..."):
            try:
                import os
                stock_file = "data/stock.pkl"
                
                if not os.path.exists(stock_file):
                    st.warning("⚠️ 주가 데이터가 없습니다. 메인 대시보드에서 시그널을 먼저 수집해주세요.")
                else:
                    # 캐싱된 데이터 로드
                    kospi_df = load_kospi_data()
                    
                    if kospi_df.empty:
                        st.error("❌ 데이터를 불러올 수 없습니다.")
                    else:
                        # 캐싱된 종목명 매핑 로드
                        name_map = load_kospi_name_map()
                        
                        # 캐싱된 데이터 전처리
                        kospi_df = prepare_sector_analysis_data(kospi_df, name_map)
                        
                        if not kospi_df.empty:
                            # 섹터별 수익률
                            sector_perf = SectorAnalyzer.get_sector_performance(kospi_df)
                            
                            st.markdown("### 📈 섹터별 등락률")
                            st.dataframe(sector_perf, use_container_width=True)
                            
                            # 히트맵
                            if not sector_perf.empty:
                                fig = SectorAnalyzer.create_sector_heatmap(sector_perf)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # 섹터별 상위 종목
                            st.markdown("### 📊 섹터별 상위 종목")
                            top_stocks = SectorAnalyzer.get_top_stocks_by_sector(kospi_df, top_n=5)
                            
                            if top_stocks:
                                for sector, df_sector in top_stocks.items():
                                    with st.expander(f"{sector} ({len(df_sector)}개)"):
                                        st.dataframe(df_sector, use_container_width=True)
                            else:
                                st.info("이용 가능한 섹터 데이터가 없습니다.")
                        else:
                            st.warning("분석할 데이터가 없습니다.")
            
            except Exception as e:
                st.error(f"오류: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


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
    page, selected_main_tab = render_sidebar_menu()
    
    # 메인 대시보드에서 포트폴리오 페이지로 이동 요청이 있는 경우
    if 'selected_menu' in st.session_state:
        # 메뉴명을 페이지 키로 변환
        menu_to_page = {
            "🏠 메인 대시보드": "main",
            "📈 종목분석": "analysis",
            "💼 포트폴리오": "portfolio",
            "⚙️ 설정": "settings"
        }
        selected_menu = st.session_state.selected_menu
        if selected_menu in menu_to_page:
            page = menu_to_page[selected_menu]
            # 세션 상태 제거
            del st.session_state.selected_menu

    if page != "main":
        selected_main_tab = st.session_state.get("active_tab", "📊 시그널")
    
    # 페이지 라우팅
    if page == "portfolio":
        page_portfolio()
    elif page == "settings":
        page_settings()
    elif page == "analysis":
        # 종목분석 페이지
        from app.ui import render_stock_analysis_page
        from app.data import load_stock_data, load_kospi_list, load_finance_data
        from app.signals import build_signals
        
        # 데이터 로딩
        with st.spinner("📊 데이터 로딩 중..."):
            df = load_stock_data("data")
            kospi = load_kospi_list("data")
            finance_df = load_finance_data("data")
        
        # 시그널 생성
        with st.spinner("🔍 시그널 분석 중..."):
            signals = build_signals(
                df,
                10,  # turnover_window
                3.0,  # turnover_multiplier
                20, 5.0, 20, 2.0, 20, 2.0,
                ["Turnover Spike"],
                "ANY",
            )
            signals = signals.merge(kospi, on="code", how="left")
        
        # 기본 params
        params = {
            "data_dir": "data",
            "turnover_window": 10,
            "turnover_multiplier": 3.0,
        }
        
        render_stock_analysis_page(df, signals, finance_df, params)
    else:
        # 기존 메인 대시보드
        run_app(selected_main_tab)
