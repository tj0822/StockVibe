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
    """포트폴리오 관리 페이지 (간소화 버전)"""
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
    
    # KOSPI 200 종목 데이터 로드
    kospi_dict = load_kospi_stocks()  # {종목코드: 종목명}
    
    # 2열 레이아웃: 왼쪽(보유종목), 오른쪽(종목추가)
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📊 보유 종목")
        
        portfolio = portfolio_mgr.load_portfolio()
        
        if not portfolio:
            st.info("보유 종목이 없습니다. 오른쪽에서 종목을 추가해주세요.")
        else:
            # 종목 목록 표시
            for code, info in portfolio.items():
                col_name, col_btn = st.columns([4, 1])
                with col_name:
                    st.markdown(f"**{info['name']}** ({code})")
                with col_btn:
                    if st.button("🗑️", key=f"del_{code}", help="삭제"):
                        portfolio_mgr.remove_from_portfolio(code)
                        st.rerun()
            
            st.markdown("---")
            st.caption(f"총 {len(portfolio)}개 종목")
    
    with col_right:
        st.subheader("➕ 종목 추가")
        
        if kospi_dict:
            # 현재 포트폴리오의 종목코드 목록
            current_codes = set(portfolio_mgr.load_portfolio().keys())
            
            # 이미 보유한 종목 제외
            available_stocks = {code: name for code, name in kospi_dict.items() 
                               if code not in current_codes}
            
            if not available_stocks:
                st.success("모든 KOSPI 200 종목을 보유 중입니다!")
            else:
                # 드롭다운 옵션: 종목명 (종목코드) 형식
                stock_options = ["선택하세요..."] + [f"{name} ({code})" 
                                for code, name in sorted(available_stocks.items(), key=lambda x: x[1])]
                
                selected = st.selectbox(
                    "종목 선택",
                    options=stock_options,
                    label_visibility="collapsed"
                )
                
                if selected and selected != "선택하세요...":
                    # 종목코드와 종목명 추출
                    parts = selected.split(" (")
                    stock_name = parts[0]
                    stock_code = parts[1].rstrip(")")
                    
                    if st.button("추가", type="primary", use_container_width=True):
                        # 기본값으로 저장 (수량=1, 가격=0)
                        portfolio_mgr.add_to_portfolio(stock_code, stock_name, 1, 0)
                        st.success(f"✅ {stock_name} 추가됨")
                        st.rerun()
        else:
            st.error("종목 데이터를 불러올 수 없습니다.")


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
    page, selected_main_tab = render_sidebar_menu()
    
    # 메인 대시보드에서 포트폴리오 페이지로 이동 요청이 있는 경우
    if 'selected_menu' in st.session_state:
        # 메뉴명을 페이지 키로 변환
        menu_to_page = {
            "🏠 메인 대시보드": "main",
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
    else:
        # 기존 메인 대시보드
        from app.ui import run_app
        run_app(selected_main_tab)
