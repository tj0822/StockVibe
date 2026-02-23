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
from long_term_analyzer import LongTermAnalyzer, create_investment_portfolio_recommendation

# 산업 트렌드 분석 (선택적 import)
try:
    from industry_trend_analyzer import IndustryTrendAnalyzer, get_industry_trend_recommendation
    HAS_INDUSTRY_TREND = True
except ImportError as e:
    HAS_INDUSTRY_TREND = False
    print(f"⚠️ 산업 트렌드 분석 모듈 로드 실패: {e}")


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
        "� 중장기 투자": "long_term",
        "�💼 포트폴리오": "portfolio",
        "⚙️ 설정": "settings"
    }
    
    selected = st.sidebar.radio("메뉴", ["메인 대시보드", "종목분석", "뉴스 트렌드", "중장기 투자", "포트폴리오", "설정"])
    
    page_map = {
        "메인 대시보드": "main",
        "종목분석": "analysis",
        "뉴스 트렌드": "industry_trend",
        "중장기 투자": "long_term",
        "포트폴리오": "portfolio",
        "설정": "settings"
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


def page_industry_trends():
    """
    🔥 뉴스 트렌드 분석 페이지
    최신 뉴스를 분석하여 유망한 산업을 식별하고 관련 종목을 추천
    """
    st.title("🔥 뉴스 트렌드 분석")
    
    st.markdown("""
    ### 📰 산업 트렌드 분석
    - **목표**: 최신 뉴스 분석을 통해 유망한 산업 식별
    - **방법**: 산업별 뉴스 수 + 감정 분석 + 시장 반응
    - **활용**: 미래 성장 가능성 높은 산업의 종목 투자
    """)
    
    # 모듈 가용성 확인
    if not HAS_INDUSTRY_TREND:
        st.error("""
        ❌ **뉴스 트렌드 분석 기능을 사용할 수 없습니다.**
        
        필요한 모듈이 로드되지 않았습니다. 다음을 확인해주세요:
        1. `industry_trend_analyzer.py` 파일이 존재하는지 확인
        2. 터미널에서 다음 명령 실행:
           ```
           pip install transformers torch
           ```
        3. Streamlit을 다시 시작해주세요
        
        임시로 중장기 투자 기능을 사용해주세요.
        """)
        return
    
    try:
        # 분석 옵션
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📊 산업 트렌드 분석")
            analysis_days = st.slider("분석 기간 (일)", min_value=1, max_value=30, value=7)
        
        with col2:
            if st.button("🔍 분석 실행", type="primary", key="analyze_trends", use_container_width=True):
                with st.spinner("📰 산업 뉴스 트렌드 분석 중... (이 과정은 1-2분 소요될 수 있습니다)"):
                    try:
                        analyzer = IndustryTrendAnalyzer()
                        st.session_state.industry_analyzer = analyzer
                        
                        # 트렌딩 산업 분석
                        trending_df = analyzer.get_trending_industries(top_n=10)
                        st.session_state.trending_industries = trending_df
                        
                        st.success("✅ 분석 완료!")
                    
                    except Exception as e:
                        st.error(f"❌ 분석 중 오류: {e}")
                        return
        
        if 'trending_industries' in st.session_state:
            trending_df = st.session_state.trending_industries
            
            # 탭 생성
            tab1, tab2, tab3 = st.tabs(["🏆 유망 산업", "📰 산업별 뉴스", "📈 종목 추천"])
            
            # ===== TAB 1: 유망 산업 =====
            with tab1:
                st.markdown("### 🔥 TOP 유망 산업")
                
                # 상점 메트릭 (상위 3개)
                if len(trending_df) >= 3:
                    col1, col2, col3 = st.columns(3)
                    
                    for idx, (col, row) in enumerate(zip([col1, col2, col3], trending_df.head(3).itertuples())):
                        with col:
                            # 산업명과 점수
                            st.metric(
                                row.산업,
                                f"{row.유망도:.1f}/100",
                                f"외신 {int(row.뉴스수)}개"
                            )
                
                st.markdown("---")
                
                # 전체 산업 랭킹
                st.markdown("### 📊 산업별 유망도 순위")
                
                # 데이터 표시
                display_df = trending_df.copy()
                display_df = display_df.astype({
                    '유망도': float,
                    '뉴스수': int,
                    '감정점수': float,
                    '긍정': int,
                    '중립': int,
                    '부정': int
                })
                
                # 컬럼 재정렬
                display_df = display_df[['산업', '유망도', '뉴스수', '감정점수', '긍정', '중립', '부정']]
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # 차트
                col1, col2 = st.columns(2)
                
                with col1:
                    # 유망도 차트
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=trending_df['산업'],
                        x=trending_df['유망도'],
                        orientation='h',
                        marker=dict(
                            color=trending_df['유망도'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="유망도")
                        ),
                        text=trending_df['유망도'].round(1),
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title="산업별 유망도",
                        xaxis_title="유망도 점수",
                        yaxis_title="산업",
                        height=500,
                        margin=dict(l=150)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 감정 점수와 뉴스 수
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        y=trending_df['산업'],
                        x=trending_df['감정점수'],
                        mode='markers',
                        name='감정점수',
                        marker=dict(
                            size=trending_df['뉴스수'] * 2,
                            color=trending_df['감정점수'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="감정점수"),
                            line=dict(color='white', width=2)
                        ),
                        text=trending_df['산업'],
                        customdata=trending_df['뉴스수'].astype(int),
                        hovertemplate="<b>%{text}</b><br>감정점수: %{x:.1f}<br>뉴스 수: %{customdata}<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        title="감정점수 vs 뉴스수 (버블 크기 = 뉴스수)",
                        xaxis_title="감정점수",
                        yaxis_title="산업",
                        height=500,
                        hovermode='closest'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # ===== TAB 2: 산업별 뉴스 =====
            with tab2:
                st.markdown("### 📰 산업별 최근 뉴스")
                
                selected_industry = st.selectbox(
                    "산업 선택",
                    options=trending_df['산업'].tolist(),
                    index=0
                )
                
                if selected_industry and 'industry_analyzer' in st.session_state:
                    analyzer = st.session_state.industry_analyzer
                    industry_details = analyzer.get_industry_details(selected_industry)
                    
                    if industry_details:
                        # 산업 정보 요약
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("유망도", f"{industry_details['trend_score']:.1f}/100")
                        with col2:
                            st.metric("감정점수", f"{industry_details['sentiment_score']:.1f}/100")
                        with col3:
                            st.metric("뉴스수", industry_details['news_count'])
                        with col4:
                            st.metric("관련종목수", len(industry_details['stocks']))
                        
                        st.markdown("---")
                        
                        # 분석 의견
                        st.markdown("### 🔍 산업 분석")
                        st.info(industry_details['analysis'])
                        
                        st.markdown("---")
                        
                        # 최근 뉴스
                        st.markdown("### 🌟 최근 뉴스")
                        
                        recent_news = industry_details['recent_news']
                        
                        if not recent_news.empty:
                            for idx, news in recent_news.head(10).iterrows():
                                # 뉴스 카드
                                with st.expander(f"📰 {news['제목'][:60]}..."):
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.caption(f"출처: {news.get('출처', 'N/A')}")
                                        st.caption(f"종목: {news.get('name', 'N/A')} ({news.get('code', 'N/A')})")
                                    
                                    with col2:
                                        st.caption(f"날짜: {news.get('날짜', 'N/A')}")
                        else:
                            st.info("최근 뉴스가 없습니다.")
            
            # ===== TAB 3: 종목 추천 =====
            with tab3:
                st.markdown("### 📈 유망 산업 관련 종목 추천")
                
                # 추천 산업 선택
                selected_rec_industry = st.selectbox(
                    "산업 선택",
                    options=trending_df['산업'].tolist(),
                    index=0,
                    key="recommendation_industry"
                )
                
                # 추천 종목 수
                num_recommendations = st.slider(
                    "추천 종목 수",
                    min_value=3,
                    max_value=20,
                    value=10,
                    key="num_recommendations"
                )
                
                if st.button("📊 종목 추천", type="primary", use_container_width=True):
                    if selected_rec_industry and 'industry_analyzer' in st.session_state:
                        analyzer = st.session_state.industry_analyzer
                        industry_data = analyzer.get_industry_details(selected_rec_industry)
                        
                        if industry_data:
                            from industry_trend_analyzer import get_industry_trend_recommendation
                            
                            recommendations = get_industry_trend_recommendation(
                                industry_data,
                                kospi_list=None
                            )
                            
                            st.session_state.industry_recommendations = recommendations
                            st.success(f"✅ {selected_rec_industry} 관련 {len(recommendations)}개 종목 추천 완료!")
                
                # 추천 결과 표시
                if 'industry_recommendations' in st.session_state:
                    recommendations = st.session_state.industry_recommendations
                    
                    # 메트릭
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("산업 유망도", 
                                trending_df[trending_df['산업'] == selected_rec_industry]['유망도'].values[0] 
                                if selected_rec_industry in trending_df['산업'].values else 0)
                    with col2:
                        st.metric("추천 종목 수", len(recommendations))
                    
                    st.markdown("---")
                    
                    # 종목 정보 테이블
                    st.markdown("### 📋 추천 종목 목록")
                    
                    display_rec = recommendations.head(num_recommendations).copy()
                    display_rec = display_rec[['code', 'name', 'level', 'score']]
                    display_rec.columns = ['종목코드', '종목명', '추천등급', '점수']
                    display_rec['점수'] = display_rec['점수'].round(1)
                    
                    st.dataframe(display_rec, use_container_width=True, hide_index=True)
                    
                    # 점수 차트
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=recommendations['name'],
                        y=recommendations['score'],
                        text=recommendations['score'].round(1),
                        textposition='auto',
                        marker=dict(
                            color=recommendations['score'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="점수")
                        ),
                        hovertemplate="<b>%{x}</b><br>점수: %{y:.1f}<extra></extra>"
                    ))
                    fig.update_layout(
                        title=f"{selected_rec_industry} 추천 종목",
                        xaxis_title="종목",
                        yaxis_title="추천 점수",
                        height=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👈 왼쪽에서 '분석 실행' 버튼을 클릭하여 산업 트렌드 분석을 시작하세요.")
    
    except Exception as e:
        st.error(f"❌ 오류: {e}")
        import traceback
        with st.expander("🔍 상세 오류 정보"):
            st.error(traceback.format_exc())


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


def page_long_term_investment():
    """
    💎 중장기 투자 추천 페이지
    재무 정보와 뉴스를 종합하여 3-5년 이상 우상향할 수 있는 종목 추천
    """
    st.title("💎 중장기 투자 추천")
    
    st.markdown("""
    ### 📊 시스템 소개
    - **분석 기준**: 재무 건전성 (40%) + 밸류에이션 (30%) + 모멘텀 (30%)
    - **투자 기간**: 최소 3~5년 장기 보유 기준
    - **목표**: 안정적인 배당 + 꾸준한 성장성
    """)
    
    # 데이터 로드
    from app.data import load_finance_data, load_stock_data
    
    with st.spinner("📊 데이터 로딩 중..."):
        try:
            finance_df = load_finance_data("data")
            price_df = load_stock_data("data")
            kospi_dict = load_kospi_name_map()
            
            if finance_df.empty:
                st.error("❌ 재무 데이터가 없습니다.")
                st.info("""
                📋 **해결 방법:**
                1. 메인 대시보드 → 🔄 데이터 탭으로 이동
                2. 데이터 수집 버튼 클릭
                3. 재무 데이터와 주가 데이터 모두 로드 완료 후 다시 시도
                """)
                return
            
            if price_df.empty:
                st.error("❌ 주가 데이터가 없습니다.")
                st.info("""
                📋 **해결 방법:**
                1. 메인 대시보드 → 🔄 데이터 탭으로 이동
                2. 데이터 수집 버튼 클릭
                3. 주가 데이터 로드 완료 후 다시 시도
                """)
                return
            
            # 데이터 통계 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 재무 데이터 행", len(finance_df))
            with col2:
                st.metric("💹 주가 데이터 행", len(price_df))
            with col3:
                unique_codes = finance_df['code'].nunique() if 'code' in finance_df.columns else 0
                st.metric("🏢 종목 수", unique_codes)
            
        except Exception as e:
            st.error(f"❌ 데이터 로드 오류: {e}")
            st.error(f"세부 정보: {str(e)}")
            return
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["🎯 추천 종목", "📊 상세 분석", "💰 포트폴리오 구성"])
    
    # ===== TAB 1: 추천 종목 =====
    with tab1:
        st.subheader("🎯 추천 종목 순위")
        
        st.markdown("""
        💡 **팁**: 최소 점수가 높을수록 더 우수한 종목만 선별됩니다.
        처음 실행 시 최소 점수를 **30~40점** 정도로 낮춘 후 시작하는 것을 추천합니다.
        """)
        
        # 분석 파라미터 섹션
        st.markdown("### ⚙️ 분석 파라미터")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_stocks = st.slider("추천 종목 수", 5, 30, 10)
        
        with col2:
            min_score = st.slider("최소 재무 건전성 점수", 20, 80, 40)
        
        # 가중치 조정 섹션
        st.markdown("### 📊 분석 기준 가중치")
        st.markdown("""
        가중치를 조정하여 분석 초점을 변경할 수 있습니다.
        **합계가 100%가 되어야 합니다.**
        """)
        
        # 슬라이더 컬럼 레이아웃
        col_w1, col_w2, col_w3 = st.columns(3)
        
        with col_w1:
            weight_fundamental = st.slider(
                "재무 건전성 (%)",
                min_value=10,
                max_value=70,
                value=40,
                step=5,
                key="weight_fund"
            )
        
        with col_w2:
            weight_valuation = st.slider(
                "밸류에이션 (%)",
                min_value=10,
                max_value=70,
                value=30,
                step=5,
                key="weight_val"
            )
        
        with col_w3:
            weight_momentum = st.slider(
                "모멘텀 (%)",
                min_value=10,
                max_value=70,
                value=30,
                step=5,
                key="weight_mom"
            )
        
        # 가중치 합계 확인
        total_weight = weight_fundamental + weight_valuation + weight_momentum
        
        # 합계 표시 및 검증
        col_sum1, col_sum2 = st.columns([2, 1])
        with col_sum1:
            if total_weight == 100:
                st.success(f"✅ 가중치 합계: {total_weight}%")
            else:
                st.warning(f"⚠️ 가중치 합계: {total_weight}% (100%가 되도록 조정해주세요)")
        
        with col_sum2:
            # 기본값으로 리셋 버튼
            if st.button("🔄 기본값 리셋", use_container_width=True):
                st.session_state.weight_fund = 40
                st.session_state.weight_val = 30
                st.session_state.weight_mom = 30
                st.rerun()
        
        # 가중치 설명
        with st.expander("🔍 가중치별 투자 성향"):
            st.markdown("""
            **재무 건전성** (기본: 40%)
            - 높을수록: ROE, 영업이익률 등 기업 기본 체질 중시
            - 안정성, 배당금, 장기 수익성 추구
            
            **밸류에이션** (기본: 30%)
            - 높을수록: PER, PBR 등 저평가 상태 중시
            - 단기 상승 여력, 가성비 투자 추구
            
            **모멘텀** (기본: 30%)
            - 높을수록: 최근 주가 추세 중시
            - 최근 상승 추세, 시장 인기도 중시
            
            **추천 조합:**
            - 🟢 안정형: 재무 50% + 밸류 30% + 모멘텀 20%
            - 🟡 균형형: 재무 40% + 밸류 30% + 모멘텀 30% (기본)
            - 🔴 공격형: 재무 30% + 밸류 20% + 모멘텀 50%
            """)
        
        if st.button("🔍 추천 종목 분석", use_container_width=True, type="primary"):
            # 가중치 검증
            if total_weight != 100:
                st.error("❌ 가중치 합계가 100%가 아닙니다. 다시 조정해주세요.")
            else:
                with st.spinner(f"🔍 {num_stocks}개 종목 분석 중... (가중치: 재무 {weight_fundamental}%, 밸류 {weight_valuation}%, 모멘텀 {weight_momentum}%)"):
                    try:
                        analyzer = LongTermAnalyzer(finance_df, price_df)
                        recommendations = analyzer.recommend_long_term_stocks(
                            num_stocks=num_stocks,
                            min_fundamental_score=min_score,
                            kospi_list=kospi_dict,
                            weight_fundamental=weight_fundamental / 100,
                            weight_valuation=weight_valuation / 100,
                            weight_momentum=weight_momentum / 100
                        )
                        
                        if recommendations.empty:
                            st.warning("""
                            ⚠️ **조건에 맞는 추천 종목이 없습니다.**
                            
                            📋 **원인 분석:**
                            - 선택한 최소 점수가 너무 높을 수 있습니다 (권장: 30~40점)
                            - 재무 데이터 컬럼명이 예상과 다를 수 있습니다
                            - 주가 데이터가 부족할 수 있습니다
                            
                            🔧 **해결 방법:**
                            1. **최소 점수를 낮춰보세요** (현재: {0}점 → 30점으로 시도)
                            2. **메인 대시보드**에서 데이터 재수집
                            3. 페이지 새로고침 (F5) 후 다시 시도
                            """.format(min_score))
                        else:
                            # 세션에 저장 (가중치도 함께)
                            st.session_state.long_term_recommendations = recommendations
                            st.session_state.last_weights = {
                                'fundamental': weight_fundamental / 100,
                                'valuation': weight_valuation / 100,
                                'momentum': weight_momentum / 100
                            }
                            
                            st.success(f"✅ {len(recommendations)}개 종목 분석 완료!")
                            
                            # 추천 종목 테이블
                            st.markdown("### 📋 추천 종목 목록")
                            
                            display_df = recommendations[
                                ['name', 'code', 'total_score', 'fundamental_score', 
                                 'valuation_score', 'momentum_score', 'roe', 'per', 
                                 'pbr', 'trend', 'reasons']
                            ].copy()
                            
                            display_df.columns = [
                                '종목명', '코드', '종합점수',
                                '재무점수', '밸류에이션점수', '모멘텀점수',
                                'ROE(%)', 'PER(배)', 'PBR(배)', '추세', '추천이유'
                            ]
                            
                            display_df = display_df.sort_values('종합점수', ascending=False)
                            
                            # 조건부 포매팅이 있는 데이터프레임
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                            
                            # 점수 분포 시각화
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # 종합점수 차트
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    y=recommendations['name'],
                                    x=recommendations['total_score'],
                                    orientation='h',
                                    marker=dict(
                                        color=recommendations['total_score'],
                                        colorscale='RdYlGn',
                                        showscale=True,
                                        colorbar=dict(title="종합점수")
                                    ),
                                    text=recommendations['total_score'].round(1),
                                    textposition='auto',
                                ))
                                fig.update_layout(
                                    title="종합 투자점수",
                                    xaxis_title="점수",
                                    yaxis_title="종목",
                                    height=600,
                                    margin=dict(l=150)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # 평가 구성 요소별 스택
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    y=recommendations['name'],
                                    x=recommendations['fundamental_score'],
                                    name='재무 건전성',
                                    orientation='h',
                                    marker=dict(color='#636EFA')
                                ))
                                fig.add_trace(go.Bar(
                                    y=recommendations['name'],
                                    x=recommendations['valuation_score'],
                                    name='밸류에이션',
                                    orientation='h',
                                    marker=dict(color='#EF553B')
                                ))
                                fig.add_trace(go.Bar(
                                    y=recommendations['name'],
                                    x=recommendations['momentum_score'],
                                    name='모멘텀',
                                    orientation='h',
                                    marker=dict(color='#00CC96')
                                ))
                                fig.update_layout(
                                    barmode='stack',
                                    title="평가 점수 구성",
                                    xaxis_title="점수",
                                    yaxis_title="종목",
                                    height=600,
                                    margin=dict(l=150),
                                    hovermode='y unified'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ 분석 중 오류: {e}")
                        import traceback
                        with st.expander("🔍 상세 오류 정보"):
                            st.error(traceback.format_exc())
    
    # ===== TAB 2: 상세 분석 =====
    with tab2:
        st.subheader("📊 종목별 상세 분석")
        
        if 'long_term_recommendations' not in st.session_state:
            st.info("👈 먼저 추천 종목을 분석해주세요.")
        else:
            recommendations = st.session_state.long_term_recommendations
            
            # 종목 선택
            stock_options = [f"{row['name']} ({row['code']})" 
                           for _, row in recommendations.iterrows()]
            
            selected_stock = st.selectbox("종목 선택", stock_options)
            
            if selected_stock:
                # 선택한 종목의 데이터 추출
                selected_idx = stock_options.index(selected_stock)
                selected_code = recommendations.iloc[selected_idx]['code']
                selected_name = recommendations.iloc[selected_idx]['name']
                
                with st.spinner(f"📊 {selected_name} 상세 분석 중..."):
                    try:
                        analyzer = LongTermAnalyzer(finance_df, price_df)
                        
                        # 저장된 가중치 불러오기 (없으면 기본값 사용)
                        weights = st.session_state.get('last_weights', {
                            'fundamental': 0.40,
                            'valuation': 0.30,
                            'momentum': 0.30
                        })
                        
                        details = analyzer.get_stock_recommendation_details(
                            selected_code,
                            selected_name,
                            weight_fundamental=weights.get('fundamental', 0.40),
                            weight_valuation=weights.get('valuation', 0.30),
                            weight_momentum=weights.get('momentum', 0.30)
                        )
                        
                        # 상단: 추천 레벨 및 점수
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("추천 레벨", details['level'])
                        with col2:
                            st.metric("종합 점수", f"{details['total_score']:.1f}/100")
                        with col3:
                            st.metric("재무 점수", f"{details['fundamental']['score']}")
                        with col4:
                            st.metric("밸류에이션 점수", f"{details['valuation']['score']}")
                        
                        st.markdown("---")
                        
                        # 재무 지표
                        st.markdown("### 💰 재무 지표")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("ROE", f"{details['fundamental'].get('roe', 0):.1f}%")
                        with col2:
                            st.metric("영업이익률", f"{details['fundamental'].get('operating_margin', 0):.1f}%")
                        with col3:
                            st.metric("이익 성장률", f"{details['fundamental'].get('profit_growth', 0):.1f}%")
                        with col4:
                            st.metric("PBR", f"{details['valuation'].get('pbr', 0):.2f}배")
                        
                        st.markdown("---")
                        
                        # 밸류에이션
                        st.markdown("### 💵 밸류에이션")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("PER", f"{details['valuation'].get('per', 0):.1f}배")
                            st.caption("낮을수록 저평가 상태")
                        
                        with col2:
                            st.metric("1년 수익률", f"{details['momentum'].get('return_1y', 0):.1f}%")
                            st.caption(f"추세: {details['momentum'].get('trend', '미정')}")
                        
                        st.markdown("---")
                        
                        # 투자 전망
                        st.markdown("### 🔮 투자 전망")
                        for outlook in details.get('outlook', []):
                            st.markdown(f"- {outlook}")
                        
                        st.markdown("---")
                        
                        # 추천 이유 정리
                        st.markdown("### ✅ 추천 이유")
                        
                        reasons_list = []
                        
                        # 재무 기반
                        if details['fundamental'].get('roe', 0) > 15:
                            reasons_list.append(f"✔️ **높은 수익성**: ROE {details['fundamental'].get('roe', 0):.1f}% (안정적 배당 기대)")
                        
                        if details['fundamental'].get('operating_margin', 0) > 10:
                            reasons_list.append(f"✔️ **우수한 경영효율**: 영업이익률 {details['fundamental'].get('operating_margin', 0):.1f}%")
                        
                        if details['fundamental'].get('profit_growth', 0) > 5:
                            reasons_list.append(f"✔️ **실적 성장성**: 이익 성장률 {details['fundamental'].get('profit_growth', 0):.1f}%")
                        
                        # 밸류에이션 기반
                        if details['valuation'].get('per', 20) < 15:
                            reasons_list.append(f"✔️ **저평가 상태**: PER {details['valuation'].get('per', 20):.1f}배 (상승 여유 있음)")
                        
                        if details['valuation'].get('pbr', 1.0) < 1.0:
                            reasons_list.append(f"✔️ **주가순자산비율 저가**: PBR {details['valuation'].get('pbr', 1.0):.2f}배 (순자산 대비 저가)")
                        
                        # 모멘텀 기반
                        if details['momentum'].get('return_1y', 0) > 10:
                            reasons_list.append(f"✔️ **긍정적 추세**: 1년 수익률 {details['momentum'].get('return_1y', 0):.1f}%")
                        
                        if reasons_list:
                            for reason in reasons_list:
                                st.markdown(reason)
                        else:
                            st.info("기본적으로 우량한 종목입니다.")
                        
                        st.markdown("---")
                        
                        # 주의사항
                        st.warning("""
                        ⚠️ **투자 유의사항**
                        - 과거 실적이 미래를 보장하지 않습니다.
                        - 본 분석은 참고자료일 뿐 투자 권고가 아닙니다.
                        - 충분한 조사 후 신중하게 투자 결정하세요.
                        - 본인의 투자 목표와 위험도를 고려하여 포트폴리오를 구성하세요.
                        """)
                    
                    except Exception as e:
                        st.error(f"❌ 상세 분석 오류: {e}")
    
    # ===== TAB 3: 포트폴리오 구성 =====
    with tab3:
        st.subheader("💰 추천 포트폴리오 구성")
        
        if 'long_term_recommendations' not in st.session_state:
            st.info("👈 먼저 추천 종목을 분석해주세요.")
        else:
            recommendations = st.session_state.long_term_recommendations
            
            col1, col2 = st.columns(2)
            
            with col1:
                total_investment = st.number_input(
                    "총 투자 금액 (원)",
                    value=10_000_000,
                    step=1_000_000,
                    min_value=1_000_000
                )
            
            with col2:
                num_portfolio_stocks = st.slider(
                    "포트폴리오 구성 종목 수",
                    min_value=3,
                    max_value=len(recommendations),
                    value=min(10, len(recommendations))
                )
            
            if st.button("📊 포트폴리오 구성", use_container_width=True, type="primary"):
                with st.spinner("포트폴리오 구성 중..."):
                    try:
                        # 상위 종목으로 구성
                        top_stocks = recommendations.head(num_portfolio_stocks)
                        
                        # 포트폴리오 배분
                        portfolio_result = create_investment_portfolio_recommendation(
                            top_stocks,
                            total_investment=total_investment
                        )
                        
                        st.session_state.portfolio_result = portfolio_result
                        
                        # 포트폴리오 전략
                        st.markdown("### 🎯 포트폴리오 전략")
                        st.info(f"**{portfolio_result['strategy']}**")
                        
                        # 배분 현황
                        st.markdown("### 💼 종목별 배분")
                        
                        portfolio_df = pd.DataFrame(portfolio_result['portfolio'])
                        portfolio_df['비중(%)'] = (
                            portfolio_df['allocation'] / total_investment * 100
                        ).round(1)
                        portfolio_df['배분액'] = portfolio_df['allocation'].apply(
                            lambda x: f"₩{x:,.0f}"
                        )
                        
                        display_df = portfolio_df[['code', 'name', '비중(%)', '배분액', 'total_score']]
                        display_df.columns = ['코드', '종목명', '비중', '배분액', '종합점수']
                        display_df['종합점수'] = display_df['종합점수'].round(1)
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # 배분 차트
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 파이 차트
                            fig = go.Figure(data=[go.Pie(
                                labels=portfolio_df['name'],
                                values=portfolio_df['allocation'],
                                textinfo="label+percent",
                                hovertemplate="<b>%{label}</b><br>₩%{value:,.0f}<br>%{percent}"
                            )])
                            fig.update_layout(
                                title="포트폴리오 구성 비중",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # 종목 점수 비교
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=portfolio_df['name'],
                                y=portfolio_df['total_score'],
                                text=portfolio_df['total_score'].round(1),
                                textposition='auto',
                                marker=dict(
                                    color=portfolio_df['total_score'],
                                    colorscale='RdYlGn',
                                    showscale=True,
                                    colorbar=dict(title="종합점수")
                                )
                            ))
                            fig.update_layout(
                                title="포트폴리오 종목 점수",
                                xaxis_title="종목",
                                yaxis_title="점수",
                                height=500,
                                xaxis_tickangle=-45
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 요약 통계
                        st.markdown("### 📊 포트폴리오 통계")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_score = portfolio_df['total_score'].mean()
                            st.metric("평균 종합점수", f"{avg_score:.1f}/100")
                        
                        with col2:
                            min_score = portfolio_df['total_score'].min()
                            st.metric("최저 종합점수", f"{min_score:.1f}/100")
                        
                        with col3:
                            st.metric("구성 종목 수", f"{len(portfolio_df)}개")
                        
                        with col4:
                            st.metric("총 배분액", f"₩{total_investment:,.0f}")
                        
                        # 포트폴리오 다운로드
                        st.markdown("---")
                        st.markdown("### 📥 포트폴리오 저장")
                        
                        # CSV로 변환
                        csv = display_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 포트폴리오 CSV 다운로드",
                            data=csv,
                            file_name=f"long_term_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"❌ 포트폴리오 구성 오류: {e}")
                        import traceback
                        st.error(traceback.format_exc())


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
    elif page == "long_term":
        page_long_term_investment()
    elif page == "industry_trend":
        page_industry_trends()
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
