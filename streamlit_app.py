import streamlit as st
from app.phase4_ui import run_phase4_app

# 페이지 설정
st.set_page_config(
    page_title="StockVibe Pro - AI Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 세션 초기화
if 'page' not in st.session_state:
    st.session_state.page = '메인 대시보드'
if 'data_refresh_time' not in st.session_state:
    import datetime
    st.session_state.data_refresh_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 커스텀 CSS - 다크 테마 최적화
st.markdown("""
<style>
    /* 메인 컨테이너 */
    .main > div {
        padding-top: 1rem;
    }
    
    /* 헤더 스타일 */
    h1 {
        color: #58a6ff;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #79c0ff;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #8b949e;
        font-weight: 500;
    }
    
    /* 카드 스타일 */
    .stExpander {
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        background-color: #161b22;
    }
    
    /* 메트릭 카드 */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 600;
        color: #c9d1d9;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #8b949e;
    }
    
    /* 버튼 스타일 */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s;
        border: 1px solid #30363d;
        background-color: #21262d;
        color: #c9d1d9;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border-color: #58a6ff;
        background-color: #30363d;
    }
    
    /* 다운로드 버튼 */
    .stDownloadButton>button {
        background-color: #238636;
        color: white;
        border: none;
    }
    
    .stDownloadButton>button:hover {
        background-color: #2ea043;
    }
    
    /* 테이블 스타일 */
    .dataframe {
        font-size: 0.85rem;
        border-radius: 8px;
        overflow: hidden;
        background-color: #161b22;
        color: #c9d1d9;
    }
    
    /* 테이블 헤더 */
    .dataframe thead tr th {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border-bottom: 1px solid #30363d !important;
    }
    
    /* 테이블 행 */
    .dataframe tbody tr {
        border-bottom: 1px solid #21262d !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #1c2128 !important;
    }
    
    /* 사이드바 */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 1rem;
    }
    
    /* 탭 스타일 */
    .stRadio > div {
        gap: 0.5rem;
        background-color: #161b22;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stRadio > div > label {
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        transition: all 0.2s;
        background-color: transparent;
        font-weight: 500;
        color: #8b949e;
    }
    
    .stRadio > div > label:has(input:checked) {
        background-color: #21262d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: #58a6ff;
    }
    
    /* 입력 필드 */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 6px;
        border: 1px solid #30363d;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #58a6ff;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border-color: #30363d;
    }
    
    /* Info 박스 */
    .stAlert {
        border-radius: 8px;
        background-color: #161b22;
        border: 1px solid #30363d;
    }
    
    /* 캡션 */
    .css-1v0mbdj, [data-testid="stCaptionContainer"] {
        color: #8b949e;
        font-size: 0.85rem;
    }
    
    /* DateInput */
    .stDateInput > div > div > input {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    
    /* 차트 배경 */
    .js-plotly-plot {
        background-color: #0d1117 !important;
    }
    
    /* 스피너 */
    .stSpinner > div {
        border-color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# Phase 4 앱 실행
if __name__ == "__main__":
    run_phase4_app()
