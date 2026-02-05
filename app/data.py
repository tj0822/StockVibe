import os
import pickle
import pandas as pd
import streamlit as st


DATA_DIR_DEFAULT = "data"


@st.cache_data(ttl=600)  # 10분마다 캐시 갱신
def load_stock_data(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "stock.pkl")
    with open(path, "rb") as fh:
        df = pickle.load(fh)
    required_cols = {"date", "code", "open", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in stock.pkl: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["code", "date"])
    return df


@st.cache_data(ttl=600)  # 10분마다 캐시 갱신
def load_kospi_list(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "kospi_list.pkl")
    with open(path, "rb") as fh:
        df = pickle.load(fh)
    required_cols = {"date", "code", "name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in kospi_list.pkl: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["code", "date"])
    latest = df.groupby("code").tail(1)[["code", "name"]]
    return latest


@st.cache_data(ttl=600)  # 10분마다 캐시 갱신
def load_kospi_index(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "kospi_index.pkl")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "index"])
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("kospi_index.pkl is not a DataFrame")
    if "date" not in df.columns or "index" not in df.columns:
        raise ValueError("Missing columns in kospi_index.pkl")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[["date", "index"]]


@st.cache_data(ttl=600)  # 10분마다 캐시 갱신
def load_finance_data(data_dir: str) -> pd.DataFrame:
    """재무 데이터 로드 (PER, PBR, EPS, BPS 등)"""
    path = os.path.join(data_dir, "stock_finance_data.pkl")
    if not os.path.exists(path):
        return pd.DataFrame()
    
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("stock_finance_data.pkl is not a DataFrame")
    
    # 날짜 파싱 (여러 형식 지원)
    if 'date' in df.columns:
        # '2021.02.26 기준(장마감)' 형식 처리
        df['date'] = df['date'].astype(str).str.extract(r'(\d{4}\.\d{2}\.\d{2})')[0]
        df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d', errors='coerce')
    
    # 숫자 컬럼 변환
    numeric_cols = ['per', 'eps', 'pbr', 'bps', 'dvr', 'estimate_per', 'estimate_eps',
                    'forignerHaveCnt', 'totalCnt', 'min52week', 'max52week']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 외국인 보유 비율 계산
    if 'forignerHaveCnt' in df.columns and 'totalCnt' in df.columns:
        df['foreigner_ratio'] = (df['forignerHaveCnt'] / df['totalCnt'] * 100).fillna(0)
    
    return df.dropna(subset=['date', 'code']).sort_values(['code', 'date'])
