import os
import pickle
import pandas as pd
import streamlit as st
from datetime import datetime


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


@st.cache_data(ttl=600)  # 10분마다 캐시 갱신
def load_marketcap_history(data_dir: str) -> pd.DataFrame:
    """
    Returns:
    - date
    - code
    - name
    - market_cap

    NOTE:
    - 별도 시가총액 히스토리 파일이 없으면 close * totalCnt(상장주식수) 프록시를 사용합니다.
    - TODO: 추후 실제 시가총액 히스토리 소스가 생기면 본 로직을 교체하세요.
    """
    price_df = load_stock_data(data_dir)
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=["date", "code", "name", "market_cap"])

    px = price_df[["date", "code", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["date", "code", "close"]).sort_values(["code", "date"])

    finance_df = load_finance_data(data_dir)
    if finance_df is not None and not finance_df.empty and "totalCnt" in finance_df.columns:
        fin = finance_df[["date", "code", "totalCnt"]].copy()
        fin["date"] = pd.to_datetime(fin["date"], errors="coerce")
        fin["code"] = fin["code"].astype(str).str.zfill(6)
        fin["totalCnt"] = pd.to_numeric(fin["totalCnt"], errors="coerce")
        fin = fin.dropna(subset=["date", "code"]).sort_values(["code", "date"])
        if not fin.empty:
            merged = px.merge(fin, on=["date", "code"], how="left")
            merged["totalCnt"] = (
                merged.sort_values(["code", "date"])
                .groupby("code")["totalCnt"]
                .ffill()
                .bfill()
            )
            merged["market_cap"] = merged["close"] * merged["totalCnt"]
        else:
            merged = px.copy()
            merged["market_cap"] = merged["close"]
    else:
        merged = px.copy()
        merged["market_cap"] = merged["close"]

    name_df = load_kospi_list(data_dir)
    if name_df is None or name_df.empty:
        name_df = pd.DataFrame(columns=["code", "name"])
    else:
        name_df = name_df[["code", "name"]].copy()
        name_df["code"] = name_df["code"].astype(str).str.zfill(6)

    out = merged.merge(name_df, on="code", how="left")
    out["name"] = out["name"].fillna(out["code"])
    out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
    out = out.dropna(subset=["date", "code", "market_cap"]).copy()
    return out[["date", "code", "name", "market_cap"]].sort_values(["date", "market_cap"], ascending=[True, False])


@st.cache_data(ttl=120)
def get_data_status(data_dir: str = DATA_DIR_DEFAULT) -> dict:
    files = {
        "price": os.path.join(data_dir, "stock.pkl"),
        "finance": os.path.join(data_dir, "stock_finance_data.pkl"),
        "news": os.path.join(data_dir, "trading_input_log.json"),
    }

    status = {
        "price_ts": "-",
        "finance_ts": "-",
        "news_ts": "-",
        "last_success_fetch": "-",
    }

    timestamps = []
    for key, path in files.items():
        if os.path.exists(path):
            ts = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
            status[f"{key}_ts"] = ts
            timestamps.append(ts)

    if timestamps:
        status["last_success_fetch"] = max(timestamps)

    return status
