"""기술적 지표 계산 모듈"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_moving_averages(df: pd.DataFrame, periods: list = [5, 20, 60, 120]) -> pd.DataFrame:
    """이동평균선 계산"""
    result = df.copy()
    
    for period in periods:
        result[f'ma{period}'] = result['close'].rolling(window=period).mean()
    
    return result


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI (Relative Strength Index) 계산"""
    result = df.copy()
    
    # 가격 변화량
    delta = result['close'].diff()
    
    # 상승/하락 분리
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # RS 계산
    rs = gain / loss
    
    # RSI 계산
    result['rsi'] = 100 - (100 / (1 + rs))
    
    return result


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence) 계산"""
    result = df.copy()
    
    # EMA 계산
    ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
    
    # MACD 라인
    result['macd'] = ema_fast - ema_slow
    
    # 시그널 라인
    result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
    
    # 히스토그램
    result['macd_hist'] = result['macd'] - result['macd_signal']
    
    return result


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """볼린저 밴드 계산"""
    result = df.copy()
    
    # 중심선 (이동평균)
    result['bb_middle'] = result['close'].rolling(window=period).mean()
    
    # 표준편차
    std = result['close'].rolling(window=period).std()
    
    # 상단/하단 밴드
    result['bb_upper'] = result['bb_middle'] + (std * std_dev)
    result['bb_lower'] = result['bb_middle'] - (std * std_dev)
    
    # %B (현재가 위치)
    result['bb_pct'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
    
    return result


def calculate_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    """스토캐스틱 오실레이터 계산"""
    result = df.copy()
    
    # 최고가/최저가
    low_min = result['low'].rolling(window=period).min()
    high_max = result['high'].rolling(window=period).max()
    
    # %K
    result['stoch_k'] = 100 * (result['close'] - low_min) / (high_max - low_min)
    
    # Smooth %K
    result['stoch_k'] = result['stoch_k'].rolling(window=smooth_k).mean()
    
    # %D (시그널)
    result['stoch_d'] = result['stoch_k'].rolling(window=smooth_d).mean()
    
    return result


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """OBV (On-Balance Volume) 계산"""
    result = df.copy()
    
    # 가격 변화
    price_change = result['close'].diff()
    
    # OBV 계산
    obv = []
    obv_value = 0
    
    for i in range(len(result)):
        if i == 0:
            obv.append(0)
        elif price_change.iloc[i] > 0:
            obv_value += result['volume'].iloc[i]
            obv.append(obv_value)
        elif price_change.iloc[i] < 0:
            obv_value -= result['volume'].iloc[i]
            obv.append(obv_value)
        else:
            obv.append(obv_value)
    
    result['obv'] = obv
    
    return result


def detect_golden_cross(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> bool:
    """골든크로스 감지 (단기 이평선이 장기 이평선을 상향 돌파)"""
    if f'ma{short_period}' not in df.columns or f'ma{long_period}' not in df.columns:
        df = calculate_moving_averages(df, [short_period, long_period])
    
    # 최근 2일 데이터
    recent = df.tail(2)
    
    if len(recent) < 2:
        return False
    
    # 전날: 단기 < 장기, 오늘: 단기 > 장기
    prev_short = recent.iloc[0][f'ma{short_period}']
    prev_long = recent.iloc[0][f'ma{long_period}']
    curr_short = recent.iloc[1][f'ma{short_period}']
    curr_long = recent.iloc[1][f'ma{long_period}']
    
    if pd.isna(prev_short) or pd.isna(prev_long) or pd.isna(curr_short) or pd.isna(curr_long):
        return False
    
    return prev_short < prev_long and curr_short > curr_long


def detect_dead_cross(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> bool:
    """데드크로스 감지 (단기 이평선이 장기 이평선을 하향 돌파)"""
    if f'ma{short_period}' not in df.columns or f'ma{long_period}' not in df.columns:
        df = calculate_moving_averages(df, [short_period, long_period])
    
    # 최근 2일 데이터
    recent = df.tail(2)
    
    if len(recent) < 2:
        return False
    
    # 전날: 단기 > 장기, 오늘: 단기 < 장기
    prev_short = recent.iloc[0][f'ma{short_period}']
    prev_long = recent.iloc[0][f'ma{long_period}']
    curr_short = recent.iloc[1][f'ma{short_period}']
    curr_long = recent.iloc[1][f'ma{long_period}']
    
    if pd.isna(prev_short) or pd.isna(prev_long) or pd.isna(curr_short) or pd.isna(curr_long):
        return False
    
    return prev_short > prev_long and curr_short < curr_long


def analyze_technical_indicators(df: pd.DataFrame) -> Dict:
    """종합 기술적 지표 분석"""
    if df.empty:
        return {}
    
    # 모든 지표 계산
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_stochastic(df)
    df = calculate_obv(df)
    
    # 최근 데이터
    latest = df.iloc[-1]
    
    analysis = {
        'price': latest['close'],
        'volume': latest['volume'],
    }
    
    # 이동평균선 분석
    ma_signals = []
    for period in [5, 20, 60, 120]:
        ma_key = f'ma{period}'
        if ma_key in latest and pd.notna(latest[ma_key]):
            ma_value = latest[ma_key]
            if latest['close'] > ma_value:
                ma_signals.append(f"MA{period} 상향")
            else:
                ma_signals.append(f"MA{period} 하향")
    
    analysis['ma_signals'] = ma_signals
    
    # 골든크로스/데드크로스
    analysis['golden_cross'] = detect_golden_cross(df)
    analysis['dead_cross'] = detect_dead_cross(df)
    
    # RSI 분석
    if 'rsi' in latest and pd.notna(latest['rsi']):
        rsi = latest['rsi']
        analysis['rsi'] = rsi
        
        if rsi > 70:
            analysis['rsi_signal'] = '과매수'
        elif rsi < 30:
            analysis['rsi_signal'] = '과매도'
        else:
            analysis['rsi_signal'] = '중립'
    
    # MACD 분석
    if 'macd' in latest and 'macd_signal' in latest:
        if pd.notna(latest['macd']) and pd.notna(latest['macd_signal']):
            macd = latest['macd']
            macd_signal = latest['macd_signal']
            
            analysis['macd'] = macd
            analysis['macd_signal'] = macd_signal
            
            if macd > macd_signal:
                analysis['macd_trend'] = '상승'
            else:
                analysis['macd_trend'] = '하락'
    
    # 볼린저 밴드 분석
    if 'bb_pct' in latest and pd.notna(latest['bb_pct']):
        bb_pct = latest['bb_pct']
        analysis['bb_pct'] = bb_pct
        
        if bb_pct > 0.8:
            analysis['bb_signal'] = '상단 근접 (과매수)'
        elif bb_pct < 0.2:
            analysis['bb_signal'] = '하단 근접 (과매도)'
        else:
            analysis['bb_signal'] = '중립'
    
    # 스토캐스틱 분석
    if 'stoch_k' in latest and pd.notna(latest['stoch_k']):
        stoch_k = latest['stoch_k']
        analysis['stoch_k'] = stoch_k
        
        if stoch_k > 80:
            analysis['stoch_signal'] = '과매수'
        elif stoch_k < 20:
            analysis['stoch_signal'] = '과매도'
        else:
            analysis['stoch_signal'] = '중립'
    
    return analysis


def get_technical_score(analysis: Dict) -> float:
    """기술적 지표 종합 점수 (-100 ~ +100)"""
    score = 0
    
    # 골든크로스/데드크로스 (±30점)
    if analysis.get('golden_cross'):
        score += 30
    if analysis.get('dead_cross'):
        score -= 30
    
    # RSI (±20점)
    if 'rsi_signal' in analysis:
        if analysis['rsi_signal'] == '과매도':
            score += 20  # 매수 기회
        elif analysis['rsi_signal'] == '과매수':
            score -= 20  # 매도 신호
    
    # MACD (±20점)
    if 'macd_trend' in analysis:
        if analysis['macd_trend'] == '상승':
            score += 20
        else:
            score -= 20
    
    # 볼린저 밴드 (±15점)
    if 'bb_signal' in analysis:
        if '하단' in analysis['bb_signal']:
            score += 15  # 반등 기회
        elif '상단' in analysis['bb_signal']:
            score -= 15
    
    # 스토캐스틱 (±15점)
    if 'stoch_signal' in analysis:
        if analysis['stoch_signal'] == '과매도':
            score += 15
        elif analysis['stoch_signal'] == '과매수':
            score -= 15
    
    return score
