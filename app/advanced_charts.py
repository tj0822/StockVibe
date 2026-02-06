"""
고급 시각화 & 패턴 인식 모듈
- 캔들 패턴 인식
- 상관관계 분석
- 거래량 분석
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CandlePatternRecognizer:
    """캔들 패턴 인식"""
    
    @staticmethod
    def detect_doji(df: pd.DataFrame) -> List[int]:
        """도지 패턴 감지"""
        patterns = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            body = abs(row['Close'] - row['Open'])
            range_val = row['High'] - row['Low']
            
            # 몸통이 전체 범위의 10% 이하
            if range_val > 0 and (body / range_val) < 0.1:
                patterns.append(i)
        
        return patterns
    
    @staticmethod
    def detect_hammer(df: pd.DataFrame) -> List[int]:
        """망치형 패턴 감지"""
        patterns = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            body = abs(row['Close'] - row['Open'])
            lower_shadow = min(row['Open'], row['Close']) - row['Low']
            upper_shadow = row['High'] - max(row['Open'], row['Close'])
            
            # 아래꼬리가 몸통의 2배 이상, 위꼬리는 짧음
            if body > 0 and lower_shadow > 2 * body and upper_shadow < body:
                patterns.append(i)
        
        return patterns
    
    @staticmethod
    def detect_shooting_star(df: pd.DataFrame) -> List[int]:
        """유성형 패턴 감지"""
        patterns = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            body = abs(row['Close'] - row['Open'])
            lower_shadow = min(row['Open'], row['Close']) - row['Low']
            upper_shadow = row['High'] - max(row['Open'], row['Close'])
            
            # 위꼬리가 몸통의 2배 이상, 아래꼬리는 짧음
            if body > 0 and upper_shadow > 2 * body and lower_shadow < body:
                patterns.append(i)
        
        return patterns
    
    @staticmethod
    def detect_engulfing(df: pd.DataFrame) -> List[Tuple[int, str]]:
        """포용형 패턴 감지"""
        patterns = []
        
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            prev_body = abs(prev['Close'] - prev['Open'])
            curr_body = abs(curr['Close'] - curr['Open'])
            
            # 강세 포용형 (이전 음봉 + 현재 큰 양봉)
            if prev['Close'] < prev['Open'] and curr['Close'] > curr['Open']:
                if curr['Close'] > prev['Open'] and curr['Open'] < prev['Close']:
                    patterns.append((i, 'bullish'))
            
            # 약세 포용형 (이전 양봉 + 현재 큰 음봉)
            elif prev['Close'] > prev['Open'] and curr['Close'] < curr['Open']:
                if curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']:
                    patterns.append((i, 'bearish'))
        
        return patterns
    
    @staticmethod
    def annotate_patterns_on_chart(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
        """차트에 패턴 표시"""
        # 도지
        doji_indices = CandlePatternRecognizer.detect_doji(df)
        for idx in doji_indices:
            if idx < len(df):
                date = df.index[idx]
                price = df.iloc[idx]['High']
                fig.add_annotation(
                    x=date, y=price,
                    text="도지",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="blue",
                    font=dict(size=10, color="blue")
                )
        
        # 망치형
        hammer_indices = CandlePatternRecognizer.detect_hammer(df)
        for idx in hammer_indices:
            if idx < len(df):
                date = df.index[idx]
                price = df.iloc[idx]['Low']
                fig.add_annotation(
                    x=date, y=price,
                    text="망치",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    font=dict(size=10, color="green")
                )
        
        # 유성형
        star_indices = CandlePatternRecognizer.detect_shooting_star(df)
        for idx in star_indices:
            if idx < len(df):
                date = df.index[idx]
                price = df.iloc[idx]['High']
                fig.add_annotation(
                    x=date, y=price,
                    text="유성",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(size=10, color="red")
                )
        
        return fig


class CorrelationAnalyzer:
    """상관관계 분석"""
    
    @staticmethod
    def calculate_correlation(stock_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """종목 간 상관관계 계산"""
        # 데이터프레임으로 변환
        df = pd.DataFrame(stock_returns)
        
        # 상관계수 계산
        correlation = df.corr()
        
        return correlation
    
    @staticmethod
    def create_correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
        """상관관계 히트맵"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="상관계수")
        ))
        
        fig.update_layout(
            title="종목 간 상관관계 히트맵",
            xaxis_title="종목",
            yaxis_title="종목",
            height=600,
            width=800
        )
        
        return fig
    
    @staticmethod
    def find_similar_stocks(target_stock: str, 
                          correlation: pd.DataFrame, 
                          top_n: int = 5) -> pd.DataFrame:
        """유사한 움직임을 보이는 종목 찾기"""
        if target_stock not in correlation.columns:
            return pd.DataFrame()
        
        # 대상 종목과의 상관계수 정렬
        similar = correlation[target_stock].sort_values(ascending=False)
        
        # 자기 자신 제외
        similar = similar[similar.index != target_stock]
        
        result = pd.DataFrame({
            '종목': similar.head(top_n).index,
            '상관계수': similar.head(top_n).values
        })
        
        return result


class VolumeAnalyzer:
    """거래량 분석"""
    
    @staticmethod
    def detect_volume_spike(df: pd.DataFrame, threshold: float = 2.0) -> List[int]:
        """거래량 급증 감지
        
        Args:
            df: 가격 데이터 (Volume 컬럼 필요)
            threshold: 평균 대비 배수 (기본 2.0배)
        """
        if 'Volume' not in df.columns:
            return []
        
        # 20일 이동평균
        avg_volume = df['Volume'].rolling(window=20).mean()
        
        spikes = []
        for i in range(len(df)):
            if i >= 20:  # 이동평균 계산 가능한 시점부터
                if df.iloc[i]['Volume'] > avg_volume.iloc[i] * threshold:
                    spikes.append(i)
        
        return spikes
    
    @staticmethod
    def create_volume_profile_chart(df: pd.DataFrame) -> go.Figure:
        """거래량 프로파일 차트"""
        if df.empty or 'Volume' not in df.columns:
            return go.Figure()
        
        # 가격대별 거래량 집계
        price_min = df['Low'].min()
        price_max = df['High'].max()
        bins = np.linspace(price_min, price_max, 50)
        
        volume_profile = []
        for i in range(len(bins) - 1):
            low_bin = bins[i]
            high_bin = bins[i + 1]
            
            # 해당 가격대에 해당하는 거래량 합계
            mask = (df['Low'] <= high_bin) & (df['High'] >= low_bin)
            vol = df[mask]['Volume'].sum()
            
            volume_profile.append({
                'price': (low_bin + high_bin) / 2,
                'volume': vol
            })
        
        profile_df = pd.DataFrame(volume_profile)
        
        # 수평 막대 그래프
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=profile_df['volume'],
            y=profile_df['price'],
            orientation='h',
            marker=dict(color='rgba(50, 171, 96, 0.6)'),
            name='거래량'
        ))
        
        fig.update_layout(
            title="거래량 프로파일",
            xaxis_title="누적 거래량",
            yaxis_title="가격",
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_price_volume_chart(df: pd.DataFrame) -> go.Figure:
        """가격-거래량 결합 차트"""
        if df.empty:
            return go.Figure()
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('주가', '거래량'),
            vertical_spacing=0.05
        )
        
        # 캔들스틱
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='주가'
            ),
            row=1, col=1
        )
        
        # 거래량 (색상: 양봉 빨강, 음봉 파랑)
        colors = ['red' if close >= open else 'blue' 
                 for close, open in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                name='거래량',
                opacity=0.6
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="날짜", row=2, col=1)
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        
        return fig
    
    @staticmethod
    def analyze_volume_trend(df: pd.DataFrame) -> Dict:
        """거래량 추세 분석"""
        if df.empty or 'Volume' not in df.columns:
            return {}
        
        recent_volume = df['Volume'].tail(5).mean()
        avg_volume = df['Volume'].mean()
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # 최근 거래량 급증/급감 여부
        if volume_ratio > 1.5:
            trend = "급증"
            signal = "강한 관심"
        elif volume_ratio > 1.2:
            trend = "증가"
            signal = "관심 증가"
        elif volume_ratio < 0.7:
            trend = "감소"
            signal = "관심 감소"
        else:
            trend = "보통"
            signal = "정상"
        
        return {
            'recent_avg': recent_volume,
            'overall_avg': avg_volume,
            'ratio': volume_ratio,
            'trend': trend,
            'signal': signal
        }
