"""
비교 분석 & 섹터 분석 모듈
- 여러 종목 동시 비교
- 섹터별 분석
- 히트맵
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px

class ComparisonAnalyzer:
    """종목 비교 분석"""
    
    @staticmethod
    def compare_stocks(stock_data: Dict[str, Dict]) -> pd.DataFrame:
        """여러 종목 비교
        
        Args:
            stock_data: {code: {'name': str, 'price': float, 'change': float, ...}}
        """
        data = []
        
        for code, info in stock_data.items():
            data.append({
                '종목코드': code,
                '종목명': info.get('name', ''),
                '현재가': info.get('price', 0),
                '등락률': info.get('change', 0),
                '거래량': info.get('volume', 0),
                '시가총액': info.get('market_cap', 0),
                'PER': info.get('per', 0),
                'PBR': info.get('pbr', 0),
                'ROE': info.get('roe', 0),
                'AI점수': info.get('ai_score', 0)
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('AI점수', ascending=False)
    
    @staticmethod
    def create_comparison_chart(stock_prices: Dict[str, pd.DataFrame], 
                                normalize: bool = True) -> go.Figure:
        """종목 비교 차트
        
        Args:
            stock_prices: {stock_name: price_df}
            normalize: True면 시작점 100 기준 정규화
        """
        fig = go.Figure()
        
        for name, df in stock_prices.items():
            if df.empty:
                continue
            
            if normalize:
                # 시작점 100 기준
                normalized = (df['Close'] / df['Close'].iloc[0]) * 100
                y_data = normalized
                y_label = "수익률 지수"
            else:
                y_data = df['Close']
                y_label = "주가"
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=y_data,
                name=name,
                mode='lines',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="종목 비교",
            xaxis_title="날짜",
            yaxis_title=y_label,
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_radar_chart(stock_scores: Dict[str, Dict[str, float]]) -> go.Figure:
        """레이더 차트 (다차원 비교)
        
        Args:
            stock_scores: {stock_name: {'재무': 80, '기술': 70, ...}}
        """
        fig = go.Figure()
        
        for name, scores in stock_scores.items():
            categories = list(scores.keys())
            values = list(scores.values())
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=500,
            title="종목 다차원 비교"
        )
        
        return fig


class SectorAnalyzer:
    """섹터 분석"""
    
    # KOSPI 200 주요 섹터 분류
    SECTOR_MAP = {
        '005930': '반도체',
        '000660': '반도체',
        '035420': 'IT',
        '035720': 'IT',
        '051910': '화학',
        '005380': '자동차',
        '005490': '자동차',
        '012330': '자동차',
        '000270': '건설',
        '028260': '건설',
        '055550': '금융',
        '086790': '금융',
        '003490': '유통',
        '051900': '유통',
    }
    
    @staticmethod
    def classify_sector(code: str) -> str:
        """종목 섹터 분류"""
        return SectorAnalyzer.SECTOR_MAP.get(code, '기타')
    
    @staticmethod
    def get_sector_performance(kospi_data: pd.DataFrame) -> pd.DataFrame:
        """섹터별 수익률"""
        if kospi_data.empty:
            return pd.DataFrame()
        
        sector_data = {}
        
        for _, row in kospi_data.iterrows():
            code = row['종목코드']
            sector = SectorAnalyzer.classify_sector(code)
            change = row.get('등락률', 0)
            
            if sector not in sector_data:
                sector_data[sector] = []
            
            sector_data[sector].append(change)
        
        # 섹터별 평균 계산
        result = []
        for sector, changes in sector_data.items():
            result.append({
                '섹터': sector,
                '종목수': len(changes),
                '평균등락률': np.mean(changes),
                '상승종목': len([c for c in changes if c > 0]),
                '하락종목': len([c for c in changes if c < 0])
            })
        
        df = pd.DataFrame(result)
        return df.sort_values('평균등락률', ascending=False)
    
    @staticmethod
    def create_sector_heatmap(sector_performance: pd.DataFrame) -> go.Figure:
        """섹터 히트맵"""
        if sector_performance.empty:
            return go.Figure()
        
        # 히트맵용 데이터 준비
        sectors = sector_performance['섹터'].tolist()
        changes = sector_performance['평균등락률'].tolist()
        
        # 색상 맵 (빨강-회색-초록)
        colors = []
        for change in changes:
            if change > 2:
                colors.append('green')
            elif change > 0:
                colors.append('lightgreen')
            elif change > -2:
                colors.append('lightcoral')
            else:
                colors.append('red')
        
        fig = go.Figure(data=[go.Bar(
            x=sectors,
            y=changes,
            marker_color=colors,
            text=[f"{c:.2f}%" for c in changes],
            textposition='auto',
        )])
        
        fig.update_layout(
            title="섹터별 등락률",
            xaxis_title="섹터",
            yaxis_title="평균 등락률 (%)",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_market_heatmap(kospi_data: pd.DataFrame) -> go.Figure:
        """시장 전체 히트맵 (트리맵)"""
        if kospi_data.empty:
            return go.Figure()
        
        # 섹터 정보 추가
        kospi_data = kospi_data.copy()
        kospi_data['섹터'] = kospi_data['종목코드'].apply(SectorAnalyzer.classify_sector)
        
        # 시가총액이 있는 경우만
        if '시가총액' in kospi_data.columns:
            fig = px.treemap(
                kospi_data,
                path=['섹터', '종목명'],
                values='시가총액',
                color='등락률',
                color_continuous_scale=['red', 'yellow', 'green'],
                color_continuous_midpoint=0,
                title="코스피 200 시장 히트맵"
            )
        else:
            # 시가총액 없으면 종목 수 기준
            fig = px.treemap(
                kospi_data,
                path=['섹터', '종목명'],
                color='등락률',
                color_continuous_scale=['red', 'yellow', 'green'],
                color_continuous_midpoint=0,
                title="코스피 200 시장 히트맵"
            )
        
        fig.update_layout(height=600)
        
        return fig
    
    @staticmethod
    def get_top_stocks_by_sector(kospi_data: pd.DataFrame, 
                                 top_n: int = 3) -> Dict[str, pd.DataFrame]:
        """섹터별 상위 종목"""
        if kospi_data.empty:
            return {}
        
        kospi_data = kospi_data.copy()
        kospi_data['섹터'] = kospi_data['종목코드'].apply(SectorAnalyzer.classify_sector)
        
        result = {}
        for sector in kospi_data['섹터'].unique():
            sector_df = kospi_data[kospi_data['섹터'] == sector]
            top_stocks = sector_df.nlargest(top_n, '등락률')
            result[sector] = top_stocks[['종목명', '현재가', '등락률', 'AI점수']]
        
        return result
    
    @staticmethod
    def detect_sector_rotation(historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """섹터 로테이션 감지
        
        Args:
            historical_data: {sector_name: price_df}
        """
        rotation_data = []
        
        for sector, df in historical_data.items():
            if len(df) < 20:
                continue
            
            # 최근 5일 vs 이전 5일 수익률 비교
            recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
            previous_return = (df['Close'].iloc[-5] / df['Close'].iloc[-10] - 1) * 100
            
            momentum = recent_return - previous_return
            
            rotation_data.append({
                '섹터': sector,
                '최근수익률': recent_return,
                '이전수익률': previous_return,
                '모멘텀': momentum,
                '상태': '강세' if momentum > 2 else ('약세' if momentum < -2 else '중립')
            })
        
        df = pd.DataFrame(rotation_data)
        return df.sort_values('모멘텀', ascending=False)
