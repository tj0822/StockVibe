"""
비교 분석 & 섹터 분석 모듈
- 여러 종목 동시 비교
- 섹터별 분석
- 히트맵
"""
import pandas as pd
import numpy as np
from typing import Dict
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
    
    # 한국거래소 표준 섹터 분류 (KOSPI 200 - 공식 분류 기준)
    SECTOR_MAP = {
        # 반도체
        '005930': '반도체',  # 삼성전자
        '000660': '반도체',  # SK하이닉스
        '042700': '반도체',  # 한미반도체
        '003230': '반도체',  # 삼성SDI
        
        # 디스플레이
        '000810': '디스플레이',  # LG디스플레이
        '034730': '디스플레이',  # SK이노베이션
        
        # 전자/전자부품
        '009150': '전자부품',  # 삼성전기
        '079550': '전자부품',  # LG이노텍
        '066570': '전자부품',  # LG전자
        '042660': '전자부품',  # 한화
        
        # 통신/정보통신
        '030200': '통신/IT',  # KT
        '017670': '통신/IT',  # SK텔레콤
        '032640': '통신/IT',  # LG유플러스
        '035420': '통신/IT',  # NAVER
        '035720': '통신/IT',  # 카카오
        '036570': '통신/IT',  # 엔씨소프트
        '025540': '통신/IT',  # 카카오뱅크
        
        # 자동차 완성차
        '005380': '자동차',  # 현대자동차
        '012330': '자동차',  # 기아
        
        # 자동차부품/타이어
        '005490': '자동차부품',  # 현대모비스
        '086280': '자동차부품',  # 현대글로비스
        '161390': '자동차부품',  # 한국타이어앤테크놀로지
        
        # 조선
        '009540': '조선',  # 현대중공업
        '011200': '조선',  # HMM(현대상선)
        
        # 기계
        '006400': '기계',  # 삼성엔지니어링
        '006390': '기계',  # 동부엔지니어링
        '204320': '기계',  # 현대로템
        
        # 철강
        '005250': '철강',  # GS칼텍스
        
        # 화학
        '011170': '화학',  # 롯데케미칼
        '051910': '화학',  # LG화학
        '010950': '화학',  # S-Oil
        
        # 의약품/바이오
        '000250': '의약품',  # GC녹십자
        '068270': '의약품',  # 셀트리온
        '128940': '의약품',  # 한미약품
        
        # 건설
        '000720': '건설',  # 현대건설
        '006360': '건설',  # GS건설
        '028260': '건설',  # 삼성물산
        '001940': '건설',  # SK C&C
        '047040': '건설',  # 대우건설
        
        # 비금속 광물
        '010130': '비금속',  # 고려아연
        
        # 운송
        '003490': '운송',  # 대한항공
        '000270': '운송',  # CJ대한통운
        '071050': '운송',  # 한진칼
        
        # 유통/소매
        '005680': '유통',  # 롯데쇼핑
        '001800': '유통',  # 오리온
        
        # 금융
        '055550': '금융',  # KB금융
        '055000': '금융',  # 신한금융
        '086790': '금융',  # 우리금융
        '086000': '금융',  # 하나금융
        '003520': '금융',  # 신한투자증권
        '078930': '금융',  # BNK금융
        '001430': '금융',  # SK증권
        
        # 보험
        '032830': '보험',  # 삼성생명
        
        # 음식료
        '097950': '음식료',  # CJ제일제당
        '010780': '음식료',  # 아이에스동서
        '001120': '음식료',  # LF
        
        # 섬유
        '111770': '섬유',  # 영원무역
        
        # 에너지/전력
        '010060': '에너지',  # OCI
        '051600': '에너지',  # 한국전력
        
        # 부동산
        '001440': '부동산',  # SK네트웍스
        
        # 미디어/엔터테인먼트
        '352820': '미디어',  # 하이브
        '009410': '미디어',  # YG엔터테인먼트
    }
    
    @staticmethod
    def classify_sector(code: str) -> str:
        """종목 섹터 분류 (한국거래소 표준)"""
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
            # 종목코드와 종목명을 함께 표시하는 컬럼 선택
            display_cols = ['종목코드', '종목명', '현재가', '등락률']
            # AI점수가 있으면 추가
            if 'AI점수' in top_stocks.columns:
                display_cols.append('AI점수')
            result[sector] = top_stocks[display_cols]
        
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
