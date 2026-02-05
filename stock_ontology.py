"""주식 온톨로지 및 예측 모듈"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


def simple_sentiment_analysis(text: str) -> str:
    """간단한 키워드 기반 감성분석"""
    positive_keywords = [
        '상승', '증가', '호재', '성장', '확대', '개선', '긍정', '상향', 
        '흑자', '최고', '신고가', '돌파', '급등', '호조', '수주', '계약',
        '투자', '확장', '인수', '혁신', '개발', '출시', '성공'
    ]
    
    negative_keywords = [
        '하락', '감소', '악재', '부진', '축소', '악화', '부정', '하향',
        '적자', '최저', '급락', '불안', '리스크', '위험', '손실', '파산',
        '부도', '감원', '철수', '중단', '실패', '지연', '취소'
    ]
    
    positive_count = sum(1 for keyword in positive_keywords if keyword in text)
    negative_count = sum(1 for keyword in negative_keywords if keyword in text)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'


@dataclass
class NewsEvent:
    """뉴스 이벤트"""
    date: datetime
    title: str
    content: str
    sentiment: str
    source: str


@dataclass
class PriceEvent:
    """주가 이벤트"""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    change_rate: float


@dataclass
class FinanceEvent:
    """재무 이벤트"""
    date: datetime
    per: Optional[float]
    pbr: Optional[float]
    eps: Optional[float]
    bps: Optional[float]
    dvr: Optional[float]


class StockOntology:
    """주식 온톨로지 - 뉴스, 주가, 재무정보 간의 관계 분석"""
    
    def __init__(self):
        self.news_events: List[NewsEvent] = []
        self.price_events: List[PriceEvent] = []
        self.finance_events: List[FinanceEvent] = []
        self.relationships: List[Dict] = []
    
    def add_news_event(self, event: NewsEvent):
        """뉴스 이벤트 추가"""
        self.news_events.append(event)
    
    def add_price_event(self, event: PriceEvent):
        """주가 이벤트 추가"""
        self.price_events.append(event)
    
    def add_finance_event(self, event: FinanceEvent):
        """재무 이벤트 추가"""
        self.finance_events.append(event)
    
    def find_correlations(self) -> List[Dict]:
        """뉴스-주가 상관관계 분석"""
        correlations = []
        
        for news in self.news_events:
            # 뉴스 발생 후 5일간의 주가 변화 추적
            news_date = news.date
            future_prices = [
                p for p in self.price_events 
                if news_date < p.date <= news_date + timedelta(days=5)
            ]
            
            if future_prices:
                # 평균 수익률 계산
                avg_return = np.mean([p.change_rate for p in future_prices])
                
                # 뉴스 감성과 주가 변동의 일치도
                sentiment_match = (
                    (news.sentiment == 'positive' and avg_return > 0) or
                    (news.sentiment == 'negative' and avg_return < 0) or
                    (news.sentiment == 'neutral')
                )
                
                correlations.append({
                    'news_date': news_date,
                    'news_title': news.title,
                    'sentiment': news.sentiment,
                    'avg_return_5d': avg_return,
                    'sentiment_match': sentiment_match,
                    'confidence': min(abs(avg_return) / 5, 1.0)  # 0-1 사이로 정규화
                })
        
        return correlations
    
    def analyze_price_patterns(self) -> Dict:
        """주가 패턴 분석"""
        if not self.price_events:
            return {}
        
        prices = [p.close for p in self.price_events]
        volumes = [p.volume for p in self.price_events]
        
        # 최근 추세
        recent_prices = prices[-20:] if len(prices) >= 20 else prices
        trend = 'uptrend' if recent_prices[-1] > recent_prices[0] else 'downtrend'
        
        # 변동성
        returns = [self.price_events[i].change_rate for i in range(len(self.price_events))]
        volatility = np.std(returns) if returns else 0
        
        # 거래량 추세
        recent_volumes = volumes[-20:] if len(volumes) >= 20 else volumes
        avg_volume = np.mean(volumes) if volumes else 0
        recent_avg_volume = np.mean(recent_volumes) if recent_volumes else 0
        volume_trend = 'increasing' if recent_avg_volume > avg_volume * 1.2 else 'stable'
        
        return {
            'trend': trend,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'avg_volume': avg_volume,
            'recent_avg_volume': recent_avg_volume
        }
    
    def analyze_finance_trends(self) -> Dict:
        """재무지표 추세 분석"""
        if len(self.finance_events) < 2:
            return {}
        
        # 최신 2개 재무 데이터 비교
        latest = self.finance_events[-1]
        previous = self.finance_events[-2]
        
        trends = {}
        
        if latest.per and previous.per:
            trends['per_trend'] = 'improving' if latest.per < previous.per else 'worsening'
            trends['per_change'] = ((latest.per - previous.per) / previous.per * 100)
        
        if latest.pbr and previous.pbr:
            trends['pbr_trend'] = 'improving' if latest.pbr < previous.pbr else 'worsening'
            trends['pbr_change'] = ((latest.pbr - previous.pbr) / previous.pbr * 100)
        
        if latest.eps and previous.eps:
            trends['eps_trend'] = 'improving' if latest.eps > previous.eps else 'worsening'
            trends['eps_change'] = ((latest.eps - previous.eps) / previous.eps * 100)
        
        if latest.dvr and previous.dvr:
            trends['dividend_trend'] = 'improving' if latest.dvr > previous.dvr else 'stable'
        
        return trends
    
    def generate_prediction(self) -> Dict:
        """종합 예측 생성"""
        correlations = self.find_correlations()
        price_patterns = self.analyze_price_patterns()
        finance_trends = self.analyze_finance_trends()
        
        # 예측 점수 계산 (간단한 규칙 기반)
        prediction_score = 0
        factors = []
        detailed_reasons = []  # 상세 근거
        
        # 1. 뉴스 감성 분석 (가중치: 30%)
        positive_news_list = []
        negative_news_list = []
        
        if correlations:
            positive_news = [c for c in correlations if c['sentiment'] == 'positive']
            negative_news = [c for c in correlations if c['sentiment'] == 'negative']
            
            news_score = (len(positive_news) - len(negative_news)) / len(correlations) * 30
            prediction_score += news_score
            
            if len(positive_news) > len(negative_news):
                factors.append(f"긍정 뉴스 {len(positive_news)}건 우세")
                detailed_reasons.append({
                    'category': '뉴스 분석',
                    'impact': 'positive',
                    'score': news_score,
                    'description': f"긍정 뉴스 {len(positive_news)}건, 부정 뉴스 {len(negative_news)}건",
                    'news_list': positive_news[:3]  # 상위 3개
                })
                positive_news_list = positive_news
            elif len(negative_news) > len(positive_news):
                factors.append(f"부정 뉴스 {len(negative_news)}건 우세")
                detailed_reasons.append({
                    'category': '뉴스 분석',
                    'impact': 'negative',
                    'score': news_score,
                    'description': f"부정 뉴스 {len(negative_news)}건, 긍정 뉴스 {len(positive_news)}건",
                    'news_list': negative_news[:3]
                })
                negative_news_list = negative_news
        
        # 2. 주가 추세 (가중치: 40%)
        if price_patterns:
            trend = price_patterns.get('trend')
            volume_trend = price_patterns.get('volume_trend')
            volatility = price_patterns.get('volatility', 0)
            
            if trend == 'uptrend':
                prediction_score += 20
                factors.append("상승 추세 지속")
                detailed_reasons.append({
                    'category': '주가 추세',
                    'impact': 'positive',
                    'score': 20,
                    'description': f"최근 20일간 상승 추세 확인 (변동성: {volatility:.2f}%)"
                })
            else:
                prediction_score -= 20
                factors.append("하락 추세")
                detailed_reasons.append({
                    'category': '주가 추세',
                    'impact': 'negative',
                    'score': -20,
                    'description': f"최근 20일간 하락 추세 (변동성: {volatility:.2f}%)"
                })
            
            if volume_trend == 'increasing':
                prediction_score += 20
                factors.append("거래량 증가")
                avg_vol = price_patterns.get('avg_volume', 0)
                recent_avg_vol = price_patterns.get('recent_avg_volume', 0)
                vol_increase = ((recent_avg_vol - avg_vol) / avg_vol * 100) if avg_vol > 0 else 0
                detailed_reasons.append({
                    'category': '거래량',
                    'impact': 'positive',
                    'score': 20,
                    'description': f"최근 거래량 {vol_increase:.1f}% 증가 (평균 {recent_avg_vol:,.0f})"
                })
        
        # 3. 재무 추세 (가중치: 30%)
        if finance_trends:
            improving_items = []
            worsening_items = []
            
            for key, value in finance_trends.items():
                if 'improving' in str(value):
                    improving_items.append(key.replace('_trend', '').upper())
                elif 'worsening' in str(value):
                    worsening_items.append(key.replace('_trend', '').upper())
            
            improving_count = len(improving_items)
            worsening_count = len(worsening_items)
            
            finance_score = (improving_count - worsening_count) * 10
            prediction_score += finance_score
            
            if improving_count > 0:
                factors.append(f"재무지표 {improving_count}개 개선")
                change_info = []
                for key, value in finance_trends.items():
                    if 'change' in key and key.replace('_change', '_trend') in [k for k, v in finance_trends.items() if 'improving' in str(v)]:
                        indicator = key.replace('_change', '').upper()
                        change_info.append(f"{indicator} {value:+.1f}%")
                
                detailed_reasons.append({
                    'category': '재무 개선',
                    'impact': 'positive',
                    'score': improving_count * 10,
                    'description': f"{', '.join(improving_items)} 개선" + (f" ({', '.join(change_info)})" if change_info else "")
                })
            
            if worsening_count > 0:
                factors.append(f"재무지표 {worsening_count}개 악화")
                change_info = []
                for key, value in finance_trends.items():
                    if 'change' in key and key.replace('_change', '_trend') in [k for k, v in finance_trends.items() if 'worsening' in str(v)]:
                        indicator = key.replace('_change', '').upper()
                        change_info.append(f"{indicator} {value:+.1f}%")
                
                detailed_reasons.append({
                    'category': '재무 악화',
                    'impact': 'negative',
                    'score': worsening_count * -10,
                    'description': f"{', '.join(worsening_items)} 악화" + (f" ({', '.join(change_info)})" if change_info else "")
                })
        
        # 예측 방향 결정
        if prediction_score > 20:
            direction = 'strong_buy'
            direction_text = '강력 매수'
            confidence = min(abs(prediction_score), 100) / 100
        elif prediction_score > 0:
            direction = 'buy'
            direction_text = '매수'
            confidence = min(abs(prediction_score), 100) / 100
        elif prediction_score < -20:
            direction = 'strong_sell'
            direction_text = '강력 매도'
            confidence = min(abs(prediction_score), 100) / 100
        elif prediction_score < 0:
            direction = 'sell'
            direction_text = '매도'
            confidence = min(abs(prediction_score), 100) / 100
        else:
            direction = 'hold'
            direction_text = '보유'
            confidence = 0.5
        
        return {
            'direction': direction,
            'direction_text': direction_text,
            'score': prediction_score,
            'confidence': confidence,
            'factors': factors,
            'detailed_reasons': detailed_reasons,  # 상세 근거 추가
            'positive_news': positive_news_list,  # 긍정 뉴스 리스트
            'negative_news': negative_news_list,  # 부정 뉴스 리스트
            'correlations': correlations,
            'price_patterns': price_patterns,
            'finance_trends': finance_trends
        }


def build_stock_ontology(
    code: str,
    price_df: pd.DataFrame,
    news_data: List[Dict],
    finance_df: pd.DataFrame,
    sentiment_analyzer=None
) -> StockOntology:
    """주식 온톨로지 구축"""
    ontology = StockOntology()
    
    # 주가 데이터 추가
    stock_prices = price_df[price_df['code'] == code].copy()
    stock_prices = stock_prices.sort_values('date')
    
    for _, row in stock_prices.iterrows():
        event = PriceEvent(
            date=pd.to_datetime(row['date']),
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            change_rate=((row['close'] - row['open']) / row['open'] * 100) if row['open'] > 0 else 0
        )
        ontology.add_price_event(event)
    
    # 뉴스 데이터 추가
    for news in news_data:
        # 감성분석 - sentiment_analyzer가 있으면 사용, 없으면 간단한 키워드 방식
        sentiment = 'neutral'
        text = news.get('title', '') + ' ' + news.get('body', '')
        
        if sentiment_analyzer:
            try:
                analysis = sentiment_analyzer.analyze(text)
                sentiment = analysis.get('sentiment', 'neutral')
            except:
                sentiment = simple_sentiment_analysis(text)
        else:
            sentiment = simple_sentiment_analysis(text)
        
        # 날짜 파싱
        try:
            if isinstance(news.get('date'), str):
                # "YYYY.MM.DD" 형식 또는 다른 형식 처리
                date_str = news.get('date', '')
                if '.' in date_str:
                    date_obj = pd.to_datetime(date_str, format='%Y.%m.%d')
                else:
                    date_obj = pd.to_datetime(date_str)
            else:
                date_obj = pd.to_datetime(news.get('date', datetime.now()))
        except:
            date_obj = datetime.now()
        
        event = NewsEvent(
            date=date_obj,
            title=news.get('title', ''),
            content=news.get('body', ''),
            sentiment=sentiment,
            source=news.get('source', '')
        )
        ontology.add_news_event(event)
    
    # 재무 데이터 추가
    if not finance_df.empty:
        stock_finance = finance_df[finance_df['code'] == code].copy()
        stock_finance = stock_finance.sort_values('date')
        
        for _, row in stock_finance.iterrows():
            event = FinanceEvent(
                date=pd.to_datetime(row['date']),
                per=row.get('per'),
                pbr=row.get('pbr'),
                eps=row.get('eps'),
                bps=row.get('bps'),
                dvr=row.get('dvr')
            )
            ontology.add_finance_event(event)
    
    return ontology
