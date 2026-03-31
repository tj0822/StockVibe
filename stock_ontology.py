"""주식 온톨로지 및 예측 모듈"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from technical_indicators import analyze_technical_indicators, get_technical_score


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

    def build_ontology_df(
        self,
        stock_master_df: pd.DataFrame,
        sector_rank_df: pd.DataFrame,
        regime,
    ) -> pd.DataFrame:
        """시장/섹터/종목/전략/시그널을 연결한 통합 온톨로지 프레임을 생성한다."""
        if stock_master_df is None or stock_master_df.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "stock",
                    "sector",
                    "sector_power",
                    "final_score_adjusted",
                    "momentum_score",
                    "signal",
                    "strategy",
                    "market_regime",
                ]
            )

        work = stock_master_df.copy()

        stock_col = "name" if "name" in work.columns else ("stock" if "stock" in work.columns else None)
        if stock_col is None:
            work["stock"] = work.get("code", "")
            stock_col = "stock"

        if "sector" not in work.columns:
            work["sector"] = "Unknown"

        # sector_power가 없으면 sector_rank 최신값으로 0~100 스케일 근사치 생성
        if "sector_power" not in work.columns:
            sector_power_map = {}
            if sector_rank_df is not None and not sector_rank_df.empty and {"sector", "rank"}.issubset(set(sector_rank_df.columns)):
                rank_work = sector_rank_df.copy()
                if "date" in rank_work.columns:
                    rank_work["date"] = pd.to_datetime(rank_work["date"], errors="coerce")
                    rank_work = rank_work.sort_values("date")
                    latest_date = rank_work["date"].dropna().max()
                    if pd.notna(latest_date):
                        rank_work = rank_work[rank_work["date"] == latest_date]
                rank_work = rank_work.dropna(subset=["sector", "rank"]).copy()
                if not rank_work.empty:
                    rank_work["rank"] = pd.to_numeric(rank_work["rank"], errors="coerce")
                    rank_work = rank_work.dropna(subset=["rank"])
                    max_rank = float(rank_work["rank"].max()) if not rank_work.empty else 1.0
                    if max_rank <= 1:
                        rank_work["sector_power"] = 100.0
                    else:
                        rank_work["sector_power"] = 100.0 * (1.0 - (rank_work["rank"] - 1.0) / (max_rank - 1.0))
                    sector_power_map = dict(zip(rank_work["sector"], rank_work["sector_power"]))
            work["sector_power"] = work["sector"].map(sector_power_map).fillna(50.0)

        work["final_score_adjusted"] = pd.to_numeric(work.get("final_score_adjusted", 0), errors="coerce").fillna(0.0)
        work["momentum_score"] = pd.to_numeric(work.get("momentum_score", 0), errors="coerce").fillna(0.0)
        work["signal"] = work.get("signal", "").fillna("").astype(str)

        strategy_col = "triggered_strategies" if "triggered_strategies" in work.columns else "strategy"
        work["strategy"] = work.get(strategy_col, "").fillna("").astype(str)

        if "date" in work.columns:
            work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
            work["date"] = work["date"].fillna(pd.Timestamp.today().normalize())
        else:
            work["date"] = pd.Timestamp.today().normalize()

        market_regime = "UNKNOWN"
        if isinstance(regime, dict):
            market_regime = str(
                regime.get("market_regime")
                or regime.get("primary_regime")
                or regime.get("regime")
                or "UNKNOWN"
            )
        elif regime is not None:
            market_regime = str(regime)

        ontology_df = pd.DataFrame(
            {
                "date": work["date"],
                "stock": work[stock_col],
                "sector": work["sector"],
                "sector_power": pd.to_numeric(work["sector_power"], errors="coerce").fillna(0.0),
                "final_score_adjusted": work["final_score_adjusted"],
                "momentum_score": work["momentum_score"],
                "signal": work["signal"],
                "strategy": work["strategy"],
                "market_regime": market_regime,
            }
        )

        return ontology_df

    def compute_priority_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """우선순위 점수를 계산하여 priority 컬럼을 추가한다."""
        if df is None or df.empty:
            return df

        work = df.copy()
        work["sector_power"] = pd.to_numeric(work.get("sector_power", 0), errors="coerce").fillna(0.0)
        work["final_score_adjusted"] = pd.to_numeric(work.get("final_score_adjusted", 0), errors="coerce").fillna(0.0)
        work["momentum_score"] = pd.to_numeric(work.get("momentum_score", 0), errors="coerce").fillna(0.0)

        if "signal_strength" in work.columns:
            signal_strength = pd.to_numeric(work["signal_strength"], errors="coerce").fillna(0.0)
        else:
            signal_series = work.get("signal", "").fillna("").astype(str).str.upper()
            signal_strength = signal_series.map({"BUY": 100.0, "SELL": -100.0}).fillna(0.0)

        work["signal_strength"] = signal_strength

        # 뉴스 임팩트 점수 (없으면 0)
        work["news_impact_score"] = pd.to_numeric(work.get("news_impact_score", 0), errors="coerce").fillna(0.0)

        work["priority"] = (
            0.3 * work["sector_power"]
            + 0.3 * work["final_score_adjusted"]
            + 0.2 * work["momentum_score"]
            + 0.2 * work["signal_strength"]
            + 0.15 * work["news_impact_score"]
        )
        return work

    def build_news_impact_map(self, stock_news_map: Dict[str, List[Dict]], stock_sector_map: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
        """종목별 뉴스 데이터를 기반으로 임팩트 점수와 근거를 생성한다."""
        if not stock_news_map:
            return {}

        impact_map: Dict[str, Dict] = {}

        for stock_code, news_items in stock_news_map.items():
            if not news_items:
                impact_map[str(stock_code)] = {
                    "news_impact_score": 0.0,
                    "news_positive_count": 0,
                    "news_negative_count": 0,
                    "news_neutral_count": 0,
                    "dominant_news_event": "NONE",
                    "news_impact_reason": "관련 뉴스 없음",
                }
                continue

            weighted_score = 0.0
            pos_count = 0
            neg_count = 0
            neu_count = 0
            reasons: List[str] = []
            event_count: Dict[str, int] = {"M&A": 0, "ORDER": 0, "REGULATION": 0, "EARNINGS": 0, "GENERAL": 0}

            sector_text = ""
            if stock_sector_map is not None:
                sector_text = str(stock_sector_map.get(str(stock_code).zfill(6), "") or "")
            if not sector_text:
                sector_text = str(news_items[0].get("sector", "") or "")

            for idx, item in enumerate(news_items[:8]):
                title = str(item.get("title", "") or "")
                body = str(item.get("body", "") or "")
                text = f"{title} {body}".strip()
                if not text:
                    continue

                sentiment = simple_sentiment_analysis(text)
                recency_weight = max(0.4, 1.0 - (idx * 0.1))
                event_type = self._detect_event_type(text)
                event_count[event_type] = event_count.get(event_type, 0) + 1

                event_weight = self._get_event_weight(event_type)
                sector_weight = self._get_sector_weight(sector_text, event_type)
                unit_weight = 10.0 * recency_weight * event_weight * sector_weight

                if sentiment == "positive":
                    weighted_score += unit_weight
                    pos_count += 1
                    reasons.append(f"+[{event_type}] {title[:24]}")
                elif sentiment == "negative":
                    weighted_score -= unit_weight
                    neg_count += 1
                    reasons.append(f"-[{event_type}] {title[:24]}")
                else:
                    neu_count += 1

            news_impact_score = float(np.clip(weighted_score, -100.0, 100.0))

            if reasons:
                impact_reason = " | ".join(reasons[:3])
            else:
                impact_reason = "중립/유효 뉴스 중심"

            dominant_event = max(event_count.items(), key=lambda x: x[1])[0] if event_count else "GENERAL"

            impact_map[str(stock_code)] = {
                "news_impact_score": news_impact_score,
                "news_positive_count": pos_count,
                "news_negative_count": neg_count,
                "news_neutral_count": neu_count,
                "dominant_news_event": dominant_event,
                "news_impact_reason": impact_reason,
            }

        return impact_map

    def _detect_event_type(self, text: str) -> str:
        """뉴스 텍스트에서 이벤트 타입 분류"""
        t = str(text or "")
        event_keywords = {
            "M&A": ["인수", "합병", "m&a", "지분", "매각", "스핀오프"],
            "ORDER": ["수주", "계약", "공급", "납품", "수출", "발주"],
            "REGULATION": ["규제", "승인", "허가", "제재", "정책", "공시", "법안"],
            "EARNINGS": ["실적", "매출", "영업이익", "순이익", "어닝", "가이던스", "컨센서스"],
        }
        for event_type, keywords in event_keywords.items():
            if any(k.lower() in t.lower() for k in keywords):
                return event_type
        return "GENERAL"

    def _get_event_weight(self, event_type: str) -> float:
        """이벤트 타입별 임팩트 가중치"""
        weights = {
            "M&A": 1.25,
            "ORDER": 1.20,
            "REGULATION": 1.15,
            "EARNINGS": 1.30,
            "GENERAL": 1.00,
        }
        return float(weights.get(str(event_type), 1.00))

    def _get_sector_weight(self, sector: str, event_type: str) -> float:
        """섹터/이벤트 조합별 가중치"""
        s = str(sector or "").lower()
        e = str(event_type or "GENERAL")

        # 기본값
        w = 1.0

        # 섹터 민감도 룰
        if any(k in s for k in ["반도체", "it", "소프트웨어", "통신"]):
            if e == "EARNINGS":
                w = 1.20
            elif e == "ORDER":
                w = 1.15
            elif e == "REGULATION":
                w = 1.10
        elif any(k in s for k in ["바이오", "제약", "헬스", "의료"]):
            if e == "REGULATION":
                w = 1.35
            elif e == "M&A":
                w = 1.10
            elif e == "EARNINGS":
                w = 1.10
        elif any(k in s for k in ["건설", "조선", "기계", "방산"]):
            if e == "ORDER":
                w = 1.35
            elif e == "M&A":
                w = 1.10
        elif any(k in s for k in ["금융", "은행", "증권", "보험"]):
            if e == "REGULATION":
                w = 1.25
            elif e == "EARNINGS":
                w = 1.15
        elif any(k in s for k in ["자동차", "배터리", "2차전지", "화학"]):
            if e == "EARNINGS":
                w = 1.20
            elif e == "REGULATION":
                w = 1.10

        return float(w)

    def attach_news_impact(self, df: pd.DataFrame, stock_news_map: Dict[str, List[Dict]]) -> pd.DataFrame:
        """온톨로지 프레임에 뉴스 임팩트 점수/근거를 결합한다."""
        if df is None or df.empty:
            return df

        work = df.copy()
        if "code" not in work.columns:
            return work

        work["code"] = work["code"].astype(str).str.zfill(6)
        stock_sector_map = {}
        if "sector" in work.columns:
            stock_sector_map = dict(
                zip(
                    work["code"].astype(str).str.zfill(6),
                    work["sector"].astype(str),
                )
            )

        impact_map = self.build_news_impact_map(stock_news_map, stock_sector_map=stock_sector_map)
        if not impact_map:
            work["news_impact_score"] = 0.0
            work["news_impact_reason"] = "관련 뉴스 없음"
            work["news_positive_count"] = 0
            work["news_negative_count"] = 0
            return work

        impact_rows = []
        for stock_code, values in impact_map.items():
            row = {"code": str(stock_code).zfill(6)}
            row.update(values)
            impact_rows.append(row)

        impact_df = pd.DataFrame(impact_rows)
        work = work.merge(impact_df, on="code", how="left")
        work["news_impact_score"] = pd.to_numeric(work.get("news_impact_score", 0), errors="coerce").fillna(0.0)
        work["news_impact_reason"] = work.get("news_impact_reason", "관련 뉴스 없음").fillna("관련 뉴스 없음")
        work["news_positive_count"] = pd.to_numeric(work.get("news_positive_count", 0), errors="coerce").fillna(0).astype(int)
        work["news_negative_count"] = pd.to_numeric(work.get("news_negative_count", 0), errors="coerce").fillna(0).astype(int)
        work["dominant_news_event"] = work.get("dominant_news_event", "NONE").fillna("NONE")
        return work

    def infer_buy_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """조건 기반 매수 후보를 추론하고 priority 상위 종목을 반환한다."""
        if df is None or df.empty:
            return pd.DataFrame()

        work = self.compute_priority_score(df)
        filtered = work[
            (work.get("signal", "").astype(str).str.upper() == "BUY")
            & (pd.to_numeric(work.get("sector_power", 0), errors="coerce").fillna(0.0) > 60)
            & (pd.to_numeric(work.get("final_score_adjusted", 0), errors="coerce").fillna(0.0) > 60)
        ].copy()

        if filtered.empty:
            return filtered

        return filtered.sort_values("priority", ascending=False).head(20).reset_index(drop=True)

    def build_sector_opportunity(self, df: pd.DataFrame) -> pd.DataFrame:
        """섹터 단위 기회 지표(avg_priority, avg_return, sector_power)를 생성한다."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["sector", "avg_priority", "avg_return", "sector_power"])

        work = self.compute_priority_score(df)

        if "avg_return" in work.columns:
            return_col = "avg_return"
        elif "return_1m" in work.columns:
            return_col = "return_1m"
        elif "return" in work.columns:
            return_col = "return"
        else:
            work["_tmp_return"] = 0.0
            return_col = "_tmp_return"

        sector_df = (
            work.groupby("sector", dropna=False)
            .agg(
                avg_priority=("priority", "mean"),
                avg_return=(return_col, "mean"),
                sector_power=("sector_power", "mean"),
            )
            .reset_index()
            .sort_values("avg_priority", ascending=False)
            .reset_index(drop=True)
        )
        return sector_df
    
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
    
    def analyze_technical_indicators(self, price_df: pd.DataFrame, code: str) -> Dict:
        """기술적 지표 분석"""
        try:
            # 해당 종목 데이터 추출
            stock_data = price_df[price_df['code'] == code].copy()
            if stock_data.empty:
                return {}
            
            # 날짜순 정렬
            stock_data = stock_data.sort_values('date')
            
            # 기술 지표 분석
            tech_analysis = analyze_technical_indicators(stock_data)
            
            return tech_analysis
        except Exception as e:
            return {}
    
    def generate_prediction(self, price_df: pd.DataFrame = None, code: str = None) -> Dict:
        """종합 예측 생성"""
        correlations = self.find_correlations()
        price_patterns = self.analyze_price_patterns()
        finance_trends = self.analyze_finance_trends()
        
        # 기술적 지표 분석 추가
        technical_analysis = {}
        technical_score = 0
        if price_df is not None and code:
            technical_analysis = self.analyze_technical_indicators(price_df, code)
            if technical_analysis:
                technical_score = get_technical_score(technical_analysis)
        
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
        
        # 4. 기술적 지표 (가중치: 20%) - NEW!
        if technical_analysis and technical_score != 0:
            prediction_score += technical_score * 0.2  # 20% 가중치
            
            tech_factors = []
            if technical_analysis.get('golden_cross'):
                tech_factors.append("골든크로스 발생")
            if technical_analysis.get('dead_cross'):
                tech_factors.append("데드크로스 발생")
            if 'rsi_signal' in technical_analysis and technical_analysis['rsi_signal'] != '중립':
                tech_factors.append(f"RSI {technical_analysis['rsi_signal']}")
            if 'macd_trend' in technical_analysis:
                tech_factors.append(f"MACD {technical_analysis['macd_trend']}")
            
            if tech_factors:
                factors.append(f"기술적: {', '.join(tech_factors[:2])}")
                
                detailed_reasons.append({
                    'category': '기술적 지표',
                    'impact': 'positive' if technical_score > 0 else 'negative',
                    'score': technical_score * 0.2,
                    'description': ' | '.join(tech_factors)
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
            'negative_news': negative_news_list,  # 부정 뉴스 리스트            'technical_analysis': technical_analysis,  # 기술적 지표 (NEW!)            'correlations': correlations,
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
