"""
중장기 투자 추천 시스템 - 재무 및 뉴스 기반 분석
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class LongTermAnalyzer:
    """중장기 투자 추천 분석 엔진"""
    
    def __init__(self, finance_df: pd.DataFrame, price_df: pd.DataFrame):
        """
        Args:
            finance_df: 재무 데이터 (코드, 날짜, PER, PBR, ROE, EPS, BPS 등)
            price_df: 주가 데이터 (코드, 날짜, 종가 등)
        """
        self.finance_df = finance_df
        self.price_df = price_df
    
    def analyze_fundamentals(self, code: str) -> Dict:
        """
        재무 지표 분석
        
        Returns:
            {
                'roe': ROE %,
                'operating_margin': 영업이익률 %,
                'debt_ratio': 부채비율 %,
                'sales_growth': 매출 성장률 %,
                'profit_growth': 이익 성장률 %,
                'per': PER,
                'pbr': PBR,
                'score': 재무 건전성 점수 (0-100)
            }
        """
        try:
            # 해당 종목의 최근 재무 데이터
            finance_data = self.finance_df[self.finance_df['code'] == code]
            
            if finance_data.empty:
                return {'score': 0, 'reason': '재무 데이터 없음'}
            
            # 최신 데이터 3개 정렬
            finance_data = finance_data.sort_values('date', ascending=False).head(3)
            latest = finance_data.iloc[0]
            prev = finance_data.iloc[1] if len(finance_data) > 1 else None
            
            # 컬럼명 정규화 (다양한 형식 지원)
            def get_column_value(row, col_names):
                """여러 가능한 컬럼명 중 첫 번째로 존재하는 값 반환"""
                for col_name in col_names:
                    if col_name in row and pd.notna(row[col_name]):
                        return float(row[col_name])
                return 0
            
            # 기본 지표 (여러 컬럼명 지원)
            result = {
                'roe': get_column_value(latest, ['roe', 'ROE', 'roe(%)', 'ROE(%)']),
                'operating_margin': get_column_value(latest, ['operating_margin', 'op_margin', 'op_margin(%)', '영업이익률']),
                'per': get_column_value(latest, ['per', 'PER', 'per배']),
                'pbr': get_column_value(latest, ['pbr', 'PBR', 'pbr배']),
                'eps': get_column_value(latest, ['eps', 'EPS']),
            }
            
            # PER, PBR이 0이면 기본값 설정 (데이터 부족 시)
            if result['per'] == 0:
                result['per'] = 15  # 평균 PER
            if result['pbr'] == 0:
                result['pbr'] = 1.0  # 평균 PBR
            
            # 성장률 계산
            if prev is not None:
                # EPS 성장률
                eps_latest = get_column_value(latest, ['eps', 'EPS'])
                eps_prev = get_column_value(prev, ['eps', 'EPS'])
                result['profit_growth'] = ((eps_latest - eps_prev) / abs(eps_prev) * 100) if eps_prev != 0 else 0
            else:
                result['profit_growth'] = 0
            
            # 재무 건전성 점수 계산
            score = self._calculate_fundamental_score(result)
            result['score'] = score
            
            return result
            
        except Exception as e:
            logger.error(f"재무 분석 실패 (코드: {code}): {e}")
            return {'score': 0, 'reason': str(e)}
    
    def _calculate_fundamental_score(self, metrics: Dict) -> int:
        """재무 건전성 점수 계산 (0-100)"""
        score = 0
        
        # ROE: 15% 이상 우수 (최대 25점)
        roe = metrics.get('roe', 0)
        if roe >= 15:
            score += min(25, (roe / 20) * 25)
        else:
            score += max(0, (roe / 15) * 25)
        
        # 영업이익률: 10% 이상 우수 (최대 20점)
        op_margin = metrics.get('operating_margin', 0)
        if op_margin >= 10:
            score += min(20, (op_margin / 15) * 20)
        else:
            score += max(0, (op_margin / 10) * 20)
        
        # PER: 10-20 적정 (최대 25점)
        per = metrics.get('per', 20)
        if 10 <= per <= 20:
            score += 25
        elif 5 <= per < 10 or 20 < per <= 30:
            score += 15
        else:
            score += 5
        
        # PBR: 0.5-2.0 적정 (최대 20점)
        pbr = metrics.get('pbr', 1.0)
        if 0.5 <= pbr <= 2.0:
            score += 20
        elif 0.3 <= pbr < 0.5 or 2.0 < pbr <= 3.0:
            score += 10
        else:
            score += 5
        
        # 이익 성장률: 10% 이상 우수 (최대 10점)
        profit_growth = metrics.get('profit_growth', 0)
        if profit_growth >= 10:
            score += min(10, (profit_growth / 20) * 10)
        elif profit_growth > 0:
            score += (profit_growth / 10) * 10
        
        return int(min(100, score))
    
    def analyze_valuation(self, code: str) -> Dict:
        """밸류에이션 분석"""
        try:
            finance_data = self.finance_df[self.finance_df['code'] == code]
            
            if finance_data.empty:
                # 데이터가 없으면 중립적 점수 반환 (50점)
                return {'score': 50, 'per': 15, 'pbr': 1.0}
            
            latest = finance_data.sort_values('date', ascending=False).iloc[0]
            
            # 컬럼명 정규화
            def get_column_value(row, col_names, default=0):
                for col_name in col_names:
                    if col_name in row and pd.notna(row[col_name]):
                        return float(row[col_name])
                return default
            
            per = get_column_value(latest, ['per', 'PER', 'per배'], default=15)
            pbr = get_column_value(latest, ['pbr', 'PBR', 'pbr배'], default=1.0)
            
            # 0 값 처리 (데이터 부족 시 기본값)
            if per == 0:
                per = 15
            if pbr == 0:
                pbr = 1.0
            
            # 밸류에이션 점수 (0-100)
            # PER이 낮을수록, PBR이 낮을수록 좋음
            per_score = max(0, min(50, (20 / per) * 50)) if per > 0 else 25
            pbr_score = max(0, min(50, (1.0 / pbr) * 50)) if pbr > 0 else 25
            
            score = int((per_score + pbr_score) / 2)
            
            return {
                'score': score,
                'per': per,
                'pbr': pbr
            }
        
        except Exception as e:
            logger.error(f"밸류에이션 분석 실패 (코드: {code}): {e}")
            # 오류 시 중립적 점수
            return {'score': 50, 'per': 15, 'pbr': 1.0}
    
    def analyze_momentum(self, code: str, period_days: int = 252) -> Dict:
        """모멘텀 분석 (최근 1년 주가 추세)"""
        try:
            price_data = self.price_df[self.price_df['code'] == code]
            
            if price_data.empty:
                # 데이터가 없으면 중립적인 점수 반환
                return {'score': 50, 'trend': '데이터 없음', 'return_1y': 0}
            
            # 최근 데이터 정렬
            price_data = price_data.sort_values('date', ascending=False)
            
            if len(price_data) < 5:
                # 데이터가 너무 적으면 중립적인 점수 반환
                return {'score': 50, 'trend': '데이터 부족', 'return_1y': 0}
            
            # 최신 종가와 예전 종가 비교 (이용 가능한 데이터 범위 내)
            latest_price = float(price_data.iloc[0]['close'])
            
            # period_days 만큼 거슬러 올라감 (또는 최대한 멀리)
            old_idx = min(period_days - 1, len(price_data) - 1)
            old_price = float(price_data.iloc[old_idx]['close'])
            
            # 수익률 계산
            return_1y = ((latest_price - old_price) / old_price * 100) if old_price > 0 else 0
            
            # 모멘텀 점수 (기본: 50점 중립, 수익률로 조정)
            momentum_score = min(100, max(0, 50 + (return_1y / 2)))
            
            # 추세 판정
            if return_1y > 20:
                trend = '강상승 ↑'
            elif return_1y > 5:
                trend = '상승 ↗'
            elif return_1y > -5:
                trend = '보합 →'
            elif return_1y > -20:
                trend = '하락 ↘'
            else:
                trend = '약세 ↓'
            
            return {
                'score': int(momentum_score),
                'trend': trend,
                'return_1y': return_1y
            }
        
        except Exception as e:
            logger.error(f"모멘텀 분석 실패 (코드: {code}): {e}")
            # 오류 시에도 중절적인 점수 반환
            return {'score': 50, 'trend': '분석 오류', 'return_1y': 0}
    
    def recommend_long_term_stocks(self, 
                                   num_stocks: int = 10,
                                   min_fundamental_score: int = 50,
                                   kospi_list: Dict = None,
                                   weight_fundamental: float = 0.40,
                                   weight_valuation: float = 0.30,
                                   weight_momentum: float = 0.30) -> pd.DataFrame:
        """
        중장기 투자 추천 종목 선정
        
        Args:
            num_stocks: 추천 종목 수
            min_fundamental_score: 최소 재무 건전성 점수
            kospi_list: KOSPI 200 종목 리스트
            weight_fundamental: 재무 건전성 가중치 (기본: 0.40)
            weight_valuation: 밸류에이션 가중치 (기본: 0.30)
            weight_momentum: 모멘텀 가중치 (기본: 0.30)
            
        Returns:
            추천 종목 DataFrame
        """
        recommendations = []
        filtered_count = 0
        error_count = 0
        
        # 분석할 코드 리스트
        if kospi_list:
            codes = list(kospi_list.keys())
        else:
            codes = self.finance_df['code'].unique() if not self.finance_df.empty else []
        
        # 데이터 검증
        if len(codes) == 0:
            logger.warning("분석할 종목이 없습니다")
            return pd.DataFrame()
        
        for code in codes:
            try:
                # 종목명
                if kospi_list and code in kospi_list:
                    name = kospi_list[code]
                else:
                    name = code
                
                # 재무 분석
                fundamental = self.analyze_fundamentals(code)
                
                # 점수가 0인 경우는 데이터 부족으로 스킵 (최소값보다 엄격함)
                if fundamental.get('score', 0) == 0:
                    error_count += 1
                    continue
                
                # 최소 요구 사항 필터링 (좀 더 유연하게)
                if fundamental.get('score', 0) < min_fundamental_score:
                    filtered_count += 1
                    continue
                
                # 밸류에이션 분석
                valuation = self.analyze_valuation(code)
                
                # 모멘텀 분석
                momentum = self.analyze_momentum(code)
                
                # 종합 점수 계산 (사용자 정의 가중치 사용)
                total_score = (
                    fundamental.get('score', 0) * weight_fundamental +
                    valuation.get('score', 0) * weight_valuation +
                    momentum.get('score', 0) * weight_momentum
                )
                
                # 추천 이유 생성
                reasons = []
                if fundamental.get('roe', 0) > 15:
                    reasons.append(f"고ROE ({fundamental.get('roe', 0):.1f}%)")
                if fundamental.get('operating_margin', 0) > 10:
                    reasons.append(f"높은 영업이익률 ({fundamental.get('operating_margin', 0):.1f}%)")
                if valuation.get('per', 20) < 15:
                    reasons.append(f"저PER ({valuation.get('per', 20):.1f}배)")
                if valuation.get('pbr', 1.0) < 1.0:
                    reasons.append(f"저PBR ({valuation.get('pbr', 1.0):.2f}배)")
                if momentum.get('return_1y', 0) > 5:
                    reasons.append(f"긍정적 모멘텀 ({momentum.get('return_1y', 0):.1f}%)")
                
                recommendations.append({
                    'code': code,
                    'name': name,
                    'total_score': total_score,
                    'fundamental_score': fundamental.get('score', 0),
                    'valuation_score': valuation.get('score', 0),
                    'momentum_score': momentum.get('score', 0),
                    'roe': fundamental.get('roe', 0),
                    'operating_margin': fundamental.get('operating_margin', 0),
                    'per': valuation.get('per', 0),
                    'pbr': valuation.get('pbr', 0),
                    'trend': momentum.get('trend', ''),
                    'return_1y': momentum.get('return_1y', 0),
                    'reasons': ' | '.join(reasons) if reasons else '기본 우량주'
                })
            
            except Exception as e:
                logger.debug(f"종목 {code} 분석 중 오류: {e}")
                error_count += 1
                continue
        
        # 분석 결과 로깅
        logger.info(f"분석 대상: {len(codes)}개, 부적격(점수 0): {error_count}개, 필터링됨: {filtered_count}개, 추천: {len(recommendations)}개")
        
        # 종합 점수 기준으로 정렬
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            rec_df = rec_df.sort_values('total_score', ascending=False).head(num_stocks)
            return rec_df
        else:
            return pd.DataFrame()
    
    
    def get_stock_recommendation_details(self, code: str, name: str = None,
                                        weight_fundamental: float = 0.40,
                                        weight_valuation: float = 0.30,
                                        weight_momentum: float = 0.30) -> Dict:
        """종목별 추천 상세 정보
        
        Args:
            code: 종목코드
            name: 종목명
            weight_fundamental: 재무 건전성 가중치
            weight_valuation: 밸류에이션 가중치
            weight_momentum: 모멘텀 가중치
        """
        try:
            fundamental = self.analyze_fundamentals(code)
            valuation = self.analyze_valuation(code)
            momentum = self.analyze_momentum(code)
            
            # 종합 점수 (사용자 정의 가중치 사용)
            total_score = (
                fundamental.get('score', 0) * weight_fundamental +
                valuation.get('score', 0) * weight_valuation +
                momentum.get('score', 0) * weight_momentum
            )
            
            # 추천 레벨 결정
            if total_score >= 80:
                level = '🟢 강력 추천'
            elif total_score >= 70:
                level = '🟡 추천'
            elif total_score >= 50:
                level = '🔵 관심'
            else:
                level = '🟤 재검토'
            
            # 전망 생성
            outlook = []
            
            # 재무 기반 전망
            if fundamental.get('roe', 0) > 15 and fundamental.get('operating_margin', 0) > 10:
                outlook.append("✅ 강한 수익성 - 안정적인 배당 기대")
            if fundamental.get('profit_growth', 0) > 10:
                outlook.append("📈 이익 성장세 - 향후 추가 상승 가능성")
            
            # 밸류에이션 기반 전망
            if valuation.get('per', 20) < 12 and valuation.get('pbr', 1.0) < 0.8:
                outlook.append("💰 저평가 상태 - 상승 여력 큼")
            elif valuation.get('per', 20) > 25:
                outlook.append("⚠️ 고평가 주의 - 조정 가능성")
            
            # 모멘텀 기반 전망
            if momentum.get('return_1y', 0) > 15:
                outlook.append("🚀 강한 상승 추세 - 모멘텀 지속 가능")
            elif momentum.get('return_1y', 0) < -15:
                outlook.append("⬇️ 약세 지속 - 저점 근처일 가능성")
            
            return {
                'code': code,
                'name': name or code,
                'level': level,
                'total_score': round(total_score, 1),
                'fundamental': fundamental,
                'valuation': valuation,
                'momentum': momentum,
                'outlook': outlook if outlook else ['⏳ 추가 분석 필요']
            }
        
        except Exception as e:
            logger.error(f"상세 분석 실패 (코드: {code}): {e}")
            return {'error': str(e)}


def create_investment_portfolio_recommendation(recommendations_df: pd.DataFrame,
                                              total_investment: int = 10_000_000) -> Dict:
    """
    추천 종목들 기반 투자 포트폴리오 구성
    
    Args:
        recommendations_df: 추천 종목 DataFrame
        total_investment: 총 투자 금액 (원)
        
    Returns:
        포트폴리오 배분안
    """
    if recommendations_df.empty:
        return {'error': '추천 종목이 없습니다'}
    
    # 점수 기반 가중치 계산
    score_sum = recommendations_df['total_score'].sum()
    recommendations_df = recommendations_df.copy()
    recommendations_df['weight'] = recommendations_df['total_score'] / score_sum
    recommendations_df['allocation'] = recommendations_df['weight'] * total_investment
    
    # 최소 투자 금액 (100만원)
    recommendations_df['allocation'] = recommendations_df['allocation'].apply(
        lambda x: max(x, 1_000_000) if x >= 500_000 else x
    )
    
    # 재정규화
    allocation_sum = recommendations_df['allocation'].sum()
    if allocation_sum > total_investment * 1.1:
        recommendations_df['allocation'] = (
            recommendations_df['allocation'] / allocation_sum * total_investment
        )
    
    return {
        'portfolio': recommendations_df[['code', 'name', 'total_score', 'allocation']].to_dict('records'),
        'total_allocation': recommendations_df['allocation'].sum(),
        'diversification': len(recommendations_df),
        'strategy': '가치 + 성장 + 안정성 균형 전략'
    }
