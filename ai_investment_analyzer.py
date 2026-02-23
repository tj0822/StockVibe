"""
OpenAI API를 이용한 고성능 AI 투자 분석 모듈
토큰 절약을 위해 분석 결과를 캐싱합니다.
"""

import json
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
import traceback


class AIInvestmentAnalyzer:
    """OpenAI API를 이용한 투자 분석"""
    
    def __init__(self, api_key: str = None):
        """
        초기화
        
        Args:
            api_key: OpenAI API 키 (None이면 환경변수에서 로드)
        """
        # openai 패키지 lazy loading
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            raise ImportError(
                "openai 패키지가 설치되지 않았습니다. "
                "다음 명령어로 설치해주세요: pip install openai"
            )
        
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API 키가 없습니다. OPENAI_API_KEY 환경변수를 설정하세요.")
        
        # OpenAI 클라이언트 초기화
        self.client = self.OpenAI(api_key=api_key)
        self.cache_dir = Path("data/ai_analysis_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_days = 7  # 캐시 유효 기간 (7일)
    
    def _generate_cache_key(self, stock_code: str, stock_name: str, analysis_type: str, 
                            financial_data: dict = None) -> str:
        """
        분석 데이터를 기반으로 캐시 키 생성
        
        Args:
            stock_code: 종목 코드
            stock_name: 종목명
            analysis_type: 분석 유형 (fundamental, technical, sentiment, recommendation)
            financial_data: 재무 데이터
        
        Returns:
            캐시 키 (MD5 해시)
        """
        cache_input = f"{stock_code}_{stock_name}_{analysis_type}"
        
        if financial_data:
            # 재무 데이터의 주요 값들을 포함하여 캐시 키 생성
            financial_str = json.dumps(financial_data, sort_keys=True, default=str)
            cache_input += f"_{financial_str}"
        
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_cached_analysis(self, cache_key: str) -> dict:
        """
        캐시에서 분석 결과 로드
        
        Args:
            cache_key: 캐시 키
        
        Returns:
            캐시된 분석 결과 또는 None
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 캐시 만료 확인
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(days=self.cache_expiry_days):
                cache_file.unlink()  # 만료된 캐시 삭제
                return None
            
            return cache_data['analysis']
        
        except Exception as e:
            print(f"⚠️ 캐시 로드 실패: {e}")
            return None
    
    def _save_cache(self, cache_key: str, analysis: dict):
        """
        분석 결과를 캐시에 저장
        
        Args:
            cache_key: 캐시 키
            analysis: 분석 결과
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"⚠️ 캐시 저장 실패: {e}")
    
    def analyze_fundamental(self, stock_code: str, stock_name: str, 
                           financial_data: dict) -> dict:
        """
        재무 기초 분석
        
        Args:
            stock_code: 종목 코드
            stock_name: 종목명
            financial_data: 재무 데이터 (PER, PBR, ROE, 영업이익률 등)
        
        Returns:
            분석 결과 (평가, 근거, 점수)
        """
        cache_key = self._generate_cache_key(stock_code, stock_name, "fundamental", financial_data)
        
        # 캐시 확인
        cached = self._get_cached_analysis(cache_key)
        if cached:
            st.info(f"💾 캐시된 분석 결과 사용 (종목: {stock_name})")
            cached['from_cache'] = True  # 캐시 플래그 추가
            return cached
        
        try:
            prompt = f"""
            다음 재무 데이터를 분석하여 투자 관점에서 평가해주세요.
            
            종목: {stock_name} ({stock_code})
            
            재무 데이터:
            {json.dumps(financial_data, ensure_ascii=False, indent=2)}
            
            분석 항목:
            1. 현재 밸류에이션 평가 (저평가/적정가/고평가)
            2. 기업 기본 체질 평가
            3. 수익성 분석
            4. 배당 매력도
            5. 장기 투자 적합성
            
            응답 형식:
            {{
                "valuation": "저평가/적정가/고평가",
                "rating": "강력추천/추천/중립/약한매도/강한매도",
                "score": 0-100,
                "strengths": ["장점1", "장점2"],
                "weaknesses": ["단점1", "단점2"],
                "conclusion": "종합 판단",
                "token_saving_note": "캐시됨"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 경험이 많은 증권 애널리스트입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                timeout=30  # 30초 타임아웃
            )
            
            analysis_text = response.choices[0].message.content
            
            # JSON 파싱 시도
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트를 구조화된 형식으로 반환
                analysis = {
                    "rating": "분석중",
                    "score": 50,
                    "conclusion": analysis_text,
                    "token_usage": response.usage.total_tokens
                }
            
            # 토큰 사용량 추가
            analysis['token_usage'] = response.usage.total_tokens
            analysis['from_cache'] = False  # API로부터 새로 생성됨
            
            # 캐시 저장
            self._save_cache(cache_key, analysis)
            
            return analysis
        
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # 상세 에러 정보 로깅
            if "RateLimitError" in error_type:
                error_msg = "OpenAI API 호출 초과. 잠시 후 다시 시도해주세요."
            elif "AuthenticationError" in error_type:
                error_msg = "OpenAI API 키가 유효하지 않습니다."
            elif "APIError" in error_type:
                error_msg = f"OpenAI API 오류: {error_msg}"
            elif "Timeout" in error_type or "timeout" in error_msg.lower():
                error_msg = "요청 시간 초과. 잠시 후 다시 시도해주세요."
            
            return {
                "error": error_msg,
                "error_type": error_type,
                "api_response": None,
                "score": 0
            }
    
    def analyze_technical(self, stock_code: str, stock_name: str, 
                         price_data: dict) -> dict:
        """
        기술적 분석
        
        Args:
            stock_code: 종목 코드
            stock_name: 종목명
            price_data: 가격 데이터 (현재가, 이동평균, RSI, MACD 등)
        
        Returns:
            분석 결과
        """
        cache_key = self._generate_cache_key(stock_code, stock_name, "technical", price_data)
        
        # 캐시 확인
        cached = self._get_cached_analysis(cache_key)
        if cached:
            st.info(f"💾 캐시된 기술 분석 사용 (종목: {stock_name})")
            cached['from_cache'] = True  # 캐시 플래그 추가
            return cached
        
        try:
            prompt = f"""
            다음 기술적 지표를 분석하여 단기 투자 신호를 제시해주세요.
            
            종목: {stock_name} ({stock_code})
            
            기술 지표:
            {json.dumps(price_data, ensure_ascii=False, indent=2)}
            
            분석 항목:
            1. 트렌드 방향 (상승/하강/횡보)
            2. 저항선 및 지지선
            3. 매수/매도 신호
            4. 단기 목표가
            5. 위험 수위
            
            응답 형식:
            {{
                "trend": "상승/하강/횡보",
                "signal": "강한매수/매수/중립/매도/강한매도",
                "score": 0-100,
                "support_level": 0,
                "resistance_level": 0,
                "target_price": 0,
                "risk_level": "낮음/중간/높음",
                "conclusion": "기술적 분석 종합의견"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 기술적 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                timeout=30  # 30초 타임아웃
            )
            
            analysis_text = response.choices[0].message.content
            
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis = {
                    "signal": "분석중",
                    "score": 50,
                    "conclusion": analysis_text,
                    "token_usage": response.usage.total_tokens
                }
            
            analysis['token_usage'] = response.usage.total_tokens
            analysis['from_cache'] = False  # API로부터 새로 생성됨
            self._save_cache(cache_key, analysis)
            
            return analysis
        
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            if "RateLimitError" in error_type:
                error_msg = "OpenAI API 호출 초과. 잠시 후 다시 시도해주세요."
            elif "AuthenticationError" in error_type:
                error_msg = "OpenAI API 키가 유효하지 않습니다."
            elif "APIError" in error_type:
                error_msg = f"OpenAI API 오류: {error_msg}"
            elif "Timeout" in error_type or "timeout" in error_msg.lower():
                error_msg = "요청 시간 초과. 잠시 후 다시 시도해주세요."
            
            return {
                "error": error_msg,
                "error_type": error_type,
                "api_response": None,
                "score": 0
            }
    
    def generate_recommendation(self, stock_code: str, stock_name: str,
                               fundamental_analysis: dict, 
                               technical_analysis: dict,
                               sentiment_score: float = 50) -> dict:
        """
        종합 투자 추천
        
        Args:
            stock_code: 종목 코드
            stock_name: 종목명
            fundamental_analysis: 기초 분석 결과
            technical_analysis: 기술 분석 결과
            sentiment_score: 감정 점수 (0-100)
        
        Returns:
            종합 추천 결과
        """
        cache_key = self._generate_cache_key(stock_code, stock_name, "recommendation", 
                                            {
                                                "fundamental": fundamental_analysis,
                                                "technical": technical_analysis,
                                                "sentiment": sentiment_score
                                            })
        
        # 캐시 확인
        cached = self._get_cached_analysis(cache_key)
        if cached:
            st.info(f"💾 캐시된 투자 추천 사용 (종목: {stock_name})")
            cached['from_cache'] = True  # 캐시 플래그 추가
            return cached
        
        try:
            prompt = f"""
            다음 분석 결과들을 종합하여 최종 투자 추천을 제시해주세요.
            
            종목: {stock_name} ({stock_code})
            
            기초 분석 결과:
            {json.dumps(fundamental_analysis, ensure_ascii=False, indent=2)}
            
            기술 분석 결과:
            {json.dumps(technical_analysis, ensure_ascii=False, indent=2)}
            
            시장 심리 점수: {sentiment_score}/100
            
            최종 추천:
            1. 종합 평가
            2. 투자 기간별 관점 (단기/중기/장기)
            3. 목표가 및 리스크
            4. 투자 방식 제안
            5. 주의사항
            
            응답 형식:
            {{
                "overall_rating": "강력추천/추천/중립/약한매도/강한매도",
                "overall_score": 0-100,
                "short_term_outlook": "긍정/중립/부정",
                "medium_term_outlook": "긍정/중립/부정",
                "long_term_outlook": "긍정/중립/부정",
                "target_price": 0,
                "stop_loss": 0,
                "investment_horizon": "3-6개월/6-12개월/1-3년",
                "key_risks": ["위험요소1", "위험요소2"],
                "recommendation": "최종 종합 의견"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 칭찬받는 투자 자문가입니다. 객관적이고 균형잡힌 조언을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,  # 더 일관성 있는 추천을 위해 낮은 temperature
                max_tokens=1500,
                timeout=30  # 30초 타임아웃
            )
            
            analysis_text = response.choices[0].message.content
            
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis = {
                    "overall_rating": "분석중",
                    "overall_score": 50,
                    "recommendation": analysis_text,
                    "token_usage": response.usage.total_tokens
                }
            
            analysis['token_usage'] = response.usage.total_tokens
            analysis['from_cache'] = False  # API로부터 새로 생성됨
            self._save_cache(cache_key, analysis)
            
            return analysis
        
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            if "RateLimitError" in error_type:
                error_msg = "OpenAI API 호출 초과. 잠시 후 다시 시도해주세요."
            elif "AuthenticationError" in error_type:
                error_msg = "OpenAI API 키가 유효하지 않습니다."
            elif "APIError" in error_type:
                error_msg = f"OpenAI API 오류: {error_msg}"
            elif "Timeout" in error_type or "timeout" in error_msg.lower():
                error_msg = "요청 시간 초과. 잠시 후 다시 시도해주세요."
            
            return {
                "error": error_msg,
                "error_type": error_type,
                "api_response": None,
                "overall_score": 0
            }
    
    def get_cache_stats(self) -> dict:
        """
        캐시 통계 조회
        
        Returns:
            캐시 파일 수, 총 크기 등
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        # 만료된 캐시 정리
        expired_count = 0
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                cache_time = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - cache_time > timedelta(days=self.cache_expiry_days):
                    cache_file.unlink()
                    expired_count += 1
            except:
                pass
        
        return {
            'cache_count': len(cache_files) - expired_count,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'expired_removed': expired_count,
            'cache_expiry_days': self.cache_expiry_days
        }
