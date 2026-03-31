"""
LLMInsightEngine: DecisionOrchestrator 결과에 대한 자연어 해설 생성
- 종목 선정 근거 설명
- 뉴스 감성 분석
- 시장 국면 해설
- MD5 기반 7일 TTL 캐싱
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Dict, Any
import streamlit as st
from dotenv import load_dotenv


# .env 파일 로드
load_dotenv()


class LLMInsightEngine:
    """LLM 기반 투자 인사이트 생성 엔진 (캐싱 포함)"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, cache_dir: str = "data/llm_insight_cache"):
        """
        Args:
            api_key: OpenAI API 키 (미제공시 환경변수에서 자동 로드)
            model: OpenAI 모델 (미제공시 환경변수 또는 기본값)
            cache_dir: 캐시 저장 디렉토리
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # 모델 우선순위: 인자 > 환경변수(OPENAI_INSIGHT_MODEL) > 환경변수(OPENAI_MODEL) > 기본값
        self.model = model or os.getenv("OPENAI_INSIGHT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
        self.cache_dir = cache_dir
        self.client = None
        self.cache_expiry_days = 7
        
        # 캐시 디렉토리 생성
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _lazy_load_client(self):
        """OpenAI 클라이언트 lazy load"""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai 패키지가 설치되지 않았습니다. pip install openai를 실행하세요.")
    
    def _generate_cache_key(self, insight_type: str, payload: Dict) -> str:
        """
        MD5 기반 캐시 키 생성
        
        Args:
            insight_type: "stock_selection", "news_sentiment", "market_regime"
            payload: 캐시 키 생성에 사용할 데이터 (날짜, 종목코드 등)
        
        Returns:
            MD5 해시 문자열
        """
        hash_input = json.dumps(
            {
                "insight_type": insight_type,
                "payload": payload,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cached_insight(self, cache_key: str) -> Optional[Dict]:
        """
        캐시 파일에서 인사이트 로드 (TTL 확인)
        
        Returns:
            캐시된 insight 또는 None
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # TTL 확인
            cached_at = datetime.fromisoformat(cache_data.get("cached_at", ""))
            if datetime.now() - cached_at > timedelta(days=self.cache_expiry_days):
                # 캐시 만료
                os.remove(cache_file)
                return None
            
            return cache_data.get("insight")
        except Exception:
            return None
    
    def _save_cache(self, cache_key: str, insight: Dict) -> None:
        """캐시 파일에 인사이트 저장"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            cache_data = {
                "cached_at": datetime.now().isoformat(),
                "insight": insight,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 캐시 저장 실패는 무시
    
    def explain_stock_selection(
        self,
        top_candidates: list[Dict],
        decision: Dict,
        date: str,
    ) -> str:
        """
        종목 선정 근거를 자연어로 설명
        
        Args:
            top_candidates: [{stock, sector, signal, score}, ...]
            decision: DecisionOrchestrator 결과 (market_bias, confidence 등)
            date: 기준일자 (YYYY-MM-DD)
        
        Returns:
            자연어 해설 문자열
        """
        # 캐시 키 생성
        payload = {
            "date": date,
            "stocks": [c.get("stock") for c in top_candidates[:5]],
            "count": len(top_candidates),
        }
        cache_key = self._generate_cache_key("stock_selection", payload)
        
        # 캐시 확인
        cached = self._get_cached_insight(cache_key)
        if cached is not None:
            return cached.get("explanation", "")
        
        # LLM 호출
        try:
            self._lazy_load_client()
            
            candidate_text = "\n".join([
                f"- {c.get('stock')} ({c.get('code')}) | {c.get('sector')} | {c.get('signal')} | 점수 {c.get('score', 0):.1f}"
                for c in top_candidates[:5]
            ])
            
            prompt = f"""당신은 투자 분석가입니다. 다음 투자 후보 종목들이 왜 선택되었는지 간단히 설명해주세요 (100자 이내).

기준일: {date}
시장 판단: {decision.get('market_bias', 'NEUTRAL')}
신뢰도: {decision.get('confidence', 0.5):.1%}

추천 종목 (Top 5):
{candidate_text}

주의:
- 종목 코드나 구체적 주가 예측은 하지 마세요
- 선택 근거의 핵심만 간결하게 설명하세요
- 면책조항을 붙이세요"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "투자 분석 조수로서 간결하고 객관적인 설명을 제공합니다."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200,
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # 캐시 저장
            self._save_cache(cache_key, {"explanation": explanation})
            
            return explanation
        
        except Exception as e:
            st.warning(f"⚠️ 종목 설명 생성 실패: {e}")
            return f"추천 종목: {', '.join([c.get('stock') for c in top_candidates[:3]])}"
    
    def analyze_news_sentiment(
        self,
        top_candidates: list[Dict],
        date: str,
    ) -> str:
        """
        뉴스 기반 시장 심리 분석
        
        Args:
            top_candidates: 추천 종목 리스트 (각 종목의 news_sentiment 포함)
            date: 기준일자
        
        Returns:
            뉴스 감성 분석 결과 문자열
        """
        payload = {
            "date": date,
            "count": len(top_candidates),
        }
        cache_key = self._generate_cache_key("news_sentiment", payload)
        
        cached = self._get_cached_insight(cache_key)
        if cached is not None:
            return cached.get("sentiment_summary", "")
        
        # 간단한 규칙 기반 분석 (LLM 없이 우선 처리)
        try:
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for candidate in top_candidates[:5]:
                sentiment_score = candidate.get("sentiment_score", 0)
                if sentiment_score > 0.3:
                    positive_count += 1
                elif sentiment_score < -0.3:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            if positive_count > negative_count:
                summary = f"시장 심리: 긍정적 {positive_count}건 우세 📈"
            elif negative_count > positive_count:
                summary = f"시장 심리: 부정적 {negative_count}건 우세 📉"
            else:
                summary = f"시장 심리: 중립적 ({positive_count}+, {neutral_count}○, {negative_count}─)"
            
            # 캐시 저장
            self._save_cache(cache_key, {"sentiment_summary": summary})
            
            return summary
        
        except Exception as e:
            return f"뉴스 심리 분석 불가"
    
    def explain_market_regime(
        self,
        regime: Dict,
        sector_strength: Dict,
        date: str,
    ) -> str:
        """
        현재 시장 국면의 의미 설명
        
        Args:
            regime: {regime: "BULL" | "BEAR" | "SIDEWAYS", confidence: 0.0~1.0}
            sector_strength: {sector_name: power_score, ...}
            date: 기준일자
        
        Returns:
            시장 국면 해설 문자열
        """
        payload = {
            "date": date,
            "regime": regime.get("regime", "UNKNOWN"),
        }
        cache_key = self._generate_cache_key("market_regime", payload)
        
        cached = self._get_cached_insight(cache_key)
        if cached is not None:
            return cached.get("regime_explanation", "")
        
        try:
            self._lazy_load_client()
            
            regime_type = regime.get("regime", "UNKNOWN")
            confidence = regime.get("confidence", 0.5)
            
            # 섹터 강도 요약
            sorted_sectors = sorted(
                sector_strength.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_sectors = ", ".join([f"{s}({v:.0f})" for s, v in sorted_sectors[:3]])
            
            prompt = f"""당신은 시장 분석가입니다. 현재 시장 국면을 간단히 설명해주세요 (80자 이내).

기준일: {date}
시장 국면: {regime_type}
확신도: {confidence:.1%}
강세 섹터: {top_sectors}

예시:
- "상승장 초입, 반도체·에너지 섹터에 자금 쏠림"
- "조정 국면 지속, 방어주 중심 쏠림"

주의: 구체적 주가 예측 없이 국면만 설명하세요."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "시장 분석가로서 객관적이고 간결한 국면 설명을 제공합니다."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=150,
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # 캐시 저장
            self._save_cache(cache_key, {"regime_explanation": explanation})
            
            return explanation
        
        except Exception as e:
            st.warning(f"⚠️ 국면 설명 생성 실패: {e}")
            regime_emoji = "📈" if regime_type == "BULL" else "📉" if regime_type == "BEAR" else "➡️"
            return f"{regime_emoji} 현재 시장 국면: {regime_type}"
    
    def generate_comprehensive_insights(
        self,
        top_candidates: list[Dict],
        decision: Dict,
        regime: Dict,
        sector_strength: Dict,
        date: str,
    ) -> Dict[str, str]:
        """
        3가지 LLM 인사이트를 한번에 생성
        
        Returns:
            {"stock_explanation": str, "sentiment_summary": str, "regime_explanation": str}
        """
        return {
            "stock_explanation": self.explain_stock_selection(top_candidates, decision, date),
            "sentiment_summary": self.analyze_news_sentiment(top_candidates, date),
            "regime_explanation": self.explain_market_regime(regime, sector_strength, date),
        }
