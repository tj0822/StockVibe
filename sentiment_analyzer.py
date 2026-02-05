"""뉴스 감성분석 모듈"""
import re
from typing import Literal, Optional
import torch


class SentimentAnalyzer:
    """뉴스 텍스트 감성분석기 (GPU 지원)"""
    
    def __init__(self, use_model: bool = True):
        """
        Args:
            use_model: True면 딥러닝 모델 사용, False면 키워드 기반
        """
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        self.device = None
        
        if use_model:
            self._initialize_model()
        
        # 키워드 기반 분석용 (fallback)
        self._initialize_keywords()
    
    def _initialize_model(self):
        """GPU 기반 감성분석 모델 초기화"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # GPU 사용 가능 여부 확인
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 한국어 감성분석 모델 로드
            # beomi/KcELECTRA-base-v2022 또는 snunlp/KR-FinBert-SC 등 사용 가능
            model_name = "beomi/KcELECTRA-base-v2022"
            
            print(f"감성분석 모델 로딩 중... (device: {self.device})")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3  # 긍정, 중립, 부정
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ 모델 로드 완료 ({self.device})")
            
        except Exception as e:
            print(f"⚠ 모델 로딩 실패: {e}")
            print("키워드 기반 분석으로 전환합니다.")
            self.use_model = False
    
    def _initialize_keywords(self):
        """키워드 사전 초기화"""
        # 긍정 키워드
        self.positive_keywords = [
            '상승', '증가', '호재', '성장', '확대', '개선', '호황', '급등', '상향',
            '긍정', '회복', '반등', '호조', '상승세', '강세', '선전', '최고',
            '신고가', '돌파', '성공', '수혜', '기대', '흑자', '이익', '실적개선',
            '주가상승', '매수', '투자', '확장', '활황', '호평', '양호', '우수',
            '증가세', '플러스', '수익', '성과', '달성', '초과', '증대', '향상',
            '호실적', '신규', '혁신', '도약', '활발', '수주', '계약', '진출'
        ]
        
        # 부정 키워드
        self.negative_keywords = [
            '하락', '감소', '악재', '위축', '침체', '불황', '급락', '하향',
            '부정', '악화', '둔화', '하락세', '약세', '부진', '최저', '저점',
            '신저가', '붕괴', '실패', '타격', '우려', '적자', '손실', '실적악화',
            '주가하락', '매도', '위기', '축소', '불안', '부실',
            '감소세', '마이너스', '손해', '하회', '미달', '부담', '악영향',
            '위험', '리스크', '부진', '저조', '취소', '철회', '중단'
        ]
        
        # 중립/불확실 키워드
        self.neutral_keywords = [
            '예정', '계획', '전망', '예상', '가능성', '검토', '추진', '발표',
            '협의', '논의', '조정', '유지', '보합', '변동', '관망', '대기'
        ]
    
    def analyze_with_model(self, text: str) -> dict:
        """딥러닝 모델 기반 감성분석"""
        if not self.model or not self.tokenizer:
            return self.analyze_with_keywords(text)
        
        try:
            # 텍스트 길이 제한 (512 토큰)
            if len(text) > 500:
                text = text[:500]
            
            # 토크나이징
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_class].item()
            
            # 클래스 매핑 (0: 부정, 1: 중립, 2: 긍정)
            sentiment_map = {
                0: 'negative',
                1: 'neutral',
                2: 'positive'
            }
            
            emoji_map = {
                'positive': '😊',
                'neutral': '😐',
                'negative': '😞'
            }
            
            sentiment = sentiment_map.get(predicted_class, 'neutral')
            
            # 점수 변환 (-1 ~ 1)
            if predicted_class == 2:  # 긍정
                score = confidence
            elif predicted_class == 0:  # 부정
                score = -confidence
            else:  # 중립
                score = 0.0
            
            return {
                'sentiment': sentiment,
                'score': round(score, 2),
                'confidence': round(confidence, 2),
                'emoji': emoji_map[sentiment],
                'method': 'model'
            }
            
        except Exception as e:
            print(f"모델 분석 실패: {e}, 키워드 분석으로 전환")
            return self.analyze_with_keywords(text)
    
    def analyze_with_keywords(self, text: str) -> dict:
        """키워드 기반 감성분석 (fallback)"""
        if not text or not isinstance(text, str):
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'emoji': '😐',
                'method': 'keyword'
            }
        
        # 텍스트 정규화
        text = text.lower()
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 키워드 카운팅
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text)
        neutral_count = sum(1 for keyword in self.neutral_keywords if keyword in text)
        
        # 가중치 계산
        total_count = positive_count + negative_count + neutral_count
        
        if total_count == 0:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'emoji': '😐',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'method': 'keyword'
            }
        
        # 점수 계산 (-1.0 ~ 1.0)
        score = (positive_count - negative_count) / total_count
        
        # 감성 분류
        if score > 0.15:
            sentiment = 'positive'
            emoji = '😊'
        elif score < -0.15:
            sentiment = 'negative'
            emoji = '😞'
        else:
            sentiment = 'neutral'
            emoji = '😐'
        
        return {
            'sentiment': sentiment,
            'score': round(score, 2),
            'emoji': emoji,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'method': 'keyword'
        }
    
    def analyze(self, text: str) -> dict:
        """
        텍스트 감성분석
        
        Args:
            text: 분석할 텍스트 (제목 + 본문)
        
        Returns:
            {
                'sentiment': 'positive' | 'negative' | 'neutral',
                'score': float,  # -1.0 ~ 1.0
                'emoji': str,
                'confidence': float,  # 모델 사용 시
                'method': 'model' | 'keyword'
            }
        """
        if self.use_model and self.model is not None:
            return self.analyze_with_model(text)
        else:
            return self.analyze_with_keywords(text)
    
    def get_sentiment_color(self, sentiment: str) -> str:
        """감성에 따른 색상 반환"""
        colors = {
            'positive': 'green',
            'negative': 'red',
            'neutral': 'gray'
        }
        return colors.get(sentiment, 'gray')
    
    def get_sentiment_text(self, sentiment: str) -> str:
        """감성에 따른 한글 텍스트 반환"""
        texts = {
            'positive': '긍정',
            'negative': '부정',
            'neutral': '중립'
        }
        return texts.get(sentiment, '중립')
