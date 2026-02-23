import torch
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import torch.nn.functional as F

class NewsAnalysis:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NewsAnalysis 초기화 - Device: {self.device}")
        self.summary_tokenizer = None
        self.summary_model = None
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        

    def _load_summary_model(self):
        """지연 로드: 요약 모델"""
        try:
            from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
            self.tokenizer_model_name = "digit82/KoBART-summarization"
            self.summary_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_model_name)
            self.summary_model = BartForConditionalGeneration.from_pretrained(self.tokenizer_model_name)
            self.summary_model.to(self.device)
            print(f"요약 모델 로드 완료 - Device: {self.device}")
        except Exception as e:
            print(f"⚠️ 요약 모델 로드 실패: {e}")
            raise
    
    def _load_sentiment_model(self):
        """지연 로드: 감정 분석 모델"""
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            self.sentiment_model_name = "C:\\workspace\\HfModels\\korean_sentiment_analysis_kcelectra"
            self.sentiment_tokenizer = BertTokenizer.from_pretrained(self.sentiment_model_name)
            self.sentiment_model = BertForSequenceClassification.from_pretrained(self.sentiment_model_name, num_labels=2)
            self.sentiment_model.to(self.device)
            print(f"감정 분석 모델 로드 완료 - Device: {self.device}")
            return True
        except Exception as e:
            print(f"⚠️ 감정 분석 모델 로드 실패: {e}")
            self.sentiment_tokenizer = None
            self.sentiment_model = None
            return False

    def summarize_text(self, text):
        if self.summary_tokenizer is None or self.summary_model is None:
            self._load_summary_model()
            
        inputs = self.summary_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        # 입력 데이터를 모델과 동일한 디바이스로 이동
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        summary_ids = self.summary_model.generate(inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)
        return self.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def batch_summarize_texts(self, texts, max_length=200, num_beams=4, early_stopping=True):
        if self.summary_tokenizer is None or self.summary_model is None:
            self._load_summary_model()
        
        # 여러 문장을 배치로 토큰화 (padding 적용)
        inputs = self.summary_tokenizer(
            texts, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True, 
            padding=True
        )
        # 입력 데이터를 모델과 동일한 디바이스로 이동
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        # 배치로 요약 생성
        summary_ids = self.summary_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping
        )
        # 각 요약 결과 디코딩하여 리스트로 반환
        del self.summary_model

        return [self.summary_tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]


    # def predict_sentiment(self, text):
    #     # 📌 KoBERT 감성 분석 모델 & 토크나이저 로드
    #     inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    #     outputs = self.sentiment_model(**inputs)        
    #     prediction = torch.argmax(outputs.logits, dim=-1).item()        
    #     return "긍정 😊" if prediction == 1 else "부정 😡"
    
    # def predict_sentiment2(self, text):
    #     """
    #     단일 문장 감성 분석 (Batch보다 속도 느림)
    #     """
    #     return self.predict_sentiment_batch([text])[0]
    
    def predict_sentiment_batch(self, texts):
        """뉴스 제목을 배치로 감성 분석"""
        if not texts:
            return []  # 빈 입력일 경우 빈 리스트 반환
        
        # 감정 분석 모델 로드 확인
        if self.sentiment_tokenizer is None or self.sentiment_model is None:
            try:
                success = self._load_sentiment_model()
                if not success:
                    # 모델 로드 실패 시 중립 감정으로 반환
                    return ['🔘'] * len(texts)
            except Exception as e:
                print(f"⚠️ 감정 모델 로드 중 오류: {e}")
                # 모델 로드 실패 시 중립 감정으로 반환
                return ['🔘'] * len(texts)
        
        try:
            with torch.no_grad():
                inputs = self.sentiment_tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding="longest",  # 가장 긴 문장 기준 패딩
                    max_length=128
                ).to(self.device)

                outputs = self.sentiment_model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)  # 확률 변환
                predictions = torch.argmax(probs, dim=-1).cpu().tolist()

                # ✅ 확률 기반 감성 분류
                sentiment_results = []
                for i, pred in enumerate(predictions):
                    positive_prob = probs[i][1].item()
                    negative_prob = probs[i][0].item()

                    # ✅ 확률 threshold 적용
                    if positive_prob >= 0.6:
                        sentiment_results.append(f"😊")
                    elif negative_prob >= 0.6:
                        sentiment_results.append(f"😡")
                    else:
                        sentiment_results.append(f"🔘")  # 애매하면 중립

                return sentiment_results
        
        except Exception as e:
            print(f"⚠️ 감정 분석 중 오류: {e}")
            # 분석 중 오류 발생 시 중립 감정으로 반환
            return ['🔘'] * len(texts)
    

    # 📌 병렬 처리 적용 (ThreadPoolExecutor 활용)
    def batch_sentiment_analysis(self, news_df, batch_size=16):        

        """데이터프레임의 뉴스 제목들을 배치로 감성 분석"""
        if "제목" not in news_df.columns:
            st.error("⚠ '제목' 컬럼을 찾을 수 없습니다. 데이터 형식을 확인하세요!")
            return news_df  # 오류 발생 시 원본 데이터 반환
        
        # 감정 분석 모델 로드 시도
        try:
            if self.sentiment_tokenizer is None or self.sentiment_model is None:
                success = self._load_sentiment_model()
                if not success:
                    # 모델 로드 실패 시 중립 감정으로 처리
                    news_df["감정"] = ['🔘'] * len(news_df)
                    return news_df
        except Exception as e:
            st.warning(f"⚠️ 감정 분석 모델 로드 실패: {e}")
            # 모델 로드 실패 시 중립 감정으로 처리
            news_df["감정"] = ['🔘'] * len(news_df)
            return news_df

        news_list = news_df["제목"].tolist()  # 뉴스 제목 리스트
        results = []

        try:
            with ThreadPoolExecutor() as executor:
                for i in range(0, len(news_list), batch_size):
                    batch = news_list[i : i + batch_size]

                    try:
                        batch_result = self.predict_sentiment_batch(batch)
                        results.extend(batch_result)

                    except Exception as e:
                        st.error(f"⚠ 감성 분석 중 오류 발생: {e}")
                        # 오류 발생 시 해당 배치는 중립으로 처리
                        results.extend(['🔘'] * len(batch))

        except Exception as e:
            st.error(f"⚠ 배치 처리 중 오류 발생: {e}")
            # 전체 실패 시 중립 감정으로 처리
            results = ['🔘'] * len(news_df)

        # ✅ 최종 결과 길이 검증
        if len(results) != len(news_df):
            st.error(f"⚠ 감성 분석 결과 길이({len(results)})와 데이터프레임 길이({len(news_df)})가 다릅니다!")
            # 길이 맞추기 (부족하면 중립으로 채우기)
            while len(results) < len(news_df):
                results.append('🔘')
            results = results[:len(news_df)]

        news_df["감정"] = results

        
        try:
            del self.sentiment_model
        except:
            pass
        torch.cuda.empty_cache()

        return news_df
    
    def empty_cache_model(self):
        # del self.sentiment_model
        # del self.summary_model
        torch.cuda.empty_cache()

    # def load_sentiment_model(self):        
    #     self.sentiment_model_name = "monologg/kobert"  # 모델명을 올바르게 변경
    #     self.sentiment_tokenizer = BertTokenizer.from_pretrained(self.sentiment_model_name)
    #     self.sentiment_model = BertForSequenceClassification.from_pretrained(self.sentiment_model_name, num_labels=2)
    #     self.sentiment_model.to(self.device)


    # def load_summary_model(self):
    #     # KoBART 요약 모델 & 토크나이저 로드        
    #     self.tokenizer_model_name = "digit82/KoBART-summarization"
    #     self.summary_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_model_name)
    #     self.summary_model = BartForConditionalGeneration.from_pretrained(self.tokenizer_model_name)
    #     self.summary_model.to(self.device)

        
    # def empty_cache_sentiment_model(self):
    #     del self.sentiment_model
    #     torch.cuda.empty_cache()

    # def empty_cache_summary_model(self):
    #     del self.summary_model
    #     torch.cuda.empty_cache()