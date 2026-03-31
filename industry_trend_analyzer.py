"""
산업 트렌드 분석 모듈
최신 뉴스를 바탕으로 유망한 산업을 식별하고 관련 종목을 추천합니다.
"""

import pandas as pd
from collections import defaultdict, Counter
import streamlit as st
from naver_news_crawler import NaverNewsCrawler
from crawling_kospi import CrawlingKospi


class IndustryTrendAnalyzer:
    """산업 트렌드 분석 클래스"""
    
    def __init__(self):
        self.crawler = NaverNewsCrawler()
        self.kospi_crawler = CrawlingKospi()
        self.industry_keywords = self._initialize_industry_keywords()
        self.news_analyzer = None  # 지연 로드
        
    def _get_news_analyzer(self):
        """지연 로드: 필요할 때만 NewsAnalysis 임포트"""
        if self.news_analyzer is None:
            try:
                from news_analysis import NewsAnalysis
                self.news_analyzer = NewsAnalysis()
            except Exception as e:
                st.warning(f"⚠️ 감정 분석 모델 로드 실패: {e}")
                # 모델 없이도 작동하도록 None 반환
                return None
        return self.news_analyzer
        
    def _initialize_industry_keywords(self) -> dict:
        """산업별 키워드 사전 초기화"""
        return {
            "AI/반도체": ["AI", "반도체", "칩", "GPU", "NVIDIA", "인공지능", "머신러닝", "딥러닝", "ChatGPT", "생성형AI"],
            "바이오/제약": ["바이오", "제약", "신약", "임상", "백신", "생명공학", "DNA", "단백질"],
            "전기차/배터리": ["전기차", "EV", "배터리", "수소", "충전소", "전동", "친환경", "탄소중립"],
            "재정에너지": ["태양광", "풍력", "재생에너지", "수소", "핵융합", "그린에너지"],
            "메타버스/게임": ["메타버스", "게임", "NFT", "블록체인", "가상현실", "VR", "웹3"],
            "5G/통신": ["5G", "6G", "통신", "네트워크", "IoT", "통신사"],
            "클라우드": ["클라우드", "데이터센터", "AWS", "클라우드 컴퓨팅", "SaaS"],
            "로봇/자동화": ["로봇", "자동화", "산업용로봇", "협동로봇", "자율주행"],
            "화학/소재": ["화학", "소재", "배터리 소재", "반도체 소재", "디스플레이"],
            "금융기술": ["핀테크", "블록체인", "암호화폐", "금융기술", "디지털뱅킹"],
            "식음료": ["식품", "음료", "외식", "편의점", "카페"],
            "유통/전자상거래": ["쇠퇴", "온라인", "전자상거래", "이커머스", "배송"],
            "건설": ["건설", "부동산", "주택", "인프라"],
            "자동차": ["자동차", "자동차부품", "완성차"],
            "해운/물류": ["해운", "물류", "항공", "배송"],
            "금융": ["은행", "보험", "증권", "금융"],
            "유틸리티": ["전력", "가스", "수도"],
            "소비재": ["의류", "신발", "패션", "화장품", "생활용품"],
            "의료": ["의료기기", "병원", "진단", "의료"],
        }
    
    def get_kospi_stocks_by_industry(self) -> dict:
        """KOSPI 200 종목을 산업별로 분류"""
        try:
            kospi_dict = self.kospi_crawler.GetKospi200()
            
            # 산업 코드 매핑 (확대된 종목)
            industry_map = {
                # AI/반도체 (14개)
                "005930": "AI/반도체",  # 삼성전자
                "000660": "AI/반도체",  # SK하이닉스
                "035420": "AI/반도체",  # NAVER
                "035720": "AI/반도체",  # 카카오
                "058270": "AI/반도체",  # SK
                "012330": "AI/반도체",  # 현대모비스
                "010100": "AI/반도체",  # 삼성 C&T
                "095570": "AI/반도체",  # AJ네트웍스
                "003550": "AI/반도체",  # LG
                "047050": "AI/반도체",  # POSCO홀딩스
                "011780": "AI/반도체",  # 현대일렉트로니
                "086520": "AI/반도체",  # 에코프로
                "028260": "AI/반도체",  # 삼성물산
                "034730": "AI/반도체",  # SK텔레콤
                
                # 전기차/배터리 (12개)
                "005380": "전기차/배터리",  # 현대차
                "051910": "전기차/배터리",  # LG화학
                "373220": "전기차/배터리",  # LG에너지솔루션
                "096770": "전기차/배터리",  # SK이노베이션
                "066970": "전기차/배터리",  # 엘앤에프
                "005490": "전기차/배터리",  # POSCO DX
                "078600": "전기차/배터리",  # 대주전자
                "138040": "전기차/배터리",  # 메리츠화재
                "000100": "전기차/배터리",  # CJ대한통운
                "024110": "전기차/배터리",  # 기아
                "161390": "전기차/배터리",  # 한국타이어
                "042660": "전기차/배터리",  # 현대중공업
                
                # 바이오/제약 (10개)
                "068270": "바이오/제약",  # 셀트리온
                "207940": "바이오/제약",  # SM C&C
                "297050": "바이오/제약",  # 롯데정보통신
                "096300": "바이오/제약",  # 롯데지주
                "011200": "바이오/제약",  # 휴젤
                "019680": "바이오/제약",  # 개인화학
                "323410": "바이오/제약",  # 카카오페이
                "024780": "바이오/제약",  # 삼성피엔엠
                "128940": "바이오/제약",  # 케이씨지
                "051375": "바이오/제약",  # LG이노텍
                
                # 5G/통신 (8개)
                "030200": "5G/통신",  # KT
                "017670": "5G/통신",  # SK텔레콤
                "034730": "5G/통신",  # SK텔레콤(중복)
                "003410": "5G/통신",  # 쌍용C&E
                "008770": "5G/통신",  # 호텔신라
                "021240": "5G/통신",  # 신한기업자산
                "039130": "5G/통신",  # 하나칠공
                "001450": "5G/통신",  # 현대해상
                
                # 금융 (15개)
                "055550": "금융",  # 신한지주
                "086790": "금융",  # 하나금융지주
                "024110": "금융",  # 우리금융지주
                "000040": "금융",  # 삼성중공업
                "032830": "금융",  # 삼성생명
                "016360": "금융",  # 삼성증권
                "061970": "금융",  # LG이노텍(증권)
                "035720": "금융",  # 카카오뱅크
                "323410": "금융",  # 카카오페이
                "066570": "금융",  # LG전자
                "0033000": "금융",  # SK C&C
                "034730": "금융",  # SK텔레콤(금융)
                "001310": "금융",  # 삼성화재
                "005959": "금융",  # SK하이닉스(금융)
                "105560": "금융",  # KB금융
                
                # 자동차 (8개)
                "005380": "자동차",  # 현대차
                "024110": "자동차",  # 기아
                "012330": "자동차",  # 현대모비스
                "042660": "자동차",  # 현대중공업
                "015590": "자동차",  # 현대글로비스
                "161390": "자동차",  # 한국타이어
                "006110": "자동차",  # LG전자(자동차부품)
                "028260": "자동차",  # 삼성물산
                
                # 화학/소재 (10개)
                "011170": "화학/소재",  # 롯데케미칼
                "001570": "화학/소재",  # 금호타이어
                "078600": "화학/소재",  # 대주전자
                "086520": "화학/소재",  # 에코프로
                "066970": "화학/소재",  # 엘앤에프
                "011780": "화학/소재",  # 현대일렉트로니
                "003850": "화학/소재",  # 보령
                "058470": "화학/소재",  # 표준C&D
                "161390": "화학/소재",  # 한국타이어(중복)
                "009150": "화학/소재",  # 삼성전기
                
                # 식음료 (6개)
                "000080": "식음료",  # 길음식
                "022100": "식음료",  # 포스코
                "109740": "식음료",  # 오젤엔터테인먼트
                "035250": "식음료",  # 강원랜드
                "336260": "식음료",  # 웹젠
                "006200": "식음료",  # 롯데리자인(식음료)
                
                # 클라우드/IT (8개)
                "035420": "클라우드",  # NAVER
                "035720": "클라우드",  # 카카오
                "010100": "클라우드",  # 삼성 C&T
                "003410": "클라우드",  # 쌍용C&E
                "021240": "클라우드",  # 신한기업자산
                "039130": "클라우드",  # 하나칠공
                "001450": "클라우드",  # 현대해상
                "089860": "클라우드",  # SK하이닉스(IT)
            }
            
            # 산업별 그룹핑
            industry_stocks = defaultdict(dict)
            
            for code, name in kospi_dict.items():
                industry = industry_map.get(code, "기타")
                industry_stocks[industry][code] = name
            
            return dict(industry_stocks)
        
        except Exception as e:
            print(f"산업 분류 오류: {e}")
            return {}
    
    def analyze_industry_trends(self, days: int = 7) -> dict:
        """
        산업별 뉴스 트렌드 분석
        
        Args:
            days: 분석 기간 (일)
        
        Returns:
            산업별 트렌드 분석 결과
        """
        industry_stocks = self.get_kospi_stocks_by_industry()
        
        if not industry_stocks:
            return {}
        
        industry_trends = {}
        news_analyzer = self._get_news_analyzer()  # 지연 로드
        
        for industry, stocks in industry_stocks.items():
            # 산업별 뉴스 수집
            all_news = []
            all_sentiments = []
            
            for code, name in list(stocks.items())[:10]:  # 산업당 최대 10개 종목 (증가)
                try:
                    news_df = self.crawler.get_recent_news(code, max_news=20)  # 종목당 최대 20개 (증가)
                    
                    if not news_df.empty:
                        news_df['code'] = code
                        news_df['name'] = name
                        all_news.append(news_df)
                        
                        # 감정 분석 (모델이 있으면 수행)
                        if news_analyzer is not None:
                            try:
                                titles = news_df['제목'].tolist()
                                sentiments = news_analyzer.predict_sentiment_batch(titles)
                                all_sentiments.extend(sentiments)
                            except Exception as e:
                                # 모델 분석 실패시 기본값 사용
                                st.warning(f"⚠️ {name}({code}) 감정 분석 실패: {str(e)[:100]}")
                                all_sentiments.extend(['🔘'] * len(news_df))
                        else:
                            # 모델 없을 시 중립 감정으로 설정
                            all_sentiments.extend(['🔘'] * len(news_df))
                
                except Exception as e:
                    continue
            
            if all_news:
                # 산업별 통합 뉴스 데이터
                industry_news_df = pd.concat(all_news, ignore_index=True)
                
                # 감정 점수 계산 (개선: 실제 감정 분포 확인)
                sentiment_distribution = Counter(all_sentiments)
                sentiment_score = self._calculate_sentiment_score(all_sentiments)
                
                # 뉴스 빈도
                news_count = len(industry_news_df)
                
                # 산업 유망도 점수 (뉴스 빈도 + 긍정 감성) - 스케일 조정
                # 뉴스 수가 많을수록, 감정점수가 높을수록 유망도 증가
                trend_score = (min(news_count, 50) / 50) * 40 + sentiment_score * 60  # 감정점수 가중치 증가
                trend_score = min(100, max(0, trend_score))
                
                industry_trends[industry] = {
                    'news_count': news_count,
                    'sentiment_score': sentiment_score,  # 0~100 (높을수록 긍정)
                    'trend_score': trend_score,  # 0~100 (높을수록 유망)
                    'recent_news': industry_news_df.head(5),
                    'stocks': stocks,
                    'sentiment_distribution': sentiment_distribution
                }
        
        return industry_trends
    
    def _calculate_sentiment_score(self, sentiments: list) -> float:
        """
        감정 리스트를 종합 점수로 변환 (0~100)
        
        😊 = 100 (긍정)
        🔘 = 50 (중립)
        😡 = 0 (부정)
        """
        if not sentiments:
            return 50
        
        scores = {
            '😊': 100,
            '🔘': 50,
            '😡': 0
        }
        
        total_score = sum(scores.get(s, 50) for s in sentiments)
        return total_score / len(sentiments)
    
    def get_trending_industries(self, top_n: int = 5) -> pd.DataFrame:
        """
        상위 유망 산업 반환
        
        Args:
            top_n: 상위 N개 산업
        
        Returns:
            산업, 유망도, 뉴스 수, 감정 점수 포함한 DataFrame
        """
        trends = self.analyze_industry_trends()
        
        if not trends:
            return pd.DataFrame()
        
        # DataFrame으로 변환
        trend_data = []
        for industry, data in trends.items():
            trend_data.append({
                '산업': industry,
                '유망도': data['trend_score'],
                '뉴스수': data['news_count'],
                '감정점수': data['sentiment_score'],
                '긍정': data['sentiment_distribution'].get('😊', 0),
                '중립': data['sentiment_distribution'].get('🔘', 0),
                '부정': data['sentiment_distribution'].get('😡', 0),
            })
        
        df = pd.DataFrame(trend_data).sort_values('유망도', ascending=False)
        return df.head(top_n)
    
    def get_top_companies_by_industry(self, industry: str, top_n: int = 10) -> dict:
        """
        특정 산업의 상위 종목 반환
        
        Args:
            industry: 산업명
            top_n: 상위 N개
        
        Returns:
            {'코드': '종목명'} 형태의 딕셔너리
        """
        industry_stocks = self.get_kospi_stocks_by_industry()
        
        if industry not in industry_stocks:
            return {}
        
        stocks = industry_stocks[industry]
        return dict(list(stocks.items())[:top_n])
    
    def get_industry_details(self, industry: str) -> dict:
        """
        산업 상세 정보 조회
        
        Args:
            industry: 산업명
        
        Returns:
            산업의 상세 정보 (뉴스, 감정, 관련 종목 등)
        """
        trends = self.analyze_industry_trends()
        
        if industry not in trends:
            return None
        
        industry_data = trends[industry]
        
        return {
            'industry': industry,
            'trend_score': industry_data['trend_score'],
            'sentiment_score': industry_data['sentiment_score'],
            'news_count': industry_data['news_count'],
            'recent_news': industry_data['recent_news'],
            'stocks': industry_data['stocks'],
            'sentiment_distribution': dict(industry_data['sentiment_distribution']),
            'analysis': self._generate_industry_analysis(industry_data),
        }
    
    def _generate_industry_analysis(self, industry_data: dict) -> str:
        """산업 분석 의견 생성"""
        trend_score = industry_data['trend_score']
        sentiment_score = industry_data['sentiment_score']
        news_count = industry_data['news_count']
        
        # 분석 의견 생성
        if trend_score >= 75:
            trend_opinion = "🔥 **매우 유망** - 최근 뉴스와 긍정 감정이 높음"
        elif trend_score >= 60:
            trend_opinion = "🟢 **유망** - 긍정적인 시장 신호 감지"
        elif trend_score >= 45:
            trend_opinion = "🟡 **중립** - 업계 뉴스 주목 필요"
        else:
            trend_opinion = "🔴 **약세** - 부정적 뉴스 우세"
        
        # 뉴스 활성도
        if news_count > 20:
            news_opinion = "📰 매우 활발한 뉴스 커버리지"
        elif news_count > 10:
            news_opinion = "📰 적당한 뉴스 활동"
        else:
            news_opinion = "📰 신문 주목도 낮음"
        
        # 시장 감정
        if sentiment_score >= 70:
            sentiment_opinion = "😊 긍정적 시장 반응"
        elif sentiment_score >= 50:
            sentiment_opinion = "🔘 중립적 시장 반응"
        else:
            sentiment_opinion = "😡 부정적 시장 반응"
        
        return f"{trend_opinion}\n{news_opinion}\n{sentiment_opinion}"


def get_industry_trend_recommendation(industry_data: dict, kospi_list: dict) -> pd.DataFrame:
    """
    산업 데이터를 바탕으로 투자 추천 결과 생성
    
    Args:
        industry_data: 산업 상세 정보
        kospi_list: KOSPI 200 종목 정보
    
    Returns:
        추천 종목 DataFrame
    """
    stocks = industry_data['stocks']
    
    # 종목별 가중치 설정 (간단한 순위 기반)
    recommendation_data = []
    
    for idx, (code, name) in enumerate(list(stocks.items())[:10]):
        rec_score = 100 - (idx * 8)  # 상위일수록 높은 점수
        
        recommendation_data.append({
            'code': code,
            'name': name,
            'score': rec_score,
            'level': '강력 추천' if rec_score >= 80 else '추천' if rec_score >= 60 else '관심',
            'industry_trend': industry_data['trend_score'],
            'industry_sentiment': industry_data['sentiment_score'],
        })
    
    return pd.DataFrame(recommendation_data)
