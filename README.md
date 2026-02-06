"# 📊 StockVibe Pro - AI 주식 분석 플랫폼

**AI 기반 종합 주식 분석 및 투자 관리 시스템**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## ✨ 주요 기능

### 🎯 Core Features (Phase 1-3)
- **AI 온톨로지 예측** - 뉴스, 주가, 재무, 기술지표 종합 분석
- **실시간 데이터** - KOSPI 200 종목 자동 크롤링
- **GPU 감성분석** - 뉴스 긍정/부정 AI 분석
- **고급 기술지표** - RSI, MACD, 볼린저밴드, 스토캐스틱, OBV
- **인터랙티브 차트** - Plotly 캔들스틱 차트

### 🚀 Advanced Features (Phase 4 - NEW!)
- **💼 포트폴리오 관리** - 보유/관심 종목 추적 및 수익률 분석
- **📈 백테스팅** - 전략 검증 및 AI 예측 정확도 추적
- **🔔 스마트 알림** - 가격/등락률/뉴스 키워드 알림
- **🔍 종목 비교** - 다종목 동시 비교 및 레이더 차트
- **🌐 섹터 분석** - 섹터별 수익률, 히트맵, 로테이션 감지
- **📊 고급 차트** - 캔들 패턴 인식, 거래량 분석, 상관관계
- **📥 데이터 내보내기** - Excel/CSV/PDF 리포트 생성
- **⚙️ 개인화 설정** - 테마, 기본값, 알림 설정

---

## 🖥️ 스크린샷

### 메인 대시보드
![Main Dashboard](docs/screenshots/main_dashboard.png)

### 포트폴리오 관리
![Portfolio](docs/screenshots/portfolio.png)

### 백테스팅
![Backtesting](docs/screenshots/backtesting.png)

---

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/StockVibe.git
cd StockVibe
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 앱 실행
```bash
streamlit run streamlit_app.py
```

### 4. 브라우저 접속
```
http://localhost:8501
```

---

## 📦 필수 패키지

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
torch>=2.0.0
transformers>=4.30.0
selenium>=4.15.0
beautifulsoup4>=4.12.0
openpyxl>=3.1.0
```

전체 목록: [requirements.txt](requirements.txt)

---

## 📖 문서

- **[Phase 4 기능 가이드](docs/PHASE4_GUIDE.md)** - 새 기능 상세 설명
- **[온톨로지 예측 시스템](docs/ONTOLOGY_PREDICTION.md)** - AI 예측 알고리즘
- **[API 문서](docs/API.md)** - 프로그래밍 인터페이스

---

## 🎨 메뉴 구조

```
📊 StockVibe Pro
├── 🏠 메인 대시보드
│   ├── KOSPI 200 종목 분석
│   ├── AI 예측 점수
│   └── 실시간 뉴스
├── 💼 포트폴리오
│   ├── 보유 종목 현황
│   ├── 관심 종목
│   └── 종목 추가
├── 📈 백테스팅
│   ├── 전략 백테스팅
│   ├── AI 예측 정확도
│   └── 성과 비교
├── 🔔 알림 관리
│   ├── 가격 알림
│   ├── 등락률 알림
│   └── 뉴스 키워드 알림
├── 🔍 종목 비교
├── 🌐 섹터 분석
├── 📊 고급 차트
│   ├── 캔들 패턴 인식
│   ├── 거래량 분석
│   └── 상관관계 분석
├── 📥 데이터 내보내기
└── ⚙️ 설정
```

---

## 🧠 AI 온톨로지 예측 시스템

### 예측 알고리즘
```
종합 점수 = 뉴스(30%) + 주가(40%) + 재무(10%) + 기술지표(20%)
```

### 분석 요소
1. **뉴스 감성 분석** (30%)
   - GPU 가속 BERT 모델
   - 최근 뉴스 긍정/부정 판단

2. **주가 추세 분석** (40%)
   - 단기/중기/장기 추세
   - 변동성 분석

3. **재무 건전성** (10%)
   - PER, PBR, ROE
   - 영업이익률, 부채비율

4. **기술적 지표** (20%)
   - RSI, MACD, 볼린저밴드
   - 스토캐스틱, OBV
   - 골든크로스/데드크로스

---

## 📊 백테스팅 전략

### 지원 전략
1. **골든크로스 전략**
   - MA5 > MA20 매수
   - MA5 < MA20 매도

2. **RSI 전략**
   - RSI < 30 매수 (과매도)
   - RSI > 70 매도 (과매수)

3. **커스텀 전략**
   - 사용자 정의 전략 추가 가능

---

## 🔧 기술 스택

- **Frontend**: Streamlit
- **Backend**: Python 3.10+
- **AI/ML**: PyTorch, Transformers
- **Data**: Pandas, NumPy
- **Visualization**: Plotly
- **Web Scraping**: Selenium, BeautifulSoup
- **Storage**: JSON (로컬)

---

## 📂 프로젝트 구조

```
StockVibe/
├── app/
│   ├── ui.py                 # 메인 UI (Phase 1-3)
│   ├── phase4_ui.py          # Phase 4 UI
│   ├── data.py               # 데이터 로더
│   ├── signals.py            # 시그널 생성
│   ├── portfolio.py          # 포트폴리오 관리
│   ├── backtesting.py        # 백테스팅 엔진
│   ├── alerts.py             # 알림 시스템
│   ├── comparison.py         # 비교 분석
│   ├── export.py             # 데이터 내보내기
│   ├── advanced_charts.py    # 고급 차트
│   └── settings.py           # 설정 관리
├── data/                     # 데이터 저장소
├── docs/                     # 문서
├── exports/                  # 내보내기 파일
├── crawling_kospi.py         # KOSPI 크롤러
├── naver_news_crawler.py     # 뉴스 크롤러
├── stock_ontology.py         # AI 온톨로지
├── technical_indicators.py   # 기술 지표
├── sentiment_analyzer.py     # 감성 분석
├── streamlit_app.py          # 메인 앱
└── requirements.txt          # 의존성
```

---

## 🎯 사용 시나리오

### 시나리오 1: 일일 시장 분석
```
1. 메인 대시보드에서 KOSPI 200 종합 현황 확인
2. AI 점수 상위 종목 클릭
3. 탭별로 상세 분석 (AI예측/차트/뉴스)
4. 관심 종목에 추가
```

### 시나리오 2: 포트폴리오 관리
```
1. 포트폴리오 메뉴에서 보유 종목 등록
2. 실시간 수익률 확인
3. 섹터별 분산 투자 현황 체크
4. 필요시 비중 조정
```

### 시나리오 3: 전략 검증
```
1. 백테스팅 메뉴 선택
2. 관심 종목 + 전략 선택
3. 과거 데이터로 검증
4. 수익률/승률 확인
5. 최적 전략 선택
```

### 시나리오 4: 알림 설정
```
1. 알림 관리 메뉴
2. 목표가 또는 등락률 알림 설정
3. 자동 모니터링
4. 조건 충족 시 알림 수신
```

---

## 📈 성능 최적화

- **캐싱**: AI 예측 결과 1시간 캐시
- **병렬 처리**: 크롤링 10 workers
- **GPU 가속**: 감성 분석 CUDA 지원
- **데이터 압축**: Git LFS로 대용량 파일 관리

---

## 🛠️ 개발

### 로컬 개발
```bash
# 가상환경 생성
python -m venv venv

# 활성화 (Windows)
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 개발 모드 실행
streamlit run streamlit_app.py --server.runOnSave=true
```

### 테스트
```bash
pytest tests/
```

---

## 🤝 기여

기여를 환영합니다! Pull Request를 보내주세요.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

---

## 🙏 감사의 말

- **Streamlit** - 웹 프레임워크
- **Hugging Face** - AI 모델
- **Plotly** - 차트 라이브러리
- **네이버 금융** - 주식 데이터

---

## 📞 연락처

프로젝트 링크: [https://github.com/yourusername/StockVibe](https://github.com/yourusername/StockVibe)

---

## 🎓 학습 자료

- [Streamlit 문서](https://docs.streamlit.io/)
- [Plotly 가이드](https://plotly.com/python/)
- [기술적 분석 입문](https://www.investopedia.com/technical-analysis-4689657)
- [백테스팅 전략](https://www.quantstart.com/articles/Backtesting-Strategies-with-Python/)

---

**Made with ❤️ by StockVibe Team**

*투자는 본인의 판단과 책임 하에 이루어져야 하며, 본 도구는 참고 자료로만 활용해야 합니다.*" 
