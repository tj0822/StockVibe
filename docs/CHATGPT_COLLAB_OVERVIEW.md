# StockVibe 협업 요약 (ChatGPT 공유용)

## 1. 프로젝트 개요
- **프로젝트명**: StockVibe Pro
- **목적**: 한국 주식(KOSPI 중심) 데이터 기반으로 시그널 분석, 시뮬레이션/최적화, 종목/섹터 분석, 포트폴리오 관리, AI 투자분석을 통합 제공
- **기술 스택**: Python, Streamlit, Pandas, Plotly, OpenAI API, (선택) GPU 가속

## 2. 실행 구조 (Entry -> Router)
1. `streamlit_app.py`
2. `app/phase4_ui.py`의 `run_phase4_app()`
3. 메인 대시보드 탭은 `app/ui.py`의 `run_app()`로 위임

### 페이지 라우팅
- `main` -> `app/ui.py` (시그널/시뮬레이션/최적화/데이터)
- `analysis` -> `app/ui.py`의 `render_stock_analysis_page(...)`
- `sector-analysis` -> `app/ui.py`의 `render_sector_analysis_page(...)`
- `portfolio` -> `app/phase4_ui.py`의 `page_portfolio()`
- `settings` -> `app/phase4_ui.py`의 `page_settings()`

## 3. 현재 주요 기능
### A. 시그널/전략
- 거래대금 급등, 모멘텀, 변동성 돌파, 평균회귀 신호 생성
- 전략 레지스트리 기반으로 전략 활성/검증/프로덕션 반영
- 프로덕션 전략만 메인 시그널 탭에 반영

**핵심 파일**
- `app/signals.py`
- `app/strategies/base.py`
- `app/strategies/registry.py`
- `app/strategies/registry_store.py`
- `app/strategies/validation.py`
- `app/strategies/impl/turnover_spike.py`

### B. 시뮬레이션/백테스트/최적화
- 전략 기반 백테스트, 수익률 곡선, 거래 기록 분석
- 파라미터 조합 탐색(그리드/랜덤), 병렬 최적화 실행
- Top 후보 파라미터를 시뮬레이션 탭에 적용 가능

**핵심 파일**
- `app/ui.py` (`run_turnover_strategy_backtest`, `render_optimizer_page`)
- `optimizer.py` (`BacktestOptimizer`)
- `app/backtesting.py` (`BacktestingEngine`)
- `app/backtest_analyzer.py`
- `app/yearly_backtest.py`

### C. 종목/섹터/장기 분석
- 종목 상세 분석 페이지(시그널/재무/기술 지표 기반)
- 섹터 분석 페이지
- 뉴스 기반 산업 트렌드 분석
- 중장기 투자 추천 및 포트폴리오 구성

**핵심 파일**
- `app/ui.py` (`render_stock_analysis_page`, `render_sector_analysis_page`)
- `industry_trend_analyzer.py` (`IndustryTrendAnalyzer`)
- `long_term_analyzer.py` (`LongTermAnalyzer`)

### D. AI 투자 분석 (OpenAI)
- 재무 분석, 기술 분석, 종합 추천 수행
- 분석 결과 파일 캐시(기본 7일)로 토큰/비용 절감
- API 에러/타임아웃 처리 포함

**핵심 파일**
- `ai_investment_analyzer.py` (`AIInvestmentAnalyzer`)
- `app/phase4_ui.py` (`page_ai_analysis`)
- 캐시 폴더: `data/ai_analysis_cache/`

### E. 포트폴리오/거래 관리
- 보유종목 추가/삭제/초기화
- 매수/매도 거래 기록 및 거래 이력 보관
- 입력 로그 저장

**핵심 파일**
- `app/portfolio.py` (`PortfolioManager`)
- 데이터 파일: `data/portfolio.json`, `data/trading_history.json`, `data/trading_input_log.json`

### F. 데이터 수집/로딩
- KOSPI 종목/지수/가격/재무 데이터 로딩
- 데이터 파일 최신 갱신 상태 제공
- 뉴스 크롤링, 카카오 알림 연계

**핵심 파일**
- `app/data.py`
- `crawling_kospi.py`
- `naver_news_crawler.py`
- `kakao_message.py`

## 4. 소스 구조 요약
```text
StockVibe/
├─ streamlit_app.py                  # 엔트리 포인트
├─ ai_investment_analyzer.py         # OpenAI 투자 분석
├─ industry_trend_analyzer.py        # 산업 트렌드 분석
├─ long_term_analyzer.py             # 장기 투자 분석
├─ optimizer.py                      # 파라미터 최적화 엔진
├─ stock_ontology.py                 # 뉴스/가격/재무 결합 분석
├─ technical_indicators.py           # 기술 지표 계산
├─ app/
│  ├─ phase4_ui.py                   # 전체 메뉴 라우팅
│  ├─ ui.py                          # 메인 대시보드 핵심 로직
│  ├─ data.py                        # 데이터 로더
│  ├─ data_pipeline.py               # 마스터 DF 구성
│  ├─ signals.py                     # 시그널 엔진
│  ├─ portfolio.py                   # 포트폴리오 관리
│  ├─ backtesting.py                 # 백테스트 유틸
│  ├─ backtest_analyzer.py           # 배치 백테스트 분석
│  ├─ yearly_backtest.py             # 연도별 백테스트
│  ├─ alerts.py                      # 알림/자동 새로고침
│  ├─ comparison.py                  # 종목/섹터 비교
│  ├─ advanced_charts.py             # 고급 차트 분석
│  ├─ export.py                      # 내보내기/리포트
│  ├─ settings.py                    # 사용자 설정
│  ├─ ui_components.py               # 공통 UI 컴포넌트
│  └─ strategies/
│     ├─ base.py                     # 전략 인터페이스
│     ├─ registry.py                 # 전략 디스커버리/조회
│     ├─ registry_store.py           # 전략 상태 저장소
│     ├─ validation.py               # 전략 검증
│     └─ impl/turnover_spike.py      # 기본 구현 전략
└─ data/
   ├─ strategy_registry.json         # 전략 운영 상태
   ├─ portfolio.json                 # 포트폴리오
   ├─ trading_history.json           # 거래 이력 아카이브
   ├─ trading_input_log.json         # 거래 입력 로그
   └─ ai_analysis_cache/             # AI 분석 캐시
```

## 5. 핵심 데이터 흐름
1. 데이터 로딩: `app/data.py`
2. 시그널 생성: `app/signals.py` 또는 `app/strategies/*`
3. 화면 렌더링: `app/ui.py` / `app/phase4_ui.py`
4. 백테스트/최적화: `app/ui.py` + `optimizer.py`
5. 포트폴리오 반영: `app/portfolio.py`
6. AI 분석 요청/캐시: `ai_investment_analyzer.py` -> `data/ai_analysis_cache/`

## 6. ChatGPT에게 전달할 빠른 설명 템플릿
아래 블록을 그대로 붙여넣어 사용하면 됩니다.

```text
이 프로젝트는 Streamlit 기반 StockVibe이며, 실행 흐름은 streamlit_app.py -> app/phase4_ui.py(run_phase4_app)입니다.
메인 대시보드(시그널/시뮬레이션/최적화/데이터)는 app/ui.py(run_app)에서 처리합니다.
시그널 엔진은 app/signals.py와 app/strategies 구조를 사용하고, 운영 전략 상태는 data/strategy_registry.json으로 관리합니다.
포트폴리오/거래 데이터는 data/portfolio.json, data/trading_history.json, data/trading_input_log.json에 저장됩니다.
AI 분석은 ai_investment_analyzer.py(OpenAI API, 7일 캐시) 기반입니다.
```

## 7. 협업 시 우선 확인 파일
- `app/phase4_ui.py`
- `app/ui.py`
- `app/signals.py`
- `app/strategies/impl/turnover_spike.py`
- `optimizer.py`
- `app/portfolio.py`
- `ai_investment_analyzer.py`
- `app/data.py`
