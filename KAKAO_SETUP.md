# 카카오톡 메시지 전송 기능 사용 가이드

StockVibe 시그널 분석 결과를 카카오톡으로 전송할 수 있습니다.

## 설정 방법

### 1. 카카오 개발자 계정 설정

1. **카카오 개발자 센터 접속**
   - https://developers.kakao.com 접속
   - 카카오 계정으로 로그인

2. **애플리케이션 추가**
   - '내 애플리케이션' 메뉴 선택
   - '애플리케이션 추가하기' 클릭
   - 앱 이름 입력 (예: StockVibe)
   - 회사명 입력 (선택사항)

3. **REST API 키 확인**
   - 생성한 앱 선택
   - '앱 키' 탭에서 'REST API 키' 복사
   - 이 키를 StockVibe 앱의 입력창에 붙여넣기

### 2. 플랫폼 설정

1. **Web 플랫폼 등록**
   - 애플리케이션 설정 > 플랫폼
   - 'Web 플랫폼 등록' 클릭
   - 사이트 도메인: `http://localhost:8501` 입력

### 3. 카카오 로그인 설정

1. **카카오 로그인 활성화**
   - 제품 설정 > 카카오 로그인
   - '활성화 설정' 상태를 ON으로 변경

2. **Redirect URI 설정**
   - 카카오 로그인 > Redirect URI
   - `http://localhost:8501` 등록 (**주의: HTTPS가 아닌 HTTP**)

### 4. 동의항목 설정

1. **권한 설정**
   - 제품 설정 > 카카오 로그인 > 동의항목
   - 필요한 경우 추가 권한 설정

## 사용 방법

### 1. 애플리케이션 실행

```bash
streamlit run streamlit_app.py
```

### 2. 카카오톡 연동

1. 시그널 탭으로 이동
2. 하단의 "카카오톡으로 전송" 섹션에서 REST API 키 입력
3. "🔐 카카오톡 연동하기" 버튼 클릭
4. "카카오 인증하기" 버튼을 클릭하면 새 탭에서 카카오 로그인 페이지가 열림
5. 카카오 계정 로그인 및 동의
6. **인증 완료 후 자동으로 Streamlit 앱으로 리다이렉트되며 연동이 완료됩니다**

### 3. 메시지 전송

1. 원하는 날짜의 시그널 확인
2. "📤 카카오톡으로 전송" 버튼 클릭
3. 카카오톡 나에게 보내기로 시그널 분석 결과 수신

## 전송되는 메시지 형식

```
📊 StockVibe 시그널 분석
📅 날짜: 2026-02-04
📈 종목 수: 5개

🔴 삼성전자 (005930)
  종가: 75,000원 | BUY
  거래량 급증: +150%

🔴 SK하이닉스 (000660)
  종가: 125,000원 | BUY
  거래량 급증: +230%

... 외 3개 종목
```

## 환경 변수 설정 (선택사항)

더 편리한 사용을 위해 환경 변수로 API 키를 설정할 수 있습니다:

### Windows (PowerShell)
```powershell
$env:KAKAO_REST_API_KEY="your_rest_api_key_here"
```

### Windows (영구 설정)
1. 시스템 환경 변수 편집
2. 새 환경 변수 추가:
   - 변수 이름: `KAKAO_REST_API_KEY`
   - 변수 값: REST API 키

### Linux/Mac
```bash
export KAKAO_REST_API_KEY="your_rest_api_key_here"
```

## 주의사항

- **개인용 기능**: 나에게 보내기는 본인 계정으로만 메시지 전송 가능
- **토큰 만료**: 인증 토큰은 일정 시간 후 만료되며, 자동으로 갱신 시도
- **메시지 제한**: 카카오 API 정책에 따른 일일 메시지 전송 제한 존재
- **보안**: REST API 키는 공개하지 마세요

## 문제 해결

### "KAKAO_REST_API_KEY 환경변수가 설정되지 않았습니다"
- StockVibe 앱에서 직접 REST API 키를 입력하세요

### "인증에 실패했습니다"
- Redirect URI가 정확히 `http://localhost:8501`로 설정되었는지 확인 (**HTTPS가 아닌 HTTP**)
- 카카오 로그인이 활성화되어 있는지 확인
- Streamlit 앱이 실행 중인지 확인 (http://localhost:8501에서 접속 가능해야 함)

### "토큰이 만료되었습니다"
- "카카오톡 연동하기"를 다시 클릭하여 재인증

### "메시지 전송 실패"
- 카카오 개발자 센터에서 앱 상태 확인
- 일일 메시지 전송 한도 확인
- 네트워크 연결 상태 확인

## API 참고 문서

- [카카오 개발자 문서](https://developers.kakao.com)
- [카카오 로그인 가이드](https://developers.kakao.com/docs/latest/ko/kakaologin/common)
- [메시지 API](https://developers.kakao.com/docs/latest/ko/message/rest-api)
