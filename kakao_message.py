import json
import os
import re
import requests
import webbrowser
from datetime import datetime
from typing import Optional
import pandas as pd


class KakaoMessageSender:
    """카카오톡 나에게 보내기 기능을 제공하는 클래스"""
    
    def __init__(self):
        self.config_file = "kakao_config.json"
        self.rest_api_key = self._load_api_key()
        self.redirect_uri = "http://localhost:8501"
        self.token_file = "kakao_token.json"
        self.access_token = None
        self.refresh_token = None
        self._load_tokens()
    
    def _load_api_key(self) -> str:
        """저장된 REST API 키 로드"""
        # 환경 변수에서 먼저 확인
        env_key = os.getenv("KAKAO_REST_API_KEY", "")
        if env_key:
            return env_key
        
        # 파일에서 로드
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("rest_api_key", "")
            except Exception:
                pass
        return ""
    
    def save_api_key(self, api_key: str) -> None:
        """
REST API 키 저장"""
        self.rest_api_key = api_key
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump({
                "rest_api_key": api_key,
                "updated_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def _load_tokens(self) -> None:
        """저장된 토큰 로드"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r", encoding="utf-8") as f:
                    tokens = json.load(f)
                    self.access_token = tokens.get("access_token")
                    self.refresh_token = tokens.get("refresh_token")
            except Exception:
                pass
    
    def _save_tokens(self, access_token: str, refresh_token: str) -> None:
        """토큰 저장"""
        with open(self.token_file, "w", encoding="utf-8") as f:
            json.dump({
                "access_token": access_token,
                "refresh_token": refresh_token,
                "updated_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        self.access_token = access_token
        self.refresh_token = refresh_token
    
    def get_auth_url(self) -> str:
        """카카오 인증 URL 생성"""
        if not self.rest_api_key:
            raise ValueError("KAKAO_REST_API_KEY 환경변수가 설정되지 않았습니다.")
        
        auth_url = (
            f"https://kauth.kakao.com/oauth/authorize?"
            f"client_id={self.rest_api_key}&"
            f"redirect_uri={self.redirect_uri}&"
            f"response_type=code&"
            f"scope=talk_message"
        )
        return auth_url
    
    def get_token(self, code: str) -> tuple[bool, str]:
        """인증 코드로 토큰 발급"""
        url = "https://kauth.kakao.com/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "client_id": self.rest_api_key,
            "redirect_uri": self.redirect_uri,
            "code": code
        }
        
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                tokens = response.json()
                self._save_tokens(
                    tokens["access_token"],
                    tokens.get("refresh_token", "")
                )
                return True, "인증이 완료되었습니다!"
            else:
                error_data = response.json()
                error_msg = error_data.get("error_description", error_data.get("error", "알 수 없는 오류"))
                return False, f"카카오 인증 실패: {error_msg}"
        except Exception as e:
            return False, f"토큰 발급 오류: {str(e)}"
    
    def refresh_access_token(self) -> bool:
        """액세스 토큰 갱신"""
        if not self.refresh_token:
            return False
        
        url = "https://kauth.kakao.com/oauth/token"
        data = {
            "grant_type": "refresh_token",
            "client_id": self.rest_api_key,
            "refresh_token": self.refresh_token
        }
        
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                tokens = response.json()
                self._save_tokens(
                    tokens["access_token"],
                    tokens.get("refresh_token", self.refresh_token)
                )
                return True
            return False
        except Exception:
            return False
    
    def send_message(self, signals_df: pd.DataFrame, selected_date: str) -> tuple[bool, str]:
        """시그널 분석 결과를 카카오톡으로 전송"""
        if not self.access_token:
            return False, "카카오톡 인증이 필요합니다. 먼저 '카카오톡 연동' 버튼을 클릭하세요."
        
        # 메시지 내용 구성
        message_text = self._format_signals(signals_df, selected_date)
        
        url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        template = {
            "object_type": "text",
            "text": message_text,
            "link": {
                "web_url": "https://finance.naver.com",
                "mobile_web_url": "https://finance.naver.com"
            },
            "button_title": " "
        }
        
        data = {
            "template_object": json.dumps(template, ensure_ascii=False)
        }
        
        try:
            response = requests.post(url, headers=headers, data=data)
            
            if response.status_code == 200:
                return True, "카카오톡으로 메시지를 전송했습니다!"
            elif response.status_code == 401:
                # 토큰 만료, 갱신 시도
                if self.refresh_access_token():
                    return self.send_message(signals_df, selected_date)
                return False, "토큰이 만료되었습니다. 다시 인증해주세요."
            else:
                error_msg = response.json().get("msg", "알 수 없는 오류")
                return False, f"메시지 전송 실패: {error_msg}"
                
        except Exception as e:
            return False, f"전송 중 오류 발생: {str(e)}"
    
    def _format_signals(self, df: pd.DataFrame, selected_date: str) -> str:
        """시그널 데이터를 메시지 형식으로 포맷팅"""
        if df.empty:
            return f"감지일자: {selected_date}\n\n조건에 맞는 시그널이 없습니다."
        
        lines = [f"감지일자: {selected_date}", ""]
        
        for idx, row in df.iterrows():
            name = row.get("name", "")
            code = row.get("code", "")
            signal = row.get("signal", "")
            
            if name and signal:
                # HTML 태그 제거
                clean_name = re.sub(r'<[^>]+>', '', str(name))
                lines.append(f"{clean_name} : {signal}")
                
                # 종목 뉴스 링크 추가
                if code:
                    news_link = f"https://finance.naver.com/item/news.naver?code={code}"
                    lines.append(f"{news_link}")
                    lines.append("")  # 종목 간 구분을 위한 빈 줄
        
        return "\n".join(lines)
    
    def is_authenticated(self) -> bool:
        """인증 상태 확인"""
        return bool(self.access_token)
