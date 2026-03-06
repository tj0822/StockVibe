"""
개인화 설정 모듈
- 사용자 설정 저장/불러오기
- 테마, 기본값 관리
"""
import json
import os
from typing import Dict, Any
from datetime import datetime

class UserSettings:
    """사용자 설정 관리"""
    
    DEFAULT_SETTINGS = {
        # 화면 설정
        'theme': 'light',  # 'light' or 'dark'
        'chart_height': 600,
        'show_volume': True,
        'show_indicators': True,
        
        # 기본값
        'default_period': '1y',  # 기본 조회 기간
        'default_chart_type': 'candle',  # 'candle' or 'line'
        'auto_expand_top': True,  # 1위 종목 자동 펼치기
        
        # 새로고침
        'auto_refresh': False,
        'refresh_interval': 300,  # 초
        
        # 알림
        'enable_alerts': True,
        'alert_sound': False,
        
        # 분석
        'show_ai_prediction': True,
        'show_technical': True,
        'show_news': True,
        'show_financial': True,
        
        # 즐겨찾기
        'favorites': [],  # 종목코드 리스트
        
        # 표시 설정
        'num_news': 5,  # 표시할 뉴스 개수
        'num_kospi': 10,  # 표시할 코스피 종목 수
        
        # 고급
        'enable_caching': True,
        'show_debug_info': False,
        
        # 백테스팅
        'backtest_period': '1y',
        'backtest_strategy': 'golden_cross',
        
        # 내보내기
        'export_format': 'excel',  # 'excel' or 'csv'
        'include_charts': True,

        # 포트폴리오 정책 임계값
        'concentration_warning_pct': 35.0,
        'concentration_critical_pct': 50.0,
        'compliant_score_threshold': 70.0,
        'watch_score_threshold': 60.0,
    }
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.settings_file = os.path.join(data_dir, "user_settings.json")
        
        os.makedirs(data_dir, exist_ok=True)
        
        # 설정 파일이 없으면 기본값으로 생성
        if not os.path.exists(self.settings_file):
            self.save_settings(self.DEFAULT_SETTINGS.copy())
    
    def load_settings(self) -> Dict[str, Any]:
        """설정 불러오기"""
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # 기본값과 병합 (새 설정 항목 추가 대비)
            merged = self.DEFAULT_SETTINGS.copy()
            merged.update(settings)
            
            return merged
        except Exception as e:
            print(f"설정 파일 로드 실패: {e}")
            return self.DEFAULT_SETTINGS.copy()
    
    def save_settings(self, settings: Dict[str, Any]):
        """설정 저장"""
        settings['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """특정 설정값 가져오기"""
        settings = self.load_settings()
        return settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """특정 설정값 변경"""
        settings = self.load_settings()
        settings[key] = value
        self.save_settings(settings)
    
    def reset_to_default(self):
        """기본값으로 초기화"""
        self.save_settings(self.DEFAULT_SETTINGS.copy())
    
    def add_favorite(self, code: str):
        """즐겨찾기 추가"""
        settings = self.load_settings()
        favorites = settings.get('favorites', [])
        
        if code not in favorites:
            favorites.append(code)
            settings['favorites'] = favorites
            self.save_settings(settings)
    
    def remove_favorite(self, code: str):
        """즐겨찾기 제거"""
        settings = self.load_settings()
        favorites = settings.get('favorites', [])
        
        if code in favorites:
            favorites.remove(code)
            settings['favorites'] = favorites
            self.save_settings(settings)
    
    def get_favorites(self) -> list:
        """즐겨찾기 목록"""
        return self.get('favorites', [])
    
    def export_settings(self, filepath: str):
        """설정 내보내기"""
        settings = self.load_settings()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    
    def import_settings(self, filepath: str):
        """설정 가져오기"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            self.save_settings(settings)
            return True
        except Exception as e:
            print(f"설정 가져오기 실패: {e}")
            return False


class ThemeManager:
    """테마 관리"""
    
    THEMES = {
        'light': {
            'background': '#FFFFFF',
            'text': '#000000',
            'primary': '#FF6B6B',
            'secondary': '#4ECDC4',
            'success': '#95E1D3',
            'warning': '#F38181',
            'info': '#AA96DA',
        },
        'dark': {
            'background': '#1E1E1E',
            'text': '#FFFFFF',
            'primary': '#FF6B6B',
            'secondary': '#4ECDC4',
            'success': '#95E1D3',
            'warning': '#F38181',
            'info': '#AA96DA',
        },
        'blue': {
            'background': '#F0F8FF',
            'text': '#000080',
            'primary': '#4169E1',
            'secondary': '#87CEEB',
            'success': '#90EE90',
            'warning': '#FFD700',
            'info': '#ADD8E6',
        },
    }
    
    @staticmethod
    def get_theme(theme_name: str = 'light') -> Dict[str, str]:
        """테마 가져오기"""
        return ThemeManager.THEMES.get(theme_name, ThemeManager.THEMES['light'])
    
    @staticmethod
    def get_plotly_template(theme_name: str = 'light') -> str:
        """Plotly 차트 템플릿"""
        if theme_name == 'dark':
            return 'plotly_dark'
        else:
            return 'plotly_white'


class DisplaySettings:
    """화면 표시 설정"""
    
    CHART_PERIODS = {
        '1개월': '1mo',
        '3개월': '3mo',
        '6개월': '6mo',
        '1년': '1y',
        '3년': '3y',
        '5년': '5y',
        '전체': 'max'
    }
    
    CHART_TYPES = {
        '캔들스틱': 'candle',
        '라인': 'line',
        '영역': 'area',
        'OHLC': 'ohlc'
    }
    
    INDICATORS = [
        'RSI',
        'MACD',
        '볼린저밴드',
        '스토캐스틱',
        'OBV',
        '이동평균선'
    ]
    
    @staticmethod
    def get_period_days(period: str) -> int:
        """기간을 일수로 변환"""
        period_map = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '3y': 1095,
            '5y': 1825,
            'max': 3650
        }
        return period_map.get(period, 365)
