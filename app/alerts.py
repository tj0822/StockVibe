"""
실시간 모니터링 & 알림 모듈
- 자동 새로고침
- 가격 알림
- 뉴스 알림
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import streamlit as st

class AlertManager:
    """알림 관리 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.alerts_file = os.path.join(data_dir, "alerts.json")
        
        os.makedirs(data_dir, exist_ok=True)
    
    def load_alerts(self) -> Dict:
        """알림 설정 불러오기"""
        if os.path.exists(self.alerts_file):
            with open(self.alerts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_alerts(self, alerts: Dict):
        """알림 설정 저장"""
        with open(self.alerts_file, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, ensure_ascii=False, indent=2)
    
    def add_price_alert(self, code: str, name: str, alert_type: str, 
                       target_price: float, current_price: float):
        """가격 알림 추가
        
        Args:
            code: 종목코드
            name: 종목명
            alert_type: 'above' (이상) 또는 'below' (이하)
            target_price: 목표가
            current_price: 현재가
        """
        alerts = self.load_alerts()
        
        alert_id = f"{code}_{alert_type}_{target_price}"
        
        alerts[alert_id] = {
            'code': code,
            'name': name,
            'type': 'price',
            'alert_type': alert_type,
            'target_price': target_price,
            'current_price': current_price,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'triggered': False
        }
        
        self.save_alerts(alerts)
        return alert_id
    
    def add_change_alert(self, code: str, name: str, change_type: str, 
                        threshold: float):
        """등락률 알림 추가
        
        Args:
            code: 종목코드
            name: 종목명
            change_type: 'surge' (급등) 또는 'plunge' (급락)
            threshold: 등락 기준 (%)
        """
        alerts = self.load_alerts()
        
        alert_id = f"{code}_{change_type}_{threshold}"
        
        alerts[alert_id] = {
            'code': code,
            'name': name,
            'type': 'change',
            'change_type': change_type,
            'threshold': threshold,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'triggered': False
        }
        
        self.save_alerts(alerts)
        return alert_id
    
    def add_news_alert(self, code: str, name: str, keywords: List[str]):
        """뉴스 키워드 알림 추가"""
        alerts = self.load_alerts()
        
        alert_id = f"{code}_news_{'_'.join(keywords[:2])}"
        
        alerts[alert_id] = {
            'code': code,
            'name': name,
            'type': 'news',
            'keywords': keywords,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'triggered': False
        }
        
        self.save_alerts(alerts)
        return alert_id
    
    def remove_alert(self, alert_id: str):
        """알림 제거"""
        alerts = self.load_alerts()
        if alert_id in alerts:
            del alerts[alert_id]
            self.save_alerts(alerts)
        return True
    
    def check_price_alerts(self, code: str, current_price: float) -> List[Dict]:
        """가격 알림 체크"""
        alerts = self.load_alerts()
        triggered = []
        
        for alert_id, alert in alerts.items():
            if alert['code'] == code and alert['type'] == 'price' and not alert['triggered']:
                if alert['alert_type'] == 'above' and current_price >= alert['target_price']:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alert['triggered_price'] = current_price
                    triggered.append(alert)
                elif alert['alert_type'] == 'below' and current_price <= alert['target_price']:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alert['triggered_price'] = current_price
                    triggered.append(alert)
        
        if triggered:
            self.save_alerts(alerts)
        
        return triggered
    
    def check_change_alerts(self, code: str, change_rate: float) -> List[Dict]:
        """등락률 알림 체크"""
        alerts = self.load_alerts()
        triggered = []
        
        for alert_id, alert in alerts.items():
            if alert['code'] == code and alert['type'] == 'change' and not alert['triggered']:
                if alert['change_type'] == 'surge' and change_rate >= alert['threshold']:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alert['triggered_change'] = change_rate
                    triggered.append(alert)
                elif alert['change_type'] == 'plunge' and change_rate <= -alert['threshold']:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alert['triggered_change'] = change_rate
                    triggered.append(alert)
        
        if triggered:
            self.save_alerts(alerts)
        
        return triggered
    
    def check_news_alerts(self, code: str, news_titles: List[str]) -> List[Dict]:
        """뉴스 키워드 알림 체크"""
        alerts = self.load_alerts()
        triggered = []
        
        for alert_id, alert in alerts.items():
            if alert['code'] == code and alert['type'] == 'news':
                # 키워드 매칭
                for title in news_titles:
                    for keyword in alert['keywords']:
                        if keyword in title:
                            alert['matched_keyword'] = keyword
                            alert['matched_title'] = title
                            alert['matched_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            triggered.append(alert.copy())
                            break
        
        return triggered
    
    def get_active_alerts(self, code: str = None) -> List[Dict]:
        """활성 알림 목록"""
        alerts = self.load_alerts()
        
        active = []
        for alert_id, alert in alerts.items():
            if not alert['triggered']:
                if code is None or alert['code'] == code:
                    active.append(alert)
        
        return active
    
    def get_triggered_alerts(self, code: str = None, limit: int = 10) -> List[Dict]:
        """최근 발동된 알림"""
        alerts = self.load_alerts()
        
        triggered = []
        for alert_id, alert in alerts.items():
            if alert['triggered']:
                if code is None or alert['code'] == code:
                    triggered.append(alert)
        
        # 발동 시간 기준 정렬
        triggered.sort(key=lambda x: x.get('triggered_at', ''), reverse=True)
        
        return triggered[:limit]


class AutoRefreshManager:
    """자동 새로고침 관리"""
    
    @staticmethod
    def get_refresh_intervals() -> Dict[str, int]:
        """새로고침 간격 옵션 (초)"""
        return {
            '사용 안 함': 0,
            '30초': 30,
            '1분': 60,
            '5분': 300,
            '10분': 600,
            '30분': 1800
        }
    
    @staticmethod
    def should_refresh(last_refresh: datetime, interval: int) -> bool:
        """새로고침 필요 여부"""
        if interval == 0:
            return False
        
        elapsed = (datetime.now() - last_refresh).total_seconds()
        return elapsed >= interval
