"""
포트폴리오 관리 모듈
- 관심 종목 저장/관리
- 보유 종목 수익률 추적
- 포트폴리오 분석
"""
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import streamlit as st

class PortfolioManager:
    """포트폴리오 관리 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.portfolio_file = os.path.join(data_dir, "portfolio.json")
        self.watchlist_file = os.path.join(data_dir, "watchlist.json")
        
        # 데이터 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        
    def load_portfolio(self) -> Dict:
        """보유 종목 불러오기"""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_portfolio(self, portfolio: Dict):
        """보유 종목 저장"""
        with open(self.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(portfolio, f, ensure_ascii=False, indent=2)
    
    def load_watchlist(self) -> List[str]:
        """관심 종목 불러오기"""
        if os.path.exists(self.watchlist_file):
            with open(self.watchlist_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_watchlist(self, watchlist: List[str]):
        """관심 종목 저장"""
        with open(self.watchlist_file, 'w', encoding='utf-8') as f:
            json.dump(watchlist, f, ensure_ascii=False, indent=2)
    
    def add_to_portfolio(self, code: str, name: str, quantity: int, 
                        avg_price: float, purchase_date: str = None):
        """포트폴리오에 종목 추가"""
        portfolio = self.load_portfolio()
        
        if purchase_date is None:
            purchase_date = datetime.now().strftime("%Y-%m-%d")
        
        if code in portfolio:
            # 기존 종목 업데이트 (평균단가 재계산)
            old_qty = portfolio[code]['quantity']
            old_price = portfolio[code]['avg_price']
            new_qty = old_qty + quantity
            new_avg = (old_qty * old_price + quantity * avg_price) / new_qty
            
            portfolio[code]['quantity'] = new_qty
            portfolio[code]['avg_price'] = new_avg
            portfolio[code]['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            # 새 종목 추가
            portfolio[code] = {
                'name': name,
                'quantity': quantity,
                'avg_price': avg_price,
                'purchase_date': purchase_date,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        self.save_portfolio(portfolio)
        return True
    
    def remove_from_portfolio(self, code: str, quantity: int = None):
        """포트폴리오에서 종목 제거 (일부 또는 전체)"""
        portfolio = self.load_portfolio()
        
        if code not in portfolio:
            return False
        
        if quantity is None or quantity >= portfolio[code]['quantity']:
            # 전체 제거
            del portfolio[code]
        else:
            # 일부 제거
            portfolio[code]['quantity'] -= quantity
            portfolio[code]['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.save_portfolio(portfolio)
        return True
    
    def add_to_watchlist(self, code: str):
        """관심 종목 추가"""
        watchlist = self.load_watchlist()
        if code not in watchlist:
            watchlist.append(code)
            self.save_watchlist(watchlist)
        return True
    
    def remove_from_watchlist(self, code: str):
        """관심 종목 제거"""
        watchlist = self.load_watchlist()
        if code in watchlist:
            watchlist.remove(code)
            self.save_watchlist(watchlist)
        return True
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """포트폴리오 현재 가치 계산 (세금 포함)"""
        portfolio = self.load_portfolio()
        
        if not portfolio:
            return pd.DataFrame()
        
        # 증권거래세율 (매도 시)
        TAX_RATE = 0.0023  # 0.23%
        
        data = []
        for code, info in portfolio.items():
            current_price = current_prices.get(code, 0)
            quantity = info['quantity']
            avg_price = info['avg_price']
            
            purchase_value = quantity * avg_price
            current_value = quantity * current_price
            
            # 매도 시 세금 계산
            tax = current_value * TAX_RATE
            current_value_after_tax = current_value - tax
            
            profit = current_value_after_tax - purchase_value
            profit_rate = (profit / purchase_value * 100) if purchase_value > 0 else 0
            
            data.append({
                '종목코드': code,
                '종목명': info['name'],
                '보유수량': quantity,
                '평균단가': avg_price,
                '현재가': current_price,
                '매입금액': purchase_value,
                '평가금액(세전)': current_value,
                '예상세금': tax,
                '평가금액': current_value_after_tax,
                '평가손익': profit,
                '수익률': profit_rate,
                '매수일': info['purchase_date']
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('평가손익', ascending=False)
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """포트폴리오 요약 통계"""
        df = self.calculate_portfolio_value(current_prices)
        
        if df.empty:
            return {
                'total_value': 0,
                'total_profit': 0,
                'total_profit_rate': 0,
                'total_purchase': 0,
                'total_tax': 0,
                'num_stocks': 0
            }
        
        return {
            'total_value': df['평가금액'].sum(),
            'total_profit': df['평가손익'].sum(),
            'total_profit_rate': (df['평가손익'].sum() / df['매입금액'].sum() * 100) if df['매입금액'].sum() > 0 else 0,
            'total_purchase': df['매입금액'].sum(),
            'total_tax': df['예상세금'].sum(),
            'num_stocks': len(df)
        }
    
    def get_sector_allocation(self, sector_info: Dict[str, str]) -> pd.DataFrame:
        """섹터별 비중 계산"""
        portfolio = self.load_portfolio()
        
        if not portfolio or not sector_info:
            return pd.DataFrame()
        
        sector_data = {}
        for code, info in portfolio.items():
            sector = sector_info.get(code, '기타')
            value = info['quantity'] * info['avg_price']
            
            if sector in sector_data:
                sector_data[sector] += value
            else:
                sector_data[sector] = value
        
        df = pd.DataFrame(list(sector_data.items()), columns=['섹터', '금액'])
        df['비중(%)'] = df['금액'] / df['금액'].sum() * 100
        return df.sort_values('비중(%)', ascending=False)
    
    def calculate_risk_metrics(self, price_history: Dict[str, pd.DataFrame]) -> Dict:
        """포트폴리오 리스크 지표 계산"""
        portfolio = self.load_portfolio()
        
        if not portfolio:
            return {}
        
        # 각 종목의 수익률 변동성 계산
        volatilities = []
        weights = []
        
        for code, info in portfolio.items():
            if code in price_history:
                df = price_history[code]
                if len(df) > 1:
                    returns = df['Close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5)  # 연간 변동성
                    volatilities.append(volatility)
                    weights.append(info['quantity'] * info['avg_price'])
        
        if not volatilities:
            return {}
        
        total_weight = sum(weights)
        weighted_volatility = sum(v * w / total_weight for v, w in zip(volatilities, weights))
        
        return {
            'portfolio_volatility': weighted_volatility,
            'avg_volatility': sum(volatilities) / len(volatilities),
            'max_volatility': max(volatilities),
            'min_volatility': min(volatilities)
        }
    
    def analyze_sell_timing(self, code: str, current_price: float, 
                           price_history: pd.DataFrame = None,
                           stop_loss_rate: float = -5.0) -> Dict:
        """매도 타이밍 분석
        
        Args:
            code: 종목코드
            current_price: 현재가
            price_history: 가격 이력 (선택사항)
            stop_loss_rate: 손절 기준 (기본값 -5%)
        
        Returns:
            매도 타이밍 분석 결과
        """
        portfolio = self.load_portfolio()
        
        if code not in portfolio:
            return {'error': '보유하지 않은 종목입니다'}
        
        info = portfolio[code]
        avg_price = info['avg_price']
        quantity = info['quantity']
        
        # 수익률 계산
        profit_rate = ((current_price - avg_price) / avg_price) * 100
        profit_amount = (current_price - avg_price) * quantity
        
        # 매도 신호 판단
        signals = []
        recommendation = "보유"
        
        # 손절 기준 체크
        if profit_rate <= stop_loss_rate:
            signals.append(f"⚠️ 손절 기준 {stop_loss_rate}% 도달 - 손절 검토")
            recommendation = "손절 검토"
        
        # 큰 손실 (손절 기준의 2배)
        if profit_rate <= stop_loss_rate * 2:
            signals.append(f"🚨 큰 손실 {stop_loss_rate * 2}% - 즉시 손절 권장")
            recommendation = "즉시 손절"
        
        # 5. 가격 이력이 있는 경우 기술적 분석
        if price_history is not None and len(price_history) > 20:
            # 이동평균 계산
            ma5 = price_history['Close'].rolling(window=5).mean().iloc[-1]
            ma20 = price_history['Close'].rolling(window=20).mean().iloc[-1]
            
            # 데드크로스 감지
            if ma5 < ma20 and profit_rate > 0:
                signals.append("📉 데드크로스 발생 - 하락 추세 시작")
                if recommendation == "보유":
                    recommendation = "부분 매도 고려"
            
            # 골든크로스 감지
            if ma5 > ma20 and profit_rate < 0:
                signals.append("📈 골든크로스 발생 - 반등 기대")
                if recommendation == "손절 검토":
                    recommendation = "추가 관찰"
        
        return {
            'code': code,
            'name': info['name'],
            'avg_price': avg_price,
            'current_price': current_price,
            'quantity': quantity,
            'profit_rate': profit_rate,
            'profit_amount': profit_amount,
            'total_value': current_price * quantity,
            'signals': signals if signals else ["📊 정상 범위 - 보유 유지"],
            'recommendation': recommendation,
            'hold_days': self._calculate_hold_days(info['purchase_date'])
        }
    
    def _calculate_hold_days(self, purchase_date: str) -> int:
        """보유 일수 계산"""
        from datetime import datetime
        try:
            purchase = datetime.strptime(purchase_date, "%Y-%m-%d")
            today = datetime.now()
            return (today - purchase).days
        except:
            return 0
    
    def get_all_sell_signals(self, current_prices: Dict[str, float], 
                            price_histories: Dict[str, pd.DataFrame] = None,
                            stop_loss_rate: float = -5.0) -> List[Dict]:
        """모든 보유 종목의 매도 신호 확인
        
        Args:
            current_prices: {code: current_price}
            price_histories: {code: price_df} (선택사항)
            stop_loss_rate: 손절 기준 (기본값 -5%)
        
        Returns:
            매도 신호가 있는 종목 리스트
        """
        portfolio = self.load_portfolio()
        sell_signals = []
        
        for code in portfolio.keys():
            if code in current_prices:
                price_hist = price_histories.get(code) if price_histories else None
                analysis = self.analyze_sell_timing(code, current_prices[code], price_hist, stop_loss_rate)
                
                # 매도 신호가 있는 경우만
                if analysis['recommendation'] not in ["보유", "추가 관찰"]:
                    sell_signals.append(analysis)
        
        return sell_signals
