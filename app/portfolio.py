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
import numpy as np

class PortfolioManager:
    """포트폴리오 관리 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.portfolio_file = os.path.join(data_dir, "portfolio.json")
        self.watchlist_file = os.path.join(data_dir, "watchlist.json")
        self.asset_history_file = os.path.join(data_dir, "asset_history.json")
        self.cash_file = os.path.join(data_dir, "cash.json")
        
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
    
    def load_cash(self) -> float:
        """현금 예수금 조회"""
        if os.path.exists(self.cash_file):
            with open(self.cash_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('amount', 0.0)
        return 0.0
    
    def save_cash(self, amount: float) -> bool:
        """현금 예수금 저장"""
        cash_data = {
            'amount': amount,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.cash_file, 'w', encoding='utf-8') as f:
            json.dump(cash_data, f, ensure_ascii=False, indent=2)
        return True
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """포트폴리오 현재 가치 계산"""
        portfolio = self.load_portfolio()
        
        if not portfolio:
            return pd.DataFrame()
        
        data = []
        for code, info in portfolio.items():
            current_price = current_prices.get(code, 0)
            quantity = info['quantity']
            avg_price = info['avg_price']
            
            # 금액 계산
            purchase_value = quantity * avg_price  # 매입금액
            current_value = quantity * current_price  # 평가금액
            
            # 손익 계산
            profit = current_value - purchase_value
            profit_rate = (profit / purchase_value * 100) if purchase_value > 0 else 0
            
            data.append({
                '종목코드': code,
                '종목명': info['name'],
                '보유수량': quantity,
                '평균단가': avg_price,
                '현재가': current_price,
                '매입금액': purchase_value,
                '평가금액': current_value,
                '평가손익': profit,
                '수익률': profit_rate,
                '매수일': info['purchase_date']
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('평가손익', ascending=False)
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """포트폴리오 요약 통계 (현금 포함)"""
        df = self.calculate_portfolio_value(current_prices)
        cash = self.load_cash()
        
        if df.empty:
            return {
                'stock_value': 0,
                'total_value': cash,  # 현금만
                'total_profit': 0,
                'total_profit_rate': 0,
                'total_purchase': 0,
                'num_stocks': 0,
                'cash': cash
            }
        
        stock_value = df['평가금액'].sum()
        total_value = stock_value + cash
        
        return {
            'stock_value': stock_value,
            'total_value': total_value,
            'total_profit': df['평가손익'].sum(),
            'total_profit_rate': (df['평가손익'].sum() / df['매입금액'].sum() * 100) if df['매입금액'].sum() > 0 else 0,
            'total_purchase': df['매입금액'].sum(),
            'num_stocks': len(df),
            'cash': cash
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
                           stop_loss_rate: float = -7.0) -> Dict:
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
    
    def load_asset_history(self) -> Dict:
        """일자별 자산현황 히스토리 불러오기"""
        if os.path.exists(self.asset_history_file):
            try:
                with open(self.asset_history_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, ValueError):
                # 손상된 JSON 파일은 무시하고 빈 딕셔너리 반환
                return {}
        return {}
    
    def save_asset_history(self, history: Dict):
        """일자별 자산현황 히스토리 저장"""
        with open(self.asset_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def record_daily_asset(self, stock_value: float, total_value: float, total_profit: float, 
                          purchase_value: float, num_stocks: int, cash: float) -> bool:
        """현재 자산 정보를 일자별로 기록 (현금 포함)"""
        history = self.load_asset_history()
        today = datetime.now().strftime("%Y-%m-%d")
        
        # pandas int64/float64를 Python 기본 타입으로 변환 (JSON 직렬화 가능)
        history[today] = {
            'stock_value': float(stock_value),       # 주식 평가금액
            'cash': float(cash),                     # 현금
            'total_value': float(total_value),       # 총 자산 (주식+현금)
            'total_profit': float(total_profit),
            'purchase_value': float(purchase_value),
            'num_stocks': int(num_stocks),
            'recorded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.save_asset_history(history)
        return True
    
    def get_asset_history_dataframe(self, days: int = 30) -> pd.DataFrame:
        """일자별 자산현황을 DataFrame으로 반환 (현금 포함)
        
        Args:
            days: 조회할 일수 (기본값: 30일)
        
        Returns:
            일자, 주식평가금액, 현금, 총자산, 평가손익, 매입금액, 보유종목수로 구성된 DataFrame
        """
        history = self.load_asset_history()
        
        if not history:
            return pd.DataFrame()
        
        # 날짜순으로 정렬하고 최근 N일만 조회
        sorted_dates = sorted(history.keys(), reverse=True)[:days]
        sorted_dates = sorted(sorted_dates)  # 오름차순으로 정렬
        
        data = []
        for date in sorted_dates:
            record = history[date]
            # 이전 형식과의 호환성을 위해 stock_value가 없으면 total_value 사용
            stock_value = record.get('stock_value', record.get('total_value', 0))
            cash = record.get('cash', 0)
            total_value = record.get('total_value', stock_value + cash)
            
            data.append({
                '날짜': date,
                '주식평가금액': stock_value,
                '현금': cash,
                '총자산': total_value,
                '평가손익': record.get('total_profit', 0),
                '매입금액': record.get('purchase_value', 0),
                '보유종목수': record.get('num_stocks', 0)
            })
        
        df = pd.DataFrame(data)
        
        # 날짜를 datetime으로 변환
        df['날짜'] = pd.to_datetime(df['날짜'])
        
        return df
    
    def simulate_take_profit(self, 
                            price_history: pd.DataFrame,
                            current_price: float,
                            avg_price: float,
                            target_profit_pct: float = 0.05,
                            holding_days: int = 60,
                            num_simulations: int = 1000) -> Dict:
        """
        익절 시뮬레이션: 목표 수익률에 도달할 확률 계산
        
        Args:
            price_history: 'close' 컬럼을 가진 가격 히스토리 DataFrame
            current_price: 현재 가격
            avg_price: 매입 평균가
            target_profit_pct: 목표 수익률 (기본값: 5%)
            holding_days: 보유 기간 (기본값: 60일)
            num_simulations: 시뮬레이션 횟수 (기본값: 1000회)
        
        Returns:
            {
                'win_rate': 승률 (0~1),
                'avg_days_to_profit': 평균 도달 일수,
                'target_profit_price': 목표 익절가,
                'max_drawdown': 최대 낙폭 확률,
                'simulations_hit_target': 목표 달성 시뮬레이션 수
            }
        """
        
        # 가격 히스토리가 부족한 경우
        if price_history.empty or len(price_history) < 30:
            return {
                'win_rate': 0.5,
                'avg_days_to_profit': holding_days,
                'target_profit_price': avg_price * (1 + target_profit_pct),
                'max_drawdown': 0.1,
                'simulations_hit_target': 0,
                'note': '데이터 부족'
            }
        
        # 수익률 계산
        closes = price_history['close'].values
        returns = np.diff(closes) / closes[:-1]
        
        # 일일 수익률의 평균과 표준편차
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            std_return = 0.01  # 변동성이 없는 경우 기본값
        
        # 목표가
        target_price = avg_price * (1 + target_profit_pct)
        
        # 몬테카를로 시뮬레이션
        hit_target = 0
        days_to_target = []
        max_drawdowns = []
        
        np.random.seed(42)
        
        for _ in range(num_simulations):
            # 임의의 수익률로 가격 경로 생성
            price_path = [current_price]
            max_price = current_price
            
            for day in range(holding_days):
                # 정규분포에서 일일 수익률 샘플링
                daily_return = np.random.normal(mean_return, std_return)
                new_price = price_path[-1] * (1 + daily_return)
                price_path.append(new_price)
                
                # 최대가 추적
                if new_price > max_price:
                    max_price = new_price
                
                # 목표가 달성했는지 확인
                if new_price >= target_price and len(days_to_target) < (hit_target + 1):
                    hit_target += 1
                    days_to_target.append(day + 1)
                    break
            
            # 최대 낙폭 계산 (현재가 기준)
            min_price = min(price_path)
            drawdown = (min_price - current_price) / current_price
            max_drawdowns.append(abs(drawdown))
        
        # 결과 집계
        win_rate = hit_target / num_simulations
        avg_days = np.mean(days_to_target) if days_to_target else holding_days
        
        return {
            'win_rate': round(win_rate, 4),
            'avg_days_to_profit': int(avg_days) if days_to_target else 0,
            'target_profit_price': round(target_price, 0),
            'max_drawdown': round(np.mean(max_drawdowns), 4),
            'simulations_hit_target': hit_target,
            'num_simulations': num_simulations
        }
    
    def simulate_portfolio_take_profit(self, 
                                      portfolio_df: pd.DataFrame,
                                      price_histories: Dict[str, pd.DataFrame],
                                      target_profit_pct: float = 0.05,
                                      holding_days: int = 60) -> pd.DataFrame:
        """
        포트폴리오 전체에 대한 익절 시뮬레이션
        
        Args:
            portfolio_df: calculate_portfolio_value()에서 반환된 DataFrame
            price_histories: {종목코드: 가격히스토리 DataFrame} 딕셔너리
            target_profit_pct: 목표 수익률
            holding_days: 보유 기간
        
        Returns:
            시뮬레이션 결과 DataFrame
        """
        if portfolio_df.empty:
            return pd.DataFrame()
        
        # 필요한 컬럼 확인
        required_cols = ['종목코드', '종목명', '현재가', '평균단가', '보유수량']
        for col in required_cols:
            if col not in portfolio_df.columns:
                return pd.DataFrame()
        
        results = []
        
        for idx, row in portfolio_df.iterrows():
            try:
                code = row['종목코드']
                name = row['종목명']
                current_price = row['현재가']
                avg_price = row['평균단가']
                qty = row['보유수량']
                
                # 해당 종목의 가격 히스토리가 있는지 확인
                if code not in price_histories or price_histories[code].empty:
                    results.append({
                        '종목코드': code,
                        '종목명': name,
                        '승률': '0.0%',
                        '목표가': int(avg_price * (1 + target_profit_pct)),
                        '평균도달일': 0,
                        '최대낙폭': '0.0%',
                        '현재가': int(current_price),
                        '보유수량': int(qty),
                        '비고': '데이터 부족'
                    })
                    continue
                
                # 시뮬레이션 실행
                sim_result = self.simulate_take_profit(
                    price_history=price_histories[code],
                    current_price=current_price,
                    avg_price=avg_price,
                    target_profit_pct=target_profit_pct,
                    holding_days=holding_days,
                    num_simulations=1000
                )
                
                results.append({
                    '종목코드': code,
                    '종목명': name,
                    '승률': f"{sim_result['win_rate']*100:.1f}%",
                    '목표가': int(sim_result['target_profit_price']),
                    '평균도달일': sim_result['avg_days_to_profit'],
                    '최대낙폭': f"{sim_result['max_drawdown']*100:.2f}%",
                    '현재가': int(current_price),
                    '보유수량': int(qty)
                })
            except Exception as e:
                # 개별 종목에서 오류 발생 시 스킵
                continue
        
        return pd.DataFrame(results)
