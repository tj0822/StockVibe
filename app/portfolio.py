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
    DEFAULT_INITIAL_CASH = 50_000_000.0
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.portfolio_file = os.path.join(data_dir, "portfolio.json")
        self.watchlist_file = os.path.join(data_dir, "watchlist.json")
        self.asset_history_file = os.path.join(data_dir, "asset_history.json")
        self.cash_file = os.path.join(data_dir, "cash.json")
        self.user_settings_file = os.path.join(data_dir, "user_settings.json")
        self.trading_history_file = os.path.join(data_dir, "trading_history.json")  # 거래 이력 보관
        
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
    
    def load_trading_history(self) -> Dict:
        """거래 이력 불러오기 (삭제된 종목 포함)"""
        if os.path.exists(self.trading_history_file):
            with open(self.trading_history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_trading_history(self, trading_history: Dict):
        """거래 이력 저장 (삭제된 종목의 거래도 보관)"""
        with open(self.trading_history_file, 'w', encoding='utf-8') as f:
            json.dump(trading_history, f, ensure_ascii=False, indent=2)
    
    def save_trading_input_log(self, input_text: str, parsed_trades: List[Dict], 
                              apply_results: Dict[str, str]):
        """
        사용자가 입력한 거래 이력을 로그 파일에 저장
        
        Args:
            input_text: 원본 입력 텍스트
            parsed_trades: 파싱된 거래 목록
            apply_results: 적용 결과 메시지
        """
        log_file = os.path.join(self.data_dir, 'trading_input_log.json')
        
        # 기존 로그 로드
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # 새 로그 엔트리 생성
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_text': input_text,
            'parsed_trades': parsed_trades,
            'apply_results': apply_results,
            'total_trades': len(parsed_trades)
        }
        
        logs.append(log_entry)
        
        # 로그 저장 (최근 100개만 유지)
        if len(logs) > 100:
            logs = logs[-100:]
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        print(f"[DEBUG] 거래 입력 로그 저장됨: {log_entry['timestamp']}")
    
    def backup_trading_history_on_delete(self, code: str, stock_info: Dict):
        """
        종목 삭제 시 거래 이력을 별도 파일에 백업
        
        Args:
            code: 종목 코드
            stock_info: 포트폴리오에서 삭제할 종목의 정보
        """
        # 기존 거래 이력 로드
        trading_history = self.load_trading_history()
        
        # 해당 종목의 거래 이력을 거래 이력 파일에 추가
        if code not in trading_history:
            trading_history[code] = {
                'name': stock_info.get('name', ''),
                'trades': stock_info.get('trades', []),
                'deleted_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            # 이미 있으면 거래 이력 병합
            existing_trades = trading_history[code].get('trades', [])
            new_trades = stock_info.get('trades', [])
            
            # 중복 제거
            existing_dates = {t.get('date') for t in existing_trades}
            for trade in new_trades:
                if trade.get('date') not in existing_dates:
                    existing_trades.append(trade)
            
            trading_history[code]['trades'] = existing_trades
            trading_history[code]['deleted_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 거래 이력 저장
        self.save_trading_history(trading_history)
        print(f"[DEBUG] {code}({stock_info.get('name')})의 거래 이력을 보관했습니다.")
    
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
    
    def remove_from_portfolio(self, code: str) -> bool:
        """
        포트폴리오에서 종목 제거 (거래 이력은 별도 파일에 보관)
        """
        portfolio = self.load_portfolio()
        
        if code in portfolio:
            stock_info = portfolio[code]
            
            # 삭제 전에 거래 이력 백업
            self.backup_trading_history_on_delete(code, stock_info)
            
            # 포트폴리오에서 삭제
            del portfolio[code]
            self.save_portfolio(portfolio)
            print(f"[DEBUG] {code}를 포트폴리오에서 제거했습니다.")
            return True
        return False
    
    def clear_portfolio(self) -> bool:
        """포트폴리오 전체 비우기 (모든 종목만 삭제, 거래 이력은 유지)"""
        empty_portfolio = {}
        self.save_portfolio(empty_portfolio)
        return True
    
    def full_reset(self) -> bool:
        """완전 초기화: 포트폴리오 + 거래 이력 + 입력 로그 모두 삭제"""
        try:
            # 1. 포트폴리오 삭제
            empty_portfolio = {}
            self.save_portfolio(empty_portfolio)
            
            # 2. 거래 이력(아카이브) 삭제
            empty_history = {}
            self.save_trading_history(empty_history)
            
            # 3. 입력 로그 삭제
            log_file = os.path.join(self.data_dir, 'trading_input_log.json')
            if os.path.exists(log_file):
                os.remove(log_file)

            # 4. 예수금 초기화
            self.save_cash(self._get_initial_cash())
            
            print("[DEBUG] 완전 초기화 완료: 포트폴리오, 거래 이력, 입력 로그 모두 삭제됨")
            return True
        except Exception as e:
            print(f"[DEBUG] 완전 초기화 오류: {str(e)}")
            return False
    
    def add_trade(self, code: str, name: str, trade_type: str, quantity: int, 
                 price: float, trade_date: str = None) -> Dict[str, str]:
        """
        거래 기록 추가 (매수/매도)
        
        Args:
            code: 종목 코드
            name: 종목명
            trade_type: 'BUY' 또는 'SELL'
            quantity: 수량
            price: 가격 (매수/매도 단가)
            trade_date: 거래 날짜 (기본값: 현재시간)
        
        Returns:
            {'status': 'success'|'error', 'message': '...'} 또는 {'status': 'error', 'message': '...'}
        """
        # 입력값 유효성 검증
        if not name or not name.strip():
            return {'status': 'error', 'message': '[ERROR] 종목명이 없습니다.'}
        
        if quantity <= 0:
            return {'status': 'error', 'message': f"[ERROR] {name} - 수량이 0 이상이어야 합니다. (입력: {quantity})"}
        
        if price <= 0:
            return {'status': 'error', 'message': f"[ERROR] {name} - 가격이 0 이상이어야 합니다. (입력: {price})"}
        
        trade_type = trade_type.upper()
        if trade_type not in ['BUY', 'SELL']:
            return {'status': 'error', 'message': f"[ERROR] {name} - 거래 타입은 BUY 또는 SELL이어야 합니다. (입력: {trade_type})"}
        
        portfolio = self.load_portfolio()
        
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 종목이 포트폴리오에 없으면 추가
        if code not in portfolio:
            portfolio[code] = {
                'name': name,
                'quantity': 0,
                'avg_price': 0,
                'trades': [],
                'purchase_date': trade_date,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # SELL 거래의 경우 사전 검증
        if trade_type == 'SELL':
            current_qty = portfolio[code].get('quantity', 0)
            if current_qty < quantity:
                return {
                    'status': 'error',
                    'message': f"[ERROR] {name} - 매도 불가: 보유수량({current_qty}주) < 매도수량({quantity}주)"
                }
        
        # 거래 기록 추가
        trade_record = {
            'type': trade_type,
            'quantity': int(quantity),
            'price': float(price),
            'date': trade_date,
            'return_pct': 0 if trade_type == 'BUY' else None  # 매도 시 계산됨
        }
        
        if 'trades' not in portfolio[code]:
            portfolio[code]['trades'] = []
        
        # BUY 거래 처리
        if trade_type == 'BUY':
            old_qty = portfolio[code].get('quantity', 0)
            old_price = portfolio[code].get('avg_price', 0)
            
            new_qty = old_qty + quantity
            # 평균 단가 계산
            if new_qty > 0:
                portfolio[code]['avg_price'] = (old_qty * old_price + quantity * price) / new_qty
            else:
                portfolio[code]['avg_price'] = price
            
            portfolio[code]['quantity'] = new_qty

            # 현금 반영: 매수 금액 차감
            current_cash = float(self.load_cash())
            current_cash -= float(quantity) * float(price)
            self.save_cash(current_cash)
            
            # 거래 기록에 수익률 저장 (BUY는 항상 0)
            trade_record['return_pct'] = 0
            
            message = f"[OK] {name} 매수 {quantity}주 @ {price:,}원 (평균단가: {portfolio[code]['avg_price']:,.0f}원, 총 {new_qty}주)"
        
        # SELL 거래 처리
        else:  # SELL
            current_qty = portfolio[code].get('quantity', 0)
            avg_buy_price = portfolio[code].get('avg_price', 0)
            
            # 수익률 계산
            if avg_buy_price > 0:
                return_pct = ((price - avg_buy_price) / avg_buy_price) * 100
                trade_record['return_pct'] = round(return_pct, 2)
            else:
                return_pct = 0
                trade_record['return_pct'] = 0
            
            # SELL 기록에 평균 매입가 저장
            trade_record['avg_buy_price'] = round(avg_buy_price, 0)
            
            # 수량 감소
            new_qty = max(0, current_qty - quantity)
            portfolio[code]['quantity'] = new_qty

            # 현금 반영: 매도 금액 증가
            current_cash = float(self.load_cash())
            current_cash += float(quantity) * float(price)
            self.save_cash(current_cash)
            
            # 손익 계산
            profit_loss_total = quantity * (price - avg_buy_price)
            
            # 수익 / 손실 표현
            if return_pct > 0:
                result_emoji = "[수익]"
            elif return_pct < 0:
                result_emoji = "[손실]"
            else:
                result_emoji = "[분기]"
            
            # 상세 메시지 생성
            detail_msg = f"(매입가: {avg_buy_price:,.0f}원, 매도가: {price:,}원, 수익률: {return_pct:+.2f}%)"
            profit_loss_str = f"{profit_loss_total:,.0f}원"
            
            if new_qty == 0:
                message = f"[OK] {name} 매도 {quantity}주 {result_emoji} {detail_msg} 손익: {profit_loss_str} (완전 매도 - 포트폴리오에서 제거됨)"
            else:
                message = f"[OK] {name} 매도 {quantity}주 {result_emoji} {detail_msg} 손익: {profit_loss_str} (남은 {new_qty}주)"
        
        # 거래 기록 추가
        portfolio[code]['trades'].append(trade_record)
        portfolio[code]['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.save_portfolio(portfolio)
        
        # 매도로 인해 수량이 0이 되면 포트폴리오에서 삭제 (거래 이력은 백업됨)
        if trade_type == 'SELL' and new_qty == 0:
            self.remove_from_portfolio(code)
        
        print(f"[DEBUG] 거래 기록 저장: {trade_record}")
        print(f"[DEBUG] {code} 현재 수량: {portfolio[code]['quantity']}, 평균단가: {portfolio[code]['avg_price']}")
        
        return {'status': 'success', 'message': message}
    
    def get_trading_history(self, code: str) -> list:
        """
        종목의 거래 이력 조회
        
        Args:
            code: 종목 코드
        
        Returns:
            거래 이력 리스트 (시간 역순)
        """
        portfolio = self.load_portfolio()
        
        if code not in portfolio:
            return []
        
        trades = portfolio[code].get('trades', [])
        # 최신 거래가 위에 오도록 역순 정렬
        return sorted(trades, key=lambda x: x.get('date', ''), reverse=True)
    
    def calculate_portfolio_stats(self) -> Dict:
        """
        포트폴리오 통계 계산
        
        Returns:
            {
                'total_stocks': 보유 종목 수,
                'realized_return': 실현 수익 (원),
                'realized_return_pct': 실현 수익률 (%),
                'unrealized_value': 미실현 평가액 (원),
                'total_invested': 총 투입액 (원)
            }
        """
        portfolio = self.load_portfolio()
        
        total_realized = 0  # 실현 수익
        total_invested = 0  # 총 투입액
        
        for code, info in portfolio.items():
            trades = info.get('trades', [])
            
            for trade in trades:
                if trade['type'] == 'BUY':
                    total_invested += trade['quantity'] * trade['price']
                elif trade['type'] == 'SELL' and trade.get('return_pct') is not None:
                    # 매도 시 수익 계산
                    sell_price = trade['price']
                    quantity = trade['quantity']
                    return_pct = trade['return_pct']
                    
                    # 매수가 추정 (평균 단가로부터)
                    avg_buy_price = sell_price / (1 + return_pct / 100) if return_pct != 0 else sell_price
                    realized = quantity * (sell_price - avg_buy_price)
                    total_realized += realized
        
        return {
            'total_stocks': len(portfolio),
            'realized_return': total_realized,
            'realized_return_pct': (total_realized / total_invested * 100) if total_invested > 0 else 0,
            'total_invested': total_invested
        }
    
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
        return self._get_initial_cash()

    def _load_cash_info(self) -> Dict:
        """현금 데이터(amount, last_updated) 조회"""
        if os.path.exists(self.cash_file):
            with open(self.cash_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                'amount': float(data.get('amount', 0.0)),
                'last_updated': data.get('last_updated')
            }
        return {
            'amount': float(self._get_initial_cash()),
            'last_updated': None
        }

    def _get_initial_cash(self) -> float:
        """초기 예수금 조회 (user_settings.json -> 기본값)"""
        try:
            if os.path.exists(self.user_settings_file):
                with open(self.user_settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                raw = settings.get('initial_cash')
                if raw is not None:
                    parsed = float(raw)
                    if parsed >= 0:
                        return parsed
        except Exception:
            pass
        return float(self.DEFAULT_INITIAL_CASH)

    def _parse_dt(self, value: str):
        """YYYY-mm-dd HH:MM:SS 또는 YYYY-mm-dd 날짜 문자열을 datetime으로 변환"""
        if not value or not isinstance(value, str):
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _latest_trade_datetime_in_portfolio(self):
        """현재 포트폴리오에 남아있는 종목의 최신 거래 시각"""
        portfolio = self.load_portfolio()
        latest_dt = None
        for _, info in portfolio.items():
            for trade in info.get('trades', []):
                trade_dt = self._parse_dt(trade.get('date'))
                if trade_dt is not None and (latest_dt is None or trade_dt > latest_dt):
                    latest_dt = trade_dt
        return latest_dt

    def _estimate_cash_from_current_portfolio_trades(self) -> float:
        """현재 보유 종목의 거래기록 기준으로 예수금 추정"""
        initial_cash = self._get_initial_cash()
        portfolio = self.load_portfolio()

        net_buy_flow = 0.0
        for _, info in portfolio.items():
            trades = info.get('trades', [])
            if trades:
                for trade in trades:
                    ttype = str(trade.get('type', '')).upper()
                    qty = float(pd.to_numeric(trade.get('quantity', 0), errors='coerce') or 0.0)
                    price = float(pd.to_numeric(trade.get('price', 0), errors='coerce') or 0.0)
                    if qty <= 0 or price <= 0:
                        continue
                    if ttype == 'BUY':
                        net_buy_flow += qty * price
                    elif ttype == 'SELL':
                        net_buy_flow -= qty * price
            else:
                qty = float(pd.to_numeric(info.get('quantity', 0), errors='coerce') or 0.0)
                avg_price = float(pd.to_numeric(info.get('avg_price', 0), errors='coerce') or 0.0)
                if qty > 0 and avg_price > 0:
                    net_buy_flow += qty * avg_price

        return float(initial_cash - net_buy_flow)
    
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
            # 가격 키를 코드 정규화(앞자리 0 포함)로 조회하고, 없으면 평균단가/최근 거래가로 fallback
            code_key = str(code).zfill(6)
            current_price = current_prices.get(code)
            if current_price is None:
                current_price = current_prices.get(code_key)

            quantity_raw = pd.to_numeric(info.get('quantity', 0), errors='coerce')
            quantity = int(quantity_raw) if pd.notna(quantity_raw) else 0
            if quantity <= 0:
                continue
            avg_price = info['avg_price']

            if current_price is None or pd.isna(current_price) or float(current_price) <= 0:
                trades = info.get('trades', [])
                if trades:
                    trades_sorted = sorted(trades, key=lambda x: x.get('date', ''))
                    current_price = trades_sorted[-1].get('price', avg_price)
                else:
                    current_price = avg_price

            current_price = float(current_price)
            
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
    
    def get_holding_summary(self, current_prices: Dict[str, float] = None) -> pd.DataFrame:
        """
        보유 종목의 상세 요약 정보 조회
        
        Args:
            current_prices: {code: price} 현재가 정보 (선택사항)
        
        Returns:
            DataFrame with columns: 종목명, 종목코드, 매수일자, 매수가격, 보유수량, 매수횟수, 현재수익률(%)
        """
        portfolio = self.load_portfolio()
        
        if not portfolio:
            return pd.DataFrame()
        
        summaries = []
        
        for code, info in portfolio.items():
            quantity_raw = pd.to_numeric(info.get('quantity', 0), errors='coerce')
            quantity = int(quantity_raw) if pd.notna(quantity_raw) else 0
            if quantity <= 0:
                # 수량이 0이면 스킵 (완전 매도된 종목)
                continue
            
            name = info['name']
            avg_price = info['avg_price']
            trades = info.get('trades', [])
            
            # 매수 거래만 필터
            buy_trades = [t for t in trades if t.get('type') == 'BUY']
            buy_count = len(buy_trades)
            
            # 첫 번째 매수일자
            if buy_trades:
                # 날짜순으로 정렬
                buy_trades_sorted = sorted(buy_trades, key=lambda x: x.get('date', ''))
                purchase_date = buy_trades_sorted[0]['date'].split(' ')[0]  # 날짜 부분만
            else:
                purchase_date = info.get('purchase_date', 'N/A')
            
            # 현재 수익률 계산
            current_price = None
            
            # 1. 사용자 제공 현재가
            if current_prices and code in current_prices:
                current_price = current_prices[code]
            elif current_prices:
                current_price = current_prices.get(str(code).zfill(6))
            
            # 2. 최근 거래가 사용 (현재가가 없으면 거래 이력의 마지막 가격 사용)
            if current_price is None and trades:
                # 모든 거래를 시간순으로 정렬
                trades_sorted = sorted(trades, key=lambda x: x.get('date', ''))
                # 마지막 거래 가격
                current_price = trades_sorted[-1].get('price', None)
            
            # 수익률 계산
            if current_price is not None and avg_price > 0:
                profit_rate = ((current_price - avg_price) / avg_price) * 100
            else:
                profit_rate = None
            
            summaries.append({
                '종목명': name,
                '종목코드': code,
                '매수일자': purchase_date,
                '평균매수가격': int(avg_price),
                '현재가격': int(current_price) if current_price is not None else None,
                '보유수량': int(quantity),
                '매수횟수': buy_count,
                '현재수익률(%)': round(profit_rate, 2) if profit_rate is not None else 'N/A'
            })
        
        df = pd.DataFrame(summaries)
        
        # 종목명순으로 정렬 (한글/영문 혼합 지원)
        if not df.empty:
            # 한글을 먼저 정렬하고 영문을 뒤에 배치
            def sort_key(name):
                # 첫 문자가 한글인지 확인
                if name and ord(name[0]) >= 0xAC00:  # 한글 범위
                    return (0, name)  # 한글은 앞 (0)
                else:
                    return (1, name)  # 영문은 뒤 (1)
            
            df['_sort_key'] = df['종목명'].apply(sort_key)
            df = df.sort_values('_sort_key').drop('_sort_key', axis=1).reset_index(drop=True)
        
        return df
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """포트폴리오 요약 통계 (현금 제외, 평가금액 기준)"""
        df = self.calculate_portfolio_value(current_prices)
        
        if df.empty:
            return {
                'stock_value': 0,
                'total_value': 0,
                'total_profit': 0,
                'total_profit_rate': 0,
                'total_purchase': 0,
                'num_stocks': 0,
                'cash': 0
            }
        
        stock_value = df['평가금액'].sum()
        total_value = stock_value
        
        return {
            'stock_value': stock_value,
            'total_value': total_value,
            'total_profit': df['평가손익'].sum(),
            'total_profit_rate': (df['평가손익'].sum() / df['매입금액'].sum() * 100) if df['매입금액'].sum() > 0 else 0,
            'total_purchase': df['매입금액'].sum(),
            'num_stocks': int(df['종목코드'].astype(str).nunique()) if '종목코드' in df.columns else len(df),
            'cash': 0
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
    def parse_trading_history_text(self, text: str, kospi_dict: Dict[str, str] = None, default_date: str = None) -> List[Dict]:
        """
        키움증권 거래 알림 텍스트를 파싱하여 거래 기록 추출
        
        지원 형식 1 (최신):
        2024년 2월 7일 오전 10:44
        2024년 2월 7일 오전 10:44, 키움증권 체결알림 : [키움]체결통보
        삼성전자
        매수 10주
        평균단가 50,000원
        
        지원 형식 2 (기존):
        --------------- 2026년 1월 15일 목요일 ---------------
        [키움증권 체결알림] [오전 8:01] [키움]체결통보
        한화
        매수 16주
        평균단가 127,100원
        
        Args:
            text: 거래 이력 텍스트
            kospi_dict: {code: name} 형식의 종목 매핑 (선택사항)
            default_date: 기본 거래 날짜 (YYYY-MM-DD 형식, 선택사항)
        
        Returns:
            [{'name': 종목명, 'action': 'BUY'|'SELL', 'quantity': 수량, 'price': 가격, 'date': 거래datetime}, ...]
        """
        import re
        from datetime import datetime
        
        trades = []
        lines = text.strip().split('\n')
        
        # 기본 거래 날짜 초기 설정
        if default_date:
            try:
                base_date = datetime.strptime(default_date, "%Y-%m-%d").date()
            except:
                base_date = datetime.now().date()
        else:
            base_date = datetime.now().date()
        
        default_time = datetime.combine(base_date, datetime.min.time())
        
        # 거래 상태
        current_stock = None
        current_action = None
        current_quantity = 0
        current_price = None
        current_time = default_time
        
        # 중복 처리 방지를 위한 마지막 처리된 datetime 기록
        last_processed_datetime = None
        
        for idx, line in enumerate(lines):
            line_clean = line.strip()
            
            # **형식 1 우선: 날짜+시간이 함께 있는 경우 (2024년 2월 7일 오전 10:44)**
            datetime_match = re.search(r'(\d{4})년\s+(\d{1,2})월\s+(\d{1,2})일\s+(오전|오후)\s+(\d{1,2}):(\d{2})', line_clean)
            if datetime_match:
                # 중복 방지: 같은 datetime이면 스킵
                current_datetime_key = f"{datetime_match.group(1)}{datetime_match.group(2)}{datetime_match.group(3)}{datetime_match.group(5)}{datetime_match.group(6)}"
                if current_datetime_key == last_processed_datetime:
                    continue
                
                # 이전 거래 저장
                if current_stock and current_action and current_quantity > 0:
                    if current_price is None:
                        current_price = 0
                    
                    trade_info = {
                        'name': current_stock,
                        'action': current_action,
                        'quantity': current_quantity,
                        'price': current_price if current_price else 0,
                        'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    trades.append(trade_info)
                
                # 새 날짜/시간 적용
                year = int(datetime_match.group(1))
                month = int(datetime_match.group(2))
                day = int(datetime_match.group(3))
                period = datetime_match.group(4)
                hour = int(datetime_match.group(5))
                minute = int(datetime_match.group(6))
                
                # 시간 변환
                if period == '오후' and hour != 12:
                    hour += 12
                elif period == '오전' and hour == 12:
                    hour = 0
                
                base_date = datetime(year, month, day).date()
                current_time = datetime(year, month, day, hour, minute)
                last_processed_datetime = current_datetime_key
                
                # 상태 초기화
                current_stock = None
                current_action = None
                current_quantity = 0
                current_price = None
                continue
            
            # **형식 2: 날짜만 있는 경우 (기존 형식의 구분선)**
            date_match = re.search(r'(\d{4})년\s+(\d{1,2})월\s+(\d{1,2})일', line_clean)
            if date_match and '-------' in line_clean:  # 구분선 확인
                # 이전 거래 저장
                if current_stock and current_action and current_quantity > 0:
                    if current_price is None:
                        current_price = 0
                    
                    trade_info = {
                        'name': current_stock,
                        'action': current_action,
                        'quantity': current_quantity,
                        'price': current_price if current_price else 0,
                        'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    trades.append(trade_info)
                
                # 새 날짜 적용
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                day = int(date_match.group(3))
                base_date = datetime(year, month, day).date()
                default_time = datetime.combine(base_date, datetime.min.time())
                current_time = default_time
                
                # 상태 초기화
                current_stock = None
                current_action = None
                current_quantity = 0
                current_price = None
                continue
            
            # 빈 줄 처리
            if not line_clean:
                if current_stock and current_action and current_quantity > 0:
                    if current_price is None:
                        current_price = 0
                    
                    trade_info = {
                        'name': current_stock,
                        'action': current_action,
                        'quantity': current_quantity,
                        'price': current_price if current_price else 0,
                        'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    trades.append(trade_info)
                
                # 상태 초기화
                current_stock = None
                current_action = None
                current_quantity = 0
                current_price = None
                current_time = default_time
                continue
            
            # 시간 패턴 추출: [오전/오후 HH:MM] (기존 형식)
            time_match = re.search(r'\[(오전|오후)\s+(\d{1,2}):(\d{2})\]', line_clean)
            if time_match:
                period = time_match.group(1)
                hour = int(time_match.group(2))
                minute = int(time_match.group(3))
                
                # 시간 변환
                if period == '오후' and hour != 12:
                    hour += 12
                elif period == '오전' and hour == 12:
                    hour = 0
                
                current_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute))
                continue
            
            # 종목명 감지: 한글 또는 영문 (거래 정보 라인 제외)
            is_trade_info_line = any(keyword in line_clean for keyword in ['키움', '체결', '평균단가', '원', '통보', '매수', '매도'])
            is_stock_name = re.match(r'^[가-힣a-zA-Z0-9\s&\-\.]+$', line_clean)
            
            if not is_trade_info_line and is_stock_name and len(line_clean) > 0:
                # 이전 거래 저장
                if current_stock and current_action and current_quantity > 0:
                    if current_price is None:
                        current_price = 0
                    
                    trade_info = {
                        'name': current_stock,
                        'action': current_action,
                        'quantity': current_quantity,
                        'price': current_price if current_price else 0,
                        'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    trades.append(trade_info)
                
                # 새 종목 시작
                current_stock = line_clean
                current_action = None
                current_quantity = 0
                current_price = None
                continue
            
            # 매수 처리
            if '매수' in line_clean:
                qty_match = re.search(r'매수\s*(\d+)\s*주', line_clean)
                if qty_match:
                    current_action = 'BUY'
                    current_quantity = int(qty_match.group(1))
                    
                    # 같은 라인에 가격이 있는지 확인
                    price_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s*원', line_clean)
                    if price_match:
                        price_str = price_match.group(1).replace(',', '')
                        try:
                            current_price = int(price_str)
                        except ValueError:
                            current_price = None
            
            # 매도 처리
            elif '매도' in line_clean:
                qty_match = re.search(r'매도\s*(\d+)\s*주', line_clean)
                if qty_match:
                    current_action = 'SELL'
                    current_quantity = int(qty_match.group(1))
                    
                    # 같은 라인에 가격이 있는지 확인
                    price_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s*원', line_clean)
                    if price_match:
                        price_str = price_match.group(1).replace(',', '')
                        try:
                            current_price = int(price_str)
                        except ValueError:
                            current_price = None
            
            # 평균단가/단가 처리
            elif '평균단가' in line_clean or ('단가' in line_clean and '평균' not in line_clean):
                price_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s*원', line_clean)
                if price_match and current_price is None:
                    price_str = price_match.group(1).replace(',', '')
                    try:
                        current_price = int(price_str)
                    except ValueError:
                        pass
        
        # 마지막 거래 처리
        if current_stock and current_action and current_quantity > 0:
            if current_price is None:
                current_price = 0
            
            trade_info = {
                'name': current_stock,
                'action': current_action,
                'quantity': current_quantity,
                'price': current_price if current_price else 0,
                'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            trades.append(trade_info)
        
        return trades
        
        print(f"[DEBUG] 파싱 시작. 입력 라인 수: {len(lines)}, 기본 날짜: {default_date}")
        
        # 기본 거래 날짜 초기 설정
        if default_date:
            # 사용자가 날짜를 지정한 경우
            try:
                base_date = datetime.strptime(default_date, "%Y-%m-%d").date()
            except:
                base_date = datetime.now().date()
        else:
            # 날짜 지정 없음 - 현재 날짜 사용
            base_date = datetime.now().date()
        
        default_time = datetime.combine(base_date, datetime.min.time())
        
        current_stock = None
        current_action = None
        current_quantity = 0
        current_price = None
        current_time = default_time
        
        for idx, line in enumerate(lines):
            line_clean = line.strip()
            print(f"[DEBUG] 라인 {idx}: '{line_clean}'")
            
            # 날짜+시간 라인 감지: YYYY년 M월 D일 오전/오후 HH:MM 형식
            datetime_match = re.search(r'(\d{4})년\s+(\d{1,2})월\s+(\d{1,2})일\s+(오전|오후)\s+(\d{1,2}):(\d{2})', line_clean)
            if datetime_match:
                # 현재 거래 저장
                if current_stock and current_action and current_quantity > 0:
                    if current_price is None:
                        current_price = 0
                    
                    trade_info = {
                        'name': current_stock,
                        'action': current_action,
                        'quantity': current_quantity,
                        'price': current_price if current_price else 0,
                        'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    trades.append(trade_info)
                    print(f"[DEBUG] 거래 추가 (날짜시간변경): {trade_info}")
                
                # 새 날짜 + 시간 적용
                year = int(datetime_match.group(1))
                month = int(datetime_match.group(2))
                day = int(datetime_match.group(3))
                period = datetime_match.group(4)
                hour = int(datetime_match.group(5))
                minute = int(datetime_match.group(6))
                
                # 시간 변환 (오전/오후)
                if period == '오후' and hour != 12:
                    hour += 12
                elif period == '오전' and hour == 12:
                    hour = 0
                
                base_date = datetime(year, month, day).date()
                current_time = datetime(year, month, day, hour, minute)
                default_time = datetime.combine(base_date, datetime.min.time())
                
                print(f"[DEBUG] 날짜+시간 변경: {current_time}")
                
                # 상태 초기화
                current_stock = None
                current_action = None
                current_quantity = 0
                current_price = None
                continue
            
            # 날짜 라인 감지: YYYY년 M월 D일 형식
            date_match = re.search(r'(\d{4})년\s+(\d{1,2})월\s+(\d{1,2})일', line_clean)
            if date_match:
                # 현재 거래 저장
                if current_stock and current_action and current_quantity > 0:
                    if current_price is None:
                        current_price = 0
                    
                    trade_info = {
                        'name': current_stock,
                        'action': current_action,
                        'quantity': current_quantity,
                        'price': current_price if current_price else 0,
                        'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    trades.append(trade_info)
                    print(f"[DEBUG] 거래 추가 (날짜변경): {trade_info}")
                
                # 새 날짜 적용
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                day = int(date_match.group(3))
                base_date = datetime(year, month, day).date()
                default_time = datetime.combine(base_date, datetime.min.time())
                current_time = default_time
                
                print(f"[DEBUG] 날짜 변경: {base_date}")
                
                # 상태 초기화
                current_stock = None
                current_action = None
                current_quantity = 0
                current_price = None
                continue
            
            if not line_clean:
                # 빈 줄: 현재 거래가 완료되었다고 판단
                if current_stock and current_action and current_quantity > 0:
                    if current_price is None:
                        print(f"[DEBUG] [WARN] {current_stock} {current_action} - 가격 없음")
                        current_price = 0
                    
                    trade_info = {
                        'name': current_stock,
                        'action': current_action,
                        'quantity': current_quantity,
                        'price': current_price if current_price else 0,
                        'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    trades.append(trade_info)
                    print(f"[DEBUG] 거래 추가 (빈줄): {trade_info}")
                
                # 상태 초기화
                current_stock = None
                current_action = None
                current_quantity = 0
                current_price = None
                current_time = default_time
                continue
            
            # 시간 패턴 추출: [오전/오후 HH:MM]
            time_match = re.search(r'\[(오전|오후)\s+(\d{1,2}):(\d{2})\]', line_clean)
            if time_match:
                period = time_match.group(1)
                hour = int(time_match.group(2))
                minute = int(time_match.group(3))
                
                # 오후 처리
                if period == '오후' and hour != 12:
                    hour += 12
                elif period == '오전' and hour == 12:
                    hour = 0
                
                current_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute))
                print(f"[DEBUG] 시간 업데이트: {current_time}")
                continue
            
            # 종목명 감지: 한글 또는 영문 (거래 정보 라인은 제외)
            # 무시할 라인: 키움증권, 체결, 평균단가 등 거래 정보
            is_trade_info_line = any(keyword in line_clean for keyword in ['키움', '체결', '평균단가', '원'])
            is_stock_name = re.match(r'^[가-힣a-zA-Z0-9\s]+$', line_clean)
            
            if not is_trade_info_line and is_stock_name:
                # 이전 거래 저장
                if current_stock and current_action and current_quantity > 0:
                    if current_price is None:
                        current_price = 0
                    
                    trade_info = {
                        'name': current_stock,
                        'action': current_action,
                        'quantity': current_quantity,
                        'price': current_price if current_price else 0,
                        'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    trades.append(trade_info)
                    print(f"[DEBUG] 거래 추가 (새 종목): {trade_info}")
                
                # 새 종목 시작
                current_stock = line_clean
                current_action = None
                current_quantity = 0
                current_price = None
                print(f"[DEBUG] 종목명: {current_stock}")
                continue
            
            # 매수 처리
            if '매수' in line_clean:
                qty_match = re.search(r'매수\s*(\d+)\s*주', line_clean)
                if qty_match and current_stock:
                    current_action = 'BUY'
                    current_quantity = int(qty_match.group(1))
                    
                    # 같은 라인에 가격이 있는지 확인
                    price_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s*원', line_clean)
                    if price_match:
                        price_str = price_match.group(1).replace(',', '')
                        try:
                            current_price = int(price_str)
                        except ValueError:
                            current_price = None
                    
                    print(f"[DEBUG] 매수 감지: {current_quantity}주, 가격: {current_price}")
            
            # 매도 처리
            elif '매도' in line_clean:
                qty_match = re.search(r'매도\s*(\d+)\s*주', line_clean)
                if qty_match and current_stock:
                    current_action = 'SELL'
                    current_quantity = int(qty_match.group(1))
                    
                    # 같은 라인에 가격이 있는지 확인
                    price_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s*원', line_clean)
                    if price_match:
                        price_str = price_match.group(1).replace(',', '')
                        try:
                            current_price = int(price_str)
                        except ValueError:
                            current_price = None
                    
                    print(f"[DEBUG] 매도 감지: {current_quantity}주, 가격: {current_price}")
            
            # 평균단가/단가 처리
            elif '평균단가' in line_clean or '단가' in line_clean:
                price_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s*원', line_clean)
                if price_match and current_price is None:  # 아직 가격이 없을 때만
                    price_str = price_match.group(1).replace(',', '')
                    try:
                        current_price = int(price_str)
                        print(f"[DEBUG] 별도 라인에서 가격 추출: {current_price}")
                    except ValueError:
                        pass
        
        # 마지막 거래 처리
        if current_stock and current_action and current_quantity > 0:
            if current_price is None:
                current_price = 0
            
            trade_info = {
                'name': current_stock,
                'action': current_action,
                'quantity': current_quantity,
                'price': current_price if current_price else 0,
                'date': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            trades.append(trade_info)
            print(f"[DEBUG] 거래 추가 (마지막): {trade_info}")
        
        print(f"\n[DEBUG] 파싱 완료. 총 거래 수: {len(trades)}")
        for idx, trade in enumerate(trades):
            print(f"[DEBUG]   거래 {idx+1}: {trade['name']} {trade['action']} {trade['quantity']}주 @ {trade['price']:,}원 ({trade['date']})")
        
        return trades
    
    def load_kospi_mapping(self) -> Dict[str, str]:
        """
        KOSPI 종목명 -> 종목코드 매핑 로드
        
        Returns:
            {종목명: 종목코드} 딕셔너리
        """
        import os
        kospi_list_file = "data/kospi_list.pkl"
        mapping = {}
        
        if os.path.exists(kospi_list_file):
            try:
                kospi_df = pd.read_pickle(kospi_list_file)
                # 최신 데이터만 사용 (code 기준 중복 제거)
                kospi_df = kospi_df.drop_duplicates(subset=['code'], keep='last')
                mapping = kospi_df[['name', 'code']].set_index('name')['code'].to_dict()
                print(f"[DEBUG] KOSPI 매핑 로드: {len(mapping)}개 종목")
            except Exception as e:
                print(f"[DEBUG] KOSPI 매핑 로드 실패: {e}")
        
        return mapping
    
    def apply_trading_history(self, trades: List[Dict]) -> Dict[str, str]:
        """
        거래 이력을 포트폴리오에 반영 (시간순 정렬 처리)
        
        Args:
            trades: parse_trading_history_text()에서 반환된 거래 기록 리스트
                [{'name': 종목명, 'action': 'BUY'|'SELL', 'quantity': 수량, 'price': 가격, 'date': 날짜}, ...]
        
        Returns:
            {종목명: 결과 메시지} 형식의 결과
        """
        from datetime import datetime
        
        results = {}
        
        if not trades:
            return {'ERROR': '[ERROR] 파싱된 거래가 없습니다.'}
        
        # 거래를 날짜 순서대로 정렬 (과거 → 미래)
        sorted_trades = sorted(trades, key=lambda x: x.get('date', ''))
        print(f"[DEBUG] 거래 정렬 완료. 순서: {[t.get('date', '') for t in sorted_trades]}")
        
        # 종목 매핑 로드 (KOSPI + 포트폴리오)
        kospi_mapping = self.load_kospi_mapping()
        portfolio = self.load_portfolio()
        portfolio_mapping = {info['name']: code for code, info in portfolio.items()}
        
        # 거래에 포함된 고유 종목들을 먼저 파악
        unique_stocks = {}  # {종목명: 코드}
        unknown_stocks = []  # KOSPI에도 없고 포트폴리오에도 없는 종목
        
        for trade in sorted_trades:
            name = trade.get('name', '').strip()
            if name and name not in unique_stocks:
                if name in portfolio_mapping:
                    # 포트폴리오에 있는 종목
                    unique_stocks[name] = portfolio_mapping[name]
                elif name in kospi_mapping:
                    # KOSPI에 있는 종목
                    unique_stocks[name] = kospi_mapping[name]
                else:
                    # 찾을 수 없는 종목
                    unknown_stocks.append(name)
        
        # 찾을 수 없는 종목 경고
        if unknown_stocks:
            for unknown in unknown_stocks:
                results[unknown] = f"[ERROR] {unknown} - KOSPI 종목에 없습니다. 종목명을 확인해주세요."
        
        # 정렬된 순서대로 거래 적용 (시간순)
        for trade in sorted_trades:
            name = trade.get('name', '').strip()
            action = trade.get('action', '').upper()
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            trade_date = trade.get('date')
            
            # 찾을 수 없는 종목은 스킵
            if name in unknown_stocks:
                continue
            
            # 거래 정보 검증
            if not name:
                results['ERROR'] = '[ERROR] 종목명이 없는 거래가 있습니다.'
                continue
            
            if not action or action not in ['BUY', 'SELL']:
                if name not in results:
                    results[name] = f"[ERROR] {name} - 거래 타입 오류: {action}"
                continue
            
            if quantity <= 0:
                if name not in results:
                    results[name] = f"[ERROR] {name} - 수량 오류: {quantity}주"
                continue
            
            if price <= 0:
                if name not in results:
                    results[name] = f"[WARN] {name} - 경고: 가격이 없음 ({action} {quantity}주)"
                continue
            
            # 중복 거래 체크
            code = unique_stocks[name]
            is_duplicate = self._is_duplicate_trade(code, action, quantity, price, trade_date)
            if is_duplicate:
                results[name] = f"[INFO] {name} - 중복 거래입니다. 건너뜀 ({action} {quantity}주 @ {price:,.0f}원, {trade_date})"
                print(f"[DEBUG] 중복 거래 감지 및 스킵: {name} {action} {quantity}주 {trade_date}")
                continue
            
            try:
                # add_trade 메서드를 사용하여 거래 기록
                result = self.add_trade(
                    code=code,
                    name=name,
                    trade_type=action,
                    quantity=quantity,
                    price=price,
                    trade_date=trade_date
                )
                
                results[name] = result['message']
                
                if result['status'] == 'error':
                    print(f"[DEBUG] 거래 적용 오류: {result['message']}")
                else:
                    print(f"[DEBUG] 거래 적용 성공: {result['message']}")
            
            except Exception as e:
                results[name] = f"[ERROR] '{name}' - 오류: {str(e)}"
                print(f"[DEBUG] 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def _is_duplicate_trade(self, code: str, trade_type: str, quantity: float, price: float, trade_date: str) -> bool:
        """
        이미 포트폴리오에 기록된 동일한 거래가 있는지 확인
        
        Args:
            code: 종목 코드
            trade_type: BUY 또는 SELL
            quantity: 수량
            price: 가격
            trade_date: 거래 날짜 (YYYY-MM-DD HH:MM:SS 형식)
        
        Returns:
            중복 거래이면 True, 아니면 False
        """
        portfolio = self.load_portfolio()
        
        if code not in portfolio:
            return False
        
        existing_trades = portfolio[code].get('trades', [])
        
        for existing_trade in existing_trades:
            # 다음 모든 조건이 일치하면 중복
            if (existing_trade.get('type') == trade_type and
                existing_trade.get('quantity') == quantity and
                existing_trade.get('price') == price and
                existing_trade.get('date') == trade_date):
                return True
        
        # 삭제된 종목 이력에서도 확인
        trading_history = self.load_trading_history()
        if code in trading_history:
            existing_trades = trading_history[code].get('trades', [])
            for existing_trade in existing_trades:
                if (existing_trade.get('type') == trade_type and
                    existing_trade.get('quantity') == quantity and
                    existing_trade.get('price') == price and
                    existing_trade.get('date') == trade_date):
                    return True
        
        return False
    
    def get_all_trades(self) -> List[Dict]:
        """
        포트폴리오의 모든 거래를 통합하여 반환 (현재 포트폴리오 + 삭제된 종목 포함)
        
        Returns:
            거래 리스트 (시간 역순) - 각 거래는 code, name 포함
        """
        portfolio = self.load_portfolio()
        trading_history = self.load_trading_history()  # 삭제된 종목의 거래 이력
        
        all_trades = []
        
        # 현재 포트폴리오의 거래
        for code, info in portfolio.items():
            trades = info.get('trades', [])
            for trade in trades:
                trade_with_info = {
                    'code': code,
                    'name': info['name'],
                    **trade
                }
                all_trades.append(trade_with_info)
        
        # 삭제된 종목의 거래 이력
        for code, history_info in trading_history.items():
            if code not in portfolio:  # 현재 포트폴리오에 없는 종목만
                trades = history_info.get('trades', [])
                for trade in trades:
                    trade_with_info = {
                        'code': code,
                        'name': history_info.get('name', ''),
                        **trade
                    }
                    all_trades.append(trade_with_info)
        
        # 시간 역순 정렬 (최신순)
        all_trades.sort(key=lambda x: x.get('date', ''), reverse=True)
        return all_trades
    
    def calculate_trade_analysis(self) -> Dict:
        """
        거래 기반 수익률 분석 (현재 포트폴리오 + 삭제된 종목의 거래 이력 포함)
        
        Returns:
            {
                'total_trades': 총 거래 수,
                'buy_count': 매수 거래 수,
                'sell_count': 매도 거래 수,
                'winning_trades': 수익 거래 수,
                'losing_trades': 손실 거래 수,
                'win_rate': 승률 (%),
                'avg_win_pct': 평균 수익률 (%),
                'avg_loss_pct': 평균 손실률 (%),
                'total_return_pct': 총 수익률 (%),
                'best_trade_pct': 최고 수익률 (%),
                'worst_trade_pct': 최악 손실률 (%),
                'total_realized_pnl': 총 실현 손익 (원),
                'trades_by_stock': {code: [거래 기록, ...]}
            }
        """
        portfolio = self.load_portfolio()
        trading_history = self.load_trading_history()  # 삭제된 종목의 거래 이력
        
        all_sell_trades = []  # 매도 거래만 분석
        buy_count = 0
        sell_count = 0
        total_realized_pnl = 0
        returns_pct = []
        
        trades_by_stock = {}
        
        # 1. 현재 포트폴리오 종목의 거래 분석
        for code, info in portfolio.items():
            trades = info.get('trades', [])
            stock_trades = []
            
            for trade in trades:
                if trade['type'] == 'BUY':
                    buy_count += 1
                elif trade['type'] == 'SELL':
                    sell_count += 1
                    return_pct = trade.get('return_pct', 0)
                    if return_pct is not None:
                        # 실현 손익 계산
                        sell_price = trade['price']
                        quantity = trade['quantity']
                        avg_buy_price = trade.get('avg_buy_price', sell_price)
                        if avg_buy_price == 0:
                            avg_buy_price = sell_price / (1 + return_pct / 100) if return_pct != 0 else sell_price
                        pnl = quantity * (sell_price - avg_buy_price)
                        
                        all_sell_trades.append({
                            'code': code,
                            'name': info['name'],
                            'date': trade['date'],
                            'quantity': trade['quantity'],
                            'price': trade['price'],
                            'avg_buy_price': trade.get('avg_buy_price', 0),
                            'return_pct': return_pct,
                            'pnl': pnl  # 수익금액(원)
                        })
                        returns_pct.append(return_pct)
                        total_realized_pnl += pnl
                
                stock_trades.append(trade)
            
            if stock_trades:
                trades_by_stock[code] = stock_trades
        
        # 2. 삭제된 종목의 거래 이력도 분석에 포함
        for code, history_info in trading_history.items():
            # 현재 포트폴리오에는 없지만 거래 이력에만 있는 경우
            if code not in trades_by_stock:
                trades = history_info.get('trades', [])
                stock_trades = []
                
                for trade in trades:
                    if trade['type'] == 'BUY':
                        buy_count += 1
                    elif trade['type'] == 'SELL':
                        sell_count += 1
                        return_pct = trade.get('return_pct', 0)
                        if return_pct is not None:
                            # 실현 손익 계산
                            sell_price = trade['price']
                            quantity = trade['quantity']
                            avg_buy_price = trade.get('avg_buy_price', sell_price)
                            if avg_buy_price == 0:
                                avg_buy_price = sell_price / (1 + return_pct / 100) if return_pct != 0 else sell_price
                            pnl = quantity * (sell_price - avg_buy_price)
                            
                            all_sell_trades.append({
                                'code': code,
                                'name': history_info.get('name', ''),
                                'date': trade['date'],
                                'quantity': trade['quantity'],
                                'price': trade['price'],
                                'avg_buy_price': trade.get('avg_buy_price', 0),
                                'return_pct': return_pct,
                                'pnl': pnl  # 수익금액(원)
                            })
                            returns_pct.append(return_pct)
                            total_realized_pnl += pnl
                            total_realized_pnl += pnl
                    
                    stock_trades.append(trade)
                
                if stock_trades:
                    trades_by_stock[code] = stock_trades
        
        # 통계 계산
        total_trades = buy_count + sell_count
        winning_trades = len([r for r in returns_pct if r > 0])
        losing_trades = len([r for r in returns_pct if r < 0])
        break_even = len([r for r in returns_pct if r == 0])
        
        win_rate = (winning_trades / sell_count * 100) if sell_count > 0 else 0
        
        win_returns = [r for r in returns_pct if r > 0]
        loss_returns = [r for r in returns_pct if r < 0]
        
        avg_win_pct = np.mean(win_returns) if win_returns else 0
        avg_loss_pct = np.mean(loss_returns) if loss_returns else 0
        total_return_pct = np.sum(returns_pct) if returns_pct else 0
        best_trade_pct = max(returns_pct) if returns_pct else 0
        worst_trade_pct = min(returns_pct) if returns_pct else 0
        
        return {
            'total_trades': total_trades,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'break_even_trades': break_even,
            'win_rate': round(win_rate, 2),
            'avg_win_pct': round(avg_win_pct, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'total_return_pct': round(total_return_pct, 2),
            'best_trade_pct': round(best_trade_pct, 2),
            'worst_trade_pct': round(worst_trade_pct, 2),
            'total_realized_pnl': round(total_realized_pnl),
            'all_sell_trades': all_sell_trades,
            'trades_by_stock': trades_by_stock
        }
    
    def get_trade_summary_by_stock(self) -> pd.DataFrame:
        """
        종목별 거래 요약
        
        Returns:
            DataFrame: {
                '종목명', '매수횟수', '매도횟수', '평균수익률', '최고수익률', 
                '최악손실', '총수익', '승률'
            }
        """
        portfolio = self.load_portfolio()
        summaries = []
        
        for code, info in portfolio.items():
            trades = info.get('trades', [])
            
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            buy_count = len([t for t in trades if t['type'] == 'BUY'])
            sell_count = len(sell_trades)
            
            returns = [t.get('return_pct', 0) for t in sell_trades 
                      if t.get('return_pct') is not None]
            
            if returns:
                avg_return = np.mean(returns)
                max_return = max(returns)
                min_return = min(returns)
                win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
                
                summaries.append({
                    '종목명': info['name'],
                    '종목코드': code,
                    '매수': buy_count,
                    '매도': sell_count,
                    '평균수익률': round(avg_return, 2),
                    '최고수익률': round(max_return, 2),
                    '최악손실': round(min_return, 2),
                    '총수익률': round(sum(returns), 2),
                    '승률': round(win_rate, 1)
                })
        
        return pd.DataFrame(summaries)
    
    def simulate_daily_trading_constraints(self, 
                                          trades_list: List[Dict],
                                          max_buy_stocks_per_day: int = 5,
                                          transaction_unit: int = 2000000) -> Dict:
        """
        일일 거래 제약 조건을 적용한 시뮬레이션
        
        Args:
            trades_list: 거래 목록 [{code, name, type, price, quantity, date}, ...]
            max_buy_stocks_per_day: 하루 최대 매수 종목 수 (기본값: 5)
            transaction_unit: 거래금액 단위 (기본값: 2,000,000원)
        
        Returns:
            {
                'executed_trades': [실행된 거래들],
                'skipped_trades': [제약조건으로 인해 스킵된 거래들],
                'simulation_stats': {
                    'total_trades': 입력된 거래 수,
                    'executed_count': 실행된 거래 수,
                    'skipped_count': 스킵된 거래 수,
                    'max_daily_stock_limit_exceeded': 일일 한도 초과 횟수,
                    'transaction_unit_constraint_violated': 거래금액 제약 위반 횟수
                }
            }
        """
        from collections import defaultdict
        
        executed_trades = []
        skipped_trades = []
        
        # 날짜별로 거래를 그룹화
        trades_by_date = defaultdict(list)
        for trade in trades_list:
            date = trade.get('date', '')[:10]  # YYYY-MM-DD 형식
            trades_by_date[date].append(trade)
        
        # 날짜별로 제약조건 적용
        max_daily_stock_limit_exceeded = 0
        transaction_unit_constraint_violated = 0
        
        for date in sorted(trades_by_date.keys()):
            daily_trades = trades_by_date[date]
            buy_trades_today = []
            
            for trade in daily_trades:
                trade_type = trade.get('type', '').upper()
                trade_amount = trade.get('quantity', 0) * trade.get('price', 0)
                
                # BUY 거래에만 제약 조건 적용
                if trade_type == 'BUY':
                    # 1. 거래금액 단위 체크
                    if trade_amount % transaction_unit != 0:
                        transaction_unit_constraint_violated += 1
                        skipped_trades.append({
                            **trade,
                            'skip_reason': f'거래금액이 {transaction_unit:,}원의 배수가 아닙니다. (실제: {trade_amount:,.0f}원)'
                        })
                        continue
                    
                    # 2. 하루 최대 매수 종목 제한 체크
                    if len(buy_trades_today) >= max_buy_stocks_per_day:
                        max_daily_stock_limit_exceeded += 1
                        skipped_trades.append({
                            **trade,
                            'skip_reason': f'하루 최대 매수 종목을 초과했습니다. ({len(buy_trades_today)}/{max_buy_stocks_per_day})'
                        })
                        continue
                    
                    buy_trades_today.append(trade)
                    executed_trades.append({**trade, 'status': 'executed'})
                
                # SELL 거래는 제약조건 없음
                elif trade_type == 'SELL':
                    executed_trades.append({**trade, 'status': 'executed'})
        
        return {
            'executed_trades': executed_trades,
            'skipped_trades': skipped_trades,
            'simulation_stats': {
                'total_trades': len(trades_list),
                'executed_count': len(executed_trades),
                'skipped_count': len(skipped_trades),
                'max_daily_stock_limit_exceeded': max_daily_stock_limit_exceeded,
                'transaction_unit_constraint_violated': transaction_unit_constraint_violated
            }
        }
