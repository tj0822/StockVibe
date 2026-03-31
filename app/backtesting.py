"""
백테스팅 & 정확도 검증 모듈
- AI 예측 정확도 추적
- 기술지표 신호 성공률 통계
- 전략 백테스팅
"""
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict

class BacktestingEngine:
    """백테스팅 엔진"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.prediction_history_file = os.path.join(data_dir, "prediction_history.json")
        
        os.makedirs(data_dir, exist_ok=True)
    
    def save_prediction(self, code: str, prediction: Dict):
        """AI 예측 저장 (정확도 검증용)"""
        history = self.load_prediction_history()
        
        if code not in history:
            history[code] = []
        
        prediction['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction['verification_date'] = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        history[code].append(prediction)
        
        # 최근 50개만 보관
        if len(history[code]) > 50:
            history[code] = history[code][-50:]
        
        with open(self.prediction_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def load_prediction_history(self) -> Dict:
        """예측 히스토리 불러오기"""
        if os.path.exists(self.prediction_history_file):
            with open(self.prediction_history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def verify_predictions(self, code: str, actual_price: float) -> Dict:
        """예측 검증 (1주일 후 실제 가격과 비교)"""
        history = self.load_prediction_history()
        
        if code not in history:
            return {'total': 0, 'verified': 0, 'accuracy': 0}
        
        today = datetime.now().strftime("%Y-%m-%d")
        verified_count = 0
        correct_count = 0
        
        for pred in history[code]:
            if pred.get('verified', False):
                verified_count += 1
                if pred.get('was_correct', False):
                    correct_count += 1
            elif pred['verification_date'] <= today:
                # 검증 가능한 예측
                predicted_direction = pred.get('direction', 'neutral')
                original_price = pred.get('current_price', 0)
                
                actual_change = (actual_price - original_price) / original_price * 100 if original_price > 0 else 0
                
                # 정확도 판정 (방향 일치 & 변화율 유사)
                was_correct = False
                if predicted_direction == 'up' and actual_change > 2:
                    was_correct = True
                elif predicted_direction == 'down' and actual_change < -2:
                    was_correct = True
                elif predicted_direction == 'neutral' and abs(actual_change) < 2:
                    was_correct = True
                
                pred['verified'] = True
                pred['was_correct'] = was_correct
                pred['actual_change'] = actual_change
                
                verified_count += 1
                if was_correct:
                    correct_count += 1
        
        # 업데이트된 히스토리 저장
        with open(self.prediction_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        accuracy = (correct_count / verified_count * 100) if verified_count > 0 else 0
        
        return {
            'total': len(history[code]),
            'verified': verified_count,
            'correct': correct_count,
            'accuracy': accuracy
        }
    
    def backtest_technical_strategy(self, price_df: pd.DataFrame, 
                                    strategy: str = 'golden_cross') -> Dict:
        """기술적 지표 전략 백테스팅"""
        if price_df.empty or len(price_df) < 30:
            return {'total_return': 0, 'win_rate': 0, 'num_trades': 0}
        
        # 이동평균 계산
        price_df['MA5'] = price_df['Close'].rolling(window=5).mean()
        price_df['MA20'] = price_df['Close'].rolling(window=20).mean()
        
        signals = []
        position = None
        
        for i in range(1, len(price_df)):
            prev_row = price_df.iloc[i-1]
            curr_row = price_df.iloc[i]
            
            if strategy == 'golden_cross':
                # 골든크로스 전략
                if prev_row['MA5'] <= prev_row['MA20'] and curr_row['MA5'] > curr_row['MA20']:
                    # 매수 신호
                    if position is None:
                        position = {
                            'type': 'long',
                            'entry_price': curr_row['Close'],
                            'entry_date': curr_row.name
                        }
                elif prev_row['MA5'] >= prev_row['MA20'] and curr_row['MA5'] < curr_row['MA20']:
                    # 매도 신호
                    if position is not None and position['type'] == 'long':
                        exit_price = curr_row['Close']
                        profit = (exit_price - position['entry_price']) / position['entry_price'] * 100
                        
                        signals.append({
                            'entry_date': position['entry_date'],
                            'exit_date': curr_row.name,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit
                        })
                        position = None
        
        if not signals:
            return {'total_return': 0, 'win_rate': 0, 'num_trades': 0}
        
        # 통계 계산
        profits = [s['profit'] for s in signals]
        total_return = sum(profits)
        win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
        avg_profit = sum(profits) / len(profits)
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(signals),
            'avg_profit': avg_profit,
            'max_profit': max(profits),
            'max_loss': min(profits),
            'signals': signals
        }
    
    def backtest_rsi_strategy(self, price_df: pd.DataFrame, 
                             oversold: int = 30, overbought: int = 70) -> Dict:
        """RSI 전략 백테스팅"""
        if price_df.empty or len(price_df) < 20:
            return {'total_return': 0, 'win_rate': 0, 'num_trades': 0}
        
        # RSI 계산
        delta = price_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        price_df['RSI'] = 100 - (100 / (1 + rs))
        
        signals = []
        position = None
        
        for i in range(1, len(price_df)):
            curr_row = price_df.iloc[i]
            
            if position is None and curr_row['RSI'] < oversold:
                # 과매도 구간 매수
                position = {
                    'type': 'long',
                    'entry_price': curr_row['Close'],
                    'entry_date': curr_row.name
                }
            elif position is not None and curr_row['RSI'] > overbought:
                # 과매수 구간 매도
                exit_price = curr_row['Close']
                profit = (exit_price - position['entry_price']) / position['entry_price'] * 100
                
                signals.append({
                    'entry_date': position['entry_date'],
                    'exit_date': curr_row.name,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit
                })
                position = None
        
        if not signals:
            return {'total_return': 0, 'win_rate': 0, 'num_trades': 0}
        
        profits = [s['profit'] for s in signals]
        
        return {
            'total_return': sum(profits),
            'win_rate': len([p for p in profits if p > 0]) / len(profits) * 100,
            'num_trades': len(signals),
            'avg_profit': sum(profits) / len(profits),
            'max_profit': max(profits),
            'max_loss': min(profits),
            'signals': signals
        }
    
    def compare_strategies(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """여러 전략 비교"""
        strategies = {
            '골든크로스': self.backtest_technical_strategy(price_df, 'golden_cross'),
            'RSI(30/70)': self.backtest_rsi_strategy(price_df, 30, 70),
            'RSI(20/80)': self.backtest_rsi_strategy(price_df, 20, 80),
        }
        
        data = []
        for name, result in strategies.items():
            data.append({
                '전략': name,
                '총 수익률': f"{result['total_return']:.2f}%",
                '승률': f"{result['win_rate']:.1f}%",
                '거래 횟수': result['num_trades'],
                '평균 수익': f"{result.get('avg_profit', 0):.2f}%",
                '최대 수익': f"{result.get('max_profit', 0):.2f}%",
                '최대 손실': f"{result.get('max_loss', 0):.2f}%"
            })
        
        return pd.DataFrame(data)
    
    def get_prediction_accuracy_trend(self, code: str) -> pd.DataFrame:
        """예측 정확도 추이"""
        history = self.load_prediction_history()
        
        if code not in history:
            return pd.DataFrame()
        
        verified_predictions = [p for p in history[code] if p.get('verified', False)]
        
        if not verified_predictions:
            return pd.DataFrame()
        
        data = []
        for pred in verified_predictions:
            data.append({
                '날짜': pred['timestamp'][:10],
                '예측방향': pred.get('direction', 'neutral'),
                '정확여부': '✓' if pred.get('was_correct', False) else '✗',
                '실제변화율': f"{pred.get('actual_change', 0):.2f}%"
            })
        
        return pd.DataFrame(data)
