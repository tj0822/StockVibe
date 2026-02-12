"""
백테스트 파라미터 최적화 모듈
"""
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Tuple
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed


class BacktestOptimizer:
    """백테스트 파라미터 최적화 클래스"""
    
    def __init__(self, price_df: pd.DataFrame, kospi_index: pd.DataFrame):
        self.price_df = price_df
        self.kospi_index = kospi_index
        
    def optimize_parameters(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        param_ranges: Dict[str, List],
        initial_cash: float = 50_000_000,
        fee_rate: float = 0.00015,
        slippage_rate: float = 0.0005,
        sell_tax_rate: float = 0.0018,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        파라미터 그리드 서치를 통한 최적화
        
        Args:
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            param_ranges: 최적화할 파라미터 범위
                {
                    'max_daily_buys': [1, 2, 3],
                    'rolling_days': [10, 20, 30],
                    'volume_threshold': [1.5, 2.0, 2.5, 3.0]
                }
            initial_cash: 초기 자산
            progress_callback: 진행상황 콜백 함수
            
        Returns:
            최적화 결과 DataFrame (파라미터 조합별 성과)
        """
        from app.signals import build_signals
        from app.ui import run_turnover_strategy_backtest
        
        # 파라미터 조합 생성
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))
        
        total_combinations = len(combinations)
        results = []
        completed_count = 0
        
        # 병렬 처리를 위한 작업 함수
        def test_combination(combo):
            from app.signals import build_signals
            from app.ui import run_turnover_strategy_backtest
            
            params = dict(zip(param_names, combo))
            
            try:
                # 시그널 생성
                signals = build_signals(
                    df=self.price_df,
                    turnover_window=params.get('rolling_days', 20),
                    turnover_multiplier=params.get('volume_threshold', 2.0),
                    momentum_window=20,
                    momentum_threshold_pct=5.0,
                    vol_window=20,
                    vol_multiplier=2.0,
                    mr_window=20,
                    mr_z=2.0,
                    enabled_algos=["Turnover Spike"],
                    combine_mode="OR"
                )
                
                # 백테스트 실행
                equity_df, trades_df = run_turnover_strategy_backtest(
                    price_df=self.price_df,
                    signal_df=signals,
                    kospi_index=self.kospi_index,
                    start_date=start_date,
                    top_n=10,
                    initial_cash=initial_cash,
                    max_daily_buys=params.get('max_daily_buys', 2),
                    kospi_bullish_only=params.get('kospi_bullish_only', False),
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    sell_tax_rate=sell_tax_rate,
                )
                
                if not equity_df.empty:
                    # 기간 필터링
                    equity_df['date'] = pd.to_datetime(equity_df['date'])
                    mask = (equity_df['date'] >= start_date) & (equity_df['date'] <= end_date)
                    period_equity = equity_df[mask].copy()
                    
                    if not period_equity.empty and 'equity' in period_equity.columns:
                        # 성과 지표 계산
                        final_equity = period_equity['equity'].iloc[-1]
                        total_return = (final_equity / initial_cash - 1) * 100
                        net_return = total_return
                        
                        # KOSPI 수익률 계산
                        kospi_index_copy = self.kospi_index.copy()
                        kospi_index_copy['date'] = pd.to_datetime(kospi_index_copy['date'])
                        kospi_period = kospi_index_copy[
                            (kospi_index_copy['date'] >= start_date) & 
                            (kospi_index_copy['date'] <= end_date)
                        ].copy()
                        
                        if not kospi_period.empty:
                            # 'index' 또는 'close' 컬럼 확인
                            value_col = 'index' if 'index' in kospi_period.columns else 'close'
                            kospi_return = (
                                kospi_period[value_col].iloc[-1] / kospi_period[value_col].iloc[0] - 1
                            ) * 100
                        else:
                            kospi_return = 0
                        
                        # 샤프 비율 계산 (일간 수익률 기준)
                        period_equity['daily_return'] = period_equity['equity'].pct_change()
                        sharpe_ratio = self._calculate_sharpe_ratio(period_equity['daily_return'])
                        
                        # MDD 계산
                        mdd = self._calculate_mdd(period_equity['equity'])
                        
                        # 승률 계산
                        if not trades_df.empty and 'pnl' in trades_df.columns:
                            profitable_trades = len(trades_df[trades_df['pnl'] > 0])
                            win_rate = (profitable_trades / len(trades_df) * 100) if len(trades_df) > 0 else 0
                        else:
                            win_rate = 0
                        
                        return {
                            'max_daily_buys': params.get('max_daily_buys', 2),
                            'rolling_days': params.get('rolling_days', 20),
                            'volume_threshold': params.get('volume_threshold', 2.0),
                            'kospi_bullish_only': params.get('kospi_bullish_only', False),
                            'total_return': total_return,
                            'net_return': net_return,
                            'kospi_return': kospi_return,
                            'excess_return': total_return - kospi_return,
                            'net_excess_return': net_return - kospi_return,
                            'sharpe_ratio': sharpe_ratio,
                            'mdd': mdd,
                            'win_rate': win_rate,
                            'total_trades': len(trades_df) if not trades_df.empty else 0,
                            'final_equity': final_equity,
                            'params': params
                        }
                    
            except Exception as e:
                # 에러 발생 시 상세 정보 출력
                import traceback
                error_details = traceback.format_exc()
                print(f"파라미터 {params} 테스트 중 오류:")
                print(error_details)
            
            return None
        
        # 병렬 실행 (CPU 코어 수의 절반 사용)
        import os
        max_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 작업 제출
            future_to_combo = {executor.submit(test_combination, combo): combo for combo in combinations}
            
            # 완료된 작업 처리
            for future in as_completed(future_to_combo):
                completed_count += 1
                combo = future_to_combo[future]
                params = dict(zip(param_names, combo))
                
                # 진행상황 업데이트
                if progress_callback:
                    progress_callback(completed_count, total_combinations, params)
                
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Future 결과 처리 중 오류: {str(e)}")
        
        return pd.DataFrame(results)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """샤프 비율 계산"""
        if returns.empty or returns.std() == 0:
            return 0
        
        # 연율화된 샤프 비율 (252 거래일 기준)
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
        return sharpe
    
    def _calculate_mdd(self, equity: pd.Series) -> float:
        """최대 낙폭(MDD) 계산"""
        if equity.empty:
            return 0
        
        cummax = equity.expanding().max()
        drawdown = (equity - cummax) / cummax * 100
        return drawdown.min()
    
    def get_optimal_params(
        self,
        results_df: pd.DataFrame,
        metric: str = 'total_return'
    ) -> Dict:
        """
        최적 파라미터 조합 반환
        
        Args:
            results_df: 최적화 결과 DataFrame
            metric: 최적화 기준 지표 ('total_return', 'sharpe_ratio', 'excess_return' 등)
            
        Returns:
            최적 파라미터 딕셔너리
        """
        if results_df.empty:
            return {}
        
        best_idx = results_df[metric].idxmax()
        best_row = results_df.loc[best_idx]
        
        return {
            'max_daily_buys': int(best_row['max_daily_buys']),
            'rolling_days': int(best_row['rolling_days']),
            'volume_threshold': float(best_row['volume_threshold']),
            'kospi_bullish_only': bool(best_row.get('kospi_bullish_only', False)),
            f'best_{metric}': float(best_row[metric])
        }


def get_period_dates(period: str, end_date: pd.Timestamp = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    투자 기간에 따른 시작일/종료일 계산
    
    Args:
        period: '1Y', '3Y', '5Y'
        end_date: 종료일 (None이면 최신 데이터 날짜)
        
    Returns:
        (시작일, 종료일) 튜플
    """
    if end_date is None:
        end_date = pd.Timestamp.now().normalize()
    
    if period == '1Y':
        start_date = end_date - pd.DateOffset(years=1)
    elif period == '3Y':
        start_date = end_date - pd.DateOffset(years=3)
    elif period == '5Y':
        start_date = end_date - pd.DateOffset(years=5)
    else:
        raise ValueError(f"Invalid period: {period}. Use '1Y', '3Y', or '5Y'")
    
    return start_date, end_date
