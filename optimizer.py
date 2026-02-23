"""
백테스트 파라미터 최적화 모듈 (병렬처리 지원)
"""
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Tuple
import streamlit as st
import multiprocessing as mp
from multiprocessing import Pool
import random
import os
import logging
import sys

# GPU 지원 확인
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


# 모듈 레벨 함수 (병렬 처리용 최소화된 문제)
def _test_combination_worker(
    combo_idx: int,
    combo,
    param_names,
    price_df,
    kospi_index,
    start_date,
    end_date,
    initial_cash,
    fee_rate,
    slippage_rate,
    sell_tax_rate,
    open_pivot,
    close_pivot,
    kospi_bullish_dates,
    signals_by_key,  # 미리 계산된 signal dict
):
    """백테스트 조합 테스트 - signal은 미리 계산됨"""
    from app.ui import run_turnover_strategy_backtest
    
    params = dict(zip(param_names, combo))
    
    try:
        # 미리 계산된 signal 조회 (생성 X)
        rolling_days = params.get('rolling_days', 20)
        volume_threshold = params.get('volume_threshold', 2.0)
        cache_key = (int(rolling_days), float(volume_threshold))
        
        if cache_key not in signals_by_key:
            # signal이 없으면 스킵
            return None
        
        signals = signals_by_key[cache_key]
        
        # 백테스트 실행
        equity_df, trades_df = run_turnover_strategy_backtest(
            price_df=price_df,
            signal_df=signals,
            kospi_index=kospi_index,
            start_date=start_date,
            top_n=10,
            initial_cash=initial_cash,
            max_daily_buys=params.get('max_daily_buys', 2),
            buy_unit=params.get('buy_unit', 2_000_000),
            add_buy_threshold_pct=params.get('add_buy_threshold_pct', -7.0),
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            sell_tax_rate=sell_tax_rate,
            open_pivot=open_pivot,
            close_pivot=close_pivot,
            kospi_bullish_dates=kospi_bullish_dates,
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
                kospi_index_copy = kospi_index.copy()
                if not kospi_index_copy.empty:
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
                else:
                    kospi_return = 0
                
                # 샤프 비율 계산 (일간 수익률 기준)
                period_equity['daily_return'] = period_equity['equity'].pct_change()
                sharpe_ratio = _calculate_sharpe_ratio_simple(period_equity['daily_return'])
                
                # MDD 계산
                mdd = _calculate_mdd_simple(period_equity['equity'])
                
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
                    'add_buy_threshold_pct': params.get('add_buy_threshold_pct', -7.0),
                    'buy_unit': params.get('buy_unit', 2_000_000),
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
        import sys
        error_details = traceback.format_exc()
        print(f"\n[ERROR] 파라미터 {params} 테스트 중 오류:", file=sys.stderr)
        print(error_details, file=sys.stderr)
    
    return None


def _calculate_sharpe_ratio_simple(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """샤프 비율 계산 (간단한 버전)"""
    if returns.empty or returns.std() == 0:
        return 0
    return (returns.mean() - risk_free_rate / 252) / returns.std() * np.sqrt(252)


def _calculate_mdd_simple(equity: pd.Series) -> float:
    """MDD 계산 (간단한 버전)"""
    if equity.empty:
        return 0
    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax
    return drawdown.min()


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
        progress_callback=None,
        search_mode: str = "grid",
        sample_size: int = 0,
        random_seed: int = 42,
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
        # 파라미터 조합 생성
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))

        if search_mode == "random" and combinations:
            rng = random.Random(random_seed)
            effective_size = max(1, min(int(sample_size), len(combinations)))
            combinations = rng.sample(combinations, effective_size)
        
        total_combinations = len(combinations)
        results = []
        completed_count = 0

        price_data = self.price_df[["date", "code", "open", "close"]].copy()
        price_data["date"] = pd.to_datetime(price_data["date"], errors="coerce").dt.normalize()
        price_data = price_data.dropna(subset=["date", "code", "open", "close"])
        price_data = price_data.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last")
        open_pivot = price_data.pivot(index="date", columns="code", values="open")
        close_pivot = price_data.pivot(index="date", columns="code", values="close")

        kospi_bullish_dates = set()
        if not self.kospi_index.empty:
            ki = self.kospi_index.copy()
            ki["date"] = pd.to_datetime(ki["date"], errors="coerce").dt.normalize()
            ki = ki.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            ki["prev_index"] = ki["index"].shift(1)
            ki["is_bullish"] = ki["index"] > ki["prev_index"]
            kospi_bullish_dates = set(ki[ki["is_bullish"]]["date"].values)
        
        # 디버깅: 초기 정보 출력
        import sys
        print(f"[최적화 시작] 기간: {start_date} ~ {end_date}", file=sys.stderr)
        print(f"[데이터] price_df rows: {len(self.price_df)}, kospi_index rows: {len(self.kospi_index)}", file=sys.stderr)
        print(f"[pivot] open_pivot shape: {open_pivot.shape}, date range: {open_pivot.index.min()} ~ {open_pivot.index.max()}", file=sys.stderr)
        print(f"[pivot] start_date in index: {pd.to_datetime(start_date) in open_pivot.index}", file=sys.stderr)
        dates_in_range = [d for d in open_pivot.index if d >= pd.to_datetime(start_date)]
        print(f"[pivot] dates >= start_date count: {len(dates_in_range)}", file=sys.stderr)
        
        # 병렬 실행을 위해 필요한 unique signal 조합 먼저 생성 (메인 프로세스)
        print(f"[신호 생성] Unique signal 조합 미리 계산 중...", file=sys.stderr)
        
        from app.signals import build_signals
        
        # param_ranges에서 rolling_days와 volume_threshold 조합 추출
        rolling_days_list = param_ranges.get('rolling_days', [20])
        volume_threshold_list = param_ranges.get('volume_threshold', [2.0])
        
        # 중복 제거 + unique 조합만 생성
        unique_signal_keys = set()
        for rd in rolling_days_list:
            for vt in volume_threshold_list:
                unique_signal_keys.add((int(rd), float(vt)))
        
        # 메인 프로세스에서 signal 미리 생성
        signals_by_key = {}
        signal_count = len(unique_signal_keys)
        print(f"[신호 생성] {signal_count}개 signal 조합 생성 중...", file=sys.stderr)
        
        for idx, (rolling_days, volume_threshold) in enumerate(sorted(unique_signal_keys)):
            try:
                signals = build_signals(
                    df=self.price_df,
                    turnover_window=rolling_days,
                    turnover_multiplier=volume_threshold,
                    momentum_window=20,
                    momentum_threshold_pct=5.0,
                    vol_window=20,
                    vol_multiplier=2.0,
                    mr_window=20,
                    mr_z=2.0,
                    enabled_algos=["Turnover Spike"],
                    combine_mode="OR"
                )
                signals_by_key[(rolling_days, volume_threshold)] = signals
                
                if (idx + 1) % max(1, signal_count // 5) == 0:
                    print(f"[신호 생성] {idx + 1}/{signal_count} 완료", file=sys.stderr)
            except Exception as e:
                print(f"[경고] Signal 생성 실패 (rolling_days={rolling_days}, volume_threshold={volume_threshold}): {e}", file=sys.stderr)
        
        print(f"[신호 생성] 완료! {len(signals_by_key)}개 signal 준비됨", file=sys.stderr)
        
        # 병렬 실행 설정
        num_workers = min(os.cpu_count() or 4, 8)  # 최대 8개 워커로 제한 (Windows 안정성)
        signals_cache_dict = {}  # (미사용 - signal은 이미 메인에서 생성)
        
        print(f"[병렬처리] multiprocessing.Pool 사용 (워커 수: {num_workers})", file=sys.stderr)
        print(f"[병렬처리] 총 파라미터 조합: {total_combinations}개", file=sys.stderr)
        
        # multiprocessing.Pool 사용 (Windows 호환성 더 좋음)
        try:
            # Windows 안정성을 위해 'spawn' context 명시 설정
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=num_workers) as pool:
                # worker 함수 호출용 헬퍼 함수
                def worker_wrapper(idx_and_combo):
                    i, combo = idx_and_combo
                    return _test_combination_worker(
                        i,
                        combo,
                        param_names,
                        self.price_df,
                        self.kospi_index,
                        start_date,
                        end_date,
                        initial_cash,
                        fee_rate,
                        slippage_rate,
                        sell_tax_rate,
                        open_pivot,
                        close_pivot,
                        kospi_bullish_dates,
                        signals_by_key,  # 미리 생성된 signal 딕셔너리
                    )
                
                # 작업 제출 (점진적 처리)
                combo_with_idx = [(i, combo) for i, combo in enumerate(combinations)]
                
                # imap_unordered로 완료된 작업부터 처리 (진행상황 반영)
                # chunksize를 20으로 증가 (CPU-bound 작업이므로 더 큰 chunk가 효율적)
                for result in pool.imap_unordered(worker_wrapper, combo_with_idx, chunksize=20):
                    completed_count += 1
                    
                    # 진행상황 업데이트
                    if progress_callback and result is not None:
                        progress_callback(completed_count, total_combinations, result.get('params', {}))
                    
                    if result is not None:
                        results.append(result)
                    
                    # 진행률 표시 (콘솔)
                    if completed_count % max(1, total_combinations // 20) == 0:
                        print(f"[진행] {completed_count}/{total_combinations} 완료 ({completed_count*100//total_combinations}%)", file=sys.stderr)
        
        except Exception as e:
            print(f"[경고] multiprocessing.Pool 사용 중 오류: {str(e)}", file=sys.stderr)
            print(f"[폴백] 순차 처리(Sequential)로 변경합니다.", file=sys.stderr)
            
            # 순차 처리로 폴백 (느리지만 안정적)
            for i, combo in enumerate(combinations):
                completed_count += 1
                
                result = _test_combination_worker(
                    i,
                    combo,
                    param_names,
                    self.price_df,
                    self.kospi_index,
                    start_date,
                    end_date,
                    initial_cash,
                    fee_rate,
                    slippage_rate,
                    sell_tax_rate,
                    open_pivot,
                    close_pivot,
                    kospi_bullish_dates,
                    signals_by_key,  # 미리 생성된 signal 딕셔너리
                )
                
                if progress_callback and result is not None:
                    progress_callback(completed_count, total_combinations, result.get('params', {}))
                
                if result is not None:
                    results.append(result)
        
        print(f"[완료] 최적화 완료 (결과: {len(results)}개)", file=sys.stderr)
        return pd.DataFrame(results)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """샤프 비율 계산 (Numba 가속화)"""
        if returns.empty or returns.std() == 0:
            return 0
        
        # Numba를 사용한 가속화
        if CUDA_AVAILABLE:
            try:
                return self._calculate_sharpe_ratio_gpu(returns.values, risk_free_rate)
            except Exception:
                pass
        
        # CPU 폴백
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
        return float(sharpe)
    
    @staticmethod
    def _calculate_sharpe_ratio_gpu(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        """GPU를 사용한 샤프 비율 계산"""
        try:
            # Numba CUDA 함수
            from numba import cuda, float64
            import numpy as np
            
            @cuda.jit(float64(float64[:], float64))
            def sharpe_gpu(ret_array, rf_rate):
                # GPU에서 평균, 표준편차 계산
                mean_val = 0.0
                for i in range(ret_array.size):
                    mean_val += ret_array[i]
                mean_val /= ret_array.size
                
                var_val = 0.0
                for i in range(ret_array.size):
                    var_val += (ret_array[i] - mean_val) ** 2
                var_val /= ret_array.size
                std_val = var_val ** 0.5
                
                if std_val == 0:
                    return 0.0
                
                excess_mean = mean_val - (rf_rate / 252.0)
                return excess_mean / std_val * (252.0 ** 0.5)
            
            returns_gpu = cuda.to_device(returns.astype(np.float64))
            result = sharpe_gpu[1, 1](returns_gpu, risk_free_rate)
            return float(result)
        except Exception as e:
            logger.debug(f"GPU Sharpe 계산 실패: {e}")
            return 0.0
    
    def _calculate_mdd(self, equity: pd.Series) -> float:
        """최대 낙폭(MDD) 계산 (Numba 가속화)"""
        if equity.empty:
            return 0
        
        if CUDA_AVAILABLE:
            try:
                return self._calculate_mdd_gpu(equity.values)
            except Exception:
                pass
        
        # CPU 폴백
        equity_array = equity.values.astype(np.float64)
        cummax = np.maximum.accumulate(equity_array)
        drawdown = ((equity_array - cummax) / cummax * 100)
        return float(np.min(drawdown))
    
    @staticmethod
    def _calculate_mdd_gpu(equity: np.ndarray) -> float:
        """GPU를 사용한 MDD 계산"""
        try:
            from numba import cuda, float64
            import numpy as np
            
            @cuda.jit(float64(float64[:]))
            def mdd_gpu(eq_array):
                # GPU에서 누적 최대값 계산 및 MDD 계산
                max_val = eq_array[0]
                min_dd = 0.0
                
                for i in range(1, eq_array.size):
                    if eq_array[i] > max_val:
                        max_val = eq_array[i]
                    
                    drawdown = (eq_array[i] - max_val) / max_val * 100.0
                    if drawdown < min_dd:
                        min_dd = drawdown
                
                return min_dd
            
            equity_gpu = cuda.to_device(equity.astype(np.float64))
            result = mdd_gpu[1, 1](equity_gpu)
            return float(result)
        except Exception as e:
            logger.debug(f"GPU MDD 계산 실패: {e}")
            return 0.0
    
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
            'add_buy_threshold_pct': float(best_row.get('add_buy_threshold_pct', -7.0)),
            f'best_{metric}': float(best_row[metric]),
            'best_total_return': float(best_row.get('total_return', 0)),
            'best_kospi_return': float(best_row.get('kospi_return', 0)),
            'best_excess_return': float(best_row.get('excess_return', 0)),
            'best_sharpe_ratio': float(best_row.get('sharpe_ratio', 0)),
            'best_mdd': float(best_row.get('mdd', 0)),
            'total_trades': int(best_row.get('total_trades', 0)),
            'win_rate': float(best_row.get('win_rate', 0))
        }


def get_period_dates(period: str, end_date: pd.Timestamp = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    투자 기간에 따른 시작일/종료일 계산
    
    Args:
        period: '2024', '2025', '2026' (연도) 또는 '1Y', '2Y', '3Y', '4Y', '5Y' (기간)
        end_date: 종료일 (None이면 최신 데이터 날짜)
        
    Returns:
        (시작일, 종료일) 튜플
    """
    if end_date is None:
        end_date = pd.Timestamp.now().normalize()
    
    # 연도 형식 (예: '2024')
    if period.isdigit() and len(period) == 4:
        year = int(period)
        start_date = pd.Timestamp(f'{year}-01-01')
        end_date = pd.Timestamp(f'{year}-12-31')
        return start_date, end_date
    
    # 기간 형식 (예: '1Y', '2Y')
    period_years = {
        '1Y': 1,
        '2Y': 2,
        '3Y': 3,
        '4Y': 4,
        '5Y': 5
    }
    
    if period in period_years:
        years = period_years[period]
        start_date = end_date - pd.DateOffset(years=years)
        return start_date, end_date
    
    raise ValueError(f"Invalid period: {period}. Use '2024', '2025', '2026' or '1Y', '2Y', '3Y', '4Y', '5Y'")


def get_available_years(df: pd.DataFrame) -> Dict[int, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    데이터에서 이용 가능한 연도 목록 반환 (해당 연도의 실제 데이터 범위 포함)
    
    Args:
        df: 주가 데이터프레임
        
    Returns:
        {연도: (시작일, 종료일)} 딕셔너리 (내림차순)
    """
    if df.empty or 'date' not in df.columns:
        return {}
    
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['date'])
    
    year_ranges = {}
    years = sorted(df_copy['date'].dt.year.unique(), reverse=True)
    
    for year in years:
        year_data = df_copy[df_copy['date'].dt.year == year]
        if not year_data.empty:
            start_date = year_data['date'].min()
            end_date = year_data['date'].max()
            year_ranges[year] = (start_date, end_date)
    
    return year_ranges


# ===== GPU 병렬처리 정보 및 유틸리티 =====

def get_gpu_info() -> Dict[str, any]:
    """GPU 정보 조회"""
    info = {
        'cuda_available': CUDA_AVAILABLE,
        'num_cpus': os.cpu_count() or 4,
        'processing_mode': 'GPU (Numba CUDA)' if CUDA_AVAILABLE else 'CPU (멀티프로세싱)',
    }
    
    if CUDA_AVAILABLE:
        try:
            from numba import cuda
            info['gpu_count'] = cuda.gpuci.get_numba_cuda_api_count() if hasattr(cuda, 'gpuci') else 1
            info['compute_capability'] = 'Available (CUDA 지원)'
        except Exception as e:
            info['gpu_info'] = f"GPU 정보 조회 실패: {e}"
    
    return info


def print_optimization_info():
    """최적화 설정 정보 출력"""
    gpu_info = get_gpu_info()
    
    print("\n" + "="*60)
    print("🚀 파라미터 최적화 설정")
    print("="*60)
    print(f"CPU 코어 수: {gpu_info['num_cpus']}")
    print(f"처리 모드: {gpu_info['processing_mode']}")
    
    if gpu_info['cuda_available']:
        print(f"✅ GPU (CUDA) 가속화 활성화")
        if 'gpu_count' in gpu_info:
            print(f"GPU 장치 수: {gpu_info['gpu_count']}")
    else:
        print("ℹ️ GPU 없음 → CPU 멀티프로세싱 사용")
    
    print(f"병렬처리 방식: ProcessPoolExecutor (GIL 우회)")
    print("="*60 + "\n")