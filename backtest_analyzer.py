"""
백테스트 분석 유틸리티

여러 종목에 대한 배치 백테스트 및 결과 분석을 위한 함수들
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def run_backtest_batch(stocks_data: List[Dict], 
                       strategy_func,
                       **strategy_kwargs) -> pd.DataFrame:
    """
    여러 종목에 대한 배치 백테스트 실행
    
    Args:
        stocks_data: 종목 데이터 리스트
        strategy_func: 백테스트 함수
        **strategy_kwargs: 전략 파라미터
    
    Returns:
        백테스트 결과 DataFrame
    """
    results = []
    
    for stock in stocks_data:
        try:
            result = strategy_func(stock, **strategy_kwargs)
            if result:
                result['code'] = stock.get('code')
                result['name'] = stock.get('name')
                results.append(result)
        except Exception as e:
            print(f"Error processing {stock.get('name')}: {e}")
            continue
    
    return pd.DataFrame(results)


def deduplicate_results(results_df: pd.DataFrame, 
                       score_col: str = "final_score",
                       return_col: str = "total_return") -> pd.DataFrame:
    """
    종목별 최고 수익률 결과만 유지 (중복 제거)
    
    Args:
        results_df: 백테스트 결과 DataFrame
        score_col: 점수 컬럼명
        return_col: 수익률 컬럼명
    
    Returns:
        중복 제거된 DataFrame
    """
    if results_df.empty:
        return results_df
    
    # 종목별로 그룹화하여 최고 수익률 행만 유지
    return results_df.loc[results_df.groupby('code')[return_col].idxmax()]


def calculate_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    수익률 배열로부터 기본 지표 계산
    
    Args:
        returns: 일일 수익률 배열 (%)
    
    Returns:
        Dict with: win_rate, avg_return, median_return, max_drawdown
    """
    if len(returns) == 0:
        return {
            'win_rate': 0,
            'avg_return': 0,
            'median_return': 0,
            'max_drawdown': 0
        }
    
    # Win rate
    wins = np.sum(returns > 0)
    win_rate = (wins / len(returns) * 100) if len(returns) > 0 else 0
    
    # Average return
    avg_return = np.mean(returns)
    
    # Median return
    median_return = np.median(returns)
    
    # Cumulative return
    cum_returns = np.cumprod(1 + returns / 100) - 1
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / (1 + running_max) * 100
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    
    return {
        'win_rate': win_rate,
        'avg_return': avg_return,
        'median_return': median_return,
        'max_drawdown': max_drawdown
    }


def score_bucket_report(results_df: pd.DataFrame, 
                       score_col: str = "final_score",
                       return_col: str = "total_return",
                       buckets: Optional[List[int]] = None) -> pd.DataFrame:
    """
    점수 버킷별 백테스트 성과 보고서
    
    높은 점수가 더 좋은 성과를 내는지 검증하는 sanity check 함수
    
    Args:
        results_df: 백테스트 결과 DataFrame
            - final_score (0~100): 종목의 점수
            - return_col: 수익률 (예: total_return, avg_return등)
            - 선택사항: win_rate, max_drawdown 등
        score_col: 점수 컬럼명
        return_col: 수익률 컬럼명
        buckets: 버킷 경계값. 기본: [0,20,40,60,80,100]
    
    Returns:
        pd.DataFrame with columns:
            bucket, count, win_rate, avg_return, median_return, max_drawdown
    """
    
    if results_df.empty:
        print("Warning: Empty results_df")
        return pd.DataFrame()
    
    # 기본 버킷 설정
    if buckets is None:
        buckets = [0, 20, 40, 60, 80, 100]
    
    # 점수 컬럼 존재 확인
    if score_col not in results_df.columns:
        raise ValueError(f"Column '{score_col}' not found in results_df")
    
    if return_col not in results_df.columns:
        raise ValueError(f"Column '{return_col}' not found in results_df")
    
    report_rows = []
    
    # 각 버킷별 처리
    for i in range(len(buckets) - 1):
        lower = buckets[i]
        upper = buckets[i + 1]
        bucket_label = f"{lower}-{upper}"
        
        # 이 버킷에 해당하는 데이터 필터링
        mask = (results_df[score_col] >= lower) & (results_df[score_col] < upper)
        bucket_data = results_df[mask]
        
        count = len(bucket_data)
        
        if count == 0:
            # 빈 버킷 처리
            report_rows.append({
                'bucket': bucket_label,
                'count': 0,
                'win_rate': np.nan,
                'avg_return': np.nan,
                'median_return': np.nan,
                'max_drawdown': np.nan
            })
        else:
            returns = bucket_data[return_col].values
            
            # Win rate 계산
            if 'win_rate' in bucket_data.columns:
                # 이미 계산되어 있으면 평균
                win_rate = bucket_data['win_rate'].mean()
            else:
                # 수익률로부터 계산
                win_rate = (np.sum(returns > 0) / count) * 100
            
            avg_return = np.mean(returns)
            median_return = np.median(returns)
            
            # Max drawdown
            if 'max_drawdown' in bucket_data.columns:
                max_drawdown = bucket_data['max_drawdown'].mean()
            else:
                max_drawdown = np.nan
            
            report_rows.append({
                'bucket': bucket_label,
                'count': count,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'median_return': median_return,
                'max_drawdown': max_drawdown
            })
    
    return pd.DataFrame(report_rows)


# ============================================================================
# Example / Main
# ============================================================================

if __name__ == "__main__":
    # 예제: Score bucket report 실행 예시
    
    print("=" * 80)
    print("📊 Score Bucket Report - Example")
    print("=" * 80)
    
    # 샘플 데이터 생성
    sample_data = {
        'code': ['001'] * 5 + ['002'] * 5 + ['003'] * 5,
        'final_score': [10, 15, 25, 35, 45, 50, 55, 65, 75, 85, 20, 30, 40, 60, 90],
        'total_return': [2.5, -1.2, 3.5, 5.2, 4.8, 6.1, 7.3, 5.9, 8.2, 9.5, -2.1, 2.8, 4.5, 7.1, 10.2],
        'win_rate': [60, 40, 70, 75, 80, 75, 80, 85, 85, 90, 50, 65, 75, 80, 95]
    }
    
    results_df = pd.DataFrame(sample_data)
    
    print("\nSample Results DataFrame:")
    print(results_df.to_string(index=False))
    
    # Score bucket report 생성
    bucket_report = score_bucket_report(results_df, 
                                       score_col="final_score",
                                       return_col="total_return")
    
    print("\n\n📈 Score Bucket Report:")
    print(bucket_report.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Key Insight: Do higher score buckets produce better returns?")
    print("If trend shows avg_return increasing with score buckets → ✅ Scoring works!")
    print("=" * 80)
    
    
    # ============================================================================
    # Ablation Test Example
    # ============================================================================
    
    print("\n\n" + "=" * 80)
    print("🧪 ABLATION TEST EXAMPLE - Factor Weight Impact Analysis")
    print("=" * 80)
    
    from financial_utils import StockScoringEngine, get_default_weights
    
    # 테스트 데이터 (예시)
    sample_stock = {
        'financial_dict': {
            'roe': 15.0,
            'debt_ratio': 45.0,
            'operating_margin': 12.0,
            'free_cash_flow': 1000000000
        },
        'current_per': 12.0,
        'industry_per': 15.0,
        'current_pbr': 1.2,
        'industry_pbr': 1.5,
    }
    
    print("\nScenario: Evaluate same stock with different factor weights\n")
    
    # 시나리오 1: 기본 가중치
    print("1️⃣  Baseline (Default Weights):")
    engine_default = StockScoringEngine()
    result_default = engine_default.calculate_final_score(**sample_stock)
    print(f"   Final Score: {result_default['final_score']}/100")
    print(f"   Weights: {engine_default.get_weights()}")
    
    # 시나리오 2: 산업 태풍 제거 (industry_tailwind = 0)
    print("\n2️⃣  Ablation Test (Industry Tailwind = 0):")
    ablation_weights_1 = {
        'quality': 0.25,
        'valuation': 0.25,
        'growth': 0.20,
        'momentum': 0.20,
        'industry_tailwind': 0.0  # 제거
    }
    engine_no_industry = StockScoringEngine(weights=ablation_weights_1)
    result_no_industry = engine_no_industry.calculate_final_score(**sample_stock)
    print(f"   Final Score: {result_no_industry['final_score']}/100")
    print(f"   Weights: {engine_no_industry.get_weights()}")
    print(f"   Impact: {result_default['final_score'] - result_no_industry['final_score']:+.1f} points")
    
    # 시나리오 3: 모멘텀 강조
    print("\n3️⃣  Momentum-Heavy Weights:")
    ablation_weights_2 = {
        'quality': 0.15,
        'valuation': 0.15,
        'growth': 0.15,
        'momentum': 0.50,  # 50%로 증가
        'industry_tailwind': 0.05
    }
    engine_momentum_heavy = StockScoringEngine(weights=ablation_weights_2)
    result_momentum_heavy = engine_momentum_heavy.calculate_final_score(**sample_stock)
    print(f"   Final Score: {result_momentum_heavy['final_score']}/100")
    print(f"   Weights: {engine_momentum_heavy.get_weights()}")
    print(f"   Impact: {result_default['final_score'] - result_momentum_heavy['final_score']:+.1f} points")
    
    # 시나리오 4: 가치주 중심
    print("\n4️⃣  Value-Focused Weights:")
    ablation_weights_3 = {
        'quality': 0.20,
        'valuation': 0.50,  # 50%로 증가
        'growth': 0.10,
        'momentum': 0.10,
        'industry_tailwind': 0.10
    }
    engine_value_heavy = StockScoringEngine(weights=ablation_weights_3)
    result_value_heavy = engine_value_heavy.calculate_final_score(**sample_stock)
    print(f"   Final Score: {result_value_heavy['final_score']}/100")
    print(f"   Weights: {engine_value_heavy.get_weights()}")
    print(f"   Impact: {result_default['final_score'] - result_value_heavy['final_score']:+.1f} points")
    
    print("\n" + "=" * 80)
    print("Key Insight: Which factors drive the score most significantly?")
    print("Use ablation tests to validate factor importance and optimize weights!")
    print("=" * 80)

