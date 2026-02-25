"""
Smoke tests for financial_utils.calculate_stock_score()

이 스크립트는 calculate_stock_score() 함수의 기본 동작을 검증합니다.
- 정상적인 입력 처리
- 결측값 처리
- 극단값 처리
- 일관된 출력 스키마
"""

import sys
import os

# 부모 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_utils import calculate_stock_score
import traceback
from typing import Dict, Any


# ============================================================================
# Test Cases 데이터
# ============================================================================

TEST_CASES = {
    "1_normal": {
        "description": "Normal realistic financial inputs (all metrics present)",
        "input": {
            'roe': 15.5,
            'debt_ratio': 45.0,
            'operating_margin': 12.3,
            'free_cash_flow': 1000000000,
            'per': 12.5,
            'pbr': 1.2,
            'roe_consistency': 85,
            'eps_growth': 8.5,
            'price_momentum_90d': 5.2,
            'sector': 'IT',
            'dividend_yield': 2.1,
            'current_price': 50000
        }
    },
    
    "2_missing_optional": {
        "description": "Missing optional fields (no dividend_yield, no bps)",
        "input": {
            'roe': 12.0,
            'debt_ratio': 50.0,
            'operating_margin': 10.0,
            'free_cash_flow': 800000000,
            'per': 14.0,
            'pbr': 1.5,
            'roe_consistency': 75,
            'eps_growth': 6.0,
            'price_momentum_90d': 3.0,
            'sector': 'Manufacturing'
            # dividend_yield, current_price 없음
        }
    },
    
    "3_zero_negatives": {
        "description": "Zero and negative values (negative EPS growth, negative FCF)",
        "input": {
            'roe': 5.0,
            'debt_ratio': 80.0,
            'operating_margin': -2.0,
            'free_cash_flow': -500000000,
            'per': 20.0,
            'pbr': 2.0,
            'roe_consistency': 40,
            'eps_growth': -5.0,
            'price_momentum_90d': -2.0,
            'sector': 'Healthcare'
        }
    },
    
    "4_high_per_pbr": {
        "description": "Extremely high PER/PBR values",
        "input": {
            'roe': 18.0,
            'debt_ratio': 30.0,
            'operating_margin': 15.0,
            'free_cash_flow': 1500000000,
            'per': 80.0,  # 매우 높음
            'pbr': 5.0,   # 매우 높음
            'roe_consistency': 90,
            'eps_growth': 12.0,
            'price_momentum_90d': 15.0,
            'sector': 'IT'
        }
    },
    
    "5_low_per_pbr": {
        "description": "Very low PER/PBR values (potential value)",
        "input": {
            'roe': 10.0,
            'debt_ratio': 60.0,
            'operating_margin': 8.0,
            'free_cash_flow': 500000000,
            'per': 5.0,   # 매우 낮음
            'pbr': 0.3,   # 매우 낮음
            'roe_consistency': 60,
            'eps_growth': 3.0,
            'price_momentum_90d': -1.0,
            'sector': 'Finance'
        }
    },
    
    "6_high_debt": {
        "description": "High debt ratio or None debt ratio",
        "input": {
            'roe': 8.0,
            'debt_ratio': 250.0,  # 매우 높은 부채
            'operating_margin': 6.0,
            'free_cash_flow': 100000000,
            'per': 18.0,
            'pbr': 1.8,
            'roe_consistency': 50,
            'eps_growth': 2.0,
            'price_momentum_90d': 0.5,
            'sector': 'Construction'
        }
    },
    
    "7_short_momentum": {
        "description": "Price series short length (momentum 0 or insufficient history)",
        "input": {
            'roe': 14.0,
            'debt_ratio': 40.0,
            'operating_margin': 11.0,
            'free_cash_flow': 900000000,
            'per': 13.0,
            'pbr': 1.3,
            'roe_consistency': 80,
            'eps_growth': 7.5,
            'price_momentum_90d': 0.0,  # 모멘텀 없음
            'sector': 'Retail'
        }
    },
    
    "8_unknown_sector": {
        "description": "Industry info missing or unknown sector",
        "input": {
            'roe': 13.0,
            'debt_ratio': 55.0,
            'operating_margin': 9.5,
            'free_cash_flow': 700000000,
            'per': 15.5,
            'pbr': 1.4,
            'roe_consistency': 70,
            'eps_growth': 5.5,
            'price_momentum_90d': 2.0,
            'sector': 'UNKNOWN_SECTOR_XYZ'
        }
    },
}


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_output_schema(output: Dict[str, Any], test_case_name: str) -> bool:
    """
    출력이 올바른 스키마를 가지는지 검증
    
    Returns:
        bool: 검증 성공 여부
    """
    required_keys = ['final_score', 'grade', 'confidence', 'breakdown']
    
    for key in required_keys:
        if key not in output:
            print(f"  ❌ FAIL: Missing key '{key}'")
            return False
    
    # final_score 검증
    score = output['final_score']
    if not isinstance(score, (int, float)) or not (0 <= score <= 100):
        print(f"  ❌ FAIL: final_score={score} not in range [0,100]")
        return False
    
    # grade 검증
    grade = output['grade']
    valid_grades = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F']
    if grade not in valid_grades:
        print(f"  ❌ FAIL: grade={grade} not in {valid_grades}")
        return False
    
    # confidence 검증
    confidence = output['confidence']
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 100):
        print(f"  ❌ FAIL: confidence={confidence} not in range [0,100]")
        return False
    
    # breakdown 검증
    breakdown = output['breakdown']
    factor_keys = ['quality', 'valuation', 'growth', 'momentum', 'industry']
    for key in factor_keys:
        if key not in breakdown:
            print(f"  ❌ FAIL: breakdown missing key '{key}'")
            return False
        factor_score = breakdown[key]
        if not isinstance(factor_score, (int, float)) or not (0 <= factor_score <= 100):
            print(f"  ❌ FAIL: breakdown[{key}]={factor_score} not in range [0,100]")
            return False
    
    return True


def run_test_case(case_name: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 테스트 케이스 실행
    
    Returns:
        Dict with keys: name, success, score, grade, confidence, output
    """
    print(f"\n📋 Test: {case_name}")
    print(f"   {case_data['description']}")
    
    result = {
        'name': case_name,
        'success': False,
        'score': None,
        'grade': None,
        'confidence': None,
        'output': None,
        'error': None
    }
    
    try:
        # 함수 호출
        output = calculate_stock_score(case_data['input'])
        result['output'] = output
        
        # 스키마 검증
        if not assert_output_schema(output, case_name):
            print(f"  ❌ FAIL: Schema validation failed")
            return result
        
        # 값 추출
        result['score'] = output['final_score']
        result['grade'] = output['grade']
        result['confidence'] = output['confidence']
        result['success'] = True
        
        print(f"  ✅ PASS: score={result['score']}, grade={result['grade']}, confidence={result['confidence']}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ❌ FAIL: Exception raised")
        print(f"     {type(e).__name__}: {e}")
        traceback.print_exc()
    
    return result


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """모든 테스트 케이스 실행 및 보고"""
    
    print("=" * 80)
    print("🧪 SCORING SMOKE TESTS - financial_utils.calculate_stock_score()")
    print("=" * 80)
    
    results = []
    
    # 모든 테스트 케이스 실행
    for case_name, case_data in TEST_CASES.items():
        result = run_test_case(case_name, case_data)
        results.append(result)
    
    # 결과 요약 테이블
    print("\n" + "=" * 80)
    print("📊 SUMMARY TABLE")
    print("=" * 80)
    
    # 헤더
    print(f"{'Case':<20} {'Score':<10} {'Grade':<8} {'Conf%':<8} {'Status':<10}")
    print("-" * 80)
    
    pass_count = 0
    fail_count = 0
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        score_str = f"{result['score']}" if result['score'] is not None else "N/A"
        grade_str = result['grade'] if result['grade'] else "N/A"
        conf_str = f"{result['confidence']}" if result['confidence'] is not None else "N/A"
        
        print(f"{result['name']:<20} {score_str:<10} {grade_str:<8} {conf_str:<8} {status:<10}")
        
        if result['success']:
            pass_count += 1
        else:
            fail_count += 1
    
    print("-" * 80)
    print(f"TOTAL: {pass_count} passed, {fail_count} failed out of {len(results)} tests")
    print("=" * 80)
    
    # Factor Breakdown 샘플 표시
    if results[0]['success'] and results[0]['output']:
        print("\n📈 Factor Breakdown Example (Test 1 - Normal):")
        breakdown = results[0]['output']['breakdown']
        for factor, score in breakdown.items():
            print(f"   {factor:<15}: {score:>3} / 100")
    
    # 종료 코드
    if fail_count > 0:
        print("\n❌ TESTS FAILED")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
