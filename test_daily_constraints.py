#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""거래 제약 조건 시뮬레이션 테스트"""

from app.portfolio import PortfolioManager
import pandas as pd

def test_daily_trading_constraints():
    print("=" * 80)
    print("테스트: 일일 거래 제약 조건 시뮬레이션")
    print("=" * 80)
    
    portfolio_mgr = PortfolioManager()
    
    # 테스트 거래 데이터
    test_trades = [
        # 2026-01-15: 3개 종목 매수
        {
            'code': '005930',
            'name': '삼성전자',
            'type': 'BUY',
            'quantity': 10,
            'price': 2000000,  # 2천만원
            'date': '2026-01-15 09:00:00'
        },
        {
            'code': '000660',
            'name': 'SK하이닉스',
            'type': 'BUY',
            'quantity': 5,
            'price': 4000000,  # 2천만원
            'date': '2026-01-15 09:30:00'
        },
        {
            'code': '035420',
            'name': 'NAVER',
            'type': 'BUY',
            'quantity': 8,
            'price': 2500000,  # 2천만원
            'date': '2026-01-15 10:00:00'
        },
        # 2026-01-15: 4번째 매수 시도 (한도 초과)
        {
            'code': '051910',
            'name': 'LG화학',
            'type': 'BUY',
            'quantity': 4,
            'price': 5000000,  # 2천만원
            'date': '2026-01-15 10:30:00'
        },
        # 2026-01-15: 5번째 매수 시도 (한도 초과)
        {
            'code': '055550',
            'name': '신한지주',
            'type': 'BUY',
            'quantity': 20,
            'price': 1000000,  # 2천만원
            'date': '2026-01-15 11:00:00'
        },
        # 2026-01-15: 거래금액이 단위가 아닌 경우
        {
            'code': '066570',
            'name': 'LG전자',
            'type': 'BUY',
            'quantity': 7,
            'price': 1500000,  # 1천050만원 (200만의 배수 아님)
            'date': '2026-01-15 11:30:00'
        },
        # 2026-01-16: SELL은 제약이 없음
        {
            'code': '005930',
            'name': '삼성전자',
            'type': 'SELL',
            'quantity': 10,
            'price': 2100000,
            'date': '2026-01-16 15:00:00'
        },
        # 2026-01-16: 새로운 날짜이므로 한도 초기화
        {
            'code': '034220',
            'name': '카카오뱅크',
            'type': 'BUY',
            'quantity': 50,
            'price': 400000,  # 2천만원
            'date': '2026-01-16 09:00:00'
        },
    ]
    
    print("\n[단계 1] 테스트 데이터 요약")
    print(f"총 {len(test_trades)}개 거래")
    for i, trade in enumerate(test_trades, 1):
        date = trade['date'][:10]
        trade_type = '매수' if trade['type'] == 'BUY' else '매도'
        amount = trade['quantity'] * trade['price']
        print(f"  {i}. [{date}] {trade['name']:10} {trade_type:2} {trade['quantity']:3} units @ {trade['price']:>10,}원 = {amount:>15,}원")
    
    print("\n[단계 2-1] 시나리오 A: 하루 최대 3종목, 거래금액 단위 200만원")
    print("-" * 80)
    sim_result_a = portfolio_mgr.simulate_daily_trading_constraints(
        trades_list=test_trades,
        max_buy_stocks_per_day=3,
        transaction_unit=2000000
    )
    
    stats_a = sim_result_a['simulation_stats']
    print(f"  총 거래: {stats_a['total_trades']:2}건")
    print(f"  실행됨:  {stats_a['executed_count']:2}건 ✅")
    print(f"  스킵됨:  {stats_a['skipped_count']:2}건 ❌")
    print(f"  - 일일 한도 초과: {stats_a['max_daily_stock_limit_exceeded']:2}건")
    print(f"  - 거래금액 제약 위반: {stats_a['transaction_unit_constraint_violated']:2}건")
    
    print("\n  실행된 거래:")
    for trade in sim_result_a['executed_trades']:
        print(f"    ✅ [{trade['date'][:10]}] {trade['name']:10} {trade['type']} {trade['quantity']}units")
    
    print("\n  스킵된 거래:")
    for trade in sim_result_a['skipped_trades']:
        print(f"    ❌ [{trade['date'][:10]}] {trade['name']:10} {trade['type']} - {trade['skip_reason']}")
    
    print("\n[단계 2-2] 시나리오 B: 하루 최대 5종목, 거래금액 단위 300만원")
    print("-" * 80)
    sim_result_b = portfolio_mgr.simulate_daily_trading_constraints(
        trades_list=test_trades,
        max_buy_stocks_per_day=5,
        transaction_unit=3000000
    )
    
    stats_b = sim_result_b['simulation_stats']
    print(f"  총 거래: {stats_b['total_trades']:2}건")
    print(f"  실행됨:  {stats_b['executed_count']:2}건 ✅")
    print(f"  스킵됨:  {stats_b['skipped_count']:2}건 ❌")
    print(f"  - 일일 한도 초과: {stats_b['max_daily_stock_limit_exceeded']:2}건")
    print(f"  - 거래금액 제약 위반: {stats_b['transaction_unit_constraint_violated']:2}건")
    
    print("\n  실행된 거래:")
    for trade in sim_result_b['executed_trades']:
        print(f"    ✅ [{trade['date'][:10]}] {trade['name']:10} {trade['type']} {trade['quantity']}units")
    
    print("\n  스킵된 거래:")
    if sim_result_b['skipped_trades']:
        for trade in sim_result_b['skipped_trades']:
            print(f"    ❌ [{trade['date'][:10]}] {trade['name']:10} {trade['type']} - {trade['skip_reason']}")
    else:
        print("    (없음)")
    
    print("\n" + "=" * 80)
    print("✓ 시뮬레이션 테스트 완료")
    print("=" * 80)
    
    # 검증
    print("\n[검증 결과]")
    
    # 시나리오 A 검증
    expected_a = {
        'executed': 5,  # 3개 + SELL 1개 + 다음날 1개
        'skipped': 3    # 초과 2개 + 거래금액 1개
    }
    
    if stats_a['executed_count'] == expected_a['executed'] and stats_a['skipped_count'] == expected_a['skipped']:
        print(f"✓ PASS (시나리오 A): 예상과 일치")
    else:
        print(f"✗ FAIL (시나리오 A): 예상 {expected_a['executed']}/{expected_a['skipped']}, 실제 {stats_a['executed_count']}/{stats_a['skipped_count']}")
    
    # 시나리오 B 검증
    expected_b = {
        'executed': 6,  # 모든 매수 + SELL + 다음날
        'skipped': 2    # 거래금액 위반
    }
    
    if stats_b['executed_count'] == expected_b['executed'] and stats_b['skipped_count'] == expected_b['skipped']:
        print(f"✓ PASS (시나리오 B): 예상과 일치")
    else:
        print(f"✗ FAIL (시나리오 B): 예상 {expected_b['executed']}/{expected_b['skipped']}, 실제 {stats_b['executed_count']}/{stats_b['skipped_count']}")

if __name__ == '__main__':
    test_daily_trading_constraints()
