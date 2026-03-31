#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""거래 중복 감지 테스트"""

from app.portfolio import PortfolioManager

def test_duplicate_detection():
    print("=" * 70)
    print("테스트: 거래 중복 감지")
    print("=" * 70)
    
    portfolio_mgr = PortfolioManager()
    
    # 테스트 거래 데이터
    test_trades = [
        # 삼성전자 (이미 포트폴리오에 있는 종목의 기존 거래와 세 개 동일)
        {
            'name': 'LG',
            'action': 'BUY',
            'quantity': 24,
            'price': 84700.0,
            'date': '2025-12-17 09:01:00'  # 이미 기록된 거래와 동일
        },
        # 다른 거래 (새로운 거래)
        {
            'name': 'LG',
            'action': 'BUY',
            'quantity': 30,
            'price': 85000.0,
            'date': '2026-02-26 10:00:00'  # 새로운 거래
        },
        # 또 다른 중복 거래
        {
            'name': 'LG',
            'action': 'BUY',
            'quantity': 25,
            'price': 80300.0,
            'date': '2025-12-29 09:40:00'  # 이미 기록된 거래와 동일
        },
    ]
    
    print("\n[단계 1] 포트폴리오 현재 상태:")
    portfolio = portfolio_mgr.load_portfolio()
    print(f"포트폴리오에 기록된 종목: {list(portfolio.keys())}")
    if '003550' in portfolio:
        lg_trades = portfolio['003550'].get('trades', [])
        print(f"\nLG의 기록된 거래:")
        for i, trade in enumerate(lg_trades, 1):
            print(f"  {i}. {trade['type']} {trade['quantity']}주 @ {trade['price']:,}원 ({trade['date']})")
    
    print("\n[단계 2] 중복 거래 포함된 거래 리스트 적용:")
    results = portfolio_mgr.apply_trading_history(test_trades)
    
    print("\n[단계 3] 적용 결과:")
    for name, message in results.items():
        print(f"  {name}: {message}")
    
    print("\n[단계 4] 적용 후 포트폴리오 상태:")
    portfolio_after = portfolio_mgr.load_portfolio()
    if '003550' in portfolio_after:
        lg_trades_after = portfolio_after['003550'].get('trades', [])
        print(f"LG의 거래 (총 {len(lg_trades_after)}건):")
        for i, trade in enumerate(lg_trades_after, 1):
            print(f"  {i}. {trade['type']} {trade['quantity']}주 @ {trade['price']:,}원 ({trade['date']})")
    
    print("\n" + "=" * 70)
    print("✓ 중복 거래 감지 테스트 완료")
    print("=" * 70)
    
    # 검증
    expected_new_trades = 1  # 새로운 거래 1건만 추가되어야 함
    actual_new_trades = len(lg_trades_after) - len(lg_trades)
    
    if actual_new_trades == expected_new_trades:
        print(f"✓ PASS: 새로운 거래 {actual_new_trades}건만 추가됨 (예상: {expected_new_trades}건)")
    else:
        print(f"✗ FAIL: 새로운 거래 {actual_new_trades}건 추가됨 (예상: {expected_new_trades}건)")

if __name__ == '__main__':
    test_duplicate_detection()
