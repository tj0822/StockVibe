#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.portfolio import PortfolioManager

pm = PortfolioManager()

print("="*70)
print("테스트: 거래분석의 수익금액 표시")
print("="*70)

# 포트폴리오 초기화
pm.clear_portfolio()

# 테스트 데이터: 매수 후 매도
test_trades = [
    {'name': '삼성전자', 'action': 'BUY', 'quantity': 10, 'price': 75000, 'date': '2024-01-10 10:00:00'},
    {'name': '삼성전자', 'action': 'SELL', 'quantity': 5, 'price': 80000, 'date': '2024-01-15 11:00:00'},
    {'name': '삼성전자', 'action': 'SELL', 'quantity': 5, 'price': 82000, 'date': '2024-01-20 14:00:00'},
    {'name': 'LG전자', 'action': 'BUY', 'quantity': 20, 'price': 70000, 'date': '2024-01-25 09:00:00'},
    {'name': 'LG전자', 'action': 'SELL', 'quantity': 20, 'price': 66000, 'date': '2024-02-01 15:00:00'},
]

print("\n[Step 1] 거래 이력 적용")
results = pm.apply_trading_history(test_trades)
print(f"✓ {len(results)}개 거래 적용")

print("\n[Step 2] 거래분석 조회")
print("-" * 70)
analysis = pm.calculate_trade_analysis()

print(f"총 거래: {analysis['total_trades']}건 (매수 {analysis['buy_count']} / 매도 {analysis['sell_count']})")
print(f"승률: {analysis['win_rate']:.1f}%")
print(f"총 실현 손익: ₩{analysis['total_realized_pnl']:,}")

print("\n[Step 3] 매도 거래 상세 (수익금액 포함)")
print("-" * 70)
print(f"{'거래일시':<20} {'종목명':<10} {'수량':<6} {'매입가':<10} {'매도가':<10} {'수익금액':<10} {'수익률':<10}")
print("-" * 70)

sell_trades = analysis['all_sell_trades']
for trade in sell_trades:
    date = trade['date']
    name = trade['name']
    qty = trade['quantity']
    avg_price = trade['avg_buy_price']
    price = trade['price']
    pnl = trade['pnl']
    return_pct = trade['return_pct']
    
    print(f"{date:<20} {name:<10} {qty:<6} ₩{avg_price:>8,.0f}  ₩{price:>8,.0f}  ₩{pnl:>8,.0f}  {return_pct:>8.2f}%")

print("\n" + "="*70)
print("✓ 검증:")
print("="*70)

# 1. 거래일시 정렬 확인
dates = [t['date'] for t in sell_trades]
is_sorted = dates == sorted(dates)
print(f"1. 거래일시 오름차순 정렬: {'✓ PASS' if is_sorted else '✗ FAIL'}")

# 2. 수익금액 계산 확인
correct_calcs = True
for trade in sell_trades:
    expected_pnl = trade['quantity'] * (trade['price'] - trade['avg_buy_price'])
    if abs(trade['pnl'] - expected_pnl) > 0.01:
        correct_calcs = False
        break

print(f"2. 수익금액 계산 정확도: {'✓ PASS' if correct_calcs else '✗ FAIL'}")

# 3. 자료 구성 확인
has_pnl = all('pnl' in t for t in sell_trades)
print(f"3. 수익금액 필드 포함: {'✓ PASS' if has_pnl else '✗ FAIL'}")

print("="*70 + "\n")
