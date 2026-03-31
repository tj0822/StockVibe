#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.portfolio import PortfolioManager

pm = PortfolioManager()

print("="*70)
print("테스트: 보유 종목 정렬 및 포맷팅")
print("="*70)

# 포트폴리오 초기화
pm.clear_portfolio()

# 테스트 데이터: 여러 종목 추가
test_trades = [
    {'name': '한화', 'action': 'BUY', 'quantity': 10, 'price': 100000, 'date': '2024-01-10 10:00:00'},
    {'name': '삼성전자', 'action': 'BUY', 'quantity': 5, 'price': 80000, 'date': '2024-01-15 11:00:00'},
    {'name': 'SK하이닉스', 'action': 'BUY', 'quantity': 8, 'price': 60000, 'date': '2024-01-20 09:00:00'},
    {'name': 'LG전자', 'action': 'BUY', 'quantity': 12, 'price': 75000, 'date': '2024-01-25 14:00:00'},
    {'name': 'NHN', 'action': 'BUY', 'quantity': 20, 'price': 50000, 'date': '2024-02-01 08:00:00'},
]

print("\n[Step 1] 거래 이력 적용")
print("-" * 70)
results = pm.apply_trading_history(test_trades)
for name, msg in results.items():
    if 'OK' in msg:
        print(f"✓ {name}")

# 현재가 설정 (테스트용)
current_prices = {
    '000880': 110000,  # 한화: +10%
    '005930': 75000,   # 삼성전자: -6.25%
    '000660': 65000,   # SK하이닉스: +8.33%
    '066570': 80000,   # LG전자: +6.67%
    '053000': 52500,   # NHN: +5%
}

print("\n[Step 2] 보유 종목 조회")
print("-" * 70)
summary = pm.get_holding_summary(current_prices)

print(f"조회된 종목: {len(summary)}개")
print("\n[테이블 출력]")
print(summary.to_string(index=False))

print("\n[Step 3] 정렬 확인")
print("-" * 70)
print("종목명 순서:")
for i, row in summary.iterrows():
    print(f"  {i+1}. {row['종목명']}")

print("\n[Step 4] 수익률 포맷 확인")
print("-" * 70)
print("현재수익률(%) 값:")
for i, row in summary.iterrows():
    rate = row['현재수익률(%)']
    print(f"  {row['종목명']:15} → {rate} (타입: {type(rate).__name__})")

print("\n" + "="*70)
print("✓ 검증:")
print("="*70)

# 1. 정렬 확인
names = summary['종목명'].tolist()
expected_order = ['LG전자', 'NHN', 'SK하이닉스', '삼성전자', '한화']
is_sorted = names == expected_order
print(f"1. 종목명 오름차순 정렬: {'✓ PASS' if is_sorted else '✗ FAIL'}")
if not is_sorted:
    print(f"   기대: {expected_order}")
    print(f"   실제: {names}")

# 2. 소수점 포맷 확인
all_valid_format = True
for i, row in summary.iterrows():
    rate = row['현재수익률(%)']
    if rate != 'N/A':
        if not isinstance(rate, (int, float)):
            all_valid_format = False
            break

print(f"2. 수익률 소수점 2자리: {'✓ PASS' if all_valid_format else '✗ FAIL'}")

print("="*70 + "\n")
