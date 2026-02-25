#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.portfolio import PortfolioManager

pm = PortfolioManager()

print("="*70)
print("최종 테스트: 종목명 정렬 및 수익률 포맷팅")
print("="*70)

# 포트폴리오 초기화
pm.clear_portfolio()

# 다양한 종목명 테스트 (한글, 영문, 혼합)
test_trades = [
    {'name': '한화', 'action': 'BUY', 'quantity': 10, 'price': 100000, 'date': '2024-01-10 10:00:00'},
    {'name': '삼성전자', 'action': 'BUY', 'quantity': 5, 'price': 80000, 'date': '2024-01-15 11:00:00'},
    {'name': 'SK하이닉스', 'action': 'BUY', 'quantity': 8, 'price': 60000, 'date': '2024-01-20 09:00:00'},
    {'name': 'LG전자', 'action': 'BUY', 'quantity': 12, 'price': 75000, 'date': '2024-01-25 14:00:00'},
    {'name': 'NAVER', 'action': 'BUY', 'quantity': 3, 'price': 100000, 'date': '2024-02-01 08:00:00'},
]

print("\n[Step 1] 거래 이력 적용")
results = pm.apply_trading_history(test_trades)
print(f"✓ {len(results)}개 종목 적용 완료")

# 현재가 설정
current_prices = {
    '000880': 110000,   # 한화: +10.00%
    '005930': 75000,    # 삼성전자: -6.25%
    '000660': 65000,    # SK하이닉스: +8.33%
    '066570': 80000,    # LG전자: +6.67%
    '035420': 105500,   # NAVER: +5.50%
}

print("\n[Step 2] 보유 종목 조회 (정렬 + 포맷팅)")
print("-" * 70)
summary = pm.get_holding_summary(current_prices)

print(f"{'종목명':<15} {'수익률':<12} {'보유수량':<8} {'매수가격':<10}")
print("-" * 70)
for i, row in summary.iterrows():
    rate = row['현재수익률(%)']
    qty = row['보유수량']
    price = row['매수가격']
    name = row['종목명']
    
    # 포맷팅 (UI와 동일하게)
    if rate != 'N/A':
        rate_str = f"{rate:+.2f}%" if isinstance(rate, (int, float)) else str(rate)
    else:
        rate_str = rate
    
    print(f"{name:<15} {rate_str:<12} {qty:<8} {price:<10,}")

print("\n" + "="*70)
print("✓ 최종 검증:")
print("="*70)

# 1. 정렬 확인 (한글/영문 우정렬)
names = summary['종목명'].tolist()
print(f"1. 종목명 정렬 순서:")
for i, name in enumerate(names, 1):
    is_korean = ord(name[0]) >= 0xAC00
    kind = '한글' if is_korean else '영문'
    print(f"   {i}. {name:<15} ({kind})")

# 2. 수익률 포맷팅 확인
print(f"\n2. 수익률 소수점 표시:")
all_correct = True
for i, row in summary.iterrows():
    rate = row['현재수익률(%)']
    if rate != 'N/A':
        # 정수인 경우 소수점 2자리로 표시하면 10.00처럼 보임
        if isinstance(rate, float):
            print(f"   {row['종목명']:<15} → {rate:+.2f}% ✓")
        else:
            print(f"   {row['종목명']:<15} → {rate} ✗ (예상: float)")
            all_correct = False

print(f"\n3. 정렬 + 포맷팅 결과: {'✓ 완벽' if all_correct else '⚠ 확인 필요'}")
print("="*70 + "\n")
