#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.portfolio import PortfolioManager

pm = PortfolioManager()

# 신규 형식 + 기존 형식 혼합 테스트
test_text = '''2024년 2월 7일 오전 10:44
삼성전자
매수 6주
평균단가 75,200원

2024년 2월 7일 오전 10:45
SK하이닉스
매수 10주
평균단가 60,000원

--------------- 2026년 1월 15일 목요일 ---------------
[키움증권 체결알림] [오전 8:01] [키움]체결통보
한화
매수 16주
평균단가 127,100원

--------------- 2026년 1월 16일 금요일 ---------------
[키움증권 체결알림] [오전 9:30] [키움]체결통보
신한지주
매도 5주
평균단가 67,500원

2024년 4월 1일 오후 3:20
네이버
매수 8주
평균단가 80,000원
'''

print("="*60)
print("파서 테스트: 신규 형식(날짜+시간) + 기존 형식(구분선) 혼합")
print("="*60)

result = pm.parse_trading_history_text(test_text, default_date='2026-02-25')

print(f'\n\n=== 파싱 결과 (총 {len(result)}건) ===')
for i, trade in enumerate(result, 1):
    print(f'{i}. {trade["name"]:10} {trade["action"]:4} {trade["quantity"]:3}주 @ {trade["price"]:7,}원 | {trade["date"]}')

print("\n\n=== 검증 ===")
print(f"✓ 기대값: 5건 거래")
print(f"✓ 실제값: {len(result)}건 거래")
print(f"✓ 상태: {'PASS' if len(result) == 5 else 'FAIL'}")

# 구체적인 검증
expected = [
    ('삼성전자', 'BUY', 6, 75200, '2024-02-07'),
    ('SK하이닉스', 'BUY', 10, 60000, '2024-02-07'),
    ('한화', 'BUY', 16, 127100, '2026-01-15'),
    ('신한지주', 'SELL', 5, 67500, '2026-01-16'),
    ('네이버', 'BUY', 8, 80000, '2024-04-01'),
]

print("\n=== 세부 검증 ===")
all_valid = True
for i, (exp_name, exp_action, exp_qty, exp_price, exp_date) in enumerate(expected):
    if i < len(result):
        trade = result[i]
        
        name_match = trade['name'] == exp_name
        action_match = trade['action'] == exp_action
        qty_match = trade['quantity'] == exp_qty
        price_match = trade['price'] == exp_price
        date_match = trade['date'].startswith(exp_date)
        
        status = "✓" if (name_match and action_match and qty_match and price_match and date_match) else "✗"
        
        print(f"{status} 거래 {i+1}: {trade['name']} vs {exp_name} | "
              f"{trade['action']} vs {exp_action} | {trade['quantity']} vs {exp_qty} | "
              f"{trade['price']} vs {exp_price} | {trade['date'][:10]} vs {exp_date}")
        
        if not (name_match and action_match and qty_match and price_match and date_match):
            all_valid = False
    else:
        print(f"✗ 거래 {i+1}: 누락됨")
        all_valid = False

print(f"\n=== 최종 판정: {'PASS ✓' if all_valid else 'FAIL ✗'} ===")
