#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.portfolio import PortfolioManager
import json

pm = PortfolioManager()

# 포트폴리오 초기화
pm.clear_portfolio()

print("="*60)
print("테스트: 매수 후 완전 매도 시 포트폴리오에서 자동 삭제")
print("="*60)

# 테스트 1: 매수
print("\n[테스트 1] 삼성전자 매수")
result = pm.add_trade(
    code='005930',
    name='삼성전자',
    trade_type='BUY',
    quantity=10,
    price=75000,
    trade_date='2024-02-07 10:44:00'
)
print(f"결과: {result['message']}")

portfolio = pm.load_portfolio()
print(f"포트폴리오 현황: {list(portfolio.keys())}")
print(f"삼성전자 보유 수량: {portfolio['005930']['quantity']}주")

# 테스트 2: 부분 매도
print("\n[테스트 2] 삼성전자 부분 매도 (5주)")
result = pm.add_trade(
    code='005930',
    name='삼성전자',
    trade_type='SELL',
    quantity=5,
    price=80000,
    trade_date='2024-02-08 14:30:00'
)
print(f"결과: {result['message']}")

portfolio = pm.load_portfolio()
print(f"포트폴리오에 있는 종목: {list(portfolio.keys())}")
if '005930' in portfolio:
    print(f"삼성전자 보유 수량: {portfolio['005930']['quantity']}주 (아직 남아있음)")
else:
    print(f"삼성전자: 포트폴리오에 없음")

# 테스트 3: 완전 매도 (포트폴리오에서 삭제되어야 함)
print("\n[테스트 3] 삼성전자 완전 매도 (5주)")
result = pm.add_trade(
    code='005930',
    name='삼성전자',
    trade_type='SELL',
    quantity=5,
    price=82000,
    trade_date='2024-02-09 09:15:00'
)
print(f"결과: {result['message']}")

portfolio = pm.load_portfolio()
print(f"포트폴리오에 있는 종목: {list(portfolio.keys())}")
if '005930' in portfolio:
    print(f"✗ FAIL: 삼성전자가 아직 포트폴리오에 있음 (수량: {portfolio['005930']['quantity']}주)")
else:
    print(f"✓ PASS: 삼성전자가 포트폴리오에서 제거됨")

# 테스트 4: 거래 이력이 백업되었는지 확인
print("\n[테스트 4] 거래 이력 백업 확인")
trading_history = pm.load_trading_history()
if '005930' in trading_history:
    trades = trading_history['005930']['trades']
    print(f"✓ PASS: 거래 이력 백업됨 (총 {len(trades)}건)")
    for i, trade in enumerate(trades, 1):
        print(f"  {i}. {trade['type']} {trade['quantity']}주 @ {trade['price']:,}원 ({trade['date']})")
else:
    print(f"✗ FAIL: 거래 이력 백업이 없음")

print("\n" + "="*60)
print("테스트 완료")
print("="*60)
