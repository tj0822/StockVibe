#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.portfolio import PortfolioManager
import json
import os

pm = PortfolioManager()

print("="*70)
print("테스트: 완전 초기화 (full_reset) 기능")
print("="*70)

# Step 1: 현재 상태 확인
print("\n[Step 1] 초기화 전 상태")
print("-" * 70)
portfolio = pm.load_portfolio()
trading_history = pm.load_trading_history()
log_file = os.path.join(pm.data_dir, 'trading_input_log.json')

print(f"포트폴리오: {len(portfolio)}개 종목")
print(f"거래 이력: {len(trading_history)}개 종목")
print(f"입력 로그: {'있음' if os.path.exists(log_file) else '없음'}")

if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    print(f"  (로그 항목: {len(logs)}개)")

# Step 2: full_reset() 실행
print("\n[Step 2] 완전 초기화 실행")
print("-" * 70)
success = pm.full_reset()
print(f"결과: {'성공 ✓' if success else '실패 ✗'}")

# Step 3: 초기화 후 상태 확인
print("\n[Step 3] 초기화 후 상태")
print("-" * 70)
portfolio = pm.load_portfolio()
trading_history = pm.load_trading_history()

print(f"포트폴리오: {len(portfolio)}개 종목 (기대값: 0)")
print(f"거래 이력: {len(trading_history)}개 종목 (기대값: 0)")
print(f"입력 로그: {'있음 ✗' if os.path.exists(log_file) else '없음 ✓'}")

print("\n" + "="*70)
print("✓ 결론:")
print("="*70)
if len(portfolio) == 0 and len(trading_history) == 0 and not os.path.exists(log_file):
    print("완전 초기화 성공: 모든 데이터 삭제됨 ✓")
else:
    print("완전 초기화 실패: 일부 데이터가 남아있음 ✗")
print("="*70 + "\n")
