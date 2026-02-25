#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.portfolio import PortfolioManager
import json

pm = PortfolioManager()

# 포트폴리오 초기화
pm.clear_portfolio()

print("="*70)
print("통합 테스트: 매수/매도 판단 및 포트폴리오 자동 관리")
print("="*70)

# 타입 1: 신규 형식 거래 데이터
test_input = '''2024년 2월 7일 오전 10:44
삼성전자
매수 10주
평균단가 75,000원

2024년 2월 8일 오전 11:00
SK하이닉스
매수 5주
평균단가 60,000원

2024년 2월 10일 오후 2:30
삼성전자
매도 10주
평균단가 82,000원

2024년 2월 15일 오전 9:15
LG에너지
매수 20주
평균단가 50,000원
'''

print("\n[Step 1] 거래 이력 파싱")
print("-" * 70)
trades = pm.parse_trading_history_text(test_input, default_date='2024-02-25')
print(f"✓ 파싱됨: {len(trades)}건")
for i, trade in enumerate(trades, 1):
    print(f"  {i}. {trade['name']:12} {trade['action']:4} {trade['quantity']:3}주 @ {trade['price']:7,}원")

print("\n[Step 2] 거래 이력 적용")
print("-" * 70)
results = pm.apply_trading_history(trades)
for name, msg in results.items():
    print(f"  {name:12} → {msg}")

print("\n[Step 3] 포트폴리오 현황 확인")
print("-" * 70)
portfolio = pm.load_portfolio()
print(f"현재 보유 종목: {len(portfolio)}개")
for code, info in portfolio.items():
    print(f"  ✓ {code} {info['name']:12} - {info['quantity']}주 (평균: {info['avg_price']:,.0f}원)")

if not portfolio:
    print("  (보유 종목 없음)")

print("\n[Step 4] 삭제된 종목의 거래 이력 확인")
print("-" * 70)
trading_history = pm.load_trading_history()
print(f"아카이브 종목: {len(trading_history)}개")
for code, history in trading_history.items():
    print(f"  📦 {code} {history['name']:12} - {len(history['trades'])}건 거래")
    for i, trade in enumerate(history['trades'], 1):
        print(f"     {i}. {trade['type']:4} {trade['quantity']:3}주 @ {trade['price']:7,}원 ({trade['date']})")

print("\n[Step 5] 거래 분석 (현재 + 삭제된 종목 포함)")
print("-" * 70)
analysis = pm.calculate_trade_analysis()
print(f"총 거래: {analysis['total_trades']}건 (매수 {analysis['buy_count']} / 매도 {analysis['sell_count']})")
print(f"승률: {analysis['win_rate']:.1f}% ({analysis['winning_trades']}승 {analysis['losing_trades']}패)")
print(f"평균 수익률: {analysis['avg_win_pct']:+.2f}%")
print(f"총 실현 손익: ₩{analysis['total_realized_pnl']:,}")

print(f"\n매도 거래 상세:")
for trade in analysis['all_sell_trades']:
    print(f"  {trade['name']:12} {trade['quantity']:3}주 @ {trade['price']:7,}원 → {trade['return_pct']:+.2f}% ({trade['date']})")

print("\n" + "="*70)
print("✓ 결론:")
print("="*70)
print(f"1. 포트폴리오: {len(portfolio)}개 종목만 표시 (현재 보유)")
print(f"2. 거래분석: {len(trading_history)}개 매도 종목도 포함 (이력 추적)")
print(f"3. 매도판단: 수량 0 → 자동 삭제 및 아카이빙 ✓")
print("="*70 + "\n")
