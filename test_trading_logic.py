#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""거래이력 처리 로직 테스트"""

from app.portfolio import PortfolioManager

# 포트폴리오 초기화
pm = PortfolioManager()
pm.clear_portfolio()

# 테스트: 거래이력 파싱
test_text = """[키움증권 체결알림] [오전 9:30] [키움]체결통보
대웅제약
매수10주
평균단가185,100원

[키움증권 체결알림] [오전 10:15] [키움]체결통보
한미사이언스
매수5주
평균단가51,700원

[키움증권 체결알림] [오전 11:00] [키움]체결통보
대웅제약
매도5주
평균단가190,000원"""

print("=" * 60)
print("거래이력 처리 로직 테스트")
print("=" * 60)

trades = pm.parse_trading_history_text(test_text)

print("\n[OK] 파싱된 거래:")
for i, trade in enumerate(trades, 1):
    print(f"  {i}. {trade['name']} {trade['action']} {trade['quantity']}주 @ {trade['price']:,}원 ({trade['date']})")

# 테스트: 거래이력 적용
print("\n[OK] 거래 적용 결과:")
results = pm.apply_trading_history(trades)
for name, message in results.items():
    print(f"  {message}")

# 최종 포트폴리오 상태
print("\n[OK] 최종 포트폴리오 상태:")
portfolio = pm.load_portfolio()
for code, info in portfolio.items():
    print(f"\n  {info['name']} ({code})")
    print(f"    보유수량: {info['quantity']}주")
    print(f"    평균단가: {info['avg_price']:,.0f}원")
    if info.get('trades'):
        print(f"    거래 이력:")
        for trade in info['trades']:
            return_pct = trade.get('return_pct', 0)
            print(f"      - {trade['date']}: {trade['type']} {trade['quantity']}주 @ {trade['price']:,}원 (수익률: {return_pct:+.2f}%)")

# 거래 분석
print("\n[OK] 거래 분석:")
analysis = pm.calculate_trade_analysis()
print(f"  총 거래 수: {analysis['total_trades']}건 (매수: {analysis['buy_count']}, 매도: {analysis['sell_count']})")
print(f"  승률: {analysis['win_rate']:.1f}% ({analysis['winning_trades']}승 {analysis['losing_trades']}패)")
print(f"  총 수익률: {analysis['total_return_pct']:+.2f}%")
print(f"  총 실현 손익: {analysis['total_realized_pnl']:,.0f}원")

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)

