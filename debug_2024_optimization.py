"""
2024년 최적화 문제 디버깅 스크립트
"""
import pandas as pd
import os
import sys
from datetime import datetime

# 데이터 파일 경로
data_dir = "data"

# 1. 데이터 파일 존재 확인
print("=" * 80)
print("1. 데이터 파일 확인")
print("=" * 80)

price_df = None
kospi_index = None

try:
    stock_file = os.path.join(data_dir, "stock.pkl")
    if os.path.exists(stock_file):
        price_df = pd.read_pickle(stock_file)
        print(f"✅ stock.pkl 로드 성공")
        print(f"   - 행 수: {len(price_df)}")
        print(f"   - 컬럼: {list(price_df.columns)}")
        print(f"   - 날짜 범위: {price_df['date'].min()} ~ {price_df['date'].max()}")
        print(f"   - 종목 수: {price_df['code'].nunique()}")
    else:
        print(f"❌ stock.pkl 파일이 없습니다")
except Exception as e:
    print(f"❌ stock.pkl 로드 실패: {e}")

try:
    kospi_file = os.path.join(data_dir, "kospi_index.pkl")
    if os.path.exists(kospi_file):
        kospi_index = pd.read_pickle(kospi_file)
        print(f"\n✅ kospi_index.pkl 로드 성공")
        print(f"   - 행 수: {len(kospi_index)}")
        print(f"   - 컬럼: {list(kospi_index.columns)}")
        print(f"   - 날짜 범위: {kospi_index['date'].min()} ~ {kospi_index['date'].max()}")
    else:
        print(f"\n❌ kospi_index.pkl 파일이 없습니다")
except Exception as e:
    print(f"\n❌ kospi_index.pkl 로드 실패: {e}")

if price_df is None or kospi_index is None:
    print("\n❌ 필수 데이터가 없습니다!")
    sys.exit(1)

# 2. 2024년 데이터 확인
print("\n" + "=" * 80)
print("2. 2024년 데이터 확인")
print("=" * 80)

price_df['date'] = pd.to_datetime(price_df['date'])
kospi_index['date'] = pd.to_datetime(kospi_index['date'])

year_2024_price = price_df[(price_df['date'].dt.year == 2024)]
year_2024_kospi = kospi_index[(kospi_index['date'].dt.year == 2024)]

print(f"2024년 주가 데이터:")
print(f"   - 행 수: {len(year_2024_price)}")
if not year_2024_price.empty:
    print(f"   - 날짜 범위: {year_2024_price['date'].min()} ~ {year_2024_price['date'].max()}")
    print(f"   - 종목 수: {year_2024_price['code'].nunique()}")
else:
    print(f"   ❌ 2024년 주가 데이터가 없습니다!")

print(f"\n2024년 KOSPI 데이터:")
print(f"   - 행 수: {len(year_2024_kospi)}")
if not year_2024_kospi.empty:
    print(f"   - 날짜 범위: {year_2024_kospi['date'].min()} ~ {year_2024_kospi['date'].max()}")
else:
    print(f"   ❌ 2024년 KOSPI 데이터가 없습니다!")

if year_2024_price.empty or year_2024_kospi.empty:
    print("\n❌ 2024년 데이터가 부족합니다!")
    sys.exit(1)

# 3. 시그널 생성 테스트
print("\n" + "=" * 80)
print("3. 시그널 생성 테스트")
print("=" * 80)

try:
    from app.signals import build_signals
    
    # 테스트 파라미터 (자주 사용되는 값)
    test_params = {
        'rolling_days': 10,
        'volume_threshold': 2.0,
    }
    
    print(f"시그널 생성 중... (rolling_days={test_params['rolling_days']}, volume_threshold={test_params['volume_threshold']})")
    
    signals = build_signals(
        df=price_df,
        turnover_window=test_params['rolling_days'],
        turnover_multiplier=test_params['volume_threshold'],
        momentum_window=20,
        momentum_threshold_pct=5.0,
        vol_window=20,
        vol_multiplier=2.0,
        mr_window=20,
        mr_z=2.0,
        enabled_algos=["Turnover Spike"],
        combine_mode="OR"
    )
    
    print(f"✅ 시그널 생성 성공")
    print(f"   - 전체 시그널 수: {len(signals)}")
    if not signals.empty:
        print(f"   - 날짜 범위: {signals['date'].min()} ~ {signals['date'].max()}")
        print(f"   - BUY 신호: {len(signals[signals['signal'] == 'BUY'])}")
        print(f"   - SELL 신호: {len(signals[signals['signal'] == 'SELL'])}")
    
    # 2024년 시그널 확인
    signals['date'] = pd.to_datetime(signals['date'])
    year_2024_signals = signals[(signals['date'].dt.year == 2024)]
    print(f"\n2024년 시그널:")
    print(f"   - 시그널 수: {len(year_2024_signals)}")
    if not year_2024_signals.empty:
        print(f"   - BUY 신호: {len(year_2024_signals[year_2024_signals['signal'] == 'BUY'])}")
        print(f"   - SELL 신호: {len(year_2024_signals[year_2024_signals['signal'] == 'SELL'])}")
    else:
        print(f"   ❌ 2024년 시그널이 없습니다!")
        
except Exception as e:
    print(f"❌ 시그널 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 백테스트 데이터 가용성 확인
print("\n" + "=" * 80)
print("4. 백테스트 데이터 가용성 확인")
print("=" * 80)

# 2024-01-02 ~ 2024-12-30 기간 확인
start_date = pd.Timestamp('2024-01-02')
end_date = pd.Timestamp('2024-12-30')

# price_df에서 pivot 생성 (백테스트가 하는 방식)
price_data = price_df[["date", "code", "open", "close"]].copy()
price_data["date"] = pd.to_datetime(price_data["date"], errors="coerce").dt.normalize()
price_data = price_data.dropna(subset=["date", "code", "open", "close"])
price_data = price_data.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last")

open_pivot = price_data.pivot(index="date", columns="code", values="open")
close_pivot = price_data.pivot(index="date", columns="code", values="close")

print(f"Pivot 데이터:")
print(f"   - open_pivot shape: {open_pivot.shape}")
print(f"   - 날짜 범위: {open_pivot.index.min()} ~ {open_pivot.index.max()}")

dates_in_range = [d for d in open_pivot.index if d >= start_date and d <= end_date]
print(f"\n2024년 (2024-01-02 ~ 2024-12-30) 범위 내 거래일:")
print(f"   - 거래일 수: {len(dates_in_range)}")
if len(dates_in_range) == 0:
    print(f"   ❌ 지정된 기간에 거래일이 없습니다!")
else:
    print(f"   - 첫 거래일: {dates_in_range[0]}")
    print(f"   - 마지막 거래일: {dates_in_range[-1]}")

# 5. 최적화 실행 시뮬레이션
print("\n" + "=" * 80)
print("5. 최적화 조합 테스트 (단일 조합)")
print("=" * 80)

try:
    from app.ui import run_turnover_strategy_backtest
    from optimizer import _test_combination_worker
    import json
    
    # 테스트 파라미터
    test_combo = (1, 10, 2.0, -7.0)  # max_daily_buys, rolling_days, volume_threshold, add_buy_threshold_pct
    param_names = ['max_daily_buys', 'rolling_days', 'volume_threshold', 'add_buy_threshold_pct']
    
    signals_cache_dict = {}
    kospi_bullish_dates = set()
    if not kospi_index.empty:
        ki = kospi_index.copy()
        ki["date"] = pd.to_datetime(ki["date"], errors="coerce").dt.normalize()
        ki = ki.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        ki["prev_index"] = ki["index"].shift(1)
        ki["is_bullish"] = ki["index"] > ki["prev_index"]
        kospi_bullish_dates = set(ki[ki["is_bullish"]]["date"].values)
    
    print(f"테스트 파라미터: {dict(zip(param_names, test_combo))}")
    print(f"기간: {start_date.date()} ~ {end_date.date()}")
    
    result = _test_combination_worker(
        combo_idx=0,
        combo=test_combo,
        param_names=param_names,
        price_df=price_df,
        kospi_index=kospi_index,
        start_date=start_date,
        end_date=end_date,
        initial_cash=50_000_000,
        fee_rate=0.00015,
        slippage_rate=0.0005,
        sell_tax_rate=0.0018,
        open_pivot=open_pivot,
        close_pivot=close_pivot,
        kospi_bullish_dates=kospi_bullish_dates,
        signals_cache_dict=signals_cache_dict,
    )
    
    if result is not None:
        print(f"\n✅ 백테스트 성공!")
        print(f"   - 수익률: {result['total_return']:.2f}%")
        print(f"   - KOSPI 수익률: {result['kospi_return']:.2f}%")
        print(f"   - 초과 수익률: {result['excess_return']:.2f}%")
        print(f"   - 샤프 비율: {result['sharpe_ratio']:.2f}")
        print(f"   - MDD: {result['mdd']:.2f}")
        print(f"   - 총 거래: {result['total_trades']}")
        print(f"   - 승률: {result['win_rate']:.2f}%")
    else:
        print(f"\n❌ 백테스트 결과가 None입니다!")
        print(f"   stderr 출력을 확인하세요.")
        
except Exception as e:
    print(f"❌ 백테스트 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("디버깅 완료")
print("=" * 80)
