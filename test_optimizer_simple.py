import pandas as pd
from app.data import load_stock_data
from optimizer import BacktestOptimizer
import warnings
import sys
import time

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    # Load data
    print("Loading data...", file=sys.stderr)
    df = load_stock_data('data')
    kospi = pd.DataFrame()  # Empty kospi for this test
    
    # Test optimization for 2026
    optimizer = BacktestOptimizer(df, kospi)
    start_date = pd.Timestamp('2026-01-02')
    end_date = pd.Timestamp('2026-02-20')
    
    param_ranges = {
        'max_daily_buys': [1, 2],
        'rolling_days': [5],  # Just one value for quick test
        'volume_threshold': [2.0],  # Just one value for quick test
        'add_buy_threshold_pct': [-7.0],  # Just one value for quick test
    }
    
    print(f"Optimizing for {start_date.date()} ~ {end_date.date()}", file=sys.stderr)
    start_time = time.time()
    results_df = optimizer.optimize_parameters(
        start_date=start_date,
        end_date=end_date,
        param_ranges=param_ranges,
        initial_cash=50_000_000,
        search_mode='grid',
    )
    elapsed = time.time() - start_time
    
    print(f"\nResults: {len(results_df)} rows in {elapsed:.2f}s", file=sys.stderr)
    if results_df.empty:
        print("WARNING: No results returned!", file=sys.stderr)
    else:
        print("\nSuccess! First result:", file=sys.stderr)
        print(results_df.iloc[0].to_dict(), file=sys.stderr)
