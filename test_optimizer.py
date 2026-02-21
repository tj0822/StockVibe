import pandas as pd
from app.data import load_stock_data
from optimizer import BacktestOptimizer
import warnings
import sys
warnings.filterwarnings('ignore')

# Load data
df = load_stock_data('data')
kospi = pd.read_pickle('data/kospi.pkl') if __import__('os').path.exists('data/kospi.pkl') else pd.DataFrame()

print("Data loaded:", file=sys.stderr)
print(f"  df shape: {df.shape}", file=sys.stderr)
print(f"  df date range: {df['date'].min()} ~ {df['date'].max()}", file=sys.stderr)
print(f"  kospi shape: {kospi.shape}", file=sys.stderr)
print(f"  kospi date range: {kospi['date'].min() if not kospi.empty else 'N/A'} ~ {kospi['date'].max() if not kospi.empty else 'N/A'}", file=sys.stderr)

# Test optimization for 2026
optimizer = BacktestOptimizer(df, kospi)
start_date = pd.Timestamp('2026-01-02')
end_date = pd.Timestamp('2026-02-20')

param_ranges = {
    'max_daily_buys': [1, 2],
    'rolling_days': [5, 10],
    'volume_threshold': [2.0, 2.5],
    'add_buy_threshold_pct': [-10.0, -7.0],
}

print(f"\nOptimizing for {start_date.date()} ~ {end_date.date()}", file=sys.stderr)
results_df = optimizer.optimize_parameters(
    start_date=start_date,
    end_date=end_date,
    param_ranges=param_ranges,
    initial_cash=50_000_000,
    search_mode='grid',
)

print(f"\nResults: {len(results_df)} rows", file=sys.stderr)
if results_df.empty:
    print("WARNING: No results returned!", file=sys.stderr)
else:
    print(results_df.head(), file=sys.stderr)
