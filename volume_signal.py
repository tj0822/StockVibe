import argparse
import os
import pickle
import pandas as pd


def load_stock_data(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "stock.pkl")
    with open(path, "rb") as fh:
        df = pickle.load(fh)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("stock.pkl is not a DataFrame")
    required_cols = {"date", "code", "open", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in stock.pkl: {sorted(missing)}")
    return df


def load_kospi_list(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "kospi_list.pkl")
    with open(path, "rb") as fh:
        df = pickle.load(fh)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("kospi_list.pkl is not a DataFrame")
    required_cols = {"date", "code", "name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in kospi_list.pkl: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["code", "date"])
    # keep latest name per code
    latest = df.groupby("code").tail(1)[["code", "name"]]
    return latest


def compute_volume_spike_signals(
    df: pd.DataFrame,
    window: int,
    multiplier: float,
) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values(["code", "date"])

    # turnover = average price (open/close/high/low) * volume
    data["turnover"] = ((data["open"] + data["close"] + data["high"] + data["low"]) / 4) * data["volume"]

    # rolling mean of previous N trading days (exclude current day)
    rolling_mean = (
        data.groupby("code")["turnover"]
        .rolling(window=window, min_periods=window)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    data["avg_turnover"] = rolling_mean
    data = data.dropna(subset=["avg_turnover"])

    data["turnover_spike"] = data["turnover"] >= data["avg_turnover"] * multiplier
    data["spike_ratio"] = data["turnover"] / data["avg_turnover"]
    data["candle"] = data.apply(
        lambda r: "BULL" if r["close"] > r["open"] else ("BEAR" if r["close"] < r["open"] else "DOJI"),
        axis=1,
    )
    data["signal"] = data.apply(
        lambda r: "BUY" if r["turnover_spike"] and r["candle"] == "BULL" else (
            "SELL" if r["turnover_spike"] and r["candle"] == "BEAR" else ""
        ),
        axis=1,
    )

    signals = data[data["signal"] != ""].copy()
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Volume spike buy/sell signal generator")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--window", type=int, default=20, help="Rolling window (trading days)")
    parser.add_argument("--multiplier", type=float, default=2.0, help="Volume spike multiplier")
    parser.add_argument("--latest-only", action="store_true", help="Show only latest date signals")
    args = parser.parse_args()

    df = load_stock_data(args.data_dir)
    kospi_list = load_kospi_list(args.data_dir)
    signals = compute_volume_spike_signals(df, args.window, args.multiplier)
    signals = signals.merge(kospi_list, on="code", how="left")

    # sort by date desc, spike_ratio desc
    signals = signals.sort_values(["date", "spike_ratio"], ascending=[False, False])

    if signals.empty:
        print("No signals found.")
        return

    # Basic summary
    total = len(signals)
    buy_cnt = (signals["signal"] == "BUY").sum()
    sell_cnt = (signals["signal"] == "SELL").sum()
    print(f"Signals: {total} (BUY: {buy_cnt}, SELL: {sell_cnt})")

    if args.latest_only:
        latest_date = signals["date"].max()
        latest = signals[signals["date"] == latest_date]
        print(f"Latest date: {latest_date.date()} | rows: {len(latest)}")
        print(latest[["date", "code", "name", "open", "close", "volume", "turnover", "avg_turnover", "spike_ratio", "signal"]].head(50))
    else:
        print(signals[["date", "code", "name", "open", "close", "volume", "turnover", "avg_turnover", "spike_ratio", "signal"]].head(50))



if __name__ == "__main__":
    main()
