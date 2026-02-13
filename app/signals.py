import pandas as pd
import numpy as np

_GPU_AVAILABLE: bool | None = None


def _load_cudf():
    try:
        import cudf  # type: ignore
        return cudf
    except Exception:
        return None


def is_gpu_available() -> bool:
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        _GPU_AVAILABLE = _load_cudf() is not None
    return _GPU_AVAILABLE


def prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["turnover"] = data["close"] * data["volume"]
    data["candle"] = np.where(
        data["close"] > data["open"],
        "BULL",
        np.where(data["close"] < data["open"], "BEAR", "DOJI"),
    )
    return data


def add_turnover_spike_signals(
    data: pd.DataFrame,
    window: int,
    multiplier: float,
    use_gpu: bool | None = None,
) -> pd.DataFrame:
    if (use_gpu is not False) and is_gpu_available():
        try:
            cudf = _load_cudf()
            if cudf is None:
                raise RuntimeError("cudf unavailable")
            gdf = cudf.from_pandas(data[["code", "date", "turnover"]])
            gdf = gdf.sort_values(["code", "date"])
            rolling = (
                gdf.groupby("code")["turnover"]
                .rolling(window=window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )
            rolling = rolling.shift(1)
            gdf["avg_turnover"] = rolling
            gdf["turnover_spike"] = gdf["turnover"] >= gdf["avg_turnover"] * multiplier
            gdf["spike_ratio"] = gdf["turnover"] / gdf["avg_turnover"]

            gpu_out = gdf[["avg_turnover", "turnover_spike", "spike_ratio"]].to_pandas()
            data["avg_turnover"] = gpu_out["avg_turnover"].values
            data["turnover_spike"] = gpu_out["turnover_spike"].values
            data["spike_ratio"] = gpu_out["spike_ratio"].values
            return data
        except Exception:
            pass

    rolling_mean = (
        data.groupby("code")["turnover"]
        .rolling(window=window, min_periods=window)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    data["avg_turnover"] = rolling_mean
    data["turnover_spike"] = data["turnover"] >= data["avg_turnover"] * multiplier
    data["spike_ratio"] = data["turnover"] / data["avg_turnover"]
    return data


def add_momentum_signals(data: pd.DataFrame, window: int, threshold_pct: float) -> pd.DataFrame:
    returns = (
        data.groupby("code")["close"]
        .pct_change(periods=window)
        .reset_index(level=0, drop=True)
    )
    data["momentum_return"] = returns
    data["momentum_buy"] = data["momentum_return"] >= (threshold_pct / 100)
    data["momentum_sell"] = data["momentum_return"] <= -(threshold_pct / 100)
    return data


def add_volatility_breakout_signals(data: pd.DataFrame, window: int, multiplier: float) -> pd.DataFrame:
    rolling_std = (
        data.groupby("code")["close"]
        .rolling(window=window, min_periods=window)
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    data["price_std"] = rolling_std
    data["upper_band"] = data["close"].shift(1) + data["price_std"] * multiplier
    data["lower_band"] = data["close"].shift(1) - data["price_std"] * multiplier
    data["vol_breakout_buy"] = data["close"] >= data["upper_band"]
    data["vol_breakout_sell"] = data["close"] <= data["lower_band"]
    return data


def add_mean_reversion_signals(data: pd.DataFrame, window: int, z_threshold: float) -> pd.DataFrame:
    rolling_mean = (
        data.groupby("code")["close"]
        .rolling(window=window, min_periods=window)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    rolling_std = (
        data.groupby("code")["close"]
        .rolling(window=window, min_periods=window)
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    data["mr_mean"] = rolling_mean
    data["mr_std"] = rolling_std
    data["mr_zscore"] = (data["close"] - data["mr_mean"]) / data["mr_std"]
    data["mr_buy"] = data["mr_zscore"] <= -abs(z_threshold)
    data["mr_sell"] = data["mr_zscore"] >= abs(z_threshold)
    return data


def build_signals(
    df: pd.DataFrame,
    turnover_window: int,
    turnover_multiplier: float,
    momentum_window: int,
    momentum_threshold_pct: float,
    vol_window: int,
    vol_multiplier: float,
    mr_window: int,
    mr_z: float,
    enabled_algos: list[str],
    combine_mode: str,
    use_gpu: bool | None = None,
) -> pd.DataFrame:
    data = prepare_base_features(df)
    if "Turnover Spike" in enabled_algos:
        data = add_turnover_spike_signals(data, turnover_window, turnover_multiplier, use_gpu=use_gpu)
    if "Momentum" in enabled_algos:
        data = add_momentum_signals(data, momentum_window, momentum_threshold_pct)
    if "Volatility Breakout" in enabled_algos:
        data = add_volatility_breakout_signals(data, vol_window, vol_multiplier)
    if "Mean Reversion" in enabled_algos:
        data = add_mean_reversion_signals(data, mr_window, mr_z)

    buy_conditions = []
    sell_conditions = []

    if "Turnover Spike" in enabled_algos:
        buy_conditions.append(data["turnover_spike"] & (data["candle"] == "BULL"))
        sell_conditions.append(data["turnover_spike"] & (data["candle"] == "BEAR"))
    if "Momentum" in enabled_algos:
        buy_conditions.append(data["momentum_buy"])
        sell_conditions.append(data["momentum_sell"])
    if "Volatility Breakout" in enabled_algos:
        buy_conditions.append(data["vol_breakout_buy"])
        sell_conditions.append(data["vol_breakout_sell"])
    if "Mean Reversion" in enabled_algos:
        buy_conditions.append(data["mr_buy"])
        sell_conditions.append(data["mr_sell"])

    if not buy_conditions:
        data["signal"] = ""
        return data

    if combine_mode == "ALL":
        buy_mask = buy_conditions[0]
        sell_mask = sell_conditions[0]
        for cond in buy_conditions[1:]:
            buy_mask = buy_mask & cond
        for cond in sell_conditions[1:]:
            sell_mask = sell_mask & cond
    else:
        buy_mask = buy_conditions[0]
        sell_mask = sell_conditions[0]
        for cond in buy_conditions[1:]:
            buy_mask = buy_mask | cond
        for cond in sell_conditions[1:]:
            sell_mask = sell_mask | cond

    if "spike_ratio" in data.columns:
        data["spike_rank"] = data.groupby("date")["spike_ratio"].rank(method="dense", ascending=False)
        top_mask = data["spike_rank"] <= 10
    else:
        top_mask = True

    data["signal"] = ""
    data.loc[buy_mask & top_mask, "signal"] = "BUY"
    data.loc[sell_mask & top_mask, "signal"] = "SELL"

    return data
