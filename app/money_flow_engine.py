from __future__ import annotations

import numpy as np
import pandas as pd


class MoneyFlowEngine:
    """Capital flow detection engine for stocks and sectors."""

    def compute_stock_money_flow(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """Compute stock-level money flow metrics.

        Returns columns:
        - stock
        - sector
        - money_flow
        - flow_change
        - flow_momentum
        """
        if stock_df is None or stock_df.empty:
            return pd.DataFrame(
                columns=[
                    "code",
                    "stock",
                    "sector",
                    "money_flow",
                    "flow_change",
                    "flow_momentum",
                    "volume_spike",
                    "price_momentum",
                ]
            )

        work = stock_df.copy()
        if "date" in work.columns:
            work["date"] = pd.to_datetime(work["date"], errors="coerce")
        else:
            work["date"] = pd.NaT

        if "code" in work.columns:
            work["code"] = work["code"].astype(str).str.zfill(6)
            key_col = "code"
        else:
            work["code"] = ""
            key_col = "stock"

        stock_col = "name" if "name" in work.columns else ("stock" if "stock" in work.columns else "code")
        work["stock"] = work[stock_col].astype(str)

        if "sector" not in work.columns:
            work["sector"] = "Unknown"

        work["close"] = pd.to_numeric(work.get("close", np.nan), errors="coerce")
        work["volume"] = pd.to_numeric(work.get("volume", np.nan), errors="coerce")
        work = work.dropna(subset=["close", "volume"]).copy()
        if work.empty:
            return pd.DataFrame(
                columns=[
                    "code",
                    "stock",
                    "sector",
                    "money_flow",
                    "flow_change",
                    "flow_momentum",
                    "volume_spike",
                    "price_momentum",
                ]
            )

        work = work.sort_values([key_col, "date"]).copy()
        work["money_flow"] = work["close"] * work["volume"]

        work["prev_money_flow"] = work.groupby(key_col)["money_flow"].shift(1)
        work["flow_change"] = np.where(
            work["prev_money_flow"].abs() > 0,
            ((work["money_flow"] - work["prev_money_flow"]) / work["prev_money_flow"].abs()) * 100.0,
            0.0,
        )

        rolling_flow = work.groupby(key_col)["money_flow"].transform(lambda x: x.rolling(5, min_periods=2).mean())
        work["flow_momentum"] = np.where(
            rolling_flow.abs() > 0,
            ((work["money_flow"] - rolling_flow) / rolling_flow.abs()) * 100.0,
            0.0,
        )

        rolling_vol = work.groupby(key_col)["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
        work["volume_spike"] = np.where(
            rolling_vol.abs() > 0,
            ((work["volume"] - rolling_vol) / rolling_vol.abs()) * 100.0,
            0.0,
        )

        prev_close_20 = work.groupby(key_col)["close"].shift(20)
        work["price_momentum"] = np.where(
            prev_close_20.abs() > 0,
            ((work["close"] - prev_close_20) / prev_close_20.abs()) * 100.0,
            0.0,
        )

        latest = (
            work.sort_values([key_col, "date"])
            .groupby(key_col, as_index=False)
            .tail(1)
            .copy()
        )

        out = latest[
            [
                "code",
                "stock",
                "sector",
                "money_flow",
                "flow_change",
                "flow_momentum",
                "volume_spike",
                "price_momentum",
            ]
        ].copy()
        return out

    def compute_sector_money_flow(self, stock_flow_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate stock flow metrics to sector level."""
        if stock_flow_df is None or stock_flow_df.empty:
            return pd.DataFrame(columns=["sector", "sector_money_flow", "sector_flow_change"])

        work = stock_flow_df.copy()
        if "sector" not in work.columns:
            work["sector"] = "Unknown"

        agg = (
            work.groupby("sector", dropna=False)
            .agg(
                sector_money_flow=("money_flow", "sum"),
                sector_flow_change=("flow_change", "mean"),
                flow_momentum=("flow_momentum", "mean"),
                volume_spike=("volume_spike", "mean"),
                price_momentum=("price_momentum", "mean"),
            )
            .reset_index()
        )
        return agg

    def compute_flow_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute flow score.

        flow_score =
            0.4 * flow_momentum +
            0.3 * flow_change +
            0.2 * volume_spike +
            0.1 * price_momentum
        """
        if df is None or df.empty:
            return df

        work = df.copy()
        flow_momentum = pd.to_numeric(self._series_or_default(work, "flow_momentum", 0.0), errors="coerce").fillna(0.0)
        flow_change_base = self._series_or_default(work, "flow_change", np.nan)
        if flow_change_base.isna().all():
            flow_change_base = self._series_or_default(work, "sector_flow_change", 0.0)
        flow_change = pd.to_numeric(flow_change_base, errors="coerce").fillna(0.0)
        volume_spike = pd.to_numeric(self._series_or_default(work, "volume_spike", 0.0), errors="coerce").fillna(0.0)
        price_momentum = pd.to_numeric(self._series_or_default(work, "price_momentum", 0.0), errors="coerce").fillna(0.0)

        # Robust normalization to 0-100 centered at 50
        fm_norm = self._normalize_centered(flow_momentum)
        fc_norm = self._normalize_centered(flow_change)
        vs_norm = self._normalize_centered(volume_spike)
        pm_norm = self._normalize_centered(price_momentum)

        work["flow_score"] = (
            0.4 * fm_norm
            + 0.3 * fc_norm
            + 0.2 * vs_norm
            + 0.1 * pm_norm
        )
        return work

    def rank_flow_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return sector flow ranking table."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["rank", "sector", "sector_money_flow", "sector_flow_change", "flow_score"])

        work = df.copy()
        if "flow_score" not in work.columns:
            work = self.compute_flow_score(work)

        rank_df = work.sort_values("flow_score", ascending=False).reset_index(drop=True)
        rank_df.insert(0, "rank", range(1, len(rank_df) + 1))
        return rank_df

    def top_flow_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return top 20 stocks attracting capital."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["rank", "stock", "sector", "money_flow", "flow_change", "flow_momentum", "flow_score"])

        work = df.copy()
        if "flow_score" not in work.columns:
            work = self.compute_flow_score(work)

        top = work.sort_values(["flow_score", "money_flow"], ascending=[False, False]).head(20).reset_index(drop=True)
        top.insert(0, "rank", range(1, len(top) + 1))
        return top

    @staticmethod
    def _normalize_centered(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        if s.empty:
            return s
        scaled = 50.0 + (np.tanh(s / 50.0) * 50.0)
        return scaled.clip(0, 100)

    @staticmethod
    def _series_or_default(df: pd.DataFrame, column: str, default_value) -> pd.Series:
        if column in df.columns:
            return df[column]
        return pd.Series([default_value] * len(df), index=df.index)
