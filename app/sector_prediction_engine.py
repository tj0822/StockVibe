from __future__ import annotations

import numpy as np
import pandas as pd

from app.money_flow_engine import MoneyFlowEngine


class SectorPredictionEngine:
    """Predict sectors likely to attract capital before broad price confirmation."""

    def compute_sector_features(
        self,
        sector_flow_df: pd.DataFrame,
        sector_rank_df: pd.DataFrame,
        stock_master_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build sector feature frame.

        Returns columns:
        - sector
        - sector_flow
        - flow_momentum
        - rank_change
        - leader_strength
        """
        base = pd.DataFrame(columns=["sector", "sector_flow", "flow_momentum", "rank_change", "leader_strength", "volume_spike", "sector_power"])

        sectors = set()
        if sector_flow_df is not None and not sector_flow_df.empty and "sector" in sector_flow_df.columns:
            sectors.update(sector_flow_df["sector"].dropna().astype(str).unique().tolist())
        if sector_rank_df is not None and not sector_rank_df.empty and "sector" in sector_rank_df.columns:
            sectors.update(sector_rank_df["sector"].dropna().astype(str).unique().tolist())
        if stock_master_df is not None and not stock_master_df.empty and "sector" in stock_master_df.columns:
            sectors.update(stock_master_df["sector"].dropna().astype(str).unique().tolist())

        if not sectors:
            return base

        out = pd.DataFrame({"sector": sorted(sectors)})

        # sector_flow, flow_momentum, volume_spike from money flow aggregation
        if sector_flow_df is not None and not sector_flow_df.empty:
            flow_df = sector_flow_df.copy()
            flow_df["sector"] = flow_df["sector"].astype(str)
            flow_df = flow_df.groupby("sector", as_index=False).agg(
                sector_flow=("sector_money_flow", "mean") if "sector_money_flow" in flow_df.columns else ("money_flow", "sum"),
                flow_momentum=("flow_momentum", "mean") if "flow_momentum" in flow_df.columns else ("sector_flow_change", "mean"),
                volume_spike=("volume_spike", "mean") if "volume_spike" in flow_df.columns else ("sector_flow_change", "mean"),
            )
            out = out.merge(flow_df, on="sector", how="left")
        else:
            out["sector_flow"] = 0.0
            out["flow_momentum"] = 0.0
            out["volume_spike"] = 0.0

        # rank_change: previous_rank - latest_rank (positive means improving)
        rank_change_df = pd.DataFrame(columns=["sector", "rank_change"])
        if sector_rank_df is not None and not sector_rank_df.empty and {"sector", "rank"}.issubset(set(sector_rank_df.columns)):
            rk = sector_rank_df.copy()
            rk["sector"] = rk["sector"].astype(str)
            rk["rank"] = pd.to_numeric(rk["rank"], errors="coerce")
            rk = rk.dropna(subset=["rank"])

            if "date" in rk.columns:
                rk["date"] = pd.to_datetime(rk["date"], errors="coerce")
                rk = rk.dropna(subset=["date"]).sort_values(["sector", "date"])
                rank_change_df = (
                    rk.groupby("sector", as_index=False)
                    .apply(lambda g: pd.Series({
                        "rank_change": (float(g.iloc[-2]["rank"]) - float(g.iloc[-1]["rank"])) if len(g) >= 2 else 0.0
                    }))
                    .reset_index(drop=True)
                )
            else:
                rank_change_df = rk.groupby("sector", as_index=False).agg(rank_change=("rank", "mean"))
                rank_change_df["rank_change"] = 0.0

        out = out.merge(rank_change_df, on="sector", how="left")

        # leader_strength: sector leaders with high momentum + money flow
        leaders = self.detect_leader_stocks(stock_master_df)
        if leaders is not None and not leaders.empty and "sector" in leaders.columns:
            ls = leaders.groupby("sector", as_index=False).agg(leader_strength=("leader_strength", "mean"))
            out = out.merge(ls, on="sector", how="left", suffixes=("", "_lead"))

        # sector_power from stock_master
        if stock_master_df is not None and not stock_master_df.empty and "sector" in stock_master_df.columns:
            sm = stock_master_df.copy()
            sm["sector"] = sm["sector"].astype(str)
            sm["sector_power"] = pd.to_numeric(self._series_or_default(sm, "sector_power", 50.0), errors="coerce").fillna(50.0)
            sp = sm.groupby("sector", as_index=False).agg(sector_power=("sector_power", "mean"))
            out = out.merge(sp, on="sector", how="left")
        else:
            out["sector_power"] = 50.0

        out["sector_flow"] = pd.to_numeric(self._series_or_default(out, "sector_flow", 0.0), errors="coerce").fillna(0.0)
        out["flow_momentum"] = pd.to_numeric(self._series_or_default(out, "flow_momentum", 0.0), errors="coerce").fillna(0.0)
        out["rank_change"] = pd.to_numeric(self._series_or_default(out, "rank_change", 0.0), errors="coerce").fillna(0.0)
        out["leader_strength"] = pd.to_numeric(self._series_or_default(out, "leader_strength", 50.0), errors="coerce").fillna(50.0)
        out["volume_spike"] = pd.to_numeric(self._series_or_default(out, "volume_spike", 0.0), errors="coerce").fillna(0.0)
        out["sector_power"] = pd.to_numeric(self._series_or_default(out, "sector_power", 50.0), errors="coerce").fillna(50.0)
        return out[["sector", "sector_flow", "flow_momentum", "rank_change", "leader_strength", "volume_spike", "sector_power"]]

    def compute_prediction_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """prediction_score = 0.35*flow_momentum + 0.25*rank_change + 0.2*volume_spike + 0.2*leader_strength"""
        if df is None or df.empty:
            return df

        work = df.copy()
        fm = self._normalize_centered(pd.to_numeric(self._series_or_default(work, "flow_momentum", 0.0), errors="coerce").fillna(0.0))
        rc = self._normalize_centered(pd.to_numeric(self._series_or_default(work, "rank_change", 0.0), errors="coerce").fillna(0.0) * 10.0)
        vs = self._normalize_centered(pd.to_numeric(self._series_or_default(work, "volume_spike", 0.0), errors="coerce").fillna(0.0))
        ls = pd.to_numeric(self._series_or_default(work, "leader_strength", 50.0), errors="coerce").fillna(50.0).clip(0, 100)

        work["prediction_score"] = (0.35 * fm) + (0.25 * rc) + (0.2 * vs) + (0.2 * ls)
        return work

    def classify_sector_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify sector state by rules.

        - prediction_score > 70 -> Emerging
        - sector_power > 75 -> Leading
        - prediction_score < 40 -> Weakening
        """
        if df is None or df.empty:
            return df

        work = df.copy()
        ps = pd.to_numeric(self._series_or_default(work, "prediction_score", 0.0), errors="coerce").fillna(0.0)
        sp = pd.to_numeric(self._series_or_default(work, "sector_power", 50.0), errors="coerce").fillna(50.0)

        state = np.where(
            sp > 75,
            "Leading",
            np.where(ps > 70, "Emerging", np.where(ps < 40, "Weakening", "Neutral")),
        )
        work["sector_state_pred"] = state
        return work

    def detect_leader_stocks(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """Detect sector leader stocks.

        Conditions:
        - momentum_score > 70
        - money_flow_score high
        """
        if stock_df is None or stock_df.empty:
            return pd.DataFrame(columns=["sector", "code", "stock", "momentum_score", "money_flow_score", "leader_strength"])

        work = stock_df.copy()
        if "sector" not in work.columns:
            work["sector"] = "Unknown"
        stock_col = "name" if "name" in work.columns else ("stock" if "stock" in work.columns else "code")
        work["stock"] = self._series_or_default(work, stock_col, "").astype(str)
        work["code"] = self._series_or_default(work, "code", "").astype(str).str.zfill(6)

        work["momentum_score"] = pd.to_numeric(self._series_or_default(work, "momentum_score", 0.0), errors="coerce").fillna(0.0)
        if "money_flow_score" in work.columns:
            work["money_flow_score"] = pd.to_numeric(work["money_flow_score"], errors="coerce").fillna(50.0)
        elif "flow_score" in work.columns:
            work["money_flow_score"] = pd.to_numeric(work["flow_score"], errors="coerce").fillna(50.0)
        else:
            work["money_flow_score"] = 50.0

        high_threshold = float(work["money_flow_score"].quantile(0.7)) if not work.empty else 60.0
        leaders = work[
            (work["momentum_score"] > 70)
            & (work["money_flow_score"] >= max(60.0, high_threshold))
        ].copy()

        if leaders.empty:
            return pd.DataFrame(columns=["sector", "code", "stock", "momentum_score", "money_flow_score", "leader_strength"])

        leaders["leader_strength"] = (0.5 * leaders["momentum_score"]) + (0.5 * leaders["money_flow_score"])
        leaders = leaders.sort_values(["sector", "leader_strength"], ascending=[True, False]).groupby("sector", as_index=False).head(3)
        return leaders[["sector", "code", "stock", "momentum_score", "money_flow_score", "leader_strength"]].reset_index(drop=True)

    def predict_future_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return top emerging sectors."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["sector", "prediction_score", "sector_state_pred"])

        work = df.copy()
        if "prediction_score" not in work.columns:
            work = self.compute_prediction_score(work)
        if "sector_state_pred" not in work.columns:
            work = self.classify_sector_state(work)

        out = work[work["sector_state_pred"].isin(["Emerging", "Leading"])].copy()
        if out.empty:
            out = work.copy()
        return out.sort_values("prediction_score", ascending=False).head(10).reset_index(drop=True)

    def build_prediction_snapshot(
        self,
        stock_df: pd.DataFrame,
        sector_rank_df: pd.DataFrame,
        sector_map_df: pd.DataFrame,
        as_of_date,
    ) -> pd.DataFrame:
        """Build a historical sector prediction snapshot as of a given date."""
        if stock_df is None or stock_df.empty:
            return pd.DataFrame()

        as_of_ts = pd.to_datetime(as_of_date, errors="coerce")
        if pd.isna(as_of_ts):
            return pd.DataFrame()

        px = stock_df.copy()
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px = px.dropna(subset=["date"])
        px = px[px["date"] <= as_of_ts].copy()
        if px.empty:
            return pd.DataFrame()

        px["code"] = px["code"].astype(str).str.zfill(6)
        if sector_map_df is not None and not sector_map_df.empty:
            sector_map = sector_map_df.copy()
            sector_map["code"] = sector_map["code"].astype(str).str.zfill(6)
            keep_cols = [c for c in ["code", "name", "sector"] if c in sector_map.columns]
            sector_map = sector_map[keep_cols].drop_duplicates(subset=["code"], keep="last")
            px = px.merge(sector_map, on="code", how="left")

        if "name" not in px.columns:
            px["name"] = px["code"]
        if "sector" not in px.columns:
            px["sector"] = "Unknown"

        # Historical stock-level money flow snapshot
        money_engine = MoneyFlowEngine()
        stock_flow_df = money_engine.compute_stock_money_flow(px)
        stock_flow_df = money_engine.compute_flow_score(stock_flow_df)
        sector_flow_df = money_engine.compute_sector_money_flow(stock_flow_df)
        sector_flow_df = money_engine.compute_flow_score(sector_flow_df)

        # Latest stock snapshot up to as_of_date with lightweight historical momentum proxy
        latest_px = (
            px.sort_values(["code", "date"])
            .groupby("code", as_index=False)
            .tail(1)
            .copy()
        )
        ret20 = px.sort_values(["code", "date"]).groupby("code")["close"].pct_change(20) * 100.0
        ret60 = px.sort_values(["code", "date"]).groupby("code")["close"].pct_change(60) * 100.0
        px_sorted = px.sort_values(["code", "date"]).copy()
        px_sorted["ret20"] = ret20.values
        px_sorted["ret60"] = ret60.values
        px_sorted["momentum_score"] = (
            50.0
            + 0.7 * px_sorted["ret20"].fillna(0.0)
            + 0.3 * px_sorted["ret60"].fillna(0.0)
        ).clip(0, 100)
        latest_momentum = (
            px_sorted.groupby("code", as_index=False)
            .tail(1)[["code", "momentum_score"]]
            .copy()
        )

        stock_snapshot = latest_px[[c for c in ["code", "name", "sector"] if c in latest_px.columns]].copy()
        stock_snapshot = stock_snapshot.merge(latest_momentum, on="code", how="left")
        if not stock_flow_df.empty:
            stock_snapshot = stock_snapshot.merge(
                stock_flow_df[["code", "flow_score"]].rename(columns={"flow_score": "money_flow_score"}),
                on="code",
                how="left",
            )

        # historical sector power from latest known rank snapshot up to as_of_date
        if sector_rank_df is not None and not sector_rank_df.empty and {"date", "sector", "rank"}.issubset(set(sector_rank_df.columns)):
            rk = sector_rank_df.copy()
            rk["date"] = pd.to_datetime(rk["date"], errors="coerce")
            rk = rk.dropna(subset=["date"])
            rk = rk[rk["date"] <= as_of_ts].copy()
            if not rk.empty:
                latest_rank_date = rk["date"].max()
                rank_slice = rk[rk["date"] == latest_rank_date].copy()
                rank_slice["rank"] = pd.to_numeric(rank_slice["rank"], errors="coerce")
                max_rank = float(rank_slice["rank"].max()) if not rank_slice.empty else 1.0
                if max_rank <= 1:
                    rank_slice["sector_power"] = 100.0
                else:
                    rank_slice["sector_power"] = 100.0 * (1.0 - (rank_slice["rank"] - 1.0) / (max_rank - 1.0))
                stock_snapshot = stock_snapshot.merge(
                    rank_slice[["sector", "sector_power"]].drop_duplicates(subset=["sector"], keep="last"),
                    on="sector",
                    how="left",
                )

        stock_snapshot["momentum_score"] = pd.to_numeric(self._series_or_default(stock_snapshot, "momentum_score", 50.0), errors="coerce").fillna(50.0)
        stock_snapshot["money_flow_score"] = pd.to_numeric(self._series_or_default(stock_snapshot, "money_flow_score", 50.0), errors="coerce").fillna(50.0)
        stock_snapshot["sector_power"] = pd.to_numeric(self._series_or_default(stock_snapshot, "sector_power", 50.0), errors="coerce").fillna(50.0)

        rank_input = pd.DataFrame(columns=["date", "sector", "rank"])
        if sector_rank_df is not None and not sector_rank_df.empty:
            rank_input = sector_rank_df.copy()
            if "date" in rank_input.columns:
                rank_input["date"] = pd.to_datetime(rank_input["date"], errors="coerce")
                rank_input = rank_input.dropna(subset=["date"])
                rank_input = rank_input[rank_input["date"] <= as_of_ts].copy()

        features_df = self.compute_sector_features(sector_flow_df, rank_input, stock_snapshot)
        scored_df = self.compute_prediction_score(features_df)
        classified_df = self.classify_sector_state(scored_df)
        classified_df["as_of_date"] = as_of_ts.normalize()
        return classified_df

    def backtest_prediction_accuracy(
        self,
        stock_df: pd.DataFrame,
        sector_rank_df: pd.DataFrame,
        sector_map_df: pd.DataFrame,
        horizon_months: tuple[int, ...] = (3, 6, 12),
        top_n: int = 3,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Backtest sector prediction accuracy using monthly snapshots.

        A prediction is counted as a hit when predicted sectors outperform the median
        sector forward return for the selected horizon.
        """
        if stock_df is None or stock_df.empty:
            empty_summary = pd.DataFrame(columns=["horizon", "samples", "hit_rate", "avg_predicted_return", "avg_benchmark_return"])
            empty_detail = pd.DataFrame(columns=["as_of_date", "horizon", "sector", "prediction_score", "future_return", "benchmark_return", "hit"])
            return empty_summary, empty_detail

        px = stock_df.copy()
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px = px.dropna(subset=["date"]).copy()
        px["code"] = px["code"].astype(str).str.zfill(6)

        if sector_map_df is not None and not sector_map_df.empty:
            sector_map = sector_map_df.copy()
            sector_map["code"] = sector_map["code"].astype(str).str.zfill(6)
            keep_cols = [c for c in ["code", "name", "sector"] if c in sector_map.columns]
            sector_map = sector_map[keep_cols].drop_duplicates(subset=["code"], keep="last")
            px = px.merge(sector_map, on="code", how="left")
        if "sector" not in px.columns:
            px["sector"] = "Unknown"

        unique_dates = pd.Series(sorted(px["date"].dropna().unique()))
        if unique_dates.empty:
            empty_summary = pd.DataFrame(columns=["horizon", "samples", "hit_rate", "avg_predicted_return", "avg_benchmark_return"])
            empty_detail = pd.DataFrame(columns=["as_of_date", "horizon", "sector", "prediction_score", "future_return", "benchmark_return", "hit"])
            return empty_summary, empty_detail

        # Monthly snapshot dates to keep runtime practical.
        monthly_dates = (
            pd.DataFrame({"date": unique_dates})
            .assign(month=lambda x: x["date"].dt.to_period("M"))
            .groupby("month", as_index=False)
            .tail(1)["date"]
            .sort_values()
            .tolist()
        )

        detail_rows: list[dict] = []
        for as_of_date in monthly_dates:
            snapshot_df = self.build_prediction_snapshot(px, sector_rank_df, sector_map_df, as_of_date)
            if snapshot_df is None or snapshot_df.empty:
                continue

            predicted_df = snapshot_df.sort_values("prediction_score", ascending=False).head(top_n).copy()
            if predicted_df.empty:
                continue

            for months in horizon_months:
                future_cutoff = pd.Timestamp(as_of_date) + pd.DateOffset(months=months)
                candidate_future = unique_dates[unique_dates >= future_cutoff]
                if candidate_future.empty:
                    continue
                future_date = pd.Timestamp(candidate_future.iloc[0])

                sector_forward_df = self._compute_sector_forward_returns(px, as_of_date, future_date)
                if sector_forward_df.empty:
                    continue

                benchmark_return = float(sector_forward_df["future_return"].median())
                merged = predicted_df.merge(sector_forward_df, on="sector", how="left")
                merged["future_return"] = pd.to_numeric(self._series_or_default(merged, "future_return", np.nan), errors="coerce")
                merged = merged.dropna(subset=["future_return"]).copy()
                if merged.empty:
                    continue

                merged["hit"] = merged["future_return"] > benchmark_return
                for _, row in merged.iterrows():
                    detail_rows.append(
                        {
                            "as_of_date": pd.Timestamp(as_of_date).normalize(),
                            "future_date": future_date.normalize(),
                            "horizon": f"{months}M",
                            "sector": row.get("sector"),
                            "prediction_score": float(pd.to_numeric(row.get("prediction_score"), errors="coerce")),
                            "future_return": float(pd.to_numeric(row.get("future_return"), errors="coerce")),
                            "benchmark_return": benchmark_return,
                            "hit": bool(row.get("hit", False)),
                        }
                    )

        detail_df = pd.DataFrame(detail_rows)
        if detail_df.empty:
            empty_summary = pd.DataFrame(columns=["horizon", "samples", "hit_rate", "avg_predicted_return", "avg_benchmark_return"])
            return empty_summary, detail_df

        summary_df = (
            detail_df.groupby("horizon", as_index=False)
            .agg(
                samples=("hit", "size"),
                hit_rate=("hit", "mean"),
                avg_predicted_return=("future_return", "mean"),
                avg_benchmark_return=("benchmark_return", "mean"),
            )
            .sort_values("horizon")
            .reset_index(drop=True)
        )
        summary_df["hit_rate"] = summary_df["hit_rate"] * 100.0
        return summary_df, detail_df.sort_values(["as_of_date", "horizon", "prediction_score"], ascending=[False, True, False]).reset_index(drop=True)

    @staticmethod
    def _compute_sector_forward_returns(stock_df: pd.DataFrame, as_of_date, future_date) -> pd.DataFrame:
        px = stock_df.copy()
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px = px.dropna(subset=["date"]).copy()
        px["close"] = pd.to_numeric(SectorPredictionEngine._series_or_default(px, "close", np.nan), errors="coerce")
        px = px.dropna(subset=["close"]).copy()
        px = px.sort_values(["code", "date"])

        start_px = (
            px[px["date"] <= pd.Timestamp(as_of_date)]
            .groupby("code", as_index=False)
            .tail(1)[["code", "sector", "close"]]
            .rename(columns={"close": "start_close"})
        )
        future_px = (
            px[px["date"] <= pd.Timestamp(future_date)]
            .groupby("code", as_index=False)
            .tail(1)[["code", "close"]]
            .rename(columns={"close": "future_close"})
        )
        merged = start_px.merge(future_px, on="code", how="inner")
        if merged.empty:
            return pd.DataFrame(columns=["sector", "future_return"])

        merged["future_return"] = np.where(
            merged["start_close"].abs() > 0,
            ((merged["future_close"] - merged["start_close"]) / merged["start_close"].abs()) * 100.0,
            np.nan,
        )
        merged = merged.dropna(subset=["future_return", "sector"])
        if merged.empty:
            return pd.DataFrame(columns=["sector", "future_return"])

        return merged.groupby("sector", as_index=False).agg(future_return=("future_return", "mean"))

    @staticmethod
    def _normalize_centered(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return (50.0 + 50.0 * np.tanh(s / 50.0)).clip(0, 100)

    @staticmethod
    def _series_or_default(df: pd.DataFrame, column: str, default_value) -> pd.Series:
        if column in df.columns:
            return df[column]
        return pd.Series([default_value] * len(df), index=df.index)
