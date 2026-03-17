from __future__ import annotations

import pandas as pd
import numpy as np

from sentiment_analyzer import SentimentAnalyzer


class SectorPredictionAI:
    """AI-like sector prediction layer using structured signals and news sentiment."""

    def __init__(self) -> None:
        self.sentiment_analyzer = SentimentAnalyzer(use_model=False)

    def build_sector_feature_table(
        self,
        sector_flow_df: pd.DataFrame,
        sector_rank_df: pd.DataFrame,
        news_df: pd.DataFrame,
        stock_master_df: pd.DataFrame,
    ) -> pd.DataFrame:
        sectors = set()
        for df in [sector_flow_df, sector_rank_df, stock_master_df]:
            if df is not None and not df.empty and "sector" in df.columns:
                sectors.update(df["sector"].dropna().astype(str).tolist())

        if not sectors:
            return pd.DataFrame(
                columns=[
                    "sector",
                    "sector_power",
                    "flow_momentum",
                    "flow_change",
                    "rank_change",
                    "leader_strength",
                    "news_sentiment",
                    "news_volume",
                ]
            )

        out = pd.DataFrame({"sector": sorted(sectors)})

        if sector_flow_df is not None and not sector_flow_df.empty:
            flow = sector_flow_df.copy()
            flow["sector"] = flow["sector"].astype(str)
            agg_map = {}
            if "flow_momentum" in flow.columns:
                agg_map["flow_momentum"] = ("flow_momentum", "mean")
            elif "flow_score" in flow.columns:
                agg_map["flow_momentum"] = ("flow_score", "mean")

            if "sector_flow_change" in flow.columns:
                agg_map["flow_change"] = ("sector_flow_change", "mean")
            elif "flow_change" in flow.columns:
                agg_map["flow_change"] = ("flow_change", "mean")

            flow_sector = flow.groupby("sector", as_index=False).agg(**agg_map) if agg_map else pd.DataFrame(columns=["sector"])
            out = out.merge(flow_sector, on="sector", how="left")

        rank_change_df = self._compute_rank_change(sector_rank_df)
        out = out.merge(rank_change_df, on="sector", how="left")

        leader_df = self.predict_leader_stocks(stock_master_df)
        if leader_df is not None and not leader_df.empty:
            leader_strength = leader_df.groupby("sector", as_index=False).agg(leader_strength=("leader_strength", "mean"))
            out = out.merge(leader_strength, on="sector", how="left")

        if stock_master_df is not None and not stock_master_df.empty and "sector" in stock_master_df.columns:
            sm = stock_master_df.copy()
            sm["sector"] = sm["sector"].astype(str)
            sm["sector_power"] = pd.to_numeric(self._series_or_default(sm, "sector_power", 50.0), errors="coerce").fillna(50.0)
            power_df = sm.groupby("sector", as_index=False).agg(sector_power=("sector_power", "mean"))
            out = out.merge(power_df, on="sector", how="left")

        news_features = self._build_news_features(news_df, stock_master_df)
        out = out.merge(news_features, on="sector", how="left")

        out["sector_power"] = pd.to_numeric(self._series_or_default(out, "sector_power", 50.0), errors="coerce").fillna(50.0)
        out["flow_momentum"] = pd.to_numeric(self._series_or_default(out, "flow_momentum", 0.0), errors="coerce").fillna(0.0)
        out["flow_change"] = pd.to_numeric(self._series_or_default(out, "flow_change", 0.0), errors="coerce").fillna(0.0)
        out["rank_change"] = pd.to_numeric(self._series_or_default(out, "rank_change", 0.0), errors="coerce").fillna(0.0)
        out["leader_strength"] = pd.to_numeric(self._series_or_default(out, "leader_strength", 50.0), errors="coerce").fillna(50.0)
        out["news_sentiment"] = pd.to_numeric(self._series_or_default(out, "news_sentiment", 50.0), errors="coerce").fillna(50.0)
        out["news_volume"] = pd.to_numeric(self._series_or_default(out, "news_volume", 0.0), errors="coerce").fillna(0.0)
        return out[["sector", "sector_power", "flow_momentum", "flow_change", "rank_change", "leader_strength", "news_sentiment", "news_volume"]]

    def compute_prediction_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        work = df.copy()
        fm = self._normalize_centered(pd.to_numeric(self._series_or_default(work, "flow_momentum", 0.0), errors="coerce").fillna(0.0))
        rc = self._normalize_centered(pd.to_numeric(self._series_or_default(work, "rank_change", 0.0), errors="coerce").fillna(0.0) * 10.0)
        ls = pd.to_numeric(self._series_or_default(work, "leader_strength", 50.0), errors="coerce").fillna(50.0).clip(0, 100)
        sp = pd.to_numeric(self._series_or_default(work, "sector_power", 50.0), errors="coerce").fillna(50.0).clip(0, 100)
        ns = pd.to_numeric(self._series_or_default(work, "news_sentiment", 50.0), errors="coerce").fillna(50.0).clip(0, 100)
        nv = self._normalize_centered(pd.to_numeric(self._series_or_default(work, "news_volume", 0.0), errors="coerce").fillna(0.0) * 5.0)

        work["prediction_score"] = (
            0.30 * fm
            + 0.20 * rc
            + 0.15 * ls
            + 0.15 * sp
            + 0.10 * ns
            + 0.10 * nv
        )
        return work

    def classify_sector_state(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        work = df.copy()
        score = pd.to_numeric(self._series_or_default(work, "prediction_score", 0.0), errors="coerce").fillna(0.0)
        state = np.where(
            score > 75,
            "Dominant",
            np.where(score > 60, "Emerging", np.where(score > 45, "Strengthening", np.where(score < 35, "Weakening", "Neutral"))),
        )
        work["sector_state_ai"] = state
        return work

    def predict_leader_stocks(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        if stock_df is None or stock_df.empty:
            return pd.DataFrame(columns=["sector", "stock", "momentum_score", "money_flow_score", "financial_score", "leader_strength"])

        work = stock_df.copy()
        work["sector"] = self._series_or_default(work, "sector", "Unknown").astype(str)
        stock_col = "name" if "name" in work.columns else ("stock" if "stock" in work.columns else "code")
        work["stock"] = self._series_or_default(work, stock_col, "").astype(str)
        work["momentum_score"] = pd.to_numeric(self._series_or_default(work, "momentum_score", 0.0), errors="coerce").fillna(0.0)
        work["money_flow_score"] = pd.to_numeric(self._series_or_default(work, "money_flow_score", 50.0), errors="coerce").fillna(50.0)
        if "financial_score" in work.columns:
            work["financial_score"] = pd.to_numeric(work["financial_score"], errors="coerce").fillna(50.0)
        else:
            work["financial_score"] = pd.to_numeric(self._series_or_default(work, "final_score_adjusted", 50.0), errors="coerce").fillna(50.0)

        leaders = work[
            (work["momentum_score"] > 70)
            & (work["money_flow_score"] > 60)
            & (work["financial_score"] > 60)
        ].copy()
        if leaders.empty:
            return pd.DataFrame(columns=["sector", "stock", "momentum_score", "money_flow_score", "financial_score", "leader_strength"])

        leaders["leader_strength"] = (
            0.4 * leaders["momentum_score"]
            + 0.3 * leaders["money_flow_score"]
            + 0.3 * leaders["financial_score"]
        )
        leaders = leaders.sort_values(["sector", "leader_strength"], ascending=[True, False]).groupby("sector", as_index=False).head(3)
        return leaders[["sector", "stock", "momentum_score", "money_flow_score", "financial_score", "leader_strength"]].reset_index(drop=True)

    def predict_future_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["sector", "prediction_score", "sector_state_ai"])

        work = df.copy()
        if "prediction_score" not in work.columns:
            work = self.compute_prediction_score(work)
        if "sector_state_ai" not in work.columns:
            work = self.classify_sector_state(work)

        predicted = work[work["sector_state_ai"].isin(["Dominant", "Emerging", "Strengthening"])].copy()
        if predicted.empty:
            predicted = work.copy()
        return predicted.sort_values("prediction_score", ascending=False).head(5).reset_index(drop=True)

    def _build_news_features(self, news_df: pd.DataFrame, stock_master_df: pd.DataFrame) -> pd.DataFrame:
        if news_df is None or news_df.empty:
            return pd.DataFrame(columns=["sector", "news_sentiment", "news_volume"])

        work = news_df.copy()
        if "sector" not in work.columns:
            work = self._map_news_to_sector(work, stock_master_df)
        if work.empty or "sector" not in work.columns:
            return pd.DataFrame(columns=["sector", "news_sentiment", "news_volume"])

        if "sentiment_score" not in work.columns:
            title_col = None
            for candidate in ["title", "제목", "headline"]:
                if candidate in work.columns:
                    title_col = candidate
                    break
            if title_col is not None:
                work["sentiment_score"] = work[title_col].astype(str).apply(lambda text: self.sentiment_analyzer.analyze(text).get("score", 0.0))
            else:
                work["sentiment_score"] = 0.0

        work["sentiment_score"] = pd.to_numeric(work["sentiment_score"], errors="coerce").fillna(0.0)
        # convert -1~1 sentiment to 0~100 scale
        work["news_sentiment"] = ((work["sentiment_score"] + 1.0) * 50.0).clip(0, 100)

        out = (
            work.groupby("sector", as_index=False)
            .agg(
                news_sentiment=("news_sentiment", "mean"),
                news_volume=("sector", "size"),
            )
        )
        return out

    def _map_news_to_sector(self, news_df: pd.DataFrame, stock_master_df: pd.DataFrame) -> pd.DataFrame:
        if stock_master_df is None or stock_master_df.empty:
            return news_df

        ref = stock_master_df.copy()
        merge_cols = [c for c in ["code", "name", "sector"] if c in ref.columns]
        ref = ref[merge_cols].copy()
        if "code" in ref.columns:
            ref["code"] = ref["code"].astype(str).str.zfill(6)

        work = news_df.copy()
        if "code" in work.columns and "code" in ref.columns:
            work["code"] = work["code"].astype(str).str.zfill(6)
            work = work.merge(ref[[c for c in ["code", "sector"] if c in ref.columns]], on="code", how="left")
        elif "name" in work.columns and "name" in ref.columns:
            work = work.merge(ref[[c for c in ["name", "sector"] if c in ref.columns]].drop_duplicates(subset=["name"], keep="last"), on="name", how="left")
        return work

    def _compute_rank_change(self, sector_rank_df: pd.DataFrame) -> pd.DataFrame:
        if sector_rank_df is None or sector_rank_df.empty or not {"sector", "rank"}.issubset(set(sector_rank_df.columns)):
            return pd.DataFrame(columns=["sector", "rank_change"])

        rk = sector_rank_df.copy()
        rk["sector"] = rk["sector"].astype(str)
        rk["rank"] = pd.to_numeric(rk["rank"], errors="coerce")
        rk = rk.dropna(subset=["rank"])
        if "date" in rk.columns:
            rk["date"] = pd.to_datetime(rk["date"], errors="coerce")
            rk = rk.dropna(subset=["date"]).sort_values(["sector", "date"])
            out = (
                rk.groupby("sector", as_index=False)
                .apply(lambda g: pd.Series({"rank_change": (float(g.iloc[-2]["rank"]) - float(g.iloc[-1]["rank"])) if len(g) >= 2 else 0.0}))
                .reset_index(drop=True)
            )
            return out
        return rk.groupby("sector", as_index=False).agg(rank_change=("rank", "mean")).assign(rank_change=0.0)

    @staticmethod
    def _normalize_centered(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return (50.0 + 50.0 * np.tanh(s / 50.0)).clip(0, 100)

    @staticmethod
    def _series_or_default(df: pd.DataFrame, column: str, default_value) -> pd.Series:
        if column in df.columns:
            return df[column]
        return pd.Series([default_value] * len(df), index=df.index)