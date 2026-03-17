from __future__ import annotations

from typing import Any

import pandas as pd

from app.strategies.registry_store import load_registry


class InvestmentEngine:
    """Final Investment Engine: 시장/섹터/종목/전략 신호를 통합해 투자 후보를 랭킹한다."""

    def build_investment_df(
        self,
        stock_master_df: pd.DataFrame,
        sector_rank_df: pd.DataFrame,
        regime: Any,
    ) -> pd.DataFrame:
        """투자 엔진 입력 프레임 생성.

        Required output columns:
        - stock, sector, sector_power, financial_score, momentum_score, signal, strategy_fit
        """
        if stock_master_df is None or stock_master_df.empty:
            return pd.DataFrame(
                columns=[
                    "stock",
                    "sector",
                    "sector_power",
                    "financial_score",
                    "momentum_score",
                    "signal",
                    "strategy_fit",
                    "money_flow_score",
                    "sector_prediction_score",
                    "final_score_adjusted",
                    "signal_strength",
                    "triggered_strategy",
                    "market_regime",
                ]
            )

        work = stock_master_df.copy()

        stock_col = "name" if "name" in work.columns else ("stock" if "stock" in work.columns else None)
        if stock_col is None:
            stock_col = "code" if "code" in work.columns else None
        if stock_col is None:
            work["stock"] = ""
            stock_col = "stock"

        if "sector" not in work.columns:
            work["sector"] = "Unknown"

        if "sector_power" not in work.columns:
            sector_power_map: dict[str, float] = {}
            if sector_rank_df is not None and not sector_rank_df.empty and {"sector", "rank"}.issubset(set(sector_rank_df.columns)):
                rank_work = sector_rank_df.copy()
                if "date" in rank_work.columns:
                    rank_work["date"] = pd.to_datetime(rank_work["date"], errors="coerce")
                    latest_date = rank_work["date"].dropna().max()
                    if pd.notna(latest_date):
                        rank_work = rank_work[rank_work["date"] == latest_date]
                rank_work["rank"] = pd.to_numeric(rank_work["rank"], errors="coerce")
                rank_work = rank_work.dropna(subset=["sector", "rank"])
                if not rank_work.empty:
                    max_rank = float(rank_work["rank"].max())
                    if max_rank <= 1:
                        rank_work["sector_power"] = 100.0
                    else:
                        rank_work["sector_power"] = 100.0 * (1.0 - (rank_work["rank"] - 1.0) / (max_rank - 1.0))
                    sector_power_map = dict(zip(rank_work["sector"], rank_work["sector_power"]))
            work["sector_power"] = work["sector"].map(sector_power_map).fillna(50.0)

        momentum_raw = self._series_or_default(work, "momentum_score", 0.0)
        final_score_raw = self._series_or_default(work, "final_score_adjusted", None)
        if final_score_raw.isna().all():
            final_score_raw = self._series_or_default(work, "final_score", 50.0)
        financial_raw = self._series_or_default(work, "financial_score", None)
        financial_raw = financial_raw.fillna(final_score_raw)

        work["momentum_score"] = pd.to_numeric(momentum_raw, errors="coerce").fillna(0.0)
        work["final_score_adjusted"] = pd.to_numeric(final_score_raw, errors="coerce").fillna(50.0)
        work["financial_score"] = pd.to_numeric(financial_raw, errors="coerce").fillna(50.0)

        signal_series = self._series_or_default(work, "signal", "").fillna("").astype(str).str.upper()
        work["signal"] = signal_series

        if "signal_strength" in work.columns:
            work["signal_strength"] = pd.to_numeric(work["signal_strength"], errors="coerce").fillna(0.0)
        else:
            work["signal_strength"] = signal_series.map({"BUY": 100.0, "SELL": -100.0}).fillna(0.0)

        if "money_flow_score" in work.columns:
            work["money_flow_score"] = pd.to_numeric(work["money_flow_score"], errors="coerce").fillna(50.0)
        elif "flow_score" in work.columns:
            work["money_flow_score"] = pd.to_numeric(work["flow_score"], errors="coerce").fillna(50.0)
        else:
            work["money_flow_score"] = 50.0

        if "sector_prediction_score" in work.columns:
            work["sector_prediction_score"] = pd.to_numeric(work["sector_prediction_score"], errors="coerce").fillna(50.0)
        elif "prediction_score" in work.columns:
            work["sector_prediction_score"] = pd.to_numeric(work["prediction_score"], errors="coerce").fillna(50.0)
        else:
            work["sector_prediction_score"] = 50.0

        strategy_col = "triggered_strategies" if "triggered_strategies" in work.columns else "strategy"
        work["triggered_strategy"] = self._series_or_default(work, strategy_col, "").fillna("").astype(str)

        registry = load_registry()
        work["strategy_fit"] = work["triggered_strategy"].apply(lambda x: self._compute_strategy_fit(x, registry, default_signal="BUY" if x else ""))

        market_regime = "UNKNOWN"
        if isinstance(regime, dict):
            market_regime = str(regime.get("market_regime") or regime.get("primary_regime") or regime.get("regime") or "UNKNOWN")
        elif regime is not None:
            market_regime = str(regime)

        out = pd.DataFrame(
            {
                "stock": work[stock_col],
                "sector": work["sector"],
                "sector_power": pd.to_numeric(work["sector_power"], errors="coerce").fillna(0.0),
                "financial_score": work["financial_score"],
                "momentum_score": work["momentum_score"],
                "signal": work["signal"],
                "strategy_fit": work["strategy_fit"],
                "money_flow_score": work["money_flow_score"],
                "sector_prediction_score": work["sector_prediction_score"],
                "final_score_adjusted": work["final_score_adjusted"],
                "signal_strength": work["signal_strength"],
                "triggered_strategy": work["triggered_strategy"],
                "market_regime": market_regime,
            }
        )
        return out

    def compute_investment_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """투자 점수 계산.

        score =
            existing_score +
            0.1 * sector_prediction_score

        where existing_score =
            0.2  * sector_power +
            0.2  * financial_score +
            0.15 * momentum_score +
            0.15 * signal_strength +
            0.1  * strategy_fit +
            0.2  * money_flow_score
        """
        if df is None or df.empty:
            return df

        work = df.copy()
        work["sector_power"] = pd.to_numeric(self._series_or_default(work, "sector_power", 0.0), errors="coerce").fillna(0.0)
        work["financial_score"] = pd.to_numeric(self._series_or_default(work, "financial_score", 0.0), errors="coerce").fillna(0.0)
        work["momentum_score"] = pd.to_numeric(self._series_or_default(work, "momentum_score", 0.0), errors="coerce").fillna(0.0)
        work["signal_strength"] = pd.to_numeric(self._series_or_default(work, "signal_strength", 0.0), errors="coerce").fillna(0.0)
        work["strategy_fit"] = pd.to_numeric(self._series_or_default(work, "strategy_fit", 0.0), errors="coerce").fillna(0.0)
        work["money_flow_score"] = pd.to_numeric(self._series_or_default(work, "money_flow_score", 50.0), errors="coerce").fillna(50.0)
        work["sector_prediction_score"] = pd.to_numeric(self._series_or_default(work, "sector_prediction_score", 50.0), errors="coerce").fillna(50.0)

        existing_score = (
            0.2 * work["sector_power"]
            + 0.2 * work["financial_score"]
            + 0.15 * work["momentum_score"]
            + 0.15 * work["signal_strength"]
            + 0.1 * work["strategy_fit"]
            + 0.2 * work["money_flow_score"]
        )
        work["investment_score"] = existing_score + (0.1 * work["sector_prediction_score"])
        return work

    def filter_buy_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """매수 후보 필터링 규칙 적용."""
        if df is None or df.empty:
            return pd.DataFrame()

        work = df.copy()
        signal_upper = self._series_or_default(work, "signal", "").astype(str).str.upper()
        filtered = work[
            (signal_upper == "BUY")
            & (pd.to_numeric(self._series_or_default(work, "sector_power", 0.0), errors="coerce").fillna(0.0) > 60)
            & (pd.to_numeric(self._series_or_default(work, "final_score_adjusted", 0.0), errors="coerce").fillna(0.0) > 60)
            & (pd.to_numeric(self._series_or_default(work, "momentum_score", 0.0), errors="coerce").fillna(0.0) > 50)
        ].copy()
        return filtered

    def rank_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """investment_score 기준 내림차순 상위 20개 반환."""
        if df is None or df.empty:
            return pd.DataFrame()

        work = df.copy()
        if "investment_score" not in work.columns:
            work = self.compute_investment_score(work)

        ranked = work.sort_values("investment_score", ascending=False).head(20).reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        return ranked

    @staticmethod
    def _compute_strategy_fit(triggered_strategy: str, registry: dict[str, Any], default_signal: str = "") -> float:
        """전략 활성/검증 상태 기반 전략 적합도 점수(0~100) 계산."""
        text = str(triggered_strategy or "").strip()
        if not text:
            if str(default_signal).upper() == "BUY":
                return 60.0
            return 50.0

        ids = [sid.strip() for sid in text.split(",") if sid.strip()]
        if not ids:
            return 50.0

        scores: list[float] = []
        for sid in ids:
            state = registry.get(sid, {}) if isinstance(registry, dict) else {}
            status = str(state.get("status", "draft")).lower()
            enabled = bool(state.get("enabled", True))
            in_prod = bool(state.get("in_production", False))

            if in_prod or status == "approved":
                s = 100.0
            elif status == "validated":
                s = 85.0
            elif enabled:
                s = 65.0
            else:
                s = 40.0
            scores.append(s)

        return float(sum(scores) / len(scores)) if scores else 50.0

    @staticmethod
    def _series_or_default(df: pd.DataFrame, column: str, default_value: Any) -> pd.Series:
        """컬럼이 없을 때도 인덱스를 맞춘 Series를 반환한다."""
        if column in df.columns:
            return df[column]
        return pd.Series([default_value] * len(df), index=df.index)