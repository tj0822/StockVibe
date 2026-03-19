from __future__ import annotations

from typing import Any

import pandas as pd


class DecisionOrchestrator:
    """Integrate engine outputs into a single investment decision."""

    def collect_inputs(
        self,
        regime,
        money_flow,
        sector_prediction,
        investment_df,
    ):
        regime_data = regime if isinstance(regime, dict) else {}

        money_flow_data = money_flow if isinstance(money_flow, dict) else {}
        stock_flow_df = money_flow_data.get("stock_flow_df") if isinstance(money_flow_data.get("stock_flow_df"), pd.DataFrame) else pd.DataFrame()
        sector_flow_df = money_flow_data.get("sector_flow_df") if isinstance(money_flow_data.get("sector_flow_df"), pd.DataFrame) else pd.DataFrame()

        if isinstance(sector_prediction, dict):
            sector_prediction_df = sector_prediction.get("prediction_df") if isinstance(sector_prediction.get("prediction_df"), pd.DataFrame) else pd.DataFrame()
        elif isinstance(sector_prediction, pd.DataFrame):
            sector_prediction_df = sector_prediction.copy()
        else:
            sector_prediction_df = pd.DataFrame()

        if isinstance(investment_df, pd.DataFrame):
            investment_work = investment_df.copy()
        else:
            investment_work = pd.DataFrame()

        return {
            "regime": regime_data,
            "stock_flow_df": stock_flow_df,
            "sector_flow_df": sector_flow_df,
            "sector_prediction_df": sector_prediction_df,
            "investment_df": investment_work,
        }

    def compute_market_bias(self, data):
        regime = data.get("regime", {}) if isinstance(data, dict) else {}
        primary = str(regime.get("primary_regime", regime.get("market_regime", "SIDEWAYS"))).upper()
        secondary = str(regime.get("secondary_regime", "NEUTRAL")).upper()
        volatility = str(regime.get("volatility_regime", "NORMAL_VOL")).upper()
        regime_conf = float(regime.get("confidence", 55.0) or 55.0)

        sector_prediction_df = data.get("sector_prediction_df", pd.DataFrame())
        sector_score = 50.0
        if isinstance(sector_prediction_df, pd.DataFrame) and not sector_prediction_df.empty:
            base = pd.to_numeric(sector_prediction_df.get("prediction_score"), errors="coerce").dropna()
            if not base.empty:
                sector_score = float(base.head(5).mean())

        investment_df = data.get("investment_df", pd.DataFrame())
        investment_score = 50.0
        if isinstance(investment_df, pd.DataFrame) and not investment_df.empty:
            score_col = "investment_score" if "investment_score" in investment_df.columns else "final_score_adjusted"
            base = pd.to_numeric(investment_df.get(score_col), errors="coerce").dropna()
            if not base.empty:
                investment_score = float(base.head(5).mean())

        composite = (0.5 * regime_conf) + (0.25 * sector_score) + (0.25 * investment_score)

        if primary == "BEAR" or secondary == "RISK_OFF":
            return "DEFENSIVE" if volatility == "HIGH_VOL" else "CAUTIOUS"
        if primary == "BULL" and secondary == "RISK_ON" and composite >= 65:
            return "BULLISH"
        if primary == "BULL":
            return "SELECTIVE_BULLISH"
        if composite >= 62:
            return "ROTATION_OPPORTUNITY"
        return "NEUTRAL"

    def select_sectors(self, data):
        pred_df = data.get("sector_prediction_df", pd.DataFrame())
        flow_df = data.get("sector_flow_df", pd.DataFrame())

        sector_scores: dict[str, float] = {}

        if isinstance(pred_df, pd.DataFrame) and not pred_df.empty and "sector" in pred_df.columns:
            work = pred_df.copy()
            work["sector"] = work["sector"].astype(str)
            work["prediction_score"] = pd.to_numeric(work.get("prediction_score"), errors="coerce").fillna(0.0)
            work["state_bonus"] = work.get("sector_state_pred", "").astype(str).map({"Leading": 8.0, "Emerging": 5.0}).fillna(0.0)
            work["combined"] = work["prediction_score"] + work["state_bonus"]
            for _, row in work.iterrows():
                sector_scores[str(row["sector"])] = float(row.get("combined", 0.0))

        if isinstance(flow_df, pd.DataFrame) and not flow_df.empty and "sector" in flow_df.columns:
            work = flow_df.copy()
            work["sector"] = work["sector"].astype(str)
            base = "flow_score" if "flow_score" in work.columns else "sector_flow_change"
            work[base] = pd.to_numeric(work.get(base), errors="coerce").fillna(0.0)
            for _, row in work.iterrows():
                sector = str(row["sector"])
                sector_scores[sector] = sector_scores.get(sector, 0.0) + (0.35 * float(row.get(base, 0.0)))

        ranked = sorted(sector_scores.items(), key=lambda item: item[1], reverse=True)
        return [sector for sector, _ in ranked[:5]]

    def select_strategies(self, data):
        regime = data.get("regime", {}) if isinstance(data, dict) else {}
        guidance = regime.get("strategy_guidance", {}) if isinstance(regime, dict) else {}
        recommended = guidance.get("recommended", []) if isinstance(guidance, dict) else []

        selected = [str(item) for item in recommended if str(item).strip()]

        investment_df = data.get("investment_df", pd.DataFrame())
        if isinstance(investment_df, pd.DataFrame) and not investment_df.empty and "triggered_strategy" in investment_df.columns:
            strategies: dict[str, int] = {}
            top_rows = investment_df.head(10)
            for raw in top_rows["triggered_strategy"].fillna(""):
                for item in [part.strip() for part in str(raw).split(",") if part.strip()]:
                    strategies[item] = strategies.get(item, 0) + 1
            ranked = sorted(strategies.items(), key=lambda item: (-item[1], item[0]))
            for strategy, _ in ranked:
                if strategy not in selected:
                    selected.append(strategy)

        return selected[:5]

    def compute_risk_level(self, data):
        regime = data.get("regime", {}) if isinstance(data, dict) else {}
        primary = str(regime.get("primary_regime", regime.get("market_regime", "SIDEWAYS"))).upper()
        secondary = str(regime.get("secondary_regime", "NEUTRAL")).upper()
        volatility = str(regime.get("volatility_regime", "NORMAL_VOL")).upper()

        if volatility == "HIGH_VOL" and (primary == "BEAR" or secondary == "RISK_OFF"):
            return "HIGH"
        if volatility == "HIGH_VOL" or primary == "SIDEWAYS":
            return "MEDIUM"
        if primary == "BULL" and secondary == "RISK_ON":
            return "LOW"
        return "MEDIUM"

    def build_final_decision(self, data):
        market_bias = self.compute_market_bias(data)
        recommended_sectors = self.select_sectors(data)
        recommended_strategies = self.select_strategies(data)
        risk_level = self.compute_risk_level(data)

        regime = data.get("regime", {}) if isinstance(data, dict) else {}
        regime_conf = float(regime.get("confidence", 55.0) or 55.0)

        investment_df = data.get("investment_df", pd.DataFrame())
        top_candidates = []
        candidate_score = 55.0
        if isinstance(investment_df, pd.DataFrame) and not investment_df.empty:
            work = investment_df.copy()
            score_col = "investment_score" if "investment_score" in work.columns else "final_score_adjusted"
            work[score_col] = pd.to_numeric(work.get(score_col), errors="coerce").fillna(0.0)
            work = work.sort_values(score_col, ascending=False)

            if not work.empty:
                candidate_score = float(work[score_col].head(5).mean())

            for _, row in work.head(5).iterrows():
                top_candidates.append(
                    {
                        "code": str(row.get("code", "")),
                        "stock": str(row.get("stock", "-")),
                        "sector": str(row.get("sector", "-")),
                        "signal": str(row.get("signal", "")),
                        "triggered_strategy": str(row.get("triggered_strategy", "")),
                        "market_regime": str(row.get("market_regime", regime.get("primary_regime", "UNKNOWN"))),
                        "score": round(float(row.get(score_col, 0.0) or 0.0), 1),
                    }
                )

        confidence = round(max(0.0, min(100.0, (0.65 * regime_conf) + (0.35 * candidate_score))), 1)

        sector_text = ", ".join(recommended_sectors[:3]) if recommended_sectors else "섹터 선택 보류"
        strategy_text = ", ".join(recommended_strategies[:3]) if recommended_strategies else "전략 선택 보류"
        explanation = (
            f"현재 판단은 {market_bias}입니다. "
            f"유망 섹터는 {sector_text} 중심으로 보고, "
            f"전략은 {strategy_text} 우선 적용이 적합합니다. "
            f"리스크 레벨은 {risk_level}이며 신뢰도는 {confidence:.1f}입니다."
        )

        return {
            "market_bias": market_bias,
            "recommended_sectors": recommended_sectors,
            "recommended_strategies": recommended_strategies,
            "top_candidates": top_candidates,
            "risk_level": risk_level,
            "confidence": confidence,
            "explanation": explanation,
        }