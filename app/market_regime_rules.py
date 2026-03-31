from __future__ import annotations



def classify_primary_regime(features: dict) -> str:
    r20 = float(features.get("kospi_return_20d", 0.0))
    above_ma60 = float(features.get("above_ma60_ratio", 0.5))

    if r20 > 0.05 and above_ma60 > 0.55:
        return "BULL"
    if r20 < -0.05 and above_ma60 < 0.45:
        return "BEAR"
    return "SIDEWAYS"


def classify_secondary_regime(features: dict) -> str:
    leading_cnt = float(features.get("leading_sector_count", 0.0))
    concentration = float(features.get("top_sector_concentration", 0.0))
    buy_sell_ratio = float(features.get("buy_sell_ratio", 1.0))

    if leading_cnt >= 4 and concentration < 0.45 and buy_sell_ratio > 1.2:
        return "RISK_ON"
    if buy_sell_ratio < 0.8 and leading_cnt <= 2:
        return "RISK_OFF"
    return "NEUTRAL"


def classify_volatility_regime(features: dict) -> str:
    vol_20d = float(features.get("kospi_vol_20d", 0.0))
    vol_threshold = float(features.get("vol_threshold", 0.02))
    return "HIGH_VOL" if vol_20d > vol_threshold else "NORMAL_VOL"


def compute_regime_confidence(features: dict, regimes: dict) -> float:
    score = 55.0

    r20 = float(features.get("kospi_return_20d", 0.0))
    r60 = float(features.get("kospi_return_60d", 0.0))
    above_ma60 = float(features.get("above_ma60_ratio", 0.5))
    buy_sell = float(features.get("buy_sell_ratio", 1.0))
    temp = float(features.get("market_temperature", 50.0))
    rot = float(features.get("rotation_strength", 0.0))
    conc = float(features.get("top_sector_concentration", 0.0))

    primary = regimes.get("primary_regime", "SIDEWAYS")
    secondary = regimes.get("secondary_regime", "NEUTRAL")
    vol_regime = regimes.get("volatility_regime", "NORMAL_VOL")

    if primary == "BULL":
        if r20 > 0.07:
            score += 8
        if r60 > 0.10:
            score += 6
        if above_ma60 > 0.60:
            score += 6
    elif primary == "BEAR":
        if r20 < -0.07:
            score += 8
        if r60 < -0.10:
            score += 6
        if above_ma60 < 0.40:
            score += 6
    else:
        if abs(r20) < 0.03:
            score += 5

    if secondary == "RISK_ON":
        if buy_sell > 1.4:
            score += 5
        if conc < 0.40:
            score += 4
    elif secondary == "RISK_OFF":
        if buy_sell < 0.7:
            score += 5
        if temp < 45:
            score += 4

    if vol_regime == "HIGH_VOL":
        if rot > 4.5:
            score += 4
        else:
            score -= 3

    return round(max(0.0, min(100.0, score)), 1)


def generate_regime_interpretation(regimes: dict, features: dict) -> dict:
    p = regimes.get("primary_regime", "SIDEWAYS")
    s = regimes.get("secondary_regime", "NEUTRAL")
    v = regimes.get("volatility_regime", "NORMAL_VOL")

    if p == "BULL" and s == "RISK_ON":
        summary = "상승 추세 속 성장/모멘텀 선호가 강한 국면입니다."
    elif p == "BEAR" or s == "RISK_OFF":
        summary = "방어 성향이 우세한 구간으로 리스크 관리가 중요한 국면입니다."
    else:
        summary = "뚜렷한 방향성보다 섹터별 순환과 선별 대응이 중요한 국면입니다."

    if v == "HIGH_VOL":
        risk_note = "변동성이 높아 추격 매수보다 분할 진입과 손절 규율이 중요합니다."
    else:
        risk_note = "변동성은 안정적이지만 섹터 리더 변화 속도를 함께 점검하세요."

    return {"summary": summary, "risk_note": risk_note}


def map_regime_to_strategy_guidance(regimes: dict) -> dict:
    p = regimes.get("primary_regime", "SIDEWAYS")
    s = regimes.get("secondary_regime", "NEUTRAL")
    v = regimes.get("volatility_regime", "NORMAL_VOL")

    recommended = []
    avoid = []

    if p == "BULL" and s == "RISK_ON":
        recommended = ["turnover_spike", "momentum_breakout", "sector_leader_follow"]
        avoid = ["mean_reversion"]
    elif p == "BULL" and v == "HIGH_VOL":
        recommended = ["turnover_spike", "quality_breakout_conservative"]
        avoid = ["aggressive_averaging"]
    elif p == "SIDEWAYS":
        recommended = ["mean_reversion", "quality_filter"]
        avoid = ["pure_breakout"]
    elif p == "BEAR":
        recommended = ["defensive_low_vol", "risk_control"]
        avoid = ["momentum_chase"]
    elif s == "RISK_OFF":
        recommended = ["largecap_quality", "dividend_focus", "weak_exposure_reduce"]
        avoid = ["smallcap_momentum"]
    else:
        recommended = ["turnover_spike", "quality_filter"]
        avoid = ["aggressive_averaging"]

    return {"recommended": recommended, "avoid": avoid}
