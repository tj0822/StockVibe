"""
재무 지표 분석 유틸리티 함수
기업의 재무 건전성과 수익성을 평가하는 함수들
"""

# ============================================================================
# Default Weights for Ablation Testing
# ============================================================================

DEFAULT_WEIGHTS = {
    'quality': 0.25,           # 재무 건전성: 25%
    'valuation': 0.25,         # 상대 밸류에이션: 25%
    'growth': 0.20,            # 성장성: 20%
    'momentum': 0.20,          # 모멘텀: 20%
    'industry_tailwind': 0.10   # 산업 태풍: 10%
}


def get_default_weights():
    """
    기본 가중치 반환 (ablation testing용)
    
    Returns:
        dict: {'quality': 0.25, 'valuation': 0.25, ...}
    """
    return DEFAULT_WEIGHTS.copy()



    """
    기업의 재무 건전성 및 수익성 점수를 계산 (0-100)
    
    ROE, 부채비율, 영업이익률, 자유현금흐름을 종합 평가
    각 지표를 정규화하고 동등한 가중치(25%)로 결합
    
    Args:
        financial_dict (dict): 재무 지표 딕셔너리
            - roe (float): ROE (%). 예: 15.5
            - debt_ratio (float): 부채비율 (%). 예: 45.0
            - operating_margin (float): 영업이익률 (%). 예: 12.3
            - free_cash_flow (float or bool): FCF (원). 양수면 1점, 음수면 0점
    
    Returns:
        int: 0-100 범위의 점수
    
    Example:
        >>> financial_dict = {
        ...     'roe': 15.0,
        ...     'debt_ratio': 50.0,
        ...     'operating_margin': 12.0,
        ...     'free_cash_flow': 1000000000
        ... }
        >>> score = calculate_quality_score(financial_dict)
        >>> print(score)
        68
    
    Normalization Logic:
        - ROE: 0-30% → 0-100 (30% 이상은 100)
        - Debt Ratio: 0-300% → 100-0 (낮을수록 좋음)
        - Operating Margin: -20% to 30% → 0-100
        - FCF: 양수(1) → 25점, 음수(0) → 0점
    """
    
    # 기본값 처리 (None이나 잘못된 값 방어)
    roe = float(financial_dict.get('roe', 0)) if financial_dict.get('roe') else 0
    debt_ratio = float(financial_dict.get('debt_ratio', 0)) if financial_dict.get('debt_ratio') else 0
    operating_margin = float(financial_dict.get('operating_margin', 0)) if financial_dict.get('operating_margin') else 0
    free_cash_flow = financial_dict.get('free_cash_flow', 0)
    
    # 1. ROE 점수 (0-30% → 0-100)
    # ROE가 높을수록 좋음 (최대 30%)
    roe_score = min(100, (roe / 30) * 100) if roe > 0 else 0
    
    # 2. 부채비율 점수 (0-300% → 100-0)
    # 부채비율이 낮을수록 좋음
    # 0%: 100점, 100%: 67점, 200%: 33점, 300% 이상: 0점
    if debt_ratio <= 0:
        debt_score = 100
    elif debt_ratio >= 300:
        debt_score = 0
    else:
        # 선형 감소: debt_ratio가 증가할수록 점수 감소
        debt_score = max(0, 100 - (debt_ratio / 3))
    
    # 3. 영업이익률 점수 (-20% to 30% → 0-100)
    # 영업이익률이 높을수록 좋음
    # -20%: 0점, 0%: 40점, 30%: 100점
    if operating_margin <= -20:
        margin_score = 0
    elif operating_margin >= 30:
        margin_score = 100
    else:
        # -20%에서 30%로 정규화 (범위: 50%)
        margin_score = ((operating_margin + 20) / 50) * 100
    
    # 4. 자유현금흐름 점수
    # 양수면 25점, 음수면 0점 (부재하거나 양수 여부만 확인)
    if isinstance(free_cash_flow, bool):
        fcf_score = 25 if free_cash_flow else 0
    else:
        fcf_score = 25 if free_cash_flow > 0 else 0
    
    # 5. 종합 점수 계산 (4개 지표 동등 가중치)
    # ROE: 25%, Debt: 25%, Margin: 25%, FCF: 25%
    quality_score = (roe_score + debt_score + margin_score + fcf_score) / 4
    
    return int(round(quality_score))


def calculate_valuation_score(financial_dict):
    """
    PER, PBR을 기반으로 저평가 여부를 판단 (0-100)
    
    낮은 PER과 PBR은 저평가 신호
    
    Args:
        financial_dict (dict): 재무 지표 딕셔너리
            - per (float): PER (배). 예: 15.5
            - pbr (float): PBR (배). 예: 1.2
    
    Returns:
        int: 0-100 범위의 저평가 점수
            - 80-100: 매우저평가
            - 60-79: 저평가
            - 40-59: 적정가
            - 20-39: 고평가
            - 0-19: 매우고평가
    
    Example:
        >>> financial_dict = {'per': 10.0, 'pbr': 0.8}
        >>> score = calculate_valuation_score(financial_dict)
        >>> print(score)
        85
    
    Valuation Logic:
        - PER: 0-20배 → 100-50 (낮을수록 저평가)
        - PBR: 0-3배 → 100-40 (낮을수록 저평가)
    """
    
    per = float(financial_dict.get('per', 15)) if financial_dict.get('per') else 15
    pbr = float(financial_dict.get('pbr', 1.5)) if financial_dict.get('pbr') else 1.5
    
    # PER 점수 (0-20배 → 100-50)
    # PER이 낮을수록 저평가
    if per <= 0:
        per_score = 100
    elif per >= 30:
        per_score = 30
    else:
        # 0-20배: 100-50점, 20-30배: 50-30점
        if per <= 20:
            per_score = 100 - (per / 20) * 50
        else:
            per_score = 50 - ((per - 20) / 10) * 20
    
    # PBR 점수 (0-3배 → 100-40)
    # PBR이 낮을수록 저평가
    if pbr <= 0:
        pbr_score = 100
    elif pbr >= 5:
        pbr_score = 20
    else:
        # 0-3배: 100-40점, 3-5배: 40-20점
        if pbr <= 3:
            pbr_score = 100 - (pbr / 3) * 60
        else:
            pbr_score = 40 - ((pbr - 3) / 2) * 20
    
    # 평균값 (PER 50%, PBR 50%)
    valuation_score = (per_score + pbr_score) / 2
    
    return int(round(valuation_score))


def calculate_relative_valuation_score(current_per, industry_per, current_pbr, industry_pbr=None):
    """
    산업 평균과 비교하여 상대적 밸류에이션 점수 계산 (0-100)
    
    종목의 PER/PBR을 산업 평균과 비교하여 저평가/고평가 여부를 판단
    산업 평균 대비 낮으면 저평가(고점수), 높으면 고평가(저점수)
    
    Args:
        current_per (float): 종목의 현재 PER (배). 예: 12.5
        industry_per (float): 산업 평균 PER (배). 예: 15.0
        current_pbr (float): 종목의 현재 PBR (배). 예: 1.0
        industry_pbr (float, optional): 산업 평균 PBR (배). 기본값은 current_pbr 기반 추정
    
    Returns:
        tuple: (점수, PER_비율, PBR_비율)
            - 점수: 0-100 범위의 상대 밸류에이션 점수
            - PER_비율: current_per / industry_per (예: 0.8 = 20% 저평가)
            - PBR_비율: current_pbr / industry_pbr (예: 0.9 = 10% 저평가)
    
    Returns:
        tuple: (score_int, per_ratio_float, pbr_ratio_float)
            - 80-100: 매우저평가 (산업평균 대비 30% 이상 낮음)
            - 60-79: 저평가 (산업평균 대비 10-30% 낮음)
            - 40-59: 적정가 (산업평균 대비 ±10% 범위)
            - 20-39: 고평가 (산업평균 대비 10-30% 높음)
            - 0-19: 매우고평가 (산업평균 대비 30% 이상 높음)
    
    Example:
        >>> # 삼성전자: PER 12, PBR 1.5 vs 반도체 평균: PER 15, PBR 2.0
        >>> score, per_ratio, pbr_ratio = calculate_relative_valuation_score(
        ...     current_per=12.0,
        ...     industry_per=15.0,
        ...     current_pbr=1.5,
        ...     industry_pbr=2.0
        ... )
        >>> print(f"점수: {score}, PER비율: {per_ratio:.2f}x, PBR비율: {pbr_ratio:.2f}x")
        점수: 75, PER비율: 0.80x, PBR비율: 0.75x
    
    Relative Valuation Logic:
        PER 비교:
        - 현재PER/산업평균PER 비율 계산
        - 0.7x 이하: 매우저평가 (30% 이상 저평가)
        - 0.9x 이하: 저평가 (10-30% 저평가)
        - 0.9x ~ 1.1x: 적정가 (±10%)
        - 1.1x ~ 1.3x: 고평가 (10-30% 고평가)
        - 1.3x 이상: 매우고평가 (30% 이상 고평가)
        
        PBR 비교: 동일 로직 적용 (가중치: PER 60%, PBR 40%)
    """
    
    try:
        # 입력값 검증 및 정규화
        current_per = float(current_per) if current_per and current_per > 0 else 1
        industry_per = float(industry_per) if industry_per and industry_per > 0 else current_per
        current_pbr = float(current_pbr) if current_pbr and current_pbr > 0 else 1.0
        
        # industry_pbr 기본값: current_pbr 기반 추정
        if industry_pbr is None or industry_pbr <= 0:
            industry_pbr = current_pbr * 1.2  # 기본: 종목 PBR의 1.2배로 추정
        else:
            industry_pbr = float(industry_pbr)
        
        # PER 상대 점수 계산
        per_ratio = current_per / industry_per
        
        if per_ratio <= 0.7:
            # 매우저평가: 30% 이상 저평가
            per_score = 95
        elif per_ratio <= 0.85:
            # 저평가: 15-30% 저평가
            per_score = 80 - ((0.85 - per_ratio) / 0.15) * 15
        elif per_ratio <= 0.9:
            # 약간 저평가: 10-15% 저평가
            per_score = 80 - ((0.9 - per_ratio) / 0.05) * 20
        elif per_ratio <= 1.1:
            # 적정가: ±10%
            per_score = 50 + ((1.1 - per_ratio) / 0.2) * 10
        elif per_ratio <= 1.3:
            # 고평가: 10-30% 고평가
            per_score = 30 + ((1.3 - per_ratio) / 0.2) * 20
        else:
            # 매우고평가: 30% 이상 고평가
            per_score = 10
        
        # PBR 상대 점수 계산 (동일 로직)
        pbr_ratio = current_pbr / industry_pbr
        
        if pbr_ratio <= 0.7:
            pbr_score = 95
        elif pbr_ratio <= 0.85:
            pbr_score = 80 - ((0.85 - pbr_ratio) / 0.15) * 15
        elif pbr_ratio <= 0.9:
            pbr_score = 80 - ((0.9 - pbr_ratio) / 0.05) * 20
        elif pbr_ratio <= 1.1:
            pbr_score = 50 + ((1.1 - pbr_ratio) / 0.2) * 10
        elif pbr_ratio <= 1.3:
            pbr_score = 30 + ((1.3 - pbr_ratio) / 0.2) * 20
        else:
            pbr_score = 10
        
        # 종합 상대 밸류에이션 점수 (PER 60%, PBR 40%)
        relative_valuation_score = (per_score * 0.6) + (pbr_score * 0.4)
        
        return int(round(relative_valuation_score)), per_ratio, pbr_ratio
    
    except Exception as e:
        print(f"Error in calculate_relative_valuation_score: {e}")
        return 50, 1.0, 1.0  # 오류 시 중립값 반환


def get_valuation_grade(score):
    """
    밸류에이션 점수에 따른 평가 등급 반환
    
    Args:
        score (int): 0-100 점수
    
    Returns:
        tuple: (등급 문자열, 설명)
            - ("매우저평가", "강력 매수 신호")
            - ("저평가", "매수 신호")
            - ("적정가", "중립")
            - ("고평가", "매도 신호")
            - ("매우고평가", "강력 매도 신호")
    
    Example:
        >>> grade, description = get_valuation_grade(80)
        >>> print(f"{grade}: {description}")
        저평가: 매수 신호
    """
    if score >= 80:
        return ("매우저평가", "강력 매수 신호")
    elif score >= 60:
        return ("저평가", "매수 신호")
    elif score >= 40:
        return ("적정가", "중립")
    elif score >= 20:
        return ("고평가", "매도 신호")
    else:
        return ("매우고평가", "강력 매도 신호")


def calculate_growth_score(financial_dict):
    """
    EPS 성장률 및 배당수익률 기반 성장성 평가 (0-100)
    
    Args:
        financial_dict (dict): 재무 지표 딕셔너리
            - eps_growth (float): EPS 성장률 (%). 예: 12.5
            - dividend_yield (float): 배당수익률 (%). 예: 3.5
    
    Returns:
        int: 0-100 범위의 성장률 점수
    
    Example:
        >>> financial_dict = {'eps_growth': 15.0, 'dividend_yield': 3.0}
        >>> score = calculate_growth_score(financial_dict)
        >>> print(score)
        72
    
    Growth Logic:
        - EPS Growth: 0-30% → 0-100 (높을수록 좋음)
        - Dividend Yield: 0-10% → 0-100 (높을수록 좋음)
    """
    
    eps_growth = float(financial_dict.get('eps_growth', 0)) if financial_dict.get('eps_growth') else 0
    dividend_yield = float(financial_dict.get('dividend_yield', 0)) if financial_dict.get('dividend_yield') else 0
    
    # EPS 성장률 점수 (0-30% → 0-100)
    if eps_growth < 0:
        growth_score = 0
    elif eps_growth >= 30:
        growth_score = 100
    else:
        growth_score = (eps_growth / 30) * 100
    
    # 배당수익률 점수 (0-10% → 0-100)
    if dividend_yield <= 0:
        dividend_score = 0
    elif dividend_yield >= 10:
        dividend_score = 100
    else:
        dividend_score = (dividend_yield / 10) * 100
    
    # 평균값 (EPS Growth 60%, Dividend 40%)
    final_score = (growth_score * 0.6) + (dividend_score * 0.4)
    
    return int(round(final_score))


def calculate_cagr_growth_score(financial_history):
    """
    3년 CAGR(복합연간성장률)을 이용한 성장성 평가 (0-100)
    
    매출액과 순이익의 3년 CAGR을 계산하여 기업 성장성을 평가
    
    Args:
        financial_history (dict): 연도별 재무 지표 딕셔너리
            - revenue_history (list): [3년전, 2년전, 1년전, 현재] 매출액 (원)
            - net_income_history (list): [3년전, 2년전, 1년전, 현재] 순이익 (원)
            예: {
                'revenue_history': [1000000000, 1200000000, 1400000000, 1680000000],
                'net_income_history': [100000000, 120000000, 140000000, 168000000]
            }
    
    Returns:
        int: 0-100 범위의 성장점수
            - 80-100: 고성장 (30% 이상 CAGR)
            - 60-79: 양호한 성장 (20-30% CAGR)
            - 40-59: 보통 성장 (10-20% CAGR)
            - 20-39: 저성장 (0-10% CAGR)
            - 0-19: 마이너스/정체 (음수 또는 0% 미만)
    
    Example:
        >>> financial_history = {
        ...     'revenue_history': [1000000000, 1200000000, 1400000000, 1680000000],
        ...     'net_income_history': [100000000, 120000000, 140000000, 168000000]
        ... }
        >>> score = calculate_cagr_growth_score(financial_history)
        >>> print(score)
        75
    
    CAGR Formula:
        CAGR = (Ending Value / Beginning Value) ^ (1 / 3) - 1
        - Ending Value: 현재(4번째) 값
        - Beginning Value: 3년전(1번째) 값
        - 3: 3년 기간
    """
    
    try:
        revenue_history = financial_history.get('revenue_history', [])
        net_income_history = financial_history.get('net_income_history', [])
        
        # 데이터 검증
        if not revenue_history or not net_income_history:
            return 0
        if len(revenue_history) < 2 or len(net_income_history) < 2:
            return 0
        
        # 최근 4개 연도 데이터 사용 (없으면 사용 가능한 최대)
        rev_data = revenue_history[-4:] if len(revenue_history) >= 4 else revenue_history
        ni_data = net_income_history[-4:] if len(net_income_history) >= 4 else net_income_history
        
        # CAGR 계산 함수
        def calculate_cagr(start_value, end_value, years=3):
            """
            CAGR 계산
            Args:
                start_value: 시작 값 (3년전)
                end_value: 종료 값 (현재)
                years: 연수 (기본 3년)
            Returns:
                cagr (float): CAGR (%). 예: 20.5는 20.5% 성장
            """
            if start_value <= 0:
                return -100  # 음수는 계산 불가
            
            try:
                cagr = (((end_value / start_value) ** (1 / years)) - 1) * 100
                return cagr
            except:
                return -100
        
        # 매출액 CAGR 계산
        if len(rev_data) >= 4:
            # 3년 CAGR
            rev_cagr = calculate_cagr(rev_data[0], rev_data[3], 3)
        elif len(rev_data) >= 2:
            # 1년 성장률 (연수 조정)
            years = len(rev_data) - 1
            rev_cagr = calculate_cagr(rev_data[0], rev_data[-1], years)
        else:
            rev_cagr = 0
        
        # 순이익 CAGR 계산
        if len(ni_data) >= 4:
            # 3년 CAGR
            ni_cagr = calculate_cagr(ni_data[0], ni_data[3], 3)
        elif len(ni_data) >= 2:
            # 1년 성장률 (연수 조정)
            years = len(ni_data) - 1
            ni_cagr = calculate_cagr(ni_data[0], ni_data[-1], years)
        else:
            ni_cagr = 0
        
        # CAGR을 0-100 점수로 정규화
        def normalize_cagr_to_score(cagr):
            """
            CAGR을 0-100 점수로 정규화
            - -10% 이상: 0점
            - 0%: 20점
            - 10%: 40점
            - 20%: 60점
            - 30%: 80점
            - 40% 이상: 100점 (상한선)
            """
            if cagr < -10:
                return 0
            elif cagr < 0:
                # -10% ~ 0%: 0 ~ 20점
                return max(0, 20 + (cagr / 10) * 20)
            elif cagr >= 40:
                return 100
            else:
                # 0% ~ 40%: 20 ~ 100점
                return 20 + (cagr / 40) * 80
        
        rev_score = normalize_cagr_to_score(rev_cagr)
        ni_score = normalize_cagr_to_score(ni_cagr)
        
        # 종합 성장점수 (매출액 40%, 순이익 60%)
        # 순이익 성장이 더 중요함
        cagr_growth_score = (rev_score * 0.4) + (ni_score * 0.6)
        
        return int(round(cagr_growth_score))
    
    except Exception as e:
        print(f"Error in calculate_cagr_growth_score: {e}")
        return 0


def calculate_momentum_score(price_df):
    """
    기술적 모멘텀 점수 계산 (0-100)
    
    200일 추세 기울기, MA60 위치, 1년 수익률을 종합하여
    단기-중기 모멘텀 강도를 평가
    
    Args:
        price_df (DataFrame): 주가 데이터 ('close' 칼럼 필수)
            - index: 날짜 (시계열)
            - close: 종가 (float)
            최소 252일 (1년) 이상의 데이터 필요
    
    Returns:
        int: 0-100 범위의 모멘텀 점수
            - 80-100: 강한 상승 모멘텀 (강력 매수)
            - 60-79: 상승 모멘텀 (매수)
            - 40-59: 중립 (보유)
            - 20-39: 하락 모멘텀 (약한 매도)
            - 0-19: 강한 하락 모멘텀 (강력 매도)
    
    Example:
        >>> import pandas as pd
        >>> dates = pd.date_range('2021-01-01', periods=300)
        >>> prices = pd.DataFrame({
        ...     'close': [100 + i*0.5 for i in range(300)]
        ... }, index=dates)
        >>> score = calculate_momentum_score(prices)
        >>> print(score)
        75
    
    Momentum Components:
        1. 200-일 추세 기울기 (40% 비중):
           - 양수: 상승 추세 (높을수록 가점)
           - 음수: 하락 추세 (낮을수록 감점)
           - 정규화: -5% ~ 5% → 0 ~ 100점
        
        2. MA60 비교 (35% 비중):
           - 현재가 > MA60: 강세 신호 (75-100점)
           - 현재가 ≈ MA60: 중립 (50점)
           - 현재가 < MA60: 약세 신호 (0-25점)
        
        3. 1-년 수익률 (25% 비중):
           - 양수 수익: (return / 60%) * 100 (0-100점)
           - 음수 수익: 마이너스 등급 (0-10점)
    """
    
    try:
        if price_df is None or len(price_df) < 60:
            return 50  # 데이터 부족 시 중립값
        
        close = price_df['close'].values
        dates = price_df.index
        
        # 1. 200-일 추세 기울기 계산
        if len(close) >= 200:
            # 최근 200일 데이터로 선형회귀하여 기울기 계산
            recent_200 = close[-200:]
            x = range(len(recent_200))
            
            # 간단한 선형 기울기 계산 (시작과 끝의 차이)
            slope = (recent_200[-1] - recent_200[0]) / recent_200[0]  # 퍼센트 변화
            
            # 정규화: -5% ~ 5% → 0 ~ 100점
            if slope >= 0.05:
                trend_score = 100
            elif slope <= -0.05:
                trend_score = 0
            else:
                # -5% ~ 5%: 0 ~ 100점
                trend_score = ((slope + 0.05) / 0.1) * 100
        else:
            # 데이터가 부족한 경우 간단한 기울기 사용
            slope = (close[-1] - close[0]) / close[0]
            if slope >= 0.05:
                trend_score = 100
            elif slope <= -0.05:
                trend_score = 0
            else:
                trend_score = ((slope + 0.05) / 0.1) * 100
        
        # 2. MA60 위치 계산
        if len(close) >= 60:
            ma60 = close[-60:].mean()
            current_price = close[-1]
            
            # MA60 대비 가격 위치
            price_vs_ma60 = (current_price - ma60) / ma60
            
            if price_vs_ma60 >= 0.05:
                # 5% 이상 위: 강세
                ma60_score = 85 + (min(price_vs_ma60 / 0.2, 1.0) * 15)
            elif price_vs_ma60 >= 0:
                # 0-5% 위: 약한 강세
                ma60_score = 50 + (price_vs_ma60 / 0.05) * 35
            elif price_vs_ma60 >= -0.05:
                # 0-5% 아래: 약한 약세
                ma60_score = 50 - (abs(price_vs_ma60) / 0.05) * 25
            else:
                # 5% 이상 아래: 약세
                ma60_score = max(0, 25 - (abs(price_vs_ma60) / 0.1) * 25)
        else:
            ma60_score = 50
        
        # 3. 1-년 수익률 계산
        if len(close) >= 252:
            # 252거래일 = 약 1년
            one_year_ago_price = close[-252]
            current_price = close[-1]
            
            one_year_return = (current_price - one_year_ago_price) / one_year_ago_price
            
            if one_year_return >= 0:
                # 양수 수익: 0-60% → 0-100점
                if one_year_return >= 0.60:
                    return_score = 100
                else:
                    return_score = (one_year_return / 0.60) * 100
            else:
                # 음수 수익: -50% 이상 손실 → 0-10점
                if one_year_return <= -0.50:
                    return_score = 0
                else:
                    return_score = max(0, 10 + (one_year_return / -0.50) * (-10))
        else:
            return_score = 50
        
        # 종합 모멘텀 점수 (트렌드 40%, MA60 35%, 1년수익 25%)
        momentum_score = (trend_score * 0.4) + (ma60_score * 0.35) + (return_score * 0.25)
        
        return int(round(momentum_score))
    
    except Exception as e:
        print(f"Error in calculate_momentum_score: {e}")
        return 50


class StockScoringEngine:
    """
    종목 통합 평가 엔진
    
    재무 건전성, 상대 밸류에이션, 성장성, 기술적 모멘텀을 종합하여
    최종 투자 등급 및 점수를 산출
    """
    
    def __init__(self, weights=None):
        """
        초기화
        
        Args:
            weights (dict, optional): 커스텀 가중치. None이면 기본값 사용
                예: {'quality': 0.3, 'valuation': 0.3, 'growth': 0.2, 'momentum': 0.2, 'industry_tailwind': 0.0}
                합이 1.0이 아니면 자동 정규화됨
        """
        if weights is None:
            self.weights = get_default_weights()
        else:
            # 커스텀 가중치 설정
            self.weights = weights.copy()
            # 가중치 정규화 (0이 아닌 가중치만 고려)
            non_zero_weights = {k: v for k, v in self.weights.items() if v > 0}
            if non_zero_weights:
                total_weight = sum(non_zero_weights.values())
                self.weights = {k: v / total_weight if v > 0 else 0 for k, v in self.weights.items()}
    
    def set_weights(self, weights):
        """
        가중치 변경 (ablation testing용)
        
        Args:
            weights (dict): 새로운 가중치
        """
        self.__init__(weights)
    
    def get_weights(self):
        """현재 사용 중인 가중치 반환"""
        return self.weights.copy()
    
    def calculate_final_score(self, 
                             financial_dict=None,
                             current_per=None, 
                             industry_per=None,
                             current_pbr=None,
                             industry_pbr=None,
                             financial_history=None,
                             price_df=None,
                             industry_tailwind_score=None,
                             verbose=False):
        """
        모든 평가 지표를 종합하여 최종 점수와 등급 계산
        
        Args:
            financial_dict (dict): 재무 지표
                {'roe', 'debt_ratio', 'operating_margin', 'free_cash_flow'}
            
            current_per (float): 종목 PER
            industry_per (float): 산업 평균 PER
            current_pbr (float): 종목 PBR
            industry_pbr (float, optional): 산업 평균 PBR
            
            financial_history (dict): 재무 이력
                {'revenue_history', 'net_income_history'}
            
            price_df (DataFrame): 주가 데이터 ('close' 칼럼)
            
            industry_tailwind_score (int, optional): 산업 태풍 점수 (0-100)
            
            verbose (bool): 상세 계산 과정 출력 여부
        
        Returns:
            dict: {
                'final_score': 최종 점수 (0-100),
                'investment_grade': 투자 등급 (S/A/B/C/D),
                'confidence_level': 신뢰도 (0-100),
                'score_breakdown': {
                    'quality_score': 재무 건전성 (0-100),
                    'valuation_score': 상대 밸류에이션 (0-100),
                    'growth_score': 성장성 (0-100),
                    'momentum_score': 모멘텀 (0-100),
                    'industry_tailwind_score': 산업 태풍 (0-100, optional)
                },
                'component_weights': {각 성분의 실제 가중치},
                'investment_thesis': 투자 관점 설명
            }
        
        Example:
            >>> engine = StockScoringEngine()
            >>> result = engine.calculate_final_score(
            ...     financial_dict={'roe': 15, 'debt_ratio': 45, 'operating_margin': 12, 'free_cash_flow': 1000000000},
            ...     current_per=12.0,
            ...     industry_per=15.0,
            ...     current_pbr=1.5,
            ...     industry_pbr=2.0,
            ...     financial_history={'revenue_history': [1000, 1300, 1690, 2197], 'net_income_history': [100, 130, 169, 220]},
            ...     price_df=price_dataframe,
            ...     industry_tailwind_score=70
            ... )
            >>> print(f"최종 점수: {result['final_score']}/100")
            >>> print(f"투자 등급: {result['investment_grade']}")
        """
        
        try:
            scores = {}
            weights_used = {}
            
            # 1. 재무 건전성 점수 (Quality Score)
            if financial_dict:
                quality_score = calculate_quality_score(financial_dict)
                scores['quality'] = quality_score
                weights_used['quality'] = self.weights['quality']
                if verbose:
                    print(f"✓ 재무 건전성: {quality_score}/100")
            
            # 2. 상대 밸류에이션 점수 (Relative Valuation Score)
            if current_per is not None and industry_per is not None and current_pbr is not None:
                val_score, per_ratio, pbr_ratio = calculate_relative_valuation_score(
                    current_per, industry_per, current_pbr, industry_pbr
                )
                scores['valuation'] = val_score
                weights_used['valuation'] = self.weights['valuation']
                if verbose:
                    print(f"✓ 상대 밸류에이션: {val_score}/100 (PER {per_ratio:.2f}x, PBR {pbr_ratio:.2f}x)")
            
            # 3. 성장성 점수 (CAGR Growth Score)
            if financial_history:
                growth_score = calculate_cagr_growth_score(financial_history)
                scores['growth'] = growth_score
                weights_used['growth'] = self.weights['growth']
                if verbose:
                    print(f"✓ 3년 성장성: {growth_score}/100")
            
            # 4. 모멘텀 점수 (Momentum Score)
            if price_df is not None:
                momentum_score = calculate_momentum_score(price_df)
                scores['momentum'] = momentum_score
                weights_used['momentum'] = self.weights['momentum']
                if verbose:
                    print(f"✓ 기술적 모멘텀: {momentum_score}/100")
            
            # 5. 산업 태풍 점수 (Industry Tailwind - 옵션)
            if industry_tailwind_score is not None:
                industry_tailwind_score = max(0, min(100, int(industry_tailwind_score)))
                scores['industry_tailwind'] = industry_tailwind_score
                weights_used['industry_tailwind'] = self.weights['industry_tailwind']
                if verbose:
                    print(f"✓ 산업 태풍: {industry_tailwind_score}/100")
            
            # 가중치 정규화 (제공된 지표만 사용)
            total_weight = sum(weights_used.values())
            normalized_weights = {k: v / total_weight for k, v in weights_used.items()}
            
            # 최종 점수 계산 (가중평균)
            final_score = sum(scores[k] * normalized_weights[k] for k in scores.keys())
            final_score = int(round(final_score))
            
            # 투자 등급 결정
            investment_grade = self._get_investment_grade(final_score)
            
            # 신뢰도 계산 (제공된 지표의 수와 품질 기반)
            confidence_level = self._calculate_confidence(scores, list(scores.keys()))
            
            # 투자 관점 생성
            investment_thesis = self._generate_investment_thesis(scores, final_score, investment_grade)
            
            # 결과 딕셔너리
            result = {
                'final_score': final_score,
                'investment_grade': investment_grade,
                'confidence_level': confidence_level,
                'score_breakdown': scores.copy(),
                'component_weights': normalized_weights.copy(),
                'investment_thesis': investment_thesis
            }
            
            if verbose:
                print(f"\n{'='*50}")
                print(f"최종 투자 점수: {final_score}/100")
                print(f"투자 등급: {investment_grade}")
                print(f"신뢰도: {confidence_level}%")
                print(f"{'='*50}")
            
            return result
        
        except Exception as e:
            print(f"Error in calculate_final_score: {e}")
            return {
                'final_score': 50,
                'investment_grade': 'C',
                'confidence_level': 30,
                'score_breakdown': {},
                'component_weights': {},
                'investment_thesis': '데이터 부족으로 신뢰도 낮음'
            }
    
    def _get_investment_grade(self, score):
        """
        점수에 따른 투자 등급 반환 (S, A, B, C, D)
        
        Args:
            score (int): 0-100 점수
        
        Returns:
            str: 투자 등급
                - S: 80 이상 (강력 매수)
                - A: 65-79 (매수)
                - B: 50-64 (보유)
                - C: 35-49 (약한 매도)
                - D: 35 미만 (강력 매도)
        """
        if score >= 80:
            return 'S'
        elif score >= 65:
            return 'A'
        elif score >= 50:
            return 'B'
        elif score >= 35:
            return 'C'
        else:
            return 'D'
    
    def _calculate_confidence(self, scores, score_components):
        """
        신뢰도 계산
        
        신뢰도는 제공된 지표의 수, 점수의 일관성, 데이터 품질 기반
        
        Args:
            scores (dict): 각 성분별 점수
            score_components (list): 사용된 지표 목록
        
        Returns:
            int: 0-100 신뢰도
        """
        # 기본 신뢰도: 제공된 지표 수
        base_confidence = min(100, len(score_components) * 20)
        
        # 점수 일관성 평가 (표준편차 기반)
        if len(scores) > 1:
            score_values = list(scores.values())
            mean_score = sum(score_values) / len(score_values)
            variance = sum((s - mean_score) ** 2 for s in score_values) / len(score_values)
            std_dev = variance ** 0.5
            
            # 표준편차가 낮을수록 (요소들이 일관성 있을수록) 신뢰도 높음
            consistency_bonus = max(0, 20 - (std_dev / 10))
        else:
            consistency_bonus = 0
        
        confidence = min(100, base_confidence + consistency_bonus)
        return int(round(confidence))
    
    def _generate_investment_thesis(self, scores, final_score, grade):
        """
        투자 관점 생성
        
        Args:
            scores (dict): 각 성분별 점수
            final_score (int): 최종 점수
            grade (str): 투자 등급
        
        Returns:
            str: 투자 관점 설명
        """
        thesis = []
        
        # 강점 분석
        strengths = []
        for component, score in scores.items():
            if score >= 75:
                if component == 'quality':
                    strengths.append("탄탄한 재무 건전성")
                elif component == 'valuation':
                    strengths.append("저평가 상태")
                elif component == 'growth':
                    strengths.append("우수한 성장성")
                elif component == 'momentum':
                    strengths.append("긍정적 기술적 신호")
                elif component == 'industry_tailwind':
                    strengths.append("호황하는 산업")
        
        # 약점 분석
        weaknesses = []
        for component, score in scores.items():
            if score <= 35:
                if component == 'quality':
                    weaknesses.append("약한 재무 구조")
                elif component == 'valuation':
                    weaknesses.append("고평가 상태")
                elif component == 'growth':
                    weaknesses.append("부진한 성장성")
                elif component == 'momentum':
                    weaknesses.append("약세 기술적 신호")
                elif component == 'industry_tailwind':
                    weaknesses.append("부진한 산업 환경")
        
        # 최종 투자 의견
        if grade == 'S':
            thesis.append("강력한 매수 추천")
        elif grade == 'A':
            thesis.append("매수 추천")
        elif grade == 'B':
            thesis.append("보유 추천")
        elif grade == 'C':
            thesis.append("약한 매도 신호")
        else:
            thesis.append("강력한 매도 추천")
        
        # 상세 의견
        if strengths:
            thesis.append(f"[강점] {', '.join(strengths)}")
        
        if weaknesses:
            thesis.append(f"[약점] {', '.join(weaknesses)}")
        
        return " | ".join(thesis)


def get_quality_grade(score):
    """
    점수에 따른 등급 반환
    
    Args:
        score (int): 0-100 점수
    
    Returns:
        str: 등급 (AAA ~ D)
    
    Example:
        >>> grade = get_quality_grade(85)
        >>> print(grade)
        AA
    """
    if score >= 90:
        return "AAA"
    elif score >= 80:
        return "AA"
    elif score >= 70:
        return "A"
    elif score >= 60:
        return "BBB"
    elif score >= 50:
        return "BB"
    elif score >= 40:
        return "B"
    elif score >= 30:
        return "CCC"
    elif score >= 20:
        return "CC"
    elif score >= 10:
        return "C"
    else:
        return "D"


if __name__ == "__main__":
    # 테스트 케이스
    print("=== Financial Quality Score Test ===\n")
    
    # 우수 기업 (삼성전자 같은)
    excellent_company = {
        'roe': 15.0,
        'debt_ratio': 45.0,
        'operating_margin': 12.0,
        'free_cash_flow': 1000000000
    }
    
    quality_score = calculate_quality_score(excellent_company)
    grade = get_quality_grade(quality_score)
    print(f"우수 기업 - Quality Score: {quality_score}/100 (등급: {grade})\n")
    
    # 약한 기업
    weak_company = {
        'roe': 3.0,
        'debt_ratio': 200.0,
        'operating_margin': 1.0,
        'free_cash_flow': -500000000
    }
    
    quality_score = calculate_quality_score(weak_company)
    grade = get_quality_grade(quality_score)
    print(f"약한 기업 - Quality Score: {quality_score}/100 (등급: {grade})\n")
    
    # 저평가 기업
    print("=== Valuation Score Test ===\n")
    
    undervalued = {
        'per': 8.0,
        'pbr': 0.8
    }
    
    val_score = calculate_valuation_score(undervalued)
    print(f"저평가 기업 - Valuation Score: {val_score}/100\n")
    
    # 산업평균 비교 밸류에이션 평가
    print("=== Relative Valuation Score Test (산업평균 비교) ===\n")
    
    # 사례 1: 산업평균 대비 저평가 (PER 고평가지만 PBR은 저평가)
    # 삼성전자: PER 12, PBR 1.5 vs 반도체 산업평균: PER 15, PBR 2.0
    val_score, per_ratio, pbr_ratio = calculate_relative_valuation_score(
        current_per=12.0,
        industry_per=15.0,
        current_pbr=1.5,
        industry_pbr=2.0
    )
    val_grade, val_desc = get_valuation_grade(val_score)
    print(f"삼성전자 (반도체) - Valuation Score: {val_score}/100")
    print(f"  등급: {val_grade} ({val_desc})")
    print(f"  PER 비율: {per_ratio:.2f}x (산업평균 대비)")
    print(f"  PBR 비율: {pbr_ratio:.2f}x (산업평균 대비)\n")
    
    # 사례 2: 산업평균 대비 매우 저평가
    # NAVER: PER 20, PBR 2.0 vs IT 산업평균: PER 30, PBR 3.5
    val_score, per_ratio, pbr_ratio = calculate_relative_valuation_score(
        current_per=20.0,
        industry_per=30.0,
        current_pbr=2.0,
        industry_pbr=3.5
    )
    val_grade, val_desc = get_valuation_grade(val_score)
    print(f"NAVER (IT) - Valuation Score: {val_score}/100")
    print(f"  등급: {val_grade} ({val_desc})")
    print(f"  PER 비율: {per_ratio:.2f}x (산업평균 대비)")
    print(f"  PBR 비율: {pbr_ratio:.2f}x (산업평균 대비)\n")
    
    # 사례 3: 산업평균 대비 고평가
    # 특정 기업: PER 25, PBR 2.5 vs 동종 산업평균: PER 18, PBR 1.8
    val_score, per_ratio, pbr_ratio = calculate_relative_valuation_score(
        current_per=25.0,
        industry_per=18.0,
        current_pbr=2.5,
        industry_pbr=1.8
    )
    val_grade, val_desc = get_valuation_grade(val_score)
    print(f"고평가 기업 - Valuation Score: {val_score}/100")
    print(f"  등급: {val_grade} ({val_desc})")
    print(f"  PER 비율: {per_ratio:.2f}x (산업평균 대비)")
    print(f"  PBR 비율: {pbr_ratio:.2f}x (산업평균 대비)\n")
    
    # 사례 4: 산업평균 대비 매우 고평가
    # 투기성 기업: PER 50, PBR 4.0 vs 동종 산업평균: PER 20, PBR 1.5
    val_score, per_ratio, pbr_ratio = calculate_relative_valuation_score(
        current_per=50.0,
        industry_per=20.0,
        current_pbr=4.0,
        industry_pbr=1.5
    )
    val_grade, val_desc = get_valuation_grade(val_score)
    print(f"투기성 기업 - Valuation Score: {val_score}/100")
    print(f"  등급: {val_grade} ({val_desc})")
    print(f"  PER 비율: {per_ratio:.2f}x (산업평균 대비)")
    print(f"  PBR 비율: {pbr_ratio:.2f}x (산업평균 대비)\n")
    
    # 사례 5: 적정가 지역
    # 적정가 기업: PER 16, PBR 1.8 vs 동종 산업평균: PER 15, PBR 1.8
    val_score, per_ratio, pbr_ratio = calculate_relative_valuation_score(
        current_per=16.0,
        industry_per=15.0,
        current_pbr=1.8,
        industry_pbr=1.8
    )
    val_grade, val_desc = get_valuation_grade(val_score)
    print(f"적정가 기업 - Valuation Score: {val_score}/100")
    print(f"  등급: {val_grade} ({val_desc})")
    print(f"  PER 비율: {per_ratio:.2f}x (산업평균 대비)")
    print(f"  PBR 비율: {pbr_ratio:.2f}x (산업평균 대비)\n")
    
    # 성장성 평가
    print("=== Growth Score Test ===\n")
    
    growth_company = {
        'eps_growth': 20.0,
        'dividend_yield': 4.5
    }
    
    growth_score = calculate_growth_score(growth_company)
    print(f"성장 기업 - Growth Score: {growth_score}/100\n")
    
    # 3년 CAGR 기반 성장성 평가
    print("=== 3-Year CAGR Growth Score Test ===\n")
    
    # 고성장 기업 (30% CAGR)
    high_growth_company = {
        'revenue_history': [1000000000, 1300000000, 1690000000, 2197000000],  # 30% CAGR
        'net_income_history': [100000000, 130000000, 169000000, 219700000]    # 30% CAGR
    }
    
    cagr_score = calculate_cagr_growth_score(high_growth_company)
    grade = get_quality_grade(cagr_score)
    print(f"고성장 기업 (30% CAGR) - CAGR Growth Score: {cagr_score}/100 (등급: {grade})\n")
    
    # 보통 성장 기업 (15% CAGR)
    normal_growth_company = {
        'revenue_history': [1000000000, 1150000000, 1322500000, 1521000000],  # 15% CAGR
        'net_income_history': [100000000, 115000000, 132250000, 152100000]    # 15% CAGR
    }
    
    cagr_score = calculate_cagr_growth_score(normal_growth_company)
    grade = get_quality_grade(cagr_score)
    print(f"보통 성장 기업 (15% CAGR) - CAGR Growth Score: {cagr_score}/100 (등급: {grade})\n")
    
    # 저성장 기업 (5% CAGR)
    low_growth_company = {
        'revenue_history': [1000000000, 1050000000, 1102500000, 1157625000],  # 5% CAGR
        'net_income_history': [100000000, 90000000, 81000000, 72900000]       # -10% CAGR
    }
    
    cagr_score = calculate_cagr_growth_score(low_growth_company)
    grade = get_quality_grade(cagr_score)
    print(f"저성장 기업 (등락) - CAGR Growth Score: {cagr_score}/100 (등급: {grade})\n")
    
    # 마이너스 성장 기업 (-10% CAGR)
    negative_growth_company = {
        'revenue_history': [1000000000, 950000000, 902500000, 857375000],  # -5% CAGR
        'net_income_history': [100000000, 70000000, 49000000, 34300000]    # -30% CAGR
    }
    
    cagr_score = calculate_cagr_growth_score(negative_growth_company)
    grade = get_quality_grade(cagr_score)
    print(f"마이너스 성장 기업 (-30% CAGR) - CAGR Growth Score: {cagr_score}/100 (등급: {grade})\n")
    
    # 모멘텀 점수 평가
    print("=== Momentum Score Test ===\n")
    
    import pandas as pd
    import numpy as np
    
    # 테스트 케이스 1: 강한 상승 모멘텀
    # 지난 200일 동안 30% 상승, 현재가 MA60 위, 1년 수익 50%
    dates = pd.date_range('2023-01-01', periods=300)
    prices_uptrend = pd.DataFrame({
        'close': np.linspace(100, 130, 200).tolist() + list(np.linspace(130, 145, 100))
    }, index=dates)
    
    momentum_score = calculate_momentum_score(prices_uptrend)
    momentum_grade = get_quality_grade(momentum_score)
    print(f"강한 상승 모멘텀 - Momentum Score: {momentum_score}/100 (등급: {momentum_grade})\n")
    
    # 테스트 케이스 2: 중립 모멘텀
    # 지난 200일 거의 변화 없음, 현재가 MA60 근처, 1년 수익 5%
    prices_neutral = pd.DataFrame({
        'close': np.linspace(100, 100.5, 200).tolist() + list(np.linspace(100.5, 105, 100))
    }, index=dates)
    
    momentum_score = calculate_momentum_score(prices_neutral)
    momentum_grade = get_quality_grade(momentum_score)
    print(f"중립 모멘텀 - Momentum Score: {momentum_score}/100 (등급: {momentum_grade})\n")
    
    # 테스트 케이스 3: 낙하 모멘텀
    # 지난 200일 동안 25% 하락, 현재가 MA60 아래, 1년 수익 -30%
    prices_downtrend = pd.DataFrame({
        'close': np.linspace(100, 75, 200).tolist() + list(np.linspace(75, 70, 100))
    }, index=dates)
    
    momentum_score = calculate_momentum_score(prices_downtrend)
    momentum_grade = get_quality_grade(momentum_score)
    print(f"낙하 모멘텀 - Momentum Score: {momentum_score}/100 (등급: {momentum_grade})\n")
    
    # 테스트 케이스 4: 강한 하락 모멘텀
    # 지난 200일 동안 40% 하락, 현재가 MA60 아래, 1년 수익 -60%
    prices_strong_downtrend = pd.DataFrame({
        'close': np.linspace(100, 60, 200).tolist() + list(np.linspace(60, 40, 100))
    }, index=dates)
    
    momentum_score = calculate_momentum_score(prices_strong_downtrend)
    momentum_grade = get_quality_grade(momentum_score)
    print(f"강한 하락 모멘텀 - Momentum Score: {momentum_score}/100 (등급: {momentum_grade})\n")
    
    # StockScoringEngine 테스트
    print("=== Stock Scoring Engine Test ===\n")
    
    engine = StockScoringEngine()
    
    # 시나리오 1: 우수한 기업 (강력 매수)
    print("시나리오 1: 우수한 기업\n")
    excellent_stock = {
        'financial_dict': {
            'roe': 16.0,
            'debt_ratio': 40.0,
            'operating_margin': 13.0,
            'free_cash_flow': 1200000000
        },
        'current_per': 10.0,
        'industry_per': 14.0,
        'current_pbr': 1.2,
        'industry_pbr': 1.8,
        'financial_history': {
            'revenue_history': [1000000000, 1350000000, 1822500000, 2460375000],  # 35% CAGR
            'net_income_history': [120000000, 162000000, 218700000, 295245000]
        },
        'price_df': prices_uptrend,
        'industry_tailwind_score': 75
    }
    
    result = engine.calculate_final_score(**excellent_stock, verbose=True)
    print(f"점수 분석:")
    for component, score in result['score_breakdown'].items():
        print(f"  - {component}: {score}/100 (가중치: {result['component_weights'][component]:.1%})")
    print(f"\n투자 관점: {result['investment_thesis']}\n")
    print(f"신뢰도: {result['confidence_level']}%\n")
    
    # 시나리오 2: 중간 기업 (보유)
    print("="*60)
    print("시나리오 2: 중간 정도의 기업\n")
    medium_stock = {
        'financial_dict': {
            'roe': 10.0,
            'debt_ratio': 55.0,
            'operating_margin': 8.0,
            'free_cash_flow': 500000000
        },
        'current_per': 15.0,
        'industry_per': 16.0,
        'current_pbr': 1.5,
        'industry_pbr': 1.6,
        'financial_history': {
            'revenue_history': [1000000000, 1100000000, 1210000000, 1331000000],  # 10% CAGR
            'net_income_history': [100000000, 105000000, 110250000, 115762500]
        },
        'price_df': prices_neutral,
        'industry_tailwind_score': 50
    }
    
    result = engine.calculate_final_score(**medium_stock, verbose=True)
    print(f"점수 분석:")
    for component, score in result['score_breakdown'].items():
        print(f"  - {component}: {score}/100 (가중치: {result['component_weights'][component]:.1%})")
    print(f"\n투자 관점: {result['investment_thesis']}\n")
    print(f"신뢰도: {result['confidence_level']}%\n")
    
    # 시나리오 3: 약한 기업 (강력 매도)
    print("="*60)
    print("시나리오 3: 약한 기업\n")
    weak_stock = {
        'financial_dict': {
            'roe': 3.0,
            'debt_ratio': 180.0,
            'operating_margin': 2.0,
            'free_cash_flow': -200000000
        },
        'current_per': 25.0,
        'industry_per': 16.0,
        'current_pbr': 2.8,
        'industry_pbr': 1.6,
        'financial_history': {
            'revenue_history': [1000000000, 950000000, 902500000, 857375000],  # -5% CAGR
            'net_income_history': [100000000, 60000000, 30000000, 0]
        },
        'price_df': prices_strong_downtrend,
        'industry_tailwind_score': 25
    }
    
    result = engine.calculate_final_score(**weak_stock, verbose=True)
    print(f"점수 분석:")
    for component, score in result['score_breakdown'].items():
        print(f"  - {component}: {score}/100 (가중치: {result['component_weights'][component]:.1%})")
    print(f"\n투자 관점: {result['investment_thesis']}\n")
    print(f"신뢰도: {result['confidence_level']}%\n")
