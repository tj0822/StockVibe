"""
Streamlit UI 유틸리티 - Score Breakdown 표시 헬퍼

Score breakdown을 Streamlit에서 시각적으로 표시하기 위한 함수들
"""

import streamlit as st
import pandas as pd


def display_score_breakdown(score_result: dict, stock_name: str = ""):
    """
    점수 분석 결과를 Streamlit UI에 표시
    
    Args:
        score_result (dict): calculate_final_score() 반환 결과
            - final_score, investment_grade, confidence_level, score_breakdown
        stock_name (str): 종목명 (옵션)
    
    Example:
        >>> from financial_utils import StockScoringEngine
        >>> from app.ui_utils import display_score_breakdown
        >>> engine = StockScoringEngine()
        >>> result = engine.calculate_final_score(...)
        >>> display_score_breakdown(result, "Samsung Electronics")
    """
    
    # 최상단 메트릭
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📊 최종 점수",
            f"{score_result.get('final_score', 0)}/100",
            delta=score_result.get('investment_grade', 'N/A')
        )
    
    with col2:
        st.metric(
            "📈 투자 등급",
            score_result.get('investment_grade', 'N/A'),
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "🎯 신뢰도",
            f"{score_result.get('confidence_level', 0)}%",
            delta="높음" if score_result.get('confidence_level', 0) >= 70 else "보통"
        )
    
    # Score Breakdown 테이블
    st.markdown("---")
    st.subheader("📋 점수 분석")
    
    breakdown = score_result.get('score_breakdown', {})
    weights = score_result.get('component_weights', {})
    
    if breakdown:
        # DataFrame 생성
        breakdown_data = []
        for factor, score in breakdown.items():
            weight = weights.get(factor, 0)
            breakdown_data.append({
                'Factor': factor.replace('_', ' ').title(),
                'Score': f"{score}/100",
                'Weight': f"{weight:.1%}",
                'Contribution': f"{score * weight:.1f}"
            })
        
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Score breakdown 정보가 없습니다.")
    
    # 투자 관점 (있으면 표시)
    thesis = score_result.get('investment_thesis', '')
    if thesis:
        st.markdown("---")
        st.subheader("💡 투자 관점")
        st.info(thesis)


def display_score_comparison(results_list: list):
    """
    여러 종목의 점수를 비교 표시
    
    Args:
        results_list (list): [
            {'name': '삼성전자', 'code': '005930', 'score_result': {...}},
            {'name': 'LG화학', 'code': '051910', 'score_result': {...}},
            ...
        ]
    """
    
    comparison_data = []
    for item in results_list:
        name = item.get('name', 'N/A')
        code = item.get('code', 'N/A')
        result = item.get('score_result', {})
        
        comparison_data.append({
            '종목': f"{name} ({code})",
            '점수': result.get('final_score', 0),
            '등급': result.get('investment_grade', 'N/A'),
            '신뢰도': f"{result.get('confidence_level', 0)}%"
        })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(
            df_comparison.sort_values('점수', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("⚠️ 비교할 데이터가 없습니다.")


def get_score_color(score: int) -> str:
    """
    점수에 따른 색상 반환 (마크다운/HTML용)
    
    Args:
        score (int): 0-100 점수
    
    Returns:
        str: 색상 코드 (hex)
    """
    if score >= 80:
        return "#00FF00"  # 녹색
    elif score >= 65:
        return "#7FFF00"  # 연두
    elif score >= 50:
        return "#FFFF00"  # 노랑
    elif score >= 35:
        return "#FF8C00"  # 주황
    else:
        return "#FF0000"  # 빨강


def grade_to_emoji(grade: str) -> str:
    """
    투자 등급을 emoji로 변환
    
    Args:
        grade (str): 'S', 'A', 'B', 'C', 'D' 등
    
    Returns:
        str: emoji + grade
    """
    emoji_map = {
        'S': '🌟', 'A': '⭐', 'B': '👍',
        'C': '👌', 'D': '⚠️', 'F': '🚫'
    }
    emoji = emoji_map.get(grade, '❓')
    return f"{emoji} {grade}"
