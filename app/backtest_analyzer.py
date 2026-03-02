"""
종목별 일괄 백테스트 분석 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
import streamlit as st


def run_batch_backtest(
    df: pd.DataFrame,
    stock_codes: List[str],
    rolling_days: int,
    volume_threshold: float,
    loss_threshold_pct: float,
    start_date,
    end_date,
    build_signals_func: Callable,
    backtest_func: Callable,
    kospi_index: pd.DataFrame,
    kospi_list: Dict = None,
    initial_cash: float = 50_000_000,
    max_daily_buys: int = 2,
    buy_unit: float = 2_000_000,
    kospi_bullish_only: bool = False,
    kospi_bullish_lookback_months: int = 6,
) -> pd.DataFrame:
    """
    여러 종목에 대해 일괄 백테스트 실행
    
    Args:
        df: 가격 데이터
        stock_codes: 테스트할 종목 코드 리스트
        rolling_days: 거래량 분석 기간
        volume_threshold: 급등 기준 배수
        loss_threshold_pct: 추가매수 손실 임계값
        start_date: 시뮬레이션 시작일
        end_date: 시뮬레이션 종료일
        build_signals_func: 신호 생성 함수
        backtest_func: 백테스트 실행 함수
        kospi_index: KOSPI 지수
        kospi_list: KOSPI 종목 목록 (코드-이름 매핑)
        initial_cash: 초기 자본금
        max_daily_buys: 일일 최대 매수 수
        buy_unit: 매수 단위
        kospi_bullish_only: 코스피 상승기간에만 매수 여부
        kospi_bullish_lookback_months: 코스피 상승판정 기준 개월 수
    
    Returns:
        성과 지표 데이터프레임
    """
    all_metrics = []
    progress_placeholder = st.empty()
    results_placeholder = st.empty()
    debug_placeholder = st.empty()
    
    total_stocks = len(stock_codes)
    
    # 디버깅 카운터
    with_data = 0
    with_date = 0
    with_signal = 0
    with_backtest = 0
    error_log = []
    
    for idx, stock_code in enumerate(stock_codes):
        try:
            # 진행률 표시
            progress = (idx + 1) / total_stocks
            stock_code_str = str(stock_code)
            progress_placeholder.progress(progress, text=f"테스트 중: {idx + 1}/{total_stocks} - {stock_code_str}")
            
            # 종목별 데이터 필터링
            stock_df = df[df['code'].astype(str) == stock_code_str].copy()
            
            if stock_df.empty:
                continue
            
            with_data += 1
            
            # 날짜 필터링
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            stock_df = stock_df[(stock_df['date'] >= start_date_dt) & (stock_df['date'] <= end_date_dt)]
            
            if stock_df.empty:
                continue
            
            with_date += 1
            
            # 신호 생성
            signals = build_signals_func(
                stock_df,
                rolling_days,
                volume_threshold,
                20, 5.0,  # 추가 파라미터
                20, 2.0,
                20, 2.0,
                ["Turnover Spike"],
                "ANY",
            )
            
            if signals.empty:
                continue
            
            with_signal += 1
            
            # 백테스트 실행
            try:
                equity_df, trades_df = backtest_func(
                    stock_df,
                    signals,
                    kospi_index,
                    start_date_dt,
                    top_n=max(1, int(max_daily_buys)),
                    initial_cash=initial_cash,
                    max_daily_buys=max_daily_buys,
                    buy_unit=buy_unit,
                    kospi_bullish_only=kospi_bullish_only,
                    kospi_bullish_lookback_months=kospi_bullish_lookback_months,
                    add_buy_threshold_pct=loss_threshold_pct,
                )
            except Exception as bt_error:
                error_log.append(f"{stock_code_str}: 백테스트 오류 - {str(bt_error)}")
                continue
            
            with_backtest += 1
            
            # 종목명 추출 (우선순위: stock_df의 name 컬럼 > kospi_list > 코드)
            stock_name = stock_code_str  # 기본값
            
            if 'name' in stock_df.columns and stock_df['name'].notna().any():
                stock_name = stock_df['name'].iloc[0]
            elif isinstance(kospi_list, dict):
                stock_name = kospi_list.get(stock_code_str, stock_code_str)
            elif isinstance(kospi_list, pd.DataFrame):
                # DataFrame인 경우 code 컬럼으로 이름 찾기
                name_row = kospi_list[kospi_list['code'].astype(str) == stock_code_str]
                if not name_row.empty:
                    stock_name = name_row['name'].iloc[0]
            
            # 성과 지표 계산
            metrics = calculate_backtest_metrics(equity_df, trades_df, stock_code_str, stock_name)
            all_metrics.append(metrics)
            
            # 중간 결과 표시
            if len(all_metrics) % 5 == 0:
                metrics_df = format_metrics_for_display(all_metrics)
                with results_placeholder.container():
                    st.dataframe(
                        metrics_df.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )
        
        except Exception as e:
            import traceback
            error_log.append(f"{stock_code}: {str(e)} - {traceback.format_exc()}")
            continue
    
    progress_placeholder.empty()
    
    # 디버깅 정보 표시
    with debug_placeholder.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("전체 종목", total_stocks)
        with col2:
            st.metric("데이터 있음", with_data)
        with col3:
            st.metric("기간 내", with_date)
        with col4:
            st.metric("신호 생성", with_signal)
        with col5:
            st.metric("백테스트 성공", with_backtest)
        
        # 최종 결과
        st.metric("최종 결과", len(all_metrics))
        
        # 에러 로그 표시
        if error_log and len(error_log) <= 10:
            st.warning("⚠️ 오류 발생:")
            for err in error_log[:10]:
                st.caption(f"❌ {err}")
        elif error_log:
            st.warning(f"⚠️ {len(error_log)}개 종목에서 오류 발생 (처음 10개):")
            for err in error_log[:10]:
                st.caption(f"❌ {err}")
    
    # 최종 결과 반환
    if all_metrics:
        return format_metrics_for_display(all_metrics)
    else:
        st.warning(
            f"⚠️ 테스트 가능한 데이터가 부족합니다.\n"
            f"- 전체: {total_stocks}개\n"
            f"- 데이터 있음: {with_data}개\n"
            f"- 기간 내 데이터: {with_date}개\n"
            f"- 신호 생성: {with_signal}개"
        )
        return pd.DataFrame()


def calculate_backtest_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    stock_code: str,
    stock_name: str
) -> Dict:
    """
    백테스트 결과에서 성과 지표 계산
    
    Args:
        equity_df: 자산 변화 데이터프레임
        trades_df: 거래 내역 데이터프레임 (return_pct: 매수가격 대비 수익률)
        stock_code: 종목 코드
        stock_name: 종목명
    
    Returns:
        성과 지표 딕셔너리
        - total_return_pct: 초기 자본금 대비 수익률
        - avg_return: 개별 거래(매수가격 대비)의 평균 수익률
        - win_rate: 수익 거래 비율
        - avg_win/avg_loss: 평균 수익/손실 (매수가격 대비)
    """
    metrics = {
        'code': stock_code,
        'name': stock_name,
        'total_trades': 0,
        'win_trades': 0,
        'lose_trades': 0,
        'breakeven_trades': 0,
        'win_rate': 0.0,
        'avg_return': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'profit_loss_ratio': 0.0,
        'total_return': 0.0,
        'total_return_pct': 0.0,
        'max_drawdown': 0.0,
        'final_equity': 0.0,
        'initial_equity': 0.0,
    }
    
    try:
        # 최종 자산
        if not equity_df.empty:
            metrics['final_equity'] = float(equity_df['equity'].iloc[-1])
            metrics['initial_equity'] = float(equity_df['equity'].iloc[0])
            metrics['total_return'] = metrics['final_equity'] - metrics['initial_equity']
            metrics['total_return_pct'] = (metrics['final_equity'] / metrics['initial_equity'] - 1) * 100
            
            # 최대 낙폭 (MDD) 계산
            cummax = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - cummax) / cummax * 100
            metrics['max_drawdown'] = float(drawdown.min())
        
        # 거래 통계
        if not trades_df.empty:
            sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
            
            if not sell_trades.empty and 'return_pct' in sell_trades.columns:
                sell_trades_valid = sell_trades.dropna(subset=['return_pct'])
                
                if not sell_trades_valid.empty:
                    total_trades = len(sell_trades_valid)
                    winning_trades = len(sell_trades_valid[sell_trades_valid['return_pct'] > 0])
                    losing_trades = len(sell_trades_valid[sell_trades_valid['return_pct'] < 0])
                    breakeven_trades = len(sell_trades_valid[sell_trades_valid['return_pct'] == 0])
                    
                    metrics['total_trades'] = total_trades
                    metrics['win_trades'] = winning_trades
                    metrics['lose_trades'] = losing_trades
                    metrics['breakeven_trades'] = breakeven_trades
                    metrics['win_rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    metrics['avg_return'] = float(sell_trades_valid['return_pct'].mean())
                    
                    if winning_trades > 0:
                        metrics['avg_win'] = float(sell_trades_valid[sell_trades_valid['return_pct'] > 0]['return_pct'].mean())
                    
                    if losing_trades > 0:
                        metrics['avg_loss'] = float(sell_trades_valid[sell_trades_valid['return_pct'] < 0]['return_pct'].mean())
                    
                    # 손익비
                    if metrics['avg_loss'] != 0:
                        metrics['profit_loss_ratio'] = metrics['avg_win'] / abs(metrics['avg_loss'])
    
    except Exception as e:
        print(f"Error calculating metrics for {stock_code}: {e}")
    
    return metrics


def format_metrics_for_display(metrics_list: List[Dict]) -> pd.DataFrame:
    """
    성과 지표 리스트를 표시용 데이터프레임으로 변환
    
    Args:
        metrics_list: 성과 지표 딕셔너리 리스트
    
    Returns:
        표시용 데이터프레임 (중복 제거됨)
    """
    df = pd.DataFrame(metrics_list)
    
    if not df.empty:
        # 중복 제거: 같은 code가 여러개면 total_return_pct가 가장 높은 것만 유지
        df = df.sort_values('total_return_pct', ascending=False).drop_duplicates(subset=['code'], keep='first')
        
        # 수익률별 정렬 (내림차순)
        df = df.sort_values('total_return_pct', ascending=False).reset_index(drop=True)
        
        # 표시 형식 정렬
        columns_order = [
            'code', 'name', 'total_return_pct', 'total_return',
            'win_rate', 'avg_return', 'profit_loss_ratio', 'max_drawdown',
            'total_trades', 'win_trades', 'lose_trades'
        ]
        
        available_cols = [col for col in columns_order if col in df.columns]
        df = df[available_cols]
    
    return df


def get_top_performing_stocks(
    metrics_df: pd.DataFrame,
    sort_by: str = 'total_return_pct',
    top_n: int = 10,
    min_trades: int = 3
) -> pd.DataFrame:
    """
    성과 기준으로 상위 종목 추출
    
    Args:
        metrics_df: 성과 지표 데이터프레임
        sort_by: 정렬 기준 ('total_return_pct', 'win_rate', 'profit_loss_ratio' 등)
        top_n: 상위 N개 종목
        min_trades: 최소 거래 수
    
    Returns:
        상위 종목 데이터프레임
    """
    if metrics_df.empty:
        return pd.DataFrame()
    
    # 최소 거래 수 필터링
    filtered = metrics_df[metrics_df['total_trades'] >= min_trades].copy()
    
    if filtered.empty:
        return filtered
    
    # 정렬
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=False)
    
    return filtered.head(top_n).reset_index(drop=True)


def get_filter_criteria(metrics_df: pd.DataFrame) -> Dict:
    """
    필터링을 위한 기준값 계산
    
    Args:
        metrics_df: 성과 지표 데이터프레임
    
    Returns:
        필터링 기준 딕셔너리
    """
    if metrics_df.empty:
        return {
            'min_return_pct': 0.0,
            'min_win_rate': 0.0,
            'min_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    return {
        'min_return_pct': float(metrics_df['total_return_pct'].quantile(0.25)),
        'min_win_rate': float(metrics_df['win_rate'].quantile(0.25)),
        'min_ratio': float(metrics_df['profit_loss_ratio'].quantile(0.25)),
        'max_drawdown': float(metrics_df['max_drawdown'].quantile(0.75))  # 최악의 낙폭
    }
