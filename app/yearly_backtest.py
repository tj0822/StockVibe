"""
연도별 개별 종목 시뮬레이션 분석 모듈
"""
import pandas as pd
from typing import Dict, List, Callable
import streamlit as st


def run_yearly_backtest(
    df: pd.DataFrame,
    stock_code: str,
    stock_name: str,
    rolling_days: int,
    volume_threshold: float,
    loss_threshold_pct: float,
    build_signals_func: Callable,
    backtest_func: Callable,
    kospi_index: pd.DataFrame,
    initial_cash: float = 50_000_000,
    max_daily_buys: int = 2,
    buy_unit: float = 2_000_000,
    start_year: int = 2010,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    특정 종목에 대해 연도별로 백테스트 실행
    
    Args:
        df: 가격 데이터
        stock_code: 종목 코드
        stock_name: 종목명
        rolling_days: 거래량 분석 기간
        volume_threshold: 급등 기준 배수
        loss_threshold_pct: 추가매수 손실 임계값
        build_signals_func: 신호 생성 함수
        backtest_func: 백테스트 실행 함수
        kospi_index: KOSPI 지수
        initial_cash: 초기 자본금
        max_daily_buys: 일일 최대 매수 수
        buy_unit: 매수 단위
        start_year: 시작 연도
        end_year: 종료 연도
    
    Returns:
        연도별 성과 지표 데이터프레임 (각 행은 한 해의 결과)
        - total_return_pct: 그 연도 초기 자본금 대비 수익률
        - avg_return: 개별 거래(매수가격 대비)의 평균 수익률
    """
    yearly_results = []
    progress_placeholder = st.empty()
    results_placeholder = st.empty()
    
    # 종목 데이터 필터링
    stock_df = df[df['code'].astype(str) == str(stock_code)].copy()
    
    if stock_df.empty:
        st.error(f"❌ 종목 {stock_code} 데이터가 없습니다.")
        return pd.DataFrame()
    
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df['year'] = stock_df['date'].dt.year
    
    years = sorted([y for y in stock_df['year'].unique() if start_year <= y <= end_year])
    
    if not years:
        st.error(f"❌ 선택한 기간({start_year}~{end_year})에 데이터가 없습니다.")
        return pd.DataFrame()
    
    for idx, year in enumerate(years):
        try:
            progress = (idx + 1) / len(years)
            progress_placeholder.progress(progress, text=f"분석 중: {year}년 ({idx + 1}/{len(years)})")
            
            # 연도별 데이터 필터링
            year_data = stock_df[stock_df['year'] == year].copy()
            
            if year_data.empty or len(year_data) < 20:
                continue
            
            start_date = pd.to_datetime(f"{year}-01-01")
            end_date = pd.to_datetime(f"{year}-12-31")
            
            # 신호 생성
            signals = build_signals_func(
                year_data,
                rolling_days,
                volume_threshold,
                20, 5.0,
                20, 2.0,
                20, 2.0,
                ["Turnover Spike"],
                "ANY",
            )
            
            if signals.empty:
                continue
            
            # 백테스트 실행
            try:
                equity_df, trades_df = backtest_func(
                    year_data,
                    signals,
                    kospi_index,
                    start_date,
                    top_n=2,
                    initial_cash=initial_cash,
                    max_daily_buys=max_daily_buys,
                    buy_unit=buy_unit,
                    add_buy_threshold_pct=loss_threshold_pct,
                )
            except Exception as e:
                st.warning(f"⚠️ {year}년 백테스트 오류: {str(e)}")
                continue
            
            # 연도별 성과 지표 계산
            metrics = calculate_yearly_metrics(
                year,
                equity_df,
                trades_df,
                stock_code,
                stock_name
            )
            yearly_results.append(metrics)
            
            # 중간 결과 표시
            if len(yearly_results) > 0:
                results_df = pd.DataFrame(yearly_results)
                with results_placeholder.container():
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                    )
        
        except Exception as e:
            st.warning(f"⚠️ {year}년 처리 오류: {str(e)}")
            continue
    
    progress_placeholder.empty()
    
    if yearly_results:
        return pd.DataFrame(yearly_results)
    else:
        st.error("❌ 분석 가능한 연도가 없습니다.")
        return pd.DataFrame()


def calculate_yearly_metrics(
    year: int,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    stock_code: str,
    stock_name: str
) -> Dict:
    """
    연도별 백테스트 결과에서 성과 지표 계산
    
    Args:
        year: 연도
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
        'year': year,
        'code': stock_code,
        'name': stock_name,
        'total_trades': 0,
        'win_trades': 0,
        'lose_trades': 0,
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
            metrics['total_return_pct'] = (metrics['final_equity'] / metrics['initial_equity'] - 1) * 100 if metrics['initial_equity'] > 0 else 0
            
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
                    
                    metrics['total_trades'] = total_trades
                    metrics['win_trades'] = winning_trades
                    metrics['lose_trades'] = losing_trades
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
        st.warning(f"지표 계산 오류 ({year}년): {e}")
    
    return metrics


def analyze_consistency(yearly_df: pd.DataFrame) -> Dict:
    """
    연도별 성과의 일관성 분석
    
    Args:
        yearly_df: 연도별 성과 지표 데이터프레임
    
    Returns:
        일관성 분석 결과 딕셔너리
    """
    if yearly_df.empty:
        return {}
    
    return {
        'avg_return_pct': yearly_df['total_return_pct'].mean(),
        'std_return_pct': yearly_df['total_return_pct'].std(),
        'min_return_pct': yearly_df['total_return_pct'].min(),
        'max_return_pct': yearly_df['total_return_pct'].max(),
        'positive_years': len(yearly_df[yearly_df['total_return_pct'] > 0]),
        'total_years': len(yearly_df),
        'win_rate_avg': yearly_df['win_rate'].mean(),
        'win_rate_std': yearly_df['win_rate'].std(),
    }


def run_yearly_backtest_batch(
    df: pd.DataFrame,
    stock_codes: List[str],
    rolling_days: int,
    volume_threshold: float,
    loss_threshold_pct: float,
    build_signals_func: Callable,
    backtest_func: Callable,
    kospi_index: pd.DataFrame,
    kospi_list: pd.DataFrame = None,
    initial_cash: float = 50_000_000,
    max_daily_buys: int = 2,
    buy_unit: float = 2_000_000,
    start_year: int = 2010,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    모든 종목에 대해 연도별로 백테스트 실행
    
    Args:
        df: 가격 데이터
        stock_codes: 테스트할 종목 코드 리스트
        rolling_days: 거래량 분석 기간
        volume_threshold: 급등 기준 배수
        loss_threshold_pct: 추가매수 손실 임계값
        build_signals_func: 신호 생성 함수
        backtest_func: 백테스트 실행 함수
        kospi_index: KOSPI 지수
        kospi_list: KOSPI 종목 목록 (코드-이름 매핑)
        initial_cash: 초기 자본금
        max_daily_buys: 일일 최대 매수 수
        buy_unit: 매수 단위
        start_year: 시작 연도
        end_year: 종료 연도
    
    Returns:
        모든 종목의 연도별 성과 지표 데이터프레임
        각 행은 (종목코드, 연도)별 결과
        - total_return_pct: 그 연도 초기 자본금 대비 수익률
        - avg_return: 개별 거래(매수가격 대비)의 평균 수익률
    """
    all_yearly_results = []
    progress_placeholder = st.empty()
    results_placeholder = st.empty()
    debug_placeholder = st.empty()
    
    total_stocks = len(stock_codes)
    processed_stocks = 0
    
    for idx, stock_code in enumerate(stock_codes):
        try:
            progress = (idx + 1) / total_stocks
            stock_code_str = str(stock_code)
            progress_placeholder.progress(progress, text=f"분석 중: {idx + 1}/{total_stocks} - {stock_code_str}")
            
            # 종목 데이터 필터링
            stock_df = df[df['code'].astype(str) == stock_code_str].copy()
            
            if stock_df.empty or len(stock_df) < 50:
                continue
            
            # 종목명 추출
            if isinstance(kospi_list, pd.DataFrame):
                name_row = kospi_list[kospi_list['code'].astype(str) == stock_code_str]
                stock_name = name_row['name'].iloc[0] if not name_row.empty else stock_code_str
            else:
                stock_name = stock_code_str
            
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            stock_df['year'] = stock_df['date'].dt.year
            
            years = sorted([y for y in stock_df['year'].unique() if start_year <= y <= end_year])
            
            if not years:
                continue
            
            # 연도별 백테스트
            for year in years:
                try:
                    year_data = stock_df[stock_df['year'] == year].copy()
                    
                    if year_data.empty or len(year_data) < 20:
                        continue
                    
                    start_date = pd.to_datetime(f"{year}-01-01")
                    end_date = pd.to_datetime(f"{year}-12-31")
                    
                    # 신호 생성
                    signals = build_signals_func(
                        year_data,
                        rolling_days,
                        volume_threshold,
                        20, 5.0,
                        20, 2.0,
                        20, 2.0,
                        ["Turnover Spike"],
                        "ANY",
                    )
                    
                    if signals.empty:
                        continue
                    
                    # 백테스트 실행
                    try:
                        equity_df, trades_df = backtest_func(
                            year_data,
                            signals,
                            kospi_index,
                            start_date,
                            top_n=2,
                            initial_cash=initial_cash,
                            max_daily_buys=max_daily_buys,
                            buy_unit=buy_unit,
                            add_buy_threshold_pct=loss_threshold_pct,
                        )
                    except Exception:
                        continue
                    
                    # 연도별 성과 지표 계산
                    metrics = calculate_yearly_metrics(
                        year,
                        equity_df,
                        trades_df,
                        stock_code_str,
                        stock_name
                    )
                    all_yearly_results.append(metrics)
                
                except Exception:
                    continue
            
            processed_stocks += 1
            
            # 중간 결과 표시
            if len(all_yearly_results) > 0 and processed_stocks % 10 == 0:
                sample_df = pd.DataFrame(all_yearly_results).tail(50)
                with results_placeholder.container():
                    st.dataframe(
                        sample_df,
                        use_container_width=True,
                        hide_index=True,
                    )
        
        except Exception as e:
            continue
    
    progress_placeholder.empty()
    
    # 디버깅 정보
    with debug_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 종목", total_stocks)
        with col2:
            st.metric("처리된 종목", processed_stocks)
        with col3:
            st.metric("총 연도별 결과", len(all_yearly_results))
    
    if all_yearly_results:
        results_df = pd.DataFrame(all_yearly_results)
        # 중복 제거: 같은 코드-연도면 total_return_pct가 높은 것만 유지
        results_df = results_df.sort_values('total_return_pct', ascending=False).drop_duplicates(subset=['code', 'year'], keep='first')
        return results_df.reset_index(drop=True)
    else:
        st.error("❌ 분석할 수 있는 데이터가 없습니다.")
        return pd.DataFrame()


def summarize_all_stocks_consistency(yearly_batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 종목의 연도별 성과에서 일관성 요약 정보 추출
    
    Args:
        yearly_batch_df: 모든 종목의 연도별 성과 데이터프레임 (run_yearly_backtest_batch 결과)
    
    Returns:
        종목별 일관성 요약 데이터프레임
        - avg_return_pct: 초기 자본금 대비 연평균 수익률
        - std_return_pct: 연 수익률의 표준편차 (값이 작을수록 일관성 높음)
        - success_rate: 수익을 본 연도의 비율
        - avg_win_rate: 개별 거래의 평균 승률 (매수가격 대비)
    """
    if yearly_batch_df.empty:
        return pd.DataFrame()
    
    # 종목별로 그룹화
    consistency_list = []
    
    for code in yearly_batch_df['code'].unique():
        stock_df = yearly_batch_df[yearly_batch_df['code'] == code]
        
        if stock_df.empty:
            continue
        
        stock_name = stock_df['name'].iloc[0] if not stock_df.empty else code
        
        consistency = {
            'code': code,
            'name': stock_name,
            'test_years': len(stock_df),  # 테스트한 연도 수
            'positive_years': len(stock_df[stock_df['total_return_pct'] > 0]),  # 수익 연도
            'success_rate': (len(stock_df[stock_df['total_return_pct'] > 0]) / len(stock_df) * 100) if len(stock_df) > 0 else 0,
            'avg_return_pct': stock_df['total_return_pct'].mean(),
            'std_return_pct': stock_df['total_return_pct'].std(),
            'min_return_pct': stock_df['total_return_pct'].min(),
            'max_return_pct': stock_df['total_return_pct'].max(),
            'avg_win_rate': stock_df['win_rate'].mean(),
            'std_win_rate': stock_df['win_rate'].std(),
            'avg_profit_loss_ratio': stock_df['profit_loss_ratio'].mean(),
        }
        
        consistency_list.append(consistency)
    
    return pd.DataFrame(consistency_list).sort_values('avg_return_pct', ascending=False)

