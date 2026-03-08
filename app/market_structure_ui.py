from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data import load_kospi_index, load_marketcap_history, load_stock_data
from app.data_pipeline import build_stock_master_df
from app.market_regime import (
    render_regime_summary,
    run_market_regime_engine,
    save_market_regime_snapshot,
)
from app.market_structure import (
    build_market_structure_watchlists,
    build_topn_sector_composition,
    get_current_market_snapshot,
    summarize_sector_rank_changes,
)
from app.market_structure_insights import (
    generate_market_structure_insights,
    render_market_insight_cards,
)
from app.marketcap_bump import (
    build_marketcap_rank_history,
    render_marketcap_bump_chart,
    summarize_rank_changes,
)
from app.signals import build_signals


@st.cache_data(ttl=3600, show_spinner=False)
def _load_stock_master_cached(data_dir: str = "data") -> pd.DataFrame:
    return build_stock_master_df(data_dir)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_price_and_sector_cached(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    price_df = load_stock_data(data_dir)
    stock_master = build_stock_master_df(data_dir)
    sector_map = stock_master[["code", "name", "sector"]].copy() if not stock_master.empty else pd.DataFrame(columns=["code", "name", "sector"])
    if not sector_map.empty:
        sector_map["code"] = sector_map["code"].astype(str).str.zfill(6)
        sector_map = sector_map.drop_duplicates(subset=["code"], keep="last")
    return price_df, sector_map


@st.cache_data(ttl=3600, show_spinner=False)
def _load_sector_rank_history_cached(data_dir: str = "data") -> pd.DataFrame:
    price_df, sector_map = _load_price_and_sector_cached(data_dir)
    if price_df is None or price_df.empty or sector_map.empty:
        return pd.DataFrame(columns=["date", "sector", "rank", "score", "turnover", "buy_count", "avg_return_1m", "sector_power"])

    px = price_df.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    if "volume" in px.columns:
        px["volume"] = pd.to_numeric(px["volume"], errors="coerce").fillna(0.0)
    else:
        px["volume"] = 1.0
    px = px.dropna(subset=["date", "code", "close"]).copy()
    if px.empty:
        return pd.DataFrame(columns=["date", "sector", "rank", "score", "turnover", "buy_count", "avg_return_1m", "sector_power"])

    px = px.merge(sector_map[["code", "sector"]], on="code", how="left")
    px = px.dropna(subset=["sector"]).copy()
    px["turnover"] = px["close"] * px["volume"]
    px["week"] = px["date"].dt.to_period("W")

    weekly_turnover = (
        px.groupby(["week", "sector"], as_index=False)
        .agg(turnover=("turnover", "sum"))
    )

    # BUY signal count (weekly)
    signal_input = price_df.copy()
    signal_input["date"] = pd.to_datetime(signal_input["date"], errors="coerce")
    signal_input["code"] = signal_input["code"].astype(str).str.zfill(6)
    signal_input["close"] = pd.to_numeric(signal_input.get("close"), errors="coerce")
    signal_input["volume"] = pd.to_numeric(signal_input.get("volume"), errors="coerce").fillna(0.0)
    if "open" in signal_input.columns:
        signal_input["open"] = pd.to_numeric(signal_input["open"], errors="coerce")
    else:
        # Fallback for legacy/partial datasets: candle classification uses open.
        signal_input["open"] = signal_input["close"]

    signal_input = signal_input.dropna(subset=["date", "code", "close", "open"]).copy()

    sig = build_signals(
        signal_input[["date", "code", "open", "close", "volume"]].copy(),
        10,
        3.0,
        20,
        5.0,
        20,
        2.0,
        20,
        2.0,
        ["Turnover Spike"],
        "ANY",
    )
    if sig is None or sig.empty:
        weekly_buy = pd.DataFrame(columns=["week", "sector", "buy_count"])
    else:
        sig = sig.copy()
        sig["date"] = pd.to_datetime(sig["date"], errors="coerce")
        sig["code"] = sig["code"].astype(str).str.zfill(6)
        sig = sig.merge(sector_map[["code", "sector"]], on="code", how="left")
        sig = sig[sig["signal"] == "BUY"]
        sig["week"] = sig["date"].dt.to_period("W")
        weekly_buy = sig.groupby(["week", "sector"], as_index=False).size().rename(columns={"size": "buy_count"})

    # weekly avg 1M return proxy from price history
    close_sorted = px.sort_values(["code", "date"]).copy()
    close_sorted["ret_20d"] = close_sorted.groupby("code")["close"].pct_change(20) * 100.0
    weekly_ret = (
        close_sorted.groupby(["week", "sector"], as_index=False)
        .agg(avg_return_1m=("ret_20d", "mean"))
    )

    stock_master = _load_stock_master_cached(data_dir)
    if stock_master is None or stock_master.empty:
        sector_power = pd.DataFrame(columns=["sector", "sector_power"])
    else:
        sector_power = (
            stock_master[["sector", "sector_power"]]
            .dropna(subset=["sector"]) 
            .drop_duplicates(subset=["sector"], keep="last")
        )

    wk = weekly_turnover.merge(weekly_buy, on=["week", "sector"], how="left")
    wk = wk.merge(weekly_ret, on=["week", "sector"], how="left")
    wk = wk.merge(sector_power, on="sector", how="left")
    wk["buy_count"] = pd.to_numeric(wk["buy_count"], errors="coerce").fillna(0.0)
    wk["avg_return_1m"] = pd.to_numeric(wk["avg_return_1m"], errors="coerce").fillna(0.0)
    wk["sector_power"] = pd.to_numeric(wk["sector_power"], errors="coerce").fillna(50.0)

    # Composite score for ranking (lightweight weighted blend)
    wk["score"] = (
        (wk["turnover"].rank(pct=True) * 0.45)
        + (wk["buy_count"].rank(pct=True) * 0.20)
        + (wk["avg_return_1m"].rank(pct=True) * 0.20)
        + (wk["sector_power"].rank(pct=True) * 0.15)
    )

    wk["date"] = wk["week"].dt.to_timestamp(how="end").dt.normalize()
    wk["rank"] = (
        wk.groupby("date")["score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    return wk[["date", "sector", "rank", "score", "turnover", "buy_count", "avg_return_1m", "sector_power"]].sort_values(["date", "rank"])


@st.cache_data(ttl=3600, show_spinner=False)
def _load_marketcap_rank_history_cached(data_dir: str, freq: str, top_n: int) -> pd.DataFrame:
    stock_master = _load_stock_master_cached(data_dir)
    sector_map = stock_master[["code", "sector"]].copy() if not stock_master.empty else pd.DataFrame(columns=["code", "sector"])
    mc_df = load_marketcap_history(data_dir)
    return build_marketcap_rank_history(mc_df, sector_map, freq=freq, top_n=top_n)


@st.cache_data(ttl=3600, show_spinner=False)
def _build_snapshot_cached(stock_master_df: pd.DataFrame, marketcap_rank_df: pd.DataFrame) -> dict:
    return get_current_market_snapshot(stock_master_df, marketcap_rank_df)


@st.cache_data(ttl=3600, show_spinner=False)
def _build_watchlists_cached(stock_master_df: pd.DataFrame, sector_rank_df: pd.DataFrame) -> dict:
    return build_market_structure_watchlists(stock_master_df, sector_rank_df)


def _render_table(df: pd.DataFrame, cols: list[str], max_rows: int = 5, key: str = "") -> None:
    show = df.copy() if df is not None else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in show.columns:
            show[c] = pd.NA
    st.dataframe(show[cols].head(max_rows), hide_index=True, use_container_width=True)


def render_market_structure_page() -> None:
    data_dir = "data"

    st.title("📡 시장구조 대시보드")
    st.caption("섹터 로테이션, 시총 상위 종목 변화, 현재 시장 리더십을 통합해서 보여줍니다.")
    st.divider()

    stock_master_df = _load_stock_master_cached(data_dir)
    kospi_index_df = load_kospi_index(data_dir)
    sector_rank_df = _load_sector_rank_history_cached(data_dir)
    marketcap_rank_snapshot_df = _load_marketcap_rank_history_cached(data_dir, freq="W", top_n=10)

    if stock_master_df is None or stock_master_df.empty:
        st.warning("시장구조 대시보드를 위한 stock master 데이터가 없습니다.")
        return

    regime_result = run_market_regime_engine(
        stock_master_df=stock_master_df,
        kospi_index_df=kospi_index_df,
        sector_rank_df=sector_rank_df,
        marketcap_rank_df=marketcap_rank_snapshot_df,
    )
    save_market_regime_snapshot(regime_result)
    render_regime_summary(regime_result)
    st.divider()

    insights = generate_market_structure_insights(
        stock_master_df,
        sector_rank_df,
        marketcap_rank_snapshot_df,
    )
    render_market_insight_cards(insights)
    st.divider()

    # Block A: Header KPI
    snapshot = _build_snapshot_cached(stock_master_df, marketcap_rank_snapshot_df)

    # Optional compact strip
    sec_comp = build_topn_sector_composition(marketcap_rank_snapshot_df)
    concentration = 0.0
    breadth = 0
    if not sec_comp.empty:
        total_n = float(sec_comp["count_in_top_n"].sum())
        top3 = float(sec_comp.head(3)["count_in_top_n"].sum())
        concentration = (top3 / total_n * 100.0) if total_n > 0 else 0.0
        breadth = int(sec_comp["sector"].nunique())
    sector_delta_summary = summarize_sector_rank_changes(sector_rank_df)

    strip1, strip2, strip3 = st.columns(3)
    with strip1:
        st.metric("시장 집중도(Top3 섹터 비중)", f"{concentration:.1f}%")
    with strip2:
        st.metric("리더십 폭(Top10 내 섹터 수)", breadth)
    with strip3:
        st.metric("회전 강도(주간 평균 rank 절대변화)", f"{sector_delta_summary.get('rotation_intensity', 0.0):.2f}")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("현재 최강 섹터", snapshot.get("strongest_sector", "-"))
    with k2:
        st.metric("LEADING 섹터 수", int(snapshot.get("leading_sector_count", 0)))
    with k3:
        st.metric("시총 리더 최다 섹터", snapshot.get("most_represented_marketcap_sector", "-"))
    with k4:
        st.metric("LEADING 섹터 BUY 신호 수", int(snapshot.get("buy_count_leading_sectors", 0)))

    st.divider()

    # Block B: Snapshot 3 columns
    st.markdown("#### 현재 시장 리더십 스냅샷")
    c1, c2, c3 = st.columns(3)

    sector_latest = (
        stock_master_df[["sector", "sector_power", "sector_rank", "sector_state"]]
        .dropna(subset=["sector"])
        .drop_duplicates(subset=["sector"], keep="last")
        .sort_values(["sector_rank", "sector_power"], ascending=[True, False])
    )

    mc_latest = marketcap_rank_snapshot_df.copy()
    if not mc_latest.empty:
        mc_latest["date"] = pd.to_datetime(mc_latest["date"], errors="coerce")
        mc_latest = mc_latest[mc_latest["date"] == mc_latest["date"].max()].sort_values("rank")

    with c1:
        st.markdown("**현재 강한 섹터 Top 5**")
        _render_table(sector_latest, ["sector", "sector_power", "sector_rank", "sector_state"], max_rows=5, key="ms_sector_top5")

    with c2:
        st.markdown("**시총 상위 리더 Top 5**")
        _render_table(mc_latest, ["rank", "name", "sector", "market_cap"], max_rows=5, key="ms_mc_top5")

    with c3:
        st.markdown("**신규 강세/약세 요약**")
        rank_change = summarize_sector_rank_changes(sector_rank_df)
        risers = rank_change.get("risers", [])
        fallers = rank_change.get("fallers", [])
        mc_change = summarize_rank_changes(marketcap_rank_snapshot_df)

        top_risers = ", ".join([f"{r.get('sector')}({int(r.get('rank_change'))})" for r in risers[:3]]) if risers else "-"
        top_fallers = ", ".join([f"{r.get('sector')}(+{int(r.get('rank_change'))})" for r in fallers[:3]]) if fallers else "-"
        new_entries = ", ".join([f"{r.get('name')}" for r in mc_change.get("new_entries", [])[:3]]) or "-"
        exits = ", ".join([f"{r.get('name')}" for r in mc_change.get("exits", [])[:3]]) or "-"

        st.markdown(f"- 급상승 섹터: {top_risers}")
        st.markdown(f"- 약화 섹터: {top_fallers}")
        st.markdown(f"- 신규 시총 진입: {new_entries}")
        st.markdown(f"- 최근 이탈: {exits}")

    st.divider()

    # Block C: Weekly sector bump chart
    st.markdown("#### 🏭 주간 섹터 수급 랭킹 변화")
    c_ctrl1, c_ctrl2 = st.columns(2)
    with c_ctrl1:
        period_weeks = st.selectbox("조회 기간", options=[8, 12, 20], index=1, format_func=lambda x: f"{x}주", key="ms_period_weeks")
    with c_ctrl2:
        top_n_view = int(
            st.slider(
                "Top N",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                key="ms_sector_view_top_n",
            )
        )

    sector_plot_df = sector_rank_df.copy()
    if not sector_plot_df.empty:
        sector_plot_df["date"] = pd.to_datetime(sector_plot_df["date"], errors="coerce")
        dates = sorted(sector_plot_df["date"].dropna().unique().tolist())
        if len(dates) > period_weeks:
            cutoff = dates[-period_weeks]
            sector_plot_df = sector_plot_df[sector_plot_df["date"] >= cutoff].copy()

        latest_date = sector_plot_df["date"].max()
        latest_rank = sector_plot_df[sector_plot_df["date"] == latest_date][["sector", "rank"]].copy()
        selected_sectors = latest_rank.sort_values("rank").head(top_n_view)["sector"].tolist()
        sector_plot_df = sector_plot_df[sector_plot_df["sector"].isin(selected_sectors)]

    if sector_plot_df.empty:
        st.info("주간 섹터 랭킹 데이터가 부족합니다.")
    else:
        fig_sector = go.Figure()
        all_dates = sorted(sector_plot_df["date"].dropna().unique().tolist())

        for sector_name, gdf in sector_plot_df.groupby("sector"):
            g = gdf.sort_values("date").set_index("date").reindex(all_dates)
            fig_sector.add_trace(
                go.Scatter(
                    x=all_dates,
                    y=g["rank"],
                    mode="lines+markers",
                    name=str(sector_name),
                    marker=dict(size=6),
                    line=dict(width=1.8),
                    opacity=0.8,
                    connectgaps=False,
                )
            )

        max_rank = int(pd.to_numeric(sector_plot_df["rank"], errors="coerce").max())
        fig_sector.update_layout(
            template="plotly_dark",
            height=560,
            xaxis_title="주간 기준일",
            yaxis_title="랭킹 (1=상위)",
            legend_title="섹터",
            legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
            margin=dict(l=10, r=140, t=20, b=10),
        )
        fig_sector.update_yaxes(autorange="reversed", tickmode="array", tickvals=list(range(1, max_rank + 1)))
        st.plotly_chart(fig_sector, use_container_width=True)

    delta_info = summarize_sector_rank_changes(sector_plot_df if not sector_plot_df.empty else sector_rank_df)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**이번 주 급상승 섹터 Top 5**")
        risers_df = pd.DataFrame(delta_info.get("risers", []))
        if risers_df.empty:
            st.caption("없음")
        else:
            st.dataframe(risers_df[["sector", "prev_rank", "latest_rank", "rank_change"]], hide_index=True, use_container_width=True)
    with d2:
        st.markdown("**이번 주 약화 섹터 Top 5**")
        fallers_df = pd.DataFrame(delta_info.get("fallers", []))
        if fallers_df.empty:
            st.caption("없음")
        else:
            st.dataframe(fallers_df[["sector", "prev_rank", "latest_rank", "rank_change"]], hide_index=True, use_container_width=True)

    st.divider()

    # Block D: Market-cap Top N bump chart
    st.markdown("#### 📈 시총 상위종목 변화")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        freq_label = st.selectbox("집계 주기", options=["주간", "월간"], index=0, key="ms_mc_freq")
    with mc2:
        top_n = int(st.selectbox("Top N", options=[10, 20, 30], index=0, key="ms_mc_topn"))
    with mc3:
        color_label = st.selectbox("색상 모드", options=["종목별", "섹터별"], index=0, key="ms_mc_color")

    freq_code = "W" if freq_label == "주간" else "M"
    color_mode = "stock" if color_label == "종목별" else "sector"

    rank_history_df = _load_marketcap_rank_history_cached(data_dir, freq=freq_code, top_n=top_n)
    if rank_history_df.empty:
        st.info("시총 Top N 랭킹 데이터를 생성할 수 없습니다.")
    else:
        st.plotly_chart(render_marketcap_bump_chart(rank_history_df, color_mode=color_mode), use_container_width=True)
        mc_summary = summarize_rank_changes(rank_history_df)

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**신규 진입 종목**")
            new_entries_df = pd.DataFrame(mc_summary.get("new_entries", []))
            if new_entries_df.empty:
                st.caption("없음")
            else:
                st.dataframe(new_entries_df, hide_index=True, use_container_width=True)
        with e2:
            st.markdown("**이탈 종목**")
            exits_df = pd.DataFrame(mc_summary.get("exits", []))
            if exits_df.empty:
                st.caption("없음")
            else:
                st.dataframe(exits_df, hide_index=True, use_container_width=True)

        st.markdown("**Top N 내 섹터 구성 요약**")
        comp_df = build_topn_sector_composition(rank_history_df)
        if comp_df.empty:
            st.caption("없음")
        else:
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.divider()

    # Block E: Sector Power vs Stock Quality Matrix
    st.markdown("#### 🧭 섹터 바람 vs 종목 체질")
    m1, m2 = st.columns(2)
    with m1:
        state_filter = st.selectbox(
            "섹터 상태 필터",
            options=["ALL", "LEADING", "STRONG", "ROTATION", "WEAK"],
            index=0,
            key="ms_state_filter",
        )
    with m2:
        signal_filter = st.selectbox("시그널 필터", options=["ALL", "BUY", "SELL", "NONE"], index=0, key="ms_signal_filter")

    matrix_df = stock_master_df.copy()
    for col in ["final_score_adjusted", "sector_power", "return_1m"]:
        matrix_df[col] = pd.to_numeric(matrix_df.get(col), errors="coerce")
    matrix_df["signal"] = matrix_df.get("signal").fillna("NONE")

    if state_filter != "ALL":
        matrix_df = matrix_df[matrix_df.get("sector_state") == state_filter]
    if signal_filter != "ALL":
        matrix_df = matrix_df[matrix_df["signal"] == signal_filter]

    matrix_df = matrix_df.dropna(subset=["final_score_adjusted", "sector_power"])

    if matrix_df.empty:
        st.info("선택한 조건의 매트릭스 데이터가 없습니다.")
    else:
        abs_ret = matrix_df["return_1m"].abs().fillna(0.0)
        if abs_ret.max() > 0:
            size = 8 + (abs_ret / abs_ret.max()) * 20
        else:
            size = pd.Series([10.0] * len(matrix_df), index=matrix_df.index)

        fig_matrix = go.Figure()
        for sector_name, gdf in matrix_df.groupby("sector"):
            fig_matrix.add_trace(
                go.Scatter(
                    x=gdf["final_score_adjusted"],
                    y=gdf["sector_power"],
                    mode="markers",
                    name=str(sector_name),
                    marker=dict(size=size.loc[gdf.index], opacity=0.78),
                    customdata=gdf[["name", "code", "signal", "return_1m"]].to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "코드: %{customdata[1]}<br>"
                        "Signal: %{customdata[2]}<br>"
                        "1M 수익률: %{customdata[3]:.2f}%<br>"
                        "조정 점수: %{x:.1f}<br>"
                        "섹터 파워: %{y:.1f}<extra></extra>"
                    ),
                )
            )

        fig_matrix.add_vline(x=70, line_width=1, line_dash="dash", line_color="#AAAAAA")
        fig_matrix.add_hline(y=70, line_width=1, line_dash="dash", line_color="#AAAAAA")
        fig_matrix.add_annotation(x=85, y=88, text="우선 관심", showarrow=False)
        fig_matrix.add_annotation(x=40, y=88, text="섹터 강세 / 종목 선별 필요", showarrow=False)
        fig_matrix.add_annotation(x=85, y=35, text="종목 우수 / 섹터 약세", showarrow=False)
        fig_matrix.add_annotation(x=40, y=35, text="회피 구간", showarrow=False)

        fig_matrix.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="final_score_adjusted",
            yaxis_title="sector_power",
            legend_title="섹터",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

    st.divider()

    # Block F: Actionable Watchlist Suggestions
    st.markdown("#### 🎯 시장구조 기반 관심 후보")
    watch = _build_watchlists_cached(stock_master_df, sector_rank_df)

    wl_cols = st.columns(3)
    labels = [
        ("섹터 주도주 후보", "leaders_df"),
        ("섹터 상승 전환 후보", "rising_df"),
        ("경고 후보", "warning_df"),
    ]

    combined_pick = []
    for col, (title, key) in zip(wl_cols, labels):
        with col:
            st.markdown(f"**{title}**")
            df = watch.get(key, pd.DataFrame())
            show_cols = [c for c in ["code", "name", "sector", "final_score_adjusted", "signal"] if c in df.columns]
            if df is None or df.empty or not show_cols:
                st.caption("없음")
            else:
                st.dataframe(df[show_cols].head(5), hide_index=True, use_container_width=True)
                combined_pick.extend(df["code"].head(5).astype(str).tolist())

    action1, action2 = st.columns(2)
    with action1:
        if st.button("시뮬레이션으로 보내기", use_container_width=True, key="ms_send_to_sim"):
            st.session_state["preselected_stocks"] = sorted(list(dict.fromkeys(combined_pick)))
            st.session_state["active_tab"] = "🎯 시뮬레이션"
            st.success("관심 종목을 session_state['preselected_stocks']에 저장했습니다.")

    with action2:
        export_frames = []
        for key, section in [("leaders_df", "leaders"), ("rising_df", "rising"), ("warning_df", "warning")]:
            df = watch.get(key, pd.DataFrame()).copy()
            if not df.empty:
                df["bucket"] = section
                export_frames.append(df)
        if export_frames:
            export_df = pd.concat(export_frames, ignore_index=True)
            st.download_button(
                label="CSV download",
                data=export_df.to_csv(index=False),
                file_name="market_structure_watchlist.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("CSV download", disabled=True, use_container_width=True)
