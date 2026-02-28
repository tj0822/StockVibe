import streamlit as st
from datetime import datetime


def _render_badge(label: str, badge_type: str = "info"):
    color_map = {
        "success": "#16a34a",
        "warning": "#d97706",
        "error": "#dc2626",
        "info": "#2563eb",
    }
    color = color_map.get(badge_type, color_map["info"])
    st.markdown(
        f"""
        <span style=\"display:inline-block;padding:2px 10px;border-radius:999px;
        background:{color}22;color:{color};border:1px solid {color}66;font-size:12px;\">{label}</span>
        """,
        unsafe_allow_html=True,
    )


def render_header(page_name: str, badges: list[dict]):
    st.markdown(f"### 📊 StockVibe Pro | {page_name}")
    if badges:
        cols = st.columns(len(badges))
        for col, badge in zip(cols, badges):
            with col:
                _render_badge(badge.get("label", "-"), badge.get("type", "info"))
    st.markdown("---")


def render_section_header(title: str, help_text: str = None):
    header_cols = st.columns([8, 1])
    with header_cols[0]:
        st.subheader(title)
    with header_cols[1]:
        if help_text:
            st.caption(help_text)


def render_kpi_row(kpis: list[dict], clickable: bool = False, key_prefix: str = ""):
    if not kpis:
        return

    clicked_key = None
    columns = st.columns(len(kpis))
    for idx, (col, item) in enumerate(zip(columns, kpis)):
        label = item.get("label", "")
        value = item.get("value", "-")
        delta = item.get("delta")
        delta_color = item.get("delta_color", "normal")
        help_text = item.get("help")
        item_key = item.get("key", f"kpi_{idx}")

        with col:
            if clickable:
                if st.button(f"{label}\n{value}", key=f"{key_prefix}{item_key}", use_container_width=True, help=help_text):
                    clicked_key = item_key
                if delta:
                    st.caption(delta)
            else:
                if delta is None:
                    st.metric(label=label, value=value, help=help_text)
                else:
                    st.metric(label=label, value=value, delta=delta, delta_color=delta_color, help=help_text)

    return clicked_key


def render_filters_row(filters: list[dict], key_prefix: str = ""):
    if not filters:
        return {}

    values = {}
    cols = st.columns(len(filters))
    for idx, (col, item) in enumerate(zip(cols, filters)):
        widget_type = item.get("type", "selectbox")
        label = item.get("label", "필터")
        options = item.get("options", [])
        default = item.get("value")
        key = item.get("key", f"filter_{idx}")
        full_key = f"{key_prefix}{key}"

        with col:
            if widget_type == "multiselect":
                values[key] = st.multiselect(label, options=options, default=default or [], key=full_key)
            elif widget_type == "date_input":
                values[key] = st.date_input(label, value=default, key=full_key)
            else:
                index = options.index(default) if (default in options) else 0
                values[key] = st.selectbox(label, options=options, index=index, key=full_key)

    return values


def render_action_row(actions: list[dict], key_prefix: str = ""):
    if not actions:
        return {}

    results = {}
    cols = st.columns(len(actions))
    for idx, (col, action) in enumerate(zip(cols, actions)):
        key = action.get("key", f"action_{idx}")
        label = action.get("label", "")
        kind = action.get("kind", "secondary")
        help_text = action.get("help")
        disabled = action.get("disabled", False)
        on_click = action.get("on_click")
        full_key = f"{key_prefix}{key}"

        with col:
            clicked = st.button(
                label,
                key=full_key,
                type=kind,
                use_container_width=True,
                help=help_text,
                disabled=disabled,
            )
        if clicked and callable(on_click):
            on_click()
        results[key] = clicked

    return results


def render_status_badge(label: str, type: str):
    _render_badge(label, type)


def render_progress_block(step: str, percent: float):
    percent = max(0.0, min(100.0, float(percent)))
    st.caption(step)
    st.progress(int(percent))


def render_progress_steps(steps: list[str], current_step_index: int, progress: float):
    safe_index = max(0, min(current_step_index, len(steps) - 1)) if steps else 0
    label = steps[safe_index] if steps else "진행 중"
    st.caption(f"[{safe_index + 1}/{len(steps)}] {label}" if steps else label)
    st.progress(max(0.0, min(1.0, float(progress))))


def render_global_status_line():
    data_refresh_time = st.session_state.get("data_refresh_time", "-")
    cache_status = st.session_state.get("ui_cache_status", "UNKNOWN")
    last_success_time = st.session_state.get("ui_last_success_time", "-")

    if cache_status == "HIT":
        cache_type = "success"
    elif cache_status == "MISS":
        cache_type = "warning"
    elif cache_status == "ERROR":
        cache_type = "error"
    else:
        cache_type = "info"

    status_cols = st.columns([3, 2, 3])
    with status_cols[0]:
        render_status_badge(f"🕒 데이터 최신 시각: {data_refresh_time}", "info")
    with status_cols[1]:
        render_status_badge(f"캐시 상태: {cache_status}", cache_type)
    with status_cols[2]:
        render_status_badge(f"마지막 성공 실행: {last_success_time}", "success")


def mark_ui_success(cache_status: str = None):
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["ui_last_success_time"] = now_ts
    st.session_state["last_run_ts"] = now_ts
    if cache_status:
        st.session_state["ui_cache_status"] = cache_status
        st.session_state["ai_cache_hit"] = cache_status
