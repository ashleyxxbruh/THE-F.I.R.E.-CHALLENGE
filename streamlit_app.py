"""
Run:
  export API_BASE_URL=http://localhost:8000
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="FIRE Routing Dashboard", layout="wide")


@st.cache_data(ttl=30)
def api_get(path: str, params: Optional[dict[str, Any]] = None) -> Any:
    url = f"{API_BASE_URL}{path}"
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def show_metric_card(label: str, value: Any) -> None:
    st.metric(label=label, value=value if value is not None else "—")


def to_chart_frame(items: list[dict[str, Any]], label_col: str = "key", value_col: str = "count") -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=[label_col, value_col])
    return pd.DataFrame(items)


def dashboard_page() -> None:
    st.title("FIRE Dashboard")
    st.caption("Routing overview from PostgreSQL via FastAPI")

    try:
        stats = api_get("/stats")
    except Exception as exc:
        st.error(f"Failed to load /stats: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric_card("Total Tickets", stats.get("total_tickets"))
    with c2:
        show_metric_card("Assigned", stats.get("total_assigned"))
    with c3:
        show_metric_card("Unassigned", stats.get("total_unassigned"))
    with c4:
        avg_priority = stats.get("avg_priority")
        show_metric_card("Avg Priority", f"{avg_priority:.2f}" if avg_priority is not None else "—")

    chart_specs = [
        ("By Office", "counts_by_office"),
        ("By Ticket Type", "counts_by_ticket_type"),
        ("By Sentiment", "counts_by_sentiment"),
        ("By Language", "counts_by_language"),
    ]

    cols = st.columns(2)
    for idx, (title, key) in enumerate(chart_specs):
        with cols[idx % 2]:
            st.subheader(title)
            frame = to_chart_frame(stats.get(key, []))
            if frame.empty:
                st.info("Not available")
            else:
                st.bar_chart(frame.set_index("key")["count"])

    st.subheader("Top Cities")
    cities_frame = to_chart_frame(stats.get("counts_by_city", []))
    if cities_frame.empty:
        st.info("Not available")
    else:
        st.dataframe(cities_frame, use_container_width=True, hide_index=True)


def tickets_page() -> None:
    st.title("Tickets")
    st.caption("Filter and inspect ticket routing with explainability")

    try:
        offices = api_get("/offices")
        stats = api_get("/stats")
    except Exception as exc:
        st.error(f"Failed to load filter data: {exc}")
        return

    office_options = ["All"] + [item["name"] for item in offices]
    type_options = ["All"] + [item["key"] for item in stats.get("counts_by_ticket_type", [])]
    sentiment_options = ["All"] + [item["key"] for item in stats.get("counts_by_sentiment", [])]
    language_options = ["All"] + [item["key"] for item in stats.get("counts_by_language", [])]

    with st.sidebar:
        st.header("Ticket Filters")
        office = st.selectbox("Office", office_options, index=0)
        status = st.selectbox("Status", ["All", "ASSIGNED", "UNASSIGNED"], index=0)
        segment = st.text_input("Segment")
        ticket_type = st.selectbox("Ticket Type", type_options, index=0)
        sentiment = st.selectbox("Sentiment", sentiment_options, index=0)
        language = st.selectbox("Language", language_options, index=0)
        priority_range = st.slider("Priority", min_value=1, max_value=10, value=(1, 10))
        limit = st.number_input("Limit", min_value=1, max_value=500, value=100, step=10)

    params: dict[str, Any] = {"limit": int(limit), "offset": 0}
    if office != "All":
        params["office"] = office
    if status != "All":
        params["status"] = status
    if segment.strip():
        params["segment"] = segment.strip()
    if ticket_type != "All":
        params["ticket_type"] = ticket_type
    if sentiment != "All":
        params["sentiment"] = sentiment
    if language != "All":
        params["language"] = language
    params["priority_min"] = priority_range[0]
    params["priority_max"] = priority_range[1]

    try:
        tickets = api_get("/tickets", params=params)
    except Exception as exc:
        st.error(f"Failed to load /tickets: {exc}")
        return

    if not tickets:
        st.info("No tickets found for selected filters.")
        return

    rows: list[dict[str, Any]] = []
    for item in tickets:
        rows.append(
            {
                "ticket_id": item["ticket_id"],
                "client_guid": item["client_guid"],
                "segment": item.get("segment") or "—",
                "city": item.get("city") or "—",
                "ticket_type": (item.get("ai") or {}).get("ticket_type") or "—",
                "sentiment": (item.get("ai") or {}).get("sentiment") or "—",
                "priority": (item.get("ai") or {}).get("priority"),
                "language": (item.get("ai") or {}).get("language") or "—",
                "status": (item.get("assignment") or {}).get("status") or "—",
                "office": (item.get("assignment") or {}).get("office_name") or "—",
                "manager": (item.get("assignment") or {}).get("manager_name") or "—",
                "assigned_at": item.get("assigned_at") or "—",
                "description": item.get("description") or "—",
                "view": f"View #{item['ticket_id']}",
            }
        )

    table = pd.DataFrame(rows)
    st.subheader("Ticket List")
    st.dataframe(table, use_container_width=True, hide_index=True)

    ticket_ids = [item["ticket_id"] for item in tickets]
    selected_ticket_id = st.selectbox("Select ticket for detail", ticket_ids, index=0)

    try:
        detail = api_get(f"/tickets/{selected_ticket_id}")
    except Exception as exc:
        st.error(f"Failed to load /tickets/{selected_ticket_id}: {exc}")
        return

    st.subheader(f"Ticket Detail #{selected_ticket_id}")
    ticket_data = detail.get("ticket", {})
    ai_data = detail.get("ai_analysis") or {}
    assignment = detail.get("assignment") or {}

    left, right = st.columns(2)
    with left:
        st.markdown("**Original Description**")
        st.write(ticket_data.get("description") or "Not available")
        st.markdown("**Client / Location**")
        st.write(
            f"GUID: {ticket_data.get('client_guid') or '—'}  \n"
            f"Segment: {ticket_data.get('segment') or '—'}  \n"
            f"City: {ticket_data.get('city') or '—'}  \n"
            f"Country: {ticket_data.get('country') or '—'}"
        )
    with right:
        st.markdown("**AI Summary**")
        st.write(ai_data.get("summary") or "Not available")
        st.markdown("**AI Recommendation**")
        st.write(ai_data.get("recommendation") or "Not available")
        st.markdown("**AI Fields**")
        st.write(
            f"Type: {ai_data.get('ticket_type') or '—'}  \n"
            f"Sentiment: {ai_data.get('sentiment') or '—'}  \n"
            f"Priority: {ai_data.get('priority') if ai_data.get('priority') is not None else '—'}  \n"
            f"Language: {ai_data.get('language') or '—'}"
        )

    st.markdown("**Assignment**")
    st.write(
        f"Status: {assignment.get('status') or 'Not available'}  \n"
        f"Office: {assignment.get('office_name') or 'Not available'}  \n"
        f"Manager: {assignment.get('manager_name') or 'Not available'}  \n"
        f"Assigned At: {assignment.get('assigned_at') or 'Not available'}"
    )

    st.markdown("**Explainability**")
    reason = assignment.get("reason")
    if reason:
        st.json(reason)
    else:
        st.json({"message": "Not available"})


def managers_page() -> None:
    st.title("Managers")
    st.caption("Current loads and assignment counts")

    try:
        managers = api_get("/managers")
    except Exception as exc:
        st.error(f"Failed to load /managers: {exc}")
        return

    if not managers:
        st.info("No manager data available.")
        return

    manager_df = pd.DataFrame(managers)
    manager_df["skills"] = manager_df["skills"].apply(
        lambda values: ", ".join(values) if isinstance(values, list) and values else "—"
    )

    offices = ["All"] + sorted(manager_df["office_name"].dropna().unique().tolist())
    col1, col2 = st.columns([2, 1])
    with col1:
        office_filter = st.selectbox("Office", offices, index=0)
    with col2:
        vip_only = st.checkbox("VIP skill only", value=False)

    filtered = manager_df.copy()
    if office_filter != "All":
        filtered = filtered[filtered["office_name"] == office_filter]
    if vip_only:
        filtered = filtered[filtered["skills"].str.contains("VIP", case=False, na=False)]

    filtered = filtered.sort_values(by=["current_load", "manager_id"], ascending=[False, True])
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if not filtered.empty:
        st.subheader("Load by Manager")
        chart_df = filtered[["full_name", "current_load"]].set_index("full_name")
        st.bar_chart(chart_df["current_load"])


def main() -> None:
    st.sidebar.title("FIRE Navigation")
    st.sidebar.caption(f"API: {API_BASE_URL}")
    page = st.sidebar.radio("Page", ["Dashboard", "Tickets", "Managers"], index=0)

    if page == "Dashboard":
        dashboard_page()
    elif page == "Tickets":
        tickets_page()
    else:
        managers_page()


if __name__ == "__main__":
    main()
