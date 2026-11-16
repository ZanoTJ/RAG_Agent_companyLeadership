"""Streamlit UI: collect a company's leadership, then chat about it."""

import pandas as pd
import streamlit as st

import database as db
import entities
import rag
import views

st.set_page_config(page_title="Company Leadership RAG Agent", page_icon="🏢", layout="wide")
db.init_db()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Collect leadership")
    url = st.text_input("Company website", placeholder="https://www.example.com")
    if st.button("Collect", type="primary", disabled=not url):
        bar = st.progress(0.0, text="Collecting leadership data...")
        steps = [0]

        def advance(_msg):
            # The total step count isn't known up front, so ease toward 95%.
            steps[0] += 1
            bar.progress(min(0.95, 1 - 0.93 ** steps[0]), text="Collecting leadership data...")

        try:
            cid = rag.ingest_company(url, log=advance)
            st.session_state.company_id = cid
            st.session_state.company_select = cid  # widget is created below
            st.session_state.messages = []
            bar.progress(1.0, text="Collection complete")
        except Exception as e:
            bar.empty()
            st.error(f"Collection failed: {e}")

    st.header("2. Pick a company")
    companies = db.list_companies()
    if companies:
        ids = [c["id"] for c in companies]
        # A stable key keeps the selection across reruns (no index recomputation).
        if st.session_state.get("company_select") not in ids:
            current = st.session_state.get("company_id")
            st.session_state.company_select = current if current in ids else ids[0]
        names = {c["id"]: f"{c['name']} ({c['leader_count']} people)" for c in companies}
        chosen = st.selectbox("Company", ids, key="company_select", format_func=names.get)
        if chosen != st.session_state.get("company_id"):
            st.session_state.company_id = chosen
            st.session_state.messages = []
    else:
        st.caption("No companies collected yet.")

# ---------------------------------------------------------------- main
st.title("🏢 Company Leadership RAG Agent")
company_id = st.session_state.get("company_id")
if not company_id:
    st.info("Enter a company website in the sidebar to get started.")
    st.stop()

leaders = db.get_leaders(company_id)
company = db.get_company(company_id)
with st.expander(f"Leadership ({len(leaders)} people)", expanded=not leaders or len(leaders) < 15):
    st.caption("Collected from public sources (company website, filings, news, public "
               "profiles). The list may not include every leader.")
    if company.get("removed_unverified"):
        st.info(f"{company['removed_unverified']} previously listed people were removed "
                "because their only evidence came from unverified sources. Collect the "
                "company again to look for official sources.")
    if leaders:
        st.dataframe(
            pd.DataFrame(views.people_rows(leaders), columns=views.PEOPLE_COLUMNS),
            column_config={
                "Source": st.column_config.LinkColumn("Source"),
                "Profile": st.column_config.LinkColumn("Profile"),
                "More sources": st.column_config.ListColumn("More sources"),
            },
            hide_index=True,
            width="stretch",
        )
        for note in views.disagreements(leaders):
            st.warning(f"Unresolved: {note}")
        roles = db.get_role_history(company_id)
    else:
        st.warning(
            "No leaders found. The site may render its team page with JavaScript, "
            "block automated requests, or not publish leadership publicly."
        )

st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about this company's leadership, e.g. 'Who runs engineering?'"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = rag.answer(company_id, question, st.session_state.messages)
            except Exception as e:
                reply = f"Error calling Gemini: {e}"
        st.markdown(reply)
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ]
