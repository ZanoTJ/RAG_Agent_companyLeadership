"""Streamlit UI: collect a company's leadership, then chat about it."""

import pandas as pd
import streamlit as st

import database as db
import entities
import rag
import views

st.set_page_config(page_title="Company Leadership RAG Agent", page_icon="🏢", layout="wide")
db.init_db()

# Data stored by older versions of the entity-resolution logic is re-resolved once
# (stored records only - nothing is re-collected).
if not st.session_state.get("reresolved"):
    outdated = [c for c in db.list_companies()
                if (c.get("resolved_version") or 0) < entities.RESOLUTION_VERSION]
    if outdated:
        with st.status("Re-checking sources and duplicates for previously collected companies...") as s:
            try:
                rag.reresolve_outdated(log=st.write)
                s.update(label="Stored data up to date", state="complete")
            except Exception as e:
                s.update(label=f"Update failed: {e}", state="error")
    st.session_state.reresolved = True

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Collect leadership")
    url = st.text_input("Company website", placeholder="https://www.example.com")
    if st.button("Collect", type="primary", disabled=not url):
        with st.status("Collecting leadership data...", expanded=True) as status:
            try:
                cid = rag.ingest_company(url, log=st.write)
                st.session_state.company_id = cid
                st.session_state.company_select = cid  # widget is created below
                st.session_state.messages = []
                status.update(label="Collection complete", state="complete")
            except Exception as e:
                status.update(label="Collection failed", state="error")
                st.error(str(e))

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
        if st.button("Re-check stored data", help="Re-check sources and duplicate people in the "
                     "stored records for this company, without re-collecting."):
            with st.status("Re-running entity resolution...", expanded=True) as s:
                try:
                    rag.reresolve_company(chosen, log=st.write)
                    s.update(label="Deduplication complete", state="complete")
                except Exception as e:
                    s.update(label="Deduplication failed", state="error")
                    st.error(str(e))
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
                "More sources": st.column_config.ListColumn("More sources"),
            },
            hide_index=True,
            width="stretch",
        )
        roles = db.get_role_history(company_id)
        if roles:
            with st.popover("Supporting evidence"):
                st.dataframe(pd.DataFrame(views.evidence_rows(roles),
                                          columns=views.EVIDENCE_COLUMNS),
                             hide_index=True, width="stretch",
                             column_config={"Source": st.column_config.LinkColumn("Source")})
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
