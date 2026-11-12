"""Streamlit UI: collect a company's leadership, then chat about it."""

import pandas as pd
import streamlit as st

import database as db
import rag

st.set_page_config(page_title="Company Leadership RAG Agent", page_icon="🏢", layout="wide")
db.init_db()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Collect leadership")
    url = st.text_input("Company website", placeholder="https://www.example.com")
    if st.button("Collect", type="primary", disabled=not url):
        with st.status("Collecting leadership data...", expanded=True) as status:
            try:
                cid = rag.ingest_company(url, log=st.write)
                st.session_state.company_id = cid
                st.session_state.messages = []
                status.update(label="Collection complete", state="complete")
            except Exception as e:
                status.update(label="Collection failed", state="error")
                st.error(str(e))

    st.header("2. Pick a company")
    companies = db.list_companies()
    if companies:
        ids = [c["id"] for c in companies]
        current = st.session_state.get("company_id", ids[0])
        chosen = st.selectbox(
            "Company",
            ids,
            index=ids.index(current) if current in ids else 0,
            format_func=lambda i: next(
                f"{c['name']} ({c['leader_count']} leaders)" for c in companies if c["id"] == i
            ),
        )
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
with st.expander(f"Leadership table ({len(leaders)} people)", expanded=not leaders or len(leaders) < 15):
    if leaders:
        df = pd.DataFrame(leaders)
        df["source_urls"] = df["source_urls"].fillna("").str.split("\n")
        st.dataframe(
            df,
            column_config={
                "source_url": st.column_config.LinkColumn("primary source"),
                "source_urls": st.column_config.ListColumn("all sources"),
                "evidence": st.column_config.TextColumn("evidence (verbatim)"),
            },
            hide_index=True,
            width="stretch",
        )
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
