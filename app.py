"""Streamlit UI: collect a company's leadership, then chat about it."""

import streamlit as st

import database as db
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
st.markdown("**Leadership**")
with st.expander(f"Leadership ({len(leaders)} people)", expanded=not leaders or len(leaders) < 15):
    st.caption("Collected from public sources (company website, filings, news, public "
               "profiles). The list may not include every leader.")
    if leaders:
        st.markdown(views.summary_html(leaders), unsafe_allow_html=True)
        for note in views.disagreements(leaders):
            st.warning(f"Unresolved: {note}")
    else:
        st.warning(
            "No leaders found. The site may render its team page with JavaScript, "
            "block automated requests, or not publish leadership publicly."
        )
st.markdown("**Person Details**")
with st.expander(f"Person Details ({len(leaders)} people)", expanded=False):
    for p in leaders:
        with st.expander(views.person_label(p)):
            st.markdown(views.details_html(p), unsafe_allow_html=True)  

st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Placed in st.bottom explicitly: it stays pinned at the bottom, but without the
# auto-scroll Streamlit adds for a top-level chat input (which jumped the page to the
# bottom whenever an expander was opened).
question = st.bottom.chat_input("Ask about this company's leadership, e.g. 'Who runs engineering?'")

# Fixed script (no user or scraped data in it): bring the newest question to the top
# of the view so the answer below it is read from its start.
_SCROLL_TO_QUESTION = """<script>setTimeout(() => {
  const q = [...document.querySelectorAll('[data-testid="stChatMessageAvatarUser"]')].pop();
  if (q) q.closest('[data-testid="stChatMessage"]').scrollIntoView({behavior: "smooth", block: "start"});
}, 150);</script>"""

if question:
    with st.chat_message("user"):
        st.markdown(question)
        st.html(_SCROLL_TO_QUESTION, unsafe_allow_javascript=True)  # show "Thinking..."
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = rag.answer(company_id, question, st.session_state.messages)
            except Exception as e:
                reply = f"Error answering question: {e}"
        st.markdown(reply)
        st.html(_SCROLL_TO_QUESTION, unsafe_allow_javascript=True)  # and the answer
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ]
