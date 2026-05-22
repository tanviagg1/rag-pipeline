"""
RAG Pipeline — Streamlit Dashboard (Phase 5)

Requires the FastAPI server running:
    uvicorn api.main:app --reload

Run:
    streamlit run dashboard/app.py

Layout:
  Sidebar  — scrape URLs, KB stats
  Main     — ask question, see agent reasoning steps + final answer
  Bottom   — hybrid search explorer
"""

import requests
import streamlit as st

st.set_page_config(page_title="RAG Pipeline", page_icon="🧠", layout="wide")

DEFAULT_API = "http://localhost:8000"

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api(path: str) -> str:
    return st.session_state.get("api_url", DEFAULT_API).rstrip("/") + path


def post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(api(path), json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect — is `uvicorn api.main:app --reload` running?")
    except requests.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.json().get('detail', str(e))}")
    return None


def get(path: str, params: dict = {}) -> dict | list | None:
    try:
        r = requests.get(api(path), params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Config")
    st.text_input("API URL", value=DEFAULT_API, key="api_url")

    st.divider()
    st.subheader("📥 Add to Knowledge Base")
    url_input = st.text_input("URL to scrape", placeholder="https://en.wikipedia.org/wiki/Raft_(algorithm)")
    force = st.checkbox("Force re-ingest")
    if st.button("Scrape & Ingest", disabled=not url_input.strip()):
        with st.spinner("Scraping..."):
            result = post("/scrape", {"url": url_input.strip(), "force": force})
        if result:
            if result["skipped"]:
                st.info(f"Already stored: {result['url']}")
            else:
                st.success(f"Ingested {result['chunks']} chunks from {result['url']}")

    st.divider()
    st.subheader("📊 Knowledge Base")
    if st.button("Refresh Stats"):
        st.rerun()
    stats = get("/stats")
    if stats:
        st.text(stats.get("stats", "No stats available"))
    else:
        st.caption("No content yet — scrape a URL first.")


# ---------------------------------------------------------------------------
# Main — Ask the Agent
# ---------------------------------------------------------------------------

st.title("🧠 RAG Pipeline — Agentic Q&A")
st.caption("Ask a question. The agent reasons through which tools to use and shows its work.")

question = st.text_area("Your question", placeholder="What is the Raft consensus algorithm?", height=80)
ask_btn = st.button("Ask Agent →", type="primary", disabled=not question.strip())

if ask_btn and question.strip():
    with st.spinner("Agent thinking..."):
        result = post("/ask", {"question": question.strip()})

    if result:
        # Reasoning trace
        if result["steps"]:
            st.subheader("Reasoning Trace")
            for i, step in enumerate(result["steps"], 1):
                with st.expander(f"Step {i} — {step['action']}({step['action_input'][:40]}...)"):
                    st.markdown(f"**Thought:** {step['thought']}")
                    st.markdown(f"**Action:** `{step['action']}({step['action_input']})`")
                    st.markdown(f"**Observation:**")
                    st.text(step["observation"][:500] + ("..." if len(step["observation"]) > 500 else ""))

        if result.get("stopped_early"):
            st.warning("Agent reached max steps — answer may be incomplete.")

        # Final answer
        st.divider()
        st.subheader("Answer")
        st.markdown(result["answer"])

# ---------------------------------------------------------------------------
# Hybrid Search Explorer
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🔍 Hybrid Search Explorer")
st.caption("See what vector + keyword search finds for a query — with RRF scores.")

col1, col2 = st.columns([4, 1])
with col1:
    search_query = st.text_input("Search query", placeholder="distributed consensus leader election")
with col2:
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

if st.button("Search", disabled=not search_query.strip()):
    results = get("/search", {"q": search_query.strip(), "top_k": top_k})
    if results:
        for i, r in enumerate(results, 1):
            with st.expander(f"[{i}] {r['title']} — RRF: {r['rrf_score']:.4f} | {r['sources']}"):
                st.caption(r["url"])
                st.text(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))
    elif results == []:
        st.info("No results — scrape some content first.")
