import streamlit as st
import requests
import json

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="NeuroAssist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .route-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .route-graph    { background: #e8f4fd; color: #1f77b4; }
    .route-papers   { background: #e8f5e9; color: #2e7d32; }
    .route-code     { background: #fff3e0; color: #e65100; }
    .route-pubmed   { background: #f3e5f5; color: #6a1b9a; }
    .source-chip {
        display: inline-block;
        background: #f0f0f0;
        border-radius: 8px;
        padding: 2px 8px;
        font-size: 0.78rem;
        margin: 2px;
        color: #333;
    }
    .score-good  { color: #2e7d32; font-weight: 600; }
    .score-ok    { color: #f57c00; font-weight: 600; }
    .score-low   { color: #c62828; font-weight: 600; }
    .example-btn { margin: 2px 0; }
    .answer-box {
        background: #fafafa;
        border-left: 4px solid #1f77b4;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def ask_api(question: str) -> dict:
    """Calls the FastAPI /ask endpoint."""
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={"question": question},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the backend is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The agent is taking too long."}
    except Exception as e:
        return {"error": str(e)}


def get_health() -> dict:
    """Calls the /health endpoint."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.json()
    except Exception:
        return {"status": "unreachable", "message": "API is not running"}


def get_examples() -> dict:
    """Fetches example questions from API."""
    try:
        response = requests.get(f"{API_BASE}/examples", timeout=5)
        return response.json().get("examples", {})
    except Exception:
        return {}


def route_color(route: str) -> str:
    colors = {
        "graph": "🔵",
        "papers": "🟢",
        "code": "🟠",
        "pubmed": "🟣",
        "papers_and_code": "🟡"
    }
    return colors.get(route, "⚪")


def score_color(score: float) -> str:
    if score >= 0.8:
        return "score-good"
    elif score >= 0.5:
        return "score-ok"
    return "score-low"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 NeuroAssist")
    st.markdown("*Research assistant for the Shrestha Lab*")
    st.divider()

    # Health status
    st.markdown("### System Status")
    health = get_health()
    status = health.get("status", "unknown")

    if status == "healthy":
        st.success("✅ API Online")
    elif status == "degraded":
        st.warning("⚠️ API Degraded")
    else:
        st.error("❌ API Offline")
        st.markdown("Start the backend with:")
        st.code("python backend/api/main.py")

    # Collection stats
    collections = health.get("collections", {})
    if collections:
        st.divider()
        st.markdown("### Knowledge Base")
        for name, info in collections.items():
            label = "📄 Papers" if "papers" in name else "💻 Code"
            st.metric(label, f"{info['count']} chunks")

    st.divider()

    # Retriever guide
    st.markdown("### How It Works")
    st.markdown("""
    🔵 **Graph** — Lab overview, structure, relationships

    🟢 **Papers** — Research findings from lab publications

    🟠 **Code** — Pipeline implementation questions

    🟣 **PubMed** — External literature search
    """)

    st.divider()
    st.markdown("### Example Questions")

    examples = get_examples()

    # Show example buttons grouped by category
    category_labels = {
        "lab_research": "🏛️ Lab Overview",
        "papers": "📄 Research",
        "pipeline_code": "💻 Code",
        "pubmed": "🔬 PubMed"
    }

    for category, label in category_labels.items():
        with st.expander(label):
            for q in examples.get(category, []):
                if st.button(q, key=f"btn_{q[:30]}", use_container_width=True):
                    st.session_state["pending_question"] = q


# ── Main content ──────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🧠 NeuroAssist</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Intelligent research assistant for the Shrestha Lab · '
    'Stony Brook University, Dept. of Neurobiology & Behavior</div>',
    unsafe_allow_html=True
)

# ── Initialize session state ──────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# ── Question input ────────────────────────────────────────────────────────────

col1, col2 = st.columns([5, 1])

with col1:
    question = st.text_input(
        "Ask a question",
        value=st.session_state.get("pending_question", ""),
        placeholder="e.g. What does the Shrestha lab study?",
        label_visibility="collapsed",
        key="question_input"
    )

with col2:
    ask_clicked = st.button("Ask →", type="primary", use_container_width=True)

# Clear pending question after it's been loaded into input
if st.session_state.get("pending_question"):
    st.session_state["pending_question"] = ""

# ── Handle question ───────────────────────────────────────────────────────────

if ask_clicked and question.strip():
    with st.spinner("🤔 Thinking..."):
        result = ask_api(question.strip())

    if "error" in result:
        st.error(f"❌ {result['error']}")
    else:
        # Add to history
        st.session_state.chat_history.insert(0, {
            "question": question.strip(),
            "result": result
        })

elif ask_clicked and not question.strip():
    st.warning("Please enter a question first.")

# ── Display chat history ──────────────────────────────────────────────────────

if st.session_state.chat_history:
    st.divider()

    # Clear history button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    for i, entry in enumerate(st.session_state.chat_history):
        q = entry["question"]
        r = entry["result"]

        with st.container():
            # Question
            st.markdown(f"**Q: {q}**")

            # Route badge + sources
            route = r.get("route", "unknown")
            sources = r.get("sources", [])
            score = r.get("faithfulness_score")

            meta_cols = st.columns([1, 3, 1])
            with meta_cols[0]:
                st.markdown(
                    f"{route_color(route)} **{route.replace('_', ' ').title()}**"
                )
            with meta_cols[1]:
                if sources:
                    sources_html = " ".join(
                        f'<span class="source-chip">{s}</span>'
                        for s in sources
                    )
                    st.markdown(sources_html, unsafe_allow_html=True)
            with meta_cols[2]:
                if score is not None:
                    css_class = score_color(score)
                    st.markdown(
                        f'<span class="{css_class}">Faithfulness: {score:.2f}</span>',
                        unsafe_allow_html=True
                    )

            # Answer
            st.markdown(
                f'<div class="answer-box">{r.get("answer", "No answer")}</div>',
                unsafe_allow_html=True
            )

            if i < len(st.session_state.chat_history) - 1:
                st.divider()

else:
    # Empty state — show welcome message
    st.markdown("---")
    st.markdown("### 👋 Welcome to NeuroAssist")
    st.markdown("""
    I can help you with:

    - **Lab research** — *"What does the Shrestha lab study?"*
    - **Published findings** — *"What did the lab find about dopamine?"*
    - **Pipeline code** — *"How does the snippet extractor work?"*
    - **External literature** — *"Find recent papers on active avoidance"*

    Try an example question from the sidebar, or type your own above.
    """)