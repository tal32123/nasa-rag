import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List, Dict

load_dotenv()

import rag_client
import ragas_evaluator

st.set_page_config(page_title="Batch Evaluation", page_icon="📊", layout="wide")
st.title("📊 Batch Evaluation")

# ── Sidebar config ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    available_backends = rag_client.discover_chroma_backends()
    if not available_backends:
        st.error("No ChromaDB backends found. Run the embedding pipeline first.")
        st.stop()

    backend_key = st.selectbox(
        "Collection",
        options=list(available_backends.keys()),
        format_func=lambda k: available_backends[k]["display_name"],
    )
    backend = available_backends[backend_key]

    openai_key = st.text_input("OpenAI API Key", type="password",
                               value=os.getenv("OPENAI_API_KEY", ""))
    if not openai_key:
        st.warning("Enter your OpenAI API key to continue.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
    n_results = st.slider("Chunks per question", 1, 10, 3)

    _MISSION_OPTIONS: List[str] = ["All Missions", "Apollo 11", "Apollo 13", "Challenger"]
    _MISSION_SLUG_MAP: Dict[str, Optional[str]] = {
        "All Missions": None, "Apollo 11": "apollo_11",
        "Apollo 13": "apollo_13", "Challenger": "challenger",
    }
    mission_filter = _MISSION_SLUG_MAP[st.selectbox("Mission filter", _MISSION_OPTIONS)]

# ── Init RAG ───────────────────────────────────────────────────────────────────
collection, ok, err = rag_client.initialize_rag_system(backend["directory"], backend["collection_name"])
if not ok:
    st.error(f"Could not connect to collection: {err}")
    st.stop()

# ── Test file picker + run ─────────────────────────────────────────────────────
test_files = [f for f in ["test_questions.json", "evaluation_dataset.txt"] if Path(f).exists()]
if not test_files:
    st.warning("No test files found (`test_questions.json` or `evaluation_dataset.txt`).")
    st.stop()

col_file, col_btn = st.columns([3, 1])
test_file = col_file.selectbox(
    "Test file",
    test_files,
    help="`test_questions.json` includes reference answers for stronger scoring.",
)
run = col_btn.button("▶ Run", type="primary", use_container_width=True)

if run:
    with st.spinner("Running evaluation — this may take a few minutes…"):
        st.session_state["batch_results"] = ragas_evaluator.batch_evaluate(
            test_file, collection, openai_key, model=model,
            n_results=n_results, mission_filter=mission_filter,
        )

# ── Results ────────────────────────────────────────────────────────────────────
results = st.session_state.get("batch_results")
if not results:
    st.info("Select a test file and click **▶ Run** to start.")
    st.stop()

# Aggregate scores
st.subheader("Aggregate Scores")
agg = results.get("aggregate", {})
if agg:
    for col, (metric, value) in zip(st.columns(len(agg)), agg.items()):
        col.metric(metric.replace("_", " ").title(), f"{value:.3f}")

st.divider()

# Per-question table
st.subheader("Per-Question Results")
rows = []
for r in results["results"]:
    row = {"Question": r["question"][:90] + ("…" if len(r["question"]) > 90 else "")}
    row.update({
        k.replace("_", " ").title(): round(v, 3)
        for k, v in r.items()
        if k not in ("question", "answer", "reference_answer") and isinstance(v, float)
    })
    rows.append(row)

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button("⬇ Download CSV", df.to_csv(index=False), "batch_eval_results.csv", "text/csv")

st.divider()

# Answer drill-down
st.subheader("Answer Details")
for r in results["results"]:
    with st.expander(r["question"][:100]):
        st.markdown(f"**Answer**\n\n{r['answer']}")
        if r.get("reference_answer"):
            st.markdown(f"---\n**Reference**\n\n{r['reference_answer']}")
