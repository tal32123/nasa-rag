import os
import json
import statistics
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not question or not question.strip():
        return {"error": "Invalid input: question, answer, and contexts must be non-empty"}
    if not answer or not answer.strip():
        return {"error": "Invalid input: question, answer, and contexts must be non-empty"}
    if not contexts or not any(c for c in contexts if c and c.strip()):
        return {"error": "Invalid input: question, answer, and contexts must be non-empty"}

    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHROMA_OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    reference = reference_answer.strip() if reference_answer and reference_answer.strip() else " ".join(contexts)

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )
    dataset = EvaluationDataset(samples=[sample])

    metrics = [
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm),
        BleuScore(),
        RougeScore(),
    ]

    try:
        result = evaluate(dataset=dataset, metrics=metrics)
        scores_df = result.to_pandas()
        row = scores_df.iloc[0]
        return {
            col: float(row[col])
            for col in scores_df.columns
            if col not in ("user_input", "response", "retrieved_contexts", "reference")
            and row[col] is not None
        }
    except Exception as e:
        return {"error": str(e)}


def batch_evaluate(
    test_file_path: str,
    rag_collection,
    openai_key: str,
    model: str = "gpt-3.5-turbo",
    n_results: int = 5,
    mission_filter: Optional[str] = None,
) -> Dict:
    """Run the full RAG pipeline on each test question and evaluate with RAGAS."""
    from rag_client import retrieve_documents, format_context
    from llm_client import generate_response

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    # Load questions
    questions = []
    if test_file_path.endswith(".json"):
        with open(test_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    q = item.get("question", "").strip()
                    if q:
                        questions.append({
                            "question": q,
                            "reference_answer": item.get("reference_answer", ""),
                        })
                elif isinstance(item, str) and item.strip():
                    questions.append({"question": item.strip(), "reference_answer": ""})
    else:
        with open(test_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append({"question": line, "reference_answer": ""})

    results = []
    for item in questions:
        question = item["question"]
        reference_answer = item.get("reference_answer", "")

        # Retrieve context from ChromaDB
        docs_result = retrieve_documents(
            rag_collection, question, n_results=n_results, mission_filter=mission_filter
        )
        docs = docs_result["documents"][0] if docs_result and docs_result.get("documents") else []
        metas = docs_result["metadatas"][0] if docs_result and docs_result.get("metadatas") else []
        context = format_context(docs, metas)

        # Generate answer via LLM
        answer = generate_response(openai_key, question, context, [], model=model)

        # Evaluate
        scores = evaluate_response_quality(
            question,
            answer,
            docs if docs else ["No context available."],
            reference_answer=reference_answer,
        )
        results.append({
            "question": question,
            "answer": answer,
            "reference_answer": reference_answer,
            **scores,
        })

    # Aggregate numeric scores per metric
    metric_keys = set()
    for r in results:
        for k in r:
            if k not in ("question", "answer", "reference_answer", "error"):
                metric_keys.add(k)

    aggregate = {}
    for key in metric_keys:
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if values:
            aggregate[key] = statistics.mean(values)

    return {"results": results, "aggregate": aggregate}
