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


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
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

    reference = " ".join(contexts)

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
) -> Dict:
    """Evaluate a set of questions loaded from a JSON or TXT file."""
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
                        questions.append({"question": q, "answer": item.get("answer", ""), "contexts": item.get("contexts", [])})
                elif isinstance(item, str) and item.strip():
                    questions.append({"question": item.strip(), "answer": "", "contexts": []})
    else:
        with open(test_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append({"question": line, "answer": "", "contexts": []})

    results = []
    for item in questions:
        scores = evaluate_response_quality(
            item["question"],
            item["answer"] if item["answer"] else "No answer provided.",
            item["contexts"] if item["contexts"] else ["No context available."],
        )
        results.append({"question": item["question"], **scores})

    # Aggregate numeric scores per metric
    metric_keys = set()
    for r in results:
        for k in r:
            if k not in ("question", "error"):
                metric_keys.add(k)

    aggregate = {}
    for key in metric_keys:
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if values:
            aggregate[key] = statistics.mean(values)

    return {"results": results, "aggregate": aggregate}
