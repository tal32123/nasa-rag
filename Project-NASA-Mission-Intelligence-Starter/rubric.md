Rubric
Use this project rubric to understand and assess the project criteria.

Embedding & Data Pipeline
Criteria	Submission Requirements
The code demonstrates a chunking strategy

Uses configurable chunk_size and chunk_overlap parameters at runtime (CLI flags or config).
Produces chunks that never exceed chunk_size characters/tokens.
Applies chunk_overlap consistently between consecutive chunks.
The code demonstrates correct embedding creation with metadata

Calls an OpenAI embedding model to vectorize each chunk.
Stores per-chunk metadata including at least source/filepath and mission (Apollo 11, Apollo 13, Challenger).
Handles non-new documents according to --update-mode with one of: skip, update, or replace.
Persist and inspect a ChromaDB collection

Writes embeddings to a ChromaDB collection at the configured --chroma-dir and --collection-name. --stats-only (or equivalent) prints collection size and at least one aggregate (e.g., number of documents/chunks).
Retrieval & LLM Integration
Criteria	Submission Requirements
The code demonstrates semantic retrieval from ChromaDB

Connects to the ChromaDB backend and issues similarity queries using the user question embedding.
Returns top-k results with k configurable at runtime. (If mission filtering is selected) restricts results to the chosen mission via metadata filtering.
The code demonstrates clean context construction for the LLM

Formats retrieved chunks into a single context string with clear separators and source attributions.
Deduplicates or sorts by score to avoid repeated snippets.
The code demonstrates a well-scoped system prompt and conversation management.

Implements a system prompt positioning the assistant as a NASA mission expert who cites retrieved sources.
Maintains conversation history across turns (role + content per turn) and includes only necessary context for each call.
Generates LLM answers grounded in retrieved context.

Passes the constructed context plus user query to the LLM and returns an answer.
Avoids ungrounded claims by instructing the model to rely on provided context and indicate uncertainty when context is insufficient.
Real-Time Evaluation
Criteria	Submission Requirements
The code demonstrates integration of RAGAS metrics

Computes at least Response Relevancy and Faithfulness for each answer.

Supports additional metrics (e.g., BLEU, ROUGE, Precision) as documented.

Evaluates a (question, context, answer) triple

Accepts inputs (question, retrieved context, model answer) and returns a structured result with metric names and values.
Handles empty or malformed inputs with a clear error message (no crashes).
Batch evaluation using a test set

Loads test questions (e.g., test_questions.json or evaluation_dataset.txt) and computes metrics end-to-end.
Outputs a summary per question and an aggregate (mean or distribution) for each metric.
Evaluation dataset

Includes evaluation_dataset.txt or test_questions.json with at least 5 mission-relevant questions spanning multiple categories (overview, emergency, disaster analysis, crew, technical, timeline).
File loads without errors and is referenced by the evaluation flow or README.