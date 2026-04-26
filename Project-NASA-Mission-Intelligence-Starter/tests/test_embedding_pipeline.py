from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import embedding_pipeline
from embedding_pipeline import ChromaEmbeddingPipelineTextOnly


def _make_pipeline(chunk_size: int = 1000, chunk_overlap: int = 200):
    """Build a pipeline with all external clients mocked out."""
    with patch("embedding_pipeline.OpenAI"), \
         patch("embedding_pipeline.chromadb.PersistentClient"), \
         patch("embedding_pipeline.OpenAIEmbeddingFunction"):
        p = ChromaEmbeddingPipelineTextOnly(
            openai_api_key="sk-test",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    p.openai_client = MagicMock()
    p.collection = MagicMock()
    return p


class TestInit:
    def test_stores_config(self):
        p = _make_pipeline(chunk_size=500, chunk_overlap=50)
        assert p.chunk_size == 500
        assert p.chunk_overlap == 50
        assert p.embedding_model == "text-embedding-3-small"

    def test_collection_created(self):
        with patch("embedding_pipeline.OpenAI"), \
             patch("embedding_pipeline.chromadb.PersistentClient") as mock_chroma, \
             patch("embedding_pipeline.OpenAIEmbeddingFunction"):
            ChromaEmbeddingPipelineTextOnly(openai_api_key="sk-test")
            mock_chroma.return_value.get_or_create_collection.assert_called_once()


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        p = _make_pipeline(chunk_size=1000)
        meta = {"source": "test", "mission": "apollo_11"}
        chunks = p.chunk_text("Short text.", meta)
        assert len(chunks) == 1
        assert chunks[0][0] == "Short text."
        assert chunks[0][1]["chunk_index"] == 0
        assert chunks[0][1]["total_chunks"] == 1

    def test_text_exactly_chunk_size_is_single_chunk(self):
        p = _make_pipeline(chunk_size=10)
        chunks = p.chunk_text("1234567890", {"source": "s", "mission": "m"})
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        p = _make_pipeline(chunk_size=100, chunk_overlap=20)
        text = "A" * 250
        chunks = p.chunk_text(text, {"source": "s", "mission": "m"})
        assert len(chunks) > 1

    def test_no_chunk_exceeds_chunk_size(self):
        p = _make_pipeline(chunk_size=100, chunk_overlap=10)
        text = "Hello world. " * 30
        chunks = p.chunk_text(text, {"source": "s", "mission": "m"})
        for chunk_text, _ in chunks:
            assert len(chunk_text) <= 100

    def test_chunk_indices_are_sequential(self):
        p = _make_pipeline(chunk_size=50, chunk_overlap=5)
        text = "X" * 200
        chunks = p.chunk_text(text, {"source": "s", "mission": "m"})
        for i, (_, meta) in enumerate(chunks):
            assert meta["chunk_index"] == i

    def test_total_chunks_consistent(self):
        p = _make_pipeline(chunk_size=50, chunk_overlap=5)
        text = "X" * 200
        chunks = p.chunk_text(text, {"source": "s", "mission": "m"})
        total = len(chunks)
        for _, meta in chunks:
            assert meta["total_chunks"] == total

    def test_metadata_copied_to_each_chunk(self):
        p = _make_pipeline(chunk_size=50, chunk_overlap=5)
        meta = {"source": "test_src", "mission": "challenger", "custom_key": "val"}
        chunks = p.chunk_text("X" * 200, meta)
        for _, chunk_meta in chunks:
            assert chunk_meta["source"] == "test_src"
            assert chunk_meta["mission"] == "challenger"
            assert chunk_meta["custom_key"] == "val"


class TestGetEmbedding:
    def test_returns_embedding_list(self):
        p = _make_pipeline()
        mock_resp = MagicMock()
        mock_resp.data[0].embedding = [0.1, 0.2, 0.3]
        p.openai_client.embeddings.create.return_value = mock_resp

        result = p.get_embedding("test text")

        assert result == [0.1, 0.2, 0.3]

    def test_calls_correct_model(self):
        p = _make_pipeline()
        mock_resp = MagicMock()
        mock_resp.data[0].embedding = [0.0]
        p.openai_client.embeddings.create.return_value = mock_resp

        p.get_embedding("text")

        p.openai_client.embeddings.create.assert_called_once_with(
            model=p.embedding_model,
            input="text",
        )


class TestGenerateDocumentId:
    def test_format_is_correct(self):
        p = _make_pipeline()
        file_path = Path("data/apollo11/a11_tec.txt")
        meta = {"mission": "apollo_11", "source": "a11_tec", "chunk_index": 5}
        doc_id = p.generate_document_id(file_path, meta)
        assert doc_id == "apollo_11_a11_tec_chunk_0005"

    def test_chunk_index_zero_padded(self):
        p = _make_pipeline()
        meta = {"mission": "challenger", "source": "sts51l", "chunk_index": 0}
        doc_id = p.generate_document_id(Path("f.txt"), meta)
        assert doc_id == "challenger_sts51l_chunk_0000"

    def test_large_chunk_index(self):
        p = _make_pipeline()
        meta = {"mission": "apollo_13", "source": "src", "chunk_index": 9999}
        doc_id = p.generate_document_id(Path("f.txt"), meta)
        assert doc_id == "apollo_13_src_chunk_9999"


class TestCheckDocumentExists:
    def test_returns_true_when_ids_present(self):
        p = _make_pipeline()
        p.collection.get.return_value = {"ids": ["apollo_11_src_chunk_0001"]}
        assert p.check_document_exists("apollo_11_src_chunk_0001") is True

    def test_returns_false_when_ids_empty(self):
        p = _make_pipeline()
        p.collection.get.return_value = {"ids": []}
        assert p.check_document_exists("nonexistent") is False
