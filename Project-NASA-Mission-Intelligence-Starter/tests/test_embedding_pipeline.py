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


class TestAddDocumentsToCollection:
    def _docs(self):
        return [
            ("text one", {"mission": "apollo_11", "source": "src", "chunk_index": 0}),
            ("text two", {"mission": "apollo_11", "source": "src", "chunk_index": 1}),
        ]

    def test_empty_documents_returns_zero_stats(self):
        p = _make_pipeline()
        stats = p.add_documents_to_collection([], Path("test.txt"))
        assert stats == {"added": 0, "updated": 0, "skipped": 0}

    def test_skip_mode_skips_existing(self):
        p = _make_pipeline()
        p.check_document_exists = MagicMock(return_value=True)
        p.get_embedding = MagicMock(return_value=[0.1])

        stats = p.add_documents_to_collection(self._docs(), Path("test.txt"), update_mode="skip")

        assert stats["skipped"] == 2
        assert stats["added"] == 0
        p.collection.add.assert_not_called()

    def test_skip_mode_adds_new(self):
        p = _make_pipeline()
        p.check_document_exists = MagicMock(return_value=False)
        p.get_embedding = MagicMock(return_value=[0.1])

        stats = p.add_documents_to_collection(self._docs(), Path("test.txt"), update_mode="skip")

        assert stats["added"] == 2
        assert stats["skipped"] == 0
        assert p.collection.add.call_count == 2

    def test_replace_mode_deletes_existing_first(self):
        p = _make_pipeline()
        p.get_file_documents = MagicMock(return_value=["old_id_0", "old_id_1"])
        p.get_embedding = MagicMock(return_value=[0.1])

        p.add_documents_to_collection(self._docs(), Path("test.txt"), update_mode="replace")

        p.collection.delete.assert_called_once_with(ids=["old_id_0", "old_id_1"])

    def test_replace_mode_adds_all_docs(self):
        p = _make_pipeline()
        p.get_file_documents = MagicMock(return_value=[])
        p.get_embedding = MagicMock(return_value=[0.1])

        stats = p.add_documents_to_collection(self._docs(), Path("test.txt"), update_mode="replace")

        assert stats["added"] == 2

    def test_update_mode_updates_existing(self):
        p = _make_pipeline()
        p.check_document_exists = MagicMock(return_value=True)
        p.update_document = MagicMock(return_value=True)

        stats = p.add_documents_to_collection(self._docs(), Path("test.txt"), update_mode="update")

        assert stats["updated"] == 2
        assert stats["added"] == 0

    def test_update_mode_adds_new_docs(self):
        p = _make_pipeline()
        p.check_document_exists = MagicMock(return_value=False)
        p.get_embedding = MagicMock(return_value=[0.1])

        stats = p.add_documents_to_collection(self._docs(), Path("test.txt"), update_mode="update")

        assert stats["added"] == 2
        assert stats["updated"] == 0


class TestGetCollectionInfo:
    def test_returns_name_and_count(self):
        p = _make_pipeline()
        p.collection.name = "nasa_missions"
        p.collection.count.return_value = 250

        info = p.get_collection_info()

        assert info["collection_name"] == "nasa_missions"
        assert info["document_count"] == 250


class TestQueryCollection:
    def test_delegates_to_collection_query(self):
        p = _make_pipeline()
        p.collection.query.return_value = {"documents": [["result"]]}

        result = p.query_collection("moon landing", n_results=3)

        p.collection.query.assert_called_once_with(
            query_texts=["moon landing"], n_results=3
        )
        assert result["documents"] == [["result"]]


class TestProcessAllTextData:
    def test_returns_stats_dict_with_expected_keys(self):
        p = _make_pipeline()
        p.scan_text_files_only = MagicMock(return_value=[])

        stats = p.process_all_text_data("./data_text")

        assert "files_processed" in stats
        assert "documents_added" in stats
        assert "total_chunks" in stats
        assert "errors" in stats
        assert "missions" in stats

    def test_processes_files_and_accumulates_stats(self, tmp_path):
        p = _make_pipeline()
        fake_file = tmp_path / "test.txt"
        fake_file.write_text("hello world")

        p.scan_text_files_only = MagicMock(return_value=[fake_file])
        p.process_text_file = MagicMock(
            return_value=[("chunk", {"mission": "apollo_11", "source": "test", "chunk_index": 0})]
        )
        p.add_documents_to_collection = MagicMock(
            return_value={"added": 1, "updated": 0, "skipped": 0}
        )

        stats = p.process_all_text_data("./data_text")

        assert stats["files_processed"] == 1
        assert stats["documents_added"] == 1
        assert stats["total_chunks"] == 1
        assert stats["errors"] == 0

    def test_errors_counted_on_exception(self):
        p = _make_pipeline()
        fake_path = MagicMock()
        fake_path.name = "bad.txt"

        p.scan_text_files_only = MagicMock(return_value=[fake_path])
        p.process_text_file = MagicMock(side_effect=Exception("read error"))

        stats = p.process_all_text_data("./data_text")

        assert stats["errors"] == 1
        assert stats["files_processed"] == 0
