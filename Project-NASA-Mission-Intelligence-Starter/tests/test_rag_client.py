import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import rag_client


class TestFormatContext:
    def test_empty_documents_returns_empty_string(self):
        assert rag_client.format_context([], []) == ""

    def test_single_document_contains_source_header(self):
        docs = ["Apollo 13 had an oxygen tank explosion."]
        metas = [{"mission": "apollo_13", "source": "AS13_PAO", "document_category": "public_affairs_officer"}]
        result = rag_client.format_context(docs, metas)
        assert "[Source 1" in result
        assert "Apollo 13" in result
        assert "AS13_PAO" in result

    def test_document_content_included(self):
        docs = ["Specific mission detail here."]
        metas = [{"mission": "apollo_11", "source": "a11tec", "document_category": "technical"}]
        result = rag_client.format_context(docs, metas)
        assert "Specific mission detail here." in result

    def test_long_document_is_truncated(self):
        docs = ["x" * 2000]
        metas = [{"mission": "apollo_11", "source": "src", "document_category": "general_document"}]
        result = rag_client.format_context(docs, metas)
        assert "..." in result
        assert "x" * 1501 not in result

    def test_short_document_not_truncated(self):
        docs = ["short"]
        metas = [{"mission": "apollo_11", "source": "src", "document_category": "technical"}]
        result = rag_client.format_context(docs, metas)
        assert "short" in result
        assert "..." not in result

    def test_multiple_documents_all_present(self):
        docs = ["content one", "content two"]
        metas = [
            {"mission": "apollo_11", "source": "src1", "document_category": "technical"},
            {"mission": "apollo_13", "source": "src2", "document_category": "flight_plan"},
        ]
        result = rag_client.format_context(docs, metas)
        assert "[Source 1" in result
        assert "[Source 2" in result
        assert "content one" in result
        assert "content two" in result

    def test_missing_metadata_uses_fallback(self):
        docs = ["some text"]
        metas = [{}]
        result = rag_client.format_context(docs, metas)
        assert "[Source 1" in result
        assert "unknown" in result.lower()

    def test_underscore_mission_is_title_cased(self):
        docs = ["text"]
        metas = [{"mission": "apollo_11", "source": "s", "document_category": "c"}]
        result = rag_client.format_context(docs, metas)
        assert "Apollo 11" in result


class TestRetrieveDocuments:
    def _make_collection(self):
        col = MagicMock()
        col.query.return_value = {"documents": [["doc"]], "metadatas": [[{"mission": "apollo_11"}]]}
        return col

    def test_no_filter_passes_none_where(self):
        col = self._make_collection()
        rag_client.retrieve_documents(col, "moon landing", n_results=3)
        col.query.assert_called_once_with(
            query_texts=["moon landing"], n_results=3, where=None
        )

    def test_mission_filter_applied(self):
        col = self._make_collection()
        rag_client.retrieve_documents(col, "explosion", n_results=5, mission_filter="apollo_13")
        col.query.assert_called_once_with(
            query_texts=["explosion"], n_results=5, where={"mission": "apollo_13"}
        )

    def test_filter_all_treated_as_no_filter(self):
        col = self._make_collection()
        rag_client.retrieve_documents(col, "query", mission_filter="all")
        call_kwargs = col.query.call_args.kwargs
        assert call_kwargs["where"] is None

    def test_returns_query_result(self):
        col = self._make_collection()
        result = rag_client.retrieve_documents(col, "query")
        assert result["documents"] == [["doc"]]


class TestInitializeRagSystem:
    def test_success_returns_collection_true_empty_error(self):
        mock_col = MagicMock()
        with patch("rag_client.chromadb.PersistentClient") as mock_cls, \
             patch("rag_client.OpenAIEmbeddingFunction"), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            mock_cls.return_value.get_collection.return_value = mock_col
            col, ok, err = rag_client.initialize_rag_system("./chroma_db", "test_col")
        assert ok is True
        assert err == ""
        assert col is mock_col

    def test_failure_returns_none_false_error_string(self):
        with patch("rag_client.chromadb.PersistentClient") as mock_cls, \
             patch("rag_client.OpenAIEmbeddingFunction"), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            mock_cls.side_effect = Exception("Cannot connect")
            col, ok, err = rag_client.initialize_rag_system("./bad", "col")
        assert ok is False
        assert col is None
        assert "Cannot connect" in err

    def test_uses_chroma_openai_api_key_env_var(self):
        with patch("rag_client.chromadb.PersistentClient"), \
             patch("rag_client.OpenAIEmbeddingFunction") as mock_fn, \
             patch.dict(os.environ, {"CHROMA_OPENAI_API_KEY": "sk-chroma"}, clear=True):
            try:
                rag_client.initialize_rag_system("./chroma_db", "col")
            except Exception:
                pass
            call_kwargs = mock_fn.call_args.kwargs
            assert call_kwargs["api_key"] == "sk-chroma"


class TestDiscoverChromaBackends:
    def test_no_chroma_dirs_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = rag_client.discover_chroma_backends()
        assert result == {}

    def test_finds_collection_and_builds_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "chroma_db_nasa").mkdir()

        mock_col = MagicMock()
        mock_col.name = "nasa_missions"
        mock_col.count.return_value = 42

        with patch("rag_client.chromadb.PersistentClient") as mock_cls:
            mock_cls.return_value.list_collections.return_value = [mock_col]
            result = rag_client.discover_chroma_backends()

        key = "chroma_db_nasa::nasa_missions"
        assert key in result
        assert result[key]["collection_name"] == "nasa_missions"
        assert result[key]["doc_count"] == "42"
        assert "chroma_db_nasa" in result[key]["display_name"]

    def test_non_chroma_dir_ignored(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "some_other_dir").mkdir()
        result = rag_client.discover_chroma_backends()
        assert result == {}

    def test_inaccessible_dir_creates_error_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "chroma_db_bad").mkdir()

        with patch("rag_client.chromadb.PersistentClient") as mock_cls:
            mock_cls.side_effect = Exception("Permission denied")
            result = rag_client.discover_chroma_backends()

        key = "chroma_db_bad::error"
        assert key in result
        assert "error" in result[key]["display_name"].lower()
        assert result[key]["doc_count"] == "0"
