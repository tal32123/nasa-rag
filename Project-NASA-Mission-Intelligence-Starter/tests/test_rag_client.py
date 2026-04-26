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
