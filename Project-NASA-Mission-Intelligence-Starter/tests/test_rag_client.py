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
