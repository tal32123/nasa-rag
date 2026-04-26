import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

import llm_client


def _mock_openai(content: str):
    """Return a patched OpenAI class whose completions return *content*."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    mock_cls = MagicMock(return_value=mock_client)
    return mock_cls, mock_client


def test_generate_response_returns_string():
    mock_cls, _ = _mock_openai("Apollo 11 landed on the moon.")
    with patch("llm_client.OpenAI", mock_cls):
        result = llm_client.generate_response(
            openai_key="sk-test",
            user_message="What was Apollo 11?",
            context="Apollo 11 was the first crewed moon landing.",
            conversation_history=[],
        )
    assert result == "Apollo 11 landed on the moon."


def test_generate_response_system_prompt_is_first_message():
    mock_cls, mock_client = _mock_openai("answer")
    with patch("llm_client.OpenAI", mock_cls):
        llm_client.generate_response("sk-test", "q", "ctx", [])

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "NASA" in messages[0]["content"]


def test_generate_response_context_appears_in_messages():
    mock_cls, mock_client = _mock_openai("answer")
    with patch("llm_client.OpenAI", mock_cls):
        llm_client.generate_response("sk-test", "question", "some NASA context", [])

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    all_content = " ".join(m["content"] for m in messages)
    assert "some NASA context" in all_content


def test_generate_response_empty_context_skips_context_message():
    mock_cls, mock_client = _mock_openai("answer")
    with patch("llm_client.OpenAI", mock_cls):
        llm_client.generate_response("sk-test", "question", "", [])

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    # Only system + user question — no context injection
    assert len(messages) == 2


def test_generate_response_includes_conversation_history():
    mock_cls, mock_client = _mock_openai("answer")
    history = [
        {"role": "user", "content": "prev question"},
        {"role": "assistant", "content": "prev answer"},
    ]
    with patch("llm_client.OpenAI", mock_cls):
        llm_client.generate_response("sk-test", "current question", "", history)

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    contents = [m["content"] for m in messages]
    assert "prev question" in contents
    assert "prev answer" in contents
    assert "current question" in contents


def test_generate_response_user_message_is_last():
    mock_cls, mock_client = _mock_openai("answer")
    with patch("llm_client.OpenAI", mock_cls):
        llm_client.generate_response("sk-test", "final question", "ctx", [])

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "final question"
