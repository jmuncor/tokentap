"""Unit tests for tokentap proxy — provider detection and MiniMax routing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tokentap.config import PROVIDERS
from tokentap.proxy import ProxyServer


class TestDetectProvider:
    """Tests for ProxyServer._detect_provider."""

    def setup_method(self):
        self.proxy = ProxyServer(port=9999)

    # --- existing providers (regression) ---

    def test_anthropic_messages(self):
        provider, path = self.proxy._detect_provider("/v1/messages")
        assert provider == "anthropic"
        assert path == "/v1/messages"

    def test_openai_chat_completions(self):
        provider, path = self.proxy._detect_provider("/v1/chat/completions")
        assert provider == "openai"
        assert path == "/v1/chat/completions"

    def test_openai_responses(self):
        provider, path = self.proxy._detect_provider("/v1/responses")
        assert provider == "openai"
        assert path == "/v1/responses"

    def test_gemini_generate(self):
        provider, path = self.proxy._detect_provider(
            "/v1beta/models/gemini:generateContent"
        )
        assert provider == "gemini"

    def test_gemini_stream(self):
        provider, path = self.proxy._detect_provider(
            "/v1beta/models/gemini:streamGenerateContent"
        )
        assert provider == "gemini"

    def test_unknown_path(self):
        provider, path = self.proxy._detect_provider("/unknown/endpoint")
        assert provider is None
        assert path == "/unknown/endpoint"

    # --- MiniMax path-prefix routing ---

    def test_minimax_chat_completions(self):
        provider, path = self.proxy._detect_provider(
            "/minimax/v1/chat/completions"
        )
        assert provider == "minimax"
        assert path == "/v1/chat/completions"

    def test_minimax_models(self):
        provider, path = self.proxy._detect_provider("/minimax/v1/models")
        assert provider == "minimax"
        assert path == "/v1/models"

    def test_minimax_prefix_stripped(self):
        """The /minimax prefix should be stripped from the cleaned path."""
        _, path = self.proxy._detect_provider("/minimax/v1/chat/completions")
        assert not path.startswith("/minimax")

    def test_minimax_upstream_url(self):
        """The upstream URL should point to api.minimax.io."""
        provider, path = self.proxy._detect_provider(
            "/minimax/v1/chat/completions"
        )
        upstream = PROVIDERS[provider]["base_url"] + path
        assert upstream == "https://api.minimax.io/v1/chat/completions"

    def test_minimax_prefix_only_not_matched(self):
        """Bare /minimax without trailing slash should not match."""
        provider, _ = self.proxy._detect_provider("/minimax")
        assert provider is None

    def test_minimax_with_query_string(self):
        provider, path = self.proxy._detect_provider(
            "/minimax/v1/chat/completions?stream=true"
        )
        assert provider == "minimax"
        assert path == "/v1/chat/completions?stream=true"


class TestProcessRequest:
    """Tests for ProxyServer._process_request with MiniMax provider."""

    def setup_method(self):
        self.events = []
        self.proxy = ProxyServer(
            port=9999, on_request=lambda e: self.events.append(e)
        )

    def test_minimax_request_parsed_as_openai_compat(self):
        body = {
            "model": "MiniMax-M3",
            "messages": [
                {"role": "user", "content": "Hello MiniMax"},
            ],
        }
        self.proxy._process_request(
            json.dumps(body).encode(),
            "/v1/chat/completions",
            "minimax",
        )
        assert len(self.events) == 1
        assert self.events[0]["provider"] == "minimax"
        assert self.events[0]["model"] == "MiniMax-M3"
        assert self.events[0]["tokens"] > 0

    def test_minimax_multimodal_content(self):
        body = {
            "model": "MiniMax-M2.7",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                    ],
                },
            ],
        }
        self.proxy._process_request(
            json.dumps(body).encode(),
            "/v1/chat/completions",
            "minimax",
        )
        assert len(self.events) == 1
        assert self.events[0]["model"] == "MiniMax-M2.7"

    def test_minimax_system_message(self):
        body = {
            "model": "MiniMax-M3",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        }
        self.proxy._process_request(
            json.dumps(body).encode(),
            "/v1/chat/completions",
            "minimax",
        )
        event = self.events[0]
        msgs = event["raw_body"]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_minimax_no_callback_no_error(self):
        proxy = ProxyServer(port=9999, on_request=None)
        body = {"model": "MiniMax-M3", "messages": []}
        # Should not raise
        proxy._process_request(
            json.dumps(body).encode(),
            "/v1/chat/completions",
            "minimax",
        )

    def test_minimax_invalid_json_no_error(self):
        self.proxy._process_request(
            b"not-json",
            "/v1/chat/completions",
            "minimax",
        )
        assert len(self.events) == 0
