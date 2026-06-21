"""Integration tests for MiniMax proxy routing end-to-end."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from tokentap.proxy import ProxyServer


class TestMiniMaxProxyIntegration:
    """Integration tests for MiniMax provider through the proxy stack."""

    def setup_method(self):
        self.events = []
        self.proxy = ProxyServer(
            port=0, on_request=lambda e: self.events.append(e)
        )

    @pytest.mark.asyncio
    async def test_minimax_request_routed_correctly(self):
        """A request with /minimax/ prefix should be routed to api.minimax.io."""
        body = json.dumps(
            {
                "model": "MiniMax-M3",
                "messages": [{"role": "user", "content": "hello"}],
            }
        ).encode()

        # Simulate request detection
        provider, cleaned = self.proxy._detect_provider(
            "/minimax/v1/chat/completions"
        )
        assert provider == "minimax"
        assert cleaned == "/v1/chat/completions"

        # Process request to verify parsing
        self.proxy._process_request(body, cleaned, provider)
        assert len(self.events) == 1
        assert self.events[0]["provider"] == "minimax"
        assert self.events[0]["model"] == "MiniMax-M3"

    @pytest.mark.asyncio
    async def test_openai_not_affected_by_minimax(self):
        """Regular /v1/chat/completions should still route to OpenAI."""
        provider, path = self.proxy._detect_provider("/v1/chat/completions")
        assert provider == "openai"
        assert path == "/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_minimax_full_event_structure(self):
        """MiniMax events should have all required dashboard fields."""
        body = json.dumps(
            {
                "model": "MiniMax-M2.7",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "What is MiniMax?"},
                ],
                "temperature": 0.7,
            }
        ).encode()

        self.proxy._process_request(
            body, "/v1/chat/completions", "minimax"
        )
        event = self.events[0]
        assert "timestamp" in event
        assert "provider" in event
        assert "model" in event
        assert "tokens" in event
        assert "messages" in event
        assert "raw_body" in event
        assert "path" in event
        assert event["provider"] == "minimax"
        assert event["model"] == "MiniMax-M2.7"
        assert event["tokens"] > 0
        assert len(event["messages"]) == 2
