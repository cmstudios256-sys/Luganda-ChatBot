"""LLM providers behind one async streaming interface.

Only httpx is required, so the app runs without vendor SDKs. Supported:
  openai      - OpenAI chat completions (also Azure-style compatible gateways)
  compatible  - any OpenAI-compatible server: vLLM, Ollama, OpenRouter, Together,
                or a locally served Sunbird/Sunflower-14B
  anthropic   - Claude messages API
  gemini      - Google Generative Language API
  mock        - offline demo replies so the UI works with no key at all
"""
from __future__ import annotations

import json
import re
from typing import AsyncIterator

import httpx

from .config import SETTINGS


class ProviderError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class BaseProvider:
    name = "base"

    def __init__(self, *, api_key: str, model: str, base_url: str = "", timeout: float = 120.0):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout)

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        system: str,
        temperature: float = 0.6,
        max_tokens: int = 900,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    async def complete(self, messages, *, system, temperature=0.6, max_tokens=900) -> str:
        chunks = [c async for c in self.stream(messages, system=system, temperature=temperature, max_tokens=max_tokens)]
        return "".join(chunks)


def _sse_lines(text: str):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            yield line[5:].strip()


# --------------------------------------------------------------------------- #
# OpenAI-compatible (streaming SSE)
# --------------------------------------------------------------------------- #
class OpenAICompatible(BaseProvider):
    name = "openai"

    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages, system, temperature, max_tokens, stream=True):
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    async def stream(self, messages, *, system, temperature=0.6, max_tokens=900):
        payload = self._payload(messages, system, temperature, max_tokens, stream=True)
        async with self._client() as client:
            async with client.stream("POST", self._endpoint(), json=payload, headers=self._headers()) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise ProviderError(f"{self.name} HTTP {resp.status_code}: {body[:500]}")
                async for raw in resp.aiter_lines():
                    raw = raw.strip()
                    if not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece


class CompatibleProvider(OpenAICompatible):
    """vLLM / Ollama / OpenRouter / any OpenAI-shaped endpoint."""
    name = "compatible"


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicProvider(BaseProvider):
    name = "anthropic"

    async def stream(self, messages, *, system, temperature=0.6, max_tokens=900):
        payload = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "temperature": min(temperature, 1.0),
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        url = (self.base_url or "https://api.anthropic.com") + "/v1/messages"
        async with self._client() as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise ProviderError(f"anthropic HTTP {resp.status_code}: {body[:500]}")
                async for raw in resp.aiter_lines():
                    raw = raw.strip()
                    if not raw.startswith("data:"):
                        continue
                    try:
                        obj = json.loads(raw[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "content_block_delta":
                        piece = (obj.get("delta") or {}).get("text")
                        if piece:
                            yield piece


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
class GeminiProvider(BaseProvider):
    name = "gemini"

    async def stream(self, messages, *, system, temperature=0.6, max_tokens=900):
        # Gemini's streaming shape differs per version; a single call keeps this
        # provider simple and predictable.
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        async with self._client() as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                raise ProviderError(f"gemini HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            try:
                text = "".join(
                    part.get("text", "")
                    for part in data["candidates"][0]["content"]["parts"]
                )
            except (KeyError, IndexError):
                raise ProviderError("gemini returned an unexpected payload")
            if text:
                yield text


# --------------------------------------------------------------------------- #
# Offline demo provider
# --------------------------------------------------------------------------- #
class MockProvider(BaseProvider):
    """No key required. Deliberately simple: keyword-routed Luganda answers so
    the UI and the API can be tested end to end before you plug in a model."""

    name = "mock"

    RULES: list[tuple[str, str]] = [
        (
            r"\b(oli otya|muli mutya|osiibye|wasuze|musiibye|hello|hi|hey|oyogera)\b",
            "Nsibye bulungi, weebale. Osiibye otya ggwe? Nze Yogera — nsobola okwogera naawe mu Luganda. "
            "Oyagala twogere ku ki leero?",
        ),
        (
            r"\b(weebale|webale|thanks|thank you)\b",
            "Kale, weebale ggwe nnyo. Bwe wabaako ky'oyagala okumanya, nze wano.",
        ),
        (
            r"\b(erinnya|name)\b|\bnze\b",
            "Kale, tusanyuse okukulaba. Nze Yogera. Nnyamba ntya leero?",
        ),
        (
            r"\b(amateeka|ssente|banka|money|loan)\b",
            "Ku ssente ne banka, nkusaba weesigame ku mukozi wa banka yo. Nze nnyinza okukunnyonnyola "
            "ebigambo ebikozesebwa mu banka, naye sisobola kuwa bulagirizi ku by'amabanja.",
        ),
        (
            r"\b(obulwadde|eddwaaliro|omusawo|health|sick|fever)\b",
            "Bw'oba olwadde, genda mu ddwaaliro okulaba omusawo amangu. Nnyinza okukunnyonnyola ebigambo, "
            "naye sisobola kuwa bulagirizi bwa ddagala.",
        ),
        (
            r"\b(okuyiga|luganda language|oluganda|grammar|translate|kyusa)\b",
            "Oluganda lulungi era lwangu okutandika. Ka tutandike ku bigambo ebikozesebwa buli lunaku: "
            "\"oli otya?\", \"weebale\", \"kale\". Bw'oba oyagala, njogere ku kibuuze kyo mu butongole.",
        ),
    ]

    FALLBACK = (
        "Ntegedde. Kino kiseera kya ndaga (demo) — tewali API key eteekeddwa, "
        "kale sisobola kulowooza bulungi. Naye ntegedde nti onjogerako ku: \"{topic}\". "
        "Nkusaba oteeke API key mu fayiro .env oba mu Settings, ndyoke nkwatireko bulungi."
    )

    async def stream(self, messages, *, system, temperature=0.6, max_tokens=900):
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = last_user.lower()
        for pattern, reply in self.RULES:
            if re.search(pattern, text):
                yield reply
                return
        topic = " ".join(last_user.split())[:120] or "ebintu ebyo"
        yield self.FALLBACK.format(topic=topic)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_provider() -> BaseProvider:
    name = SETTINGS.resolved_provider
    common = dict(
        api_key=SETTINGS.resolved_api_key,
        model=SETTINGS.resolved_model,
        base_url=SETTINGS.resolved_base_url,
        timeout=SETTINGS.request_timeout,
    )
    if name == "openai":
        return OpenAICompatible(**common)
    if name == "anthropic":
        return AnthropicProvider(**common)
    if name == "gemini":
        return GeminiProvider(**common)
    if name == "compatible":
        return CompatibleProvider(**common)
    return MockProvider(**common)


def provider_status() -> dict:
    prov = get_provider()
    return {
        "provider": prov.name,
        "model": prov.model,
        "base_url": getattr(prov, "base_url", "") or "",
        "live": SETTINGS.is_live,
        "key_present": bool(SETTINGS.resolved_api_key) or prov.name == "mock",
        "demo_mode": not SETTINGS.is_live,
    }
