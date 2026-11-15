"""Configuration: reads .env, picks a provider, exposes sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def _load_env() -> None:
    """Tiny .env loader (keeps the dependency list short)."""
    for name in (".env", ".env.local"):
        path = ROOT / name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_env()


# Sensible defaults per provider. Override with LLM_MODEL.
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-2.5-flash",
    "compatible": "Sunbird/Sunflower-14B",  # e.g. vLLM / Ollama / OpenRouter
    "mock": "yogera-demo",
}

KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "compatible": ("LLM_API_KEY", "OPENAI_API_KEY"),
    "mock": (),
}


@dataclass
class Settings:
    provider: str = "auto"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.6
    max_tokens: int = 900
    request_timeout: float = 120.0
    allow_english_fallback: bool = True
    use_few_shots: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def resolved_provider(self) -> str:
        """'auto' -> first provider that actually has a key, else mock."""
        if self.provider and self.provider != "auto":
            return self.provider
        for name in ("openai", "anthropic", "gemini", "compatible"):
            if any(os.getenv(k) for k in KEY_ENV[name]):
                return name
        return "mock"

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        prov = self.resolved_provider
        if prov == "compatible":
            return os.getenv("LLM_MODEL") or DEFAULT_MODELS["compatible"]
        return os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(prov, "gpt-4.1-mini")

    @property
    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        env = os.getenv("LLM_BASE_URL", "").strip()
        if env:
            return env.rstrip("/")
        if self.resolved_provider == "openai":
            return "https://api.openai.com/v1"
        if self.resolved_provider == "compatible":
            return "http://localhost:8000/v1"  # vLLM default
        return ""

    @property
    def resolved_api_key(self) -> str:
        for name in KEY_ENV.get(self.resolved_provider, ()):
            value = os.getenv(name, "").strip()
            if value:
                return value
        return ""

    @property
    def is_live(self) -> bool:
        """True when a real model will answer (not the offline demo)."""
        return self.resolved_provider != "mock"


def get_settings() -> Settings:
    return Settings(
        provider=os.getenv("LLM_PROVIDER", "auto").strip().lower() or "auto",
        model=os.getenv("LLM_MODEL", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "").strip(),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.6")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "900")),
        use_few_shots=os.getenv("YOGERA_FEW_SHOTS", "1").strip() not in {"0", "false", "no"},
    )


SETTINGS = get_settings()
