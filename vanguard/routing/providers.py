"""Provider catalog and routing configuration, persisted to vanguard_config.json.

The catalog ships with the OpenAI-compatible endpoints VanguardAI knows how to
speak to. A custom provider is any other OpenAI-compatible base URL.

On first use, if the config file is missing, enable every built-in provider
whose API key is present in the environment and derive a default model plus a
fallback order from those providers. After that the file is the source of
truth; new built-in providers are merged in without discarding saved settings.
"""

import json
import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Seeding reads the same .env the gateway starts with, so imports here need no
# ordering guarantee relative to app.core.config.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

CONFIG_PATH = Path(__file__).resolve().parents[2] / "vanguard_config.json"

ALIAS_NAMES = ("fast", "smart", "coding", "opus", "sonnet", "haiku")
CATEGORIES = ("cloud", "local", "custom")

# Models used to seed a default route. Only providers whose model id is stable
# and widely available belong here; everything else is picked from discovery.
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-20b",
    "openrouter": "deepseek/deepseek-chat",
}

BUILTIN_PROVIDERS = {
    "groq": {
        "name": "Groq",
        "category": "cloud",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "nvidia_nim": {
        "name": "NVIDIA NIM",
        "category": "cloud",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "openrouter": {
        "name": "OpenRouter",
        "category": "cloud",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "name": "DeepSeek",
        "category": "cloud",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "together": {
        "name": "Together AI",
        "category": "cloud",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "cerebras": {
        "name": "Cerebras",
        "category": "cloud",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
    },
    "sambanova": {
        "name": "SambaNova",
        "category": "cloud",
        "base_url": "https://api.sambanova.ai/v1",
        "api_key_env": "SAMBANOVA_API_KEY",
    },
    "fireworks": {
        "name": "Fireworks AI",
        "category": "cloud",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
    },
    "mistral": {
        "name": "Mistral",
        "category": "cloud",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
    },
    "xai": {
        "name": "xAI",
        "category": "cloud",
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
    },
    "gemini": {
        "name": "Google Gemini",
        "category": "cloud",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
    },
    "ollama": {
        "name": "Ollama",
        "category": "local",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
    },
    "lmstudio": {
        "name": "LM Studio",
        "category": "local",
        "base_url": "http://localhost:1234/v1",
        "api_key_env": None,
    },
    "llamacpp": {
        "name": "llama.cpp",
        "category": "local",
        "base_url": "http://localhost:8080/v1",
        "api_key_env": None,
    },
}

_lock = threading.RLock()
_config: dict | None = None


def _entry(provider_id: str, meta: dict) -> dict:
    api_key = os.getenv(meta["api_key_env"], "") if meta.get("api_key_env") else ""
    return {
        "id": provider_id,
        "name": meta["name"],
        "category": meta["category"],
        "base_url": meta["base_url"],
        "api_key": api_key,
        "enabled": bool(api_key),
    }


def _seed() -> dict:
    providers = {
        provider_id: _entry(provider_id, meta)
        for provider_id, meta in BUILTIN_PROVIDERS.items()
    }
    enabled_defaults = [
        (provider_id, model)
        for provider_id, model in DEFAULT_MODELS.items()
        if providers[provider_id]["enabled"]
    ]
    default_model = (
        f"{enabled_defaults[0][0]}/{enabled_defaults[0][1]}" if enabled_defaults else ""
    )
    fallback_models = [
        f"{provider_id}/{model}" for provider_id, model in enabled_defaults[1:]
    ]
    return {
        "providers": providers,
        "routing": {
            "default_model": default_model,
            "alias_overrides": {name: None for name in ALIAS_NAMES},
            "fallback_models": fallback_models,
        },
    }


def _merge(stored: dict) -> dict:
    """Overlay a stored config on the built-in catalog so new entries appear."""
    merged = _seed()
    stored_providers = stored.get("providers") or {}
    for provider_id, entry in stored_providers.items():
        if not isinstance(entry, dict):
            continue
        base = merged["providers"].get(provider_id, {})
        merged["providers"][provider_id] = {**base, **entry, "id": provider_id}

    stored_routing = stored.get("routing") or {}
    for key in ("default_model", "fallback_models"):
        if key in stored_routing:
            merged["routing"][key] = stored_routing[key]
    aliases = stored_routing.get("alias_overrides") or {}
    for name in ALIAS_NAMES:
        merged["routing"]["alias_overrides"][name] = aliases.get(name)
    return merged


def _read() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("could not read %s; using the built-in defaults", CONFIG_PATH)
        return {}


def _write(config: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("could not persist %s; the configuration is process-local", CONFIG_PATH)


def load_config() -> dict:
    """Return the effective config, seeding the file on first use."""
    global _config
    if _config is not None:
        return _config
    with _lock:
        if _config is not None:
            return _config
        if CONFIG_PATH.exists():
            _config = _merge(_read())
        else:
            _config = _seed()
            _write(_config)
        return _config


def save_config(config: dict) -> None:
    global _config
    with _lock:
        _config = config
        _write(config)


def reset_config() -> None:
    """Forget the cached config; the next read reloads or reseeds it."""
    global _config
    with _lock:
        _config = None


def list_providers() -> dict[str, dict]:
    return load_config()["providers"]


def get_provider(provider_id: str) -> dict | None:
    return list_providers().get(provider_id)


def upsert_provider(entry: dict) -> dict:
    config = load_config()
    config["providers"][entry["id"]] = entry
    save_config(config)
    return entry


def routing_config() -> dict:
    return load_config()["routing"]


def update_routing(routing: dict) -> dict:
    config = load_config()
    config["routing"] = routing
    save_config(config)
    return routing


def split_target(value: str | None, config: dict | None = None) -> tuple[str, str] | None:
    """Parse "<provider>/<model>", or None when the prefix is not a provider.

    Splitting on the first slash is what lets upstream ids that contain their
    own slash ("meta/llama-3.1-70b-instruct") pass through intact.
    """
    if not value or "/" not in value:
        return None
    provider_id, model = value.split("/", 1)
    known = (config or load_config())["providers"]
    if provider_id not in known or not model:
        return None
    return provider_id, model


def mask_key(api_key: str | None) -> str:
    """Return a display-only form; the raw key never leaves the process."""
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}...{api_key[-4:]}"
