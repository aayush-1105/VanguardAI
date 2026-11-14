"""Administration API and the single-file control panel.

The endpoints mutate provider credentials and routing, so they belong behind
whatever authentication the deployment puts in front of the gateway.
"""

import re
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from vanguard.routing import discovery, providers

router = APIRouter()
ui_router = APIRouter()

UI_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"
PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class ProviderUpsert(BaseModel):
    id: str
    name: str | None = None
    category: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    enabled: bool | None = None


class RoutingUpdate(BaseModel):
    default_model: str
    alias_overrides: dict[str, str | None] = {}
    fallback_models: list[str] = []


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": kind, "message": message}})


def _public(provider_id: str, entry: dict, catalog: dict) -> dict:
    status = (catalog.get("providers") or {}).get(provider_id, {})
    return {
        "id": provider_id,
        "name": entry.get("name", provider_id),
        "category": entry.get("category", "custom"),
        "base_url": entry.get("base_url", ""),
        "enabled": bool(entry.get("enabled")),
        "has_api_key": bool(entry.get("api_key")),
        "api_key": providers.mask_key(entry.get("api_key")),
        "model_count": status.get("model_count", 0),
        "health": status.get("status", "unknown"),
    }


@router.get("/admin/providers")
async def list_providers():
    catalog = discovery.load_catalog() or {}
    entries = [
        _public(provider_id, entry, catalog)
        for provider_id, entry in providers.list_providers().items()
    ]
    return {"providers": entries}


@router.post("/admin/providers")
async def upsert_provider(payload: ProviderUpsert):
    if not PROVIDER_ID.match(payload.id):
        return _error(400, "invalid_provider", "Provider id must be a lowercase slug.")
    if payload.category is not None and payload.category not in providers.CATEGORIES:
        return _error(400, "invalid_provider", "Category must be cloud, local, or custom.")

    existing = providers.get_provider(payload.id) or {}
    entry = {**existing, "id": payload.id}
    for field in ("name", "category", "base_url", "enabled"):
        value = getattr(payload, field)
        if value is not None:
            entry[field] = value
    # Omitted means "keep the stored key"; an empty string clears it.
    if payload.api_key is not None:
        entry["api_key"] = payload.api_key

    entry.setdefault("name", payload.id)
    entry.setdefault("category", "custom")
    entry.setdefault("base_url", "")
    entry.setdefault("api_key", "")
    entry.setdefault("enabled", False)

    if not str(entry["base_url"]).startswith(("http://", "https://")):
        return _error(400, "invalid_provider", "Base URL must start with http:// or https://.")

    providers.upsert_provider(entry)
    discovery.reset_catalog()
    return _public(payload.id, entry, {})


@router.post("/admin/models/refresh")
async def refresh_models():
    return await discovery.refresh_models()


@router.get("/admin/routing")
async def get_routing():
    return providers.routing_config()


@router.post("/admin/routing")
async def set_routing(payload: RoutingUpdate):
    if providers.split_target(payload.default_model) is None:
        return _error(
            400, "invalid_routing", "Default model must be <provider>/<model>."
        )
    unknown_aliases = set(payload.alias_overrides) - set(providers.ALIAS_NAMES)
    if unknown_aliases:
        return _error(
            400,
            "invalid_routing",
            f"Unknown alias: {', '.join(sorted(unknown_aliases))}.",
        )
    for name, target in payload.alias_overrides.items():
        if target is not None and providers.split_target(target) is None:
            return _error(400, "invalid_routing", f"Alias {name} is not a valid target.")
    for target in payload.fallback_models:
        if providers.split_target(target) is None:
            return _error(400, "invalid_routing", f"{target} is not a valid target.")

    routing = {
        "default_model": payload.default_model,
        "alias_overrides": {
            name: payload.alias_overrides.get(name) for name in providers.ALIAS_NAMES
        },
        "fallback_models": payload.fallback_models,
    }
    return providers.update_routing(routing)


@router.get("/models")
async def list_models():
    catalog = discovery.load_catalog() or {"data": []}
    return {"object": "list", "data": catalog.get("data", [])}


def _html() -> HTMLResponse | JSONResponse:
    if not UI_PATH.exists():
        return _error(404, "ui_missing", "The control panel is not installed.")
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@ui_router.get("/ui", include_in_schema=False)
@ui_router.get("/ui/", include_in_schema=False)
async def control_panel():
    return _html()
