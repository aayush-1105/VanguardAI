"""Resolve an incoming model id into an ordered provider fallback chain.

Resolution order:
  1. an explicit "<provider>/<model>", when the prefix names a provider;
  2. an alias override ("fast", "smart", "coding", "opus", "sonnet", "haiku");
  3. the configured default model, used for "auto", a missing model, or any
     string that matches neither of the above.

Every chain starts with the resolved primary and continues with the configured
fallback models. A provider that is disabled is skipped for automatic
(fallback) selection, but an explicit request for it is still honoured.
"""

from vanguard.routing import providers


def _default(routing: dict) -> tuple[str, str] | None:
    return providers.split_target(routing.get("default_model"))


def resolve(model: str | None) -> tuple[str, str] | None:
    """Return the primary (provider_id, upstream_model) for a request."""
    routing = providers.routing_config()

    if model and model != "auto":
        explicit = providers.split_target(model)
        if explicit is not None:
            return explicit
        alias = (routing.get("alias_overrides") or {}).get(model)
        if alias:
            return providers.split_target(alias) or _default(routing)
    return _default(routing)


def candidates(model: str | None) -> list[tuple[str, str]]:
    """Return the deduplicated chain of (provider_id, upstream_model) hops."""
    routing = providers.routing_config()
    chain: list[tuple[str, str]] = []
    primary = resolve(model)
    if primary is not None:
        chain.append(primary)

    known = providers.list_providers()
    for entry in routing.get("fallback_models") or []:
        target = providers.split_target(entry)
        if target is None:
            continue
        provider = known.get(target[0]) or {}
        # Fallbacks are automatic, so a disabled provider stays out of the chain.
        if not provider.get("enabled"):
            continue
        chain.append(target)

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for hop in chain:
        if hop not in seen:
            seen.add(hop)
            unique.append(hop)
    return unique
