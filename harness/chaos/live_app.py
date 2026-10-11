"""The gateway for the live demo run: real provider adapters wrapped in fault injection, so a fault
can be put on a real provider for a while and the gateway's answer to it watched on the dashboard.

Unlike llm_gateway.main it has the chaos endpoint and no verifier (the run's spend stays bounded by
harness.chaos.live's preflight); unlike harness.chaos.app its calls are real and cost money. It is
started by harness.chaos.live, which checks the budget first, not by hand.
Environment: the provider keys (from .env), LLM_GATEWAY_REDIS_URL, LIVE_LOG, LLM_GATEWAY_ADMIN_TOKEN."""
import os
from pathlib import Path

import redis.asyncio as redis

from llm_gateway.api import create_app
from llm_gateway.breaker import CircuitBreaker
from llm_gateway.chaos import ChaosControl, ChaosProvider
from llm_gateway.health import HealthTracker
from llm_gateway.jobs import JobQueue
from llm_gateway.providers.factory import build_providers
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import DEFAULT_CONFIG_PATH, Router

ADMIN_TOKEN = os.environ.get("LLM_GATEWAY_ADMIN_TOKEN", "chaos-local")

registry = load_registry()
router = Router(Path(os.environ.get("LLM_GATEWAY_ROUTING", DEFAULT_CONFIG_PATH)), set(registry))
store = redis.Redis.from_url(os.environ.get("LLM_GATEWAY_REDIS_URL", "redis://127.0.0.1:6379/0"))
control = ChaosControl(store)
health = HealthTracker(store)
real = build_providers({m.provider for m in registry.values()})
app = create_app(
    registry, {p: ChaosProvider(adapter, p, control) for p, adapter in real.items()},
    RequestLog(Path(os.environ.get("LIVE_LOG", "var/live/requests.sqlite3"))),
    router=router, health=health, breaker=CircuitBreaker(store, health, config=lambda: router.config.breaker),
    chaos=control, admin_token=ADMIN_TOKEN, jobs=JobQueue(store))
