"""The gateway as the chaos runs serve it: the real gateway code, real Redis health and breakers,
and simulated providers (recorded answers at recorded latency) wrapped in fault injection.

It builds no real provider client and no verifier judge, so it needs no API key and makes no paid
call. Run one instance per port; both share Redis, so they share health, circuits and faults.

Usage:
  uv run uvicorn harness.chaos.app:app --host 127.0.0.1 --port 8000
  uv run uvicorn harness.chaos.app:app --host 127.0.0.1 --port 8001
Environment: LLM_GATEWAY_REDIS_URL (default redis://127.0.0.1:6379/0), LLM_GATEWAY_ROUTING,
LLM_GATEWAY_ADMIN_TOKEN (default chaos-local), CHAOS_LATENCY_SCALE (default 1.0), CHAOS_LOG."""
import os
from pathlib import Path

import redis.asyncio as redis

from harness.chaos.simulated import from_recording
from llm_gateway.api import create_app
from llm_gateway.breaker import CircuitBreaker
from llm_gateway.chaos import ChaosControl, ChaosProvider
from llm_gateway.health import HealthTracker
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import DEFAULT_CONFIG_PATH, Router

ADMIN_TOKEN = os.environ.get("LLM_GATEWAY_ADMIN_TOKEN", "chaos-local")

registry = load_registry()
router = Router(Path(os.environ.get("LLM_GATEWAY_ROUTING", DEFAULT_CONFIG_PATH)), set(registry))
store = redis.Redis.from_url(os.environ.get("LLM_GATEWAY_REDIS_URL", "redis://127.0.0.1:6379/0"))
control = ChaosControl(store)
simulated = from_recording(float(os.environ.get("CHAOS_LATENCY_SCALE", "1.0")))
health = HealthTracker(store)
app = create_app(
    registry,
    {p: ChaosProvider(simulated, p, control) for p in sorted({m.provider for m in registry.values()})},
    RequestLog(Path(os.environ.get("CHAOS_LOG", "var/chaos_requests.sqlite3"))),
    router=router, health=health, breaker=CircuitBreaker(store, health, config=lambda: router.config.breaker),
    chaos=control, admin_token=ADMIN_TOKEN)
