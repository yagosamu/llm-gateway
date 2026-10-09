"""The served app: the committed registry and routing config, real provider adapters, a request log
on disk, the sampled verifier with the project's judge, and, when LLM_GATEWAY_REDIS_URL is set, the
shared health window and per-provider circuit breakers in Redis.

Usage: uv run --env-file .env uvicorn llm_gateway.main:app --host 127.0.0.1 --port 8000"""
import os
from pathlib import Path

import redis.asyncio as redis

from harness.routing.judge import VerifierJudge
from llm_gateway.api import create_app
from llm_gateway.breaker import CircuitBreaker
from llm_gateway.health import HealthTracker
from llm_gateway.providers.factory import build_providers
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import DEFAULT_CONFIG_PATH, Router

registry = load_registry()
router = Router(Path(os.environ.get("LLM_GATEWAY_ROUTING", DEFAULT_CONFIG_PATH)), set(registry))
redis_url = os.environ.get("LLM_GATEWAY_REDIS_URL")
health = breaker = None
if redis_url:
    store = redis.Redis.from_url(redis_url)
    health = HealthTracker(store)
    breaker = CircuitBreaker(store, health, config=lambda: router.config.breaker)
app = create_app(registry, build_providers({m.provider for m in registry.values()}),
                 RequestLog(Path(os.environ.get("LLM_GATEWAY_LOG", "var/requests.sqlite3"))),
                 router=router, verifier_judge=VerifierJudge(), health=health, breaker=breaker)
