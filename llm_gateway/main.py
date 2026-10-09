"""The served app: the committed registry and routing config, real provider adapters, a request log
on disk and the sampled verifier with the project's judge.

Usage: uv run --env-file .env uvicorn llm_gateway.main:app --port 8000"""
import os
from pathlib import Path

from harness.routing.judge import VerifierJudge
from llm_gateway.api import create_app
from llm_gateway.providers.factory import build_providers
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import DEFAULT_CONFIG_PATH, Router

registry = load_registry()
app = create_app(registry, build_providers({m.provider for m in registry.values()}),
                 RequestLog(Path(os.environ.get("LLM_GATEWAY_LOG", "var/requests.sqlite3"))),
                 router=Router(Path(os.environ.get("LLM_GATEWAY_ROUTING", DEFAULT_CONFIG_PATH)), set(registry)),
                 verifier_judge=VerifierJudge())
