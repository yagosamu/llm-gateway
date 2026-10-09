"""Run every pre-registered chaos scenario under every configuration, REPS times, and record the raw
results under data/chaos.

The runner starts the two simulated gateway instances itself, pointed at a routing file it rewrites
for each configuration (the instances reload it on change). Before each run it clears the gateway's
keys in Redis (health, circuits, faults), so no run inherits another's state. During a run it sends
the open-loop traffic, injects the scenario's fault into openai at FAULT_START through the admin
endpoint, clears it at FAULT_END, and polls openai's circuit state. Nothing here calls a real model.

Usage: uv run python -m harness.chaos.scenarios [--reps N] [--only SCENARIO]"""
import argparse
import asyncio
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx2
import redis.asyncio as redis
import yaml

from harness.chaos import prereg, prereg_v2
from harness.chaos.analyze import MANIFEST_PATH, RUNS_DIR
from harness.chaos.load import generate
from llm_gateway.routing import DEFAULT_CONFIG_PATH

ROUTING_PATH = Path("var/chaos/routing.yaml")
REDIS_URL = os.environ.get("LLM_GATEWAY_REDIS_URL", "redis://127.0.0.1:6379/0")
ADMIN = {"X-Admin-Token": os.environ.get("LLM_GATEWAY_ADMIN_TOKEN", "chaos-local")}


@dataclass(frozen=True)
class Study:
    """Which faults, which configurations, how many repetitions, and where the raw runs go."""
    name: str
    design: str
    faults: dict
    configs: tuple
    reps: int
    runs_dir: Path
    manifest_path: Path


STUDIES = {
    "v1": Study("v1", "harness/chaos/prereg.py", prereg.SCENARIOS, prereg.CONFIGS, prereg.REPS, RUNS_DIR, MANIFEST_PATH),
    "v2": Study("v2", "harness/chaos/prereg_v2.py", prereg_v2.SCENARIOS, prereg_v2.CONFIGS, prereg_v2.REPS,
                Path("data/chaos/v2/runs"), Path("data/chaos/v2/manifest.json")),
}


def routing_for(config: str) -> dict:
    """The committed routing config, with failover and the breaker switched as the configuration says.
    breaker_v1 and breaker_v2 are failover_breaker with the breaker's mode set to time_window or
    count_window; nothing else differs between them."""
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["version"] = f"chaos-{config}"
    breaker = dict(data.get("breaker") or {})
    if config in prereg_v2.MODES:
        breaker |= {"enabled": True, "mode": prereg_v2.MODES[config]}
    else:
        breaker |= {"enabled": config == "failover_breaker"}
    data["breaker"] = breaker
    if config == "none":
        data["failover"] = dict(data.get("failover") or {}) | {"fallbacks": {}}
    return data


def write_routing(config: str) -> None:
    ROUTING_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROUTING_PATH.write_text(yaml.safe_dump(routing_for(config), sort_keys=False), encoding="utf-8")
    now = time.time_ns()
    os.utime(ROUTING_PATH, ns=(now, now))  # a fresh mtime, so the instances reload even within a second


def start_instances() -> list[subprocess.Popen]:
    procs = []
    for port in prereg.PORTS:
        env = os.environ | {"LLM_GATEWAY_ROUTING": str(ROUTING_PATH), "CHAOS_LOG": f"var/chaos/requests_{port}.sqlite3"}
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "harness.chaos.app:app", "--host", "127.0.0.1",
                                       "--port", str(port), "--log-level", "warning"], env=env))
    return procs


async def wait_ready(client: httpx2.AsyncClient) -> None:
    for _ in range(120):
        try:
            statuses = [(await client.get(f"http://127.0.0.1:{p}/v1/health")).status_code for p in prereg.PORTS]
            if all(s == 200 for s in statuses):
                return
        except httpx2.HTTPError:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError("the chaos instances did not come up")


async def run_once(study: Study, scenario: str, config: str, rep: int, client: httpx2.AsyncClient, store) -> dict:
    write_routing(config)
    keys = [k async for k in store.scan_iter("llmgw:*")]
    if keys:
        await store.delete(*keys)
    bases = [f"http://127.0.0.1:{p}" for p in prereg.PORTS]
    timeline, stop = [], asyncio.Event()
    started = time.perf_counter()

    async def poll():
        while not stop.is_set():
            try:
                body = (await client.get(f"{bases[0]}/v1/health")).json()
                timeline.append({"t": round(time.perf_counter() - started, 3),
                                 "state": body["providers"][prereg.FAULTY_PROVIDER]["circuit"]})
            except (httpx2.HTTPError, KeyError, ValueError):
                pass
            await asyncio.sleep(prereg.TIMELINE_INTERVAL)

    async def inject():
        fault = study.faults[scenario]
        await asyncio.sleep(prereg.FAULT_START - (time.perf_counter() - started))
        if fault is not None:  # a control scenario injects nothing
            await client.post(f"{bases[0]}/admin/chaos/{prereg.FAULTY_PROVIDER}", json=fault, headers=ADMIN)
        await asyncio.sleep(prereg.FAULT_END - (time.perf_counter() - started))
        await client.delete(f"{bases[1]}/admin/chaos", headers=ADMIN)

    poller, injector = asyncio.create_task(poll()), asyncio.create_task(inject())
    requests = await generate(bases, prereg.RATE, prereg.DURATION)
    await injector
    stop.set()
    await poller
    name = f"{scenario}__{config}__{rep}"
    folder = study.runs_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    for file, rows in (("requests.jsonl", requests), ("timeline.jsonl", timeline)):
        (folder / file).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return {"name": name, "scenario": scenario, "config": config, "rep": rep,
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


async def run_all(study: Study, reps: int, scenarios: list[str]) -> list[dict]:
    store = redis.Redis.from_url(REDIS_URL)
    write_routing(study.configs[-1])
    procs = start_instances()
    entries = []
    try:
        async with httpx2.AsyncClient(timeout=30) as client:
            await wait_ready(client)
            for rep in range(1, reps + 1):  # repetitions outermost, so slow drift spreads over every pair
                for scenario in scenarios:
                    for config in study.configs:
                        entry = await run_once(study, scenario, config, rep, client, store)
                        entries.append(entry)
                        print(f"{entry['name']} done", flush=True)
    finally:
        for proc in procs:
            proc.terminate()
        await store.aclose()
    return entries


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=list(STUDIES), default="v1")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--only", default=None, help="one scenario of the study")
    args = parser.parse_args()
    study = STUDIES[args.study]
    if args.only and args.only not in study.faults:
        parser.error(f"--only must be one of {list(study.faults)}")
    scenarios = [args.only] if args.only else list(study.faults)
    entries = asyncio.run(run_all(study, args.reps or study.reps, scenarios))
    # New runs replace earlier ones of the same name; the others, and any notes, are kept.
    manifest = (json.loads(study.manifest_path.read_text(encoding="utf-8")) if study.manifest_path.exists()
                else {"design": study.design, "rate_rps": prereg.RATE, "duration_s": prereg.DURATION})
    fresh = {e["name"] for e in entries}
    manifest["runs"] = [e for e in manifest.get("runs", []) if e["name"] not in fresh] + entries
    study.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    study.manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(entries)} runs -> {study.manifest_path} ({len(manifest['runs'])} in total)")


if __name__ == "__main__":
    main()
