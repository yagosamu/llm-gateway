import asyncio
import random

import fakeredis
import httpx2
import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import create_app
from llm_gateway.jobs import DEAD, DELAYED, Backoff, IdempotencyConflict, JobQueue
from llm_gateway.providers.base import ProviderError, ProviderResult
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.worker import Worker

REGISTRY = load_registry()
BODY = {"model": "gpt-6-luna", "messages": [{"role": "user", "content": "Name a river."}]}


class Clock:
    def __init__(self, now=4_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def queue(server=None, clock=None):
    return JobQueue(fakeredis.FakeAsyncRedis(server=server or fakeredis.FakeServer()), clock or Clock())


def run(coro):
    return asyncio.run(coro)


def test_a_submitted_job_is_queued_once_per_idempotency_key():
    q = queue()

    async def scenario():
        await q.ensure_group()
        first, created = await q.submit(BODY, "t1", "open_qa", "r1", "key-1")
        again, created_again = await q.submit(BODY, "t1", "open_qa", "r2", "key-1")
        other_tenant, _ = await q.submit(BODY, "t2", "open_qa", "r3", "key-1")
        return first, created, again, created_again, other_tenant, await q.depth(), await q.get(first)

    first, created, again, created_again, other, depth, job = run(scenario())
    assert created and not created_again and first == again
    assert other != first  # keys are per tenant
    assert depth["stream"] == 2
    assert (job["status"], job["tenant"], job["body"]) == ("queued", "t1", BODY)


def test_reusing_a_key_for_another_request_is_refused():
    q = queue()

    async def scenario():
        await q.submit(BODY, "t1", "open_qa", "r1", "key-1")
        await q.submit({**BODY, "model": "auto"}, "t1", "open_qa", "r2", "key-1")

    with pytest.raises(IdempotencyConflict):
        run(scenario())


def test_full_jitter_stays_inside_the_exponential_cap():
    backoff, rng = Backoff(base_seconds=1, cap_seconds=8, max_attempts=10), random.Random(1)
    for attempt in range(1, 8):
        bound = min(8, 2 ** (attempt - 1))
        delays = [backoff.delay(attempt, rng) for _ in range(200)]
        assert all(0 <= d <= bound for d in delays) and max(delays) > 0.8 * bound


def test_a_retry_waits_for_its_time_and_is_moved_back_exactly_once():
    server, clock = fakeredis.FakeServer(), Clock()
    a, b = queue(server, clock), queue(server, clock)

    async def scenario():
        await a.ensure_group()
        job_id, _ = await a.submit(BODY, "t1", "f", "r", None)
        [(mid, _)] = await a.read("w1")
        await a.retry_later(job_id, mid, 5.0, "upstream_server_error")
        early = await a.move_due()
        clock.now += 5
        moved = [await a.move_due(), await b.move_due()]
        return early, moved, await a.depth(), (await a.get(job_id))["status"]

    early, moved, depth, status = run(scenario())
    assert early == 0 and sorted(moved) == [0, 1]
    assert depth == {"stream": 1, "delayed": 0, "dead": 0} and status == "queued"


def test_an_unacknowledged_entry_is_reclaimed_by_another_worker():
    q = queue()

    async def scenario():
        await q.ensure_group()
        job_id, _ = await q.submit(BODY, "t1", "f", "r", None)
        await q.read("dead-worker")  # delivered, never acknowledged
        return job_id, await q.read("w2", block_ms=10), await q.reclaim("w2", min_idle_seconds=0)

    job_id, fresh, reclaimed = run(scenario())
    assert fresh == [] and [j for _, j in reclaimed] == [job_id]


class Provider:
    def __init__(self, fail_times=0, kind="server_error"):
        self.fail_times, self.kind, self.calls = fail_times, kind, 0

    async def complete(self, model, request):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError(self.kind, model.provider, "down")
        return ProviderResult("The Nile.", 10, 5, "stop")


class World:
    """A gateway app and workers sharing one fake Redis; workers reach the app in process."""

    def __init__(self, provider, max_attempts=4):
        self.server, self.clock = fakeredis.FakeServer(), Clock()
        self.provider, self.log = provider, RequestLog(":memory:")
        self.queue = JobQueue(fakeredis.FakeAsyncRedis(server=self.server), self.clock)
        self.app = create_app(REGISTRY, {p: provider for p in ("openai", "anthropic", "groq")}, self.log,
                              jobs=JobQueue(fakeredis.FakeAsyncRedis(server=self.server), self.clock))
        self.backoff = Backoff(base_seconds=1, cap_seconds=4, max_attempts=max_attempts)

    def worker(self, client, name="w1"):
        return Worker(self.queue, client, name, self.backoff, random.Random(0))

    async def submit(self, client, key=None, tenant="t1"):
        headers = {"X-Tenant-Id": tenant, "X-Feature": "open_qa", "X-Request-Id": "client-1", "X-Deferrable": "true"}
        if key:
            headers["Idempotency-Key"] = key
        return (await client.post("/v1/chat/completions", json=BODY, headers=headers)).json()

    async def drain(self, worker, passes=20):
        for _ in range(passes):
            await worker.run_once(block_ms=1)
            self.clock.now += 10  # let any backoff elapse


def client_for(app):
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://gw")


def test_a_worker_answers_a_deferred_job_through_the_gateway():
    world = World(Provider())

    async def scenario():
        async with client_for(world.app) as client:
            await world.queue.ensure_group()
            submitted = await world.submit(client)
            await world.drain(world.worker(client))
            job = (await client.get(f"/v1/jobs/{submitted['id']}", headers={"X-Tenant-Id": "t1"})).json()
            return submitted, job

    submitted, job = run(scenario())
    assert submitted["status"] == "queued" and job["status"] == "done" and job["attempts"] == 1
    assert job["result"]["choices"][0]["message"]["content"] == "The Nile."
    rows = world.log.rows()
    assert [r["outcome"] for r in rows] == ["queued", "ok"]
    assert rows[1]["request_id"] == f"job:{submitted['id']}:1"


def test_a_job_outlives_an_outage_through_backoff_and_retries():
    world = World(Provider(fail_times=2))

    async def scenario():
        async with client_for(world.app) as client:
            await world.queue.ensure_group()
            job_id = (await world.submit(client))["id"]
            await world.drain(world.worker(client))
            return await world.queue.get(job_id)

    job = run(scenario())
    assert (job["status"], job["attempts"]) == ("done", 3)
    assert world.provider.calls == 3


def test_a_job_that_keeps_failing_ends_dead_after_the_attempt_budget():
    world = World(Provider(fail_times=99), max_attempts=3)

    async def scenario():
        async with client_for(world.app) as client:
            await world.queue.ensure_group()
            job_id = (await world.submit(client))["id"]
            await world.drain(world.worker(client))
            return await world.queue.get(job_id), await world.queue.redis.lrange(DEAD, 0, -1)

    job, dead = run(scenario())
    assert (job["status"], job["attempts"], job["last_error"]) == ("failed", 3, "upstream_server_error")
    assert dead == [job["id"].encode()]


def test_a_bad_request_is_not_retried():
    world = World(Provider(fail_times=99, kind="bad_request"))

    async def scenario():
        async with client_for(world.app) as client:
            await world.queue.ensure_group()
            job_id = (await world.submit(client))["id"]
            await world.drain(world.worker(client))
            return await world.queue.get(job_id)

    job = run(scenario())
    assert (job["status"], job["attempts"]) == ("failed", 1)


def test_a_dead_workers_job_is_finished_by_another_with_one_call():
    world = World(Provider())

    async def scenario():
        async with client_for(world.app) as client:
            await world.queue.ensure_group()
            job_id = (await world.submit(client))["id"]
            await world.queue.read("dead-worker")  # took it, then died before calling the gateway
            survivor = world.worker(client, "w2")
            survivor.reclaim_idle_seconds = 0
            await world.drain(survivor, passes=2)
            return await world.queue.get(job_id)

    job = run(scenario())
    assert job["status"] == "done" and world.provider.calls == 1


def test_a_worker_dying_after_the_call_and_before_storing_causes_one_duplicate_call():
    """The documented limit: at-least-once execution. The answer is stored once, but the model was
    called twice, and the request log shows both calls."""
    world = World(Provider())

    async def scenario():
        async with client_for(world.app) as client:
            await world.queue.ensure_group()
            job_id = (await world.submit(client))["id"]
            [(mid, _)] = await world.queue.read("dying-worker")
            attempt = await world.queue.start_attempt(job_id)
            await client.post("/v1/chat/completions", json=BODY, headers={  # the call goes through ...
                "X-Tenant-Id": "t1", "X-Feature": "open_qa", "X-Request-Id": f"job:{job_id}:{attempt}"})
            # ... and the worker dies before queue.finish
            survivor = world.worker(client, "w2")
            survivor.reclaim_idle_seconds = 0
            await world.drain(survivor, passes=2)
            return await world.queue.get(job_id)

    job = run(scenario())
    assert job["status"] == "done" and world.provider.calls == 2
    assert sum(r["outcome"] == "ok" for r in world.log.rows()) == 2


def test_a_finished_job_delivered_again_is_only_acknowledged():
    world = World(Provider())

    async def scenario():
        async with client_for(world.app) as client:
            await world.queue.ensure_group()
            job_id = (await world.submit(client))["id"]
            await world.drain(world.worker(client), passes=1)
            await world.queue.redis.xadd("llmgw:jobs", {"job": job_id})  # a stray redelivery
            await world.drain(world.worker(client), passes=1)
            return await world.queue.depth()

    depth = run(scenario())
    assert depth["stream"] == 0 and world.provider.calls == 1


def api(jobs=True):
    store = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())
    app = create_app(REGISTRY, {p: Provider() for p in ("openai", "anthropic", "groq")},
                     jobs=JobQueue(store) if jobs else None)
    return TestClient(app)


def defer(client, key=None, tenant="t1", body=BODY):
    headers = {"X-Tenant-Id": tenant, "X-Feature": "open_qa", "X-Request-Id": "r", "X-Deferrable": "true"}
    if key is not None:
        headers["Idempotency-Key"] = key
    return client.post("/v1/chat/completions", json=body, headers=headers)


def test_the_api_queues_deduplicates_and_isolates_tenants():
    client = api()
    first = defer(client, key="order-7")
    assert first.status_code == 202 and first.json()["deduplicated"] is False
    again = defer(client, key="order-7")
    assert again.json()["id"] == first.json()["id"] and again.json()["deduplicated"] is True
    job_id = first.json()["id"]
    assert client.get(f"/v1/jobs/{job_id}", headers={"X-Tenant-Id": "t1"}).json()["status"] == "queued"
    for tenant in ("t2", ""):
        assert client.get(f"/v1/jobs/{job_id}", headers={"X-Tenant-Id": tenant}).status_code == 404
    assert client.get("/v1/jobs/nope", headers={"X-Tenant-Id": "t1"}).status_code == 404


@pytest.mark.parametrize("key, body, code", [
    ("order-7", {**BODY, "messages": [{"role": "user", "content": "Name a lake."}]}, "idempotency_key_reused"),
    ("has space", BODY, "invalid_idempotency_key"),
])
def test_bad_idempotency_keys_are_refused(key, body, code):
    client = api()
    defer(client, key="order-7")
    response = defer(client, key=key, body=body)
    assert response.json()["error"]["code"] == code


def test_a_deferred_request_is_validated_before_it_is_queued():
    client = api()
    response = client.post("/v1/chat/completions", json=BODY, headers={"X-Deferrable": "true"})
    assert response.status_code == 400 and response.json()["error"]["code"] == "missing_metadata"
    assert defer(client, body={**BODY, "model": "gpt-9"}).status_code == 404


def test_without_a_queue_deferral_is_refused():
    response = defer(api(jobs=False))
    assert response.status_code == 400 and response.json()["error"]["code"] == "deferral_disabled"


def test_queue_depth_is_exposed():
    client = api()
    defer(client)
    assert 'llm_gateway_jobs{state="stream"} 1.0' in client.get("/metrics").text
