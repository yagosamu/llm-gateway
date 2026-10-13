"""The slice 4 evaluation, as pre-registered in harness/cache/prereg.py. Offline and free: it replays
the committed workload through the gateway (harness.cache.replay) and reads the committed verdicts and
human labels. Writes results/slice4.json and results/slice4.md; CI regenerates the JSON and fails when
it differs from the committed file.

Latency here is built from recorded latencies only (the model's answer and the embedding call, both
recorded once, live); the cache lookup itself is left out, because in this replay it runs on fakeredis
in process, which says nothing about Redis and changes from run to run.

Usage: uv run python -m harness.cache.evaluate"""
import asyncio
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from harness.cache import prereg
from harness.cache.label import HUMAN_DIR, verdicts_by_pair
from harness.cache.record import EMBEDDINGS_INDEX, EMBEDDINGS_PATH, INSTRUCTION_INDEX, INSTRUCTION_PATH, instruction_only
from harness.cache.replay import Replayer, pair_id
from harness.cache.workload import read_jsonl
from llm_gateway import embeddings

JSON_PATH, MD_PATH = Path("results/slice4.json"), Path("results/slice4.md")
SHARED_TENANT = "one-tenant"
BUST_QUERIES = 200


# --- statistics ----------------------------------------------------------------------------------

def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = prereg.ALPHA) -> tuple[float, float]:
    """Exact two-sided interval for k successes in n trials, by bisection on the binomial tails."""
    if n == 0:
        return 0.0, 1.0

    def solve(f) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > alpha / 2)
    return lower, upper


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


# --- false-hit rates -----------------------------------------------------------------------------

def false_hit_table(rows: list[dict], labels: dict[str, bool], half: str | None = None) -> dict:
    """Semantic hits (occurrences and distinct pairs) and how many were false, optionally in one half."""
    hits = [r for r in rows if r["layer"] == "semantic" and (half is None or r["half"] == half)]
    keys = [pair_id(r["text_id"], r["matched_text_id"]) for r in hits]
    labeled = [k for k in keys if k in labels]
    false = [k for k in labeled if not labels[k]]
    distinct, distinct_false = set(labeled), {k for k in labeled if not labels[k]}
    return {"hits": len(hits), "labeled": len(labeled), "false": len(false),
            "rate": len(false) / len(labeled) if labeled else None,
            "distinct_pairs": len(distinct), "distinct_false": len(distinct_false),
            "distinct_rate": len(distinct_false) / len(distinct) if distinct else None}


def choose_threshold(per_floor: dict[str, dict]) -> str | None:
    for floor in sorted(per_floor):
        tune = per_floor[floor]["tune"]
        if tune["labeled"] >= prereg.TUNE_MIN_LABELED and tune["rate"] <= prereg.TUNE_MAX_FALSE_HIT:
            return floor
    return None


# --- cost and latency ----------------------------------------------------------------------------

def economics(rows: list[dict], replayer: Replayer, semantic: bool) -> dict:
    """Cost and latency of serving the replayed rows, against sending every query to the model."""
    emb = {r["text_id"]: r for r in read_jsonl(EMBEDDINGS_INDEX)}
    no_cache_cost, cache_cost, no_cache_ms, cache_ms = 0.0, 0.0, [], []
    for r in rows:
        answer = replayer.matrix[(r["text_id"], "gpt-6-luna")]
        embed_cost = embeddings.cost_usd(emb[r["text_id"]]["tokens"]) if semantic and r["layer"] != "exact" else 0.0
        embed_ms = emb[r["text_id"]]["latency_ms"] if semantic and r["layer"] != "exact" else 0.0
        no_cache_cost += answer["cost_usd"]
        no_cache_ms.append(answer["latency_ms"])
        if r["layer"] == "miss":
            cache_cost += answer["cost_usd"] + embed_cost
            cache_ms.append(answer["latency_ms"] + embed_ms)
        else:
            cache_cost += embed_cost
            cache_ms.append(embed_ms)
    n = len(rows)
    layers = Counter(r["layer"] for r in rows)
    return {"queries": n, "exact_hits": layers["exact"], "semantic_hits": layers["semantic"], "misses": layers["miss"],
            "hit_rate": (layers["exact"] + layers["semantic"]) / n,
            "cost_per_1000_no_cache_usd": no_cache_cost / n * 1000, "cost_per_1000_cache_usd": cache_cost / n * 1000,
            "cost_change": cache_cost / no_cache_cost - 1,
            "p50_ms_no_cache": percentile(no_cache_ms, 50), "p50_ms_cache": percentile(cache_ms, 50),
            "p95_ms_no_cache": percentile(no_cache_ms, 95), "p95_ms_cache": percentile(cache_ms, 95)}


# --- counterfactuals -----------------------------------------------------------------------------

def instruction_only_matches(floor: float, replayer: Replayer) -> dict:
    """Pairs of workload texts with different context blocks whose instructions alone embed at or
    above the floor, against the same pairs embedded whole (as the cache does)."""
    full_index = {r["text_id"]: r["row"] for r in read_jsonl(EMBEDDINGS_INDEX)}
    instr_index = {r["text_id"]: r["row"] for r in read_jsonl(INSTRUCTION_INDEX)}
    full, instr = np.load(EMBEDDINGS_PATH), np.load(INSTRUCTION_PATH)
    ids = sorted(instr_index)
    contexts = {t: replayer.by_id[t]["text"].split("\n\nContext:\n", 1)[1] for t in ids}
    vi = np.stack([instr[instr_index[t]] for t in ids])
    vf = np.stack([full[full_index[t]] for t in ids])
    si, sf = vi @ vi.T, vf @ vf.T
    by_instruction = by_full = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if contexts[ids[a]] == contexts[ids[b]]:
                continue
            by_instruction += bool(si[a, b] >= floor)
            by_full += bool(sf[a, b] >= floor)
    return {"texts_with_context": len(ids), "pairs_with_different_context": len(ids) * (len(ids) - 1) // 2
            - sum(1 for a in range(len(ids)) for b in range(a + 1, len(ids)) if contexts[ids[a]] == contexts[ids[b]]),
            "matches_instruction_only": by_instruction, "matches_whole_message": by_full}


async def counterfactuals(floor: float, replayer: Replayer) -> dict:
    shared = [q | {"sent_as_tenant": SHARED_TENANT} for q in replayer.workload]
    rows = await replayer.run(floor, shared)
    cross = [r for r in rows if r["layer"] != "miss" and r["matched_tenant"] and r["matched_tenant"] != r["tenant"]]
    state: dict = {}
    await replayer.run(floor, app_state=state)
    stored_before = set(state["index"].owner)  # every entry stored under the old version
    first = replayer.workload[:BUST_QUERIES]
    after_change = await replayer.run(floor, first, app_state=state, headers={"X-Prompt-Version": "v2"})
    control = await replayer.run(floor, first, app_state=state)
    # A query repeated within the 200 can hit an answer stored minutes earlier under v2 itself; what
    # the change must prevent is a hit on an answer stored under the old version.
    hits_after = [r for r in after_change if r["layer"] != "miss"]
    return {
        "question_only_key": {"hits": sum(r["layer"] != "miss" for r in rows), "cross_tenant_hits": len(cross),
                              "cross_tenant_exact": sum(r["layer"] == "exact" for r in cross),
                              "cross_tenant_semantic": sum(r["layer"] == "semantic" for r in cross)},
        "system_prompt_change": {"queries": len(first),
                                 "hits_on_old_version_entries": sum(r["matched_entry"] in stored_before for r in hits_after),
                                 "hits_on_new_version_entries": sum(r["matched_entry"] not in stored_before for r in hits_after),
                                 "hits_control_same_version": sum(r["layer"] != "miss" for r in control)},
        "instruction_only_embedding": instruction_only_matches(floor, replayer),
    }


# --- agreement -----------------------------------------------------------------------------------

def human_agreement(labels: dict[str, bool]) -> dict:
    key = json.loads((HUMAN_DIR / "key.json").read_text(encoding="utf-8"))
    with (HUMAN_DIR / "answers.csv").open(encoding="utf-8") as f:
        answers = list(csv.DictReader(f))
    cells = Counter()
    disagreements = []
    for a in answers:
        pid = key[a["item"]]["pair_id"]
        human, judges = a["acceptable"] == "yes", labels[pid]
        cells[f"human_{'yes' if human else 'no'}_judges_{'yes' if judges else 'no'}"] += 1
        if human != judges:
            disagreements.append({"item": int(a["item"]), "pair_id": pid, "human": human, "judges": judges})
    return {"items": len(answers), "agree": sum(v for k, v in cells.items() if k.count("yes") in (0, 2)),
            "cells": dict(sorted(cells.items())), "disagreements": disagreements}


# --- the whole analysis ----------------------------------------------------------------------------

def labels_from_verdicts() -> tuple[dict[str, bool], dict]:
    verdicts = verdicts_by_pair()
    labels = {pid: all(v.values()) for pid, v in verdicts.items() if len(v) == len(prereg.JUDGES)}
    agree = sum(len(set(v.values())) == 1 for v in verdicts.values())
    return labels, {"pairs": len(verdicts), "judges_agree": agree}


async def build(directory: Path = Path("var/cache/replay")) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    replayer = Replayer(directory)
    labels, judge_stats = labels_from_verdicts()
    pairs = {p["pair_id"]: p for p in read_jsonl(Path("data/cache/pairs.jsonl"))}
    replays = {f"{t:.2f}": await replayer.run(t) for t in prereg.THRESHOLDS}
    exact_only = await replayer.run(None)
    per_floor = {f: {"tune": false_hit_table(rows, labels, "tune"), "test": false_hit_table(rows, labels, "test"),
                     "all": false_hit_table(rows, labels)} for f, rows in replays.items()}
    chosen = choose_threshold(per_floor)
    result = {
        "prereg": "harness/cache/prereg.py", "queries": len(replayer.workload),
        "judges": judge_stats, "labeled_pairs": len(labels), "false_pairs": sorted(
            [{"pair_id": k, "similarity": pairs[k]["similarity"], "same_intent": pairs[k]["same_intent"],
              "category": pairs[k]["category"], "half": pairs[k]["half"]} for k, ok in labels.items() if not ok],
            key=lambda p: -p["similarity"]),
        "per_floor": {f: {h: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in t.items()} for h, t in d.items()}
                      for f, d in per_floor.items()},
        "chosen_floor": chosen,
        "exact_only": economics(exact_only, replayer, semantic=False),
        "human": human_agreement(labels),
    }
    by_category = defaultdict(lambda: [0, 0])
    for k, ok in labels.items():
        by_category[pairs[k]["category"]][0] += 1
        by_category[pairs[k]["category"]][1] += not ok
    result["false_pairs_by_category"] = {c: {"pairs": n, "false": f} for c, (n, f) in sorted(by_category.items())}
    if chosen is not None:
        test = per_floor[chosen]["test"]
        lo, hi = clopper_pearson(test["false"], test["labeled"])
        dlo, dhi = clopper_pearson(test["distinct_false"], test["distinct_pairs"])
        result["claim"] = {"floor": chosen, "test_false_hits": test["false"], "test_semantic_hits": test["labeled"],
                           "rate": test["rate"], "ci95": [lo, hi], "below_1_percent": test["rate"] < prereg.CLAIM_MAX_FALSE_HIT,
                           "upper_below_1_percent": hi < prereg.CLAIM_MAX_FALSE_HIT,
                           "distinct": {"false": test["distinct_false"], "pairs": test["distinct_pairs"],
                                        "rate": test["distinct_rate"], "ci95": [dlo, dhi]}}
        result["at_chosen_floor"] = economics(replays[chosen], replayer, semantic=True)
        result["counterfactuals"] = await counterfactuals(float(chosen), replayer)
    return json.loads(json.dumps(result, default=float), parse_float=lambda s: round(float(s), 6))


# --- report --------------------------------------------------------------------------------------

def pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def render(r: dict) -> str:
    lines = ["# Slice 4 results: response cache", "",
             f"{r['queries']} queries replayed through the gateway; design in {r['prereg']}. Semantic hits are "
             "judged against the served model's fresh answer by two judges; a hit is false if either rejects it.", ""]
    if r["chosen_floor"] is None:
        lines += ["**No floor met the tune-half rule; the semantic layer is not shippable on this evidence.**", ""]
    else:
        c = r["claim"]
        lines += [f"**Chosen floor {c['floor']}** (lowest with at most {pct(prereg.TUNE_MAX_FALSE_HIT)} false hits over "
                  f"at least {prereg.TUNE_MIN_LABELED} labeled hits on the tune half).", "",
                  f"Test half at that floor: {c['test_false_hits']} false of {c['test_semantic_hits']} semantic hits, "
                  f"{pct(c['rate'])}, exact 95% interval [{pct(c['ci95'][0])}, {pct(c['ci95'][1])}]. "
                  f"Below 1%: {'yes' if c['below_1_percent'] else 'no'}; upper bound below 1%: "
                  f"{'yes' if c['upper_below_1_percent'] else 'no'}. Over distinct pairs: {c['distinct']['false']} of "
                  f"{c['distinct']['pairs']}, [{pct(c['distinct']['ci95'][0])}, {pct(c['distinct']['ci95'][1])}].", ""]
        e, x = r["at_chosen_floor"], r["exact_only"]
        lines += ["| | no cache | exact layer only | exact + semantic at the chosen floor |", "|---|---|---|---|",
                  f"| hit rate | - | {pct(x['hit_rate'])} | {pct(e['hit_rate'])} ({e['exact_hits']} exact, {e['semantic_hits']} semantic) |",
                  f"| cost per 1,000 queries, US$ | {e['cost_per_1000_no_cache_usd']:.4f} | {x['cost_per_1000_cache_usd']:.4f} "
                  f"({pct(x['cost_change'])}) | {e['cost_per_1000_cache_usd']:.4f} ({pct(e['cost_change'])}) |",
                  f"| p50 latency, ms | {e['p50_ms_no_cache']:.0f} | {x['p50_ms_cache']:.0f} | {e['p50_ms_cache']:.0f} |",
                  f"| p95 latency, ms | {e['p95_ms_no_cache']:.0f} | {x['p95_ms_cache']:.0f} | {e['p95_ms_cache']:.0f} |", ""]
    lines += ["## Precision by floor", "",
              "| floor | semantic hits | false | false-hit rate | tune: hits / false | test: hits / false |",
              "|---|---|---|---|---|---|"]
    for f, d in r["per_floor"].items():
        lines.append(f"| {f} | {d['all']['labeled']} | {d['all']['false']} | {pct(d['all']['rate'])} | "
                     f"{d['tune']['labeled']} / {d['tune']['false']} | {d['test']['labeled']} / {d['test']['false']} |")
    lines += ["", "## False hits", "", "| pair (query <- stored) | similarity | same intent | category | half |", "|---|---|---|---|---|"]
    lines += [f"| {p['pair_id']} | {p['similarity']:.3f} | {'yes' if p['same_intent'] else 'no'} | {p['category']} | {p['half']} |"
              for p in r["false_pairs"]]
    if "counterfactuals" in r:
        cf = r["counterfactuals"]
        q, s, io = cf["question_only_key"], cf["system_prompt_change"], cf["instruction_only_embedding"]
        lines += ["", "## Counterfactuals at the chosen floor", "",
                  f"- Key on the question alone (no tenant): {q['cross_tenant_hits']} of {q['hits']} hits would have served an "
                  f"answer stored for another tenant ({q['cross_tenant_exact']} exact, {q['cross_tenant_semantic']} semantic).",
                  f"- System prompt version changed for the first {s['queries']} queries: {s['hits_on_old_version_entries']} hits "
                  f"on answers stored under the old version ({s['hits_on_new_version_entries']} hits on answers those same "
                  f"queries stored under the new version, when one repeated another); the same queries again under the old "
                  f"version: {s['hits_control_same_version']} hits. The pre-registration said every one must miss, which "
                  "overlooked repeats within the 200; the check that matters is the first number.",
                  f"- Context left out of the embedding: among {io['pairs_with_different_context']} pairs of texts with different "
                  f"context blocks, {io['matches_instruction_only']} would match on the instruction alone, "
                  f"{io['matches_whole_message']} match with the context embedded (not judged)."]
    h = r["human"]
    lines += ["", "## Judges and the human check", "",
              f"The two judges agree on {r['judges']['judges_agree']} of {r['judges']['pairs']} pairs. Blind human labels agree "
              f"with the judges on {h['agree']} of {h['items']} sampled pairs: {h['cells']}.", "",
              "False pairs by category, over all labeled pairs: " + ", ".join(
                  f"{c} {v['false']}/{v['pairs']}" for c, v in r["false_pairs_by_category"].items()) + ".", "",
              "Latency is built from recorded latencies (the model call and the embedding call); the cache lookup is not "
              "included. Exact hits skip the embedding call; misses pay it."]
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    result = asyncio.run(build())
    JSON_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    MD_PATH.write_text(render(result), encoding="utf-8", newline="\n")
    print(render(result))


if __name__ == "__main__":
    main()
