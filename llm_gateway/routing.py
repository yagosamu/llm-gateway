"""Cost routing: pick a tier for a request sent with model "auto", then the tier's model.

The routing config is a YAML file (routing.yaml) read again whenever it changes on disk, so the
tier map, the policy and the threshold change without a deploy. A config that fails to load or
validate is refused and the last good one stays in force; serving never stops on a bad edit.

Policies decide the cheapest candidate tier (low, then medium) whose estimated chance of an
acceptable answer reaches the threshold, and fall back to high:
- always_<tier>: a fixed tier.
- feature_table: the estimate is the acceptance rate of (X-Feature, tier) measured on judged traffic.
- classifier: the estimate is a logistic regression per tier over cheap prompt features.
Estimates for the last two come from a JSON file written by the training step, named in the config.

The offline evaluation calls the same decide() the gateway calls, so what is measured is what runs."""
import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("routing.yaml")
TIERS = ("low", "medium", "high")
CANDIDATE_ORDER = ("low", "medium")  # cheapest first; high is the fallback
POLICY_KINDS = ("always_low", "always_medium", "always_high", "feature_table", "classifier")
BREAKER_MODES = ("time_window", "count_window")
CONTEXT_LABEL = re.compile(r"\n\s*Context:\s*\n")
PATTERNS = {
    "asks_list": re.compile(r"\b(list|what are some|name (a few|some|several|\d+|two|three|four|five|ten))\b", re.I),
    "asks_compare": re.compile(r"\b(compare|comparison|difference|differ|versus|vs\.?|better than)\b", re.I),
    "asks_creative": re.compile(r"\b(write|poem|story|haiku|essay|imagine|creative|funny|song|letter)\b", re.I),
    "asks_summary": re.compile(r"\b(summari[sz]e|summary|key points|in brief|tl;?dr)\b", re.I),
}
log = logging.getLogger(__name__)


class RoutingConfigError(ValueError):
    pass


def user_text(messages: list[dict]) -> str:
    return "\n\n".join(m["content"] for m in messages if m["role"] == "user")


def prompt_features(messages: list[dict], feature: str) -> dict[str, float]:
    """The pre-registered classifier features. Keyword features read only the instruction, the text
    before any context block, so a long pasted document does not trigger them."""
    text = user_text(messages)
    instruction = CONTEXT_LABEL.split(text, maxsplit=1)[0]
    features = {"log_chars": math.log1p(len(text)), "has_context": float(bool(CONTEXT_LABEL.search(text)))}
    features |= {name: float(bool(p.search(instruction))) for name, p in PATTERNS.items()}
    features[f"feature={feature}"] = 1.0
    return features


@dataclass(frozen=True)
class Decision:
    tier: str
    reason: str


class FixedTier:
    def __init__(self, tier: str):
        self.tier = tier

    def decide(self, feature: str, messages: list[dict]) -> Decision:
        return Decision(self.tier, f"policy always_{self.tier}")


class FeatureTable:
    def __init__(self, estimates: dict[str, dict[str, float]], threshold: float):
        self.estimates, self.threshold = estimates, threshold

    def decide(self, feature: str, messages: list[dict]) -> Decision:
        rates = self.estimates.get(feature)
        if rates is None:
            return Decision("high", f"feature {feature!r} has no measured acceptance")
        for tier in CANDIDATE_ORDER:
            if rates.get(tier, 0.0) >= self.threshold:
                return Decision(tier, f"{feature} accepted {rates[tier]:.2f} on {tier} >= {self.threshold}")
        return Decision("high", f"{feature}: no cheaper tier reaches {self.threshold}")


class LogisticClassifier:
    """weights: {tier: {"intercept": b, "coef": {feature name: w}}}; a missing feature counts as 0."""

    def __init__(self, weights: dict[str, dict], threshold: float):
        self.weights, self.threshold = weights, threshold

    def probability(self, tier: str, features: dict[str, float]) -> float:
        w = self.weights[tier]
        z = w["intercept"] + sum(w["coef"].get(name, 0.0) * value for name, value in features.items())
        return 1.0 / (1.0 + math.exp(-z))

    def decide(self, feature: str, messages: list[dict]) -> Decision:
        features = prompt_features(messages, feature)
        for tier in CANDIDATE_ORDER:
            p = self.probability(tier, features)
            if p >= self.threshold:
                return Decision(tier, f"P(acceptable on {tier}) = {p:.2f} >= {self.threshold}")
        return Decision("high", f"no cheaper tier reaches {self.threshold}")


@dataclass(frozen=True)
class VerifierConfig:
    """Sampled after-the-fact verification of routed answers. sample_rate 0 turns it off."""
    sample_rate: float = 0.0
    daily_budget_usd: float = 0.0


@dataclass(frozen=True)
class BreakerConfig:
    """When a provider's circuit opens, how long it stays open, and how a half-open probe is leased.
    enabled false lets every call through and never moves a circuit, for comparison runs.

    mode time_window (v1): evaluate the shared 60 s health window, failures and the p95 of successes.
    mode count_window (v2): evaluate the last window_calls calls admitted since the circuit last
    moved; a call counts as slow once it has run slow_call_seconds, even before it finishes."""
    enabled: bool = True
    mode: str = "time_window"
    error_rate_threshold: float = 0.5
    min_calls: int = 10
    p95_budget_ms: float = 30_000.0
    open_seconds: float = 15.0
    probe_lease_seconds: float = 70.0  # longer than a provider call's 60 s timeout
    window_calls: int = 20
    slow_call_seconds: float = 30.0


@dataclass(frozen=True)
class FailoverConfig:
    """For "auto" requests: the models to try, in order, after the tier's own model, and the total time
    a request may spend across all its attempts."""
    deadline_seconds: float = 90.0
    fallbacks: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingConfig:
    version: str
    policy_name: str
    tier_map: dict[str, str]
    policy: object  # FixedTier | FeatureTable | LogisticClassifier
    verifier: VerifierConfig = VerifierConfig()
    breaker: BreakerConfig = BreakerConfig()
    failover: FailoverConfig = FailoverConfig()


def _failover_config(data: dict, tier_map: dict[str, str], model_ids: set[str]) -> FailoverConfig:
    section = data.get("failover") or {}
    if not isinstance(section, dict):
        raise RoutingConfigError("failover must be a mapping")
    deadline = section.get("deadline_seconds", FailoverConfig.deadline_seconds)
    if not isinstance(deadline, (int, float)) or deadline <= 0:
        raise RoutingConfigError("failover.deadline_seconds must be a positive number")
    fallbacks = {}
    for tier, models in (section.get("fallbacks") or {}).items():
        if tier not in TIERS:
            raise RoutingConfigError(f"failover.fallbacks has an unknown tier {tier!r}")
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise RoutingConfigError(f"failover.fallbacks.{tier} must be a list of model ids")
        unknown = sorted(m for m in models if m not in model_ids)
        if unknown:
            raise RoutingConfigError(f"failover.fallbacks.{tier} names models outside the registry: {unknown}")
        if tier_map[tier] in models:
            raise RoutingConfigError(f"failover.fallbacks.{tier} repeats the tier's own model")
        fallbacks[tier] = tuple(models)
    return FailoverConfig(float(deadline), fallbacks)


def _breaker_config(data: dict) -> BreakerConfig:
    section = data.get("breaker") or {}
    if not isinstance(section, dict):
        raise RoutingConfigError("breaker must be a mapping")
    unknown = sorted(set(section) - set(BreakerConfig.__dataclass_fields__))
    if unknown:
        raise RoutingConfigError(f"breaker has unknown keys {unknown}")
    values = {}
    for key, value in section.items():
        if key == "enabled":
            if not isinstance(value, bool):
                raise RoutingConfigError("breaker.enabled must be true or false")
            values[key] = value
        elif key == "mode":
            if value not in BREAKER_MODES:
                raise RoutingConfigError(f"breaker.mode must be one of {BREAKER_MODES}")
            values[key] = value
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RoutingConfigError(f"breaker.{key} must be a number")
        else:
            values[key] = int(value) if key in ("min_calls", "window_calls") else float(value)
    config = BreakerConfig(**values)
    if not 0 < config.error_rate_threshold <= 1:
        raise RoutingConfigError("breaker.error_rate_threshold must be in (0, 1]")
    if min(config.min_calls, config.p95_budget_ms, config.open_seconds, config.probe_lease_seconds,
           config.window_calls, config.slow_call_seconds) <= 0:
        raise RoutingConfigError("breaker numbers must be positive")
    if config.window_calls < config.min_calls:
        raise RoutingConfigError("breaker.window_calls must be at least min_calls")
    return config


def _verifier_config(data: dict) -> VerifierConfig:
    section = data.get("verifier") or {}
    if not isinstance(section, dict):
        raise RoutingConfigError("verifier must be a mapping")
    rate, budget = section.get("sample_rate", 0.0), section.get("daily_budget_usd", 0.0)
    if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
        raise RoutingConfigError("verifier.sample_rate must be a number in [0, 1]")
    if not isinstance(budget, (int, float)) or budget < 0:
        raise RoutingConfigError("verifier.daily_budget_usd must be a non-negative number")
    return VerifierConfig(float(rate), float(budget))


def load_config(path: Path, model_ids: set[str]) -> RoutingConfig:
    """Parse and validate routing.yaml. Raises RoutingConfigError naming the problem."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RoutingConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RoutingConfigError(f"{path} is not a mapping")
    tier_map, policy_name = data.get("tier_map"), data.get("policy")
    if not isinstance(tier_map, dict) or set(tier_map) != set(TIERS):
        raise RoutingConfigError(f"tier_map must map exactly {TIERS}")
    unknown = sorted(m for m in tier_map.values() if m not in model_ids)
    if unknown:
        raise RoutingConfigError(f"tier_map names models outside the registry: {unknown}")
    if policy_name not in POLICY_KINDS:
        raise RoutingConfigError(f"policy must be one of {POLICY_KINDS}, got {policy_name!r}")
    if policy_name.startswith("always_"):
        policy = FixedTier(policy_name.removeprefix("always_"))
    else:
        threshold = data.get("decision_threshold")
        if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise RoutingConfigError("decision_threshold must be a number in (0, 1]")
        estimates_path = path.parent / str(data.get("estimates", ""))
        try:
            estimates = json.loads(estimates_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RoutingConfigError(f"cannot read estimates {estimates_path}: {exc}") from exc
        if policy_name not in estimates:
            raise RoutingConfigError(f"{estimates_path} has no {policy_name!r} section")
        section = estimates[policy_name]
        policy = FeatureTable(section, threshold) if policy_name == "feature_table" else LogisticClassifier(section, threshold)
    return RoutingConfig(version=str(data.get("version", "")), policy_name=policy_name, tier_map=dict(tier_map),
                         policy=policy, verifier=_verifier_config(data), breaker=_breaker_config(data),
                         failover=_failover_config(data, tier_map, model_ids))


class Router:
    """Holds the routing config and reloads it when the file's modification time changes."""

    def __init__(self, path: Path, model_ids: set[str]):
        self.path, self.model_ids = path, model_ids
        self._lock = threading.Lock()
        self._mtime = os.stat(path).st_mtime_ns
        self.config = load_config(path, model_ids)  # a bad config at startup fails the boot
        self.rejected_reloads = 0

    def _maybe_reload(self) -> None:
        try:
            mtime = os.stat(self.path).st_mtime_ns
        except OSError:
            return
        if mtime == self._mtime:
            return
        with self._lock:
            if mtime == self._mtime:
                return
            self._mtime = mtime
            try:
                self.config = load_config(self.path, self.model_ids)
                log.info("routing config reloaded: version %s, policy %s", self.config.version, self.config.policy_name)
            except RoutingConfigError as exc:
                self.rejected_reloads += 1
                log.error("routing config rejected, keeping version %s: %s", self.config.version, exc)

    def route(self, feature: str, messages: list[dict]) -> tuple[str, Decision, RoutingConfig]:
        """(model id, decision, config in force) for one request."""
        self._maybe_reload()
        config = self.config
        decision = config.policy.decide(feature, messages)
        return config.tier_map[decision.tier], decision, config
