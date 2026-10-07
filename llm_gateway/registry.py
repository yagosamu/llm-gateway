"""The model registry: which models the gateway can call, at what price, in which quality tier.
Loaded from models.yaml and validated once at startup, so a typo fails the boot, not a request."""
import datetime
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("models.yaml")
PROVIDERS = ("anthropic", "openai", "groq")
TIERS = ("high", "medium", "low")
EFFORTS = ("low", "medium", "high")


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    provider_model: str
    tier: str
    input_per_mtok: float
    output_per_mtok: float
    reasoning_effort: str
    price_source: str
    price_checked: datetime.date


def _parse(entry: dict, position: int) -> ModelConfig:
    fields = set(ModelConfig.__dataclass_fields__)
    missing, extra = sorted(fields - set(entry)), sorted(set(entry) - fields)
    if missing or extra:
        raise RegistryError(f"model #{position}: missing {missing}, unknown {extra}")
    model = ModelConfig(**entry)
    if model.provider not in PROVIDERS:
        raise RegistryError(f"{model.id}: provider {model.provider!r} is not one of {PROVIDERS}")
    if model.tier not in TIERS:
        raise RegistryError(f"{model.id}: tier {model.tier!r} is not one of {TIERS}")
    if model.reasoning_effort not in EFFORTS:
        raise RegistryError(f"{model.id}: reasoning_effort {model.reasoning_effort!r} is not one of {EFFORTS}")
    for name in ("input_per_mtok", "output_per_mtok"):
        value = getattr(model, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise RegistryError(f"{model.id}: {name} must be a non-negative number, got {value!r}")
    if not isinstance(model.price_checked, datetime.date):
        raise RegistryError(f"{model.id}: price_checked must be a YYYY-MM-DD date, got {model.price_checked!r}")
    if not model.price_source.startswith("https://"):
        raise RegistryError(f"{model.id}: price_source must be an https URL")
    return model


def load_registry(path: Path = DEFAULT_PATH) -> dict[str, ModelConfig]:
    """Model id -> ModelConfig, in file order. Raises RegistryError on a malformed or duplicate entry."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    registry: dict[str, ModelConfig] = {}
    for position, entry in enumerate(data["models"], start=1):
        model = _parse(entry, position)
        if model.id in registry:
            raise RegistryError(f"model id {model.id!r} appears more than once")
        registry[model.id] = model
    return registry


def cost_usd(model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    """What one call cost at the registry price. Output tokens include reasoning tokens."""
    return (input_tokens * model.input_per_mtok + output_tokens * model.output_per_mtok) / 1_000_000
