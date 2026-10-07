import pytest

from llm_gateway.registry import RegistryError, cost_usd, load_registry

VALID = """
  - id: m1
    provider: openai
    provider_model: m1
    tier: low
    input_per_mtok: 0.5
    output_per_mtok: 1.5
    reasoning_effort: medium
    price_source: https://example.com/pricing
    price_checked: 2026-10-07
"""


def write(tmp_path, entries: str):
    path = tmp_path / "models.yaml"
    path.write_text("models:" + entries, encoding="utf-8")
    return path


def test_the_committed_registry_loads_and_has_every_tier():
    registry = load_registry()
    assert {m.tier for m in registry.values()} == {"high", "medium", "low"}
    assert {m.provider for m in registry.values()} == {"anthropic", "openai", "groq"}


def test_a_duplicate_id_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="more than once"):
        load_registry(write(tmp_path, VALID + VALID))


@pytest.mark.parametrize("field, bad", [
    ("tier: low", "tier: cheap"),
    ("provider: openai", "provider: ollama"),
    ("input_per_mtok: 0.5", "input_per_mtok: -1"),
    ("price_checked: 2026-10-07", "price_checked: last week"),
    ("price_source: https://example.com/pricing", "price_source: example.com"),
    ("reasoning_effort: medium", "reasoning_effort: max"),
])
def test_a_bad_field_is_refused(tmp_path, field, bad):
    with pytest.raises(RegistryError):
        load_registry(write(tmp_path, VALID.replace(field, bad)))


def test_a_missing_or_unknown_field_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="missing"):
        load_registry(write(tmp_path, VALID.replace("    tier: low\n", "")))
    with pytest.raises(RegistryError, match="unknown"):
        load_registry(write(tmp_path, VALID + "    region: us\n"))


def test_cost_is_tokens_times_price_per_million(tmp_path):
    model = load_registry(write(tmp_path, VALID))["m1"]
    assert cost_usd(model, 1_000_000, 0) == pytest.approx(0.5)
    assert cost_usd(model, 2_000, 1_000) == pytest.approx((2_000 * 0.5 + 1_000 * 1.5) / 1_000_000)
