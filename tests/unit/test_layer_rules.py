"""Unit tests for tensor-key matching and layer-rule resolution."""

from __future__ import annotations

import pytest

from model_merger.exceptions import ConfigurationError
from model_merger.policies.layer_rules import CompiledRule, LayerRuleResolver
from model_merger.policies.matching import LayerRange, compile_predicate
from model_merger.utilities.patterns import extract_layer_index


def test_exact_match() -> None:
    predicate = compile_predicate(exact="model.norm.weight")
    assert predicate("model.norm.weight")
    assert not predicate("model.norm.bias")


def test_glob_match() -> None:
    predicate = compile_predicate(glob="*.bias")
    assert predicate("layer.0.bias")
    assert not predicate("layer.0.weight")


def test_regex_match() -> None:
    predicate = compile_predicate(regex=r".*embed.*")
    assert predicate("model.embed_tokens.weight")
    assert not predicate("model.norm.weight")


def test_suffix_match() -> None:
    predicate = compile_predicate(suffix=".weight")
    assert predicate("a.weight")
    assert not predicate("a.bias")


def test_layer_range_match() -> None:
    predicate = compile_predicate(layer_range=LayerRange(0, 1))
    assert predicate("model.layers.0.mlp.up_proj.weight")
    assert predicate("model.layers.1.mlp.up_proj.weight")
    assert not predicate("model.layers.5.mlp.up_proj.weight")
    assert not predicate("model.embed_tokens.weight")  # no layer index


def test_exclude() -> None:
    predicate = compile_predicate(suffix=".weight", exclude=["*embed*"])
    assert predicate("model.layers.0.mlp.up_proj.weight")
    assert not predicate("model.embed_tokens.weight")


def test_combined_and_semantics() -> None:
    predicate = compile_predicate(regex=r"layers\.\d+", suffix=".weight")
    assert predicate("model.layers.3.mlp.weight")
    assert not predicate("model.layers.3.mlp.bias")


def test_requires_at_least_one_condition() -> None:
    with pytest.raises(ConfigurationError):
        compile_predicate()


def test_invalid_regex() -> None:
    with pytest.raises(ConfigurationError):
        compile_predicate(regex="(")


def test_bad_layer_range() -> None:
    with pytest.raises(ConfigurationError):
        LayerRange(5, 1)


def test_resolver_first_match_and_priority() -> None:
    rule_low = CompiledRule("low", compile_predicate(suffix=".weight"), payload="low", priority=0)
    rule_high = CompiledRule("high", compile_predicate(glob="*embed*"), payload="high", priority=10)
    resolver = LayerRuleResolver([rule_low, rule_high], default_payload="default")
    # embed matches both; higher priority wins.
    assert resolver.resolve("model.embed_tokens.weight").payload == "high"
    # only the low rule matches this.
    assert resolver.resolve("model.norm.weight").payload == "low"
    # nothing matches -> default.
    assert resolver.resolve("model.norm.bias").is_default


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("model.layers.20.self_attn.q_proj.weight", 20),
        ("transformer.h.5.attn.weight", 5),
        ("encoder.block.3.layer.0.SelfAttention.q.weight", 3),
        ("bert.encoder.layer.11.output.dense.weight", 11),
        ("model.embed_tokens.weight", None),
    ],
)
def test_extract_layer_index(key: str, expected: int | None) -> None:
    assert extract_layer_index(key) == expected
