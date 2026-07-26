"""Merge policies: precision, key matching, layer-rule resolution, non-float handling."""

from __future__ import annotations

from .layer_rules import CompiledRule, LayerRuleResolver, ResolvedRule, RuleUsage
from .matching import LayerRange, compile_predicate
from .non_float_tensors import NonFloatTensorPolicy, tensors_equal
from .precision import PrecisionPolicy

__all__ = [
    "PrecisionPolicy",
    "compile_predicate",
    "LayerRange",
    "CompiledRule",
    "ResolvedRule",
    "LayerRuleResolver",
    "RuleUsage",
    "NonFloatTensorPolicy",
    "tensors_equal",
]
