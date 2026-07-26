"""Resolve which algorithm applies to each tensor key.

The resolver holds an ordered list of compiled rules plus a default payload.
For a given key it returns the payload of the first matching rule, or the default
if none match.  "First" is defined deterministically by ``(-priority, index)`` so
higher-priority rules win and ties fall back to declaration order.  This makes
rule precedence explicit and reproducible (documented in
``docs/configuration-reference.md``).

The payload is generic (``object``): the execution layer stores an algorithm
configuration there.  Keeping the resolver payload-agnostic avoids a dependency
cycle with the configuration models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .matching import MatchPredicate

__all__ = ["CompiledRule", "ResolvedRule", "LayerRuleResolver"]


@dataclass(frozen=True)
class CompiledRule:
    """A named match predicate with an attached payload and priority."""

    name: str
    predicate: MatchPredicate
    payload: object
    priority: int = 0


@dataclass(frozen=True)
class ResolvedRule:
    """The rule chosen for a tensor key (or the synthetic default)."""

    rule_name: str
    payload: object
    is_default: bool


class LayerRuleResolver:
    """Map tensor keys to algorithm payloads via ordered rules."""

    def __init__(
        self,
        rules: list[CompiledRule],
        default_payload: object,
        *,
        default_name: str = "default",
    ) -> None:
        # Stable sort by descending priority, then original order.
        self._rules = [
            rule
            for _, rule in sorted(enumerate(rules), key=lambda pair: (-pair[1].priority, pair[0]))
        ]
        self._default = ResolvedRule(default_name, default_payload, is_default=True)

    @property
    def rules(self) -> list[CompiledRule]:
        return list(self._rules)

    def resolve(self, key: str) -> ResolvedRule:
        """Return the resolved rule for ``key``."""

        for rule in self._rules:
            if rule.predicate(key):
                return ResolvedRule(rule.name, rule.payload, is_default=False)
        return self._default


@dataclass
class RuleUsage:
    """Accumulates how many tensors each rule handled (for reporting)."""

    counts: dict[str, int] = field(default_factory=dict)

    def record(self, rule_name: str) -> None:
        self.counts[rule_name] = self.counts.get(rule_name, 0) + 1
