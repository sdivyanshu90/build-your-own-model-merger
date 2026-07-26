"""Compile tensor-key match conditions into predicates.

A layer rule selects tensors by any combination of: exact name, glob, regex,
name suffix, and transformer layer-number range, minus an optional set of exclude
globs.  All *positive* conditions present must hold (AND), and no exclude glob may
match.  At least one positive condition is required so a rule cannot match
everything by accident.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..exceptions import ConfigurationError
from ..utilities.patterns import extract_layer_index, glob_to_regex

__all__ = ["LayerRange", "compile_predicate"]

MatchPredicate = Callable[[str], bool]


@dataclass(frozen=True)
class LayerRange:
    """Inclusive transformer layer-index range ``[start, end]``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ConfigurationError(f"layer_range start ({self.start}) exceeds end ({self.end})")

    def contains(self, index: int | None) -> bool:
        return index is not None and self.start <= index <= self.end


def compile_predicate(
    *,
    exact: str | None = None,
    glob: str | None = None,
    regex: str | None = None,
    suffix: str | None = None,
    layer_range: LayerRange | None = None,
    exclude: Sequence[str] = (),
) -> MatchPredicate:
    """Build a ``key -> bool`` predicate from match conditions.

    Raises:
        ConfigurationError: if no positive condition is supplied, or a regex is
            invalid.
    """

    positive = [exact, glob, regex, suffix, layer_range]
    if not any(condition is not None for condition in positive):
        raise ConfigurationError(
            "a match rule needs at least one of: exact, glob, regex, suffix, layer_range"
        )

    glob_re = glob_to_regex(glob) if glob is not None else None
    try:
        regex_re = re.compile(regex) if regex is not None else None
    except re.error as error:
        raise ConfigurationError(f"invalid match regex {regex!r}: {error}") from error
    exclude_res = [glob_to_regex(pattern) for pattern in exclude]

    def predicate(key: str) -> bool:
        if exact is not None and key != exact:
            return False
        if glob_re is not None and glob_re.match(key) is None:
            return False
        if regex_re is not None and regex_re.search(key) is None:
            return False
        if suffix is not None and not key.endswith(suffix):
            return False
        if layer_range is not None and not layer_range.contains(extract_layer_index(key)):
            return False
        return all(pattern.match(key) is None for pattern in exclude_res)

    return predicate
