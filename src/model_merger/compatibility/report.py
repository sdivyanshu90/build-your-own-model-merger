"""Compatibility findings and their aggregation.

A finding has a severity (:class:`~model_merger.types.Severity`), a short machine
code, and a human message.  Whether a finding *blocks* the merge depends on both
its severity and the compatibility mode:

* ``FATAL``  -- always blocks (architectures fundamentally incompatible).
* ``ERROR``  -- blocks in ``strict`` mode; downgraded to a warning in ``permissive``.
* ``WARNING``/``INFO`` -- never block.

The merge is aborted *before any output is written* if a blocking finding exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..exceptions import CompatibilityError
from ..types import CompatibilityMode, Severity

__all__ = ["Finding", "CompatibilityReport"]


@dataclass(frozen=True)
class Finding:
    """A single compatibility observation."""

    severity: Severity
    code: str
    message: str

    def is_blocking(self, mode: CompatibilityMode) -> bool:
        if self.severity is Severity.FATAL:
            return True
        if self.severity is Severity.ERROR:
            return mode is CompatibilityMode.STRICT
        return False

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity.value, "code": self.code, "message": self.message}


@dataclass
class CompatibilityReport:
    """A collection of findings with convenience aggregates."""

    findings: list[Finding] = field(default_factory=list)

    def add(self, severity: Severity, code: str, message: str) -> None:
        self.findings.append(Finding(severity=severity, code=code, message=message))

    def extend(self, other: CompatibilityReport) -> None:
        self.findings.extend(other.findings)

    @property
    def max_severity(self) -> Severity:
        if not self.findings:
            return Severity.INFO
        return max((finding.severity for finding in self.findings), key=lambda s: s.rank)

    def blocking(self, mode: CompatibilityMode) -> list[Finding]:
        return [finding for finding in self.findings if finding.is_blocking(mode)]

    def is_compatible(self, mode: CompatibilityMode) -> bool:
        return not self.blocking(mode)

    def raise_if_incompatible(self, mode: CompatibilityMode) -> None:
        """Raise :class:`CompatibilityError` if any finding blocks under ``mode``."""

        blocking = self.blocking(mode)
        if blocking:
            details = "; ".join(f"[{finding.code}] {finding.message}" for finding in blocking)
            raise CompatibilityError(f"models are not compatible ({mode.value} mode): {details}")

    def summary(self, mode: CompatibilityMode) -> dict[str, object]:
        counts: dict[str, int] = {member.value: 0 for member in Severity}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return {
            "mode": mode.value,
            "compatible": self.is_compatible(mode),
            "max_severity": self.max_severity.value,
            "counts": counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }
