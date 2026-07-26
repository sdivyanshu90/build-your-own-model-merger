"""Serialize plans and reports to JSON and Markdown.

JSON is the canonical, machine-readable form (stable keys, sorted where order is
irrelevant).  Markdown is an optional human-friendly rendering.  Neither ever
includes secrets: only paths, hashes, counts, and configuration values that have
already been through the redaction path make it into a report.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import MergePlan, MergeReport

__all__ = [
    "report_to_json",
    "write_report_json",
    "report_to_markdown",
    "write_report_markdown",
    "plan_to_json",
    "plan_to_markdown",
]


def report_to_json(report: MergeReport, *, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent, sort_keys=True)


def write_report_json(report: MergeReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(report_to_json(report) + "\n", encoding="utf-8")
    return destination


def plan_to_json(plan: MergePlan, *, include_tensor_entries: bool = False, indent: int = 2) -> str:
    return json.dumps(
        plan.to_dict(include_tensor_entries=include_tensor_entries), indent=indent, sort_keys=True
    )


def _rule_lines(per_rule_counts: dict[str, int]) -> list[str]:
    return [f"- `{name}`: {count} tensors" for name, count in sorted(per_rule_counts.items())]


def plan_to_markdown(plan: MergePlan) -> str:
    lines = [
        f"# Merge plan `{plan.run_id}`",
        "",
        f"- **Algorithm**: {plan.algorithm}",
        f"- **Models**: {', '.join(model.name for model in plan.models)}",
        f"- **Output**: `{plan.output_path}` ({plan.output_format})",
        f"- **Estimated size**: {plan.estimated_output_bytes:,} bytes "
        f"in {plan.shard_count} shard(s)",
        f"- **Tensors**: {plan.tensor_count} ({len(plan.non_float_keys)} non-float)",
        f"- **Unsafe loading required**: {plan.requires_unsafe_loading}",
        "",
        "## Per-rule tensor counts",
        *_rule_lines(plan.per_rule_counts),
    ]
    if plan.warnings:
        lines += ["", "## Warnings", *[f"- {warning}" for warning in plan.warnings]]
    return "\n".join(lines) + "\n"


def report_to_markdown(report: MergeReport) -> str:
    lines = [
        f"# Merge report `{report.run_id}`",
        "",
        f"- **Timestamp**: {report.timestamp}",
        f"- **Tool version**: {report.tool_version}",
        f"- **Algorithm**: {report.algorithm}",
        f"- **Models**: {', '.join(model.name for model in report.models)}",
        f"- **Output**: `{report.output_path}` ({report.output_format})",
        f"- **Tensors merged**: {report.tensor_count} ({report.non_float_count} non-float)",
        f"- **Duration**: {report.duration_seconds:.3f} s",
        f"- **Verification**: {'PASSED' if report.verification.passed else 'FAILED'}",
    ]
    if report.peak_memory_bytes is not None:
        lines.append(f"- **Peak memory**: {report.peak_memory_bytes:,} bytes")
    lines += ["", "## Output files"]
    lines += [
        f"- `{name}`: `{report.output_hashes[name]}`" for name in sorted(report.output_hashes)
    ]
    lines += ["", "## Per-rule tensor counts", *_rule_lines(report.per_rule_counts)]
    if report.greedy_history:
        lines += ["", "## Greedy soup decisions"]
        for step in report.greedy_history:
            verdict = "accepted" if step.accepted else "rejected"
            lines.append(
                f"- `{step.candidate}` -> {verdict} (score {step.score:.6g}): {step.reason}"
            )
    if report.warnings:
        lines += ["", "## Warnings", *[f"- {warning}" for warning in report.warnings]]
    return "\n".join(lines) + "\n"


def write_report_markdown(report: MergeReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(report_to_markdown(report), encoding="utf-8")
    return destination
