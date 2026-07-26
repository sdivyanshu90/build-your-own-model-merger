"""Domain-specific exception hierarchy for :mod:`model_merger`.

Every expected failure mode maps to exactly one exception type and one stable
CLI exit code (see :attr:`ModelMergerError.exit_code`).  User-facing code should
catch :class:`ModelMergerError` and render ``error.message`` without a traceback
unless debug mode is enabled; unexpected exceptions (anything *not* derived from
:class:`ModelMergerError`) are bugs and should surface with a full traceback.

The exit codes are part of the public contract and are documented in
``docs/cli-reference.md``.  They intentionally start at 2 because 0 means
success and 1 is reserved for "unexpected/unclassified error".
"""

from __future__ import annotations

__all__ = [
    "ModelMergerError",
    "ConfigurationError",
    "CheckpointError",
    "UnsafeCheckpointError",
    "CompatibilityError",
    "TensorMismatchError",
    "NumericalError",
    "EvaluationError",
    "OutputExistsError",
    "InsufficientDiskSpaceError",
    "MergeExecutionError",
    "VerificationError",
    "EXIT_CODES",
]


class ModelMergerError(Exception):
    """Base class for all expected, actionable failures.

    Attributes:
        message: A human-readable, actionable description of what went wrong.
        exit_code: Process exit code the CLI should return for this error class.
    """

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(ModelMergerError):
    """The configuration is syntactically or semantically invalid."""

    exit_code = 2


class CheckpointError(ModelMergerError):
    """A checkpoint could not be read, is malformed, or is unsupported."""

    exit_code = 3


class UnsafeCheckpointError(CheckpointError):
    """Loading the checkpoint would require executing untrusted pickle data.

    Raised when a pickle-backed PyTorch checkpoint is encountered without the
    explicit ``allow_unsafe`` opt-in.  This is a security boundary, not a mere
    format limitation, so it has its own exit code.
    """

    exit_code = 4


class CompatibilityError(ModelMergerError):
    """The source models are not compatible for merging."""

    exit_code = 5


class TensorMismatchError(CompatibilityError):
    """Two tensors that must align (shape/dtype/presence) do not."""

    exit_code = 6


class NumericalError(ModelMergerError):
    """A merge produced non-finite values or violated a numerical invariant."""

    exit_code = 7


class EvaluationError(ModelMergerError):
    """An evaluator (callable or external command) failed to produce a metric."""

    exit_code = 8


class OutputExistsError(ModelMergerError):
    """The output path already exists and ``overwrite`` was not requested."""

    exit_code = 9


class InsufficientDiskSpaceError(ModelMergerError):
    """The preflight disk-space estimate exceeds available free space."""

    exit_code = 10


class MergeExecutionError(ModelMergerError):
    """A failure occurred while executing the merge plan."""

    exit_code = 11


class VerificationError(ModelMergerError):
    """The written output failed post-merge verification."""

    exit_code = 12


#: Mapping of exception class name -> exit code, for documentation/tests.
EXIT_CODES: dict[str, int] = {
    cls.__name__: cls.exit_code
    for cls in (
        ModelMergerError,
        ConfigurationError,
        CheckpointError,
        UnsafeCheckpointError,
        CompatibilityError,
        TensorMismatchError,
        NumericalError,
        EvaluationError,
        OutputExistsError,
        InsufficientDiskSpaceError,
        MergeExecutionError,
        VerificationError,
    )
}
