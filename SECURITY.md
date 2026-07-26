# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅         |

Security fixes are released for the latest minor version. Older versions may not
receive patches.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately**. Do not open a public
issue. Use GitHub's private "Report a vulnerability" advisory flow on the
repository, or email the maintainers at the address in the repository metadata.

We aim to acknowledge reports within 5 business days and to provide a remediation
timeline after triage. Please include:

- A description of the issue and its impact.
- Steps to reproduce (a minimal config or checkpoint if possible).
- The version and platform affected.

## Trust model

Model Merger draws a clear boundary between the application and third-party model
files, which are treated as **untrusted input**:

- **Pickle-backed checkpoints** (`.bin`/`.pt`) are loaded with
  `weights_only=True` by default. Full unpickling (which can execute arbitrary
  code) requires an explicit `allow_unsafe_pytorch` / `--allow-unsafe` opt-in and
  emits a prominent warning. Prefer safetensors.
- **Checkpoint metadata** (shard filenames, ancillary file names) is validated to
  be safe relative members before any file is written; path traversal is
  rejected.
- **Output is never overwritten** unless `overwrite` is requested, and is written
  atomically via a staging directory so a crash cannot leave a corrupt output
  presented as success.
- **External evaluators** are executed as argument vectors with `shell=False`;
  the checkpoint path is passed as a single argument, so there is no command
  injection surface.
- **Configuration files** are treated as trusted (they are yours), but secrets in
  paths/environment variables are redacted from logs and never written to
  reports.

See [docs/security-and-trust.md](docs/security-and-trust.md) for details.
