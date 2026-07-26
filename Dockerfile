# Minimal, non-root image for running merges via the CLI.
#
#   docker build -t model-merger .
#   docker run --rm -v "$PWD:/work" -w /work model-merger merge configs/uniform_soup.example.yaml
#
# Uses the CPU-only PyTorch wheel to keep the image small; mount a CUDA-enabled
# base and install the matching torch build if you need GPU.

FROM python:3.11-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install into an isolated prefix we copy into the runtime stage.
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install .

FROM python:3.11-slim AS runtime

# Create an unprivileged user.
RUN useradd --create-home --uid 10001 merger
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER merger
WORKDIR /work

ENTRYPOINT ["model-merger"]
CMD ["--help"]
