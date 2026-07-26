"""Generate tiny, deterministic model checkpoints for tests and examples.

The whole test suite runs against these fixtures -- no multi-gigabyte downloads,
no network.  Each model is a handful of small tensors plus a minimal Hugging Face
``config.json`` and tokenizer stub, written as safetensors (and optionally a
pytorch ``.bin``).  Determinism comes from a fixed seed per model.

Usage:
    python scripts/generate_tiny_test_models.py OUTPUT_DIR [--pytorch]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

#: Small, transformer-like tensor shapes shared by every generated model.
TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    "model.embed_tokens.weight": (32, 8),
    "model.layers.0.self_attn.q_proj.weight": (8, 8),
    "model.layers.0.self_attn.k_proj.weight": (8, 8),
    "model.layers.0.mlp.up_proj.weight": (16, 8),
    "model.layers.0.mlp.down_proj.weight": (8, 16),
    "model.layers.0.input_layernorm.weight": (8,),
    "model.layers.1.self_attn.q_proj.weight": (8, 8),
    "model.layers.1.mlp.up_proj.weight": (16, 8),
    "model.norm.weight": (8,),
    "lm_head.weight": (32, 8),
}

#: A non-float buffer that must be handled by the non-float policy.
INT_BUFFER_KEY = "model.layers.0.self_attn.rotary_emb.inv_freq_ids"


def build_state_dict(seed: int, *, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Return a deterministic state dict for the given seed."""

    generator = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {}
    for key, shape in TENSOR_SHAPES.items():
        state[key] = torch.randn(shape, generator=generator).to(dtype)
    # Integer buffer identical across models (satisfies require_equal by default).
    state[INT_BUFFER_KEY] = torch.arange(4, dtype=torch.int64)
    return state


def minimal_config(vocab_size: int = 32, hidden_size: int = 8) -> dict[str, object]:
    return {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": 2,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.44.0",
    }


def minimal_tokenizer() -> dict[str, object]:
    return {"model_type": "stub", "bos_token": "<s>", "eos_token": "</s>"}


def write_model(
    directory: Path,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
    write_pytorch: bool = False,
    with_ancillary: bool = True,
) -> Path:
    """Write one tiny model directory and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    state = build_state_dict(seed, dtype=dtype)
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    if write_pytorch:
        torch.save(state, directory / "pytorch_model.bin")
    if with_ancillary:
        (directory / "config.json").write_text(json.dumps(minimal_config(), indent=2))
        (directory / "generation_config.json").write_text(json.dumps({"max_length": 32}, indent=2))
        (directory / "tokenizer_config.json").write_text(json.dumps(minimal_tokenizer(), indent=2))
        (directory / "special_tokens_map.json").write_text(
            json.dumps({"bos_token": "<s>", "eos_token": "</s>"}, indent=2)
        )
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tiny test models.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pytorch", action="store_true", help="Also write pytorch_model.bin.")
    parser.add_argument("--count", type=int, default=3, help="Number of models.")
    args = parser.parse_args()

    for index in range(args.count):
        model_dir = args.output_dir / f"model-{chr(ord('a') + index)}"
        write_model(model_dir, seed=1000 + index, write_pytorch=args.pytorch)
        print(f"wrote {model_dir}")


if __name__ == "__main__":
    main()
