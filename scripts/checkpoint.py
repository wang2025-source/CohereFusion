"""Fusion-only checkpoint serialization and strict architecture validation."""

from collections.abc import Mapping

import torch

from model.cohere_fusion import ARCHITECTURE


FROZEN_PREFIXES = ("text_encoder.", "semantic_teacher.")


def fusion_state(model):
    return {name: tensor for name, tensor in model.state_dict().items()
            if not name.startswith(FROZEN_PREFIXES)}


def load_checkpoint(model, checkpoint):
    if not isinstance(checkpoint, Mapping) or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Expected {ARCHITECTURE} checkpoint")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    expected = fusion_state(model)
    missing, unexpected = sorted(expected.keys() - state.keys()), sorted(state.keys() - expected.keys())
    invalid = [name for name in expected.keys() & state.keys()
               if not isinstance(state[name], torch.Tensor) or state[name].shape != expected[name].shape]
    if missing or unexpected or invalid:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}, invalid={invalid}")
    model.load_state_dict(state, strict=False)
