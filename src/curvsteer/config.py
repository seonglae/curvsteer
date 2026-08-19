"""Model sites, one entry per checkpoint the sweep can run on.

A site is a residual-stream position. Everything else in this package is
model-agnostic and reads what it needs from the loaded config object.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model: str
    block: int
    blocks_attr: str
    dtype: str = "float32"
    #: Sparse-autoencoder release for this site, if one exists. The dictionary
    #: arms are only constructible when this is set, which is itself part of the
    #: argument: the method needs no dictionary, so it runs where none exists.
    sae_repo: str | None = None
    sae_file: str | None = None
    notes: str = ""


CONFIGS: dict[str, ModelConfig] = {
    "gpt2": ModelConfig(
        model="gpt2", block=6, blocks_attr="transformer.h",
        notes="Small enough to smoke-test the whole pipeline on CPU.",
    ),
    "gemma2": ModelConfig(
        model="google/gemma-2-2b-it", block=12, blocks_attr="model.layers",
        dtype="float32",
        sae_repo="google/gemma-scope-2b-pt-res",
        sae_file="layer_12/width_16k/average_l0_82/params.npz",
        notes=(
            "Carries the only dictionary arms in the package. Note the mismatch "
            "this creates: a base-model dictionary read on an instruct model, "
            "which is a consequence of letting dictionary availability pick the "
            "checkpoint. See README, 'Why the headline is not this model'."
        ),
    ),
    "gemma4": ModelConfig(
        model="google/gemma-4-12B", block=24, blocks_attr="model.layers",
        dtype="bfloat16",
        notes=(
            "The headline site. No dictionary exists for this checkpoint, so the "
            "dictionary arms are absent by necessity rather than by choice."
        ),
    ),
}
