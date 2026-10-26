"""M3 of Ahmadi & Sennrich (2026): a Whisper encoder fine-tuned to classify grid cells.

The encoder output at `pool_layer` is mean-pooled over the frames that hold audio (as in the
precomputed embeddings, maloney.embeddings.whisper) and a linear head maps it to the classes, the
centroid cells of the training municipalities. Layers above `pool_layer` are dropped, so pooling a
middle layer is also cheaper. After training the head is discarded: the pooled embeddings of the
fine-tuned encoder build cell prototypes in each mapping mode (maloney.train).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import WhisperFeatureExtractor, WhisperModel

from maloney.data.stt4sg import collate
from maloney.embeddings.whisper import featurize, pool_frames


class WhisperCollate:
    """The data module's collate plus Whisper log-mel features of every item (in the workers)."""

    def __init__(self, model: str):
        self.features = WhisperFeatureExtractor.from_pretrained(model)

    def __call__(self, items: list[dict]) -> dict:
        batch = collate(items)
        audios = [it["audio"].numpy() for it in items]
        batch.update(featurize(self.features, audios))
        del batch["audio"]  # the features replace the padded waveforms
        return batch


class WhisperGeoClassifier(nn.Module):
    def __init__(self, model: str, n_classes: int, pool_layer: int | None = None, freeze_layers: int = 0,
                 dropout: float = 0.0, gradient_checkpointing: bool = False):
        super().__init__()
        self.model_name = model
        self.encoder = WhisperModel.from_pretrained(model).get_encoder()
        n_layers = self.encoder.config.encoder_layers
        self.pool_layer = n_layers if pool_layer is None else pool_layer
        if not 1 <= self.pool_layer <= n_layers:
            raise ValueError(f"pool_layer must be in 1..{n_layers}, got {pool_layer}")
        if self.pool_layer < n_layers:
            # The output of layer k has no final layer norm (same as hidden_states[k] in the extraction).
            self.encoder.layers = self.encoder.layers[: self.pool_layer]
            self.encoder.layer_norm = nn.Identity()
            self.encoder.config.encoder_layers = self.pool_layer
        # The convolutional front end and the lowest `freeze_layers` transformer layers stay fixed.
        for module in (self.encoder.conv1, self.encoder.conv2, *self.encoder.layers[:freeze_layers]):
            module.requires_grad_(False)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.dim = self.encoder.config.d_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.dim, n_classes)

    def embed(self, batch: dict) -> torch.Tensor:
        """Pooled embeddings [n_items, dim] of a batch from WhisperCollate (on the model's device)."""
        device = self.head.weight.device
        hidden = self.encoder(batch["input_features"].to(device)).last_hidden_state
        return pool_frames(hidden.float(), batch["frames"].to(device), batch["owner"].to(device), int(batch["n_clips"]))

    def forward(self, batch: dict) -> torch.Tensor:
        """Class logits [n_items, n_classes]."""
        return self.head(self.dropout(self.embed(batch)))

    def n_parameters(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters()), trainable


def class_index(cells: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Position of every cell in the sorted array of class cells (-1 for cells that are no class)."""
    pos = np.searchsorted(classes, cells).clip(max=len(classes) - 1)
    return np.where(classes[pos] == cells, pos, -1)
