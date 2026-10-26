"""Clip embeddings from a Whisper encoder: hidden states of chosen layers, mean-pooled over time.

Whisper's encoder only takes 30 s of input (3000 log-mel frames, padded with silence) and returns
1500 frames (50 per second). The pooled embedding of a clip averages only the frames that hold its
audio, not the padding. A clip longer than 30 s is cut into 30 s chunks whose pooled embeddings are
averaged weighted by their number of frames.

`hidden_states[0]` is the input to the first layer, `hidden_states[k]` the output of layer k; the
last one includes the encoder's final layer norm. Dialect information tends to peak in the middle
layers, so several layers are kept and the choice is left to the experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

SAMPLE_RATE = 16_000
CHUNK = 30 * SAMPLE_RATE  # samples per encoder input


def default_layers(n_layers: int, n_keep: int = 8) -> list[int]:
    """`n_keep` evenly spaced layers ending with the last, e.g. 4, 8, ..., 32 for whisper-large."""
    step = max(1, n_layers // n_keep)
    return list(range(n_layers % step or step, n_layers + 1, step))


def chunk_audio(audio: np.ndarray) -> list[np.ndarray]:
    """Cut a waveform into pieces of at most 30 s (a single piece for normal clips)."""
    return [audio[i: i + CHUNK] for i in range(0, max(len(audio), 1), CHUNK)]


def featurize(feature_extractor: WhisperFeatureExtractor, audios: list[np.ndarray]) -> dict[str, torch.Tensor]:
    """Log-mel features of the clips' 30 s chunks (CPU work, meant for dataloader workers).

    Returns the features [n_chunks, n_mels, 3000], the encoder frames holding audio per chunk and
    the clip each chunk belongs to.
    """
    chunks, owner = [], []
    for i, audio in enumerate(audios):
        pieces = chunk_audio(audio)
        chunks += pieces
        owner += [i] * len(pieces)
    f = feature_extractor(chunks, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
    mel_frames = f["attention_mask"].sum(1)
    return {"input_features": f["input_features"], "frames": (mel_frames + 1) // 2,
            "owner": torch.tensor(owner), "n_clips": torch.tensor(len(audios))}


def pool_frames(hidden: torch.Tensor, frames: torch.Tensor, owner: torch.Tensor, n_items: int) -> torch.Tensor:
    """Mean over the frames with audio of chunk hidden states [n_chunks, T, ...] -> [n_items, ...].

    Chunks of the same item (`owner`) are combined weighted by their number of frames, so the result
    is the mean over all audio frames of the item. Differentiable (used for fine-tuning too).
    """
    mask = (torch.arange(hidden.shape[1], device=hidden.device)[None] < frames[:, None]).to(hidden.dtype)
    sums = (hidden * mask.view(*mask.shape, *[1] * (hidden.dim() - 2))).sum(1)
    item_sums = torch.zeros(n_items, *sums.shape[1:], device=sums.device, dtype=sums.dtype).index_add(0, owner, sums)
    item_frames = torch.zeros(n_items, device=sums.device, dtype=sums.dtype).index_add(0, owner, frames.to(sums.dtype))
    return item_sums / item_frames.clamp(min=1).view(-1, *[1] * (sums.dim() - 1))


class WhisperEmbedder:
    """Mean-pooled hidden states of `layers` of a Whisper encoder (any checkpoint on the HF Hub)."""

    def __init__(self, model: str = "openai/whisper-large-v3", layers: list[int] | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        self.model_name = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.features = WhisperFeatureExtractor.from_pretrained(model)
        self.encoder = WhisperModel.from_pretrained(model, dtype=self.dtype).get_encoder().to(self.device).eval()
        n_layers = self.encoder.config.encoder_layers
        self.layers = default_layers(n_layers) if layers is None else list(layers)
        if not all(0 <= layer <= n_layers for layer in self.layers):
            raise ValueError(f"{model} has layers 0..{n_layers}, got {self.layers}")
        self.dim = self.encoder.config.d_model

    @torch.inference_mode()
    def embed_features(self, batch: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """Pooled embeddings [n_clips, n_layers, dim] (float32) and encoder frames per clip."""
        features = batch["input_features"].to(self.device, self.dtype)
        hidden = self.encoder(features, output_hidden_states=True).hidden_states
        frames, owner, n_clips = batch["frames"].to(self.device), batch["owner"].to(self.device), int(batch["n_clips"])
        stacked = torch.stack([hidden[layer].float() for layer in self.layers], 2)  # [chunks, T, layers, dim]
        pooled = pool_frames(stacked, frames, owner, n_clips)
        clip_frames = torch.zeros(n_clips, device=self.device).index_add_(0, owner, frames.float())
        return pooled.cpu().numpy(), clip_frames.cpu().numpy().astype(np.int32)

    def embed(self, audios: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Embed waveforms (16 kHz mono float32) in one go: [n, n_layers, dim] and frames per clip."""
        return self.embed_features(featurize(self.features, audios))
