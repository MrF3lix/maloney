import numpy as np
import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from maloney.models.whisper_geo import WhisperCollate, WhisperGeoClassifier, class_index
from maloney.train import HeadMethod, autocast_dtype

TINY = "openai/whisper-tiny"
tiny_cached = isinstance(try_to_load_from_cache(TINY, "config.json"), str)


def test_class_index():
    assert class_index(np.array([5, 7, 9, 2, 100]), np.array([2, 5, 9])).tolist() == [1, -1, 2, 0, -1]


def test_head_method():
    scores = HeadMethod(np.array([3, 8])).scores(np.array([[0.0, 0.0], [10.0, 0.0]]))
    np.testing.assert_allclose(scores.sum(1), 1)
    assert scores[1].argmax() == 0


def test_autocast_dtype():
    assert autocast_dtype(torch.device("cpu"), "auto") is None
    assert autocast_dtype(torch.device("cpu"), "bf16") == torch.bfloat16


@pytest.mark.skipif(not tiny_cached, reason="needs openai/whisper-tiny in the Hugging Face cache")
def test_model_trains_on_a_batch():
    rng = np.random.default_rng(0)
    items = [{"audio": torch.tensor(rng.normal(0, 0.1, 16_000 * s).astype(np.float32)), "dialect": 0, "canton": 0,
              "zipcode": "8000", "client_id": "a", "path": "p", "n_clips": 1} for s in (2, 4)]
    batch = WhisperCollate(TINY)(items)
    model = WhisperGeoClassifier(TINY, n_classes=3, pool_layer=2, freeze_layers=1)
    assert len(model.encoder.layers) == 2
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    targets = torch.tensor([0, 2])
    losses = []
    for _ in range(5):
        loss = torch.nn.functional.cross_entropy(model(batch), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert model.encoder.conv1.weight.grad is None and model.encoder.layers[0].fc1.weight.grad is None


@pytest.mark.skipif(not tiny_cached, reason="needs openai/whisper-tiny in the Hugging Face cache")
def test_float16_checkpoint_trains_in_float32(tmp_path):
    """whisper-large-v3 is stored in float16; training its weights in float16 with AdamW gives nan."""
    from transformers import WhisperFeatureExtractor, WhisperModel

    WhisperModel.from_pretrained(TINY, dtype=torch.float16).save_pretrained(tmp_path)
    WhisperFeatureExtractor.from_pretrained(TINY).save_pretrained(tmp_path)
    model = WhisperGeoClassifier(str(tmp_path), n_classes=3)
    assert {p.dtype for p in model.parameters()} == {torch.float32}

    rng = np.random.default_rng(0)
    items = [{"audio": torch.tensor(rng.normal(0, 0.1, 16_000 * 3).astype(np.float32)), "dialect": 0, "canton": 0,
              "zipcode": "8000", "client_id": "a", "path": "p", "n_clips": 1} for _ in range(3)]
    batch = WhisperCollate(str(tmp_path))(items)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    for _ in range(5):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0, 1, 2]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
    assert all(torch.isfinite(p).all() for p in model.parameters())
