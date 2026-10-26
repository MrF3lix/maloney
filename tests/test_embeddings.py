import h5py
import numpy as np
import pandas as pd
import pytest

from maloney.embeddings import EmbeddingStore
from maloney.embeddings.whisper import chunk_audio, default_layers


def test_default_layers():
    assert default_layers(32) == [4, 8, 12, 16, 20, 24, 28, 32]
    assert default_layers(24) == [3, 6, 9, 12, 15, 18, 21, 24]
    assert default_layers(4) == [1, 2, 3, 4]


def test_chunk_audio():
    assert [len(c) for c in chunk_audio(np.zeros(16_000 * 41))] == [480_000, 176_000]
    assert len(chunk_audio(np.zeros(100))) == 1


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "emb.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("path", data=["s1/a.mp3", "s1/b.mp3", "s2/c.mp3"], dtype=h5py.string_dtype())
        f["frames"] = np.array([100, 300, 50], dtype="int32")
        f["done"] = np.ones(3, bool)
        f["layer_04"] = np.array([[1, 0], [0, 1], [1, 1]], dtype="float16")
        f.attrs.update(model="fake", layers=[4], dim=2)
    return EmbeddingStore(path)


def test_store_lookup_and_pooling(store):
    np.testing.assert_array_equal(store.get(["s2/c.mp3", "s1/a.mp3"], 4), [[1, 1], [1, 0]])
    # frame-weighted: (100 * (1, 0) + 300 * (0, 1)) / 400
    np.testing.assert_allclose(store.pooled([["s1/a.mp3", "s1/b.mp3"]], 4), [[0.25, 0.75]])
    meta = pd.DataFrame({"path": ["s1/a.mp3", "s1/b.mp3", "s2/c.mp3"], "client_id": ["s1", "s1", "s2"]})
    np.testing.assert_allclose(store.for_segments(meta, [(0, 1), (2,)], 4), [[0.25, 0.75], [1, 1]])
    speakers, pooled = store.by_speaker(meta, 4)
    assert speakers.tolist() == ["s1", "s2"] and pooled.shape == (2, 2)
    with pytest.raises(KeyError):
        store.get(["nope.mp3"], 4)
    with pytest.raises(ValueError):
        store.layer(8)


def test_incomplete_store_is_refused(store):
    with h5py.File(store.path, "a") as f:
        f["done"][1] = False
    with pytest.raises(ValueError, match="incomplete"):
        EmbeddingStore(store.path)
