"""Baselines and learnable methods on frozen embeddings (Ahmadi & Sennrich 2026, Section 5).

    B1  LogRegDialect     logistic regression on the dialect region, no grid; its spatial prediction
                          is the cell of the region's centre
    B2  NearestPrototype  cosine similarity to the cell prototypes of the mapping mode
    M1  LogRegCells       logistic regression on cells
    M2  ProtoNet          learned linear projection, cosine to projected cell means, learned temperature

As in the paper, M1 and M2 are trained on the centroid cell of each unit's municipality whatever the
mapping mode; the mode then only changes how a predicted cell maps back to a dialect region. B2 is
the only one whose cell predictions depend on the mode.

A method has classes, each standing for a grid cell (`class_cells`) and, for B1, a dialect region
(`class_regions`); `scores` gives a score per class, higher is better, which can be averaged over
the units of a speaker.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from maloney.baselines.evaluate import GeoContext

log = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def centroid_cells(units: pd.DataFrame, geo: GeoContext) -> np.ndarray:
    """The training target of the paper: the cell of each unit's municipality centroid."""
    return geo.grid.cell_of(units[["muni_E", "muni_N"]].to_numpy(float))


class Method(ABC):
    class_cells: np.ndarray                # [K] grid cell of every class
    class_regions: np.ndarray | None = None  # [K] dialect region of every class (B1 only)

    @abstractmethod
    def fit(self, x: np.ndarray, units: pd.DataFrame, geo: GeoContext) -> None:
        """Train on embeddings [n, D] of the training units (rows of `units`)."""

    @abstractmethod
    def scores(self, x: np.ndarray) -> np.ndarray:
        """Score [n, K] of every class for embeddings [n, D]."""


class _LogReg(Method):
    """Multinomial logistic regression (L-BFGS, as in the paper: C=1.0)."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, standardize: bool = True):
        self.C, self.max_iter, self.standardize = C, max_iter, standardize

    def _fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        steps = [StandardScaler()] if self.standardize else []
        self.model = make_pipeline(*steps, LogisticRegression(C=self.C, max_iter=self.max_iter))
        self.model.fit(x, y)
        return self.model.classes_

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)


class LogRegDialect(_LogReg):
    """B1: predicts the dialect region; its cell is that of the region's centre, the clip-weighted
    mean of the centroids of the training municipalities in that region."""

    def fit(self, x, units, geo):
        self.class_regions = self._fit(x, units["dialect_region"].to_numpy())
        g = geo.geolects
        centres = np.array([np.average(g.xy[g.dialect_region == r], axis=0, weights=g.n_clips[g.dialect_region == r])
                            for r in self.class_regions])
        self.class_cells = geo.grid.cell_of(centres)


class LogRegCells(_LogReg):
    """M1: predicts the centroid cell; the classes are the cells holding a training municipality."""

    def fit(self, x, units, geo):
        self.class_cells = self._fit(x, centroid_cells(units, geo))


class NearestPrototype(Method):
    """B2: the cell whose prototype (mapping mode) is most cosine-similar to the embedding.

    `center` subtracts the mean training embedding first (not in the paper; speech embeddings are
    often anisotropic, which makes all cosine similarities nearly equal).
    """

    def __init__(self, center: bool = False):
        self.center = center

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        return l2_normalize(x - self.mean if self.center else x)

    def fit(self, x, units, geo):
        self.mean = x.mean(0)
        sums, counts = geo.geolects.embedding_sums(self._prepare(x), units["bfs_nr"])
        prototypes, has_mass = geo.mapping.prototypes(geo.weights, sums, counts)
        self.class_cells = np.flatnonzero(has_mass)
        self.prototypes = prototypes[has_mass]

    def scores(self, x):
        return self._prepare(x) @ self.prototypes.T


class ProtoNet(Method):
    """M2: a linear map W (initialised to the identity) trained so that projected embeddings are
    cosine-close to the projected mean of their centroid cell, with cross-entropy over
    temperature-scaled similarities (learned temperature). The cell means are computed once; their
    projections are refreshed every epoch (Adam, lr 1e-3, 50 epochs in the paper)."""

    def __init__(self, lr: float = 1e-3, epochs: int = 50, batch_size: int = 256, temperature: float = 10.0,
                 device: str | None = None, seed: int = 0):
        self.lr, self.epochs, self.batch_size, self.temperature = lr, epochs, batch_size, temperature
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = seed

    def _similarity(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.log_tau.exp() * torch.nn.functional.normalize(z, dim=1) @ torch.nn.functional.normalize(q, dim=1).T

    def fit(self, x, units, geo):
        torch.manual_seed(self.seed)
        targets = centroid_cells(units, geo)
        self.class_cells, y = np.unique(targets, return_inverse=True)
        x = torch.as_tensor(l2_normalize(x), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, device=self.device)
        means = torch.zeros(len(self.class_cells), x.shape[1], device=self.device).index_add_(0, y, x)
        self.means = means / torch.bincount(y, minlength=len(self.class_cells))[:, None]
        self.W = torch.nn.Parameter(torch.eye(x.shape[1], device=self.device))
        self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(self.temperature)), device=self.device))
        optimizer = torch.optim.Adam([self.W, self.log_tau], lr=self.lr)
        for epoch in range(self.epochs):
            q = (self.means @ self.W.T).detach()
            total = 0.0
            for batch in torch.randperm(len(x), device=self.device).split(self.batch_size):
                loss = torch.nn.functional.cross_entropy(self._similarity(x[batch] @ self.W.T, q), y[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                log.info("ProtoNet epoch %d/%d: loss %.4f, temperature %.2f", epoch + 1, self.epochs,
                         total / len(x), self.log_tau.exp().item())

    @torch.no_grad()
    def scores(self, x):
        x = torch.as_tensor(l2_normalize(x), dtype=torch.float32, device=self.device)
        return torch.softmax(self._similarity(x @ self.W.T, self.means @ self.W.T), dim=1).cpu().numpy()
