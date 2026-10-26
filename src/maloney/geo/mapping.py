"""Geolect-to-cell mappings and cell prototypes (Ahmadi & Sennrich 2026, Section 4).

Here a geolect is a municipality: the speakers living there, located at the municipality centroid.
Its region is the municipality together with the municipalities around it in the same canton:
those whose border lies within `neighbour_distance` metres of its own (0 = direct neighbours).
A mapping is a weight matrix A [geolects, cells]:

    centroid   A[g, cell of g's centroid] = 1                                   one cell per geolect
    uniform    A[g, c] = area(c ∩ R_g) / area(R_g)                              hard-edged region R_g
    gaussian   A[g, c] = exp(-d(c, g)^2 / (2 sigma^2)), dropped below eps       smooth, beyond borders

with d the distance between cell midpoint and centroid in metres. Unlike the paper, where every
cell of a region gets weight 1, uniform spreads one unit of weight evenly over the region's area:
a cell the region only grazes gets little, and a cell shared by overlapping regions is not counted
twice. (The paper's Gaussian has the factor 1 / (2 pi sigma^2), which cancels in the weighted mean
below; leaving it out makes `eps` a fraction of the peak weight: 1e-3 cuts the kernel at 3.7 sigma.)

The prototype of a cell is the L2-normalised mean embedding of what is mapped onto it:

    centroid, uniform   p_c = sum_g A[g,c] * sum_g(e) / sum_g A[g,c] * n_g     (Eq. 4: over clips)
    gaussian            p_c = sum_g A[g,c] * mean_g(e) / sum_g A[g,c]          (Eq. 6: over geolects)

A > 0 also gives the geolects of each cell (the paper's D_c), used to recover a variety from a
predicted cell.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

from maloney.geo.grid import Grid

log = logging.getLogger(__name__)

MODES = ("centroid", "uniform", "gaussian")
# The municipality polygons are simplified one by one, which leaves gaps of up to a few tens of
# metres between neighbours; borders closer than this count as touching.
TOUCH_TOLERANCE = 100


class Municipalities:
    """Municipality polygons in LV95 with BFS number and canton (data/geo/municipalities.geojson)."""

    def __init__(self, path: str | Path):
        features = json.loads(Path(path).read_text())["features"]
        self.bfs_nr = np.array([f["properties"]["bfs_nr"] for f in features])
        self.name = np.array([f["properties"]["municipality"] for f in features])
        self.canton = np.array([f["properties"]["canton"] for f in features])
        self.geometry = np.array([shape(f["geometry"]) for f in features])
        self.tree = shapely.STRtree(self.geometry)
        self.index = {int(b): i for i, b in enumerate(self.bfs_nr)}

    def __contains__(self, bfs_nr) -> bool:
        return int(bfs_nr) in self.index

    def neighbourhood(self, bfs_nr: int, distance: float = 0) -> np.ndarray:
        """Indices of the municipality and those of the same canton within `distance` metres of it."""
        i = self.index[int(bfs_nr)]
        near = self.tree.query(self.geometry[i], predicate="dwithin", distance=distance + TOUCH_TOLERANCE)
        return np.sort(near[self.canton[near] == self.canton[i]])

    def region(self, bfs_nr: int, distance: float = 0) -> shapely.Geometry:
        """The area of `neighbourhood`, as one geometry."""
        return shapely.union_all(self.geometry[self.neighbourhood(bfs_nr, distance)])


@dataclass
class Geolects:
    """The municipalities in some clip metadata, with centroid, canton and main dialect region."""

    bfs_nr: np.ndarray          # [G] BFS municipality number
    xy: np.ndarray              # [G, 2] centroid in LV95 (E, N)
    canton: np.ndarray          # [G] canton of the municipality
    dialect_region: np.ndarray  # [G] most frequent dialect region of its clips
    n_clips: np.ndarray         # [G]

    @classmethod
    def from_meta(cls, meta: pd.DataFrame) -> Geolects:
        """From clip metadata with locations (STT4SGDataset.meta when the data module has a geo_table)."""
        g = meta.groupby("bfs_nr").agg(E=("muni_E", "first"), N=("muni_N", "first"),
                                       canton=("muni_canton", "first"), n_clips=("path", "size"),
                                       dialect_region=("dialect_region", lambda s: s.mode().iloc[0]))
        return cls(g.index.to_numpy(), g[["E", "N"]].to_numpy(float), g["canton"].to_numpy(),
                   g["dialect_region"].to_numpy(), g["n_clips"].to_numpy())

    def __len__(self) -> int:
        return len(self.bfs_nr)

    def index_of(self, bfs_nr) -> np.ndarray:
        """Geolect index of every BFS number (-1 for municipalities not in this set)."""
        lookup = pd.Series(np.arange(len(self)), index=self.bfs_nr)
        return lookup.reindex(np.asarray(bfs_nr)).fillna(-1).to_numpy(np.int64)

    def embedding_sums(self, embeddings: np.ndarray, bfs_nr) -> tuple[np.ndarray, np.ndarray]:
        """Per geolect the sum of the clip `embeddings` [N, D] and the number of clips."""
        index = self.index_of(bfs_nr)
        if (index < 0).any():
            raise ValueError(f"{(index < 0).sum()} clips are from municipalities that are not geolects")
        sums = np.zeros((len(self), embeddings.shape[1]), dtype=np.float64)
        np.add.at(sums, index, embeddings)
        return sums, np.bincount(index, minlength=len(self)).astype(np.float64)


@dataclass
class CellMapping:
    """How geolects are spread over the cells: `mode` is centroid, uniform or gaussian.

    `municipalities` (the polygons) and `neighbour_distance` (metres) are only used by the uniform
    mode, `sigma` (metres) and `eps` (fraction of the peak weight) only by the gaussian mode.
    """

    mode: str = "centroid"
    sigma: float = 20_000
    eps: float = 1e-3
    neighbour_distance: float = 0
    municipalities: str | Path | Municipalities | None = None
    _municipalities: Municipalities | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.mode == "uniform" and self.municipalities is None:
            raise ValueError("uniform mode needs the municipality polygons (municipalities=...geojson)")

    @property
    def polygons(self) -> Municipalities:
        if self._municipalities is None:
            m = self.municipalities
            self._municipalities = m if isinstance(m, Municipalities) else Municipalities(m)
        return self._municipalities

    def weights(self, grid: Grid, geolects: Geolects) -> np.ndarray:
        """A [geolects, cells]: how much of each geolect goes to each cell."""
        weights = np.zeros((len(geolects), len(grid)))
        if self.mode == "centroid":
            weights[np.arange(len(geolects)), grid.cell_of(geolects.xy)] = 1
        elif self.mode == "uniform":
            missing = [b for b in geolects.bfs_nr if b not in self.polygons]
            if missing:
                log.warning("%d municipalities have no polygon and stay in their centroid cell: %s",
                            len(missing), missing[:10])
            for g, bfs_nr in enumerate(geolects.bfs_nr):
                if bfs_nr in missing:
                    weights[g, grid.cell_of(geolects.xy[g])] = 1
                    continue
                cells, areas = grid.cell_areas(self.polygons.region(bfs_nr, self.neighbour_distance))
                weights[g, cells] = areas / areas.sum()
        else:
            d2 = ((geolects.xy[:, None, :] - grid.midpoints[None]) ** 2).sum(-1)
            weights = np.exp(-d2 / (2 * self.sigma**2))
            weights[weights < self.eps] = 0
        return weights

    def prototypes(self, weights: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """L2-normalised prototypes [cells, D] from per-geolect embedding sums and clip counts,
        and the mask [cells] of cells that received any mass (the others have no prototype)."""
        if self.mode == "gaussian":
            numerator, denominator = weights.T @ (sums / np.maximum(counts, 1)[:, None]), weights.sum(0)
        else:
            numerator, denominator = weights.T @ sums, weights.T @ counts
        has_mass = denominator > 0
        prototypes = np.zeros_like(numerator)
        prototypes[has_mass] = numerator[has_mass] / denominator[has_mass, None]
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        return np.divide(prototypes, norms, out=np.zeros_like(prototypes), where=norms > 0), has_mass

    def geolects_of_cells(self, weights: np.ndarray) -> list[np.ndarray]:
        """For every cell, the indices of the geolects mapped onto it (the paper's D_c)."""
        return [np.flatnonzero(column) for column in (weights > 0).T]
