"""Metrics of Ahmadi & Sennrich (2026, Section 5.3), with the dialect region as the variety.

Every evaluated unit (clip, segment or speaker) has a true location (its municipality centroid) and
a predicted cell ĉ. Spatial metrics:

    cell_acc         ĉ lies in the region of the true municipality under the mapping (for centroid
                     mode: ĉ is its centroid cell)
    acc@N            Chebyshev distance between ĉ and the true centroid cell is at most N (acc@0 = exact)
    mean_km/median_km  distance between the midpoint of ĉ and the true centroid

The dialect region is recovered from ĉ through the training municipalities mapped onto it (D_ĉ):

    region_acc       the paper's rule: correct if the true region is among D_ĉ's regions; otherwise
                     the region of the municipality in D_ĉ whose centroid cell is closest to ĉ
                     (Chebyshev, ties alphabetical); wrong if D_ĉ is empty
    region_f1        macro-F1 of these predicted regions
    region_acc_majority  stricter: the region with the most weight in ĉ (sum of weight x clips), since
                     the paper's rule is lenient when D_ĉ is large (uniform, gaussian)

Methods that predict regions directly (B1) use their region for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from maloney.geo import CellMapping, Geolects, Grid

NO_REGION = "<none>"


@dataclass
class GeoContext:
    """The grid, the mapping and the training municipalities (geolects) mapped onto the grid."""

    grid: Grid
    mapping: CellMapping
    geolects: Geolects
    weights: np.ndarray = field(init=False)         # [G, C] mapping weights of the training geolects
    centroid_cells: np.ndarray = field(init=False)  # [G] cell of each training geolect's centroid

    def __post_init__(self):
        self.weights = self.mapping.weights(self.grid, self.geolects)
        self.centroid_cells = self.grid.cell_of(self.geolects.xy)
        self.cell_geolects = self.mapping.geolects_of_cells(self.weights)
        regions = np.unique(self.geolects.dialect_region)
        onehot = (self.geolects.dialect_region[:, None] == regions[None]).astype(float)
        mass = self.weights.T @ (onehot * self.geolects.n_clips[:, None])  # [C, R]
        self.majority_region = np.where(mass.sum(1) > 0, regions[mass.argmax(1)], NO_REGION)

    def recover_region(self, cell: int, true_region: str) -> str:
        """The paper's variety recovery for one predicted cell."""
        members = self.cell_geolects[cell]
        if len(members) == 0:
            return NO_REGION
        regions = self.geolects.dialect_region[members]
        if true_region in regions:
            return true_region
        distance = self.grid.chebyshev(self.centroid_cells[members], np.full(len(members), cell))
        nearest = regions[distance == distance.min()]
        return sorted(nearest)[0]


def evaluate(cells: np.ndarray, units: pd.DataFrame, geo: GeoContext,
             regions: np.ndarray | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    """Metrics for predicted cells (and, for B1, predicted regions) of `units`.

    `units` has the columns of clip metadata with locations (one row per unit, see units.py): the
    true location is muni_E/muni_N. Returns the metrics and a table of per-unit predictions.
    """
    grid = geo.grid
    true_xy = units[["muni_E", "muni_N"]].to_numpy(float)
    true_cells = grid.cell_of(true_xy)
    true_regions = units["dialect_region"].to_numpy()

    # Regions of the evaluated municipalities under the same mapping, for cell accuracy.
    eval_geolects = Geolects.from_meta(units)
    eval_weights = geo.mapping.weights(grid, eval_geolects)
    in_region = eval_weights[eval_geolects.index_of(units["bfs_nr"]), cells] > 0

    if regions is None:
        regions = np.array([geo.recover_region(c, r) for c, r in zip(cells, true_regions)])
        majority = geo.majority_region[cells]
    else:
        majority = regions
    chebyshev = grid.chebyshev(cells, true_cells)
    km = grid.distance_km(grid.midpoints[cells], true_xy)
    labels = sorted(set(true_regions))
    metrics = {
        "n": len(units),
        "region_acc": float((regions == true_regions).mean()),
        "region_f1": float(f1_score(true_regions, regions, labels=labels, average="macro", zero_division=0)),
        "region_acc_majority": float((majority == true_regions).mean()),
        "cell_acc": float(in_region.mean()),
        **{f"acc@{n}": float((chebyshev <= n).mean()) for n in range(4)},
        "mean_km": float(km.mean()),
        "median_km": float(np.median(km)),
    }
    predictions = units.assign(pred_cell=cells, true_cell=true_cells, pred_region=regions,
                               pred_region_majority=majority, error_km=km, chebyshev=chebyshev)
    return metrics, predictions
