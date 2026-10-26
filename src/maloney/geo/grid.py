"""Square grid over Switzerland in Swiss LV95 coordinates (EPSG:2056, metres).

The paper's grid (Ahmadi & Sennrich 2026) uses square cells of g degrees; here the cells are
`cell_size` metres in LV95, so every cell has the same physical size. The grid is anchored at
multiples of `cell_size` (cell edges at round LV95 coordinates) and only cells that overlap
Switzerland are kept ("active" cells); they are numbered 0..len(grid)-1, which are the class ids.

Each active cell keeps its integer position (ix, iy) on the full grid, used for the Chebyshev cell
distance (the paper's Acc@N), and its midpoint. Points map to the cell they lie in; a point in a
cell that does not overlap Switzerland (e.g. a centroid just across a border or in a lake) maps to
the nearest active cell.
"""

from __future__ import annotations

import json
import math
from functools import cached_property
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape


def lv95_to_wgs84(east, north) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) in degrees for LV95 (E, N) in metres; swisstopo's approximation, accurate to ~1 m."""
    y = (np.asarray(east, dtype=float) - 2_600_000) / 1e6
    x = (np.asarray(north, dtype=float) - 1_200_000) / 1e6
    lon = 2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x**2 - 0.0436 * y**3
    lat = 16.9023892 + 3.238272 * x - 0.270978 * y**2 - 0.002528 * x**2 - 0.0447 * y**2 * x - 0.0140 * x**3
    return lat * 100 / 36, lon * 100 / 36


def read_regions(path: str | Path, key: str = "canton") -> dict[str, shapely.Geometry]:
    """Polygons of a GeoJSON file in LV95 (e.g. data/geo/cantons.geojson), by the property `key`."""
    features = json.loads(Path(path).read_text())["features"]
    return {f["properties"][key]: shape(f["geometry"]) for f in features}


class Grid:
    """Active square cells of `cell_size` metres over the union of `regions` (e.g. the cantons)."""

    def __init__(self, regions: dict[str, shapely.Geometry], cell_size: float = 10_000):
        self.cell_size = float(cell_size)
        self.regions = regions
        self.outline = shapely.union_all(list(regions.values()))
        size = self.cell_size
        xmin, ymin, xmax, ymax = self.outline.bounds
        self.x0, self.y0 = math.floor(xmin / size) * size, math.floor(ymin / size) * size
        self.nx, self.ny = math.ceil((xmax - self.x0) / size), math.ceil((ymax - self.y0) / size)

        ix, iy = (a.ravel() for a in np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij"))
        boxes = self._boxes(ix, iy)
        active = self._overlapping(boxes, self.outline)
        self.ix, self.iy, self.boxes = ix[active], iy[active], boxes[active]
        self.lookup = np.full((self.nx, self.ny), -1, dtype=np.int64)  # full grid position -> cell id
        self.lookup[self.ix, self.iy] = np.arange(len(self.ix))
        self.midpoints = np.column_stack([self.x0 + (self.ix + 0.5) * size, self.y0 + (self.iy + 0.5) * size])
        self._region_cells: dict[str, np.ndarray] = {}

    @classmethod
    def from_file(cls, regions: str | Path, cell_size: float = 10_000, key: str = "canton") -> Grid:
        return cls(read_regions(regions, key), cell_size)

    def __len__(self) -> int:
        return len(self.ix)

    def __repr__(self) -> str:
        return (f"Grid({len(self)} active cells of {self.cell_size / 1000:g} km, "
                f"{self.nx}x{self.ny} from LV95 ({self.x0:.0f}, {self.y0:.0f}))")

    def _boxes(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        x, y, s = self.x0 + ix * self.cell_size, self.y0 + iy * self.cell_size, self.cell_size
        return shapely.box(x, y, x + s, y + s)

    @staticmethod
    def _overlapping(boxes: np.ndarray, geometry: shapely.Geometry) -> np.ndarray:
        """Boxes sharing area with `geometry` (touching along an edge is not enough)."""
        shapely.prepare(geometry)
        hit = shapely.intersects(geometry, boxes)
        hit[hit] = shapely.area(shapely.intersection(boxes[hit], geometry)) > 0
        return hit

    @cached_property
    def midpoints_latlon(self) -> np.ndarray:
        """Cell midpoints as [n_cells, 2] (lat, lon)."""
        return np.column_stack(lv95_to_wgs84(self.midpoints[:, 0], self.midpoints[:, 1]))

    def cell_of(self, points) -> np.ndarray:
        """Cell id of every LV95 point ([..., 2] as (E, N); numpy or CPU tensor)."""
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 2)
        ix = np.floor((flat[:, 0] - self.x0) / self.cell_size).astype(np.int64)
        iy = np.floor((flat[:, 1] - self.y0) / self.cell_size).astype(np.int64)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        cells = np.full(len(flat), -1, dtype=np.int64)
        cells[inside] = self.lookup[ix[inside], iy[inside]]
        if (outside := cells < 0).any():
            cells[outside] = self.nearest_cell(flat[outside])
        return cells.reshape(points.shape[:-1])

    def nearest_cell(self, points) -> np.ndarray:
        """Id of the active cell whose midpoint is closest to every LV95 point ([n, 2])."""
        points = np.asarray(points, dtype=float)
        return ((points[:, None, :] - self.midpoints[None]) ** 2).sum(-1).argmin(1)

    def cells_in(self, geometry: shapely.Geometry) -> np.ndarray:
        """Ids of the active cells that share area with an LV95 geometry (e.g. a canton)."""
        return np.flatnonzero(self._overlapping(self.boxes.copy(), geometry))

    @cached_property
    def _box_tree(self) -> shapely.STRtree:
        return shapely.STRtree(self.boxes)

    def cell_areas(self, geometry: shapely.Geometry) -> tuple[np.ndarray, np.ndarray]:
        """Ids of the active cells overlapping an LV95 geometry and the overlap area of each (m²)."""
        cells = np.sort(self._box_tree.query(geometry, predicate="intersects"))
        areas = shapely.area(shapely.intersection(self.boxes[cells], geometry))
        return cells[areas > 0], areas[areas > 0]

    def region_cells(self, region: str) -> np.ndarray:
        """Ids of the active cells covering one of `regions` (cached)."""
        if region not in self._region_cells:
            self._region_cells[region] = self.cells_in(self.regions[region])
        return self._region_cells[region]

    def chebyshev(self, a, b) -> np.ndarray:
        """Cell distance max(|dx|, |dy|) between cell ids `a` and `b`: 0 same cell, 1 neighbours, ..."""
        a, b = np.asarray(a), np.asarray(b)
        return np.maximum(np.abs(self.ix[a] - self.ix[b]), np.abs(self.iy[a] - self.iy[b]))

    @staticmethod
    def distance_km(a, b) -> np.ndarray:
        """Distance between LV95 points ([..., 2]) in km. LV95 is conformal and nearly distance-true
        over Switzerland (scale error < 0.05 %), so this matches the great-circle (haversine) distance."""
        return np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=-1) / 1000
