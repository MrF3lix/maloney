import json
from pathlib import Path

import numpy as np
import pytest
import shapely

from maloney.geo import CellMapping, Geolects, Grid, Municipalities, lv95_to_wgs84

GEO_DIR = Path(__file__).resolve().parents[1] / "data" / "geo"
CANTONS = GEO_DIR / "cantons.geojson"


@pytest.fixture
def grid():
    # Two 30 x 20 km "cantons" side by side, not aligned to the 10 km grid.
    a = shapely.box(2_601_000, 1_201_000, 2_631_000, 1_221_000)
    b = shapely.box(2_631_000, 1_201_000, 2_661_000, 1_221_000)
    return Grid({"A": a, "B": b}, cell_size=10_000)


@pytest.fixture
def geolects():
    return Geolects(bfs_nr=np.array([1, 2, 3]), xy=np.array([[2_605_000, 1_205_000], [2_625_000, 1_215_000],
                                                            [2_655_000, 1_205_000]], dtype=float),
                    canton=np.array(["A", "A", "B"]), dialect_region=np.array(["x", "x", "y"]),
                    n_clips=np.array([1, 2, 3]))


def test_grid_cells(grid):
    # Edges at multiples of 10 km: x 2600-2670 km, y 1200-1230 km -> 7 x 3, all overlap the boxes.
    assert (grid.x0, grid.y0, grid.nx, grid.ny) == (2_600_000, 1_200_000, 7, 3)
    assert len(grid) == 21
    np.testing.assert_allclose(grid.midpoints[grid.cell_of([2_605_000, 1_205_000])], [2_605_000, 1_205_000])
    assert grid.cell_of(np.zeros((4, 3, 2))).shape == (4, 3)


def test_point_outside_maps_to_nearest_cell(grid):
    far = grid.cell_of([[2_700_000, 1_205_000]])[0]
    assert (grid.ix[far], grid.iy[far]) == (6, 0)


def test_region_cells_and_distances(grid):
    assert len(grid.region_cells("A")) == 12 and len(grid.region_cells("B")) == 12  # share the x=2630 km column
    a, b = grid.cell_of([[2_605_000, 1_205_000], [2_635_000, 1_225_000]])
    assert grid.chebyshev(a, b) == 3 and grid.chebyshev(a, a) == 0
    assert grid.distance_km([2_600_000, 1_200_000], [2_603_000, 1_204_000]) == pytest.approx(5)


def test_lv95_to_wgs84_bern():
    lat, lon = lv95_to_wgs84(2_600_000, 1_200_000)  # projection centre, the old observatory in Bern
    assert lat == pytest.approx(46.95108, abs=1e-4) and lon == pytest.approx(7.43864, abs=1e-4)


def test_mappings(grid, geolects):
    centroid = CellMapping("centroid").weights(grid, geolects)
    assert (centroid.sum(1) == 1).all()
    gaussian = CellMapping("gaussian", sigma=10_000).weights(grid, geolects)
    assert gaussian[0].argmax() == centroid[0].argmax() and ((gaussian > 0).sum(1) > 1).all()
    with pytest.raises(ValueError):
        CellMapping("nearest")
    with pytest.raises(ValueError):
        CellMapping("uniform")  # needs the municipality polygons


@pytest.fixture
def municipalities(tmp_path):
    """Five 10 km square municipalities in two cantons, offset by 1 km from the grid:

        5 .  .  .        canton A: 1, 2, 3, 5
        1  2  3  4       canton B: 4 (touches 3, but across the canton border)
    """
    boxes = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (0, 1)}
    features = [{"type": "Feature", "properties": {"bfs_nr": b, "municipality": f"m{b}", "canton": "B" if b == 4 else "A"},
                 "geometry": shapely.geometry.mapping(shapely.box(2_601_000 + x * 10_000, 1_201_000 + y * 10_000,
                                                                  2_611_000 + x * 10_000, 1_211_000 + y * 10_000))}
                for b, (x, y) in boxes.items()]
    path = tmp_path / "municipalities.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return Municipalities(path)


def test_neighbourhood(municipalities):
    bfs = lambda idx: municipalities.bfs_nr[idx].tolist()  # noqa: E731
    assert bfs(municipalities.neighbourhood(1)) == [1, 2, 5]
    assert bfs(municipalities.neighbourhood(3)) == [2, 3]  # 4 touches 3 but lies in canton B
    assert bfs(municipalities.neighbourhood(4)) == [4]
    assert bfs(municipalities.neighbourhood(1, distance=10_000)) == [1, 2, 3, 5]
    assert municipalities.region(1).area == pytest.approx(3 * 10_000**2)


def test_uniform_spreads_over_neighbourhood(municipalities):
    grid = Grid({c: shapely.union_all(municipalities.geometry[municipalities.canton == c]) for c in "AB"}, 10_000)
    geolects = Geolects(bfs_nr=np.array([1, 4]), xy=np.array([[2_606_000, 1_206_000], [2_636_000, 1_206_000]], float),
                        canton=np.array(["A", "B"]), dialect_region=np.array(["x", "y"]), n_clips=np.array([2, 1]))
    mapping = CellMapping("uniform", municipalities=municipalities)
    weights = mapping.weights(grid, geolects)
    np.testing.assert_allclose(weights.sum(1), 1)
    # Region of 1 = municipalities 1, 2, 5 (300 km², offset 1 km from the cells): the cell holding most
    # of it has 9 x 9 km of 2 plus a 1 x 9 km strip of 1, i.e. 90 of 300 km².
    assert weights[0].max() == pytest.approx(90 / 300)
    # 4 shares no area with 1's region, and its region stays within 4.
    assert not (weights[0] * weights[1]).any() and (weights[1] > 0).sum() == 4
    wider = CellMapping("uniform", municipalities=municipalities, neighbour_distance=10_000).weights(grid, geolects)
    assert (wider[0] > 0).sum() > (weights[0] > 0).sum()


def test_prototypes(grid, geolects):
    embeddings = np.array([[1, 0], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1]], dtype=float)
    sums, counts = geolects.embedding_sums(embeddings, [1, 2, 2, 3, 3, 3])
    np.testing.assert_allclose(counts, geolects.n_clips)
    for mode in ("centroid", "gaussian"):
        mapping = CellMapping(mode, sigma=10_000)
        prototypes, has_mass = mapping.prototypes(mapping.weights(grid, geolects), sums, counts)
        np.testing.assert_allclose(np.linalg.norm(prototypes[has_mass], axis=1), 1)
        assert (prototypes[~has_mass] == 0).all()
    # Two geolects with weight 1 in the same cell: the clip-weighted mean, normalised.
    mapping = CellMapping("centroid")
    weights = mapping.weights(grid, geolects)
    weights[1] = weights[0]
    prototypes, has_mass = mapping.prototypes(weights, sums, counts)
    np.testing.assert_allclose(prototypes[weights[0].argmax()], np.array([1, 2]) / np.sqrt(5))
    assert [d.tolist() for d in mapping.geolects_of_cells(weights) if len(d)] == [[0, 1], [2]]


@pytest.mark.skipif(not CANTONS.is_file(), reason="needs data/geo/cantons.geojson (scripts/build_zip_geo.py)")
def test_swiss_grid():
    grid = Grid.from_file(CANTONS, cell_size=10_000)
    assert 400 < len(grid) < 600
    assert len(grid.region_cells("BS")) <= 4 and len(grid.region_cells("GR")) > 60
    zurich, geneva = grid.cell_of([[2_683_000, 1_248_000], [2_500_000, 1_118_000]])
    assert grid.chebyshev(zurich, geneva) == 18


@pytest.mark.skipif(not (GEO_DIR / "municipalities.geojson").is_file(), reason="needs data/geo/municipalities.geojson")
def test_swiss_neighbourhoods():
    munis = Municipalities(GEO_DIR / "municipalities.geojson")
    zurich = munis.neighbourhood(261)  # Zürich
    names = set(munis.name[zurich])
    assert {"Zürich", "Dübendorf", "Adliswil"} <= names and (munis.canton[zurich] == "ZH").all()
    basel = munis.neighbourhood(2701)  # Basel: its neighbours Allschwil and Binningen lie in BL
    assert set(munis.name[basel]) == {"Basel", "Riehen"}
