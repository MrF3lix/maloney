import numpy as np
import pandas as pd
import pytest
import shapely

from maloney.baselines.evaluate import NO_REGION, GeoContext, evaluate
from maloney.baselines.methods import LogRegCells, LogRegDialect, NearestPrototype, ProtoNet
from maloney.baselines.units import speaker_scores
from maloney.geo import CellMapping, Geolects, Grid

# Three municipalities, 10 km cells: 1 and 2 in region "east", 3 in "west"; 2 is one cell from 1.
MUNIS = pd.DataFrame({"bfs_nr": [1, 2, 3], "muni_E": [2_605_000.0, 2_615_000.0, 2_655_000.0],
                      "muni_N": [1_205_000.0] * 3, "muni_canton": ["A", "A", "B"],
                      "dialect_region": ["east", "east", "west"]})


@pytest.fixture
def grid():
    return Grid({"A": shapely.box(2_600_000, 1_200_000, 2_630_000, 1_220_000),
                 "B": shapely.box(2_630_000, 1_200_000, 2_660_000, 1_220_000)}, cell_size=10_000)


def make_units(n_per_muni: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Clips whose embedding is the municipality's direction plus noise."""
    rng = np.random.default_rng(seed)
    units = MUNIS.loc[MUNIS.index.repeat(n_per_muni)].reset_index(drop=True)
    units = units.assign(client_id=[f"s{b}_{i % 2}" for i, b in enumerate(units["bfs_nr"])],
                         path=[f"p{i}" for i in range(len(units))])
    x = np.eye(8)[units["bfs_nr"] - 1] * 3 + rng.normal(0, 0.3, (len(units), 8)) + 5  # offset: anisotropy
    return x, units


@pytest.fixture
def geo(grid):
    _, units = make_units(10, 0)
    return GeoContext(grid, CellMapping("centroid"), Geolects.from_meta(units))


@pytest.mark.parametrize("method", [NearestPrototype(), NearestPrototype(center=True), LogRegCells(),
                                    ProtoNet(epochs=30, batch_size=8, device="cpu")])
def test_methods_find_the_cell(method, geo):
    x, units = make_units(20, 1)
    method.fit(x, units, geo)
    x_test, test = make_units(5, 2)
    cells = method.class_cells[method.scores(x_test).argmax(1)]
    metrics, _ = evaluate(cells, test, geo)
    assert metrics["acc@0"] == 1 and metrics["region_acc"] == 1 and metrics["median_km"] == 0


def test_logreg_dialect(geo):
    x, units = make_units(20, 1)
    method = LogRegDialect()
    method.fit(x, units, geo)
    assert method.class_regions.tolist() == ["east", "west"]
    # "east" centre lies between municipalities 1 and 2 (equal clips): 2610 km, the cell of 2.
    assert method.class_cells.tolist() == geo.grid.cell_of([[2_610_000, 1_205_000], [2_655_000, 1_205_000]]).tolist()
    x_test, test = make_units(5, 2)
    best = method.scores(x_test).argmax(1)
    metrics, _ = evaluate(method.class_cells[best], test, geo, method.class_regions[best])
    assert metrics["region_acc"] == 1 and metrics["acc@0"] == pytest.approx(2 / 3)


def test_evaluate_rules(geo):
    _, test = make_units(1, 0)  # one unit per municipality
    grid = geo.grid
    c1, c2, c3 = grid.cell_of(MUNIS[["muni_E", "muni_N"]].to_numpy())
    empty = grid.cell_of([[2_625_000, 1_215_000]])[0]
    # unit 1 -> cell of 2 (same region, 1 cell, 10 km), unit 2 -> empty cell, unit 3 -> cell of 1 (wrong region)
    metrics, pred = evaluate(np.array([c2, empty, c1]), test, geo)
    assert pred["pred_region"].tolist() == ["east", NO_REGION, "east"]
    assert metrics["region_acc"] == pytest.approx(1 / 3) and metrics["acc@0"] == 0
    assert metrics["acc@1"] == pytest.approx(2 / 3) and pred["error_km"].iloc[0] == pytest.approx(10)
    assert metrics["cell_acc"] == 0


def test_speaker_scores():
    units = pd.DataFrame({"client_id": ["a", "b", "a"], "n_clips": [1, 2, 3], "x": [0, 1, 2]})
    scores, speakers = speaker_scores(np.array([[1.0, 0], [0, 1], [0, 1]]), units)
    np.testing.assert_allclose(scores, [[0.5, 0.5], [0, 1]])
    assert speakers["client_id"].tolist() == ["a", "b"] and speakers["n_clips"].tolist() == [4, 2]
