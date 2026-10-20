import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import numpy as np
import pytest
from geometry import Geometry, Tracks, make_tracks, evaluate, optimize, module_plot, chip_layout


def test_perfect_wide_full_ring():
    g = Geometry(radius_cm=10, n_staves=80, stave_width_cm=2, stave_length_cm=100, tilt_deg=0,
                 chips_across=1, modules_z=1, chips_per_module_z=1,
                 chip_width_cm=2, chip_length_cm=100, eoc_width_mm=0)
    tr = make_tracks(6000, eta_max=0.5, seed=12)
    m = evaluate(g, tr)
    assert m['eff_all'] == pytest.approx(1.0)
    assert m['eff_reference'] == pytest.approx(1.0)


def test_dead_area_reduces_efficiency():
    g = Geometry(n_staves=30, stave_width_cm=5.42, tilt_deg=0, dead_x_mm=0, dead_z_mm=0)
    tr = make_tracks(12000, eta_max=1.3, seed=4)
    m0 = evaluate(g, tr)
    m1 = evaluate(replace(g, dead_x_mm=0.6, dead_z_mm=0.6), tr)
    assert m1['eff_all'] < m0['eff_all']
    assert m1['dead_only_fraction'] > m0['dead_only_fraction']


def test_radial_central_ray_works_with_tilt():
    g = Geometry(radius_cm=21, n_staves=24, stave_width_cm=5.42, stave_length_cm=129,
                 tilt_deg=3, dead_x_mm=0.1)
    tr = Tracks(np.array([0.01]), np.array([0.0]), np.array([0.0]))
    m = evaluate(g, tr)
    assert m['n_active'] == 1


def test_radius_changes_ideal_reference_eta():
    tr = make_tracks(7000, eta_max=2.0, seed=1)
    small = evaluate(Geometry(radius_cm=18), tr)
    large = evaluate(Geometry(radius_cm=24), tr)
    assert small['ideal_acceptance'] > large['ideal_acceptance']


def test_optimization_uses_valid_candidates():
    g = Geometry()
    tr = make_tracks(800, eta_max=1.5)
    bounds = {'radius_cm': (20, 22), 'n_staves': (23, 27), 'tilt_deg': (0, 6)}
    rows = optimize(g, bounds, tr, n_candidates=12)
    assert rows and rows[0]['score_pp'] >= rows[-1]['score_pp']
    assert all(23 <= c['n_staves'] <= 27 for c in rows)


def test_invalid_chip_dead_area_fails():
    with pytest.raises(ValueError):
        Geometry(dead_x_mm=14).validate()


def test_dead_edges_fixed_in_optimization():
    # Optimization must preserve supplied chip dead edges for EVERY evaluated candidate.
    g = Geometry(dead_x_mm=0.45, dead_z_mm=0.35)
    tr = make_tracks(600, eta_max=1.5, seed=11)
    bounds = {'radius_cm': (20, 23), 'n_staves': (23, 29), 'tilt_deg': (-2, 7)}
    rows = optimize(g, bounds, tr, n_candidates=14)
    assert len(rows) >= 2
    assert all('dead_x_mm' not in row and 'dead_z_mm' not in row for row in rows)
    for row in rows:
        trial = replace(g, radius_cm=row['radius_cm'], n_staves=row['n_staves'], tilt_deg=row['tilt_deg'])
        assert row['eff_all'] == evaluate(trial, tr)['eff_all']


def test_dead_edges_cannot_be_in_scan_bounds():
    g = Geometry(dead_x_mm=0.2)
    tr = make_tracks(300)
    bounds = {'radius_cm': (20, 22), 'n_staves': (23, 27), 'tilt_deg': (0, 6),
              'dead_x_mm': (0, 0.5)}
    with pytest.raises(ValueError, match='dead area'):
        optimize(g, bounds, tr, n_candidates=12)


def test_nominal_chip_dimensions_and_eoc_area():
    g = Geometry()
    g.validate()
    assert (g.chip_width_cm, g.chip_length_cm, g.eoc_width_mm) == (2.6, 3.0, 2.0)
    assert chip_layout(g)['span_u'] == pytest.approx(5.2)
    assert chip_layout(g)['span_z'] == pytest.approx(120.0)
    expected_fraction = (26 * (30 - 2)) / (26 * 30)
    m = evaluate(g, make_tracks(500, seed=7))
    assert m['active_chip_area_fraction'] == pytest.approx(expected_fraction)
    assert m['active_sensor_area_cm2'] == pytest.approx(m['n_chips'] * 2.6 * 2.8)


def test_single_sided_eoc_is_at_positive_z_edge():
    g = Geometry(tilt_deg=0)
    tr = Tracks(np.array([0., 0.]), np.array([0., 0.]),
                np.array([-57.3, -57.1]))
    metrics, masks = evaluate(g, tr, return_masks=True)
    # Chips occupy [-60,60] cm longitudinally, each chip 3 cm.
    # First chip active through z=-57.2, then EOC occupies [-57.2,-57.0].
    assert masks['physical'].tolist() == [True, True]
    assert masks['sensor'].tolist() == [True, True]
    assert masks['active'].tolist() == [True, False]
    assert metrics['n_active'] == 1
    chip = module_plot(g)['chips'][0]
    assert chip['outer'][3] - chip['active'][3] == pytest.approx(0.2)
    assert chip['eoc'][3] - chip['eoc'][1] == pytest.approx(0.2)


def test_chip_is_not_stretched_to_fill_stave():
    g = Geometry()
    lay = chip_layout(g)
    assert lay['span_u'] == pytest.approx(5.2)
    assert lay['span_u'] < g.stave_width_cm
    assert lay['span_z'] == pytest.approx(120.0)
    assert lay['span_z'] < g.stave_length_cm


def test_chip_and_eoc_fixed_in_optimization():
    g = Geometry(eoc_width_mm=2.3, chip_width_cm=2.6, chip_length_cm=3.0)
    tracks = make_tracks(450, eta_max=1.5, seed=2)
    rows = optimize(g, {'radius_cm': [20, 21], 'n_staves': [24, 26],
                        'tilt_deg': [1, 4]}, tracks, n_candidates=11)
    assert rows
    assert all('eoc_width_mm' not in x and 'chip_width_cm' not in x for x in rows)
    for row in rows:
        trial = replace(g, radius_cm=row['radius_cm'], n_staves=row['n_staves'], tilt_deg=row['tilt_deg'])
        assert row['eff_all'] == evaluate(trial, tracks)['eff_all']


def test_eoc_cannot_be_optimized():
    with pytest.raises(ValueError, match='dead area'):
        optimize(Geometry(), {'radius_cm': (20, 22), 'n_staves': (20, 25),
                 'tilt_deg': (0, 4), 'eoc_width_mm': (0, 2)}, make_tracks(200))


def test_chip_layout_requires_room_on_stave():
    with pytest.raises(ValueError, match='larghezza'):
        Geometry(chip_width_cm=3.0).validate()
