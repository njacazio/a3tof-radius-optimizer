import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import numpy as np
import pytest
from geometry import (Geometry, Tracks, make_tracks, evaluate, optimize, module_plot,
                      chip_layout, stave_clearance_report, StaveClearanceError)


def test_full_ring_by_overlapping_staves_is_now_rejected():
    g = Geometry(radius_cm=10, n_staves=80, stave_width_cm=2, stave_length_cm=100, tilt_deg=0,
                 chips_across=1, modules_z=1, chips_per_module_z=1,
                 chip_width_cm=2, chip_length_cm=100, eoc_width_mm=0)
    with pytest.raises(StaveClearanceError):
        evaluate(g, make_tracks(500, eta_max=0.5, seed=12))


def test_valid_nonoverlapping_coverage():
    g = Geometry(radius_cm=10, n_staves=20, stave_width_cm=2, stave_length_cm=100, tilt_deg=0,
                 chips_across=1, modules_z=1, chips_per_module_z=1,
                 chip_width_cm=2, chip_length_cm=100, eoc_width_mm=0)
    tr = make_tracks(6000, eta_max=0.5, seed=12)
    m = evaluate(g, tr)
    assert 0.5 < m['eff_all'] < 1
    assert m['stave_clearance']['valid']


def test_dead_area_reduces_efficiency():
    g = Geometry(n_staves=24, stave_width_cm=5.42, tilt_deg=3, dead_x_mm=0, dead_z_mm=0)
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


def test_nominal_clearance_and_default_requirement():
    g = Geometry()
    r = stave_clearance_report(g)
    assert g.stave_clearance_mm == 1.0
    assert r['valid']
    assert r['min_clearance_mm'] == pytest.approx(3.0415454313, abs=1.e-6)
    assert len(r['closest_pair']) == 2
    assert r['n_violating_pairs'] == 0
    assert evaluate(g, make_tracks(200))['stave_clearance']['valid']


def test_all_pairs_including_intersections_and_ring_wrap():
    g = Geometry(n_staves=27, tilt_deg=0)
    r = stave_clearance_report(g)
    assert not r['valid']
    assert r['min_clearance_mm'] == pytest.approx(0)
    assert r['n_violating_pairs'] >= g.n_staves
    assert 1 in r['violating_staves'] and g.n_staves in r['violating_staves']
    with pytest.raises(StaveClearanceError) as error:
        g.validate()
    assert error.value.report['n_violating_pairs'] == r['n_violating_pairs']


def test_minimum_gap_is_a_constraint_not_penetration_tolerance():
    g = Geometry(n_staves=24, tilt_deg=0)
    assert stave_clearance_report(g)['min_clearance_mm'] == pytest.approx(1.08468924596)
    assert g.validate()
    stricter = replace(g, stave_clearance_mm=1.2)
    with pytest.raises(StaveClearanceError):
        stricter.validate()
    assert stave_clearance_report(replace(g, stave_clearance_mm=1.0))['valid']


def test_stave_thickness_reduces_clearance_conservatively():
    g = Geometry(n_staves=24, tilt_deg=0, stave_thickness_mm=0.2)
    r = stave_clearance_report(g)
    assert r['min_clearance_mm'] == pytest.approx(0.88468924596)
    assert r['conservative_with_thickness']
    with pytest.raises(StaveClearanceError):
        g.validate()


def test_non_neighbour_pairs_checked():
    g = Geometry(radius_cm=5, n_staves=8, stave_width_cm=30,
                 chips_across=1, chip_width_cm=2.6)
    r = stave_clearance_report(g)
    assert not r['valid']
    assert any(p['stave_b'] - p['stave_a'] > 1 and (p['stave_a'], p['stave_b']) != (1,8)
               for p in r['violating_pairs'])


def test_optimizer_skips_intersections_and_keeps_clearance_fixed():
    g = Geometry(stave_clearance_mm=1.0, stave_thickness_mm=0.05)
    tr = make_tracks(400, eta_max=1.1, seed=19)
    rows, details = optimize(g, {'radius_cm': [21,21], 'n_staves': [24,30], 'tilt_deg': [0,0]},
                             tr, n_candidates=60, return_diagnostics=True)
    assert details['rejected_clearance'] > 0
    assert rows
    assert all(c['min_clearance_mm'] >= g.stave_clearance_mm-1.e-6 for c in rows)
    assert all('stave_clearance_mm' not in c and 'stave_thickness_mm' not in c for c in rows)
    assert all(c['n_staves'] == 24 for c in rows)  # larger counts intersect at this tilt


def test_cannot_optimize_mechanical_constraint():
    bounds = {'radius_cm': (20,22), 'n_staves': (23,27), 'tilt_deg': (0,6),
              'stave_clearance_mm': (0,2)}
    with pytest.raises(ValueError, match='Bounds non ammessi'):
        optimize(Geometry(), bounds, make_tracks(200), n_candidates=10)


def test_tilt_rotates_around_unchanged_stave_center():
    """Rotation may never shift the stave midpoint away from the nominal radius."""
    from geometry import stave_frames, stave_segments
    g = Geometry(radius_cm=21, n_staves=24, phi_offset_deg=8.5)
    midpoints = []
    for angle in (-19.0, 0.0, 3.0, 16.0):
        trial = replace(g, tilt_deg=angle)
        center, axis_u, normal = stave_frames(trial)
        segments = stave_segments(trial)
        assert np.allclose(0.5 * (segments[:, 0] + segments[:, 1]), center, atol=1.e-12)
        assert np.allclose(np.linalg.norm(center, axis=1), 21)
        assert np.allclose(segments[:, 1] - segments[:, 0], trial.stave_width_cm * axis_u)
        assert np.allclose(np.einsum('ij,ij->i', axis_u, normal), 0, atol=1.e-14)
        midpoints.append(center)
    assert all(np.allclose(center, midpoints[0]) for center in midpoints[1:])


def test_module_dimensions_recomputed_from_user_inputs():
    nominal = module_plot(Geometry())
    assert nominal['span_u_cm'] == pytest.approx(5.2)
    assert nominal['module_length_cm'] == pytest.approx(12.0)
    assert nominal['total_modules_length_cm'] == pytest.approx(120.0)
    changed = module_plot(Geometry(chip_gap_u_mm=1.0, chip_gap_z_mm=0.5,
                                   module_gap_z_mm=1.0))
    assert changed['span_u_cm'] == pytest.approx(5.3)
    assert changed['module_length_cm'] == pytest.approx(12.15)
    assert changed['total_modules_length_cm'] == pytest.approx(122.4)
    assert changed['active_area_cm2'] == pytest.approx(nominal['active_area_cm2'])
    # The preview can still draw a mechanical configuration with invalid clearance.
    assert module_plot(Geometry(n_staves=30, tilt_deg=0))['span_u_cm'] == pytest.approx(5.2)


def test_random_track_pair_classification_matches_evaluation():
    from geometry import sample_track_pair
    g = Geometry()
    pairs = [sample_track_pair(g, seed=s) for s in (10, 11, 12)]
    assert sample_track_pair(g, seed=10) == pairs[0]
    assert pairs[0] != pairs[1]
    for pair in pairs:
        assert pair['hit']['efficient'] and not pair['miss']['efficient']
        for cls, is_hit in (('hit', True), ('miss', False)):
            t = pair[cls]
            single = Tracks(np.array([np.deg2rad(t['phi_deg'])]),
                            np.array([t['eta']]), np.array([t['z0_cm']]))
            _, masks = evaluate(g, single, return_masks=True)
            assert bool(masks['reference'][0])  # Miss must not simply leave barrel.
            assert bool(masks['active'][0]) == is_hit
            assert t['x_end_cm']**2 + t['y_end_cm']**2 > g.radius_cm**2


def test_random_track_pair_enforces_clearance():
    from geometry import sample_track_pair
    with pytest.raises(StaveClearanceError):
        sample_track_pair(Geometry(n_staves=30, tilt_deg=0), seed=1)
