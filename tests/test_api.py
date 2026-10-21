import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip('flask')
from app import app


def test_optimize_api_preserves_fixed_dead_area():
    client = app.test_client()
    payload = {
        'geometry': {'dead_x_mm': 0.4, 'dead_z_mm': 0.6},
        'n_tracks_opt': 500,
        'optimization': {
            'n_candidates': 10,
            'bounds': {'radius_cm': [20, 21], 'n_staves': [23, 24], 'tilt_deg': [0, 4]}
        }
    }
    response = client.post('/api/optimize', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['candidates']
    assert all('dead_x_mm' not in row and 'dead_z_mm' not in row for row in data['candidates'])


def test_optimize_api_rejects_dead_area_bounds():
    client = app.test_client()
    response = client.post('/api/optimize', json={
        'optimization': {'bounds': {'dead_z_mm': [0, 1]}}
    })
    assert response.status_code == 400
    assert 'dead area' in response.get_json()['error']


def test_simulate_rejects_overlapping_staves_and_returns_visualization():
    client = app.test_client()
    response = client.post('/api/simulate', json={
        'geometry': {'n_staves': 27, 'tilt_deg': 0.0, 'stave_clearance_mm': 1.0},
        'n_tracks': 300
    })
    assert response.status_code == 422
    data = response.get_json()
    assert data['clearance']['n_violating_pairs'] > 0
    assert not data['clearance']['valid']
    assert data['geometry']['invalid_x']
    assert 'metrics' not in data


def test_simulate_reports_nominal_stave_clearance():
    client = app.test_client()
    response = client.post('/api/simulate', json={'n_tracks': 300})
    assert response.status_code == 200
    m = response.get_json()['metrics']
    assert m['stave_clearance']['valid']
    assert m['stave_clearance']['required_clearance_mm'] == 1.0


def test_optimizer_skips_stave_overlaps_when_baseline_invalid():
    client = app.test_client()
    response = client.post('/api/optimize', json={
        'geometry': {'n_staves': 27, 'tilt_deg': 0},
        'n_tracks_opt': 300,
        'optimization': {'n_candidates': 10, 'bounds': {
            'radius_cm': [21,21], 'n_staves': [24,27], 'tilt_deg': [0,0]
        }}
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['diagnostics']['rejected_clearance'] > 0
    assert all(row['min_clearance_mm'] >= 1 for row in data['candidates'])


def test_optimizer_does_not_allow_scanning_stave_clearance():
    client = app.test_client()
    response = client.post('/api/optimize', json={
        'optimization': {'bounds': {'stave_clearance_mm': [0, 2]}}
    })
    assert response.status_code == 400


def test_preview_updates_module_dimensions_and_tilt_center():
    client = app.test_client()
    response = client.post('/api/preview', json={'geometry': {
        'chip_gap_u_mm': 1.0, 'chip_gap_z_mm': 0.5,
        'module_gap_z_mm': 1.0, 'tilt_deg': 10.0}})
    assert response.status_code == 200
    data = response.get_json()
    assert data['module']['span_u_cm'] == pytest.approx(5.3)
    assert data['module']['module_length_cm'] == pytest.approx(12.15)
    assert len(data['geometry']['center_x']) == 24
    assert data['clearance']['required_clearance_mm'] == 1.0


def test_preview_draws_invalid_stave_clearance_without_simulating():
    client = app.test_client()
    response = client.post('/api/preview', json={'geometry': {'n_staves': 27, 'tilt_deg': 0}})
    assert response.status_code == 200
    assert not response.get_json()['clearance']['valid']


def test_pair_api_returns_a_new_hit_and_miss():
    client = app.test_client()
    response = client.post('/api/sample-tracks', json={'sample_seed': 31})
    assert response.status_code == 200
    data = response.get_json()
    assert data['hit']['efficient'] is True
    assert data['miss']['efficient'] is False
    assert client.post('/api/sample-tracks', json={'sample_seed': 31}).get_json() == data
    assert client.post('/api/sample-tracks', json={'sample_seed': 32}).get_json() != data


def test_pair_api_rejects_invalid_clearance():
    client = app.test_client()
    response = client.post('/api/sample-tracks', json={'geometry': {'n_staves': 27, 'tilt_deg': 0}})
    assert response.status_code == 400
