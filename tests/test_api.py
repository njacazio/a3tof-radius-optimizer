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
