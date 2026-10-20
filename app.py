"""Run with: python app.py; open http://127.0.0.1:5000"""
import os
from flask import Flask, jsonify, render_template, request
from geometry import Geometry, make_tracks, evaluate, diagnostic_plots, optimize

app = Flask(__name__)
app.json.sort_keys = False


def read_geom(payload):
    cfg = payload.get('geometry', {})
    if not isinstance(cfg, dict):
        raise ValueError('La geometria deve essere un oggetto JSON.')
    base = Geometry()
    allowed = set(base.__dataclass_fields__)
    extra = set(cfg) - allowed
    if extra:
        raise ValueError('Parametri sconosciuti: ' + ', '.join(sorted(extra)))
    data = {name: cfg.get(name, getattr(base, name)) for name in allowed}
    int_fields = {'n_staves', 'chips_across', 'modules_z', 'chips_per_module_z'}
    for key in int_fields:
        if not float(data[key]).is_integer():
            raise ValueError(f'{key} deve essere un numero intero.')
    data = {k: (int(v) if k in int_fields else float(v)) for k, v in data.items()}
    return Geometry(**data).validate()


def read_tracks(p, default_n=15000):
    n = int(p.get('n_tracks', default_n))
    eta_max = float(p.get('eta_max', 2.0))
    sigma = float(p.get('vertex_sigma_z_cm', 0))
    seed = int(p.get('seed', 12345))
    return make_tracks(n, eta_max, sigma, seed), eta_max


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/defaults')
def defaults():
    return jsonify({'geometry': Geometry().__dict__,
                    'tracks': {'n_tracks': 15000, 'eta_max': 2.0, 'vertex_sigma_z_cm': 0.0, 'seed': 12345}})


@app.post('/api/simulate')
def simulate():
    try:
        p = request.get_json(force=True)
        g = read_geom(p)
        tracks, eta_max = read_tracks(p)
        metrics, masks = evaluate(g, tracks, return_masks=True)
        plots = diagnostic_plots(g, tracks, masks, eta_max)
        return jsonify({'metrics': metrics, 'plots': plots})
    except (ValueError, TypeError, OverflowError, KeyError) as e:
        return jsonify({'error': str(e)}), 400


@app.post('/api/optimize')
def optimize_api():
    try:
        p = request.get_json(force=True)
        g = read_geom(p)
        tracks, _ = read_tracks({**p, 'n_tracks': int(p.get('n_tracks_opt', 4000))}, default_n=4000)
        config = p.get('optimization', {})
        bounds = config.get('bounds', {})
        allowed_bounds = {'radius_cm', 'n_staves', 'tilt_deg'}
        extra = set(bounds) - allowed_bounds
        if extra:
            raise ValueError('Intervalli di ottimizzazione non ammessi: ' + ', '.join(sorted(extra)) + '. La dead area è fissa.')
        bounds = {f: [float(x) for x in bounds.get(f, [getattr(g, f), getattr(g, f)])]
                  for f in ['radius_cm', 'n_staves', 'tilt_deg']}
        rows = optimize(g, bounds, tracks, n_candidates=int(config.get('n_candidates', 80)),
                        stave_penalty_pp=float(config.get('stave_penalty_pp', 0.1)),
                        radius_penalty_pp=float(config.get('radius_penalty_pp', 0.0)),
                        target_radius_cm=float(config.get('target_radius_cm', 21.0)),
                        seed=int(p.get('seed', 12345)))
        if not rows:
            return jsonify({'error': 'Nessuna configurazione valida nei limiti scelti.'}), 400
        return jsonify({'candidates': rows, 'best': rows[0]})
    except (ValueError, TypeError, OverflowError, KeyError) as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', '5000')), debug=False)
