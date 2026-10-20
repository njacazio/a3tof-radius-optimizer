"""Fast, geometry-only ALICE 3 iTOF barrel coverage model.

Units: cm for detector dimensions; mm for passive edges; degrees for angles.
Straight tracks originate at (0,0,z0), with uniform phi and eta by default.
This is not a reconstruction, matching or transport efficiency model.
"""
from dataclasses import dataclass, asdict, replace
import math
import numpy as np


@dataclass(frozen=True)
class Geometry:
    radius_cm: float = 21.0
    n_staves: int = 24
    stave_width_cm: float = 5.42
    stave_length_cm: float = 129.0
    tilt_deg: float = 3.0
    chips_across: int = 2
    modules_z: int = 10
    chips_per_module_z: int = 4
    dead_x_mm: float = 0.0
    dead_z_mm: float = 0.0
    phi_offset_deg: float = 0.0

    @property
    def chips_z(self):
        return self.modules_z * self.chips_per_module_z

    def validate(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError('Tutti i parametri geometrici devono essere numeri finiti.')
        if not 1 <= self.radius_cm <= 200 or not 2 <= self.n_staves <= 300:
            raise ValueError('Raggio o numero di stave fuori intervallo.')
        if not 0.05 <= self.stave_width_cm <= 100 or not 1 <= self.stave_length_cm <= 1500:
            raise ValueError('Dimensioni dello stave fuori intervallo.')
        if not -80 <= self.tilt_deg <= 80 or not -360 <= self.phi_offset_deg <= 360:
            raise ValueError('Angolo fuori intervallo.')
        if not 1 <= self.chips_across <= 10 or not 1 <= self.modules_z <= 100 or not 1 <= self.chips_per_module_z <= 20:
            raise ValueError('Segmentazione non valida.')
        if self.dead_x_mm < 0 or self.dead_z_mm < 0:
            raise ValueError('I margini passivi devono essere >= 0.')
        if 2 * self.dead_x_mm >= 10 * self.stave_width_cm / self.chips_across:
            raise ValueError('Margine morto X troppo grande rispetto al chip.')
        if 2 * self.dead_z_mm >= 10 * self.stave_length_cm / self.chips_z:
            raise ValueError('Margine morto Z troppo grande rispetto al chip.')
        return self


@dataclass(frozen=True)
class Tracks:
    phi: np.ndarray
    eta: np.ndarray
    z0: np.ndarray

    @property
    def n(self):
        return len(self.phi)


def make_tracks(n=15000, eta_max=2.0, sigma_z_cm=0.0, seed=12345):
    if not 100 <= n <= 200000 or not 0.1 <= eta_max <= 5 or not 0 <= sigma_z_cm <= 40:
        raise ValueError('Parametri delle tracce fuori intervallo.')
    rng = np.random.default_rng(int(seed))
    return Tracks(
        phi=rng.uniform(0, 2 * np.pi, n),
        eta=rng.uniform(-eta_max, eta_max, n),
        z0=rng.normal(0, sigma_z_cm, n) if sigma_z_cm else np.zeros(n),
    )


def evaluate(geometry: Geometry, tracks: Tracks, return_masks=False):
    """Find intersections with each finite stave plane and its rectangular active chips.

    Handles overlaps by logical OR: a track counts once if any active chip is crossed.
    A chip dead edge is a symmetric margin on *each* side of every chip.
    """
    g = geometry.validate()
    phi, eta, z0 = tracks.phi, tracks.eta, tracks.z0
    n = tracks.n
    active = np.zeros(n, dtype=bool)
    physical = np.zeros(n, dtype=bool)
    sinheta = np.sinh(eta)
    alpha = np.deg2rad(g.tilt_deg)
    radial_projection = g.radius_cm * np.cos(alpha)
    u_center_offset = g.radius_cm * np.sin(alpha)
    pitch_x = g.stave_width_cm / g.chips_across
    pitch_z = g.stave_length_cm / g.chips_z
    dead_x = g.dead_x_mm / 10.0
    dead_z = g.dead_z_mm / 10.0
    offset = np.deg2rad(g.phi_offset_deg)
    for stave in range(g.n_staves):
        theta = offset + stave * (2 * np.pi / g.n_staves) + alpha
        delta = phi - theta
        denom = np.cos(delta)
        # Positive radial direction and finite intersection, no line pointing backwards.
        candidate = denom > 1.e-10
        if not np.any(candidate):
            continue
        t = np.zeros(n)
        np.divide(radial_projection, denom, out=t, where=candidate)
        u = t * np.sin(delta) + u_center_offset
        z = z0 + t * sinheta
        inside = candidate & (np.abs(u) <= g.stave_width_cm / 2) & (np.abs(z) <= g.stave_length_cm / 2)
        physical |= inside
        if not np.any(inside & ~active):
            continue
        xcell = np.mod(u + g.stave_width_cm / 2, pitch_x)
        zcell = np.mod(z + g.stave_length_cm / 2, pitch_z)
        hit = inside & (xcell >= dead_x) & (xcell <= pitch_x - dead_x) & (zcell >= dead_z) & (zcell <= pitch_z - dead_z)
        active |= hit
        if np.all(active):
            break
    # Reference means: an *ideal* continuous cylinder of this radius and length,
    # inside the user's generated eta/z0 sample. NOT all tracks from a physical event.
    reference = np.abs(z0 + g.radius_cm * sinheta) <= g.stave_length_cm / 2
    nref = int(np.count_nonzero(reference))
    nactive = int(np.count_nonzero(active))
    hit_ref = int(np.count_nonzero(active & reference))
    active_w = max(0, pitch_x - 2 * dead_x)
    active_l = max(0, pitch_z - 2 * dead_z)
    metrics = {
        'eff_all': nactive / n,
        'eff_reference': hit_ref / nref if nref else None,
        'ideal_acceptance': nref / n,
        'physical_fraction': float(np.mean(physical)),
        'dead_only_fraction': float(np.mean(physical & ~active)),
        'missed_fraction': float(np.mean(~physical)),
        'n_tracks': n, 'n_ideal': nref,
        'n_active': nactive,
        'n_stave_only': int(np.count_nonzero(physical & ~active)),
        'n_outside': int(np.count_nonzero(~physical)),
        'n_chips': g.n_staves * g.chips_across * g.chips_z,
        'active_sensor_area_cm2': g.n_staves * g.chips_across * g.chips_z * active_w * active_l,
        'active_chip_area_fraction': (active_w * active_l) / (pitch_x * pitch_z),
    }
    if return_masks:
        return metrics, dict(active=active, physical=physical, reference=reference)
    return metrics


def _binned_eff(x, hit, edges):
    total = np.histogram(x, bins=edges)[0]
    found = np.histogram(x, bins=edges, weights=hit.astype(float))[0]
    values = np.divide(found, total, out=np.full(len(total), np.nan), where=total > 0)
    return (0.5 * (edges[:-1] + edges[1:])).tolist(), [float(v) if np.isfinite(v) else None for v in values]


def diagnostic_plots(g, tracks, masks, eta_max):
    phi_deg = np.rad2deg(tracks.phi)
    eta = tracks.eta
    p, eff_phi = _binned_eff(phi_deg, masks['active'], np.linspace(0, 360, 73))
    e, eff_eta = _binned_eff(eta, masks['active'], np.linspace(-eta_max, eta_max, 49))
    pbins = np.linspace(0, 360, 73)
    ebins = np.linspace(-eta_max, eta_max, 49)
    htot, _, _ = np.histogram2d(eta, phi_deg, bins=[ebins, pbins])
    hhit, _, _ = np.histogram2d(eta, phi_deg, bins=[ebins, pbins], weights=masks['active'].astype(float))
    hmap = np.divide(hhit, htot, out=np.full_like(hhit, np.nan), where=htot > 0)
    return {
        'phi': {'x': p, 'y': eff_phi},
        'eta': {'x': e, 'y': eff_eta},
        'heatmap': {'x': (0.5 * (pbins[:-1] + pbins[1:])).tolist(),
                    'y': (0.5 * (ebins[:-1] + ebins[1:])).tolist(),
                    'z': [[float(v) if np.isfinite(v) else None for v in row] for row in hmap]},
        'geometry': geometry_plot(g),
        'module': module_plot(g),
    }


def geometry_plot(g: Geometry):
    g.validate()
    base_x, base_y, active_x, active_y = [], [], [], []
    alpha = math.radians(g.tilt_deg)
    offset = math.radians(g.phi_offset_deg)
    pw = g.stave_width_cm / g.chips_across
    dx = g.dead_x_mm / 10
    for i in range(g.n_staves):
        phi = offset + 2 * math.pi * i / g.n_staves
        theta = phi + alpha
        center = np.array([g.radius_cm * math.cos(phi), g.radius_cm * math.sin(phi)])
        u = np.array([-math.sin(theta), math.cos(theta)])
        endpoints = [center + u * (j * g.stave_width_cm / 2) for j in (-1, 1)]
        base_x.extend([endpoints[0][0], endpoints[1][0], None])
        base_y.extend([endpoints[0][1], endpoints[1][1], None])
        for ch in range(g.chips_across):
            low = -g.stave_width_cm / 2 + ch * pw + dx
            high = -g.stave_width_cm / 2 + (ch + 1) * pw - dx
            a, b = center + u * low, center + u * high
            active_x.extend([a[0], b[0], None])
            active_y.extend([a[1], b[1], None])
    return {'staves_x': base_x, 'staves_y': base_y, 'active_x': active_x, 'active_y': active_y,
            'circle_x': (g.radius_cm * np.cos(np.linspace(0, 2 * np.pi, 361))).tolist(),
            'circle_y': (g.radius_cm * np.sin(np.linspace(0, 2 * np.pi, 361))).tolist()}



def module_plot(g: Geometry):
    """Rectangles representing one repeated 2D module in local stave u-z coordinates."""
    chip_u = g.stave_width_cm / g.chips_across
    length = g.stave_length_cm / g.modules_z
    chip_z = length / g.chips_per_module_z
    dx, dz = g.dead_x_mm / 10, g.dead_z_mm / 10
    chips = []
    for i in range(g.chips_across):
        for j in range(g.chips_per_module_z):
            ulo = -g.stave_width_cm / 2 + i * chip_u
            zlo = -length / 2 + j * chip_z
            chips.append({
                'outer': [ulo, zlo, ulo + chip_u, zlo + chip_z],
                'active': [ulo + dx, zlo + dz, ulo + chip_u - dx, zlo + chip_z - dz],
            })
    return {'chips': chips, 'width': g.stave_width_cm, 'length': length}


def optimize(g, bounds, tracks, n_candidates=80, stave_penalty_pp=0.1, radius_penalty_pp=0.0, target_radius_cm=21.0, seed=27):
    """Seeded random search + baseline, with common sampled tracks for fair comparison.

    Score uses absolute efficiency in percentage points, minus declared penalties.
    Dead-edge margins and all other sensor parameters are fixed throughout
    the search. Only radius, stave count and tilt are optimized.
    """
    if not 10 <= n_candidates <= 300:
        raise ValueError('Numero di configurazioni richieste fuori intervallo (10-300).')
    if not 0 <= stave_penalty_pp <= 5 or not 0 <= radius_penalty_pp <= 5:
        raise ValueError('Penalita fuori intervallo.')
    fields = ['radius_cm', 'n_staves', 'tilt_deg']
    unexpected = set(bounds) - set(fields)
    if unexpected:
        raise ValueError('Bounds non ammessi (la dead area e le altre proprietà del sensore restano fisse): ' + ', '.join(sorted(unexpected)))
    for field in fields:
        low, high = bounds[field]
        if not np.isfinite([low, high]).all() or low > high:
            raise ValueError(f'Limiti invalidi per {field}.')
    rng = np.random.default_rng(seed)
    rows = []
    seen = set()
    for field in fields:
        if field == 'n_staves' and math.ceil(bounds[field][0]) > math.floor(bounds[field][1]):
            raise ValueError('Intervallo del numero di stave senza valori interi.')
    baseline = {f: getattr(g, f) for f in fields}
    choices = [baseline] if all(bounds[f][0] <= baseline[f] <= bounds[f][1] for f in fields) else []
    for _ in range(n_candidates - len(choices)):
        candidate = {}
        for field in fields:
            low, high = bounds[field]
            candidate[field] = int(rng.integers(math.ceil(low), math.floor(high) + 1)) if field == 'n_staves' else float(rng.uniform(low, high))
        choices.append(candidate)
    for params in choices:
        key = tuple(round(float(params[f]), 4) for f in fields)
        if key in seen:
            continue
        seen.add(key)
        try:
            trial = replace(g, **params).validate()
            m = evaluate(trial, tracks)
        except ValueError:
            continue
        score_pp = 100 * m['eff_all'] - stave_penalty_pp * trial.n_staves - radius_penalty_pp * abs(trial.radius_cm - target_radius_cm)
        rows.append({**{f: getattr(trial, f) for f in fields},
                     'eff_all': m['eff_all'], 'eff_reference': m['eff_reference'],
                     'n_chips': m['n_chips'], 'score_pp': score_pp})
    rows.sort(key=lambda r: r['score_pp'], reverse=True)
    return rows


def config_dict(g):
    return asdict(g)
