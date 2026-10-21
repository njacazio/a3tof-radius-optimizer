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
    # Chip orientation: 26 mm across the stave (u), 30 mm along the beam (z).
    chip_width_cm: float = 2.6
    chip_length_cm: float = 3.0
    # One asymmetric end-of-column band, along the 26 mm edge at +z.
    eoc_width_mm: float = 2.0
    # Optional mechanical gaps. Zero until measured dimensions are supplied.
    chip_gap_u_mm: float = 0.0
    chip_gap_z_mm: float = 0.0
    module_gap_z_mm: float = 0.0
    dead_x_mm: float = 0.0
    dead_z_mm: float = 0.0
    phi_offset_deg: float = 0.0
    # Fixed mechanical constraint. Require at least this much separation in XY.
    stave_clearance_mm: float = 1.0
    # Optional stave thickness, measured perpendicular to the stave's long XY axis.
    # Zero is an ideal infinitely thin stave until a mechanical thickness is known.
    # Physical clearance is conservatively bounded by (axis-segment distance - thickness).
    stave_thickness_mm: float = 0.0

    @property
    def chips_z(self):
        return self.modules_z * self.chips_per_module_z

    def validate(self, check_clearance=True):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError('Tutti i parametri geometrici devono essere numeri finiti.')
        if not 1 <= self.radius_cm <= 200 or not 2 <= self.n_staves <= 300:
            raise ValueError('Raggio o numero di stave fuori intervallo.')
        if not 0.05 <= self.stave_width_cm <= 100 or not 1 <= self.stave_length_cm <= 1500:
            raise ValueError('Dimensioni dello stave fuori intervallo.')
        if not -80 <= self.tilt_deg <= 80 or not -360 <= self.phi_offset_deg <= 360:
            raise ValueError('Angolo fuori intervallo.')
        if not 0 <= self.stave_clearance_mm <= 100 or not 0 <= self.stave_thickness_mm <= 100:
            raise ValueError('Distanza minima tra stave e spessore devono essere tra 0 e 100 mm.')
        if not 1 <= self.chips_across <= 10 or not 1 <= self.modules_z <= 100 or not 1 <= self.chips_per_module_z <= 20:
            raise ValueError('Segmentazione non valida.')
        if self.chip_width_cm <= 0 or self.chip_length_cm <= 0:
            raise ValueError('Le dimensioni del chip devono essere positive.')
        if any(v < 0 for v in (self.dead_x_mm, self.dead_z_mm, self.eoc_width_mm,
                              self.chip_gap_u_mm, self.chip_gap_z_mm, self.module_gap_z_mm)):
            raise ValueError('Bordi inattivi e distanze tra chip/moduli devono essere >= 0.')
        if 2 * self.dead_x_mm >= 10 * self.chip_width_cm:
            raise ValueError('Margine morto X troppo grande rispetto al chip.')
        if self.eoc_width_mm + 2 * self.dead_z_mm >= 10 * self.chip_length_cm:
            raise ValueError('End-of-column + margini Z troppo grandi rispetto al chip.')
        lay = chip_layout(self)
        if lay['span_u'] > self.stave_width_cm + 1.e-9:
            raise ValueError('Il numero di chip con le dimensioni/gap indicati supera la larghezza dello stave.')
        if lay['span_z'] > self.stave_length_cm + 1.e-9:
            raise ValueError('Il numero di chip con le dimensioni/gap indicati supera la lunghezza dello stave.')
        if check_clearance:
            report = stave_clearance_report(self)
            if not report['valid']:
                raise StaveClearanceError(report)
        return self



class StaveClearanceError(ValueError):
    """Invalid mechanical layout, with a structured diagnostic for the UI."""

    def __init__(self, report):
        self.report = report
        pair = report['closest_pair']
        super().__init__(
            f"Distanza insufficiente tra stave: {report['min_clearance_mm']:.3f} mm "
            f"(coppia {pair[0]}–{pair[1]}); richiesta almeno "
            f"{report['required_clearance_mm']:.3f} mm. "
            f"Coppie non conformi: {report['n_violating_pairs']}."
        )


def stave_segments(g: Geometry):
    """2D centerline segments of all mechanical staves, shape (n,2,2), cm."""
    phi = np.deg2rad(g.phi_offset_deg) + np.arange(g.n_staves) * (2 * np.pi / g.n_staves)
    theta = phi + np.deg2rad(g.tilt_deg)
    centers = g.radius_cm * np.column_stack((np.cos(phi), np.sin(phi)))
    tangent = np.column_stack((-np.sin(theta), np.cos(theta)))
    half = 0.5 * g.stave_width_cm * tangent
    return np.stack((centers - half, centers + half), axis=1)


def stave_clearance_report(g: Geometry, max_pairs_reported=20):
    """Check *all* distinct stave pairs, including non-neighbours and ring wraparound.

    Exact minimum distance between finite, zero-thickness XY centerline segments.
    For finite thickness, subtract stave_thickness_mm as a conservative lower bound
    on the actual edge-to-edge clearance of the corresponding XY rectangles:
    any physical stave rectangle lies within the capsule of radius thickness/2.
    Negative clearance means that the inflated envelopes overlap. This is a
    sufficient but, for thick staves, not necessary condition for non-intersection.

    `stave_clearance_mm` is a *minimum allowed gap*, NOT an allowed penetration.
    Staves with zero thickness that intersect have separation zero and fail if the
    required gap is positive. No Z dependence; all staves share the same Z range.
    """
    if g.n_staves < 2:
        raise ValueError('Occorrono almeno due stave.')
    segs = stave_segments(g)
    i, j = np.triu_indices(g.n_staves, 1)
    p, p1 = segs[i, 0, :], segs[i, 1, :]
    q, q1 = segs[j, 0, :], segs[j, 1, :]
    r, s = p1 - p, q1 - q

    def cross(a, b):
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    def point_segment_distance(points, origins, directions):
        denom = np.einsum('ij,ij->i', directions, directions)
        projection = np.einsum('ij,ij->i', points - origins, directions) / denom
        closest = origins + np.clip(projection, 0, 1)[:, None] * directions
        return np.linalg.norm(points - closest, axis=1)

    d = np.minimum.reduce([
        point_segment_distance(p, q, s), point_segment_distance(p1, q, s),
        point_segment_distance(q, p, r), point_segment_distance(q1, p, r),
    ])
    div = cross(r, s)
    qp = q - p
    nonparallel = np.abs(div) > 1.e-12
    t = np.divide(cross(qp, s), div, out=np.full(len(div), -1.), where=nonparallel)
    u = np.divide(cross(qp, r), div, out=np.full(len(div), -1.), where=nonparallel)
    crossing = nonparallel & (t >= -1.e-12) & (t <= 1+1.e-12) & (u >= -1.e-12) & (u <= 1+1.e-12)
    d[crossing] = 0.
    clear_mm = d * 10 - g.stave_thickness_mm
    min_id = int(np.argmin(clear_mm))
    # Numerical epsilon: 1 nm in this centimeter-scale model.
    bad = np.flatnonzero(clear_mm < g.stave_clearance_mm - 1.e-6)
    entries = [{'stave_a': int(i[k]+1), 'stave_b': int(j[k]+1),
                'clearance_mm': float(clear_mm[k])} for k in bad[:max_pairs_reported]]
    return {
        'valid': len(bad) == 0,
        'min_clearance_mm': float(clear_mm[min_id]),
        'required_clearance_mm': g.stave_clearance_mm,
        'stave_thickness_mm': g.stave_thickness_mm,
        'closest_pair': [int(i[min_id]+1), int(j[min_id]+1)],
        'n_violating_pairs': int(len(bad)),
        'violating_pairs': entries,
        'violating_staves': sorted(set((i[bad]+1).tolist() + (j[bad]+1).tolist())),
        'conservative_with_thickness': g.stave_thickness_mm > 0,
    }


def chip_layout(g: Geometry):
    """Fixed-chip tiling, centered inside a larger mechanical stave envelope.

    Local u: transverse chip edge 26 mm; local z: chip edge 30 mm.
    EOC occupies the last eoc_width_mm in +z on *each* chip.
    """
    du = g.chip_gap_u_mm / 10.0
    dz = g.chip_gap_z_mm / 10.0
    dm = g.module_gap_z_mm / 10.0
    step_u = g.chip_width_cm + du
    step_z = g.chip_length_cm + dz
    module_length = g.chips_per_module_z * g.chip_length_cm + (g.chips_per_module_z - 1) * dz
    module_pitch = module_length + dm
    span_u = g.chips_across * g.chip_width_cm + (g.chips_across - 1) * du
    span_z = g.modules_z * module_length + (g.modules_z - 1) * dm
    return dict(step_u=step_u, step_z=step_z, module_length=module_length,
                module_pitch=module_pitch, span_u=span_u, span_z=span_z)


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
    """Straight-line intersections with finite staves and independently sized chips.

    Each chip has 26 mm transverse width, 30 mm longitudinal length by default.
    Its 2 mm end-of-column is a *single* inactive strip at its +z end.
    Overlapping staves recover hits (logical OR). Mechanical margins and optional
    inter-chip/module gaps remain insensitive, rather than stretching chip sizes.
    """
    g = geometry.validate()
    lay = chip_layout(g)
    phi, eta, z0 = tracks.phi, tracks.eta, tracks.z0
    n = tracks.n
    active = np.zeros(n, dtype=bool)
    physical = np.zeros(n, dtype=bool)
    sensor = np.zeros(n, dtype=bool)
    eoc_only = np.zeros(n, dtype=bool)
    sinheta = np.sinh(eta)
    alpha = np.deg2rad(g.tilt_deg)
    radial_projection = g.radius_cm * np.cos(alpha)
    u_center_offset = g.radius_cm * np.sin(alpha)
    dead_u = g.dead_x_mm / 10.0
    dead_z = g.dead_z_mm / 10.0
    eoc = g.eoc_width_mm / 10.0
    offset = np.deg2rad(g.phi_offset_deg)
    for stave in range(g.n_staves):
        theta = offset + stave * (2 * np.pi / g.n_staves) + alpha
        delta = phi - theta
        denom = np.cos(delta)
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

        # Chips occupy a known-sized, centered subarray of the mechanical stave.
        coord_u = u + lay['span_u'] / 2
        col = np.floor(coord_u / lay['step_u'])
        local_u = coord_u - col * lay['step_u']
        chip_u = (col >= 0) & (col < g.chips_across) & (local_u >= 0) & (local_u < g.chip_width_cm)

        coord_z = z + lay['span_z'] / 2
        mod = np.floor(coord_z / lay['module_pitch'])
        local_module_z = coord_z - mod * lay['module_pitch']
        row = np.floor(local_module_z / lay['step_z'])
        local_z = local_module_z - row * lay['step_z']
        chip_z = ((mod >= 0) & (mod < g.modules_z) &
                  (row >= 0) & (row < g.chips_per_module_z) &
                  (local_z >= 0) & (local_z < g.chip_length_cm))
        on_chip = inside & chip_u & chip_z
        sensor |= on_chip
        is_eoc = on_chip & (local_z >= g.chip_length_cm - eoc)
        eoc_only |= is_eoc
        hit = (on_chip & (local_u >= dead_u) & (local_u < g.chip_width_cm - dead_u) &
               (local_z >= dead_z) & (local_z < g.chip_length_cm - eoc - dead_z))
        active |= hit
        if np.all(active):
            break

    # Relative to an ideal continuous barrel, for the selected (phi,eta,z0) sample.
    reference = np.abs(z0 + g.radius_cm * sinheta) <= g.stave_length_cm / 2
    nref = int(np.count_nonzero(reference))
    nactive = int(np.count_nonzero(active))
    active_chip_width = g.chip_width_cm - 2 * dead_u
    active_chip_length = g.chip_length_cm - eoc - 2 * dead_z
    n_chips = g.n_staves * g.chips_across * g.chips_z
    metrics = {
        'eff_all': nactive / n,
        'eff_reference': int(np.count_nonzero(active & reference)) / nref if nref else None,
        'ideal_acceptance': nref / n,
        'physical_fraction': float(np.mean(physical)),
        'sensor_fraction': float(np.mean(sensor)),
        'dead_only_fraction': float(np.mean(physical & ~active)),
        'eoc_loss_fraction': float(np.mean(eoc_only & ~active)),
        'missed_fraction': float(np.mean(~physical)),
        'n_tracks': n, 'n_ideal': nref,
        'n_active': nactive,
        'n_stave_only': int(np.count_nonzero(physical & ~active)),
        'n_outside': int(np.count_nonzero(~physical)),
        'n_chips': n_chips,
        'active_sensor_area_cm2': n_chips * active_chip_width * active_chip_length,
        'active_chip_area_fraction': (active_chip_width * active_chip_length) / (g.chip_width_cm * g.chip_length_cm),
        'chip_area_cm2': g.chip_width_cm * g.chip_length_cm,
        'stave_clearance': stave_clearance_report(g),
    }
    if return_masks:
        return metrics, dict(active=active, physical=physical, sensor=sensor, reference=reference)
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
    g.validate(check_clearance=False)
    lay = chip_layout(g)
    base_x, base_y, active_x, active_y = [], [], [], []
    alpha = math.radians(g.tilt_deg)
    offset = math.radians(g.phi_offset_deg)
    dead_u = g.dead_x_mm / 10
    for i in range(g.n_staves):
        phi = offset + 2 * math.pi * i / g.n_staves
        theta = phi + alpha
        center = np.array([g.radius_cm * math.cos(phi), g.radius_cm * math.sin(phi)])
        axis_u = np.array([-math.sin(theta), math.cos(theta)])
        a, b = (center + axis_u * (j * g.stave_width_cm / 2) for j in (-1, 1))
        base_x.extend([a[0], b[0], None])
        base_y.extend([a[1], b[1], None])
        for ch in range(g.chips_across):
            low = -lay['span_u'] / 2 + ch * lay['step_u'] + dead_u
            high = -lay['span_u'] / 2 + ch * lay['step_u'] + g.chip_width_cm - dead_u
            a, b = center + axis_u * low, center + axis_u * high
            active_x.extend([a[0], b[0], None])
            active_y.extend([a[1], b[1], None])
    report = stave_clearance_report(g)
    bad_x, bad_y = [], []
    segments = stave_segments(g)
    for stave in report['violating_staves']:
        aa, bb = segments[stave-1]
        bad_x.extend([float(aa[0]), float(bb[0]), None])
        bad_y.extend([float(aa[1]), float(bb[1]), None])
    return {'staves_x': base_x, 'staves_y': base_y, 'active_x': active_x, 'active_y': active_y,
            'invalid_x': bad_x, 'invalid_y': bad_y,
            'circle_x': (g.radius_cm * np.cos(np.linspace(0, 2 * np.pi, 361))).tolist(),
            'circle_y': (g.radius_cm * np.sin(np.linspace(0, 2 * np.pi, 361))).tolist()}


def module_plot(g: Geometry):
    """One 2D module showing chips, single-sided EOC bands, and optional gaps."""
    g.validate()
    lay = chip_layout(g)
    dz0 = g.dead_z_mm / 10
    du0 = g.dead_x_mm / 10
    eoc = g.eoc_width_mm / 10
    chips = []
    for i in range(g.chips_across):
        for j in range(g.chips_per_module_z):
            ulo = -lay['span_u'] / 2 + i * lay['step_u']
            zlo = -lay['module_length'] / 2 + j * lay['step_z']
            chips.append({
                'outer': [ulo, zlo, ulo + g.chip_width_cm, zlo + g.chip_length_cm],
                'active': [ulo + du0, zlo + dz0, ulo + g.chip_width_cm - du0,
                           zlo + g.chip_length_cm - eoc - dz0],
                'eoc': [ulo, zlo + g.chip_length_cm - eoc,
                        ulo + g.chip_width_cm, zlo + g.chip_length_cm],
            })
    return {'chips': chips, 'width': g.stave_width_cm, 'length': lay['module_length'],
            'eoc_width_mm': g.eoc_width_mm}


def optimize(g, bounds, tracks, n_candidates=80, stave_penalty_pp=0.1, radius_penalty_pp=0.0, target_radius_cm=21.0, seed=27, return_diagnostics=False):
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
    rejected_clearance = 0
    rejected_geometry = 0
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
        except StaveClearanceError:
            rejected_clearance += 1
            continue
        except ValueError:
            rejected_geometry += 1
            continue
        score_pp = 100 * m['eff_all'] - stave_penalty_pp * trial.n_staves - radius_penalty_pp * abs(trial.radius_cm - target_radius_cm)
        rows.append({**{f: getattr(trial, f) for f in fields},
                     'eff_all': m['eff_all'], 'eff_reference': m['eff_reference'],
                     'n_chips': m['n_chips'], 'score_pp': score_pp,
                     'min_clearance_mm': m['stave_clearance']['min_clearance_mm']})
    rows.sort(key=lambda r: r['score_pp'], reverse=True)
    if return_diagnostics:
        return rows, {'evaluated': len(rows), 'rejected_clearance': rejected_clearance,
                      'rejected_geometry': rejected_geometry,
                      'duplicate_candidates': len(choices) - len(seen)}
    return rows


def config_dict(g):
    return asdict(g)
