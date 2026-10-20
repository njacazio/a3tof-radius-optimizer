# iTOF Geometry Lab (ALICE 3)

A local Python + HTML dashboard for the geometric optimization of the **ALICE 3 inner TOF barrel**.

## Start

```bash
cd itof-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000/** in your browser.

Python back end: `geometry.py`, `app.py`. HTML/JS front end: `templates/index.html`. Plotly JS is bundled in `static/` for offline plots. No ROOT/O2 dependency.

On the page:

1. Set the three design variables: radius, stave count and tilt angle. In the separate **fixed sensor parameters** section, set stave width/length, segmentation and passive edges (in mm **per side**, independently in transverse X and longitudinal Z). These remain unchanged throughout each optimization run.
2. Set generated track sample: flat phi, flat eta, vertex Gaussian z-spread, random seed.
3. Click **Simula geometria** to see transverse stave coverage, phi/eta efficiency, and an eta-vs-phi map.
4. Choose search ranges and use **Ottimizza**. The seeded random search evaluates candidates using the *same* generated tracks; rank by an objective including optionally a cost penalty per stave and deviation from preferred radius. Click **Usa** to rerun a chosen design with higher statistics.
5. Save and load configurations as JSON; export the full optimization table as CSV. Inspect the repeated module chip map, including inactive borders. Adjust `geometry.py` if changing the physics model.

## Definitions

A track is a straight line from `(0,0,z0)` parametrized by `(phi, eta)`. Each iTOF stave is a finite plane at radius `r` and rotated by `phi_stave + 90° + tilt` in XY, matching the placement convention in ALICE O2. The script intersects each track with all finite stave planes and tests its local coordinates against chip segmentation and symmetric dead edges. Overlapping staves can recover lost hits. A track is counted once if at least one active chip is intercepted.

- **Global geometric efficiency**: tracks with ≥1 active chip intersection / all generated tracks (`phi ∈ [0,2π)`, `eta ∈ [-eta_max,eta_max]`).
- **Ideal-reference efficiency**: tracks with ≥1 active chip intersection **and** intersecting an ideal cylindrical barrel within `±L/2` / number of generated tracks intersecting that ideal cylinder.
- **Ideal acceptance**: generated tracks intersecting the ideal continuous barrel / generated tracks.
- **Dead-only**: tracks crossing any physical stave but no active area / generated tracks.

These are *geometric* efficiencies; they do **not** include intrinsic chip efficiency, material effects, magnetic bending, multiple scattering, hit merging, reconstruction, or time-to-track matching. In particular, a geometrically optimal radius does not imply optimal TOF PID or tracking. The dead-edge parameters are fixed chip properties: they enter the geometric efficiency calculation, but are never scanned or optimized. To study a different chip design, change them in the fixed sensor section and rerun the search.

### Baseline values

Extracted from the ALICE O2 `dev` code on 20 October 2026:

- `r = 21 cm`, `L = 129 cm`, `24` staves
- stave width `5.42 cm`, stave tilt `3°`
- `10` modules per stave, each with `2 × 4` chips (2 transverse, 4 longitudinal)
- passive edges default to `0 mm` as **illustrative** settings; set the measured physical values before interpreting any result. `ChipSpecifics` contains passive-edge settings, but do not assume symmetric values are the final physical chip implementation. Dead area is **not** an optimization degree of freedom.

Source code:

- https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/Upgrades/ALICE3/IOTOF/simulation/src/Detector.cxx
- https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/Upgrades/ALICE3/IOTOF/simulation/src/Layer.cxx
- https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/Upgrades/ALICE3/IOTOF/base/include/IOTOFBase/IOTOFBaseParam.h

The earlier ALICE 3 conceptual study quoted `r = 19 cm`; this app intentionally uses the `dev` simulation default (`21 cm`) rather than assuming these different design snapshots are identical.

## Tests

```bash
pip install pytest
pytest -q
```

## Scope for next iteration

- Import `TGeo` / real sensor placements from `o2sim_geometry.root` and use asymmetric chip readout/top/side dead edges.
- Import track samples or O2 `MCTrack` branches; helix/ACTS propagation in B field.
- Compute reconstruction efficiency and matching conditional on `pT, eta, phi, charge`; compare with geometric upper bounds.
- Incorporate engineering constraints: maximum radius/occupied volume, material, stave budget/cost, minimum overlap and nonideal sensor response.

## REST API

- `GET /api/defaults`
- `POST /api/simulate` body `{ "geometry": { ... }, "n_tracks": 15000, "eta_max": 2, "vertex_sigma_z_cm": 0, "seed": 12345 }`
- `POST /api/optimize` same body plus `{ "optimization": { "bounds": { "radius_cm": [18,25], "n_staves": [20,32], "tilt_deg": [-5,15] }, "n_candidates": 80, "stave_penalty_pp": 0.1 } }`

The optimizer accepts bounds **only** for `radius_cm`, `n_staves`, and `tilt_deg`. Other geometry fields, including `dead_x_mm` and `dead_z_mm`, are taken from the submitted `geometry` object and are fixed during the full run. Bounds on dead area are explicitly rejected to catch stale clients.

Local use only: the web server listens to `127.0.0.1`, is not intended for public exposure.
