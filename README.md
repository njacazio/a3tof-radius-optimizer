# iTOF Geometry Lab — ALICE 3

Local Python + Flask web app for exploring and optimizing geometrical acceptance of an ALICE 3 iTOF barrel layer.

## Run (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000/**. JavaScript plots work offline: Plotly is bundled under `static/`.

## Geometry defaults

### Design variables (scanned)

| Parameter | Default |
| --- | ---: |
| Cylinder radius | 21 cm |
| Number of staves | 24 |
| Stave tilt | 3° |

### Fixed stave and chip properties

| Parameter | Default |
| --- | ---: |
| Mechanical stave width | 5.42 cm |
| Mechanical stave length | 129 cm |
| Chips across stave (`u`) | 2 |
| Modules along beam (`z`) | 10 |
| Chips per module along `z` | 4 |
| **Individual chip size** | **2.6 cm (u) × 3.0 cm (z)** |
| **End-of-column strip** | **2.0 mm, single edge at +z** |
| Extra inactive borders (each side) | 0 mm in both `u` and `z` |
| Inter-chip gaps in `u` and `z` | 0 mm (not yet supplied) |
| Inter-module gap in `z` | 0 mm (not yet supplied) |

The **end-of-column runs along the 26 mm chip edge**, and extends 2 mm *inward along its 30 mm dimension*. The default active rectangle is therefore **26 × 28 mm = 728 mm²** out of 780 mm²: **93.33% active chip fraction**, before any extra dead borders and gaps.

With 2 chips across and 40 chips longitudinally, the array envelope is **5.2 cm × 120 cm** (for zero gaps). It is centered in the larger 5.42 × 129 cm stave. The mechanically unoccupied portions are insensitive in this geometrical approximation: the chip is **not stretched** to fill the stave. If detailed module-FPC/stave-FPC placement or true inter-chip gaps become available, enter the measured gaps in the fixed-parameter section or adapt the placement model.

The +z orientation is a *coordinate convention* for this first implementation. The figure supplied establishes the side of the chip carrying EOC and its width, but does not by itself determine alternating chip orientations, precise module offsets, or independent gaps; these are **not inferred**. Every chip currently carries EOC at +z in local stave coordinates. Extending this to a measured per-chip orientation map is straightforward.

**None of the chip properties, including EOC and additional dead margins, is optimized.** Only `radius_cm`, `n_staves`, `tilt_deg` may occur in scan bounds. The API explicitly rejects other scan variables.

## What is computed

Tracks start at `(0,0,z0)` with flat generated azimuth/eta (and optionally Gaussian vertex z). Each line is intersected with the finite planar staves; a hit is counted only when its local coordinates fall within one of the *physical chip rectangles* and outside its EOC and any additional passive borders.

- `eff_all`: trajectories hitting at least one active chip / all generated trajectories.
- `eff_reference`: trajectories with an active hit *and* an intersection with an ideal radius-R cylinder inside |z| ≤ L/2, divided by trajectories reaching that ideal cylinder.
- `physical_fraction`: fraction crossing some physical stave plane envelope.
- `sensor_fraction`: fraction crossing at least one silicon chip footprint.
- `dead_only_fraction`: fraction crossing a stave envelope but not an active chip (including FPC regions and physical gaps).
- `active_chip_area_fraction`: geometrical active chip area / physical chip area (93.33% for the defaults).

Overlapping staves can recover otherwise inactive intersections. A trajectory counts once if any stave provides an active hit.

These quantities represent **geometrical coverage, not tracking/reconstruction or matching efficiency**. No magnetic bending, detector material, scattering, sensor response, timing, or matching of TRK to iTOF hits is included.

## Dashboard

1. Change design variables (radius, stave count, tilt).
2. Edit fixed chip size, EOC, extra dead edges and gaps when physical information becomes available.
3. Simulate and inspect stave XY view, hit probability vs phi/eta, eta–phi map, and a detailed local module map. EOC is explicitly drawn in orange; active silicon is teal.
4. Set scan ranges for the *three design parameters only* and optimize. The comparison uses the **same generated track sample** for each candidate. Results can be applied and exported as CSV.
5. Save/load an entire configuration as JSON.

## Test

```bash
pip install pytest
pytest -q
```

Tests cover: EOC asymmetry and orientation (+z), single-chip nominal area, chip-array envelope (no artificial stretching), validation, optimization bounds and preservation of fixed sensor geometry, and API request behavior if Flask is installed.

## Implementation and endpoints

- `geometry.py`: detector geometry, straight-line intersections, masks, metrics and optimization.
- `app.py`: Flask application and REST API.
- `templates/index.html`: dashboard, with Plotly JS bundled under `static/`.
- `example_config.json`: JSON sample with nominal chip dimensions and fixed EOC.

- `GET /api/defaults`
- `POST /api/simulate`
- `POST /api/optimize`

Example optimization body:

```json
{
  "geometry": {
    "radius_cm": 21, "n_staves": 24, "tilt_deg": 3,
    "chip_width_cm": 2.6, "chip_length_cm": 3.0,
    "eoc_width_mm": 2.0
  },
  "n_tracks_opt": 4000,
  "optimization": {
    "bounds": {
      "radius_cm": [18, 25],
      "n_staves": [20, 32],
      "tilt_deg": [-5, 15]
    },
    "n_candidates": 80
  }
}
```

The web service listens only on `127.0.0.1`. Use locally; this is not designed to be publicly exposed.

## References for stave layout

Baseline stave geometry in the ALICE O2 development code (chip dimensions and EOC values in this version come from the user's latest specifications):

- https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/Upgrades/ALICE3/IOTOF/simulation/src/Detector.cxx
- https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/Upgrades/ALICE3/IOTOF/simulation/src/Layer.cxx
