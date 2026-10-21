# iTOF Geometry Lab — ALICE 3

Local Python + Flask web app for exploring and optimizing geometrical acceptance of an ALICE 3 iTOF barrel layer.

## Run (Linux)

```bash
unzip itof-optimizer-center-tilt-live-tracks.zip
cd itof-optimizer
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
| Stave tilt | 3° (rotation about each stave midpoint) |

### Fixed stave and chip properties

| Parameter | Default |
| --- | ---: |
| Mechanical stave width | 5.42 cm |
| Mechanical stave length | 129 cm |
| **Minimum allowable stave separation** | **1 mm** (fixed constraint, configurable) |
| Transverse stave thickness | 0 mm (idealized, unknown physical dimension) |
| Chips across stave (`u`) | 2 |
| Modules along beam (`z`) | 10 |
| Chips per module along `z` | 4 |
| **Individual chip size** | **2.6 cm (u) × 3.0 cm (z)** |
| **End-of-column strip** | **2.0 mm, single edge at +z** |
| Extra inactive borders (each side) | 0 mm in both `u` and `z` |
| Inter-chip gaps in `u` and `z` | 0 mm (not yet supplied) |
| Inter-module gap in `z` | 0 mm (not yet supplied) |

### Tilt convention

The stave midpoint sits at `(R*cos(phi_i), R*sin(phi_i))` for each azimuthal slot and stays at that exact position when tilt changes. The stave is rotated about **its own midpoint**, *not an endpoint*, by `tilt_deg` from the local tangential direction (tilt = 0° is tangential to the radius-R cylinder). A dedicated `stave_frames()` helper controls the segments used for clearance, the XY plot, and the trajectory-plane intersection convention. The XY plot marks every stave midpoint/pivot.

The **end-of-column runs along the 26 mm chip edge**, and extends 2 mm *inward along its 30 mm dimension*. The default active rectangle is therefore **26 × 28 mm = 728 mm²** out of 780 mm²: **93.33% active chip fraction**, before any extra dead borders and gaps.

With 2 chips across and 40 chips longitudinally, the array envelope is **5.2 cm × 120 cm** (for zero gaps). It is centered in the larger 5.42 × 129 cm stave. The mechanically unoccupied portions are insensitive in this geometrical approximation: the chip is **not stretched** to fill the stave. If detailed module-FPC/stave-FPC placement or true inter-chip gaps become available, enter the measured gaps in the fixed-parameter section or adapt the placement model.

The +z orientation is a *coordinate convention* for this first implementation. The figure supplied establishes the side of the chip carrying EOC and its width, but does not by itself determine alternating chip orientations, precise module offsets, or independent gaps; these are **not inferred**. Every chip currently carries EOC at +z in local stave coordinates. Extending this to a measured per-chip orientation map is straightforward.

**None of the chip properties, including EOC and additional dead margins, is optimized.** Only `radius_cm`, `n_staves`, `tilt_deg` may occur in scan bounds. The API explicitly rejects other scan variables. Both the stave clearance requirement and thickness are **fixed during optimization**, and can be set in the sensor/mechanics panel.

## No-overlap / minimum stave separation constraint

All stave pairs are checked in the transverse XY plane (including nonadjacent staves and the periodic wraparound pair). Each stave is represented by a finite-width line segment whose endpoints are computed using its physical transverse width, radius, and tilt. The **exact closest distance between the centerline segments** is calculated, with intersections assigned a gap of **zero**. With nonzero stave thickness, an additional conservative allowance is subtracted:

```
conservative_clearance_mm = 10 * min_distance_between_XY_segments_cm - stave_thickness_mm
```

A geometry is allowed **only if** `conservative_clearance_mm >= stave_clearance_mm` for **every pair**. The default `stave_clearance_mm=1.0` means a required gap of **at least 1 mm**; it does *not* mean a 1 mm penetration is acceptable. This constraint ensures centerline nonintersections at zero thickness, and conservatively guarantees clearance for finite-thickness rectangular stave footprints when the thickness is supplied. It may exclude some layouts that would be possible with an exact detailed stave footprint.

**IMPORTANT:** The stave's actual physical transverse thickness has not been specified. The default `stave_thickness_mm=0.0` is only a 2D idealization: **1 mm between infinitely thin stave segments is not a verified 1 mm mechanical gap between real staves.** Set a measured thickness before using the constraint as a mechanical interference check, or replace the simplified stave envelope with CAD-derived polygons.

For the nominal `(R=21 cm, N=24, tilt=3°, stave width=5.42 cm)` geometry, the minimum centerline separation is about **3.04 mm**, which passes the default 1 mm requirement.

Manual simulations that violate the rule do not produce an efficiency. Instead, the API replies with status 422 and reports the minimum gap, offending stave pair(s), and XY geometry; the dashboard highlights the affected staves **red**. Optimization candidates that violate the clearance requirement are skipped before simulation. The optimization results show minimum gap, and report the number of rejected geometries. Both gap and thickness stay fixed across all candidates.

## What is computed

Tracks start at `(0,0,z0)` with flat generated azimuth/eta (and optionally Gaussian vertex z). Each line is intersected with the finite planar staves; a hit is counted only when its local coordinates fall within one of the *physical chip rectangles* and outside its EOC and any additional passive borders.

- `eff_all`: trajectories hitting at least one active chip / all generated trajectories.
- `eff_reference`: trajectories with an active hit *and* an intersection with an ideal radius-R cylinder inside |z| ≤ L/2, divided by trajectories reaching that ideal cylinder.
- `physical_fraction`: fraction crossing some physical stave plane envelope.
- `sensor_fraction`: fraction crossing at least one silicon chip footprint.
- `dead_only_fraction`: fraction crossing a stave envelope but not an active chip (including FPC regions and physical gaps).
- `active_chip_area_fraction`: geometrical active chip area / physical chip area (93.33% for the defaults).

Overlapping stave centerlines are forbidden by the no-overlap rule. A trajectory counts once if any active chip is hit.

These quantities represent **geometrical coverage, not tracking/reconstruction or matching efficiency**. No magnetic bending, detector material, scattering, sensor response, timing, or matching of TRK to iTOF hits is included.

## Dashboard

1. Change design variables (radius, stave count, tilt). The geometry/clearance preview and **module drawing/size readouts update live** (debounced ~160 ms) on chip, gap, segmentation, and stave input changes; full tracking efficiency is recomputed only when clicking **Simula geometria**.
2. Click **+ Traccia efficiente e inefficiente** to add **two new random tracks** to the XY panel (one active hit, one miss). Each click **adds** a pair without erasing older tracks; **Rimuovi tracce** clears the overlay. Tracks are reclassified with the same Python hit logic used by efficiency calculations. Both are conditioned on entering the ideal cylinder within the finite barrel length, and misses preferentially cross a stave but avoid active silicon. The XY plot is a **projection only**; eta and z0 determine whether silicon/EOC are hit. Markers at ray endpoints show the ray direction, **not** the actual sensor hit position. Track overlays reset when detector geometry or the track-generation eta/sigma settings change.
3. Edit fixed chip size, EOC, extra dead edges and gaps when physical information becomes available.
4. Simulate and inspect stave XY view, hit probability vs phi/eta, eta–phi map, and a detailed local module map. EOC is explicitly drawn in orange; active silicon is teal. Staves violating the distance constraint appear in red in the XY panel, with the failing stave pair and clearance shown in a banner.
5. Set scan ranges for the *three design parameters only* and optimize. The comparison uses the **same generated track sample** for each candidate. Results can be applied and exported as CSV.
6. Save/load an entire configuration as JSON.

## Test

```bash
pip install pytest
pytest -q
```

Tests cover: rotation about stave midpoint, recalculation of module dimensions, seeded random hit/miss tracks consistent with the evaluation masks, XY stave collisions, all-pair/periodic checks, minimum separation vs tolerance, fixed stave thickness in optimization, EOC asymmetry and orientation (+z), single-chip nominal area, chip-array envelope (no artificial stretching), validation, optimization bounds and preservation of fixed sensor geometry, and API request behavior if Flask is installed.

## Implementation and endpoints

- `geometry.py`: detector geometry, straight-line intersections, masks, metrics and optimization.
- `app.py`: Flask application and REST API.
- `templates/index.html`: dashboard, with Plotly JS bundled under `static/`.
- `example_config.json`: JSON sample with nominal chip dimensions and fixed EOC.

- `GET /api/defaults`
- `POST /api/simulate`
- `POST /api/optimize`
- `POST /api/preview` — rapid module, XY geometry and clearance preview without recomputing tracks (it will also draw invalid stave-clearance configurations so they can be corrected).
- `POST /api/sample-tracks` — one random active and one random inactive track, inside the ideal reference acceptance. Accepts optional `sample_seed` for a reproducible test; otherwise each request gets a fresh random sample.

Example optimization body:

```json
{
  "geometry": {
    "radius_cm": 21, "n_staves": 24, "tilt_deg": 3,
    "stave_clearance_mm": 1.0, "stave_thickness_mm": 0.0,
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
