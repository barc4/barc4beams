# barc4beams — Example Notebooks

v2025.11.10

This folder contains a minimal set of Jupyter notebooks demonstrating the core functionalities of **barc4beams**,  
from beam import and standardization to sampling, propagation, and statistical aggregation.

---

## Example 1 — Beam from ray-tracing codes
**File:** `ex_01_beam_from_raytracing_codes.ipynb`

Shows how to:
- Imports a beam and convert it into the barc4beams standard format.  
- Compute and print beam statistics (moments, FWHM, RMS, focal distances).  
- Plot beam profiles, divergence, phase-space, and caustics.  
- Save and reload standardized beams in `.h5` and `.json` formats.


---

## Example 2 — Beam from wave-propagation codes
**File:** `ex_02_beam_from_wave_propagation_codes.ipynb`

Shows how to:
- Load near-field (NF) and far-field (FF) intensity maps from SRW (or WOFRY) stored in HDF5.  
- Sample rays using `Beam.from_intensity()` following the algorithm from  
  *Rebuffi et al.*, *J. Synchrotron Rad.* **27**, 1108–1120 (2020).  
- Compare two reconstructions:
  - FF-only → assumes a point source.
  - NF+FF → includes finite spot size and angular spread.
- Visualize phase-space, beam profiles, and caustic evolution.

---

## Example 3 — Synthetic beam collection
**File:** `ex_03_beam_collection.ipynb`

Shows how to:
- Compute per-run and merged beam statistics.  
- Merge multiple runs into a single standardized beam and visualize results.
