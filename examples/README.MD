# barc4beams – Example Notebooks

This folder contains a curated set of Jupyter notebooks illustrating the core
features and typical workflows of **barc4beams**.

The examples are grouped by data origin and abstraction level:
- Ray-tracing based workflows
- Beam manipulation and ensembles
- Wavefront-based reconstruction and propagation

---

## 1. Beams from ray-tracing codes

### Example 1a — Mirror focusing
**File:** `ex_01a_beam_from_raytracing_codes-mirror_focusing.ipynb`

- Import a beam from ray-tracing output
- Convert to the standardized format
- Compute beam statistics (moments, RMS, FWHM, focal distances)
- Visualize size, divergence, phase-space, and caustics

**Key point:**  
Monochromatic-like case where intensity is approximately uniform across energies.

---

### Example 1b — CRL focusing (polychromatic case)
**File:** `ex_01b_beam_from_raytracing_code-crl_focusing.ipynb`

Same workflow as 1a, but:

- Includes a realistic energy bandwidth
- Intensity is distributed across energies

**Key point:**  
Demonstrates the impact of spectral bandwidth on beam properties.

---

## 2. Beam collections and ensembles

### Example 2a — Beam collection
**File:** `ex_02a_beam_collection.ipynb`

- Handle multiple beams (independent runs)
- Compute statistics per run
- Merge beams into a single dataset
- Visualize aggregated results

---

### Example 2b — BeamEnsemble
**File:** `ex_02b_beam_ensemble.ipynb`

- Use the `BeamEnsemble` class
- Compute ensemble-level statistics
- Store and reload ensembles
- Compare runs in a structured way

**Key point:**  
Preferred abstraction for handling multiple realizations consistently.

---

## 3. Beams from wavefront propagation

### Example 3a — Sampling from intensity (NF/FF)
**File:** `ex_03a_beam_from_wave_propagation_codes-NF_FF.ipynb`

- Load near-field (NF) and far-field (FF) intensity maps
- Sample rays using `Beam.from_intensity()`

This sampling strategy follows the approach described in:  
Rebuffi et al., *J. Synchrotron Rad.* **27**, 1108–1120 (2020).

- Compare:
  - FF-only (point-source assumption)
  - NF+FF (finite source size)

---

### Example 3b — Beam from intensity and phase
**File:** `ex_03b_beam_from_wave_propagation_codes-intensity_phase.ipynb`

- Reconstruct rays from intensity **and phase**
- Compute local wavefront slopes from the complex field
- Generate a beam consistent with wavefront propagation

**Key point:**  
Inverse approach inspired by X-ray speckle tracking methods as described in:
Celestre et al., *J. Synchrotron Rad.* **32**, 180–199 (2025).

---

### Example 3c — Applying a complex wavefront
**File:** `ex_03c_beam_from_wave_propagation_codes-intensity_apply_complex_wavefront.ipynb`

- Start from a reconstructed beam
- Apply a complex wavefront (e.g. CRL errors)
- Update intensity and ray directions

**Key point:**  
Bridges wave optics and ray-based propagation.

---

## Notes

- All examples rely on the standardized beam format used throughout
  **barc4beams**.
- The notebooks are intentionally minimal and designed as building blocks
  for custom workflows.