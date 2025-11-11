# barc4beams

**barc4beams** is a Python package providing tools for  
standardization, statistical analysis, and visualization of ray-traced photon beams  
from simulation codes such as **SHADOW3/4**, **SRW**, and **PyOptiX**.

It offers a unified data class (`Beam`), flexible import/export utilities, statistical
moment-based analysis, and a consistent plotting API for beam profile, divergence,
phase space, and caustic visualization.

---

## Features

- Import beams from **SHADOW3/4**, **PyOptiX**, or intensity maps from **SRW** / **WOFRY**  
- Convert to a standardized, schema-validated `Beam` format  
- Compute beam moments, FWHM, skewness, kurtosis, and focal distances  
- Merge and analyze ensembles of beams across runs  
- Propagate beams in free space and build caustics  
- Save and reload beams in HDF5 or JSON formats  
- Publication-ready 2D visualizations using Matplotlib

See [`examples/README.md`](examples/README.md) for detailed descriptions of the  
three reference notebooks:

1. Beam import from ray-tracing codes  
2. Beam sampling from wave-propagation intensity maps  
3. Beam collections and statistical aggregation

---

## Installation

From PyPI (once published):

```bash
pip install barc4beams