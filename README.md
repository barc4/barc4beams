# barc4beams

**barc4beams** is a Python package providing tools for standardization, statistical analysis,
 and visualization of (photon) beams from simulation codes such as **SHADOW3/4** and **PyOptiX**.
 **barc4beams** also converts intensity maps from wave-optics codes such as **SRW** and **WOFRY**.

It offers a data class called `Beam`, which offers import/export utilities, statistical
moment-based analysis, and a consistent plotting API for beam profile, divergence,
phase space, and caustic visualization.

---

## Features

- Import beams from **SHADOW3/4**, **PyOptiX**, or intensity maps from **SRW** / **WOFRY**  
- Convert to a standard `Beam` format  
- Compute beam moments, FWHM, skewness, kurtosis, and focal distances  
- Merge and analyze ensembles of beams across runs  
- Propagate beams in free space and build caustics  
- Save and reload beams in HDF5 and JSON formats (stats)
- Plotting using Matplotlib

See **examples/README.md** for detailed descriptions of the  
three reference notebooks:

1. Beam import from ray-tracing codes  
2. Beam sampling from intensity maps  
3. Beam collections and statistical aggregation

---

## Installation

From PyPI:

```bash
pip install barc4beams

---

[![PyPI](https://img.shields.io/pypi/v/barc4beams.svg)](https://pypi.org/project/barc4beams/)
[![Python](https://img.shields.io/pypi/pyversions/barc4beams.svg)](https://pypi.org/project/barc4beams/)
[![License: CeCILL-2.1](https://img.shields.io/badge/license-CeCILL--2.1-blue.svg)](https://opensource.org/licenses/CECILL-2.1)