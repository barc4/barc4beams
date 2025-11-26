Usage and examples
==================

The repository ships with three Jupyter notebooks illustrating the core
workflows.

Example 1: Beam from ray-tracing codes
--------------------------------------

File: ``ex_01_beam_from_raytracing_codes.ipynb``

This notebook shows how to:

* Import a beam and convert it into the barc4beams standard format.
* Compute and print beam statistics (moments, FWHM, RMS, focal distances).
* Plot beam profiles, divergence, phase-space, and caustics.
* Save and reload standardized beams in ``.h5`` and ``.json`` formats.

Example 2: Beam from wave-propagation codes
-------------------------------------------

File: ``ex_02_beam_from_wave_propagation_codes.ipynb``

This notebook shows how to:

* Load near-field and far-field intensity maps from SRW/WOFRY stored in HDF5.
* Sample rays using :class:`barc4beams.Beam.from_intensity`.
* Compare reconstructions using FF-only and NF+FF information.
* Visualize phase-space, beam profiles, and caustic evolution.

Example 3: Synthetic beam collection
------------------------------------

File: ``ex_03_beam_collection.ipynb``

This notebook shows how to:

* Compute per-run and merged beam statistics.
* Merge multiple runs into a single standardized beam.
* Visualize the resulting distributions.