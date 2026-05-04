Usage
=====

This section provides a concise overview of how to use **barc4beams**
through its main workflows.

The library is organized around a central abstraction: the **Beam**.

---

Core concepts
-------------

Beam
~~~~

A ``Beam`` represents a collection of rays in a standardized format:

- positions: (x, y)
- directions: (x', y')
- energy
- intensity
- polarization components
- lost-ray flags

All operations (statistics, visualization, transformations) are defined
on this structure.

---

BeamEnsemble
~~~~~~~~~~~~

A ``BeamEnsemble`` is a collection of ``Beam`` instances representing
multiple realizations (e.g. scans, simulations, tolerances).

It provides:

- ensemble statistics
- structured comparison between runs
- unified save/load interface

---

Main workflows
--------------

1. Beam from ray-tracing
~~~~~~~~~~~~~~~~~~~~~~~~

Typical pipeline:

1. Import raw ray-tracing data
2. Convert to standard format
3. Analyze and visualize

This is illustrated in:

- Example 1a (monochromatic-like case)
- Example 1b (polychromatic beam)

---

2. Beam collections and ensembles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two levels of abstraction:

- Manual handling of multiple beams → ``Beam`` (Example 2a)
- Structured handling → ``BeamEnsemble`` (Example 2b)

Use ``BeamEnsemble`` when:

- working with repeated simulations
- performing statistical analysis across runs
- saving/loading grouped datasets

---

3. Beam from wavefront data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two complementary approaches:

**Intensity-based sampling**

- Uses NF and/or FF intensity
- Assumes no phase information
- Implemented via ``Beam.from_intensity()``

(Example 3a)

---

**Phase-aware reconstruction**

- Uses complex field (amplitude + phase)
- Computes local slopes from the wavefront
- Generates physically consistent rays

(Example 3b)

This follows principles similar to X-ray speckle tracking,
but applied in reverse (wavefront → rays).

---

4. Applying wavefronts to beams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A beam can be modified by applying a complex wavefront:

- modifies ray directions (via phase gradients)
- updates intensity
- flags invalid rays if needed

(Example 3c)

This enables hybrid workflows combining:

- ray-tracing
- wave optics
- error propagation

---

Visualization and statistics
----------------------------

The library provides:

- beam size and divergence plots
- phase-space visualization
- caustics
- intensity-aware statistics

All statistics are computed using per-ray weights (intensity).

---

Persistence
-----------

Objects can be saved and reloaded:

- ``Beam`` → HDF5 + JSON metadata
- ``BeamEnsemble`` → grouped HDF5 + statistics

---

Summary
-------

Typical usage patterns:

- Ray-tracing → Beam → analysis
- Multiple runs → BeamEnsemble → statistics
- Wavefront → Beam → hybrid propagation

Refer to the example notebooks for concrete implementations.