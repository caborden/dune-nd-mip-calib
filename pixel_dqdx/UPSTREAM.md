# Upstream provenance

Restored from https://github.com/robx97/dune-nd-mip-calib/tree/r/dev/pixel_dqdx
at commit `33cf2fe926afe416b31fe930f18dff80b9ab47d8`, fetched and verified against
GitHub on this task. The directory exists on `r/dev`, not upstream `main`.
The fork includes `origin/r/dev`; it was absent from the working tree because
`fsdcube-charge-splitting` starts from `main`.

Local adaptations:

- `segments.py`: fixed the incomplete final print statement/undefined `path`;
  extracted the original physical PCA, projected windows, local-spacing dx,
  charge sum, and angle calculations into `segment_track_hits`. The old
  `run_dqdx_tpc` calls that helper. Removed unused plotting/fitting imports.
- `dqdx.py`: replaced removed `np.trapz` with SciPy `trapezoid`; made tqdm optional;
  added prebinned histogram input so the new reader does not hold all dQ/dx
  entries in memory. Initial optimizer width guesses for prebinned input use
  the histogram's weighted variance. Increased optimizer evaluations and handled
  failed fits. Added an optional width bound used by the new driver to avoid
  unbounded convolution grids. Fixed the mismatched x/fit arrays and descending interpolation
  in the ancillary FWHM routine.
- `lifetime.py`: made LaTeX-containing strings raw to avoid invalid escapes.
  Lifetime analysis is not called by the new workflow.

The upstream `landau()` is proportional to a Moyal PDF, not an exact Landau PDF.
The free amplitude absorbs its normalization. The upstream `langau()` integrates
this approximation against a Gaussian with a 0.01 integration step and ±20σ
integration bounds. These model definitions are preserved. See the SciPy Moyal
reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moyal.html

The old segmentation entry point still reads the old `tracks/label` and
`hits/reco` HDF5 layout, and retains the upstream mixed hard-coded TPC bounds.
Use `analyze_track_dqdx.py` for the merged schema-2 outputs. It uses only sane stored outer bounds for automatic face cuts, requires explicit
geometry for through-going/internal TPC cuts, and never adopts those hard-coded bounds.
