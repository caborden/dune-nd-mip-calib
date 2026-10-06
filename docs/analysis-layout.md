# Sample-based FSD-cube workflow

Use one `fsdcube/` workspace for selections, analyses, and comparisons. Sample
names describe productions; analysis names describe the method inside each run.
The repository stores code and analysis recipes. Large outputs and the local
`scratch/` mirror are excluded from Git.

```text
fsdcube/
  samples/
    data-Reflow_FSDCube_v3.binary-cosmics-2026_02/
      sample.json
      selection/v1/
        Reflow_FSDCube_v3.binary-cosmics-2026_02.track-selection.hdf5
        inputs.txt
        parts/                    # full batch HDF5 files, logs, manifest
        pilot/                    # pilot merged file and parts/
      analysis/run-001/           # imported historical data-only analysis
        config.json
        manifest.json
        shared/segments.hdf5
        dqdx/                     # plots, histogram CSVs, original analysis.json
    mc-FSDCubeSim_v1_prc256/
      sample.json
      selection/v1/
        FSDCubeSim_v1_prc256.track-selection.hdf5
        inputs.txt
        parts/
        pilot/
      analysis/                   # created when an MC-only analysis is run
  comparisons/
    reflow-v3-feb2026_vs_fsdcube-sim-v1-prc256/
      comparison.json            # references samples and selection versions
      analysis/run-001/           # imported historical paired analysis
        config.json
        manifest.json
        shared/segments.hdf5
        dqdx/
      analysis/run-002/           # new modular run, when requested
        config.json
        manifest.json
        shared/segments.hdf5
        dqdx/
  archive/
    FSDCubeSim_v1_prc256-20261003-191727/inputs.txt
  relocation.json                # migration journal and original-file SHA256s
```

Each selection version is independent of the analysis run number. A comparison
references the selected versions of its data and MC samples; it does not copy
their selections. Shared derived tables are currently reused **within a run**.
Separate runs/comparisons each have their own derived tables; there is no global
cache or automatic cross-run linking.

## Code and configuration

`analysis/segments.py` streams selected event hits into a shared segment table.
`analysis/dqdx.py` reads that table to produce dQ/dx histograms and fits.
`analysis/hit_density.py` uses the same table for hits per segment versus drift
time and data/MC ratios.
`analysis/config.py` holds the common physics defaults and validation.
`analysis/io.py` handles atomic manifests, identities, and provenance.
`analysis/run.py` resolves sample registries and executes dependencies once.
The existing `analyze_track_dqdx.py` CLI and imported helpers remain available,
including data-only, MC-only, and paired modes and the original flat output layout.

Recipes under `configs/analyses/` reference registry IDs rather than hard-coded
NERSC or local paths. `--root` chooses the machine's workspace, and `--run-id`
chooses a new output directory. Geometry is embedded in the recipe's `geometry`
object if explicit bounds are needed. The current defaults preserve 3 cm
segments/steps, the conditional 1 cm outer-face margin, and the existing fits.
Shared segmentation options belong in the top-level `segments` object;
`modules.dqdx` holds plotting/fit options. Misplaced `modules.segments` options
are rejected before creating an output run.

Available stages are `segments`, `dqdx`, and `hit_density`. Requesting either
plotting module automatically runs or reuses `segments`; `all` requests every
implemented stage. The supplied recipes run both plotting modules by default.
Future scientific modules should have separate files, output subdirectories,
and registered dependencies. The pixel-charge module will eventually use the all-event hit
view directly, keyed by sample/file/event/electronics address; segment extraction
continues using only selected hits.

Examples from the repository directory:

```bash
# Data-only, MC-only, or a paired comparison (choose one recipe).
python -m analysis.run --config configs/analyses/data.json \
  --root "$SCRATCH/fsdcube" --run-id run-002 --modules dqdx

python -m analysis.run --config configs/analyses/mc.json \
  --root "$SCRATCH/fsdcube" --run-id run-002 --modules dqdx

python -m analysis.run --config configs/analyses/data_mc.json \
  --root "$SCRATCH/fsdcube" --run-id run-002 --modules all

# Prepare only the shared table, then add plots to the same run.
python -m analysis.run --config configs/analyses/data_mc.json \
  --root "$SCRATCH/fsdcube" --run-id run-003 --modules segments

python -m analysis.run --config configs/analyses/data_mc.json \
  --root "$SCRATCH/fsdcube" --run-id run-003 --modules dqdx --resume
```

A run refuses overwriting existing results. `--resume` checks canonical input
paths, file sizes/mtimes, relevant settings, source-code hashes, dependency
signatures, package/Python versions, and output paths/sizes/mtimes. Inputs are
assumed immutable; this analysis check does not hash every large input file.
Completed stages are reused only on an exact match. Use a new run ID when
inputs/settings/code change or outputs are modified. A failed stage keeps its
diagnostic staging directory and can be retried while retaining completed
dependencies. A killed process can leave `.writer.lock`; confirm no process is
still using that run before removing that lock. Interrupted publication with an
unrecorded destination is refused for manual inspection.

`manifest.json` records per-stage completion. Overall completion applies to
the requested stages and their dependencies, not every future module. A fit
marked unavailable remains a valid completed plotting stage; consult the
module's `analysis.json` for fit success and diagnostics.

## Hits per segment versus drift time

The `hit_density` module implements the left plot on slide 10 of
`FSD_CUBE_MC_SD_2026.pdf`, plus its data/MC ratio. It uses a **segment-weighted
arithmetic mean of recorded `nhits`**, rather than unique-pixel counts. Every
segment enters the bin containing its saved median hit `t_drift`. Data and MC
share identical bin edges. Drift times remain in their original ticks; no tick
conversion or production-specific rescaling is introduced.

Defaults confirmed for this study are 3 cm windows from the shared segmentation,
nine equal bins over 0–5000 ticks, and no additional angular cuts. These bin edges
and the mean interpretation are working assumptions: the slide does not give
exact edges, angular selection, or averaging/error definitions. This reproduces
the observable and plotting style; it does not promise the original numerical
values or sample. Existing face/group/through-going settings still apply when
building the shared table.

The y-axis explicitly says mean hits per segment. Set
`observable: "hits-per-cm"` for a literal dN/dx: each segment contributes
`nhits/dx`, where `dx` is its physical projected span plus median spacing,
not the nominal window length. The module then averages those ratios, rather
than dividing the total hit count by total length.

Repeated selected hit records on one pixel count separately. Rejected event
hits remain excluded. The existing shared table contains only accepted
segmentation windows (at least two hits, positive charge sum and physical dx);
empty/one-hit/nonpositive-charge windows and partial end windows are absent.
No selection or segmentation mathematics changes for this module.

Example from the repository directory, after migration:

```bash
python -m analysis.run --config configs/analyses/data_mc.json \
  --root "$SCRATCH/fsdcube" --run-id run-004 --modules hit_density
```

Use `data.json` or `mc.json` for a single sample; use `--modules all` to produce
both dQ/dx and hit-density outputs from one shared extraction. Adding this module
changes the runner code fingerprint, so choose a new run ID when updating older
runs. Existing output directories remain intact. Once created with the current
code, the same run supports `--resume` to add the other plotting module.

Options belong in `modules.hit_density`:

```json
{
  "drift_bins": 9,
  "drift_range_ticks": [0, 5000],
  "observable": "hits-per-segment",
  "uncertainty": "track-bootstrap",
  "bootstrap_samples": 1000,
  "bootstrap_seed": 12345,
  "min_segments": 1
}
```

Optional `drift_edges_ticks` supplies custom increasing edges and overrides the
uniform range/bin count. Bins include their left edge; only the final bin includes
its right edge. Nonfinite/invalid rows, below-range rows and above-range rows are
audited separately. `min_segments` suppresses means below the requested bin count.
Empty bins produce missing values, never zero-valued measurements.

The confirmed default uncertainty is **whole-track bootstrap**, with 1000
replicates. All segments/readout groups for `(sample, file, event, track)` share
one resampling weight, and the segment-weighted mean is recomputed. It accounts
for within-track correlations, including repeated/overlapping windows. Tracks
with at least one valid in-range segment form the bootstrap population. This
treats different tracks as independent; it does not include shared event/run
systematics or fluctuations across source files. Errors are the standard
deviation of resampled means, not confidence intervals. A bin with fewer than
two contributing tracks has no bootstrap error; replicates with empty bins are
excluded and their valid count is saved. Data/MC ratio errors use first-order
propagation assuming independent samples; missing or zero MC means have no ratio.

Choose `uncertainty: "segment-sem"` for the sample standard deviation divided
by sqrt(segment count), or `"none"` for no error bars. Segment SEM is also saved
as a diagnostic even when bootstrap errors are plotted. The random seed and
separate sample streams make bootstrap outputs reproducible; adding MC does not
change data uncertainties. Input is streamed in chunks, retaining per-track/bin
statistics rather than loading the entire shared table.

Outputs live under `analysis/<run>/hit_density/`:

- Per-sample `data_hit_density`/`mc_hit_density` PNG, PDF and CSV files.
- Paired `data_mc_hit_density` PNG/PDF and `data_mc_hit_density_ratio` PNG/PDF/CSV.
- `analysis.json`: completed status, definitions, settings, input/code identities,
  segmentation settings, audit counts and all bin summaries (mean, standard
  deviation, segment SEM, displayed error, segment/track counts, valid bootstrap
  replicates). Missing values are JSON null or CSV NaN.

To use an already completed table without another extraction, the module also
has a standalone CLI. This writes independent module products and does not update
the runner's stage manifest; use a new destination outside an existing runner
stage directory:

```bash
python -m analysis.hit_density --segments /path/to/shared/segments.hdf5 \
  --output-dir /path/to/new-hit-density-output \
  --drift-bins 9 --drift-range-ticks 0 5000 --uncertainty track-bootstrap
```

## Migrate the existing NERSC directories

Finish any selection/merge/analysis jobs writing these directories first.
Migration is specific to the current folder names listed above. It checks
completed schema-2 selections, every full-batch part, batch/source counts,
historical analysis completion, and segment counts. Unexpected destinations or
registries are refused before moving files.

Run from the updated repository in the existing analysis environment:

```bash
module load conda
conda activate /global/common/software/dune/efield2_rcmanduj/

# Print and validate the plan; no changes.
python reorganize_outputs.py --root /pscratch/sd/c/cborden

# Rename within scratch, retain old-path links, create registries/import records.
python reorganize_outputs.py --root /pscratch/sd/c/cborden --apply

# Independently verify preserved bytes, completed HDF5 files, counts, and links.
python reorganize_outputs.py --root /pscratch/sd/c/cborden --check
```

The utility records SHA256 hashes of every original regular file (including
logs, plots, JSON, and notebook checkpoints), moves rather than copies the
existing outputs, and checks those hashes afterward. An interrupted move can be
continued by rerunning `--apply`; repeating a completed migration only verifies
it. Unknown sibling directories remain untouched. The abandoned inputs-only MC
directory is kept under `archive/` when present.

Old directories become relative compatibility symlinks. Old `full_parts`,
`pilot_parts`, pilot-file and analysis `segments.hdf5` paths also remain
accessible through links. Keep these links while using older notebooks or
historic manifests. Their original paths, timestamps, hashes, and HDF5 metadata
are preserved verbatim; imported analyses are labeled `imported-analysis-v1`
and require a **new** run ID for the modular runner.

The batch/merge resume code compares canonical paths. Compatibility links keep
old paths readable but do not make those old canonical-path signatures match
after relocation. Treat migrated selections as completed historical products;
use a fresh `selection/v2/` directory and fresh manifests for a new batch or
merge. This migration does not regenerate those selections or rewrite their
historical provenance.

## Add another production

Create `samples/<descriptive-id>/sample.json` with the production kind and a
relative selection path; put the input list, completed merged selection, parts
and pilot under its `selection/v1/` directory. For example:

```json
{
  "id": "mc-another-production",
  "kind": "mc",
  "selection_versions": {
    "v1": {"path": "selection/v1/another-production.track-selection.hdf5"}
  }
}
```

Copy an analysis recipe, change its target ID, and use a fresh run ID. To compare
two productions, create `comparisons/<comparison-id>/comparison.json` with
`samples.data` and `samples.mc` reference objects containing `id` and
`selection_version`. Outputs then belong to the comparison directory.
