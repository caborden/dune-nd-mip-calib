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

A run refuses implicit overwriting of existing results. `--resume` checks canonical input
paths, file sizes/mtimes, relevant settings, source-code hashes, dependency
signatures, package/Python versions, and output paths/sizes/mtimes. Inputs are
assumed immutable; this analysis check does not hash every large input file.
Completed stages are reused only on a compatible scientific match. Changes to
runner/manifest code alone do not invalidate extraction or plotting artifacts.
Change a plotting module's code/settings with `--rerun <module>` to refresh it
in the same run, or choose a new run ID. Changed inputs or segmentation require
`--rerun segments`; downstream stages are then marked stale and recomputed when
requested. A failed computation keeps the previous output at its original path
and retains diagnostics under `.history/`. A killed process can leave `.writer.lock`; confirm no process is
still using that run before removing that lock. Interrupted publication with an
unrecorded destination is refused for manual inspection.

`manifest.json` records per-stage completion. Overall completion applies to
the requested stages and their dependencies, not every future module. A fit
marked unavailable remains a valid completed plotting stage; consult the
module's `analysis.json` for fit success and diagnostics.

### Update or extend an existing run

The dQ/dx titles are **FSD Cube Data Fit** and **FSD Cube Simulation Fit**, including
when plots are regenerated from an older table whose title attributes say FSD.
From the repository directory after pulling updated code on NERSC:

```bash
# Refresh only the dQ/dx plots/fits in the existing paired run-001.
python -m analysis.run --config configs/analyses/data_mc.json \
  --root "$SCRATCH/fsdcube" --run-id run-001 --modules dqdx --rerun dqdx

# Add hit-density products to that same run, reusing compatible shared segments.
python -m analysis.run --config configs/analyses/data_mc.json \
  --root "$SCRATCH/fsdcube" --run-id run-001 --modules hit_density --resume

# Explicitly rebuild extraction and all implemented plotting modules.
python -m analysis.run --config configs/analyses/data_mc.json \
  --root "$SCRATCH/fsdcube" --run-id run-001 --modules all --rerun all
```

Use `data.json` or `mc.json` for a sample-only run. `--rerun` implies `--resume`,
and rerun names are added to the requested modules. Multiple names are supported,
e.g. `--rerun dqdx hit_density`. A forced dependency rebuild also reruns any
requested dependent modules; unrequested dependent outputs remain on disk but
their manifest status becomes `stale` until refreshed. Consult the per-stage
status when consuming an existing output directory.

Each invocation snapshots the previous configuration and manifest under
`.history/<UTC timestamp>-<id>/`. A successful replacement moves the old module
directory or shared table there, then publishes the completed staged replacement.
No previous output is removed before its replacement has been computed. History
grows with refreshes, especially when rebuilding shared HDF5 tables. Archived
compatibility links continue to point to the appropriate saved/current table.

Imported `run-001` results are automatically adopted. Reuse requires a complete
table with matching segmentation options, sample identities, input canonical
paths/sizes/mtimes and upstream segmentation hash. If those checks fail (for
example, when analyzing a local copy with different input paths/timestamps), the
error requests `--rerun segments`. Run the first example with
`--rerun segments dqdx` to rebuild those dependencies while retaining all
previous outputs in history. Original imported provenance is preserved in the
history and adopted manifest; saved HDF5 attributes are not relabeled in place.
Older modular signatures are upgraded only if their recorded provenance can be
reconstructed exactly; otherwise the runner requests an explicit rerun.

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
both dQ/dx and hit-density outputs from one shared extraction. `--resume` can add
this module to an existing compatible run; `--rerun hit_density` refreshes its
existing products without forcing segment extraction.

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
- `hit_count_distributions` PNG/PDF: one composite figure with data/MC overlaid
  in each drift bin. The default nine bins use a 3x3 grid in increasing row-major
  order. Other bin counts retain three columns (unused panels are hidden).
  Unit-width bins are centered on integer recorded-hit counts; each sample is
  normalized separately within each drift bin to show shape despite different
  sample sizes. All populated counts, including tails, are retained. Panels
  report segment counts, arithmetic means (also dashed lines), and empirical
  modes, including ties. These modes are not fitted MPVs. Empty samples are
  labeled explicitly. This diagnostic always uses raw `nhits`, including when
  the main mean plot uses `hits-per-cm`; it is not suppressed by `min_segments`.
- Per-sample `data_hit_count_distributions.csv`/`mc_hit_count_distributions.csv`
  save drift-bin edges, integer hit counts, segment frequencies and fractions.
- `analysis.json`: completed status, definitions, settings, input/code identities,
  segmentation settings, audit counts and all bin summaries (mean, standard
  deviation, segment SEM, displayed error, segment/track counts, valid bootstrap
  replicates), plus raw hit-count frequencies, means and modes.
  Missing values are JSON null or CSV NaN.

Refresh this module in an existing paired run after pulling updated code:

```bash
python -m analysis.run --config configs/analyses/data_mc.json \
  --root "$SCRATCH/fsdcube" --run-id run-002 \
  --modules hit_density --rerun hit_density
```

This reuses compatible shared segments and replaces only the hit-density stage;
the old stage is archived under the run's `.history/` directory.

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
are preserved verbatim; imported analyses are initially labeled
`imported-analysis-v1` and can be adopted by the runner as described above.
After intentional refreshes, `reorganize_outputs.py --check` verifies the original
file hashes against either their current paths or preserved `.history/` copies.
It verifies migration preservation, not the freshness of every current analysis.

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


## February 2026 periodic-reset scan

Selections remain under `samples/<sample-id>/selection/v1/`. The reset recipes
`configs/analyses/data_prc2.json`, `data_prc8.json`, and `data_prc16.json` now target
paired data/MC comparisons, despite their retained `data_` filenames. All three
use the existing `mc-FSDCubeSim_v1_prc256` selection and the same analysis settings.
Their output roots are:

```text
fsdcube/comparisons/
  reflow-v3-feb2026-prc2_vs_fsdcube-sim-v1-prc256/
  reflow-v3-feb2026-prc8_vs_fsdcube-sim-v1-prc256/
  reflow-v3-feb2026-prc16_vs_fsdcube-sim-v1-prc256/
```

Each holds `comparison.json` and `analysis/<run-id>/`, with `config.json`,
`manifest.json`, `shared/segments.hdf5`, `dqdx/`, and `hit_density/`. No sample-level
analysis directory is created by these recipes. Sample registries must already
exist and reference complete schema-2 selections. The historical unsuffixed data
sample ID is used for prc8; its canonical paths and existing analyses are preserved.
The old unsuffixed comparison directory is also retained as a historical result;
the explicitly named prc8 comparison starts a new run rather than moving it.

From the repository root on NERSC, initialize the environment and registries:

```bash
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /global/common/software/dune/efield2_rcmanduj/
python prepare_reset_comparisons.py --root "$SCRATCH/fsdcube"
```

This validates all three data/MC pairs before creating any registry. Repeated
setup reuses identical registries and refuses conflicting ones. Use
`--resets prc2 prc16` if only those pairs should be prepared.

Submit from the repository root:

```bash
ACCOUNT=YOUR_NERSC_ACCOUNT
mkdir -p "$SCRATCH/fsdcube/jobs/logs"
for reset in prc2 prc8 prc16; do
    env -u SBATCH_MEM_PER_NODE -u SBATCH_MEM_PER_CPU -u SBATCH_MEM_PER_GPU \
      sbatch -A "$ACCOUNT" --cpus-per-task=2 \
      --job-name="analysis-${reset}" \
      --output="$SCRATCH/fsdcube/jobs/logs/analysis-${reset}-%j.out" \
      jobs/analyze_sample.sbatch \
      "configs/analyses/data_${reset}.json" "$SCRATCH/fsdcube" run-001
done
squeue -u "$USER"
```

The script initializes Conda in Bash, prints startup/environment/input diagnostics,
then runs shared segmentation, dQ/dx, and hit density including hit-count
distributions and data/MC ratios. It requests two logical CPUs and no explicit
memory limit, leaving memory to the site's configured default. The submission
command also removes inherited SBATCH memory overrides. Confirm the effective
memory request with `sacct` rather than assuming a specific default. The 12 hour
walltime is a starting value; override with `sbatch --time=...` as needed.
Each comparison currently extracts its own
data and MC segment table, so the shared MC is processed once per comparison.

Use a new run ID if the destination already exists. Pass `--resume` as an extra
argument to the job script to reuse compatible completed stages. Check
`manifest.json` for per-stage completion and module `analysis.json` files for
fit/statistics diagnostics. Failed sample-only runs from the previous recipes
can remain in place: the revised comparison targets use different directories.

For the original startup failures, inspect both combined stdout/stderr logs:

```bash
sed -n '1,200p' "$SCRATCH/fsdcube/jobs/logs/analysis-prc2-59505327.out"
sed -n '1,200p' "$SCRATCH/fsdcube/jobs/logs/analysis-prc16-59505330.out"
```

The original logs identified conflicting launcher settings:
`SLURM_CPUS_PER_TASK=18` and `SLURM_TRES_PER_TASK=cpu=2`. Both describe CPUs per
task, and srun refuses to launch when they disagree. NERSC shared CPU allocations
are also constrained by memory: 32 GiB at 1952 MiB per logical CPU needs 17 logical
CPUs, rounded to 18 for nine physical cores. This is consistent with the job's
CPU count growing beyond the original request of two. The log alone does not
prove where either variable was last assigned.

The job now requests two logical CPUs with no explicit memory directive, logs
both CPU variables at entry, passes the allocated `SLURM_CPUS_PER_TASK` explicitly to
`srun --cpus-per-task`, and removes the redundant `SLURM_TRES_PER_TASK` only from
that CPU-only launcher's environment. It retains the Slurm job identity and
allocation variables. Numerical-library thread counts remain one.
This launch convention should also be used in future CPU-only job scripts.
GPU steps need their own resource handling because TRES can include GPU requests.

To check the failed jobs' actual allocations:

```bash
sacct -j 59505327,59505330 -X \
  --format=JobID,ReqCPUS,AllocCPUS,ReqMem,ReqTRES%70,AllocTRES%70
```

After resubmission, include step-level peak memory measurements:

```bash
sacct -j NEW_JOB_ID --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS
```

To retry after startup failures, use a fresh run ID such as `run-002` to avoid
collisions with partial runs or stale writer locks. Do not submit a duplicate
writer to a run that is still active.

Memory-based allocation is documented at
https://docs.nersc.gov/jobs/examples/#shared. Slurm's environment-variable mappings
are documented at https://slurm.schedmd.com/srun.html.

## Future sample groupings (proposed extension)

Use `samples/` for immutable selections and production metadata. Prefer named
`comparisons/` for new analysis products, including groups of samples and their
MC reference. The existing generic sample-only recipes remain available for
historical workflows; the reset recipes use comparison targets.

The `analysis.run` registry and segment plotting code support at most one data
sample and one MC sample. The separate `analysis.compare_dqdx` command now supports
overlays of multiple data samples from saved dQ/dx products (see below). A future
registry should list members with separate sample IDs, selection versions,
display labels, and optional references to completed analysis runs. Sample
identity must be independent of data/MC kind throughout segment tables and plot
modules. Record member run signatures, common binning/units, normalization, and
any ratio reference. Reuse compatible tables or summaries when possible; do not
merge distinct reset productions just to make an overlay.

## Overlay the prc2, prc8, and prc16 data dQ/dx distributions

`python -m analysis.compare_dqdx` consumes each completed module's
`data_dqdx_histogram.csv` and `analysis.json`. It selects the data result from each
data/MC pair, preserves its saved MPV and chi-squared/ndf, and evaluates the same
fit function with the saved parameters. It does not refit normalized histograms
or repeat selection, segmentation, or MC analysis.

The overlay uses common bins and distinct orange/green/purple fills with alpha
0.5, outlines, and matching fit curves. Each bin shows its fraction of track segments
within the saved range (currently 0–90 ke−/cm): count divided by the total in-range
count, without dividing by bin width. Bin fractions sum to one; the y-axis is
"Fraction of track segments" and has no units. The saved fit curve is also divided
by the total in-range count.
This compares shapes despite different sample sizes. Legends show the reset
label, component MPV, and chi-squared/ndf. A second legend line gives the fitted
convolution's FWHM in ke−/cm and dimensionless FWHM/component-MPV. The width is
the separation of the two half-maximum crossings around the convolution's peak,
evaluated from saved fit parameters without truncation to the displayed or fit
range. No refitting or width uncertainty calculation is performed. These derived
widths and crossing positions are recorded in each member's manifest entry;
unavailable fits/widths are labeled, and a nonpositive MPV has an undefined ratio.
Component MPV is the Moyal-like
component parameter, not the maximum of the convolved curve. The original fit
range and raw-count fit statistics are retained.

The reader refuses incomplete or stale inputs, inconsistent histogram totals,
different segmentation/fit settings or relevant analysis-code hashes, duplicate
data selections, and existing output runs. A failed fit remains visible as a
histogram with a `Fit unavailable` label. Input product paths, hashes, settings,
fit summaries, and sample identities are recorded in the overlay manifest.
The `dqdx-overlay` comparison registry is consumed by this dedicated workflow;
it is not a data/MC registry for `analysis.run`.

From the repository root on NERSC, select the actual completed source runs. The
following uses run-002 from the new jobs and the historical prc8 comparison. If
prc8 was rerun under the explicitly named prc8 comparison, change PRC8 to that
run's dqdx directory. Source paths are relative to the fsdcube root (absolute
paths are also accepted); source files are read-only.

```bash
PRC2="comparisons/reflow-v3-feb2026-prc2_vs_fsdcube-sim-v1-prc256/analysis/run-002/dqdx"
PRC8="comparisons/reflow-v3-feb2026_vs_fsdcube-sim-v1-prc256/analysis/run-001/dqdx"
PRC16="comparisons/reflow-v3-feb2026-prc16_vs_fsdcube-sim-v1-prc256/analysis/run-002/dqdx"

mkdir -p "$SCRATCH/fsdcube/jobs/logs"
env -u SBATCH_MEM_PER_NODE -u SBATCH_MEM_PER_CPU -u SBATCH_MEM_PER_GPU \
  sbatch -A "$ACCOUNT" --cpus-per-task=2 \
    --job-name=dqdx-prc-overlay \
    --output="$SCRATCH/fsdcube/jobs/logs/dqdx-prc-overlay-%j.out" \
    jobs/compare_dqdx.sbatch "$SCRATCH/fsdcube" run-001 \
    --member "prc2=$PRC2" --member "prc8=$PRC8" --member "prc16=$PRC16"
```

The batch wrapper requests two logical CPUs, default site memory, and ten minutes,
and keeps the explicit srun CPU setting and conflicting-TRES removal. In an
existing compute allocation with the environment loaded, the direct equivalent is:

```bash
python -m analysis.compare_dqdx --root "$SCRATCH/fsdcube" --run-id run-001 \
  --member "prc2=$PRC2" --member "prc8=$PRC8" --member "prc16=$PRC16"
```

Outputs are stored under:

```text
comparisons/reflow-v3-feb2026-prc2_vs_prc8_vs_prc16/
  comparison.json
  analysis/run-001/
    config.json
    manifest.json
    dqdx_overlay/
      dqdx_prc_overlay.pdf
```

The overlay saves only the PDF plot plus its configuration and provenance JSON
files. Fit summaries remain in the manifest; no duplicate histogram or fit CSVs
are exported. The source analysis CSV/JSON products are still required as inputs.
Use a fresh overlay run ID for another rendering; `--alpha` changes fill opacity. Future
data groupings can use different labeled `--member` inputs and a descriptive
`--comparison-id` without changing the source sample organization.

To render the width annotations without a batch submission and preserve the
existing plot, activate the analysis environment and run from the repository root
with the member paths above:

```bash
python -m analysis.compare_dqdx --root "$SCRATCH/fsdcube" --run-id run-fwhm-001 \
  --member "prc2=$PRC2" --member "prc8=$PRC8" --member "prc16=$PRC16"
```

This writes `analysis/run-fwhm-001/dqdx_overlay/dqdx_prc_overlay.pdf` under the
same overlay comparison. Existing run directories are refused, never overwritten.

## Component MPV versus drift time with track-bootstrap errors

`python -m analysis.mpv_drift` reads completed shared segment tables. It uses
15 equal drift bins from 0 to 5000 CRS ticks (left-inclusive; 5000 is included in
the final bin), 100 dQ/dx bins over 0–90 ke−/cm, and the existing Moyal-like
component/Gaussian convolution fit over 20–50 ke−/cm. Fits need at least 100
entries in the fit range and five populated fit bins. It validates matching
segmentation settings and positive, matching `crs_ticks` metadata for every
source in every sample. No time conversion or lifetime correction is applied.

Each point is the original histogram's fitted component MPV. Error bars are the
sample standard deviation (`ddof=1`) of successful bootstrap MPV fits, with no
division by sqrt(number of replicas). Each replica draws N tracks with replacement
from the N tracks represented in that sample's segment table, including tracks
whose segments are outside the displayed ranges. All segments and readout groups
of `(file_id, source_event_row, track_id)` receive one common multiplicity across
all drift bins. Independent data/MC streams use stable label-specific seeds.
This captures within-track correlation, but does not model correlations between
separate tracks in the same event or calibration/selection/model systematics.

The default is 1000 replicates. An error bar requires two contributing tracks,
a successful original fit, and at least 90% successful bootstrap fits (and at
least two). Success counts/fractions and failures are recorded per bin. Original
points with unavailable uncertainty use open markers; failed original fits are
omitted. No fit-covariance error is silently substituted. Fit-covariance MPV SE is
saved separately as a diagnostic. No component-width or Gaussian-width outputs
are produced.

The lower panel is data component MPV / shared MC component MPV. Each ratio uses
the same MC bootstrap array, paired by replica index with its data bootstrap.
Ratio error bars are the standard deviation of these ratios, subject to the same
success threshold and available numerator/denominator uncertainties. Saved NPZ
arrays preserve replica ordering, NaNs for failed fits, and correlations across
drift bins and across ratios sharing MC. They can support later trend fitting.

Example inputs (adjust run IDs to the completed runs):

```bash
PRC2="comparisons/reflow-v3-feb2026-prc2_vs_fsdcube-sim-v1-prc256/analysis/run-002/shared/segments.hdf5"
PRC8="comparisons/reflow-v3-feb2026_vs_fsdcube-sim-v1-prc256/analysis/run-001/shared/segments.hdf5"
PRC16="comparisons/reflow-v3-feb2026-prc16_vs_fsdcube-sim-v1-prc256/analysis/run-002/shared/segments.hdf5"
python -u -m analysis.mpv_drift --root "$SCRATCH/fsdcube" \
  --data "prc2=$PRC2" --data "prc8=$PRC8" --data "prc16=$PRC16" \
  --mc "$PRC8" --run-id mpv-drift-001 --bootstrap-samples 1000
```

Run this on compute resources. Unlike the histogram overlay, this can require
60,060 nonlinear fits: four samples times fifteen bins times 1001 original plus
bootstrap histograms. A real prc8 drift-bin fit took about 0.66 seconds on the
local development machine; full production can therefore take hours, and fit
costs vary by bin. Start with `--bootstrap-samples 20 --run-id mpv-drift-pilot`
to measure speed and reveal sparse MC bins; that pilot is not a final uncertainty
estimate. Use a separate run ID for the 1000-replica result.

Checkpoints are written every ten replicas and at sample completion. Repeat the
same command with `--resume` to continue a stopped run, with unchanged input
identities, code, runtime, seed, replica count, and settings. Replica-indexed RNG
makes continuation identical to uninterrupted execution. A killed process may
leave `.writer.lock`; confirm the old process/allocation has stopped before
removing that one lock. Never run two writers in the same output directory.

Outputs live under
`comparisons/reflow-v3-feb2026-prc2_vs_prc8_vs_prc16/analysis/<run-id>/`:
`manifest.json` contains settings, provenance, per-bin MPVs/standard errors,
success fractions, and ratios; `mpv_drift/` contains `mpv_drift.png/.pdf`,
per-sample summary JSON, and sample/ratio bootstrap NPZ arrays. The existing
histogram-overlay comparison registry is left intact.

For the full run, a batch wrapper is available with two logical CPUs, no explicit
memory request, the same launcher fix, and a twelve-hour limit:

```bash
env -u SBATCH_MEM_PER_NODE -u SBATCH_MEM_PER_CPU -u SBATCH_MEM_PER_GPU \
  sbatch -A "$ACCOUNT" --cpus-per-task=2 --job-name=mpv-drift \
    --output="$SCRATCH/fsdcube/jobs/logs/mpv-drift-%j.out" \
    jobs/mpv_drift.sbatch --root "$SCRATCH/fsdcube" \
    --data "prc2=$PRC2" --data "prc8=$PRC8" --data "prc16=$PRC16" \
    --mc "$PRC8" --run-id mpv-drift-001 --bootstrap-samples 1000
```

A local two-replica smoke test of the historical prc8 table found valid original
fits in all fifteen data bins but only eight MC bins under the existing 100-entry
fit-range threshold. Additional MC statistics may be needed for a complete ratio
curve. That smoke test validates execution and layout, not uncertainty convergence.

## Time separation of two-hit pixels

`python -m analysis.hit_time_difference` compares data and MC using the complete
schema-2 selection hit tables. Within each saved event it groups hits by
`(io_group, io_channel, chip_id, channel_id)` and keeps only pixels with exactly
two recorded hits. Source file and source event identities remain separate.
Pixels with three or more hits are excluded, rather than yielding adjacent pairs.
Multiplicity is counted before rejecting nonfinite timestamps.

The default observable is `abs(ts_pps[1] - ts_pps[0])`, in CRS ticks. Integer
subtraction precedes float conversion, preserving small differences between
large timestamps without unsigned underflow. Equal timestamps yield zero and
are retained. The optional `--time-field t_drift` uses the difference of saved
calibrated drift times instead; the script never silently substitutes fields or
applies rollover corrections. Source tick periods must be available and agree
within and between the data and MC samples.

All saved event hits participate, including hits rejected by track selection;
there are no additional charge, disabled-channel, or fiducial cuts. The population
is conditional on the saved events containing an accepted track, not an inclusive
cosmic-data sample. `pairs_both_selected` in the audit reports how many pairs have
both hits selected, without filtering the plotted population.

Each data/MC bin is normalized by the total number of exactly-two-hit
pixel/event instances in that sample, without dividing by bin width. Different
pixels in one event count separately. The denominator includes instances outside
the plotted range and those with invalid timestamps; their counts are audited.
Thus the sum of displayed bin fractions can be less than one. CSV files contain
raw counts and a `fraction` column; the manifest records `normalization_instances`. The default is 200 bins
over 0–200 ticks (one tick per bin), resolving the short separations observed in
the local prc8 sample. The manifest records pairs above the upper bound, invalid-time
pairs, zero separations, multiplicity counts, source identities, and elapsed time.
Empty histograms are explicitly labeled. This is a single pass over event hit
slices with no fitting or bootstrap; memory is bounded by one event plus the
small histogram arrays.

From the NERSC checkout with the analysis environment loaded:

```bash
for reset in prc2 prc8 prc16; do
    python -u -m analysis.hit_time_difference \
      --config "configs/analyses/data_${reset}.json" \
      --root "$SCRATCH/fsdcube" --run-id hit-timing-001 \
      --data-label "Data ${reset}"
done
```

This uses the existing per-reset data/MC comparison registries prepared with
`prepare_reset_comparisons.py`. Each comparison receives its own
`analysis/hit-timing-001/manifest.json`, `config.json`, and a
`hit_time_difference/` folder with `hit_time_difference.png/.pdf` and
`data_delta_t.csv`/`mc_delta_t.csv`. Use a fresh run ID to change the binning, e.g.
`--run-id hit-timing-zoom-001 --max-delta-ticks 500 --bins 100`. Existing runs are
refused. This standalone module reads selection outputs directly and does not
rebuild the existing segment-based analyses.

A local full prc8/MC run processed 896,118 data hits and 20,771 MC hits in about
3.3 seconds. Direct execution is suitable for these sample sizes; no batch
submission is needed for this plot. NERSC filesystem speed can change the timing.

The fraction-normalized local prc8 preview has 185,172 data instances and 5,162
MC instances. Of these, 2,185 and 3 respectively lie above 200 ticks. The MC
minimum is 33 ticks; the data's main rise is around 31–32 ticks, but this is not
a strict measured hardware limit. Data include 268 zero-separation pairs whose
linked packet fields match after excluding source bookkeeping and receipt time,
and 11 positive separations below 30 ticks. These remain included, not silently
deduplicated. Both samples report a 0.1 microsecond CRS tick. The saved metadata
do not establish the production ADC hold/busy/reset settings, so the histogram
alone cannot determine the configured minimum cycle time.

### Combined periodic-reset hit timing plot

Overlay prc2, prc8, prc16, and the shared MC once using:

```bash
python -u -m analysis.hit_time_difference \
  --config configs/analyses/hit_timing_prc_comparison.json \
  --root "$SCRATCH/fsdcube" --run-id hit-timing-fraction-001
```

This recipe resolves the existing sample registries directly; no pair-registry
preparation is required. It creates a dedicated `hit-time-comparison` registry
under `comparisons/reflow-v3-feb2026-prc2_vs_prc8_vs_prc16_vs_fsdcube-sim-v1-prc256/`.
The standalone hit-timing module consumes this multi-sample recipe; it is not a
paired registry for the segment-analysis runner. A conflicting existing registry
is rejected. Outputs are under `analysis/<run-id>/hit_time_difference/`, with one
PNG/PDF overlay and separate `data_prc2_delta_t.csv`, `data_prc8_delta_t.csv`,
`data_prc16_delta_t.csv`, and `mc_delta_t.csv` tables. Each curve retains its own
all-two-hit-instance denominator, including overflow. Data use distinct colors;
MC uses a dashed orange line. Tick periods must agree across all four samples.
Existing paired recipes and their command-line arguments remain supported.
Choose a new run ID for each rerun; the existing outputs are not overwritten.
