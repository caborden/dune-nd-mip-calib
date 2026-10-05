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

Available stages are `segments` and `dqdx`. Requesting `dqdx` automatically runs
or reuses `segments`; `all` requests every implemented stage. New scientific
modules such as hit density and pixel charge should have separate files,
output subdirectories, and registered dependencies. They are not implemented by
this organization change. The pixel module will eventually use the all-event hit
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
