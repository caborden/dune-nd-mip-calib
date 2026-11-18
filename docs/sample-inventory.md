# NERSC sample inventory

Run from the analysis checkout with the analysis Python environment loaded.

```bash
export FSD_ANALYSIS_ROOT=/pscratch/sd/c/cborden/fsdcube
python sample_inventory.py \
  --root "$FSD_ANALYSIS_ROOT" \
  --campaign-jobs /global/cfs/cdirs/dune/users/cborden/muon-skim-jobs \
  --production-root /pscratch/sd/c/cborden/fsd-muon-production
```

The generated document is `$FSD_ANALYSIS_ROOT/sample-inventory.md`. Discovery
settings are saved in `inventory-config.json` at the same root. Subsequent calls
can use just `--root`. `--selection /absolute/path.hdf5` adds an external selection
to persistent discovery. A file lock covers discovery and atomic publication so
concurrent completion hooks preserve additional selection paths.

Each resolved HDF5 path appears once with registry aliases, status, track count,
selection cut and provenance paths. Only metadata and dataset dimensions are
read. Registered missing files remain listed. Pilots are included. Individual
parts, analysis outputs, archived samples and simulation intermediates are
excluded. Files without track-selection tables are ignored. Unregistered samples
have unknown kind until a registry or campaign identifies them. A missing cut is
reported as unknown. Frozen detector YAML values are listed explicitly. An absent
ADC busy-delay override is reported as unspecified/inherited, even if the sample
name suggests a value. This avoids inferring settings from a mutable simulator.

## Completion hooks

The updated `batch_track_selection.py` refreshes after a successful merged batch,
including a successful resume. The command-line entry points of
`merge_track_outputs.py` and `track_selection_fsd.py` refresh after publishing the
output. Per-input files under a `parts` directory are skipped. A batch without a
merged output is represented when its later merge completes.

The hooks use `FSD_ANALYSIS_ROOT` when set, otherwise the output's ancestor named
`fsdcube`. Set the environment variable for selections stored outside that tree.
The first manual invocation above configures campaign discovery for later data
completion hooks. Inventory failures print a warning after selection completion.

The updated ND_Production campaign launcher copies this generator from
`$WORK/dune-nd-mip-calib/sample_inventory.py` into each newly prepared campaign's
frozen `code` directory and hashes it with the other code. It prints a warning
and disables the hook if the generator is absent. After the campaign publishes
`summary.json`, it invokes the frozen generator with its campaign and production
roots. Its inventory root is captured from `FSD_ANALYSIS_ROOT`, defaulting to
`$SCRATCH/fsdcube` at preparation time.

Existing prepared campaigns and frozen selection snapshots retain their original
code. Refresh their inventory manually after completion. New data jobs must use
the updated analysis checkout or a new snapshot including `sample_inventory.py`.
The selector script hash changes, so old batches should be resumed with their
original snapshot. Use a new output directory for new selection runs.

## Verification

```bash
python -m unittest discover -s tests -p 'test_sample_inventory.py'
```

After the first NERSC invocation, inspect the generated document's selections,
campaign summaries, missing-file statuses and read-error section. Re-running the
same command refreshes it without moving or changing any sample files.
