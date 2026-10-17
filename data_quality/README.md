# Inclusive FLOW sample quality

Compare one or more original FLOW samples without track selection or a merged
input file. Every file contributes to collective sample accumulators. There are
no per-file plots. The first sample is the reference for ratio panels.

The example config selects all matching date subdirectories below the February
`cold/scan/lt_prc16` and October `halfField_5ke_prc16_processed_flow/flow` roots.
An earlier version listed only the four February and two October filenames shown
in the initial request; reports made with that version cover only those files.
Run `--check-inputs` to verify the expanded file counts before submitting a job.
The exact expanded list is saved in `config.resolved.json` for each run.
Other operating conditions are
not established by their filenames. Paths have not been verified on NERSC from
the local development machine; run the input check below.

## Files

| File | Responsibility |
| --- | --- |
| `run.py` | CLI, config validation, input inspection, processing orchestration, provenance and completion/failure status. |
| `flow.py` | Read-only HDF5 access, input deduplication, metadata/schema inspection, and bounded forward event-to-hit reference traversal. |
| `metrics.py` | Streaming hit/packet/event histograms, per-channel occupancy and binned charge medians, exact integer timing differences, spatial projections. |
| `report.py` | Collective overlays and ratios, spatial/channel maps, CSV/NPZ tables, HTML report, multipage plot PDF, and audit JSON. |
| `config.example.json` | Full-sample file patterns across date subdirectories, labels, binning, ratio-display settings, and memory controls. |
| `run_nersc.sbatch` | CPU shared-queue job using the existing analysis conda environment and the repository's Slurm launch convention. |
| `test_data_quality.py` | Synthetic FLOW regression tests, including unequal-size files, references, timing, missing fields, and report generation. |
| `requirements.txt` | Only NumPy, h5py, and Matplotlib are needed; no h5flow or track-analysis dependencies. |
| `__init__.py` | Makes `data_quality` importable as a Python package. |

## Transfer code with Git

From the local checkout, review and commit **only this folder**. These commands
assume the current branch remains `fsdcube-charge-splitting`. Existing unrelated
untracked files elsewhere in the repository are not part of this change.

```bash
cd /Users/casey/MessierLab/dune-nd-mip-calib
git status --short
git branch --show-current
git add data_quality
git diff --cached --stat
git diff --cached -- data_quality
git commit -m "Add inclusive FLOW data quality comparisons"
git push -u origin fsdcube-charge-splitting
```

`git add` stages files, `git commit` records the staged snapshot locally, and
`git push` sends the branch to `origin` (`caborden/dune-nd-mip-calib`). If you
already have unrelated staged changes, unstage those before committing this
folder. Do not discard local modifications to resolve a rejected push.

On NERSC, enter your existing checkout and update the same branch:

```bash
cd /path/to/your/dune-nd-mip-calib
git status --short
git fetch origin
git switch fsdcube-charge-splitting
git pull --ff-only origin fsdcube-charge-splitting
```

If the branch does not exist locally, use
`git switch --track origin/fsdcube-charge-splitting` instead of `git switch`.
If Git reports local modifications or divergent history, resolve those before
updating; `--ff-only` deliberately refuses an implicit merge. Avoid switching or
updating the checkout while a job is actively importing code from it; wait for
that job or use a separate checkout. This matters if the track-selection job
mentioned in the request is running from this same directory.

## Inspect inputs on NERSC

Run from the repository root:

```bash
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /global/common/software/dune/efield2_rcmanduj/
python -c 'import numpy, h5py, matplotlib; print("Dependencies available")'

python -m data_quality.run \
  --config data_quality/config.example.json \
  --check-inputs
```

This checks all paths and required schemas, prints hit/event counts, optional
packet fields, reference availability, and CRS tick periods, and writes nothing.
Plot legends use `fields: A/B inputs`: A is the number of resolved input files
with the fields needed for that metric and B is the number configured/resolved
for the sample. This does not count all files elsewhere on disk, and fields being
present does not guarantee populated data. In particular an ADC schema can be
present even when there are no data packets selected.
It does not scan all hit values or verify all reference contents. A successful
preflight is not a completed quality analysis. Required inputs are
`charge/events/data` and the configured calibrated-hit dataset, with scalar
`Q`, `io_group`, `io_channel`, `chip_id`, and `channel_id` fields. Missing optional
fields disable only the affected metrics and are recorded in the report.

If the environment lacks dependencies, use a personal environment and install
`python -m pip install -r data_quality/requirements.txt`; do not modify the shared
environment. Update the activation line in the batch script to that environment.

## Run the analysis

For a full production-sized sample, submit a batch job from the repository root.
Replace `YOUR_NERSC_ACCOUNT` with the allocation used for your other jobs:

```bash
mkdir -p "$SCRATCH/fsdcube/data_quality/logs"
sbatch -A YOUR_NERSC_ACCOUNT \
  --output="$SCRATCH/fsdcube/data_quality/logs/dq-%j.out" \
  data_quality/run_nersc.sbatch \
  data_quality/config.example.json \
  "$SCRATCH/fsdcube/data_quality/feb_vs_oct_prc16_run001"

squeue -u "$USER"
# Replace JOBID with the number returned by sbatch.
tail -f "$SCRATCH/fsdcube/data_quality/logs/dq-JOBID.out"
```

The four-hour request is an initial allocation, not a runtime estimate; adjust
with `sbatch --time=...` if necessary. The script follows the existing repository
CPU/shared-QOS setup. It is a single-process analysis, needs no GPU, and can run
independently of track selection. It may compete for filesystem I/O with other
jobs reading the same inputs. Per-file progress messages are logging only.

Inside an existing interactive compute allocation, the equivalent command is:

```bash
python -u -m data_quality.run \
  --config data_quality/config.example.json \
  --output-dir "$SCRATCH/fsdcube/data_quality/feb_vs_oct_prc16_run001"
```

Choose a new output directory for every run (`run002`, etc.). Existing output
directories are refused, including failed runs, and inputs are always opened
read-only. A run is complete only when `summary.json` contains
`"status": "complete"` and the job exits successfully. Warnings can still indicate
unavailable or excluded data; inspect them before interpreting comparisons.

## Outputs and interpretation

- `report.html`: sample-wide plots with coverage warnings, integrity counters,
  and links to numerical outputs. Keep `plots/` beside it when moving it.
- `report.pdf`: standalone multipage plot collection. Detailed metadata and
  integrity warnings remain in the HTML/JSON report.
- `summary.json`: aggregate counters; per-file schema, calibration/geometry
  attributes, optional-field availability, timestamp envelopes, and integrity
  bookkeeping; settings, source-code hashes, runtime versions, and run status.
- `config.resolved.json`: exact expanded input list and effective settings.
- `plots/`: PNG overlays with ratios, spatial projections, occupancy and median
  charge maps by IO group, and occupancy-ratio maps.
- `tables/`: histogram CSVs (counts, fractions, reference ratios), channel CSVs
  (addresses, counts, positions, binned medians, overflow, ratios), and spatial
  histogram NPZs (counts, edges, normalization).

To copy the complete report back to your laptop, run locally, replacing the
remote username and absolute output path with yours:

```bash
scp -r YOUR_USERNAME@perlmutter.nersc.gov:/absolute/scratch/path/fsdcube/data_quality/feb_vs_oct_prc16_run001 \
  /Users/casey/MessierLab/dune-nd-mip-calib/scratch/
```

Create the local `scratch/` directory first if needed. Open `report.html` in a
browser or `report.pdf` in a PDF viewer. The repository already ignores `scratch/`.

### Populations and normalization

**Hit plots** include every row in the selected calibrated-hit dataset, even
unassociated hits and disabled-flagged hits. There are no charge, track, or
fiducial cuts. Hit and event populations therefore need not have equal totals.
**Event plots** use unique linked hit row indices for each event. Duplicate
reference links are counted and collapsed; distinct hit records are retained,
including records with identical timestamps. Empty events enter event hit count,
channel count, and charge histograms with zero, but have no duration or timing
pairs. Invalid-reference events are counted and skipped for event metrics.
An event containing nonfinite charge is excluded from the event-charge histogram
instead of being assigned a partial sum.

**ADC plots** use all packet rows with `packet_type == 0`, including bad-parity
packets; parity failures are counted separately. Without both `packet_type` and
`dataword`, ADC plots are unavailable. `adc_max` defaults to 255 and the audit
counts values at/above that configured endpoint; verify the electronics setting
before interpreting endpoint occupancy as physical saturation.
The audit records per-file packet-type counts and warns when the schema exists
but no `packet_type == 0` rows were found, including an empty packet dataset.

Counts are accumulated across files **before** normalizing. Histogram fractions
are counts divided by the total entries supplied to that histogram, including
nonfinite, underflow, and overflow entries. They are fractions per bin, not
probability densities per unit x. Missing fields and rejected events are reported
separately and do not enter that histogram's denominator. A displayed histogram
may sum to less than one. The full denominator and coverage are in `summary.json`.
Ratios with zero reference content are undefined (`nan` in CSV), with no
pseudocounts. No statistical error bands or significance tests are claimed.

One-dimensional ratio panels default to a fixed **0 to 2** display range so
sparse-denominator extremes do not compress the rest of the curve. Up/down
triangles at the panel boundary identify values outside this range; the panel
reports their count. Hollow markers identify bins with fewer than 20 reference
entries. Those bins are still shown, not rejected, and a zero numerator remains
zero. Every finite ratio is saved unmodified in the CSV, alongside reference
counts and display flags; `summary.json` records display statistics. The count
threshold is a visual aid, not a statistical significance criterion. Configure
`ratio_ylim` (e.g. `[0, 5]`) and `ratio_min_reference_count` to change this behavior.

Occupancy maps show **fraction of all sample hits**, not Hz or efficiency.
Ratios divide these fractions at matching electronics addresses. Zero numerator
hits give ratio zero; no reference hits give an undefined ratio. The report has
no external channel inventory/mask: it cannot distinguish dead, disabled,
unmapped, and enabled-but-unobserved channels. IO-group maps use hit y/z as the
readout plane, assuming drift along x; validate that convention for the sample.
Positions must remain constant to 0.001 cm per address within the sample; ratio
maps also require compatible positions between samples.
An unmatched electronics address can occupy a physical position that is populated
under a different address in the other sample. Thus blue/gray patches on an
address-matched ratio map do not by themselves establish missing physical coverage;
compare the individual occupancy maps and validate the address/geometry mapping.

Channel medians use **pooled fine charge histograms**, not averages of per-file
medians. With the default `[-20, 200, 880]` bins, each middle-order-statistic charge
is approximated by its 0.25-unit bin midpoint (up to 0.125-unit quantization error).
If either middle rank lies outside the range, the median is unavailable and
flagged in the channel CSV. Counts outside that range still contribute to the
rank calculation. Widen `channel_charge_bins` if medians are censored.

### Timing, geometry, and integrity limits

- Timing uses `ts_pps` by default. `run_info/crs_ticks` is read per file in us,
  so different known clock periods are converted before pooling. Missing or
  invalid clock periods disable time plots for that file. Set `time_field` to
  `t_drift` only deliberately; it changes the interpretation of time differences.
- Integer timestamps are differenced **before** converting to floating point.
  No rollover or clock-reset correction is inferred. Float timestamps cannot
  regain precision already lost by upstream processing.
- Event duration is `max(hit time)-min(hit time)`, not the event-builder window.
  All adjacent time-sorted hits on a channel within an event contribute to delta-t,
  including multiplicities above two. No pair crosses events or files. If any
  timestamp in a channel/event is nonfinite, that group's timing pairs are skipped
  rather than bridged across the invalid record.
- Both samples are known to be half field. No lifetime, recombination, live-time,
  or extra calibration correction is applied. Following the user's convention
  for these productions, stored `Q` is interpreted as **ke- (thousands of
  electrons)**: `Q = 5` means 5,000 electrons. Both the default and example config
  use `charge_unit: "ke-"`. This setting changes labels and recorded unit metadata
  only; it does not rescale charge values or histogram bins. Hit charge, summed
  event charge, and channel median charge all use ke-; raw ADC remains dataword.
  If using an older custom config, update its explicit `charge_unit` as well.
- Spatial ranges default to the shared finite data extent across samples, then
  a second sequential coordinate pass fills histograms. Outliers can stretch
  these ranges: set `spatial_ranges`, e.g. `{"x": [-40, 40]}`, using validated
  bounds. Configured ranges are plotting ranges, not fiducial cuts. No stored
  geometry bounds are silently assumed valid (prior FSD Cube metadata contained
  degenerate bounds). Coordinates outside plot ranges are counted separately.
- Input paths, symlinks, and hard links are deduplicated by filesystem identity.
  Overlapping per-file event timestamp envelopes are warnings, not proof of
  duplication and not measurements of live time. Copies of the same acquisition
  with different filesystem identities are **not** automatically deduplicated.
- Forward reference bounds, invalid source/target rows, duplicate links,
  unvisited reference rows, nonfinite values, disabled flags, and zero-time
  same-channel pairs are audited. Zero-time pairs are duplicate candidates, not
  proven duplicate packets. No exhaustive cross-file record/ID deduplication is
  performed, and IDs are never treated as row indices.

## Configuration and resource usage

Each sample's `files` accepts exact paths or glob strings; relative paths resolve
against the JSON's directory. Empty matches and repeated inputs are errors.
Explicit lists make reruns reproducible as new files arrive. One sample is valid
for standalone QA; multiple samples overlay against the first reference.

Histogram specifications are `[low, high, number_of_bins]`. The provided ranges
are starting points, not selection cuts; check overflow counters after running.
To customize without changing the tracked example, copy it to a local/scratch
JSON, edit the settings, and pass that path to the same CLI/batch script.

`chunk_size` bounds sequential hit/packet/reference reads. `event_batch_size`
controls how many event reference regions are prefetched together. Batches that
span more than `max_event_links` are processed event by event; an individual
reference region above that limit is counted and omitted, not read unboundedly.
Raise that limit if legitimate events are omitted and memory permits.

Memory scales with these read limits, histogram bins, and the number of distinct
observed electronics channels, not total input hits. The median estimator costs
roughly 8 bytes per charge bin per observed channel plus object overhead.
Inclusive hit/packet reads, event-associated reads, and the coordinate pass mean
the analysis reads some data more than once. It does not create a merged FLOW
file. Input size/mtime are checked again at the end to detect files being changed
during processing; use completed production files.

## Tests

From the repository root with the dependencies available:

```bash
python -m unittest data_quality.test_data_quality -v
```

Tests create tiny temporary FLOW fixtures, check pooling independently of file
partition, exercise IDs distinct from row numbers and non-contiguous reference
regions, preserve uint64 timing precision, and render a complete report. They
do not validate the physics calibration or actual NERSC input contents.
