# field-uniformity

Track selection for LArTPC data using DBSCAN and PCA fits.

On NERSC, load the existing analysis environment:

```bash
module load conda
conda activate /global/common/software/dune/efield2_rcmanduj/
python3 track_selection_fsd.py input.FLOW.hdf5 tracks.hdf5 False
```

The last argument selects 2×2 (`True`) or FSD (`False`) geometry. The output is
now HDF5, replacing the previous NumPy output. Choose a new output filename;
existing files are refused. The input stays read-only. Dependencies are listed
in `requirements-analysis.txt`; the existing environment may already provide them.

## Selection and PCA interpretation

The selection rules are unchanged: DBSCAN connects all event hits within 9.3 cm
(`min_samples=1`); clusters require at least 90 hits; the initial physical-coordinate
PCA must have relative spread ≤0.07; hits at distance ≥1.116 cm from that axis are
trimmed; at least 90 hits must remain. The legacy `PCAs()` standardized-coordinate
quality must be >0.973, and the legacy endpoint-defined length must be >55 cm.
Only 2×2 requires a light trigger with IO group 6. The anode-to-anode boundary test
is a saved flag, not an acceptance requirement. An empty coordinate-filtered event
is now safely skipped instead of passing an empty array to DBSCAN.

`efield.py` and its `PCAs()`/length/projection calculations are unchanged.
`PCAs()` standardizes each coordinate independently before PCA. Its saved
`pca_dir_standardized` is therefore **not a physical track direction**, and its
`pca_quality_standardized` remains the final acceptance quantity. Both use retained
hits. A separate output-only PCA in centimeters saves `pca_dir_physical`,
`pca_quality_physical`, `pca_center_physical_cm`, variances, and relative spread.
The initial untrimmed physical fit is also saved. PCA signs do not establish travel
direction. `projected_xyz_cm` retains the legacy projection onto the endpoint-defined
line; it is neither simulation truth nor a projection onto the added physical PCA.

## One hit table, two views

Only events containing at least one accepted track are stored. Every original
referenced calibrated prompt hit in those events is stored once, including hits
rejected by coordinate, cluster, or axis cuts. No rejected hit is automatically
associated to a track, and no additional hit-quality criteria are imposed.

| Dataset | Contents |
|---|---|
| `events/data` | Original event fields plus `file_id`, `source_event_row`, `hit_start`, `hit_stop` (exclusive), and `n_accepted_tracks` |
| `events/hits` | All original calibrated-hit fields, including electronics addresses, charge/timestamps/disabled flags where present; plus `file_id`, `source_hit_row`, `source_event_row`, `selected_track_id`, `projected_xyz_cm`, and `packet_link_count` |
| `tracks/data` | One row per accepted track, `file_id`, sequential `track_id`, source event row, both PCA definitions, initial PCA, endpoints/length, cluster/selected hit counts, and `a2a` |
| `events/t0` | All referenced t0 records, original fields plus `file_id` and source row/event row |
| `events/ext_trigs` | All referenced trigger records, original fields plus `file_id` and source row/event row |
| `events/packets` | All packets referenced by saved event hits, original fields (e.g. ADC/dataword, parity, packet/trigger/reset/CDS flags) plus `file_id` and source row/event row; each packet stored once within an event |
| `events/hit_packet_refs` | `file_id`, `hit_row` into the output hit table and `packet_row` into the output packet table; retains zero, one, or multiple links without choosing silently |
| `metadata/` | Input path, selection settings, source hashes of the three scripts, runtime versions, and copied run/geometry/hit-calibration/t0 attributes |

Hit and event IDs retain their native integer types and are distinct from dataset
row indices. Track IDs are unique within an output, including merged outputs. A single-input
output has `file_id=0`. A merge assigns file IDs by input order and remaps track IDs.
Source hits/events are identified by `(file_id, source_hit_row/source_event_row)`,
not by original IDs alone. File IDs are local to one merged output. Source event IDs are in `events/data/id`. Source hit IDs remain
in `events/hits/id`. Each `sources/<file_id>/metadata` group contains its own
input identity and run/calibration/geometry/timing/selection metadata. Optional packet/t0/trigger tables in per-input outputs have an `available` attribute;
per-source availability is also recorded on `sources/<file_id>`;
missing links give `packet_link_count=-1`, while available links with no packet give
0. All original structured fields are copied, so unavailable optional hit fields
such as `Q_raw` simply do not appear in that input's output schema.

The writer traverses direct forward FLOW references with source rows in column 0,
validates reference bounds/duplicates, and matches the selector's hit ordering by
explicit hit IDs. Unsupported references fail rather than silently misaligning
membership. Output `status` is `complete` only after successful processing;
failures leave `incomplete`. Outputs with no selected tracks still contain typed
empty tables. Packet rows shared across events may appear once per event, with
original source rows retained for auditing.

Coordinates and lengths are in cm; `t_drift` retains CRS ticks. The tick period for each file is
copied to `sources/<file_id>/metadata/run_info`'s `crs_ticks` attribute when supplied by the source.
Charge values retain the source FLOW units and calibration attributes. Timestamp
integers are preserved exactly, with no added rollover or timing correction.

```python
import h5py
import numpy as np

with h5py.File('tracks.hdf5', 'r') as f:
    event = f['events/data'][0]
    all_hits = f['events/hits'][event['hit_start']:event['hit_stop']]
    track_id = f['tracks/data'][0]['track_id']
    selected = all_hits[all_hits['selected_track_id'] == track_id]

    # Full-event versus selected charge on one selected track pixel.
    pixel = selected[0]
    address = ['io_group', 'io_channel', 'chip_id', 'channel_id']
    same_pixel = np.ones(len(all_hits), dtype=bool)
    for field in address:
        same_pixel &= all_hits[field] == pixel[field]
    recorded = all_hits[same_pixel]
    selected_pixel = recorded[recorded['selected_track_id'] == track_id]
    print(recorded['Q'].sum(), selected_pixel['Q'].sum())
```

This is an exact electronics-address grouping within one source event. When grouping
a merged hit table, include both `file_id` and `source_event_row` in the pixel key. Network aliases,
physical deposit completeness, and ownership of rejected same-pixel hits are not
inferred. The full-event view lets those questions be investigated.

## Batch processing and one merged analysis file

The recommended workflow keeps per-input files for reruns, then creates a single
merged file. Selection remains the same. Each input is processed sequentially in
its own subprocess with a separate log. Outputs are named using the original
basename plus a hash of the absolute path, avoiding collisions across directories.

From a compute allocation with the analysis environment loaded:

```bash
INPUT_DIR=/dvs_ro/cfs/cdirs/dunepro/www/data/nd-production/FSDCube/Reflow_FSDCube_v3/run-ndlar-flow/Reflow_FSDCube_v3.flow/FLOW/Feb2026/cold/scan/lt_prc8

# Pilot: one full FLOW file, per-input output plus merged-schema output.
python batch_track_selection.py \
  "$INPUT_DIR/Reflow_FSDCube_v3.flow.binary-cosmics-2026_02_27_17_17_31_PST.FLOW.hdf5" \
  --output-dir "$SCRATCH/fsdcube/pilot_parts" \
  --merged-output "$SCRATCH/fsdcube/pilot.hdf5"

# All matching files, deterministically sorted.
python batch_track_selection.py \
  --input-glob "$INPUT_DIR/Reflow_FSDCube_v3.flow.binary-cosmics-*.FLOW.hdf5" \
  --output-dir "$SCRATCH/fsdcube/full_parts" \
  --merged-output "$SCRATCH/fsdcube/full.hdf5"
```

Multiple positional paths, repeated `--input-glob` patterns, or `--input-list`
(one path per line, blank lines/# comments ignored) are supported. Duplicate
resolved paths and empty glob matches are refused. FSD is the default; pass
`--is2x2` for 2×2. Each input is processed in full; there is no event-limit option.

To resume, repeat the **same command with `--resume` added**. The batch manifest
records the requested inputs, per-file output/log paths, and completion/failure
statuses. Completed files are reused only when source path/size/mtime, detector
mode, and selector/helper source hashes match. Source files are not fully hashed.
Changing the selection code or input set requires a new batch directory.

Per-input outputs are first written to `.partial` files and published only after
successful completion and validation. An interrupted readable complete partial
can be published on resume. An incomplete or unreadable partial is preserved with
`.failed-<timestamp>` appended, and that FLOW input is rerun from its beginning.
Other completed inputs remain unchanged; no attempt is made to continue a partial
event. The batch stops at the first failure. Logs are overwritten when rerunning
an input; failed partial outputs remain available for inspection.

Merge existing completed per-input outputs independently:

```bash
python merge_track_outputs.py "$SCRATCH/fsdcube/full_parts/"*.tracks.hdf5 \
  --output "$SCRATCH/fsdcube/full.hdf5"
# Add --resume to recover an interrupted merge or reuse an identical completed one.
```

`merge_track_outputs.py` accepts schema **2.0** per-input outputs. Earlier schema
1.0 outputs must be regenerated. It preflights completion, duplicate FLOW source
paths, and schema compatibility; incompatible hit/event/packet fields or types
are refused rather than silently dropping information. Missing optional empty
packet/t0/trigger tables are supported, with availability recorded per source.
Differences in run conditions and selection metadata are retained, not normalized.

Merging copies datasets in bounded chunks (`--chunk-rows`, default 100000), remaps
track IDs, offsets event hit ranges and hit→packet links, and preserves original
source rows/IDs. Every data table includes `file_id`. Per-source metadata is stored
in `sources/<file_id>/metadata`; never assume the first file's tick period or
calibration applies to all files. Files with zero accepted tracks still receive
source metadata and completion status.

The merged file is written as `<output>.partial` and checkpoints each complete
input. On `--resume`, uncommitted rows and source groups are truncated before
reappending the interrupted input. Resume requires unchanged merge input order and
output-file sizes/mtimes. A final completed file is published without overwriting
an existing destination. Check `status=complete` before analysis. A severely
corrupted/unreadable partial merge must be rebuilt from the preserved per-input
outputs using a new destination.

Use one batch process per parts directory and one merger per destination.
Parallel jobs may write separate per-input outputs/directories and merge them
later; concurrent writers to one HDF5 destination are unsupported. The parts plus
merged output require disk space for both copies. Analysis should read event
slices or chunks rather than loading the whole merged hit table. Merging does not
associate activity across events or FLOW-file boundaries.

## Verification

```bash
python -m unittest discover -s tests -v
```

Synthetic HDF5 tests compare selected hit IDs with the legacy selection, check
standardized-cut rejection, physical versus standardized directions, rejected
same-pixel charge, multiple tracks/events, exact large timestamps, packet links,
missing optional tables, trigger rules, empty output, and failure/overwrite handling. Batch/merge tests cover repeated IDs across files,
per-source run metadata, reference remapping, schema conflicts, and interrupted
batch/merge recovery without duplicates.
Tests substitute the `h5flow` I/O boundary with a reader that deliberately reverses
hit ordering; they exercise the actual unchanged `efield.py` calculations. Production
`h5flow` reference behavior still needs a pilot run on the target FLOW file at NERSC.
