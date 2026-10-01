# FSD-cube charge-splitting tools

These tools turn reconstructed FLOW hits into auditable event–pixel and track–pixel
candidate tables. They measure **recorded** charge, not total deposited charge or
truth-level splitting. The input file stays read-only. The original selector and
`FSDcubeData.ipynb` are unchanged; the new implementation is independent of their
hard-coded geometry and 55 cm cut.

## Quick start on NERSC

From the repository root, use a Python environment with the analysis dependencies.
The notebook kernel must use that same environment. Python 3.10+ is recommended.

```bash
python -m pip install -r requirements-analysis.txt
export FSD_INPUT='/dvs_ro/cfs/cdirs/dunepro/www/data/nd-production/FSDCube/Reflow_FSDCube_v3/run-ndlar-flow/Reflow_FSDCube_v3.flow/FLOW/Feb2026/cold/scan/lt_prc8/Reflow_FSDCube_v3.flow.binary-cosmics-2026_02_27_17_17_31_PST.FLOW.hdf5'

python -m fsdcube_analysis.scan "$FSD_INPUT" \
  --output results/pilot_address --max-events 500 --no-file-locking

MPLBACKEND=Agg python -m fsdcube_analysis.plots results/pilot_address \
  --output results/pilot_address/figures --sample event
MPLBACKEND=Agg python -m fsdcube_analysis.plots results/pilot_address \
  --output results/pilot_address/figures --sample track
```

`--no-file-locking` is optional and mirrors the read-only workaround in the existing
NERSC notebook. Files are never opened for writing. An existing scan output
directory is refused; choose a new directory for each selection. Plot files in a
chosen figures directory may be regenerated.

Open **FSDCubeChargeSplitting.ipynb** in Jupyter. Every code cell has an explanation
of its question and interpretation. Set `RUN_SCAN=False` and `OUTPUT` to a completed
scan to inspect it without rerunning. Otherwise the default is a new 500-event scan.
Optional environment overrides are `FSD_INPUT`, `FSD_OUTPUT`, `FSD_MAX_EVENTS`, and
`FSD_RUN_SCAN` (`1` to run, `0` to reload).

## Scripts and modules

| File | Purpose | How to use |
|---|---|---|
| `fsdcube_analysis/flow_io.py` | Reads event→hit, hit→packet, and event→t0 references; checks link bounds/duplicates; preserves IDs separately from row indices; attaches packet values and quality flags. | `hits, audit = read_event(f, event_row, file_id)` with an open h5py file. `linked_rows` traverses a direct forward reference; `take_rows` supports arbitrary requested row order. |
| `fsdcube_analysis/charge_splitting.py` | Adds explicit pixel keys and constructs chronologically ranked hits, pixel sums, and adjacent-time gaps. | `hits = add_pixel_keys(hits, geometry_attrs, mode='address')`; `ranked, pixels, gaps = summarize_pixels(hits)`. Set `track=True` only for uniquely associated hits with nonnegative track IDs. |
| `fsdcube_analysis/tracks.py` | Finds exploratory straight tracks using one representative per pixel; trims/refits in physical coordinates; recovers eligible original hits on seed pixels; vetoes competing-line ambiguity. | `associated, candidates = associate_tracks(one_event_hits, TrackConfig(...))`. Input must already have quality checks and pixel keys. |
| `fsdcube_analysis/scan.py` | Bounded multi-file scan, per-event audit, track association, CSV output, and provenance manifest. | CLI above or `tables = scan_files(inputs, output, max_events=500, track_config=TrackConfig())`. `tables, manifest = load_results(output)` reads completed results with nullable integer packet fields. |
| `fsdcube_analysis/plots.py` | Multiplicity, timing, charge-sum, Q1/Q2, charge-fraction, and channel/chip comparison plots, plus event displays. | CLI above or `figures, channels, chips = make_plots(tables, output, sample='event', tick_us=0.1)`. Get the actual tick period from the manifest. `plot_event(hits, projection='yz')` expects a single event. |
| `tests/test_fsdcube_analysis.py` | Synthetic HDF5 integration tests for references, grouping, quality, recovery, empty samples, plotting, and file separation. | `MPLBACKEND=Agg python -m unittest discover -s tests -v`. No real data or network is required. |
| `fsdcube_analysis/__init__.py` | Package description and tool version recorded in the manifest. | Imported automatically. |

The new tools require neither `h5flow` nor `ndlar_flow` to run. Their direct reference
reader supports the source-column ordering in the inspected production. It does not
guess reverse reference paths. See [source references](FSDCube_references.md).

## Definitions and output tables

Within a scan, `(file_id, event_row, hit_row)` identifies a source hit; its original
`id` is retained separately. File IDs are mapped to absolute source paths in
`manifest.json`. **File IDs are local to one scan**: when combining separate scans,
add a scan ID or remap file IDs by source path. Do not concatenate outputs blindly.
Track identity is `(file_id, event_row, track_id)`.

The event-pixel key is `(file_id, event_row, io_group, pixel_io_channel, chip_id,
channel_id)`. In address mode, `pixel_io_channel` is the original IO channel. In
network mode it is `(io_channel-1)//n_io_channels_per_tile`; this requires
`network_agnostic=True`. Original IO channels are always retained. Network grouping
is a hypothesis pending LUT validation; consistent y/z alone does not prove it.

| Output | Contents and denominator |
|---|---|
| `events.csv` | One row per requested event actually present, including empty and reference-error events; expected/read hit counts, t0, quality counts, accepted-track count. A reference failure leaves unavailable counts blank. |
| `hits.csv` | Every readable referenced hit, original charge/position/timestamps and electronics IDs, packet row/dataword/flags, quality status, event-pixel chronological rank/gap, and track association. |
| `pixels.csv` | One row per observed event–pixel key: multiplicity, Q_total, Q1/Q2/Q3, timing, coordinate spans, quality/tie flags. There are no zero-hit pixel entries. |
| `gaps.csv` | One row per adjacent-hit interval within each event–pixel key; an N-hit pixel contributes N−1 intervals. |
| `tracks.csv` | All fitted/clustering candidates, including rejected clusters and their first rejection reason. Accepted fits include center, unit axis, spread, and projected length. Events with fewer than three usable representatives have no cluster rows. |
| `track_hits.csv` | Uniquely associated hits, reranked within track–pixel groups. Includes `association_role` (`seed` or `recovered`). |
| `track_pixels.csv` | Associated charge sums with original event-pixel counts and `association_complete`; sums may be partial and are labeled accordingly. |
| `track_gaps.csv` | Adjacent-hit intervals within each track–pixel group. |
| `manifest.json` | Completion/failure status, all selection settings, file metadata/paths, geometry/run/calibration/t0 attributes, source hashes, dependency versions, Git commit/status, units, and row counts. Full input files are not hashed. |

Q/Q_raw are in ke−, position in cm, `t_drift` in CRS ticks, and derived time gaps in
microseconds. `crs_ticks` supplies the conversion. Q1 means earliest reconstructed
drift time **within the event**, not largest charge. An equal-time tie is sorted by
hit row for reproducibility but is not considered resolved ordering. For N>3 use
the ranked hit table rather than truncating to the convenience Q1/Q2/Q3 columns.

## Quality and reference checks

- Missing required schema fields fail the scan; the manifest records the failure.
- Duplicate/out-of-bounds references skip the affected event with an explicit audit
  error. Unsupported/missing reference paths fail rather than masquerading as empty.
- Each usable calibrated hit must link to exactly one packet. Reused packet rows,
  address mismatch, bad parity, unexpected data-packet type, disabled flags, or
  nonfinite charge/time/position invalidate it for fitting and quality plots.
- Event hit-count mismatches or other than exactly one t0 record invalidate selection
  for that event. Type NONE is allowed and preserved; an external trigger is not required.
- A pixel sum passes quality only if all constituent hits pass and its y/z spans are
  at most 1e-4 cm. Ordered-charge plots also exclude timestamp ties.
- Negative and zero charges are retained. Only fractional-charge plots additionally
  require positive total charge. No trigger/reset/CDS flag interpretation is imposed.
- Packet timestamp/address/type/parity fields remain available for further audits.
  Raw timestamp offsets are not required constant across all events. No new rollover
  correction is applied: ordering uses the production's event-relative `t_drift`.

No checks prove completeness across event boundaries, correct physical t0, absence
of acquisition duplication, or correct calibration. Type NONE events have unverified
absolute x/drift position; do not infer cathode/anode contact or lifetime directly.

## Track selection and association

The earliest quality-passing hit per candidate pixel is used as one representative
so multiplicity does not directly increase fit weight. DBSCAN operates separately
per IO group in reconstructed 3D coordinates. Candidate clusters are trimmed around
an initial PCA axis, then refitted using an SVD in physical coordinates. Length is
max−min projection along the refitted axis. The direction sign is deterministic but
does not identify muon travel direction.

| CLI option | Default | Interpretation |
|---|---:|---|
| `--eps-cm` | 3.0 | DBSCAN connection scale, min_samples=1 |
| `--min-pixels` | 20 | Distinct retained representatives, before and after trimming |
| `--min-length-cm` | 10.0 | Projected track length; exploratory replacement for 55 cm |
| `--max-relative-spread` | 0.15 | sqrt(transverse variance sum / longitudinal variance) |
| `--axis-distance-cm` | 1.116 | Initial fit trimming distance |
| `--recovery-us` | 10.0 | Maximum absolute time difference from that pixel's representative |
| `--recovery-distance-cm` | 1.116 | Maximum distance from the refitted axis |
| `--endpoint-margin-cm` | 1.116 | Allowed extension of the projected track interval |

After fitting, the code revisits **all original event hits** on retained seed pixels.
Hits passing quality, time, distance, and endpoint conditions are associated. It does
not recover new pixels outside that footprint. A hit near another accepted line in
the same IO group and its time envelope is conservatively marked ambiguous (`-2`),
not assigned to the nearest line. Unassociated hits have ID `-1`. This ambiguity veto
cannot detect competing tracks merged by DBSCAN, or tracks below the acceptance cuts.

Default track plots require `association_complete=True`: all original event-pixel
hits belong to that track. This avoids silently displaying partial sums as complete,
but biases acceptance against pixels with out-of-window/noisy/ambiguous hits. The
notebook exposes those excluded groups and compares recovery windows. Even complete
associations are only complete **within the recorded event**.

These cuts are development starting points. Validate with event displays and scans
over cut values. No calibrated muon classifier, pixel path-length calculation,
Landau fitting, electron-lifetime correction, or truth matching is included.

## Additional runs

```bash
# Exact addresses versus provisional network aliases: keep outputs separate.
python -m fsdcube_analysis.scan "$FSD_INPUT" --output results/pilot_network \
  --pixel-mode network --max-events 500 --no-file-locking

# Raw candidate distributions only, without track finding.
python -m fsdcube_analysis.scan "$FSD_INPUT" --output results/pilot_no_tracks \
  --no-tracks --max-events 500 --no-file-locking

# Compare the old minimum length; this rejects the inspected event 7.
python -m fsdcube_analysis.scan "$FSD_INPUT" --output results/pilot_length55 \
  --min-length-cm 55 --max-events 500 --no-file-locking

# Next nonoverlapping batch, using the same configuration.
python -m fsdcube_analysis.scan "$FSD_INPUT" --output results/batch_500_999 \
  --start-event 500 --max-events 500 --no-file-locking
```

Multiple positional input paths are accepted; the limit applies independently to
each file. A start index beyond the file yields zero requested-present events, not
an error. The pilot implementation stores tables in memory and traverses packet
references per hit; use bounded batches rather than an unbounded production run.

The timing figure contains a native-tick 0–20 µs zoom and a full-range ECDF so long
separations are not silently discarded. Charge histograms use shared bins and counts,
not normalized densities. Channel/chip repeat-fraction CSVs give both numerator and
denominator; low-count channels are not reliable rate estimates. Distinct files
remain separate in those tables. Compare angles/path lengths and run conditions
before attributing charge-distribution differences to readout splitting.
