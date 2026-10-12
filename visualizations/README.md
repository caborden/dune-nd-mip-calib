# FSD and FSD Cube visualization notebooks

Two notebooks share small Python helpers. They read your checked-out configuration,
not copied detector constants. The default larnd-sim location is the sibling
`../larnd-sim`; set `LARND_SIM_REPO` or edit `LARND_REPO` in each notebook.

## Install and open locally

From the dune-nd-mip-calib repository root:

```bash
python3 -m venv visualizations/.venv
source visualizations/.venv/bin/activate
python -m pip install -r visualizations/requirements.txt
python -m jupyter lab visualizations/
```

Select the Python kernel from this environment and use **Run All**. No CUDA is
needed for the geometry notebook or the signal teaching example. The requirements
are for a separate notebook environment: do not use them to upgrade a working
larnd-sim CUDA environment, which may constrain NumPy and other dependencies.

- **01_detector_geometry.ipynb**: rotatable FSD/FSD Cube readout volumes, anodes,
  cathodes, electron drift arrows, tile labels; tile/chip pixel maps with nominal
  electronics addresses and simulation pixel IDs; source/destination translation
  overlay from the supplied skimming notebook.
- **02_signal_readout.ipynb**: current, amplifier/integral, and ADC panels with
  linked time axes and a 3D track/pixel view. Includes reset/threshold controls,
  actual simulation-capture loading, input-deposit context, and raw packet viewing.

Run from the repository root or a subdirectory. Outputs go into ignored
`visualizations/outputs/`. The notebooks save standalone HTML with Plotly embedded,
so rotation, zoom and hover work offline. Python widget controls require Jupyter;
exported HTML preserves the selected state, not the notebook controls.

## What the geometry means

Public helper coordinates are **edep-sim XYZ in centimeters**, with x the drift
axis, y vertical, z transverse. Plotly draws y vertically and labels all axes.
Internally larnd-sim exchanges x and z. Tile positions from YAML are millimeters;
module offsets are centimeters. Tile orientation flips are applied to the pixel
map. Grid IDs use the layout's tile map and zero-based plane indices; they are
distinct from electronics addresses and one-based layout TPC labels.

The supplied layouts have a 128 by 80 pixel-center grid (10,240 positions) at
3.72 mm pitch per tile. FSD has 40 tiles; FSD Cube has one. Values shown by the
notebook are recomputed from the selected checkout. Cathode location comes from
the anode location, orientation, and drift length. A missing single-anode drift
length is rejected. Only single-module layouts are supported.

These are readout/active-volume schematics. They do not invent cryostat, field cage,
PCB pad shape, shield geometry, or mechanical dimensions absent from the YAML.
Nominal IO routes are not a data-run channel-mask or routing-alias model. The
existing ND_Production `geometry/inspectgeom.C` and `geometry/eve_display.C` can
render detailed FSD GDML separately.

The skimming notebook's 40 regions are **harvesting windows**, not exact FSD Cube
active volumes. The translation overlay keeps their original constants and shows
their mismatch with the current FSD Cube bounds. It is a shared simulation
coordinate frame, not an FSD Cube physically installed in FSD.

## Signal modes and their limits

### Immediately runnable teaching example

Analytic positive and bipolar pulses feed a noise-free integrate/threshold/hold/
reset model. The example uses selected YAML ADC transfer parameters and shows how
reset phase/frequency and threshold can change readout. A zero periodic-reset
setting disables periodic resets. The schematic muon and pulse shapes are not
computed from one another. All plots and saved bundles say **illustrative**.

This is not a replacement for Geant4, field-response calculation, diffusion, or
larnd-sim electronics. In particular it omits stochastic noise and buffer response
and simplifies hold/busy/reset timing. Numerical agreement with production is not
claimed. The approximately zero integral of an induction-only-like pulse does not
guarantee zero thresholded readout.

### Capture a real larnd-sim run

Use the **existing larnd-sim GPU environment**, with its normal dependencies and
external response files. The wrapper itself adds only NumPy; it does not require
Plotly or Jupyter during capture. Run from the dune-nd-mip-calib root, or add that
directory to `PYTHONPATH` if your larnd-sim setup requires a different working
directory for its relative asset paths. The wrapper preserves the working directory.

```bash
# Activate your validated larnd-sim environment first.
# Use absolute input/output paths and a small, existing single-event pilot input.
python -m visualizations.capture_larnd \
  --larnd-repo /path/to/larnd-sim \
  --trace-dir /path/to/new-traces \
  --detector-name 'FSD Cube' \
  --max-channels 8 --max-batches 4 \
  -- \
  --config fsd_cube \
  --input_filename /path/to/pilot.EDEPSIM.hdf5 \
  --output_filename /path/to/pilot.LARNDSIM.hdf5 \
  --rand_seed 123
```

Everything after `--` goes to the checked-out `run_simulation` function through
Python Fire. Use the same configuration/options as your validated normal run;
`fsd_cube` above is only an example, not a recommendation for a particular reset
or far-field configuration. The detector-name option is a display label, not a
configuration selector. A complete CUDA simulation is still run even after the
capture limit is reached.

To capture chosen channels, add before `--`:

```text
--pixel-ids 123,124,125 --event-id 42
```

Those are **simulation grid IDs**, obtainable by hovering over the notebook's pixel
map, not chip/channel addresses. IDs missing from a particular digitization call
are skipped; choose enough `--max-batches` to capture the desired calls. Without
an explicit list, the first channels in each call are recorded; their physical
collection/induction category is not inferred. A real far-field-enabled run is
needed to inspect far-field induction-only signals.

The adapter wraps `do_digitize_and_update`, copies only selected current rows,
calls the original electronics function once without changing RNG state, then
records its ADC values/times. It never modifies larnd-sim source. It captures
both normal and induction-only digitization calls if that driver calls the same
hook. It was developed against the local interface at commit `6106b83`.
Branches with another interface fail explicitly and need an adapter update.
An end-to-end CUDA capture has not been validated on this Mac.

Capture safeguards:

- Use a new trace directory per run; trace files are never overwritten.
- Exactly one event per digitization call and one module are supported; other
  cases fail instead of combining ambiguous time coordinates.
- `--max-channels`, `--max-batches` and `--max-samples` bound diagnostic transfers.
  The last is channel count times samples per call, default 2,000,000. Oversized
  captures fail rather than silently truncate. Normal simulation memory is not
  reduced by these settings.
- A manifest records the command, driver/fee hashes, git commit and captured files.
  A manifest after a failed run is not proof of successful simulation completion.
- Captures are per call: the same pixel in several calls is not automatically
  combined into one event-wide waveform.

In notebook 02 set `TRACE_PATH` to a produced NPZ. Optionally set `EDEP_INPUT` to
the matching original edep-HDF5 file to show its event's deposited segments. This
context is bounded, with a truncation flag, and is not per-pixel truth backtracking.

**What is actually recorded:** the signed current entering electronics, and the
ADC codes/local times returned by larnd-sim before packet serialization. **What
is not recorded:** per-tick CSA state, random noise draws, discriminator decisions,
or exact periodic-reset times. Thus the middle panel for captures is labeled an
unreset current integral and no simulated reset markers are fabricated. The
stored global constants do not replace per-pixel gain/threshold/pedestal LUTs.

Times are local microseconds before event offsets, clock quantization, and rollover.
The normal LARNDSIM output still contains actual serialized packets. Notebook 02
can plot those separately for an exact address (with an explicit data packet type:
1 for the local v3 setup, 0 for v2). It does not guess alignment between raw packet
timestamps and diagnostic times. Bad-channel masks or invalid routes can prevent
a diagnostic ADC from becoming a saved packet. Raw packet readings include
pedestal; reading is capped at 5,000 matches by default. Decreasing raw timestamps
are rejected rather than silently treated as elapsed time.

## Portable bundle format

Compressed NPZ, schema version 1, loaded with `allow_pickle=False`:

| Key | Meaning |
|---|---|
| `time_us` | Strictly increasing, uniformly spaced sample times, shape `(T,)` |
| `current_e_per_us` | Signed current, shape `(N,T)` |
| `pixel_id` | Simulation grid IDs, shape `(N,)` |
| `pixel_xyz_cm` | Physical pixel centers in edep XYZ, shape `(N,3)` |
| `adc_channel_index` | Row in captured channel table for each ADC |
| `adc_time_us`, `adc_value` | Local ADC times and raw digitized codes |
| `metadata_json` | JSON string, including kind, version, provenance and timing convention |
| `addresses` (capture) | Nominal IO group/channel, chip/channel, shape `(N,4)` |
| `segments_xyz_cm` (optional) | Context segments, shape `(S,2,3)` |
| `csa_e`, reset fields (demo only) | Explicit teaching-model internal history |

`save_bundle` refuses overwrites and validates dimensions. `signal_figure` accepts
this same schema for later simulator instrumentation that exports extra diagnostics.

## Checks

```bash
python -m unittest discover -s visualizations/tests -v
```

Tests check geometry orientation and channel/grid round trips, known FSD/FSD Cube
layout invariants when the sibling checkout exists, analytic electronics behavior,
HDF5 selection and integer timestamps, and capture transparency with a fake driver.
These are CPU checks, not a production GPU physics validation.
