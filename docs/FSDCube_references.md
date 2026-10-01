# FSD-cube analysis references

Requested project reference: [2026 light tutorial](https://github.com/DUNE/ndlar_flow/blob/v1.6.0_Tutorial/docs/Light_Tutorial_2026/ND_Prototypes_Wkshp_0926_LRS_Tutorial.ipynb).
The inspected `v1.6.0_Tutorial` branch resolved to commit
`e1406a5ea544fac333e8817cfcceba18ef5c5785`.
This is a reference version, not a verified production commit for Reflow_FSDCube_v3.

## Source definitions

- [Calibrated prompt hits](https://github.com/DUNE/ndlar_flow/blob/e1406a5ea544fac333e8817cfcceba18ef5c5785/src/proto_nd_flow/reco/charge/calib_prompt_hits.py): Q_raw is charge in ke- before optional ADC-droop correction; Q is equal to Q_raw when that correction is disabled. Energy E additionally uses recombination and optionally electron-lifetime correction. t_drift is in charge-clock ticks relative to event t0, with PPS rollover adjustments. Calibrated x is reconstructed from drift time; y and z are pixel coordinates in cm.
- [Run data](https://github.com/DUNE/ndlar_flow/blob/e1406a5ea544fac333e8817cfcceba18ef5c5785/src/proto_nd_flow/resources/run_data.py): crs_ticks is the charge-clock period in microseconds; rollover_ticks is the nominal count between SYNC rollovers.
- [Charge timestamp correction](https://github.com/DUNE/ndlar_flow/blob/e1406a5ea544fac333e8817cfcceba18ef5c5785/src/proto_nd_flow/reco/charge/timestamp_corrector.py): corrected packet timestamps still involve modulo rollover; they are not automatically absolute chronological times.
- [Geometry](https://github.com/DUNE/ndlar_flow/blob/e1406a5ea544fac333e8817cfcceba18ef5c5785/src/proto_nd_flow/resources/geometry.py): network-agnostic geometry replicates pixel mappings across IO-channel blocks (four for this configuration). Validate a canonical pixel key against stored geometry, retaining the original electronics address.
- [FSD-cube calibration configuration](https://github.com/DUNE/ndlar_flow/blob/e1406a5ea544fac333e8817cfcceba18ef5c5785/yamls/fsdcube_flow/reco/charge/CalibHitBuilderData.yaml): compare to actual production settings rather than silently adopting these defaults.

## Applying the tutorial

Use its event-to-hit reference traversal pattern with charge/calib_prompt_hits for the inspected FSD-cube file. ref_region start/stop delimit rows of the reference table (stop exclusive), not a direct slice of the target hit dataset. Filter reference rows by source row index, then read target row indices. Detect duplicate references before deduplicating.

The tutorial uses 2x2 simulation and light data; its 16 ns waveform ticks, light geometry, and simulation-specific one-to-one charge/light matching must not be transferred to this charge-only FSD-cube sample.

Saved outputs in FSDcubeData.ipynb show adc_droop_calibration=False, crs_ticks=0.1, rollover_ticks=10000000, network_agnostic=True, and n_io_channels_per_tile=4. Validate source expectations against the actual data and establish production provenance before final interpretation.

## Additional tutorial repository

The user confirmed [jvmead/lrs_tutorials](https://github.com/jvmead/lrs_tutorials) as the second reference. Its README and notebooks 1 and 6 were inspected on 2026-09-30. Notebook 1 covers raw/processed light waveforms and imports `LUT` from `proto_nd_flow.util.lut`. Notebook 6 demonstrates reading reconstructed light hits through references and making DataFrames. Use these as access/inspection examples; do not transfer light ADC, photoelectron, or sample-period definitions to charge hits. Preserve validity masks when adapting dereference examples.

## Updated local notebook observations (2026-09-30)

The saved event-7 output contains 130 referenced hits, matching nhit, zero hits flagged disabled, and Q equal to Q_raw. Ten candidate pixel groups contain two hits each; every displayed group has zero y/z span and only one IO channel. Thus this event has 120 candidate pixels, but does not test cross-IO-channel aliases. These are repeated-hit candidates, not yet validated muon charge splitting.

The displayed coordinate ranges give a bounding-box diagonal of approximately 41.5 cm, below the selector's current 55 cm length requirement. Event 7 therefore cannot pass that length cut. Its earliest drift time is zero and it has no external triggers; inspect the associated combined/t0 record before interpreting reconstructed x as absolute drift position.
