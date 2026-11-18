# dQ/dx drift-time mosaic

`python -m analysis.dqdx_drift_mosaic` reads completed segment tables produced by
`analyze_track_dqdx.py` or `analysis.run`. It overlays data and MC in a 2-row,
5-column figure, ordered from 0–50 to 450–500 microseconds. It does not resegment
tracks or add a lifetime correction. No distribution fits are performed.

```bash
python -m analysis.dqdx_drift_mosaic \
  --data-segments /path/to/prc8/segments.hdf5 \
  --mc-segments /path/to/prc8/segments.hdf5 \
  --label 'PRC8 data / PRC8 MC (ADC busy 8)' \
  --output-dir /path/to/new/prc8-drift-mosaic --write-png --write-csv
```

The same paired file can supply both samples, or separate files may be used.
Segment settings must match. Each source's `metadata/run_info.crs_ticks` (in
microseconds per tick) converts the saved median hit drift time into microseconds.
Missing, nonpositive, or nonfinite clock periods are rejected, as are incomplete
segment tables. Different source clock periods are supported.

Time bins are left-inclusive and right-exclusive, with 500 microseconds included
in the final bin. The 100 dQ/dx bins cover 0–90 ke−/cm, including 90. Each sample
is normalized independently to 100% across its displayed dQ/dx bins in each
panel. This is a shape comparison, not an absolute-rate comparison. N denotes
displayed segments; track segments are correlated and no statistical errors are
claimed. Empty panels remain visible and labeled. All panels share axis scales.

Outputs are `dqdx_drift_mosaic.pdf`, optional PNG and histogram CSV, and
`analysis.json` with input identities, code hashes, segmentation settings, clock
periods, raw counts and exclusions. Nonfinite entries are counted first; drift
under/overflow counts apply to finite entries; charge under/overflow counts apply
only to finite entries inside the time range. These exclusions and histogrammed
entries partition the total. Output directories must be new.

The provenance in the segment table identifies its original track inputs. Check
those identities when reusing a table, especially for similarly named campaigns.
The 450–500 microsecond bin is diagnostic and need not represent physical drift
inside the nominal detector volume.

## Data-only fitted mosaic

Use `--data-only-fit` instead of `--mc-segments` to select only the data from a
paired (or data-only) segment table:

```bash
python -m analysis.dqdx_drift_mosaic \
  --data-segments /path/to/prc8/segments.hdf5 --data-only-fit \
  --label 'PRC8 data' --output-dir /path/to/new/data-fit-mosaic \
  --write-png --write-csv
```

This mode fits the raw data counts with the existing Moyal–Gaussian model over
20–50 ke−/cm (`--fit-range`), requiring at least 100 entries in that range
(`--min-fit-entries`) and five populated fit bins. The curve is scaled by the
same normalization as the histogram for display. Every panel's legend reports
the component MPV, full fitted convolution FWHM, and FWHM/component-MPV ratio.
MPV and FWHM have units ke−/cm; their ratio is dimensionless. The component MPV
is not the convolved curve's maximum. Widths are measured on the full fitted
curve, not truncated at the fit-range boundaries. Failed/low-statistics fits or
unavailable widths are explicitly labeled. Fit parameters, covariance, chi-square,
and width diagnostics are saved per drift bin in `analysis.json`. Covariance
errors do not account for segment correlations. Data-only CSVs contain no MC
column; the paired overlay mode remains available.
