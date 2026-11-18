"""Overlay data/MC segment dQ/dx in ten 50-microsecond drift bins."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .dqdx import fit_counts
from .compare_dqdx import fitted_widths
from .exports import add_export_arguments, figure_extensions
from .io import code_hashes, file_identity, write_json

DRIFT_EDGES = np.arange(0., 501., 50.)
DQDX_EDGES = np.linspace(0., 90., 101)


def histograms(path, kind):
    """Convert each source's median segment time from CRS ticks to microseconds."""
    counts = np.zeros((10, 100), dtype=np.int64)
    audit = dict(total=0, nonfinite=0, drift_underflow=0, drift_overflow=0,
                 dqdx_underflow=0, dqdx_overflow=0)
    with h5py.File(path, 'r') as f:
        if f.attrs.get('status') != 'complete':
            raise ValueError(f'Incomplete segments: {path}')
        ids = [int(i) for i, sample in f['samples'].items()
               if sample.attrs.get('sample_kind') == kind]
        if len(ids) != 1:
            raise ValueError(f'Expected exactly one {kind} sample: {path}')
        sample = f[f'samples/{ids[0]}']
        periods = {}
        for key, source in sample['sources'].items():
            period = float(source['metadata/run_info'].attrs.get('crs_ticks', np.nan))
            if not np.isfinite(period) or period <= 0:
                raise ValueError(f'Missing/invalid CRS tick period for source {key}: {path}')
            periods[int(key)] = period
        audit.update(clock_period_us_by_source=periods,
                     selection_input=str(sample.attrs['input_file']),
                     segment_settings=json.loads(f.attrs['settings']))
        table = f['segments/data']
        for start in range(0, len(table), 100000):
            rows = table[start:start + 100000]
            rows = rows[rows['sample_id'] == ids[0]]
            for source_id in np.unique(rows['file_id']):
                if int(source_id) not in periods:
                    raise ValueError(f'Missing source metadata: {source_id}')
                part = rows[rows['file_id'] == source_id]
                t = part['t_drift'] * periods[int(source_id)]
                q = part['dqdx']
                finite = np.isfinite(t) & np.isfinite(q)
                audit['total'] += len(part)
                audit['nonfinite'] += int((~finite).sum())
                t, q = t[finite], q[finite]
                audit['drift_underflow'] += int((t < 0).sum())
                audit['drift_overflow'] += int((t > 500).sum())
                inside = (t >= 0) & (t <= 500)
                audit['dqdx_underflow'] += int((inside & (q < 0)).sum())
                audit['dqdx_overflow'] += int((inside & (q > 90)).sum())
                counts += np.histogram2d(t, q, bins=(DRIFT_EDGES, DQDX_EDGES))[0].astype(np.int64)
    audit['displayed_by_drift_bin'] = counts.sum(axis=1).tolist()
    audit['histogrammed_segments'] = int(counts.sum())
    return counts, audit


def fit_panel(counts, fit_range=(20., 50.), min_fit_entries=100):
    fit, reason = fit_counts(counts, DQDX_EDGES, fit_range, min_fit_entries)
    if fit is None:
        return None, dict(status='no_fit', reason=reason or 'optimizer failed')
    summary = dict(status='fit', params=fit['params'].tolist(),
                   covariance=fit['cov'].tolist(), mpv=float(fit['params'][0]),
                   mpv_error=float(fit['mpv_error']), chi2_red=float(fit['chi2_red']))
    summary['widths'] = fitted_widths(summary)
    return fit, summary


def fit_label(summary):
    if summary['status'] != 'fit':
        return 'Fit unavailable'
    lines = ['Moyal–Gaussian fit', f"Component MPV = {summary['mpv']:.2f}"]
    widths = summary['widths']
    if widths['status'] == 'complete':
        lines.append(f"FWHM = {widths['fwhm']:.2f}")
        ratio = widths['fwhm_over_component_mpv']
        lines.append(f'FWHM/MPV = {ratio:.3f}' if ratio is not None else 'FWHM/MPV unavailable')
    else:
        lines.append('FWHM / ratio unavailable')
    return '\n'.join(lines)


def run(data_segments, mc_segments, output_dir, label='Data / MC', write_png=False, write_csv=False,
        data_only_fit=False, fit_range=(20., 50.), min_fit_entries=100):
    if not (0 <= fit_range[0] < fit_range[1] <= 90) or min_fit_entries < 1:
        raise ValueError('Invalid fit range or minimum entries')
    if data_only_fit == (mc_segments is not None):
        raise ValueError('Choose either --mc-segments or --data-only-fit')
    data, da = histograms(data_segments, 'data')
    members = [(data, 'Data', 'steelblue')]
    inputs, audits, saved_counts = dict(data=file_identity(data_segments)), dict(data=da), dict(data=data.tolist())
    if not data_only_fit:
        mc, ma = histograms(mc_segments, 'mc')
        if da['segment_settings'] != ma['segment_settings']:
            raise ValueError('Data and MC segment settings differ')
        members.append((mc, 'MC', 'darkorange'))
        inputs['mc'], audits['mc'], saved_counts['mc'] = file_identity(mc_segments), ma, mc.tolist()
    fits = [fit_panel(row, fit_range, min_fit_entries) for row in data] if data_only_fit else []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey=True)
    peak = max((100 * row.max() / row.sum() for counts, _, _ in members
                for row in counts if row.sum()), default=1.)
    if fits:
        peak = max(peak, max((100 * fit['fit_vals'].max() / data[i].sum()
                             for i, (fit, _) in enumerate(fits) if fit is not None), default=0.))
    for i, ax in enumerate(axes.flat):
        for counts, name, color in members:
            n = int(counts[i].sum())
            if n:
                ax.stairs(100 * counts[i] / n, DQDX_EDGES, color=color,
                          linewidth=1.5, label=f'{name}: N={n:,}')
            else:
                ax.plot([], [], color=color, label=f'{name}: N=0 (empty)')
        if data_only_fit:
            fit, summary = fits[i]
            if fit is not None:
                ax.plot(fit['x_fit'], 100 * fit['fit_vals'] / data[i].sum(),
                        color='red', linewidth=1.5, label=fit_label(summary))
            else:
                ax.plot([], [], color='red', label=fit_label(summary))
        upper = '≤' if i == 9 else '<'
        ax.set_title(f'{int(DRIFT_EDGES[i])} ≤ t {upper} {int(DRIFT_EDGES[i+1])} μs')
        ax.set_xlim(0, 90)
        ax.set_ylim(0, peak * (1.65 if data_only_fit else 1.15))
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle(f'{label}: dQ/dx by drift time', fontsize=17)
    fig.supxlabel('Segment dQ/dx [ke⁻/cm]', y=.055)
    fig.supylabel('Displayed segments per bin [%]')
    fig.text(.5, .015, 'Each sample normalized separately within each panel; N = segments in 0–90 ke⁻/cm. '
             'Median hit drift time; no added lifetime correction.', ha='center', fontsize=9)
    if data_only_fit:
        fig.text(.5, .04, f'Fit range {fit_range[0]:g}–{fit_range[1]:g} ke⁻/cm; MPV is the component parameter; '
                 'FWHM is the full fitted convolution width [ke⁻/cm].', ha='center', fontsize=9)
    fig.tight_layout(rect=(.015, .08, 1, .95))
    for ext in figure_extensions(write_png):
        fig.savefig(out / f'dqdx_drift_mosaic.{ext}', dpi=180)
    plt.close(fig)
    if write_csv:
        rows = [(DRIFT_EDGES[i], DRIFT_EDGES[i+1], DQDX_EDGES[j], DQDX_EDGES[j+1], *[counts[i,j] for counts, _, _ in members])
                for i in range(10) for j in range(100)]
        np.savetxt(out / 'histograms.csv', rows, delimiter=',',
                   header='drift_low_us,drift_high_us,dqdx_low,dqdx_high,' + ','.join(name.lower() + '_count' for _, name, _ in members), comments='')
    result = dict(status='complete', label=label, drift_edges_us=DRIFT_EDGES.tolist(),
                  dqdx_edges=DQDX_EDGES.tolist(), normalization='percent of displayed segments per sample per drift bin',
                  bin_policy='left inclusive, right exclusive; final upper endpoints included',
                  inputs=inputs, audits=audits, counts=saved_counts,
                  fit_settings=dict(enabled=data_only_fit, range=list(fit_range), min_entries=min_fit_entries),
                  fits=[dict(summary, drift_low_us=float(DRIFT_EDGES[i]), drift_high_us=float(DRIFT_EDGES[i+1]))
                        for i, (_, summary) in enumerate(fits)],
                  source_hashes=code_hashes(['analysis/dqdx_drift_mosaic.py', 'analysis/exports.py', 'analysis/io.py',
                                             'analysis/dqdx.py', 'analysis/compare_dqdx.py', 'pixel_dqdx/dqdx.py']))
    write_json(out / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-segments', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--data-only-fit', action='store_true', help='Plot only data with a Moyal–Gaussian fit in each panel')
    mode.add_argument('--mc-segments', help='May be the same paired segment file as --data-segments')
    parser.add_argument('--output-dir', required=True, help='New output directory')
    parser.add_argument('--label', default=None)
    parser.add_argument('--fit-range', type=float, nargs=2, default=(20., 50.))
    parser.add_argument('--min-fit-entries', type=int, default=100)
    add_export_arguments(parser)
    args = vars(parser.parse_args())
    if args['label'] is None:
        args['label'] = 'Data' if args['data_only_fit'] else 'Data / MC'
    result = run(**args)
    print(json.dumps(result['audits'], indent=2))


if __name__ == '__main__':
    main()
