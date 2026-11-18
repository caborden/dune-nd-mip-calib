"""Overlay data/MC segment dQ/dx in ten 50-microsecond drift bins."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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


def run(data_segments, mc_segments, output_dir, label='Data / MC', write_png=False, write_csv=False):
    data, da = histograms(data_segments, 'data')
    mc, ma = histograms(mc_segments, 'mc')
    if da['segment_settings'] != ma['segment_settings']:
        raise ValueError('Data and MC segment settings differ')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey=True)
    peak = max((100 * row.max() / row.sum() for counts in (data, mc)
                for row in counts if row.sum()), default=1.)
    for i, ax in enumerate(axes.flat):
        for counts, name, color in ((data, 'Data', 'steelblue'), (mc, 'MC', 'darkorange')):
            n = int(counts[i].sum())
            if n:
                ax.stairs(100 * counts[i] / n, DQDX_EDGES, color=color,
                          linewidth=1.5, label=f'{name}: N={n:,}')
            else:
                ax.plot([], [], color=color, label=f'{name}: N=0 (empty)')
        upper = '≤' if i == 9 else '<'
        ax.set_title(f'{int(DRIFT_EDGES[i])} ≤ t {upper} {int(DRIFT_EDGES[i+1])} μs')
        ax.set_xlim(0, 90)
        ax.set_ylim(0, peak * 1.15)
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle(f'{label}: dQ/dx by drift time', fontsize=17)
    fig.supxlabel('Segment dQ/dx [ke⁻/cm]', y=.055)
    fig.supylabel('Displayed segments per bin [%]')
    fig.text(.5, .015, 'Each sample normalized separately within each panel; N = segments in 0–90 ke⁻/cm. '
             'Median hit drift time; no added lifetime correction.', ha='center', fontsize=9)
    fig.tight_layout(rect=(.015, .08, 1, .95))
    for ext in figure_extensions(write_png):
        fig.savefig(out / f'dqdx_drift_mosaic.{ext}', dpi=180)
    plt.close(fig)
    if write_csv:
        rows = [(DRIFT_EDGES[i], DRIFT_EDGES[i+1], DQDX_EDGES[j], DQDX_EDGES[j+1], data[i,j], mc[i,j])
                for i in range(10) for j in range(100)]
        np.savetxt(out / 'histograms.csv', rows, delimiter=',',
                   header='drift_low_us,drift_high_us,dqdx_low,dqdx_high,data_count,mc_count', comments='')
    result = dict(status='complete', label=label, drift_edges_us=DRIFT_EDGES.tolist(),
                  dqdx_edges=DQDX_EDGES.tolist(), normalization='percent of displayed segments per sample per drift bin',
                  bin_policy='left inclusive, right exclusive; final upper endpoints included',
                  inputs=dict(data=file_identity(data_segments), mc=file_identity(mc_segments)),
                  audits=dict(data=da, mc=ma), counts=dict(data=data.tolist(), mc=mc.tolist()),
                  source_hashes=code_hashes(['analysis/dqdx_drift_mosaic.py', 'analysis/exports.py', 'analysis/io.py']))
    write_json(out / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-segments', required=True)
    parser.add_argument('--mc-segments', required=True, help='May be the same paired segment file as --data-segments')
    parser.add_argument('--output-dir', required=True, help='New output directory')
    parser.add_argument('--label', default='Data / MC')
    add_export_arguments(parser)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result['audits'], indent=2))


if __name__ == '__main__':
    main()
