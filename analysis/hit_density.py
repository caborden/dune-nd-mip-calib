"""Mean recorded hits per selected segment versus median drift time.

The slide's hits/segment observable is the default. True hits/cm is an optional
mean of nhits/dx, using the shared segmentation's physical dx. No hit deduplication,
new angular cuts, timing conversion, or selection changes are introduced.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .io import code_hashes, file_identity, write_json

SETTING_KEYS = {'drift_bins', 'drift_range_ticks', 'drift_edges_ticks', 'observable',
                'uncertainty', 'bootstrap_samples', 'bootstrap_seed', 'min_segments'}


def make_settings(drift_bins=9, drift_range_ticks=(0., 5000.), drift_edges_ticks=None,
                  observable='hits-per-segment', uncertainty='track-bootstrap',
                  bootstrap_samples=1000, bootstrap_seed=12345, min_segments=1):
    if observable not in {'hits-per-segment', 'hits-per-cm'}:
        raise ValueError('observable must be hits-per-segment or hits-per-cm')
    if uncertainty not in {'track-bootstrap', 'segment-sem', 'none'}:
        raise ValueError('uncertainty must be track-bootstrap, segment-sem or none')
    for name, value, minimum in [('drift_bins', drift_bins, 1),
                                 ('bootstrap_samples', bootstrap_samples, 2),
                                 ('bootstrap_seed', bootstrap_seed, 0),
                                 ('min_segments', min_segments, 1)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if drift_edges_ticks is None:
        interval = np.asarray(drift_range_ticks, dtype=float)
        if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] >= interval[1]:
            raise ValueError('Invalid drift_range_ticks')
        edges = np.linspace(*interval, drift_bins+1)
    else:
        edges = np.asarray(drift_edges_ticks, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('Drift edges must be finite and strictly increasing')
    return dict(drift_bins=len(edges)-1, drift_range_ticks=[float(edges[0]), float(edges[-1])],
                drift_edges_ticks=edges.tolist(), observable=observable, uncertainty=uncertainty,
                bootstrap_samples=int(bootstrap_samples), bootstrap_seed=int(bootstrap_seed),
                min_segments=int(min_segments))


def number(value):
    return float(value) if np.isfinite(value) else None


def bootstrap_errors(track_counts, track_sums, repetitions, seed, sample_id):
    """Resample entire tracks; retain the segment-weighted mean in every bin.

    Tracks are clustered by file/event/track, combining readout groups. Resampling
    one cluster applies the same weight to all its bins. Replicates with no
    segments in a bin are omitted; fewer than two contributing tracks yields no
    error estimate. Separate sample streams keep data results stable in MC-only
    and paired configurations.
    """
    ntracks, nbins = track_counts.shape
    error = np.full(nbins, np.nan)
    valid_counts = np.zeros(nbins, dtype=np.int64)
    if ntracks < 2:
        return error, valid_counts
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_id]))
    estimates = np.full((repetitions, nbins), np.nan)
    probabilities = np.full(ntracks, 1./ntracks)
    for start in range(0, repetitions, 64):
        weights = rng.multinomial(ntracks, probabilities, size=min(64, repetitions-start))
        denominator = weights @ track_counts
        numerator = weights @ track_sums
        np.divide(numerator, denominator, out=estimates[start:start+len(weights)], where=denominator > 0)
    for index in range(nbins):
        values = estimates[:, index]
        values = values[np.isfinite(values)]
        valid_counts[index] = len(values)
        if np.count_nonzero(track_counts[:, index]) >= 2 and len(values) >= 2:
            error[index] = np.std(values, ddof=1)
    return error, valid_counts


def summarize(segment_file, settings, chunk_rows=100000):
    """Read bounded chunks; retain only per-track/per-bin sufficient statistics."""
    if chunk_rows < 1:
        raise ValueError('chunk_rows must be positive')
    edges = np.asarray(settings['drift_edges_ticks'])
    nbins = len(edges)-1
    samples = {}
    with h5py.File(segment_file, 'r') as f:
        if f.attrs.get('status') != 'complete':
            raise ValueError('Shared segments are incomplete')
        dataset = f['segments/data']
        required = {'sample_id', 'file_id', 'source_event_row', 'track_id', 't_drift', 'nhits', 'dx'}
        if not required <= set(dataset.dtype.names or []):
            raise ValueError('Shared segment table lacks required identity/count/time fields')
        segment_settings = json.loads(f.attrs.get('settings', '{}'))
        for identity in sorted(f['samples'], key=int):
            sample_id = int(identity)
            if sample_id not in {0, 1}:
                raise ValueError(f'Unsupported sample_id: {sample_id}')
            attrs = f[f'samples/{identity}'].attrs
            kind = 'data' if sample_id == 0 else 'mc'
            if attrs.get('sample_kind', kind) != kind:
                raise ValueError('Sample kind disagrees with its stable sample ID')
            samples[sample_id] = dict(sample_id=sample_id, sample_kind=kind,
                input=str(attrs.get('input_file', '')), tracks={},
                counts=np.zeros(nbins, dtype=np.int64), sums=np.zeros(nbins), squares=np.zeros(nbins),
                audit=dict(total_segments=0, included_segments=0, below_range=0, above_range=0,
                           invalid_segments=0))
        for start in range(0, len(dataset), chunk_rows):
            values = dataset[start:start+chunk_rows]
            if not set(np.unique(values['sample_id'])) <= set(samples):
                raise ValueError('Segment references a missing sample group')
            for sample_id, sample in samples.items():
                rows = values[values['sample_id'] == sample_id]
                audit = sample['audit']
                audit['total_segments'] += len(rows)
                time = rows['t_drift'].astype(float)
                count = rows['nhits'].astype(float)
                valid = np.isfinite(time) & np.isfinite(count) & (count >= 0)
                if settings['observable'] == 'hits-per-cm':
                    dx = rows['dx'].astype(float)
                    valid &= np.isfinite(dx) & (dx > 0)
                    count = np.divide(count, dx, out=np.full(len(count), np.nan), where=valid)
                    valid &= np.isfinite(count)
                audit['invalid_segments'] += int((~valid).sum())
                audit['below_range'] += int((valid & (time < edges[0])).sum())
                audit['above_range'] += int((valid & (time > edges[-1])).sum())
                keep = valid & (time >= edges[0]) & (time <= edges[-1])
                rows, time, count = rows[keep], time[keep], count[keep]
                audit['included_segments'] += len(rows)
                bins = np.searchsorted(edges, time, side='right')-1
                bins[time == edges[-1]] = nbins-1  # final bin includes the final edge
                sample['counts'] += np.bincount(bins, minlength=nbins)
                sample['sums'] += np.bincount(bins, weights=count, minlength=nbins)
                sample['squares'] += np.bincount(bins, weights=count*count, minlength=nbins)
                for row, index, value in zip(rows, bins, count):
                    key = (int(row['file_id']), int(row['source_event_row']), int(row['track_id']))
                    if key not in sample['tracks']:
                        sample['tracks'][key] = (np.zeros(nbins, dtype=np.int64), np.zeros(nbins))
                    counts, sums = sample['tracks'][key]
                    counts[index] += 1
                    sums[index] += value
    results = []
    for sample_id, sample in samples.items():
        counts = sample['counts']
        mean = np.divide(sample['sums'], counts, out=np.full(nbins, np.nan), where=counts > 0)
        variance = np.divide(sample['squares']-sample['sums']*np.nan_to_num(mean), counts-1,
                             out=np.full(nbins, np.nan), where=counts > 1)
        std = np.sqrt(np.maximum(variance, 0))
        sem = np.divide(std, np.sqrt(counts), out=np.full(nbins, np.nan), where=counts > 1)
        ordered_tracks = [sample['tracks'][key] for key in sorted(sample['tracks'])]
        track_counts = np.asarray([t[0] for t in ordered_tracks], dtype=np.int64).reshape(-1, nbins)
        track_sums = np.asarray([t[1] for t in ordered_tracks], dtype=float).reshape(-1, nbins)
        ntracks = np.count_nonzero(track_counts, axis=0)
        repetitions = np.zeros(nbins, dtype=np.int64)
        if settings['uncertainty'] == 'track-bootstrap':
            errors, repetitions = bootstrap_errors(track_counts, track_sums,
                settings['bootstrap_samples'], settings['bootstrap_seed'], sample_id)
        elif settings['uncertainty'] == 'segment-sem':
            errors = sem.copy()
        else:
            errors = np.full(nbins, np.nan)
        eligible = counts >= settings['min_segments']
        bins = []
        for index in range(nbins):
            bins.append(dict(bin_low=float(edges[index]), bin_high=float(edges[index+1]),
                bin_center=float((edges[index]+edges[index+1])/2), n_segments=int(counts[index]),
                n_tracks=int(ntracks[index]), mean=number(mean[index]) if eligible[index] else None,
                std_dev=number(std[index]), segment_sem=number(sem[index]),
                error=number(errors[index]) if eligible[index] else None,
                valid_bootstrap_replicates=int(repetitions[index])))
        results.append(dict(sample_id=sample_id, sample_kind=sample['sample_kind'], input=sample['input'],
                            n_tracks=len(ordered_tracks), audit=sample['audit'], bins=bins))
    return results, segment_settings


def ratios(samples):
    by_kind = {s['sample_kind']: s for s in samples}
    if set(by_kind) != {'data', 'mc'}:
        return []
    result = []
    for data, mc in zip(by_kind['data']['bins'], by_kind['mc']['bins']):
        ratio, error = None, None
        if data['mean'] is not None and mc['mean'] is not None and mc['mean'] > 0:
            ratio = data['mean']/mc['mean']
            if data['error'] is not None and mc['error'] is not None:
                error = float(np.hypot(data['error']/mc['mean'],
                                       data['mean']*mc['error']/mc['mean']**2))
        result.append(dict(bin_low=data['bin_low'], bin_high=data['bin_high'],
                           bin_center=data['bin_center'], ratio=ratio, error=error))
    return result


def draw_series(ax, bins, value_key, label, color, error_key='error'):
    x = np.array([b['bin_center'] for b in bins])
    y = np.array([np.nan if b[value_key] is None else b[value_key] for b in bins])
    ax.plot(x, y, 'o-', label=label, color=color, linewidth=1.5, markersize=4)
    errors = np.array([np.nan if b[error_key] is None else b[error_key] for b in bins])
    valid = np.isfinite(y) & np.isfinite(errors)
    ax.errorbar(x[valid], y[valid], yerr=errors[valid], fmt='none', color=color, capsize=3)


def save_plot(directory, stem, samples, settings, segment_settings, ratio_bins=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if ratio_bins is None:
        for sample in samples:
            kind = sample['sample_kind']
            draw_series(ax, sample['bins'], 'mean', 'Data' if kind == 'data' else 'MC',
                        'tab:green' if kind == 'data' else 'tab:orange')
        length = segment_settings.get('segment_length_cm')
        label = (f'Mean N hits / {length:g} cm segment' if length is not None else 'Mean N hits / segment')
        if settings['observable'] == 'hits-per-cm':
            label = 'Mean dN/dx [hits/cm]'
        ax.set(ylabel=label, title='Hit density versus drift time')
    else:
        draw_series(ax, ratio_bins, 'ratio', 'Data / MC', 'tab:orange')
        ax.axhline(1., linestyle='--', color='0.5', linewidth=1.)
        ax.set(ylabel='Ratio Data / MC', title='Hit density ratio versus drift time')
    ax.set(xlabel='Drift time [ticks]', xlim=settings['drift_range_ticks'])
    points = ratio_bins if ratio_bins is not None else [b for s in samples for b in s['bins']]
    key = 'ratio' if ratio_bins is not None else 'mean'
    if not any(b[key] is not None for b in points):
        ax.text(.5, .5, 'No populated bins in requested drift range',
                transform=ax.transAxes, ha='center', va='center')
    ax.grid(alpha=.35)
    ax.legend()
    methods = {'track-bootstrap': 'track bootstrap errors', 'segment-sem': 'segment SEM errors',
               'none': 'error bars disabled'}
    fig.text(.5, .015, f'Selected recorded hits; {methods[settings["uncertainty"]]}', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    try:
        for extension in ['png', 'pdf']:
            fig.savefig(directory/f'{stem}.{extension}', dpi=180)
    finally:
        plt.close(fig)


def plot_segments(segment_file, output_dir, settings=None):
    """Runner module; only the completed shared segment table is read."""
    settings = make_settings(**(settings or {}))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='incomplete', settings=settings, segment_file=file_identity(segment_file),
                    generated_utc=datetime.now(timezone.utc).isoformat(),
                    source_hashes=code_hashes(['analysis/hit_density.py', 'analysis/io.py']),
                    definitions=dict(observable='segment-weighted arithmetic mean',
                        drift_time='median recorded hit t_drift of each segment, unchanged ticks',
                        hit_view='selected recorded hits; repeated hits are not deduplicated',
                        bin_edges='left inclusive/right exclusive; final edge included',
                        bootstrap_cluster='(sample_id, file_id, source_event_row, track_id)',
                        bootstrap_population='tracks with at least one valid in-range segment',
                        error={'track-bootstrap': 'standard deviation of track-bootstrap means; not a confidence interval',
                               'segment-sem': 'sample standard deviation / sqrt(n_segments); treats segments as independent',
                               'none': 'error bars disabled'}[settings['uncertainty']],
                        ratio_error='first-order propagation assuming independent data and MC samples',
                        selection='shared segmentation cuts, including >=2 hits and positive dq/dx denominators/charge'))
    try:
        samples, segment_settings = summarize(segment_file, settings)
        manifest.update(samples=samples, segment_settings=segment_settings)
        for sample in samples:
            kind = sample['sample_kind']
            keys = ['bin_low', 'bin_high', 'bin_center', 'n_segments', 'n_tracks', 'mean',
                    'std_dev', 'segment_sem', 'error', 'valid_bootstrap_replicates']
            rows = [[np.nan if b[key] is None else b[key] for key in keys] for b in sample['bins']]
            np.savetxt(directory/f'{kind}_hit_density.csv', rows, delimiter=',',
                       header=','.join(keys), comments='')
            save_plot(directory, f'{kind}_hit_density', [sample], settings, segment_settings)
        ratio_bins = ratios(samples)
        if ratio_bins:
            save_plot(directory, 'data_mc_hit_density', samples, settings, segment_settings)
            save_plot(directory, 'data_mc_hit_density_ratio', samples, settings, segment_settings, ratio_bins)
            keys = ['bin_low', 'bin_high', 'bin_center', 'ratio', 'error']
            rows = [[np.nan if b[key] is None else b[key] for key in keys] for b in ratio_bins]
            np.savetxt(directory/'data_mc_hit_density_ratio.csv', rows, delimiter=',',
                       header=','.join(keys), comments='')
        manifest.update(status='complete', ratio=ratio_bins)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        write_json(directory/'analysis.json', manifest)
        raise
    write_json(directory/'analysis.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments', required=True, help='completed shared segments.hdf5')
    parser.add_argument('--output-dir', required=True, help='new module output directory')
    parser.add_argument('--drift-bins', type=int, default=9)
    parser.add_argument('--drift-range-ticks', type=float, nargs=2, default=(0., 5000.))
    parser.add_argument('--drift-edges-ticks', type=float, nargs='+')
    parser.add_argument('--observable', choices=['hits-per-segment', 'hits-per-cm'], default='hits-per-segment')
    parser.add_argument('--uncertainty', choices=['track-bootstrap', 'segment-sem', 'none'], default='track-bootstrap')
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=12345)
    parser.add_argument('--min-segments', type=int, default=1)
    args = vars(parser.parse_args())
    segment_file = args.pop('segments')
    output_dir = Path(args.pop('output_dir')).resolve()
    settings = make_settings(**args)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = plot_segments(segment_file, output_dir, settings)
    for sample in result['samples']:
        print(sample['sample_kind'], sample['audit'])
    print(f'Complete: {output_dir}')


if __name__ == '__main__':
    main()
