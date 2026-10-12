"""Component MPV versus drift time with a joint whole-track bootstrap."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix

from .dqdx import fit_counts
from .io import code_hashes, file_identity, read_json, runtime_versions, safe_name, write_json

DRIFT_EDGES = np.linspace(0., 5000., 16)
DQDX_EDGES = np.linspace(0., 90., 101)
COLORS = ['#E69F00', '#009E73', '#CC79A7', '#0072B2']


def color_for(label, index):
    return dict(prc2=COLORS[0], prc8=COLORS[1], prc16=COLORS[2], MC=COLORS[3]).get(label, COLORS[index % len(COLORS)])


def read_tracks(path, kind):
    """Sparse per-track histograms; readout groups share one track weight."""
    keys, rows, columns = {}, [], []
    sample_id = 0 if kind == 'data' else 1
    with h5py.File(path, 'r') as f:
        if f.attrs.get('status') != 'complete':
            raise ValueError(f'Incomplete segments: {path}')
        parent = Path(path).parent.parent/'manifest.json'
        if parent.exists():
            stage = read_json(parent).get('stages', {}).get('segments')
            if stage and stage.get('status') != 'complete':
                raise ValueError(f'Stale/incomplete segmentation: {parent}')
        sample = f[f'samples/{sample_id}']
        if sample.attrs.get('sample_kind', kind) != kind:
            raise ValueError(f'Wrong sample kind: {path}')
        periods = []
        for source in sample['sources'].values():
            period = float(source['metadata/run_info'].attrs.get('crs_ticks', np.nan))
            if not np.isfinite(period) or period <= 0:
                raise ValueError(f'Missing/invalid CRS tick period: {path}')
            periods.append(period)
        if not periods or not np.allclose(periods, periods[0], rtol=1e-10, atol=0):
            raise ValueError(f'Inconsistent CRS tick periods: {path}')
        settings = json.loads(f.attrs['settings'])
        identity = str(sample.attrs['input_file'])
        total, valid = 0, 0
        for start in range(0, len(f['segments/data']), 100000):
            values = f['segments/data'][start:start+100000]
            for v in values[values['sample_id'] == sample_id]:
                key = (int(v['file_id']), int(v['source_event_row']), int(v['track_id']))
                row = keys.setdefault(key, len(keys))
                total += 1
                t, q = float(v['t_drift']), float(v['dqdx'])
                if not np.isfinite([t, q]).all() or not 0 <= t <= 5000 or not 0 <= q <= 90:
                    continue
                # Left-inclusive bins, including the last upper endpoint.
                j = min(np.searchsorted(DRIFT_EDGES, t, side='right')-1, 14)
                k = min(np.searchsorted(DQDX_EDGES, q, side='right')-1, 99)
                rows.append(row)
                columns.append(j*100+k)
                valid += 1
    matrix = coo_matrix((np.ones(len(rows), dtype=np.int64), (rows, columns)),
                        shape=(len(keys), 1500)).tocsr()
    audit = dict(selection_input=identity, segment_settings=settings, crs_ticks=periods[0],
                 tracks=len(keys), segments=total, histogrammed_segments=valid,
                 excluded_segments=total-valid,
                 bootstrap_population='all tracks represented in this sample segment table, including out-of-range tracks')
    return matrix, audit


def fit_mpv(counts):
    # The upstream implementation prints failures and can overflow in a zero tail.
    with redirect_stdout(io.StringIO()), np.errstate(over='ignore', under='ignore'):
        result, failure = fit_counts(counts, DQDX_EDGES, (20., 50.), 100)
    if result is None or not np.isfinite(result['params'][0]):
        return dict(mpv=None, fit_covariance_se=None, chi2_ndf=None,
                    status='no_fit', reason=failure or 'optimizer failed')
    return dict(mpv=float(result['params'][0]),
                fit_covariance_se=float(result['mpv_error']) if np.isfinite(result['mpv_error']) else None,
                chi2_ndf=float(result['chi2_red']), status='fit')


def resampled_histograms(matrix, rng):
    ntracks = matrix.shape[0]
    if not ntracks:
        return np.zeros((15, 100))
    weights = rng.multinomial(ntracks, np.full(ntracks, 1./ntracks))
    # One weight per whole track, used for every drift bin and readout group.
    return np.asarray(matrix.T @ weights).reshape(15, 100)


def errors(draws, eligible, minimum_fraction):
    successes = np.isfinite(draws).sum(axis=0)
    se = np.full(draws.shape[1], np.nan)
    for j in range(draws.shape[1]):
        if eligible[j] and successes[j] >= max(2, int(np.ceil(minimum_fraction*len(draws)))):
            se[j] = np.std(draws[np.isfinite(draws[:, j]), j], ddof=1)
    return se, successes


def checkpoint(path, draws, completed):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, mpv=draws, completed=completed)
    temporary.replace(path)


def bootstrap(matrix, repetitions, seed, destination, resume=False):
    draws = np.full((repetitions, 15), np.nan)
    completed = 0
    if resume and destination.exists():
        with np.load(destination) as old:
            draws = old['mpv']
            completed = int(old['completed'])
        if draws.shape != (repetitions, 15) or not 0 <= completed <= repetitions:
            raise ValueError(f'Invalid bootstrap checkpoint: {destination}')
    start = time.monotonic()
    for b in range(completed, repetitions):
        # Replica-indexed RNG makes restart exact and independent of checkpoint frequency.
        rng = np.random.default_rng(np.random.SeedSequence([seed, b]))
        histograms = resampled_histograms(matrix, rng)
        for j, counts in enumerate(histograms):
            result = fit_mpv(counts)
            if result['mpv'] is not None:
                draws[b, j] = result['mpv']
        if (b+1) % 10 == 0 or b+1 == repetitions:
            checkpoint(destination, draws, b+1)
            print(f'{destination.stem}: {b+1}/{repetitions} replicates; '
                  f'{time.monotonic()-start:.1f}s this invocation', flush=True)
    return draws


def plot(samples, ratios, destination):
    centers = (DRIFT_EDGES[:-1]+DRIFT_EDGES[1:])/2
    fig, (ax, ratio_ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
    try:
        for index, sample in enumerate(samples):
            color = color_for(sample['label'], index)
            values = np.array([b['mpv'] if b['mpv'] is not None else np.nan for b in sample['bins']])
            se = np.array([b['bootstrap_se'] if b['bootstrap_se'] is not None else np.nan for b in sample['bins']])
            good = np.isfinite(values) & np.isfinite(se)
            ax.errorbar(centers[good], values[good], yerr=se[good], fmt='o', color=color,
                        capsize=2, label=sample['label'])
            bad = np.isfinite(values) & ~np.isfinite(se)
            ax.plot(centers[bad], values[bad], 'o', color=color, fillstyle='none')
        for index, ratio in enumerate(ratios):
            color = color_for(ratio['label'], index)
            values = np.array([np.nan if x is None else x for x in ratio['mpv_ratio']])
            se = np.array([np.nan if x is None else x for x in ratio['bootstrap_se']])
            good = np.isfinite(values) & np.isfinite(se)
            ratio_ax.errorbar(centers[good], values[good], yerr=se[good], fmt='o',
                              color=color, capsize=2)
            bad = np.isfinite(values) & ~np.isfinite(se)
            ratio_ax.plot(centers[bad], values[bad], 'o', color=color, fillstyle='none')
        ax.set(title='FSD Cube component MPV versus drift time', ylabel='Component MPV [ke−/cm]')
        ax.legend()
        ratio_ax.axhline(1, color='0.4', linestyle='--', linewidth=1)
        ratio_ax.set(xlabel='Drift time [CRS ticks]', ylabel='Data / MC', xlim=(0, 5000))
        for axis in (ax, ratio_ax):
            axis.grid(alpha=.2)
        fig.text(.5, .015, 'Errors: whole-track bootstrap standard error. Open markers: uncertainty unavailable.\n'
                 'Fit: Moyal-like component convolved with Gaussian; no lifetime correction.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .07, 1, 1))
        for ext in ('png', 'pdf'):
            fig.savefig(destination/f'mpv_drift.{ext}', dpi=180)
    finally:
        plt.close(fig)


def finite_list(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def run(root, data, mc, run_id, repetitions=1000, seed=12345, minimum_fraction=.9, resume=False):
    if repetitions < 2 or seed < 0 or not 0 < minimum_fraction <= 1:
        raise ValueError('Require >=2 bootstrap replicates, nonnegative seed, and 0 < success fraction <= 1')
    labels = [label for label, _ in data]
    if not labels or len(set(labels)) != len(labels) or 'MC' in labels:
        raise ValueError('Data labels must be unique and cannot be MC')
    root = Path(root).resolve()
    sources = [(safe_name(label), root/Path(path), 'data') for label, path in data]
    sources.append(('MC', root/Path(mc), 'mc'))
    prepared = [(label, path, *read_tracks(path, kind)) for label, path, kind in sources]
    reference = prepared[0][3]
    for label, path, matrix, audit in prepared:
        if audit['segment_settings'] != reference['segment_settings']:
            raise ValueError(f'{label}: incompatible segmentation settings')
        if not np.isclose(audit['crs_ticks'], reference['crs_ticks'], rtol=1e-10, atol=0):
            raise ValueError(f'{label}: CRS tick period differs across samples')
    if len({item[3]['selection_input'] for item in prepared}) != len(prepared):
        raise ValueError('Duplicate source selections')
    signature = dict(inputs=[dict(label=label, **file_identity(path)) for label, path, _, _ in prepared],
                     repetitions=repetitions, seed=seed, minimum_success_fraction=minimum_fraction,
                     drift_edges_ticks=DRIFT_EDGES.tolist(), dqdx_edges=DQDX_EDGES.tolist(),
                     fit_range=[20, 50], min_fit_entries=100, runtime=runtime_versions(),
                     code=code_hashes(['analysis/mpv_drift.py', 'analysis/dqdx.py', 'pixel_dqdx/dqdx.py']))
    directory = root/'comparisons/reflow-v3-feb2026-prc2_vs_prc8_vs_prc16/analysis'/safe_name(run_id)
    if directory.exists():
        if not resume or read_json(directory/'manifest.json')['signature'] != signature:
            raise ValueError('Existing run: use --resume with unchanged inputs/settings/code or a new run ID')
    else:
        directory.mkdir(parents=True)
    lock = directory/'.writer.lock'
    with lock.open('x') as stream:
        stream.write('MPV drift writer\n')
    manifest = dict(status='running', signature=signature, created_utc=datetime.now(timezone.utc).isoformat(),
                    uncertainty='sample standard deviation of successful whole-track bootstrap MPVs; ddof=1',
                    ratio_uncertainty='standard deviation of paired data/MC bootstrap MPV ratios; one shared MC draw per replicate',
                    track_key=['sample', 'file_id', 'source_event_row', 'track_id'])
    write_json(directory/'manifest.json', manifest)
    output = directory/'mpv_drift'
    output.mkdir(exist_ok=True)
    try:
        samples, all_draws = [], []
        for label, path, matrix, audit in prepared:
            print(f'{label}: {matrix.shape[0]} tracks; fitting original histograms', flush=True)
            counts = np.asarray(matrix.sum(axis=0)).reshape(15, 100)
            ntracks = np.array([np.count_nonzero(np.asarray(matrix[:, j*100:(j+1)*100].sum(axis=1))) for j in range(15)])
            original = [fit_mpv(c) for c in counts]
            print(f'{label}: {sum(b["mpv"] is not None for b in original)}/15 original fits available', flush=True)
            # Separate stable streams; the MC label is constant across every ratio.
            stream_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], 'little') ^ seed
            draws = bootstrap(matrix, repetitions, stream_seed, output/f'{label}-bootstrap.npz', resume)
            eligible = (ntracks >= 2) & np.array([v['mpv'] is not None for v in original])
            se, success = errors(draws, eligible, minimum_fraction)
            print(f'{label}: {np.isfinite(se).sum()}/15 bootstrap standard errors available', flush=True)
            bins = [dict(original[j], low=float(DRIFT_EDGES[j]), high=float(DRIFT_EDGES[j+1]),
                         n_segments=int(counts[j].sum()), n_tracks=int(ntracks[j]),
                         bootstrap_se=finite_list(se)[j], successful_fits=int(success[j]),
                         success_fraction=float(success[j]/repetitions),
                         uncertainty_status='available' if np.isfinite(se[j]) else 'unavailable') for j in range(15)]
            samples.append(dict(label=label, audit=audit, bins=bins))
            all_draws.append(draws)
            write_json(output/f'{label}-summary.json', samples[-1])
        ratios = []
        for i, sample in enumerate(samples[:-1]):
            ratio_draws = np.full_like(all_draws[i], np.nan)
            valid = np.isfinite(all_draws[i]) & np.isfinite(all_draws[-1]) & (all_draws[-1] > 0)
            np.divide(all_draws[i], all_draws[-1], out=ratio_draws, where=valid)
            values = [b['mpv']/m['mpv'] if b['mpv'] is not None and m['mpv'] is not None and m['mpv'] > 0 else np.nan
                      for b, m in zip(sample['bins'], samples[-1]['bins'])]
            eligible = [b['bootstrap_se'] is not None and m['bootstrap_se'] is not None
                        for b, m in zip(sample['bins'], samples[-1]['bins'])]
            se, success = errors(ratio_draws, eligible, minimum_fraction)
            ratios.append(dict(label=sample['label'], mpv_ratio=finite_list(values), bootstrap_se=finite_list(se),
                               successful_pairs=success.tolist(), success_fraction=(success/repetitions).tolist()))
            # Preserve replica ordering for cross-bin and cross-ratio covariance analysis.
            checkpoint(output/f'{sample["label"]}-ratio-bootstrap.npz', ratio_draws, repetitions)
        plot(samples, ratios, output)
        for identity in signature['inputs']:
            if file_identity(identity['path']) != {k: v for k, v in identity.items() if k != 'label'}:
                raise ValueError('Input changed while running')
        manifest.update(status='complete', samples=samples, ratios=ratios)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(directory/'manifest.json', manifest)
        lock.unlink()
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--data', action='append', required=True, metavar='LABEL=SEGMENTS_HDF5')
    parser.add_argument('--mc', required=True, help='one shared MC segment table (may also contain data)')
    parser.add_argument('--run-id', default='mpv-drift-001')
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=12345)
    parser.add_argument('--min-success-fraction', type=float, default=.9)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    data = []
    for value in args.data:
        label, sep, path = value.partition('=')
        if not sep or not label or not path:
            parser.error('--data requires LABEL=SEGMENTS_HDF5')
        data.append((label, path))
    print('Complete:', run(args.root, data, args.mc, args.run_id, args.bootstrap_samples,
                          args.bootstrap_seed, args.min_success_fraction, args.resume))


if __name__ == '__main__':
    main()
