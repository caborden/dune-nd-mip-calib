"""Time separation of exactly-two-hit pixels in saved full events."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .io import code_hashes, file_identity, read_json, safe_name, write_json
from .run import resolve_target

ADDRESS = ['io_group', 'io_channel', 'chip_id', 'channel_id']


def pixel_deltas(hits, time_field):
    """One nonnegative separation per pixel with exactly two records in this event.

    Group BEFORE testing timestamps: invalid times must not turn a three-hit
    pixel into a two-hit pixel. Convert integer scalars to Python int before
    subtraction to avoid float rounding and unsigned overflow.
    """
    audit = Counter(hits=len(hits))
    if not len(hits):
        return np.empty(0), audit
    _, inverse, counts = np.unique(hits[ADDRESS], return_inverse=True, return_counts=True)
    audit.update(pixels=len(counts), pixels_one_hit=int((counts == 1).sum()),
                 pixels_two_hits=int((counts == 2).sum()), pixels_more_than_two_hits=int((counts > 2).sum()))
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts[:-1])]
    integer = np.issubdtype(hits.dtype[time_field], np.integer)
    deltas = []
    for start in starts[counts == 2]:
        pair = hits[order[start:start+2]]
        a, b = pair[time_field]
        if not np.isfinite(a) or not np.isfinite(b):
            audit['pairs_nonfinite_time'] += 1
            continue
        delta = abs(int(b)-int(a)) if integer else abs(float(b)-float(a))
        if not np.isfinite(delta):
            audit['pairs_nonfinite_delta'] += 1
            continue
        deltas.append(delta)
        audit['pairs_valid'] += 1
        audit['pairs_zero_delta'] += int(delta == 0)
        audit['pairs_both_selected'] += int(np.all(pair['selected_track_id'] >= 0))
    return np.asarray(deltas, dtype=float), audit


def summarize(path, edges, time_field='ts_pps'):
    counts = np.zeros(len(edges)-1, dtype=np.int64)
    audit = Counter()
    with h5py.File(path, 'r') as f:
        if f.attrs.get('schema_version') != '2.0' or f.attrs.get('status') != 'complete':
            raise ValueError(f'Requires a complete schema-2 selection: {path}')
        dataset = f['events/hits']
        required = set(ADDRESS + [time_field, 'file_id', 'source_event_row', 'selected_track_id'])
        if not required <= set(dataset.dtype.names):
            raise ValueError(f'{path}: missing hit fields {sorted(required-set(dataset.dtype.names))}')
        if dataset.dtype[time_field].kind not in 'iuf':
            raise ValueError(f'Unsupported timestamp dtype: {dataset.dtype[time_field]}')
        periods = {}
        for file_id, source in f['sources'].items():
            group = source.get('metadata/run_info')
            period = np.nan if group is None else float(group.attrs.get('crs_ticks', np.nan))
            if not np.isfinite(period) or period <= 0:
                raise ValueError(f'{path}: missing CRS tick period for source {file_id}')
            periods[file_id] = period
        if not periods or not np.allclose(list(periods.values()), next(iter(periods.values())), rtol=1e-10, atol=0):
            raise ValueError(f'{path}: inconsistent CRS tick periods')
        previous_stop = 0
        events = f['events/data']
        for offset in range(0, len(events), 1000):
            for event in events[offset:offset+1000]:
                start, stop = int(event['hit_start']), int(event['hit_stop'])
                if start != previous_stop or not start <= stop <= len(dataset):
                    raise ValueError('Invalid/noncontiguous event hit ranges')
                previous_stop = stop
                hits = dataset[start:stop]
                if (str(int(event['file_id'])) not in periods or
                        np.any(hits['file_id'] != event['file_id']) or
                        np.any(hits['source_event_row'] != event['source_event_row'])):
                    raise ValueError('Event hit range crosses source identity')
                delta, counters = pixel_deltas(hits, time_field)
                audit.update(counters)
                audit['events'] += 1
                counts += np.histogram(delta, bins=edges)[0]
                audit['pairs_above_range'] += int((delta > edges[-1]).sum())
                audit['pairs_in_histogram'] += int(((delta >= edges[0]) & (delta <= edges[-1])).sum())
        if previous_stop != len(dataset):
            raise ValueError('Unreferenced output hits after final event')
    return dict(counts=counts, audit=dict(audit), crs_ticks=next(iter(periods.values())))


def resolve_inputs(root, config, data_label):
    """Resolve a paired registry or an explicit, labeled multi-sample recipe."""
    if config['target']['kind'] != 'comparison':
        raise ValueError('Use a comparison recipe')
    if 'samples' not in config:
        target, inputs = resolve_target(root, config['target'])
        if {kind for kind, _, _ in inputs} != {'data', 'mc'}:
            raise ValueError('Comparison must contain data and MC')
        return target, [(kind, data_label if kind == 'data' else 'MC', path, kind)
                        for kind, _, path in inputs]
    references = config['samples']
    if not references or sum(r['kind'] == 'mc' for r in references.values()) != 1:
        raise ValueError('Combined comparison requires exactly one shared MC sample')
    if not any(r['kind'] == 'data' for r in references.values()):
        raise ValueError('Combined comparison requires data')
    inputs = []
    for key, reference in references.items():
        safe_name(key)
        _, resolved = resolve_target(root, dict(kind='sample', id=reference['id'],
                                               selection_version=reference.get('selection_version', 'v1')))
        kind, _, path = resolved[0]
        if kind != reference['kind']:
            raise ValueError(f'Sample kind mismatch for {key}')
        inputs.append((key, reference.get('label', key), path, kind))
    if len({path for _, _, path, _ in inputs}) != len(inputs):
        raise ValueError('Combined comparison repeats a selection file')
    target = root/'comparisons'/safe_name(config['target']['id'])
    registry = dict(id=config['target']['id'], kind='hit-time-comparison', samples=references)
    registry_path = target/'comparison.json'
    if registry_path.exists():
        if read_json(registry_path) != registry:
            raise FileExistsError(f'Existing comparison registry differs: {registry_path}')
    else:
        write_json(registry_path, registry)
    return target, inputs


def run(config_path, root, run_id, bins=200, max_delta_ticks=200., time_field='ts_pps', data_label='Data'):
    if bins < 1 or not np.isfinite(max_delta_ticks) or max_delta_ticks <= 0:
        raise ValueError('Require positive bin count and maximum time difference')
    if time_field not in {'ts_pps', 't_drift'}:
        raise ValueError('time_field must be ts_pps or t_drift')
    config = read_json(config_path)
    target, inputs = resolve_inputs(Path(root).resolve(), config, data_label)
    identities = {key: file_identity(path) for key, _, path, _ in inputs}
    edges = np.linspace(0, max_delta_ticks, bins+1)
    directory = target/'analysis'/safe_name(run_id)
    directory.mkdir(parents=True, exist_ok=False)
    output = directory/'hit_time_difference'
    output.mkdir()
    settings = dict(time_field=time_field, edges_ticks=edges.tolist(),
                    hit_view='all recorded hits in saved events; includes track-rejected hits',
                    pixel_key=['file_id', 'source_event_row', *ADDRESS], multiplicity=2,
                    delta='absolute difference; no rollover correction',
                    normalization='bin count / all exactly-two-hit pixel/event instances; includes out-of-range and invalid-time instances in denominator; no bin-width division',
                    population='only events with at least one accepted track are present in selections')
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                    inputs=identities, settings=settings, source_hashes=code_hashes(['analysis/hit_time_difference.py']))
    write_json(directory/'config.json', dict(recipe=config, settings=settings, data_label=data_label))
    write_json(directory/'manifest.json', manifest)
    start = time.monotonic()
    try:
        samples = {key: summarize(path, edges, time_field) for key, _, path, _ in inputs}
        periods = [sample['crs_ticks'] for sample in samples.values()]
        if not np.allclose(periods, periods[0], rtol=1e-10, atol=0):
            raise ValueError('Data and MC CRS tick periods differ')
        fig, ax = plt.subplots(figsize=(9, 5.5))
        try:
            colors = iter(['#0072B2', '#009E73', '#CC79A7', '#D55E00', '#56B4E9'])
            for key, label, _, kind in inputs:
                color = '#E69F00' if kind == 'mc' else next(colors, None)
                sample = samples[key]
                sample.update(label=label, kind=kind)
                counts = sample['counts']
                n = sample['audit'].get('pixels_two_hits', 0)
                sample['normalization_instances'] = n
                overflow = sample['audit'].get('pairs_above_range', 0)
                fraction = counts/n if n else np.zeros(len(counts))
                ax.stairs(fraction, edges, color=color, linewidth=1.4,
                          linestyle='--' if kind == 'mc' else '-',
                          label=f'{label}: {n:,} instances; {overflow:,} above range' if n else f'{label}: no two-hit instances')
                np.savetxt(output/f'{key}_delta_t.csv', np.column_stack([edges[:-1], edges[1:], counts, fraction]),
                           delimiter=',', header='bin_low_ticks,bin_high_ticks,count,fraction', comments='')
                print(f'{label}: {sample["audit"]}', flush=True)
            ax.set(title='Time separation in pixels with exactly two hits', xlabel='Hit time separation |Δt| [CRS ticks]',
                   ylabel='Fraction of two-hit pixel/event instances', xlim=(0, max_delta_ticks), ylim=(0, None))
            ax.legend()
            ax.grid(axis='y', alpha=.2)
            fig.text(.5, .02, 'Bin count / all two-hit pixel/event instances, including those outside the displayed range; no bin-width division.',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .06, 1, 1))
            for ext in ('png', 'pdf'):
                fig.savefig(output/f'hit_time_difference.{ext}', dpi=180)
        finally:
            plt.close(fig)
        if identities != {key: file_identity(path) for key, _, path, _ in inputs}:
            raise ValueError('Input changed during analysis')
        manifest.update(status='complete', elapsed_seconds=time.monotonic()-start,
                        samples={kind: {key: value for key, value in sample.items() if key != 'counts'}
                                 for kind, sample in samples.items()})
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(directory/'manifest.json', manifest)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='paired data/MC or combined multi-sample comparison recipe')
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-id', default='hit-timing-001')
    parser.add_argument('--bins', type=int, default=200)
    parser.add_argument('--max-delta-ticks', type=float, default=200.)
    parser.add_argument('--time-field', choices=['ts_pps', 't_drift'], default='ts_pps')
    parser.add_argument('--data-label', default='Data')
    args = vars(parser.parse_args())
    args['config_path'] = args.pop('config')
    print('Complete:', run(**args))


if __name__ == '__main__':
    main()
