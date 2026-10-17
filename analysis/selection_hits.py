"""Selected-hit charge and event-scoped electronics-pixel multiplicity."""
from collections import Counter
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .io import write_json, file_identity, code_hashes

SETTING_KEYS = {'charge_bins', 'charge_range'}
PIXEL_KEYS = ['io_group', 'io_channel', 'chip_id', 'channel_id']


def make_settings(charge_bins=100, charge_range=(0., 100.)):
    if (not isinstance(charge_bins, int) or charge_bins < 1 or len(charge_range) != 2
            or not np.isfinite(charge_range).all() or charge_range[0] >= charge_range[1]):
        raise ValueError('Invalid hit-charge histogram settings')
    return dict(charge_bins=charge_bins, charge_range=list(charge_range))


def summarize(path, settings):
    edges = np.linspace(*settings['charge_range'], settings['charge_bins'] + 1)
    counts = np.zeros(settings['charge_bins'], dtype=np.int64)
    multiplicity = Counter()
    audit = dict(selected_hits=0, nonfinite_charge=0, charge_underflow=0, charge_overflow=0,
                 invalid_pixel_hits=0, events=0)
    with h5py.File(path, 'r') as f:
        if f.attrs.get('status') != 'complete' or f.attrs.get('schema_version') != '2.0':
            raise ValueError('Expected a complete schema-2 selection')
        hits = f['events/hits']
        required = set(PIXEL_KEYS + ['Q', 'selected_track_id', 'file_id', 'source_event_row'])
        if not required <= set(hits.dtype.names):
            raise ValueError(f'Missing hit fields: {required - set(hits.dtype.names)}')
        events = f['events/data']
        for start in range(0, len(events), 4096):
            for event in events[start:start+4096]:
                rows = hits[int(event['hit_start']):int(event['hit_stop'])]
                if np.any(rows['file_id'] != event['file_id']) or np.any(rows['source_event_row'] != event['source_event_row']):
                    raise ValueError('Event hit range references a different source event')
                rows = rows[rows['selected_track_id'] >= 0]
                audit['events'] += 1
                audit['selected_hits'] += len(rows)
                q = rows['Q'].astype(float)
                finite = np.isfinite(q)
                audit['nonfinite_charge'] += int((~finite).sum())
                q = q[finite]
                audit['charge_underflow'] += int((q < edges[0]).sum())
                audit['charge_overflow'] += int((q > edges[-1]).sum())
                counts += np.histogram(q, edges)[0]
                valid = np.ones(len(rows), dtype=bool)
                for key in PIXEL_KEYS:
                    valid &= np.isfinite(rows[key]) & (rows[key] >= 0)
                audit['invalid_pixel_hits'] += int((~valid).sum())
                # Each event is independent; different tracks on the same pixel combine.
                _, n = np.unique(rows[valid][PIXEL_KEYS], return_counts=True)
                multiplicity.update(map(int, n))
    return dict(input=file_identity(path), audit=audit, charge_edges=edges.tolist(),
                charge_counts=counts.tolist(), multiplicity=dict(sorted(multiplicity.items())))


def draw(samples, directory):
    for variable in ['charge', 'multiplicity']:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sample in samples:
            if variable == 'charge':
                counts = np.asarray(sample['charge_counts'])
                ax.stairs(counts / counts.sum() if counts.sum() else counts,
                          sample['charge_edges'], label=sample['label'], color=sample.get('color'))
            else:
                values = sorted((int(k), v) for k, v in sample['multiplicity'].items())
                total = sum(v for _, v in values)
                ax.plot([k for k, _ in values], [v / total for _, v in values], 'o-', label=sample['label'], color=sample.get('color'))
        ax.set(xlabel='Selected hit Q [ke−]' if variable == 'charge' else 'Selected hits per occupied pixel per event',
               ylabel='Fraction of in-range hits' if variable == 'charge' else 'Fraction of occupied pixel-events',
               title='FSD Cube: selected-hit charge' if variable == 'charge' else 'FSD Cube: pixel hit multiplicity')
        if variable == 'charge':
            ax.set_xlim(samples[0]['charge_edges'][0], samples[0]['charge_edges'][-1])
        else:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(Path(directory)/f'{variable}.{ext}', dpi=180)
        plt.close(fig)


def analyze(samples, output_dir, settings):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for kind, title, path in samples:
        result = dict(summarize(path, settings), sample_kind=kind, label='Data' if kind == 'data' else 'Simulation')
        results.append(result)
        edges = result['charge_edges']
        np.savetxt(directory/f'{kind}_charge.csv', np.column_stack([edges[:-1], edges[1:], result['charge_counts']]),
                   delimiter=',', header='bin_low,bin_high,count', comments='')
        np.savetxt(directory/f'{kind}_multiplicity.csv', np.asarray(list(result['multiplicity'].items())).reshape(-1, 2),
                   fmt='%d', delimiter=',', header='hits_per_pixel_event,pixel_events', comments='')
    draw(results, directory)
    write_json(directory/'analysis.json', dict(status='complete', settings=settings, samples=results,
        source_hashes=code_hashes(['analysis/selection_hits.py']),
        definitions=dict(hit_view='selected_track_id >= 0; each stored hit counted once; no segment cuts',
                         pixel_key=PIXEL_KEYS, scope='file_id, source_event_row',
                         multiplicity='one entry per occupied pixel per event, pooled across events; no zero-hit pixels',
                         charge_units='Q assumed ke−; no additional charge correction',
                         normalization='charge: in-range hits; multiplicity: occupied pixel-events')))
