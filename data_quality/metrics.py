"""Streaming sample accumulators; no track selection or reconstruction required."""
from collections import Counter
from dataclasses import dataclass, field
import time

import h5py
import numpy as np

from .flow import ADDRESS, EVENTS, chunks, event_hits

DEFAULT_HISTOGRAMS = {
    'charge': [-20, 200, 220], 'adc': [-.5, 255.5, 256],
    'drift_us': [-100, 2000, 210], 'event_hits': [-.5, 4999.5, 200],
    'event_channels': [-.5, 1999.5, 200], 'event_charge': [-100, 100000, 200],
    'event_duration_us': [0, 2000, 200], 'multiplicity': [.5, 50.5, 50],
    'delta_us': [0, 100, 200],
}


def edges(spec):
    if len(spec) != 3:
        raise ValueError('Histogram specification must be [low, high, number_of_bins]')
    low, high, number = spec
    if (not np.isfinite([low, high, number]).all() or high <= low
            or int(number) != number or not 1 <= number <= 100000):
        raise ValueError(f'Invalid histogram specification: {spec}')
    return np.linspace(low, high, int(number) + 1)


@dataclass
class Histogram:
    edges: np.ndarray
    total: int = 0
    nonfinite: int = 0
    underflow: int = 0
    overflow: int = 0
    counts: np.ndarray = field(init=False)

    def __post_init__(self):
        self.counts = np.zeros(len(self.edges)-1, dtype=np.int64)

    def add(self, values):
        values = np.atleast_1d(values)
        self.total += len(values)
        good = np.isfinite(values)
        self.nonfinite += int((~good).sum())
        values = values[good]
        self.underflow += int((values < self.edges[0]).sum())
        self.overflow += int((values > self.edges[-1]).sum())
        self.counts += np.histogram(values, self.edges)[0]

    def summary(self):
        return dict(total=self.total, nonfinite=self.nonfinite,
                    underflow=self.underflow, overflow=self.overflow,
                    in_range=int(self.counts.sum()))


def positive_differences(ordered):
    """Subtract integers before float conversion, including uint64 above 2**53."""
    if ordered.dtype.kind in 'iu':
        return np.diff(ordered.astype(np.uint64)).astype(float)
    return np.diff(ordered.astype(float))


def timing(hits, time_field, tick, audit):
    """All adjacent hits per channel/event; never pair across events or files."""
    if not len(hits):
        return np.empty(0), np.empty(0)
    order = np.argsort(hits[list(ADDRESS)], order=list(ADDRESS))
    records = hits[order]
    addresses = records[list(ADDRESS)]
    cuts = np.r_[0, np.flatnonzero(addresses[1:] != addresses[:-1]) + 1, len(records)]
    multiplicities = np.diff(cuts)
    deltas = []
    if time_field in (hits.dtype.names or ()) and tick is not None:
        for start, stop in zip(cuts[:-1], cuts[1:]):
            values = records[time_field][start:stop]
            if not np.isfinite(values).all():
                audit['channel_events_nonfinite_time'] += 1
                continue  # Do not bridge across an invalid timestamp.
            if len(values) < 2:
                continue
            delta = positive_differences(np.sort(values))
            audit['same_channel_zero_time_pairs'] += int((delta == 0).sum())
            deltas.append(delta * tick)
    return multiplicities, np.concatenate(deltas) if deltas else np.empty(0)


class Sample:
    def __init__(self, definition, settings):
        self.name, self.label = definition['name'], definition['label']
        self.settings = settings
        self.histograms = {k: Histogram(edges(v)) for k, v in settings['histograms'].items()}
        self.channels = {}
        self.charge_edges = edges(settings['channel_charge_bins'])
        self.audit = Counter()
        self.files = []
        self.coverage = {k: [] for k in self.histograms}
        self.bounds = {axis: [np.inf, -np.inf] for axis in 'xyz'}
        self.projections = {}
        self.warnings = []

    def add_hits(self, hits, tick):
        names, audit = set(hits.dtype.names), self.audit
        audit['hit_rows'] += len(hits)
        q = hits['Q']
        self.histograms['charge'].add(q)
        audit['nonfinite_charge_hits'] += int((~np.isfinite(q)).sum())
        audit['nonpositive_charge_hits'] += int((np.isfinite(q) & (q <= 0)).sum())
        if 'is_disabled' in names:
            audit['hits_with_disabled_flag_available'] += len(hits)
            audit['disabled_flagged_hits'] += int(np.count_nonzero(hits['is_disabled']))
        if 't_drift' in names and tick is not None:
            self.histograms['drift_us'].add(hits['t_drift'].astype(float) * tick)
            audit['negative_drift_hits'] += int((hits['t_drift'] < 0).sum())
        for axis in 'xyz':
            if axis in names:
                finite = hits[axis][np.isfinite(hits[axis])]
                audit[f'nonfinite_{axis}_hits'] += len(hits)-len(finite)
                if len(finite):
                    self.bounds[axis][0] = min(self.bounds[axis][0], float(finite.min()))
                    self.bounds[axis][1] = max(self.bounds[axis][1], float(finite.max()))
        if {'x', 'y', 'z'} <= names:
            audit['nonfinite_xyz_hits'] += int((~np.isfinite(np.column_stack(
                [hits[a] for a in 'xyz'])).all(axis=1)).sum())
        if not len(hits):
            return
        order = np.argsort(hits[list(ADDRESS)], order=list(ADDRESS))
        sorted_hits = hits[order]
        addresses = sorted_hits[list(ADDRESS)]
        cuts = np.r_[0, np.flatnonzero(addresses[1:] != addresses[:-1]) + 1, len(hits)]
        for start, stop in zip(cuts[:-1], cuts[1:]):
            part = sorted_hits[start:stop]
            key = tuple(int(part[n][0]) for n in ADDRESS)
            if key not in self.channels:
                self.channels[key] = dict(count=0, charge=Histogram(self.charge_edges),
                                          yz_sum=np.zeros(2), yz_n=0,
                                          yz_min=np.full(2, np.inf), yz_max=np.full(2, -np.inf))
            channel = self.channels[key]
            channel['count'] += len(part)
            channel['charge'].add(part['Q'])
            if {'y', 'z'} <= names:
                yz = np.column_stack([part['y'], part['z']])
                yz = yz[np.isfinite(yz).all(axis=1)]
                if len(yz):
                    channel['yz_sum'] += yz.sum(axis=0)
                    channel['yz_n'] += len(yz)
                    channel['yz_min'] = np.minimum(channel['yz_min'], yz.min(axis=0))
                    channel['yz_max'] = np.maximum(channel['yz_max'], yz.max(axis=0))

    def add_event(self, hits, tick):
        self.histograms['event_hits'].add(len(hits))
        unique = np.unique(hits[list(ADDRESS)])
        self.histograms['event_channels'].add(len(unique))
        if np.isfinite(hits['Q']).all():
            self.histograms['event_charge'].add(float(hits['Q'].sum(dtype=float)))
        else:
            self.audit['events_excluded_from_charge_nonfinite'] += 1
        multiplicity, delta = timing(hits, self.settings['time_field'], tick, self.audit)
        self.histograms['multiplicity'].add(multiplicity)
        self.histograms['delta_us'].add(delta)
        field_name = self.settings['time_field']
        if field_name in hits.dtype.names and tick is not None and len(hits):
            values = hits[field_name]
            if np.isfinite(values).all():
                duration = positive_differences(np.array([values.min(), values.max()]))[0] * tick
                self.histograms['event_duration_us'].add(duration)
            else:
                self.audit['events_excluded_from_duration_nonfinite'] += 1
        if not len(hits):
            self.audit['empty_events'] += 1

    def process(self, info):
        path, settings = info['path'], self.settings
        print(f'[{self.name}] {path}', flush=True)
        before = self.audit.copy()
        self.files.append(info)
        with h5py.File(path, 'r') as f:
            hits = f[settings['hits_path']+'/data']
            tick = info['tick_us']
            names = set(hits.dtype.names)
            self.coverage['charge'].append(path)
            if 't_drift' in names and tick is not None:
                self.coverage['drift_us'].append(path)
            if tick is None:
                self.warnings.append(f'{path}: missing/invalid crs_ticks; time plots omit this file.')
            if settings['time_field'] not in names:
                self.warnings.append(f'{path}: missing {settings["time_field"]}; duration and delta-t omitted.')
            last_update = time.monotonic()
            processed_hits = 0
            for part in chunks(hits, settings['chunk_size']):
                self.add_hits(part, tick)
                processed_hits += len(part)
                if time.monotonic()-last_update > 30:
                    print(f'[{self.name}] hit rows {processed_hits:,}/{len(hits):,}', flush=True)
                    last_update = time.monotonic()
            self.audit['event_rows'] += info['n_events']
            # Timestamp envelopes are diagnostics, never treated as live time.
            event_data = f[EVENTS+'/data']
            if 'unix_ts' in (event_data.dtype.names or ()):
                low, high = np.inf, -np.inf
                for block in chunks(event_data.fields('unix_ts'), settings['chunk_size']):
                    finite = block[np.isfinite(block) & (block > 0)]
                    self.audit['invalid_event_unix_timestamps'] += len(block)-len(finite)
                    if len(finite):
                        low, high = min(low, float(finite.min())), max(high, float(finite.max()))
                info['event_unix_envelope'] = [low, high] if np.isfinite([low, high]).all() else None
            self.add_packets(f, info)
            if info['references_available']:
                for metric in ('event_hits', 'event_channels', 'event_charge', 'multiplicity'):
                    self.coverage[metric].append(path)
                if settings['time_field'] in names and tick is not None:
                    self.coverage['event_duration_us'].append(path)
                    self.coverage['delta_us'].append(path)
            else:
                self.warnings.append(f'{path}: no forward event-hit refs; event metrics omitted.')
            last_update = time.monotonic()
            for row, records in event_hits(f, settings, self.audit):
                self.add_event(records, tick)
                if time.monotonic()-last_update > 30:
                    print(f'[{self.name}] event row {row+1:,}/{info["n_events"]:,}', flush=True)
                    last_update = time.monotonic()
        info['audit'] = dict(self.audit - before)
        unvisited = (info['audit'].get('reference_rows', 0)
                     - info['audit'].get('reference_rows_visited_in_regions', 0))
        if unvisited:
            self.warnings.append(f'{path}: {unvisited} reference rows were not visited through event regions; '
                                 'event associations may be incomplete. See integrity counters.')

    def add_packets(self, f, info):
        path = 'charge/packets/data'
        required = {'dataword', 'packet_type'}
        if path not in f or not required <= set(f[path].dtype.names or ()):
            self.warnings.append(f'{info["path"]}: ADC omitted; need packet_type and dataword in {path}.')
            return
        self.coverage['adc'].append(info['path'])
        for block in chunks(f[path], self.settings['chunk_size']):
            self.audit['packet_rows'] += len(block)
            data = block[block['packet_type'] == 0]
            self.audit['data_packet_rows'] += len(data)
            self.histograms['adc'].add(data['dataword'])
            self.audit['adc_at_configured_max'] += int((data['dataword'] == self.settings['adc_max']).sum())
            self.audit['adc_above_configured_max'] += int((data['dataword'] > self.settings['adc_max']).sum())
            if 'valid_parity' in data.dtype.names:
                self.audit['data_packets_with_parity_available'] += len(data)
                self.audit['data_packets_invalid_parity'] += int((data['valid_parity'] == 0).sum())

    def channel_rows(self):
        for key, channel in sorted(self.channels.items()):
            hist = channel['charge']
            finite = hist.total - hist.nonfinite
            median, status = np.nan, 'no_finite_charge'
            if finite:
                # Locate both middle ranks, matching the usual even-N median.
                ranks = [(finite-1)//2, finite//2]
                cumulative = hist.underflow + np.cumsum(hist.counts)
                if ranks[0] < hist.underflow:
                    status = 'median_rank_below_range'
                elif ranks[1] >= finite - hist.overflow:
                    status = 'median_rank_above_range'
                else:
                    bins = np.searchsorted(cumulative, ranks, side='right')
                    median = float(np.mean((hist.edges[bins] + hist.edges[bins+1])/2))
                    status = 'binned_estimate'
            yz = channel['yz_sum']/channel['yz_n'] if channel['yz_n'] else [np.nan, np.nan]
            stable = bool(channel['yz_n'] and np.all(channel['yz_max']-channel['yz_min'] <= 1e-3))
            yield dict(zip(ADDRESS, key), count=channel['count'],
                       hit_fraction=channel['count']/max(self.audit['hit_rows'], 1),
                       y=float(yz[0]), z=float(yz[1]), position_stable=stable,
                       median_charge=median, median_status=status,
                       finite_charge=finite, charge_underflow=hist.underflow, charge_overflow=hist.overflow)


def spatial_edges(samples, settings):
    result = {}
    for axis in 'xyz':
        configured = settings.get('spatial_ranges', {}).get(axis)
        if configured:
            low, high = configured
        else:
            low = min(s.bounds[axis][0] for s in samples)
            high = max(s.bounds[axis][1] for s in samples)
            if not np.isfinite([low, high]).all():
                low, high = -1, 1
            pad = max((high-low)*.01, .01)
            low, high = low-pad, high+pad
        result[axis] = edges([low, high, settings['spatial_bins']])
    return result


def fill_projections(sample, axes):
    """Second sequential coordinate pass gives common, data-derived plot ranges."""
    for pair in ('xy', 'xz', 'yz'):
        sample.projections[pair] = dict(counts=np.zeros((len(axes[pair[0]])-1,
                                                       len(axes[pair[1]])-1), dtype=np.int64),
                                        total=0, nonfinite=0, outside=0, files=0)
    for info in sample.files:
        print(f'[{sample.name}] spatial pass: {info["path"]}', flush=True)
        with h5py.File(info['path'], 'r') as f:
            hits = f[sample.settings['hits_path']+'/data']
            fields = [a for a in 'xyz' if a in hits.dtype.names]
            for pair, item in sample.projections.items():
                item['files'] += int(all(a in fields for a in pair))
            if not fields:
                continue
            for block in chunks(hits.fields(fields), sample.settings['chunk_size']):
                for pair, item in sample.projections.items():
                    if not all(a in fields for a in pair):
                        continue
                    a, b = (block[n] for n in pair)
                    finite = np.isfinite(a) & np.isfinite(b)
                    counts = np.histogram2d(a[finite], b[finite], bins=[axes[n] for n in pair])[0].astype(np.int64)
                    item['counts'] += counts
                    item['total'] += len(block)
                    item['nonfinite'] += int((~finite).sum())
                    item['outside'] += int(finite.sum()-counts.sum())
