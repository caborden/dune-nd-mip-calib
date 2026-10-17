"""Read-only FLOW inspection and bounded event/hit reference traversal."""
from pathlib import Path
import glob
import os

import h5py
import numpy as np

ADDRESS = ('io_group', 'io_channel', 'chip_id', 'channel_id')
EVENTS = 'charge/events'


def json_value(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def resolve_samples(samples, config_dir):
    """Reject repeated paths, symlinks, and hard links, including across samples."""
    seen, names, result = set(), set(), []
    for sample in samples:
        name = sample['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Sample names must contain only letters, numbers, _ and -')
        if name in names:
            raise ValueError(f'Duplicate sample name: {name}')
        names.add(name)
        files = []
        for pattern in sample['files']:
            path = Path(os.path.expandvars(os.path.expanduser(pattern)))
            if not path.is_absolute():
                path = config_dir / path
            matches = sorted(glob.glob(str(path)))
            if not matches:
                raise FileNotFoundError(f'No files match: {path}')
            for match in matches:
                resolved = Path(match).resolve(strict=True)
                stat = resolved.stat()
                identity = (stat.st_dev, stat.st_ino)
                if not resolved.is_file():
                    raise ValueError(f'Not a file: {resolved}')
                if identity in seen:
                    raise ValueError(f'Repeated input (possibly an alias): {resolved}')
                seen.add(identity)
                files.append(str(resolved))
        if not files:
            raise ValueError(f'Empty sample: {name}')
        result.append(dict(name=name, label=sample.get('label', name), files=files))
    if not result:
        raise ValueError('At least one sample is required')
    return result


def tick_us(f):
    """FLOW run_info/crs_ticks is in microseconds; never guess a missing clock."""
    try:
        value = float(f['run_info'].attrs['crs_ticks'])
        return value if np.isfinite(value) and value > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def chunks(dataset, size):
    for start in range(0, len(dataset), size):
        yield dataset[start:start + size]


def inspect_file(path, settings):
    stat = Path(path).stat()
    hits_path = settings['hits_path']
    with h5py.File(path, 'r') as f:
        hits = f[f'{hits_path}/data']
        events = f[f'{EVENTS}/data']
        names = set(hits.dtype.names or ())
        missing = set(ADDRESS + ('Q',)) - names
        if missing:
            raise ValueError(f'{path}: missing required calibrated-hit fields: {sorted(missing)}')
        for name in ADDRESS + ('Q',):
            if hits.dtype[name].shape or hits.dtype[name].kind not in 'biuf':
                raise ValueError(f'{path}: unsupported hit field {name}: {hits.dtype[name]}')
        if any(hits.dtype[n].kind not in 'iu' for n in ADDRESS):
            raise ValueError(f'{path}: electronics addresses must be integers')
        base = f'{EVENTS}/ref/{hits_path}'
        references = all(p in f for p in (base+'/ref', base+'/ref_region'))
        if references:
            ref, region = f[base+'/ref'], f[base+'/ref_region']
            if (ref.ndim != 2 or ref.shape[1] != 2 or ref.dtype.kind not in 'iu'
                    or not {'start', 'stop'} <= set(region.dtype.names or ())
                    or len(region) != len(events)):
                raise ValueError(f'{path}: unsupported forward event-hit reference schema')
            if any(region.dtype[n].kind not in 'iu' for n in ('start', 'stop')):
                raise ValueError(f'{path}: reference region bounds must be integers')
        metadata = {}
        for group in ('run_info', 'geometry_info', hits_path, EVENTS, 'combined/t0'):
            if group in f:
                metadata[group] = json_value(dict(f[group].attrs))
        return dict(path=path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    n_hits=len(hits), n_events=len(events), hit_dtype=str(hits.dtype),
                    hit_fields=sorted(names), event_fields=list(events.dtype.names or ()),
                    packet_fields=list(f['charge/packets/data'].dtype.names or ())
                    if 'charge/packets/data' in f else [],
                    references_available=references, tick_us=tick_us(f), metadata=metadata)


def _read_rows(dataset, rows):
    if not len(rows):
        return np.empty(0, dtype=dataset.dtype)
    if int(rows[-1]) - int(rows[0]) + 1 == len(rows):
        return dataset[int(rows[0]):int(rows[-1]) + 1]
    # h5py point selection is slow for very long lists. Read contiguous runs.
    cuts = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1, len(rows)]
    if len(cuts) < 64:
        return np.concatenate([dataset[int(rows[a]):int(rows[b-1])+1]
                               for a, b in zip(cuts[:-1], cuts[1:])])
    return np.concatenate([dataset[rows[i:i+4096]] for i in range(0, len(rows), 4096)])


def event_hits(f, settings, audit):
    """Yield (event row, unique linked hits); reject malformed events, count them.

    Read several events together when the enclosing reference span fits the
    configured memory limit. A large batch falls back to individual events.
    References are row indices, never event IDs. Other sources in an enclosing
    ref_region are allowed by h5flow and filtered explicitly.
    """
    base = f"{EVENTS}/ref/{settings['hits_path']}"
    if base+'/ref' not in f or base+'/ref_region' not in f:
        audit['events_missing_references'] += len(f[EVENTS+'/data'])
        return
    refs, regions = f[base+'/ref'], f[base+'/ref_region']
    hits = f[settings['hits_path']+'/data']
    limit = settings['max_event_links']
    # Audit every reference, including links with invalid source rows that no
    # event's region would otherwise visit. Do not silently drop these records.
    for block in chunks(refs, settings['chunk_size']):
        audit['reference_rows'] += len(block)
        audit['reference_rows_invalid_source'] += int(np.count_nonzero(
            (block[:, 0] < 0) | (block[:, 0] >= len(regions))))
        audit['reference_rows_invalid_target'] += int(np.count_nonzero(
            (block[:, 1] < 0) | (block[:, 1] >= len(hits))))

    def read_batch(indices, spans):
        valid = []
        for row, span in zip(indices, spans):
            start, stop = int(span['start']), int(span['stop'])
            if not 0 <= start <= stop <= len(refs):
                audit['events_invalid_reference_region'] += 1
            elif stop - start > limit:
                audit['events_over_link_limit'] += 1
            else:
                valid.append((int(row), start, stop))
        if not valid:
            return
        first, last = min(v[1] for v in valid), max(v[2] for v in valid)
        if last - first > limit:
            for row, start, stop in valid:
                yield from read_batch([row], [dict(start=start, stop=stop)])
            return
        links = refs[first:last]
        per_event = []
        for row, start, stop in valid:
            window = links[start-first:stop-first]
            target = window[window[:, 0] == row, 1]
            audit['reference_rows_visited_in_regions'] += len(target)
            if (len(window) and not len(target)) or np.any((target < 0) | (target >= len(hits))):
                audit['events_invalid_hit_references'] += 1
                continue
            unique = np.unique(target).astype(np.int64)
            audit['duplicate_event_hit_links'] += len(target) - len(unique)
            per_event.append((row, unique))
        if not per_event:
            return
        all_rows = np.unique(np.concatenate([r for _, r in per_event]))
        records = _read_rows(hits, all_rows)
        for row, rows in per_event:
            audit['events_analyzed'] += 1
            yield row, records[np.searchsorted(all_rows, rows)]

    size = settings['event_batch_size']
    for start in range(0, len(regions), size):
        spans = regions[start:start+size]
        yield from read_batch(range(start, start+len(spans)), spans)
