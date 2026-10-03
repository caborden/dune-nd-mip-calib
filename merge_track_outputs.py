"""Merge completed schema-2 per-FLOW outputs in bounded chunks.

Resume checkpoints are committed per input file, never per partially written event.
Use one writer per destination. Original input files are never modified.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np

TABLES = ('events/data', 'events/hits', 'tracks/data', 'events/packets',
          'events/hit_packet_refs', 'events/t0', 'events/ext_trigs')
OPTIONAL = {'events/packets', 'events/t0', 'events/ext_trigs'}


def signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def publish(partial, output):
    # Same-directory hard link publishes without overwriting an existing output.
    os.link(partial, output)
    Path(partial).unlink()


def plan_merge(inputs):
    schemas, sources, signatures = {}, set(), []
    for path in inputs:
        signatures.append(signature(path))
        with h5py.File(path, 'r') as f:
            if f.attrs.get('schema_version') != '2.0' or f.attrs.get('status') != 'complete':
                raise ValueError(f'{path}: requires a completed schema-2 output')
            if list(f['sources']) != ['0'] or f['sources/0'].attrs['status'] != 'complete':
                raise ValueError(f'{path}: merge accepts per-input outputs, not already merged outputs')
            source = str(f['sources/0/metadata'].attrs['input_file'])
            if source in sources:
                raise ValueError(f'Duplicate FLOW source: {source}')
            sources.add(source)
            for name in TABLES:
                dtype = f[name].dtype
                if 'file_id' not in dtype.names:
                    raise ValueError(f'{path}: missing file_id in {name}')
                if name not in schemas:
                    schemas[name] = dtype
                elif dtype != schemas[name]:
                    # An unavailable optional table has only bookkeeping fields.
                    if name not in OPTIONAL:
                        raise ValueError(f'Incompatible schema for {name}: {path}')
                    old = schemas[name]
                    common = set(old.names) & set(dtype.names)
                    if any(old.fields[k][0] != dtype.fields[k][0] for k in common):
                        raise ValueError(f'Incompatible field types for {name}: {path}')
                    if set(old.names) <= set(dtype.names):
                        schemas[name] = dtype
                    elif not set(dtype.names) <= set(old.names):
                        raise ValueError(f'Incompatible optional schema for {name}: {path}')
    # Resolve optional empty schemas before creating a destination. A smaller
    # nonempty schema cannot be expanded with invented field values.
    for path in inputs:
        with h5py.File(path, 'r') as f:
            for name in TABLES:
                if len(f[name]) and f[name].dtype != schemas[name]:
                    raise ValueError(f'Incompatible nonempty schema for {name}: {path}')
    return schemas, signatures


def append(dataset, data):
    start = len(dataset)
    dataset.resize(start + len(data), axis=0)
    dataset[start:] = data


def merge_outputs(inputs, output, resume=False, chunk_rows=100000):
    if chunk_rows <= 0:
        raise ValueError('chunk_rows must be positive')
    inputs = [Path(p).resolve() for p in inputs]
    if not inputs:
        raise ValueError('No merge inputs')
    output = Path(output).resolve()
    partial = output.with_name(output.name + '.partial')
    if output in inputs or partial in inputs:
        raise ValueError('Merge output overlaps an input')
    schemas, signatures = plan_merge(inputs)
    expected = json.dumps(signatures, sort_keys=True)
    if output.exists():
        if resume:
            with h5py.File(output, 'r') as existing:
                if (existing.attrs.get('status') == 'complete' and
                        existing.attrs.get('schema_version') == '2.0' and
                        existing.attrs.get('merge_inputs') == expected):
                    return output
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists() and not resume:
        raise FileExistsError(f'{partial}; use --resume to recover')
    mode = 'r+' if partial.exists() else 'x'
    with h5py.File(partial, mode) as out:
        if mode == 'x':
            out.attrs['schema_version'] = '2.0'
            out.attrs['status'] = 'incomplete'
            out.attrs['merge_inputs'] = expected
            out.attrs['committed_inputs'] = 0
            out.attrs['committed_lengths'] = json.dumps({name: 0 for name in TABLES})
            out.create_group('sources')
            meta = out.create_group('metadata')
            meta.attrs['kind'] = 'merged; run/calibration/selection metadata are per source'
            meta.attrs['merge_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            meta.attrs['h5py_version'] = h5py.__version__
            meta.attrs['numpy_version'] = np.__version__
            for name, dtype in schemas.items():
                out.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                   chunks=True, compression='gzip')
            with h5py.File(inputs[0], 'r') as first:
                for name in TABLES:
                    for key, value in first[name].attrs.items():
                        if key != 'available':
                            out[name].attrs[key] = value
            out.flush()
        elif out.attrs['merge_inputs'] != expected:
            raise ValueError('Resume inputs/order or their size/mtime changed')
        committed = int(out.attrs['committed_inputs'])
        if committed < 0 or committed > len(inputs):
            raise ValueError('Invalid merge checkpoint')
        # Source-group commit markers are authoritative: a crash after appending
        # but before this marker leaves an entire input to roll back.
        while committed < len(inputs) and str(committed) in out['sources'] and out[f'sources/{committed}'].attrs.get('merge_committed', False):
            committed += 1
        lengths = ({name: 0 for name in TABLES} if committed == 0 else
                   json.loads(out[f'sources/{committed-1}'].attrs['output_stops']))
        for name in TABLES:
            if len(out[name]) < lengths[name]:
                raise ValueError('Output shorter than its committed checkpoint')
            out[name].resize(lengths[name], axis=0)
        for key in list(out['sources']):
            if int(key) >= committed:
                del out[f'sources/{key}']
        out.attrs['status'] = 'incomplete'
        out.attrs['committed_inputs'] = committed
        out.attrs['committed_lengths'] = json.dumps(lengths)
        out.flush()
        for file_id in range(committed, len(inputs)):
            with h5py.File(inputs[file_id], 'r') as source:
                offsets = {name: len(out[name]) for name in TABLES}
                n_tracks = len(source['tracks/data'])
                # Schema-2 selector assigns local IDs in row order.
                for start in range(0, n_tracks, chunk_rows):
                    ids = source['tracks/data'][start:start+chunk_rows]['track_id']
                    if not np.array_equal(ids, np.arange(start, start+len(ids))):
                        raise ValueError('Nonsequential local track IDs')
                for name in TABLES:
                    dataset = source[name]
                    if dataset.dtype != schemas[name] and len(dataset):
                        raise ValueError(f'Nonempty incompatible optional table: {name}')
                    for start in range(0, len(dataset), chunk_rows):
                        data = dataset[start:start+chunk_rows]
                        if np.any(data['file_id'] != 0):
                            raise ValueError('Per-input file_id must be zero')
                        data['file_id'] = file_id
                        if name == 'tracks/data':
                            data['track_id'] += offsets[name]
                        elif name == 'events/hits':
                            ids = data['selected_track_id']
                            if np.any((ids < -1) | (ids >= n_tracks)):
                                raise ValueError('Hit refers to an invalid local track')
                            data['selected_track_id'][ids >= 0] += offsets['tracks/data']
                        elif name == 'events/data':
                            if np.any((data['hit_start'] < 0) | (data['hit_stop'] < data['hit_start']) | (data['hit_stop'] > len(source['events/hits']))):
                                raise ValueError('Invalid event hit range')
                            data['hit_start'] += offsets['events/hits']
                            data['hit_stop'] += offsets['events/hits']
                        elif name == 'events/hit_packet_refs':
                            for field, target in [('hit_row', 'events/hits'), ('packet_row', 'events/packets')]:
                                if np.any((data[field] < 0) | (data[field] >= len(source[target]))):
                                    raise ValueError(f'Invalid {field} reference')
                                data[field] += offsets[target]
                        append(out[name], data)
                source.copy(source['sources/0'], out['sources'], name=str(file_id))
                group = out[f'sources/{file_id}']
                group.attrs['per_input_output'] = str(inputs[file_id])
                group.attrs['output_starts'] = json.dumps(offsets)
                stops = {name: len(out[name]) for name in TABLES}
                group.attrs['output_stops'] = json.dumps(stops)
                # Flush rows and metadata before writing the commit marker.
                out.flush()
                group.attrs['merge_committed'] = True
                out.flush()
                out.attrs['committed_inputs'] = file_id+1
                out.attrs['committed_lengths'] = json.dumps(stops)
                out.flush()
        out.attrs['status'] = 'complete'
        out.flush()
    publish(partial, output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='completed per-input track HDF5 files')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--chunk-rows', type=int, default=100000)
    args = parser.parse_args()
    merge_outputs(args.inputs, args.output, args.resume, args.chunk_rows)
