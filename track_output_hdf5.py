"""Single-copy event-hit output for the legacy DBSCAN/PCA selector.

Reads direct forward FLOW references (source in column 0). IDs are never
interpreted as dataset rows. Original structured fields retain their dtypes.
"""
from pathlib import Path
import hashlib
import platform

import h5py
import numpy as np

HITS = 'charge/calib_prompt_hits'
TRACK_DTYPE = np.dtype([
    ('file_id', '<i8'), ('track_id', '<i8'), ('source_event_row', '<i8'),
    ('length_cm', '<f8'), ('start', '<f8', (3,)), ('end', '<f8', (3,)),
    ('initial_center_cm', '<f8', (3,)), ('initial_axis_physical', '<f8', (3,)),
    ('initial_quality_physical', '<f8'), ('initial_relative_spread', '<f8'),
    ('pca_dir_standardized', '<f8', (3,)), ('pca_quality_standardized', '<f8'),
    ('pca_variance_standardized', '<f8'),
    ('pca_center_physical_cm', '<f8', (3,)), ('pca_dir_physical', '<f8', (3,)),
    ('pca_quality_physical', '<f8'), ('pca_variance_physical_cm2', '<f8', (3,)),
    ('relative_spread_physical', '<f8'),
    ('n_cluster_hits', '<i8'), ('n_selected_hits', '<i8'), ('a2a', '?'),
])


def extended_dtype(original, extra):
    names = set(original.names or ())
    if names.intersection(name for name, *_ in extra):
        raise ValueError('Input fields collide with output bookkeeping fields')
    return np.dtype([(name, original.fields[name][0]) for name in original.names] + extra)


def copy_records(records, dtype):
    result = np.zeros(len(records), dtype=dtype)
    for name in records.dtype.names:
        result[name] = records[name]
    return result


def linked_rows(f, source, target, row):
    base = f'{source}/ref/{target}'
    region = f[f'{base}/ref_region'][row]
    start, stop = int(region['start']), int(region['stop'])
    refs = f[f'{base}/ref']
    if not 0 <= start <= stop <= len(refs):
        raise ValueError(f'Invalid reference region: {base}, row {row}')
    links = refs[start:stop]
    if links.ndim != 2 or links.shape[1] != 2:
        raise ValueError(f'Unsupported reference shape: {base}')
    rows = links[links[:, 0] == row, 1].astype(np.int64)
    if stop > start and not len(rows):
        raise ValueError(f'Unsupported reference column ordering: {base}')
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f'Duplicate references: {base}, row {row}')
    if np.any((rows < 0) | (rows >= len(f[f'{target}/data']))):
        raise ValueError(f'Out-of-bounds references: {base}, row {row}')
    return rows


def take_rows(dataset, rows):
    unique, inverse = np.unique(rows, return_inverse=True)
    return dataset[unique][inverse]


class TrackOutputWriter:
    def __init__(self, output_file, input_file, settings):
        self.source = h5py.File(input_file, 'r')
        self.file = None
        try:
            required = {'id', 'x', 'y', 'z', 'Q', 't_drift',
                        'io_group', 'io_channel', 'chip_id', 'channel_id'}
            hit_dtype = self.source[f'{HITS}/data'].dtype
            missing = required - set(hit_dtype.names)
            if missing:
                raise ValueError(f'Missing required hit fields: {sorted(missing)}')
            # Refuse accidental overwrite, including overwriting the source.
            self.file = h5py.File(output_file, 'x')
            self.file.attrs['schema_version'] = '2.0'
            self.file.attrs['status'] = 'incomplete'
            meta = self.file.create_group('metadata')
            meta.attrs['input_file'] = str(Path(input_file).resolve())
            stat = Path(input_file).stat()
            meta.attrs['input_size_bytes'] = stat.st_size
            meta.attrs['input_mtime_ns'] = stat.st_mtime_ns
            meta.attrs['python_version'] = platform.python_version()
            meta.attrs['numpy_version'] = np.__version__
            meta.attrs['h5py_version'] = h5py.__version__
            for key, value in settings.items():
                meta.attrs[key] = value
            for name in ['track_selection_fsd.py', 'efield.py', 'track_output_hdf5.py']:
                path = Path(__file__).with_name(name)
                meta.attrs[f'{name}_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            for name in ['run_info', 'geometry_info', HITS, 'combined/t0']:
                if name in self.source:
                    group = meta.require_group(name)
                    for key, value in self.source[name].attrs.items():
                        group.attrs[key] = value
            self.create('events/hits', extended_dtype(hit_dtype, [
                ('file_id', '<i8'), ('source_hit_row', '<i8'), ('source_event_row', '<i8'),
                ('selected_track_id', '<i8'), ('projected_xyz_cm', '<f8', (3,)),
                ('packet_link_count', '<i8'),
            ]))
            self.create('events/data', extended_dtype(self.source['charge/events/data'].dtype, [
                ('file_id', '<i8'), ('source_event_row', '<i8'), ('hit_start', '<i8'), ('hit_stop', '<i8'),
                ('n_accepted_tracks', '<i8'),
            ]))
            self.create('tracks/data', TRACK_DTYPE)
            self.create('events/hit_packet_refs', np.dtype([('file_id', '<i8'), ('hit_row', '<i8'), ('packet_row', '<i8')]))
            self.optional_tables = {}
            for output, source, parent in [
                ('events/t0', 'combined/t0', 'charge/events'),
                ('events/ext_trigs', 'charge/ext_trigs', 'charge/events'),
                ('events/packets', 'charge/packets', HITS),
            ]:
                available = (f'{source}/data' in self.source and
                             f'{parent}/ref/{source}/ref' in self.source and
                             f'{parent}/ref/{source}/ref_region' in self.source)
                original = self.source[f'{source}/data'].dtype if available else np.dtype([])
                self.create(output, extended_dtype(original, [('file_id', '<i8'), ('source_row', '<i8'), ('source_event_row', '<i8')]))
                self.file[output].attrs['available'] = available
                self.optional_tables[output] = (source, parent, available)
            self.file['events/hits'].attrs['position_units'] = 'cm'
            self.file['events/hits'].attrs['t_drift_units'] = 'CRS ticks; period in sources/<file_id>/metadata/run_info/crs_ticks'
            self.file['events/hits'].attrs['charge_units'] = 'source FLOW Q/Q_raw units; see copied calibration attributes'
            self.file['events/hits'].attrs['pixel_key'] = 'file_id, source_event_row, io_group, io_channel, chip_id, channel_id'
            self.file['events/hits'].attrs['selected_track_id_unselected'] = -1
            self.file['events/hits'].attrs['projected_xyz_cm_definition'] = 'Legacy endpoint-line projection, not truth or physical PCA projection; NaN for unselected hits'
            self.file['tracks/data'].attrs['pca_standardized_definition'] = 'Legacy PCAs(): per-coordinate StandardScaler, retained hits; quality used in final acceptance'
            self.file['tracks/data'].attrs['pca_physical_definition'] = 'PCA(3) on retained xyz in cm; diagnostic only; axis sign does not indicate travel direction'
            source_meta = self.file.require_group('sources/0')
            self.file.copy(meta, source_meta, name='metadata')
            source_meta.attrs['status'] = 'incomplete'
            for output, (_, _, available) in self.optional_tables.items():
                source_meta.attrs[output + '_available'] = available
        except Exception:
            self.close(False)
            raise

    def create(self, name, dtype):
        self.file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                 chunks=True, compression='gzip')

    def append(self, name, values):
        dataset = self.file[name]
        start = len(dataset)
        dataset.resize(start + len(values), axis=0)
        dataset[start:] = values
        return start

    def append_event(self, event_row, manager_hits, selected_track_id, projected_xyz, tracks):
        rows = np.sort(linked_rows(self.source, 'charge/events', HITS, event_row))
        original = take_rows(self.source[f'{HITS}/data'], rows)
        # Match manager order to actual source rows by explicit ID, never by row=ID.
        ids = original['id']
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate hit IDs prevent an unambiguous manager/source mapping')
        lookup = {int(identity): i for i, identity in enumerate(ids)}
        hits = copy_records(original, self.file['events/hits'].dtype)
        hits['source_hit_row'] = rows
        hits['source_event_row'] = event_row
        hits['selected_track_id'] = -1
        hits['projected_xyz_cm'] = np.nan
        hits['packet_link_count'] = -1
        seen = set()
        for i in np.flatnonzero(selected_track_id >= 0):
            if np.ma.is_masked(manager_hits['id'][i]):
                raise ValueError('Selection includes a masked FLOW reference')
            destination = lookup[int(manager_hits['id'][i])]
            if destination in seen:
                raise ValueError('Selection contains the same source hit more than once')
            seen.add(destination)
            hits['selected_track_id'][destination] = selected_track_id[i]
            hits['projected_xyz_cm'][destination] = projected_xyz[i]
        hit_start = len(self.file['events/hits'])
        for output, (source, parent, available) in self.optional_tables.items():
            if not available:
                continue
            if output == 'events/packets':
                links = [linked_rows(self.source, parent, source, int(row)) for row in rows]
                hits['packet_link_count'] = [len(link) for link in links]
                source_rows = np.unique(np.concatenate(links)) if links else np.array([], dtype=np.int64)
                packet_start = len(self.file[output])
                packet_lookup = {int(row): packet_start + i for i, row in enumerate(source_rows)}
                refs = [(0, hit_start + i, packet_lookup[int(row)]) for i, link in enumerate(links) for row in link]
                self.append('events/hit_packet_refs', np.array(refs, dtype=self.file['events/hit_packet_refs'].dtype))
            else:
                source_rows = linked_rows(self.source, parent, source, event_row)
            records = copy_records(take_rows(self.source[f'{source}/data'], source_rows), self.file[output].dtype)
            records['source_row'] = source_rows
            records['source_event_row'] = event_row
            self.append(output, records)
        self.append('events/hits', hits)
        event = copy_records(self.source['charge/events/data'][event_row:event_row + 1], self.file['events/data'].dtype)
        event['source_event_row'] = event_row
        event['hit_start'], event['hit_stop'] = hit_start, hit_start + len(hits)
        event['n_accepted_tracks'] = len(tracks)
        self.append('events/data', event)
        data = np.zeros(len(tracks), dtype=TRACK_DTYPE)
        for i, track in enumerate(tracks):
            for name, value in track.items():
                data[name][i] = value
        self.append('tracks/data', data)

    def close(self, complete):
        if self.file is not None:
            self.file.attrs['status'] = 'complete' if complete else 'incomplete'
            if 'sources/0' in self.file:
                self.file['sources/0'].attrs['status'] = 'complete' if complete else 'incomplete'
            self.file.close()
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(exc_type is None)
