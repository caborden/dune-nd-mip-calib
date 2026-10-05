"""Shared selected-hit segmentation; event-streaming extraction."""
from collections import Counter
import json
from pathlib import Path
import h5py
import numpy as np
from pixel_dqdx.segments import segment_track_hits, is_through_going, fiducialize_hits

SEGMENT_DTYPE = np.dtype([
    ('sample_id', 'i4'), ('file_id', 'i8'), ('source_event_row', 'i8'), ('track_id', 'i8'),
    ('group', 'S64'), ('dq', 'f8'), ('dx', 'f8'), ('dqdx', 'f8'), ('nhits', 'i8'),
    ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('t_drift', 'f8'),
    ('theta', 'f8'), ('phi', 'f8'),
])


def groups_for(hits, file_id, geometry, grouping):
    config = geometry.get('sources', {}).get(str(file_id), geometry)
    if 'groups' in config:
        groups = config['groups']
        used = set()
        for group in groups:
            if not group.get('io_groups') or used.intersection(group['io_groups']):
                raise ValueError('Geometry groups must have disjoint, nonempty io_groups')
            used.update(group['io_groups'])
            for axis in 'xyz':
                bounds = group['bounds'][axis]
                if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                    raise ValueError(f'Invalid physical bounds: {group["name"]}/{axis}')
            if len(group['name'].encode()) > 64:
                raise ValueError('Group name is too long')
        unknown = set(hits['io_group'].tolist()) - used
        if unknown:
            raise ValueError(f'Geometry lacks IO groups {unknown} for file_id={file_id}')
        return groups
    if grouping == 'legacy-tpc':
        if not set(hits['io_group'].tolist()) <= {1, 2, 3, 4}:
            raise ValueError('Legacy TPC map only supports IO groups 1..4')
        return [dict(name='tpc0', io_groups=[1]), dict(name='tpc1', io_groups=[2,3,4])]
    return [dict(name=f'io_group_{int(io)}', io_groups=[int(io)]) for io in np.unique(hits['io_group'])]


def automatic_bounds(source, file_id, margin):
    """Use only finite, nondegenerate stored outer detector bounds in hit cm."""
    path = f'sources/{file_id}/metadata/geometry_info'
    if path not in source or 'lar_detector_bounds' not in source[path].attrs:
        return None, 'stored detector bounds unavailable'
    values = np.asarray(source[path].attrs['lar_detector_bounds'], dtype=float)
    if values.shape != (2,3) or not np.isfinite(values).all():
        return None, 'stored detector bounds have unsupported shape/nonfinite values'
    low, high = values.min(axis=0), values.max(axis=0)
    if np.any(high-low <= 2*margin):
        return None, 'stored detector bounds are degenerate or narrower than twice the margin'
    return {axis:[float(low[i]),float(high[i])] for i,axis in enumerate('xyz')}, 'stored outer detector bounds'


def segment_sample(source_path, out, sample_id, config):
    audit = Counter()
    with h5py.File(source_path, 'r') as source:
        if source.attrs.get('schema_version') != '2.0' or source.attrs.get('status') != 'complete':
            raise ValueError('Analysis requires a complete schema-2 track-selection output')
        provenance = out.require_group(f'samples/{sample_id}')
        provenance.attrs['input_file'] = str(Path(source_path).resolve())
        stat = Path(source_path).stat()
        provenance.attrs['input_size_bytes'], provenance.attrs['input_mtime_ns'] = stat.st_size, stat.st_mtime_ns
        source.copy(source['sources'], provenance, name='sources')
        auto_geometry = {}
        face_policy = {}
        for file_id in source['sources']:
            if config['face_cuts'] == 'auto':
                bounds, reason = automatic_bounds(source,file_id,config['face_margin_cm'])
                auto_geometry[int(file_id)] = bounds
                face_policy[file_id] = dict(bounds=bounds,reason=reason)
        provenance.attrs['automatic_face_policy'] = json.dumps(face_policy)
        audit['sources_with_usable_auto_bounds'] = sum(value['bounds'] is not None for value in face_policy.values())
        audit['sources_without_usable_auto_bounds'] = sum(value['bounds'] is None for value in face_policy.values())
        # Read the merged file by event ranges, not a full hit-table load.
        for start in range(0, len(source['events/data']), 1000):
            for event in source['events/data'][start:start+1000]:
                audit['events'] += 1
                hits = source['events/hits'][int(event['hit_start']):int(event['hit_stop'])]
                if np.any(hits['file_id'] != event['file_id']) or np.any(hits['source_event_row'] != event['source_event_row']):
                    raise ValueError('Event range crosses source-file/event identity')
                hits = hits[hits['selected_track_id'] >= 0]
                for track_id in np.unique(hits['selected_track_id']):
                    track = hits[hits['selected_track_id'] == track_id]
                    audit['selected_tracks'] += 1
                    for group in groups_for(track, int(event['file_id']), config['geometry'], config['grouping']):
                        part = track[np.isin(track['io_group'], group['io_groups'])]
                        if not len(part):
                            continue
                        audit['track_groups'] += 1
                        xyz = np.column_stack([part[d] for d in 'xyz'])
                        good = np.isfinite(xyz).all(axis=1) & np.isfinite(part['Q']) & np.isfinite(part['t_drift'])
                        audit['nonfinite_selected_hits_excluded'] += int((~good).sum())
                        part, xyz = part[good], xyz[good]
                        if len(part) < 3:
                            audit['too_few_hits'] += 1
                            continue
                        if 'bounds' not in group and auto_geometry.get(int(event['file_id'])) is not None:
                            group = dict(group,bounds=auto_geometry[int(event['file_id'])])
                        if 'bounds' in group:
                            if config['require_through_going'] and not is_through_going(xyz, group['bounds'], config['face_margin_cm']):
                                audit['not_through_going'] += 1
                                continue
                            xyz, keep = fiducialize_hits(xyz, group['bounds'], config['face_margin_cm'])
                            part = part[keep]
                            if len(part) < 3:
                                audit['too_few_fiducial_hits'] += 1
                                continue
                        elif config['require_through_going']:
                            raise ValueError('--require-through-going requires explicit geometry bounds')
                        result = segment_track_hits(xyz, part['Q'], part['t_drift'],
                                                    config['segment_length_cm'], config['step_cm'], int(track_id))
                        n = len(result.get('dq', []))
                        if not n:
                            audit['groups_without_segments'] += 1
                            continue
                        values = np.zeros(n, dtype=SEGMENT_DTYPE)
                        values['sample_id'], values['file_id'] = sample_id, event['file_id']
                        values['source_event_row'], values['track_id'] = event['source_event_row'], track_id
                        values['group'] = group['name'].encode()
                        for key in ['dq', 'dx', 'nhits', 'x', 'y', 'z', 't_drift', 'theta', 'phi']:
                            values[key] = result[key]
                        values['dqdx'] = values['dq']/values['dx']
                        dataset = out['segments/data']
                        offset = len(dataset)
                        dataset.resize(offset+n, axis=0)
                        dataset[offset:] = values
                        audit['segments'] += n
        provenance.attrs['audit'] = json.dumps(dict(audit))
    return dict(audit)



def build_segments(samples, destination, config):
    """Write one shared table. Samples are (kind, title, input path) tuples."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(destination, 'x') as out:
        out.attrs['status'] = 'incomplete'
        out.attrs['settings'] = json.dumps(config)
        out.create_dataset('segments/data', shape=(0,), maxshape=(None,),
                           dtype=SEGMENT_DTYPE, chunks=True, compression='gzip')
        for kind, title, path in samples:
            sample_id = 0 if kind == 'data' else 1
            audit = segment_sample(path, out, sample_id, config)
            out[f'samples/{sample_id}'].attrs['sample_kind'] = kind
            out[f'samples/{sample_id}'].attrs['title'] = title
            records.append(dict(sample_id=sample_id, sample_kind=kind, title=title,
                                input=str(Path(path).resolve()), audit=audit))
        out.attrs['status'] = 'complete'
    return records
