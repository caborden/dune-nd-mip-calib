"""Migrate the existing FSD-cube outputs without rewriting HDF5/provenance files.

Default: print a plan. --apply moves files and leaves relative compatibility links.
--check validates the migrated selections, analyses, links and preserved bytes.
Run only after writers using these directories have stopped.
"""
import argparse
import hashlib
import os
from pathlib import Path

import h5py

from analysis.io import read_json, write_json

DATA_ID = 'data-Reflow_FSDCube_v3.binary-cosmics-2026_02'
MC_ID = 'mc-FSDCubeSim_v1_prc256'
COMPARISON_ID = 'reflow-v3-feb2026_vs_fsdcube-sim-v1-prc256'
DATA_FILE = 'Reflow_FSDCube_v3.binary-cosmics-2026_02.track-selection.hdf5'
MC_FILE = 'FSDCubeSim_v1_prc256.track-selection.hdf5'
SPECS = [(DATA_ID, 'data', 'run-20261003-151120', DATA_FILE, 'pilot.hdf5'),
         (MC_ID, 'mc', 'mc-pilot', MC_FILE, 'FSDCubeSim_v1_prc256.pilot.track-selection.hdf5')]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def complete_selection(path):
    with h5py.File(path, 'r') as f:
        if f.attrs.get('status') != 'complete' or f.attrs.get('schema_version') != '2.0':
            raise ValueError(f'Incomplete/unsupported selection: {path}')
        return dict(sources=len(f['sources']), tracks=len(f['tracks/data']),
                    events=len(f['events/data']), hits=len(f['events/hits']))


def layout(root):
    base = root/'fsdcube'
    moves = []
    for sample, _, legacy, _, pilot in SPECS:
        destination = base/'samples'/sample/'selection/v1'
        moves += [(root/'fsdcube-charge-splitting'/legacy, destination),
                  (destination/'full_parts', destination/'parts'),
                  (destination/'pilot_parts', destination/'pilot/parts'),
                  (destination/pilot, destination/'pilot'/pilot)]
    for old, target in [('dqdx_data_v1', base/'samples'/DATA_ID),
                        ('dqdx_data_mc_v1', base/'comparisons'/COMPARISON_ID)]:
        destination = target/'analysis/run-001/dqdx'
        moves += [(base/old, destination),
                  (destination/'segments.hdf5', destination.parent/'shared/segments.hdf5')]
    archive = 'FSDCubeSim_v1_prc256-20261003-191727'
    if (root/'fsdcube-charge-splitting'/archive).exists() or (base/'archive'/archive).exists():
        moves.append((root/'fsdcube-charge-splitting'/archive, base/'archive'/archive))
    return base, moves


def origin(path, previous):
    """Map a planned destination back to its on-disk source during preflight."""
    for source, destination in reversed(previous):
        if source.is_symlink() or (not source.exists() and destination.exists()):
            continue  # This move already happened; use its current destination.
        if path.is_relative_to(destination):
            path = source/path.relative_to(destination)
    return path


def preflight(root, base, moves, validate_analyses=True):
    for index, (source, destination) in enumerate(moves):
        if source.is_symlink():
            if source.resolve() != destination.resolve() or not destination.exists():
                raise FileExistsError(f'Unexpected compatibility link: {source}')
            continue
        current_source = origin(source, moves[:index])
        current_destination = origin(destination, moves[:index])
        if current_source.exists() and current_destination.exists():
            raise FileExistsError(f'Destination already exists: {destination}')
        if not current_source.exists() and not current_destination.exists():
            raise FileNotFoundError(f'Missing migration input: {current_source}')
    for sample, kind, legacy, filename, pilot in SPECS:
        old = root/'fsdcube-charge-splitting'/legacy
        current = old if old.exists() else base/'samples'/sample/'selection/v1'
        totals = complete_selection(current/filename)
        parts = current/('full_parts' if (current/'full_parts').exists() else 'parts')
        manifest = read_json(parts/'batch_manifest.json')
        if manifest['status'] != 'complete' or any(row['status'] != 'complete' for row in manifest['files']):
            raise ValueError(f'Batch still incomplete: {parts}')
        if len(manifest['files']) != len(manifest['inputs']) or totals['sources'] != len(manifest['inputs']):
            raise ValueError(f'Merged source count disagrees with batch: {current}')
        for row in manifest['files']:
            complete_selection(parts/Path(row['output']).name)
        pilot_path = current/pilot if (current/pilot).exists() else current/'pilot'/pilot
        complete_selection(pilot_path)
    for old, target in [('dqdx_data_v1', base/'samples'/DATA_ID),
                        ('dqdx_data_mc_v1', base/'comparisons'/COMPARISON_ID)]:
        if not validate_analyses:
            continue
        current = base/old if (base/old).exists() else target/'analysis/run-001/dqdx'
        record = read_json(current/'analysis.json')
        if record['status'] != 'complete':
            raise ValueError(f'Analysis still incomplete: {current}')
        segment_file = current/'segments.hdf5'
        if not segment_file.exists():
            segment_file = current.parent/'shared/segments.hdf5'
        with h5py.File(segment_file, 'r') as f:
            if f.attrs.get('status') != 'complete':
                raise ValueError(f'Incomplete segments: {segment_file}')
            if len(f['segments/data']) != sum(s['audit']['segments'] for s in record['samples']):
                raise ValueError(f'Segment count disagrees with analysis: {current}')
    # Never overwrite existing registry records with unrelated contents.
    for path, record in registries(base).items():
        if path.exists() and read_json(path) != record:
            raise FileExistsError(f'Registry differs from migration definition: {path}')


def registries(base):
    records = {}
    for sample, kind, _, filename, _ in SPECS:
        records[base/'samples'/sample/'sample.json'] = dict(
            id=sample, kind=kind, selection_versions={'v1': {'path': f'selection/v1/{filename}'}})
    records[base/'comparisons'/COMPARISON_ID/'comparison.json'] = dict(
        id=COMPARISON_ID, samples={kind: dict(id=sample, selection_version='v1')
                                 for sample, kind, _, _, _ in SPECS})
    return records


def compatibility_link(source, destination):
    if source.is_symlink():
        if source.resolve() != destination.resolve():
            raise FileExistsError(f'Wrong compatibility link: {source}')
        return
    if source.exists():
        raise FileExistsError(f'Cannot replace compatibility path: {source}')
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(os.path.relpath(destination, source.parent), target_is_directory=destination.is_dir())


def import_analysis(directory):
    record = read_json(directory/'dqdx/analysis.json')
    write_json(directory/'config.json', dict(imported=True, settings=record['settings'],
               historical_inputs=[s['input'] for s in record['samples']]))
    write_json(directory/'manifest.json', dict(format='imported-analysis-v1', status='complete',
               stages={'segments': {'status': 'complete', 'path': 'shared/segments.hdf5'},
                       'dqdx': {'status': 'complete', 'path': 'dqdx/analysis.json'}},
               source_hashes=record.get('source_hashes', {}),
               note='Original provenance is retained verbatim. The runner can adopt and refresh this run.'))


def check(root):
    base, moves = layout(root)
    preflight(root, base, moves, validate_analyses=False)
    record = read_json(base/'relocation.json')
    if record['status'] != 'complete':
        raise ValueError('Migration has not completed; rerun --apply')
    for source, destination in moves:
        if not source.is_symlink() or source.resolve() != destination.resolve():
            raise ValueError(f'Missing/wrong compatibility link: {source}')
    histories = []
    for group in ['samples', 'comparisons']:
        for path in (base/group).glob('*/analysis/*/.history/*/history.json'):
            histories.extend(read_json(path).get('previous_outputs', []))
    for old, expected in record['preserved_files'].items():
        path = root/old
        canonical = path.resolve()
        candidates = [path]
        for history in histories:
            source = Path(history['source'])
            if canonical.is_relative_to(source):
                candidates.append(Path(history['destination'])/canonical.relative_to(source))
        if not any(candidate.is_file() and digest(candidate) == expected for candidate in candidates):
            raise ValueError(f'Historical bytes changed or are missing: {path}')
    for path, expected in registries(base).items():
        if read_json(path) != expected:
            raise ValueError(f'Invalid registry: {path}')
    print(f'Validated {len(record["preserved_files"])} preserved files and {len(moves)} compatibility links.')
    for sample, _, _, filename, _ in SPECS:
        print(sample, complete_selection(base/'samples'/sample/'selection/v1'/filename))
    return record


def migrate(root, apply=False):
    root = Path(root).resolve()
    base, moves = layout(root)
    record_path = base/'relocation.json'
    completed = record_path.exists() and read_json(record_path).get('status') == 'complete'
    preflight(root, base, moves, validate_analyses=not completed)
    for source, destination in moves:
        print(f'{source.relative_to(root)} -> {destination.relative_to(root)}')
    if not apply:
        print('Dry run: no files changed. Use --apply to move and leave compatibility links.')
        return
    if record_path.exists():
        record = read_json(record_path)
        if record['status'] == 'complete':
            return check(root)
    else:
        preserved = {}
        # Capture every original regular file, including logs and notebook checkpoints.
        for top in [root/'fsdcube-charge-splitting', base/'dqdx_data_v1', base/'dqdx_data_mc_v1']:
            if top.exists() and not top.is_symlink():
                for path in top.rglob('*'):
                    if path.is_file() and not path.is_symlink():
                        preserved[str(path.relative_to(root))] = digest(path)
        record = dict(format='fsdcube-relocation-v1', status='running', preserved_files=preserved,
                      moves=[dict(source=str(s.relative_to(root)), destination=str(d.relative_to(root)))
                             for s, d in moves], completed_moves=[])
        write_json(record_path, record)
    for source, destination in moves:
        if source.is_symlink() or (not source.exists() and destination.exists()):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        record['completed_moves'].append(str(source.relative_to(root)))
        write_json(record_path, record)
    # All moves finish before compatibility links are created.
    for source, destination in moves:
        compatibility_link(source, destination)
    for path, registry in registries(base).items():
        write_json(path, registry)
    for directory in [base/'samples'/DATA_ID/'analysis/run-001',
                      base/'comparisons'/COMPARISON_ID/'analysis/run-001']:
        import_analysis(directory)
    record['status'] = 'complete'
    write_json(record_path, record)
    return check(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='parent containing fsdcube/ and fsdcube-charge-splitting/')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    check(root) if args.check else migrate(root, args.apply)


if __name__ == '__main__':
    main()
