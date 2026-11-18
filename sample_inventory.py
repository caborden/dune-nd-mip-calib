"""Inventory saved track selections and muon campaigns without reading hit arrays."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import h5py


SKIP = {'parts', 'analysis', 'comparisons', 'data_quality', 'jobs', 'archive',
        '.git', '.history', '.ipynb_checkpoints', '__pycache__', 'output', 'logs'}


def read_json(path):
    return json.loads(path.read_text())


def atomic_write(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def scan(base):
    """Avoid production intermediates, individual parts and duplicate symlink trees."""
    for directory, dirs, files in os.walk(base):
        dirs[:] = sorted(name for name in dirs if name not in SKIP)
        for name in sorted(files):
            path = Path(directory) / name
            if name == 'sample.json':
                yield path
            elif path.suffix in {'.h5', '.hdf5'} and any(
                    word in name.lower() for word in ('tracks', 'track-selection', 'pilot')):
                yield path


def selection_metadata(path):
    if not path.is_file():
        return {'status': 'missing'}
    with h5py.File(path, 'r') as source:
        if 'tracks/data' not in source or 'sources' not in source:
            return None
        cuts = set()
        for group in source['sources'].values():
            metadata = group.get('metadata')
            value = None if metadata is None else metadata.attrs.get('min_length_cm')
            cuts.add('unknown' if value is None else f'>{float(value):g} cm')
        return {'status': str(source.attrs.get('status', 'unknown')),
                'schema': str(source.attrs.get('schema_version', 'unknown')),
                'tracks': len(source['tracks/data']),
                'cut': ', '.join(sorted(cuts)) or 'unknown'}


def detector_settings(directory):
    settings = []
    for path in sorted(directory.glob('detector*.yaml')):
        values = []
        for key in ('periodic_reset_cycles', 'adc_busy_delay', 'adc_hold_delay'):
            match = re.search(r'^' + key + r':\s*(\d+)\s*(?:#.*)?$',
                              path.read_text(), re.MULTILINE)
            values.append(f'{key}={match[1]}' if match else f'{key}=unspecified/inherited')
        settings.append(f'{path}: ' + ', '.join(values))
    return settings


def collect(root, config):
    entries = {}
    warnings = []
    campaigns = []

    def entry(path):
        path = path.resolve()
        return entries.setdefault(path, {'aliases': set(), 'kinds': set(), 'records': set(),
                                         'campaigns': []})

    bases = [root]
    production = config.get('production_root')
    if production:
        bases.extend([Path(production) / 'campaigns', Path(production) / 'track-selection'])
    for base in bases:
        if not base.is_dir():
            warnings.append(f'Search directory missing: {base}')
        for path in scan(base):
            if path.name != 'sample.json':
                entry(path)
                continue
            try:
                record = read_json(path)
                for version, details in record.get('selection_versions', {}).items():
                    item = entry(path.parent / details['path'])
                    item['aliases'].add(f"{record.get('id', path.parent.name)} / {version}")
                    item['kinds'].add(record.get('kind', 'unknown'))
                    item['records'].add(str(path))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.append(f'{path}: {exc}')

    for path in config.get('selections', []):
        entry(Path(path))

    jobs = config.get('campaign_jobs')
    if jobs:
        jobs = Path(jobs)
        if not jobs.is_dir():
            warnings.append(f'Campaign directory missing: {jobs}')
        directories = {p.parent for pattern in ('*/campaign.json', '*/summary.json')
                       for p in jobs.glob(pattern)}
        for directory in sorted(directories):
            summary_path = directory / 'summary.json'
            if not summary_path.exists():
                campaigns.append(f'{directory.name} — no summary published; {directory}')
                continue
            try:
                summary = read_json(summary_path)
                campaigns.append(f"{directory.name} — {summary.get('status', 'unknown')}; "
                                 f"{summary.get('total_tracks', 'unknown')} tracks; {summary_path}")
                if summary.get('output'):
                    item = entry(directory / summary['output'])
                    item['kinds'].add('mc')
                    item['aliases'].add(directory.name)
                    item['records'].add(str(summary_path))
                    item['campaigns'].extend(detector_settings(directory))
            except (OSError, ValueError, TypeError) as exc:
                warnings.append(f'{summary_path}: {exc}')

    selections = []
    for path, item in sorted(entries.items()):
        try:
            metadata = selection_metadata(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            metadata = {'status': 'unreadable'}
            warnings.append(f'{path}: {exc}')
        if metadata is None:
            continue
        item.update(metadata)
        item['path'] = str(path)
        selections.append(item)
    return selections, campaigns, warnings


def markdown(root, config, selections, campaigns, warnings):
    lines = ['# NERSC track sample inventory', '',
             f'Generated {datetime.now(timezone.utc).isoformat()}.', '',
             f'Analysis root `{root}`.', '',
             'One entry per resolved selection path. Aliases share the same file. '
             'Track counts and cuts come from HDF5 metadata. Unknown cuts are not inferred '
             'from names. Electronics values come from frozen campaign YAMLs. '
             'Unspecified values may be inherited from the simulator.', '',
             'Includes registered selections, discovered merged selections and pilots. '
             'Individual parts, archived selections and simulation intermediates are excluded.', '',
             f"Campaign jobs `{config.get('campaign_jobs', 'not configured')}`.", '',
             f"Production root `{config.get('production_root', 'not configured')}`.", '',
             f'## Selections ({len(selections)})', '']
    for item in selections:
        path = item['path']
        lines.extend([f'### {Path(path).name}', '', f'- Path `{path}`',
                      f"- Kind {', '.join(sorted(item['kinds'])) or 'unknown'}",
                      f"- Status {item['status']}",
                      f"- Tracks {item.get('tracks', 'unknown')}",
                      f"- Selection cut {item.get('cut', 'unknown')}",
                      f"- Schema {item.get('schema', 'unknown')}"])
        lines.extend(f'- Alias `{alias}`' for alias in sorted(item['aliases']))
        lines.extend(f'- Record `{record}`' for record in sorted(item['records']))
        lines.extend(f'- Detector settings `{settings}`' for settings in item['campaigns'])
        lines.append('')
    lines.extend(['## Campaign summaries', ''])
    lines.extend(f'- {campaign}' for campaign in campaigns)
    lines.extend(['', '## Read errors and missing search directories', ''])
    lines.extend(f'- {warning}' for warning in warnings)
    if not warnings:
        lines.append('None.')
    return '\n'.join(lines) + '\n'


def generate(root, campaign_jobs=None, production_root=None, selections=()):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Lock covers discovery as well as publication, preventing stale concurrent scans.
    with (root / '.sample-inventory.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config_path = root / 'inventory-config.json'
        config = read_json(config_path) if config_path.exists() else {}
        if campaign_jobs is not None:
            config['campaign_jobs'] = str(Path(campaign_jobs).resolve())
        if production_root is not None:
            config['production_root'] = str(Path(production_root).resolve())
        config['selections'] = sorted(set(config.get('selections', [])) |
                                      {str(Path(path).resolve()) for path in selections})
        samples, campaigns, warnings = collect(root, config)
        atomic_write(config_path, json.dumps(config, indent=2) + '\n')
        output = root / 'sample-inventory.md'
        atomic_write(output, markdown(root, config, samples, campaigns, warnings))
    print(f'Inventory: {output} ({len(samples)} selections, {len(warnings)} warnings)', flush=True)
    return output


def refresh_after_selection(output):
    """Best-effort refresh for standalone selections and merged data batches."""
    output = Path(output).resolve()
    if 'parts' in output.parts or output.suffix == '.partial':
        return
    root = os.environ.get('FSD_ANALYSIS_ROOT')
    if not root:
        root = next((parent for parent in output.parents if parent.name == 'fsdcube'), None)
    if root is None:
        return
    try:
        generate(root, selections=[output])
    except Exception as exc:
        print(f'WARNING: selection completed, but inventory refresh failed: {exc}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=os.environ.get('FSD_ANALYSIS_ROOT'),
                        help='analysis root; defaults to FSD_ANALYSIS_ROOT')
    parser.add_argument('--campaign-jobs', help='persist the directory containing campaign records')
    parser.add_argument('--production-root', help='persist the MC production directory')
    parser.add_argument('--selection', action='append', default=[], help='additional selection path')
    args = parser.parse_args()
    if not args.root:
        parser.error('provide --root or set FSD_ANALYSIS_ROOT')
    generate(args.root, args.campaign_jobs, args.production_root, args.selection)


if __name__ == '__main__':
    main()
