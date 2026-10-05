"""Run selected analysis modules for a registered sample or comparison."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

import h5py

from .config import make_settings, SEGMENT_KEYS
from .dqdx import plot_segments
from .segments import build_segments
from .io import (read_json, write_json, fingerprint, file_identity, code_hashes,
                 runtime_versions, safe_name)

# Register future modules here, with explicit shared-table dependencies.
MODULES = {
    'segments': dict(dependencies=[], code=['analysis/segments.py', 'pixel_dqdx/segments.py'],
                     settings=SEGMENT_KEYS, output='shared/segments.hdf5'),
    'dqdx': dict(dependencies=['segments'], code=['analysis/dqdx.py', 'pixel_dqdx/dqdx.py'],
                 settings={'bins', 'hist_range', 'fit_range', 'min_fit_entries'}, output='dqdx'),
}


def resolve_target(root, target):
    kind, name = target['kind'], safe_name(target['id'])
    if kind not in {'sample', 'comparison'}:
        raise ValueError('Target kind must be sample or comparison')
    directory = root/('samples' if kind == 'sample' else 'comparisons')/name
    if kind == 'sample':
        registry = read_json(directory/'sample.json')
        references = {registry['kind']: dict(id=name, selection_version=target.get('selection_version', 'v1'))}
    else:
        references = read_json(directory/'comparison.json')['samples']
    if not references or not set(references) <= {'data', 'mc'}:
        raise ValueError('Registries must reference data, MC, or both')
    samples = []
    for sample_kind in ['data', 'mc']:
        if sample_kind not in references:
            continue
        reference = references[sample_kind]
        sample_dir = root/'samples'/safe_name(reference['id'])
        record = read_json(sample_dir/'sample.json')
        if record['kind'] != sample_kind:
            raise ValueError('Sample kind disagrees with registry reference')
        version = safe_name(reference.get('selection_version', 'v1'))
        path = (sample_dir/record['selection_versions'][version]['path']).resolve(strict=True)
        with h5py.File(path, 'r') as source:
            if source.attrs.get('status') != 'complete' or source.attrs.get('schema_version') != '2.0':
                raise ValueError(f'Expected complete schema-2 selection: {path}')
        title = 'FSD Data Fit' if sample_kind == 'data' else 'FSD Simulation Fit'
        samples.append((sample_kind, title, path))
    return directory, samples


def dependency_order(requested):
    order = []
    def add(name):
        if name not in MODULES:
            raise ValueError(f'Unknown module {name!r}; available: {", ".join(MODULES)}')
        for dependency in MODULES[name]['dependencies']:
            add(dependency)
        if name not in order:
            order.append(name)
    for name in requested:
        add(name)
    if not order:
        raise ValueError('Request at least one module')
    return order


def artifact_identities(path):
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    return [file_identity(p) for p in files]


def run(config_path, root, run_id, modules=None, resume=False):
    config = read_json(config_path)
    unexpected = set(config) - {'target', 'segments', 'modules', 'run_modules'}
    if unexpected:
        raise ValueError(f'Unknown configuration keys: {unexpected}')
    root = Path(root).resolve()
    target_dir, samples = resolve_target(root, config['target'])
    segment_settings = config.get('segments', {})
    if not set(segment_settings) <= SEGMENT_KEYS:
        raise ValueError('Only segmentation options belong in segments settings')
    module_settings = config.get('modules', {})
    for name, options in module_settings.items():
        if name == 'segments':
            raise ValueError('Configure shared segment options under the top-level segments key')
        if name not in MODULES or not set(options) <= MODULES[name]['settings']:
            raise ValueError(f'Unknown module/options: {name}')
    settings = make_settings(**segment_settings, **module_settings.get('dqdx', {}))
    requested = modules if modules is not None else config.get('run_modules', ['dqdx'])
    if requested == ['all']:
        requested = list(MODULES)
    order = dependency_order(requested)
    inputs = {kind: file_identity(path) for kind, _, path in samples}
    resolved = dict(target=config['target'], inputs=inputs, settings=settings)
    directory = target_dir/'analysis'/safe_name(run_id)
    manifest_path = directory/'manifest.json'
    if directory.exists() and not resume:
        raise FileExistsError(f'{directory} exists; use --resume or a new run ID')
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive writer lock; a killed process requires checking the job before removing it.
    lock = directory/'.writer.lock'
    with lock.open('x') as handle:
        handle.write(str(__import__('os').getpid()))
    try:
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get('format') != 'modular-analysis-v1':
                raise ValueError('Imported historical run: choose a new run ID')
            if manifest['resolved']['inputs'] != inputs or manifest['resolved']['target'] != config['target']:
                raise ValueError('Run inputs/target changed: choose a new run ID')
        else:
            manifest = dict(format='modular-analysis-v1', stages={},
                            created_utc=datetime.now(timezone.utc).isoformat(), resolved=resolved)
        runtime = runtime_versions()
        signatures = {}
        # Reject stale completed stages before changing any historical run record.
        for name in order:
            module = MODULES[name]
            signatures[name] = fingerprint(dict(inputs=inputs, runtime=runtime,
                settings={key: settings[key] for key in module['settings']},
                code=code_hashes(module['code'] + ['analysis/config.py', 'analysis/io.py', 'analysis/run.py']),
                dependencies={dep: signatures[dep] for dep in module['dependencies']}))
            prior = manifest['stages'].get(name, {})
            if prior.get('status') == 'complete':
                if prior['signature'] != signatures[name]:
                    raise ValueError(f'{name} settings/code changed: choose a new run ID')
                if prior['artifacts'] != artifact_identities(directory/module['output']):
                    raise ValueError(f'{name} outputs changed or are missing: choose a new run ID')
        manifest.update(status='running', requested_modules=requested, resolved=resolved,
                        runtime=runtime, source_hashes=code_hashes())
        write_json(directory/'config.json', config)
        write_json(manifest_path, manifest)
        started = True
        for name in order:
            module = MODULES[name]
            signature = signatures[name]
            prior = manifest['stages'].get(name, {})
            destination = directory/module['output']
            if prior.get('status') == 'complete':
                print(f'Reuse {name}: {destination}', flush=True)
                continue
            if destination.exists():
                raise FileExistsError(f'Unrecorded output requires inspection: {destination}')
            stage = dict(status='running', signature=signature)
            manifest['stages'][name] = stage
            write_json(manifest_path, manifest)
            temporary = directory/f'.{name}-{uuid.uuid4().hex}.tmp'
            temporary.mkdir()
            print(f'Run {name}: {destination}', flush=True)
            try:
                if name == 'segments':
                    build_segments(samples, temporary/'segments.hdf5',
                                   {key: settings[key] for key in SEGMENT_KEYS})
                    destination.parent.mkdir(exist_ok=True)
                    (temporary/'segments.hdf5').rename(destination)
                    temporary.rmdir()
                else:
                    plot_segments(directory/'shared/segments.hdf5', temporary, settings)
                    temporary.rename(destination)
                # Input files must remain unchanged during extraction/plotting.
                if inputs != {kind: file_identity(path) for kind, _, path in samples}:
                    raise ValueError('Input changed during the run')
                stage.update(status='complete', artifacts=artifact_identities(destination))
                write_json(manifest_path, manifest)
            except Exception as exc:
                stage.update(status='failed', error=str(exc))
                # Keep failed artifacts for inspection; retry uses a fresh staging directory.
                if destination.exists():
                    destination.rename(directory/f'.{name}-{uuid.uuid4().hex}.failed')
                if temporary.exists():
                    temporary.rename(temporary.with_suffix('.failed'))
                raise
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        manifest.pop('error', None)
        write_json(manifest_path, manifest)
        return directory
    except Exception as exc:
        if locals().get('started'):
            manifest.update(status='failed', error=str(exc))
            write_json(manifest_path, manifest)
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--root', required=True, help='fsdcube root containing samples/ and comparisons/')
    parser.add_argument('--run-id', required=True, help='e.g. run-002; existing outputs require --resume')
    parser.add_argument('--modules', nargs='+', choices=[*MODULES, 'all'])
    parser.add_argument('--resume', action='store_true')
    args = vars(parser.parse_args())
    args['config_path'] = args.pop('config')
    print(f'Complete: {run(**args)}')


if __name__ == '__main__':
    main()
