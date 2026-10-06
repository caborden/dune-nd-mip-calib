"""Run selected analysis modules for a registered sample or comparison."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

import h5py

from .config import make_settings, SEGMENT_KEYS
from .dqdx import plot_segments, sample_title
from .hit_density import (plot_segments as plot_hit_density, make_settings as make_hit_density_settings,
                          SETTING_KEYS as HIT_DENSITY_KEYS)
from .segments import build_segments
from .run_state import components, upgrade_legacy, adopt_imported, start_history, publish
from .io import (read_json, write_json, fingerprint, file_identity, code_hashes,
                 runtime_versions, safe_name)

# Register future modules here, with explicit shared-table dependencies.
MODULES = {
    'segments': dict(dependencies=[], code=['analysis/segments.py', 'pixel_dqdx/segments.py'],
                     settings=SEGMENT_KEYS, output='shared/segments.hdf5'),
    'dqdx': dict(dependencies=['segments'], code=['analysis/dqdx.py', 'pixel_dqdx/dqdx.py'],
                 settings={'bins', 'hist_range', 'fit_range', 'min_fit_entries'}, output='dqdx'),
    'hit_density': dict(dependencies=['segments'], code=['analysis/hit_density.py'],
                        settings=HIT_DENSITY_KEYS, output='hit_density'),
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
        title = sample_title(sample_kind)
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


def run(config_path, root, run_id, modules=None, resume=False, rerun=None):
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
    density_settings = make_hit_density_settings(**module_settings.get('hit_density', {}))
    stage_settings = {
        'segments': {key: settings[key] for key in SEGMENT_KEYS},
        'dqdx': {key: settings[key] for key in MODULES['dqdx']['settings']},
        'hit_density': density_settings,
    }
    requested = modules if modules is not None else config.get('run_modules', ['dqdx'])
    if requested == ['all']:
        requested = list(MODULES)
    forced = set(MODULES) if rerun == ['all'] else set(rerun or [])
    dependency_order(list(forced)) if forced else None
    requested = list(dict.fromkeys([*requested, *sorted(forced)]))
    order = dependency_order(requested)
    resume = resume or bool(forced)
    inputs = {kind: file_identity(path) for kind, _, path in samples}
    resolved = dict(target=config['target'], inputs=inputs, settings=settings,
                    module_settings={'hit_density': density_settings})
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
        runtime = runtime_versions()
        if manifest_path.exists():
            original = read_json(manifest_path)
            if original.get('format') == 'imported-analysis-v1':
                manifest = adopt_imported(directory, original, resolved, runtime, stage_settings,
                                          MODULES, 'segments' in forced)
            elif original.get('format') == 'modular-analysis-v1':
                manifest = upgrade_legacy(original, MODULES)
                if (manifest['resolved']['inputs'] != inputs or
                        manifest['resolved']['target'] != config['target']) and 'segments' not in forced:
                    raise ValueError('Run inputs/target changed: choose a new run ID or use --rerun segments')
            else:
                raise ValueError('Unsupported run manifest')
        else:
            manifest = dict(format='modular-analysis-v1', stages={},
                            created_utc=datetime.now(timezone.utc).isoformat(), resolved=resolved)
        plans = {}
        # Complete preflight before changing any output or run record.
        for name in order:
            module = MODULES[name]
            prior = manifest['stages'].get(name, {})
            scientific = components(name, MODULES, inputs, runtime, stage_settings[name], plans)
            dep_changed = any(plans[dep]['execute'] for dep in module['dependencies'])
            execute = name in forced or dep_changed or prior.get('status') in {'stale', 'failed', 'running'} or not prior
            destination = directory/module['output']
            if not execute:
                if prior.get('status') != 'complete' or prior.get('signature_components') != scientific:
                    raise ValueError(f'{name} settings/code changed or unverified: use --rerun {name}')
                if prior.get('artifacts') != artifact_identities(destination):
                    raise ValueError(f'{name} outputs changed or are missing: use --rerun {name}')
            if not prior and destination.exists():
                raise FileExistsError(f'Unrecorded output requires inspection: {destination}')
            plans[name] = dict(signature_components=scientific, signature=fingerprint(scientific),
                               generation=uuid.uuid4().hex if execute else prior['generation'],
                               execute=execute, prior=prior)
        history, history_record = start_history(directory)
        manifest.update(status='running', state_version=2, requested_modules=requested, resolved=resolved,
                        runtime=runtime, source_hashes=code_hashes())
        write_json(directory/'config.json', config)
        write_json(manifest_path, manifest)
        started = True
        for name in order:
            module = MODULES[name]
            plan = plans[name]
            destination = directory/module['output']
            if not plan['execute']:
                print(f'Reuse {name}: {destination}', flush=True)
                continue
            stage = {key: plan[key] for key in ['signature_components', 'signature', 'generation']}
            stage['status'] = 'running'
            manifest['stages'][name] = stage
            write_json(manifest_path, manifest)
            temporary = directory/f'.{name}-{uuid.uuid4().hex}.tmp'
            temporary.mkdir()
            print(f'Run {name}: {destination}', flush=True)
            try:
                if name == 'segments':
                    build_segments(samples, temporary/'segments.hdf5', stage_settings['segments'])
                    ready = temporary/'segments.hdf5'
                elif name == 'dqdx':
                    plot_segments(directory/'shared/segments.hdf5', temporary, settings)
                    ready = temporary
                    # Preserve the old flat-layout segments path when refreshing imported plots.
                    if (destination/'segments.hdf5').is_symlink():
                        (temporary/'segments.hdf5').symlink_to('../shared/segments.hdf5')
                elif name == 'hit_density':
                    plot_hit_density(directory/'shared/segments.hdf5', temporary, density_settings)
                    ready = temporary
                if inputs != {kind: file_identity(path) for kind, _, path in samples}:
                    raise ValueError('Input changed during the run')
                publish(directory, destination, ready, history, history_record)
                if temporary.exists():
                    temporary.rmdir()
                stage.update(status='complete', artifacts=artifact_identities(destination))
                # Refreshing a dependency invalidates every downstream stage, even if settings match.
                invalid = {name}
                for _ in MODULES:
                    for other, spec in MODULES.items():
                        if other != name and invalid.intersection(spec['dependencies']):
                            invalid.add(other)
                for other in invalid - {name}:
                    if other in manifest['stages']:
                        manifest['stages'][other].update(status='stale', reason=f'{name} was regenerated')
                write_json(manifest_path, manifest)
            except Exception as exc:
                # A computation failure leaves the old output in its original location.
                if plan['prior'] and destination.exists():
                    manifest['stages'][name] = dict(plan['prior'], last_attempt_error=str(exc))
                else:
                    stage.update(status='failed', error=str(exc))
                if temporary.exists():
                    failed = history/'failed'/name
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    temporary.rename(failed)
                raise
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        manifest.pop('error', None)
        history_record.update(status='complete', requested_modules=requested, rerun_modules=sorted(forced))
        write_json(history/'history.json', history_record)
        write_json(manifest_path, manifest)
        return directory
    except Exception as exc:
        if locals().get('started'):
            manifest.update(status='failed', error=str(exc))
            write_json(manifest_path, manifest)
            history_record.update(status='failed', error=str(exc))
            write_json(history/'history.json', history_record)
        raise

    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--root', required=True, help='fsdcube root containing samples/ and comparisons/')
    parser.add_argument('--run-id', required=True, help='e.g. run-002; existing outputs require --resume')
    parser.add_argument('--modules', nargs='+', choices=[*MODULES, 'all'])
    parser.add_argument('--resume', action='store_true', help='reuse completed stages and add missing modules')
    parser.add_argument('--rerun', nargs='+', choices=[*MODULES, 'all'],
                        help='refresh selected stages in this run; implies --resume; old outputs go to .history/')
    args = vars(parser.parse_args())
    args['config_path'] = args.pop('config')
    print(f'Complete: {run(**args)}')


if __name__ == '__main__':
    main()
