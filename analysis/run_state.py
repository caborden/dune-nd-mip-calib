"""Compatibility checks and preserved history for updating analysis runs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

import h5py

from .config import SEGMENT_KEYS
from .io import code_hashes, fingerprint, file_identity, write_json


def components(name, modules, inputs, runtime, settings, stages):
    # Runner/manifest-only edits do not invalidate scientific artifacts.
    return dict(inputs=inputs, runtime=runtime, settings=settings,
        code=code_hashes(modules[name]['code'] + ['analysis/config.py']),
        dependencies={dep: dict(signature=stages[dep]['signature'], generation=stages[dep]['generation'])
                      for dep in modules[name]['dependencies']})


def upgrade_legacy(manifest, modules):
    """Recover v1 provenance only when its original fingerprint reconstructs exactly."""
    old_stages = manifest['stages']
    originals = {name: dict(stage) for name, stage in old_stages.items()}
    for stage in old_stages.values():
        stage.setdefault('generation', uuid.uuid4().hex)
    for name in modules:
        stage = old_stages.get(name, {})
        if not stage or 'signature_components' in stage or stage.get('status') != 'complete':
            continue
        try:
            options = manifest['resolved']['settings']
            if name == 'hit_density':
                options = manifest['resolved']['module_settings']['hit_density']
            selected = {key: options[key] for key in modules[name]['settings']}
            old_code = {key: manifest['source_hashes'][key] for key in
                        modules[name]['code'] + ['analysis/config.py', 'analysis/io.py', 'analysis/run.py']}
            legacy = dict(inputs=manifest['resolved']['inputs'], runtime=manifest['runtime'],
                settings=selected, code=old_code,
                dependencies={dep: originals[dep]['signature'] for dep in modules[name]['dependencies']})
            if fingerprint(legacy) != stage['signature']:
                raise ValueError('Original fingerprint does not match the available provenance')
            scientific = dict(legacy)
            scientific['code'] = {key: old_code[key] for key in modules[name]['code'] + ['analysis/config.py']}
            scientific['dependencies'] = {dep: dict(signature=old_stages[dep]['signature'],
                generation=old_stages[dep]['generation']) for dep in modules[name]['dependencies']}
            stage['previous_signature'] = stage['signature']
            stage['signature_components'] = scientific
            stage['signature'] = fingerprint(scientific)
        except (KeyError, ValueError):
            stage['status'] = 'unverified'
            stage['reason'] = 'Old provenance cannot be reconstructed; explicitly rerun this module'
    return manifest


def adopt_imported(directory, original, resolved, runtime, stage_settings, modules, rebuild_segments):
    """Validate saved input identities/settings before reusing an imported table."""
    table = directory/'shared/segments.hdf5'
    reusable = not rebuild_segments
    if reusable:
        with h5py.File(table, 'r') as f:
            stored = json.loads(f.attrs['settings'])
            if f.attrs.get('status') != 'complete' or any(
                    stored.get(key) != stage_settings['segments'][key] for key in SEGMENT_KEYS):
                raise ValueError('Imported segment settings differ; use --rerun segments')
            expected_ids = {str(0 if kind == 'data' else 1) for kind in resolved['inputs']}
            if set(f['samples']) != expected_ids:
                raise ValueError('Imported samples differ; use --rerun segments')
            for kind, identity in resolved['inputs'].items():
                attrs = f[f'samples/{0 if kind == "data" else 1}'].attrs
                old_path = Path(str(attrs.get('input_file', '')))
                same_path = old_path.exists() and str(old_path.resolve()) == identity['path']
                same_stat = (attrs.get('input_size_bytes') == identity['size'] and
                             attrs.get('input_mtime_ns') == identity['mtime_ns'])
                if not (same_path and same_stat):
                    raise ValueError('Cannot verify imported input path/size/mtime; use --rerun segments')
        expected_hash = original.get('source_hashes', {}).get('pixel_dqdx/segments.py')
        if expected_hash != code_hashes(['pixel_dqdx/segments.py'])['pixel_dqdx/segments.py']:
            raise ValueError('Imported segmentation code differs; use --rerun segments')
    stages = {}
    for name, old in original['stages'].items():
        if name not in modules:
            continue
        stages[name] = dict(status='stale', generation=uuid.uuid4().hex,
                            imported_provenance=old, reason='Imported plotting module needs refresh')
    if reusable:
        stage = stages['segments']
        scientific = components('segments', modules, resolved['inputs'], runtime, stage_settings['segments'], stages)
        stage.update(status='complete', signature_components=scientific, signature=fingerprint(scientific),
                     artifacts=[file_identity(table)], origin='verified imported table')
    return dict(format='modular-analysis-v1', state_version=2, resolved=resolved, runtime=runtime,
                stages=stages, imported_original=original,
                created_utc=datetime.now(timezone.utc).isoformat())


def start_history(directory):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    history = directory/'.history'/stamp
    history.mkdir(parents=True)
    for name in ['config.json', 'manifest.json']:
        source = directory/name
        if source.exists():
            shutil.copy2(source, history/name)
    record = dict(status='running', previous_outputs=[])
    write_json(history/'history.json', record)
    return history, record


def publish(directory, destination, ready, history, record):
    """Replace only after successful computation; preserve the previous destination."""
    saved = history/destination.relative_to(directory)
    links = {}
    if destination.is_dir():
        links = {str(p.relative_to(destination)): p.resolve() for p in destination.rglob('*') if p.is_symlink()}
    replaced = destination.exists()
    if replaced:
        saved.parent.mkdir(parents=True, exist_ok=True)
        record['previous_outputs'].append(dict(source=str(destination), destination=str(saved)))
        write_json(history/'history.json', record)
        destination.rename(saved)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        ready.rename(destination)
    except BaseException:
        if replaced:
            saved.rename(destination)
        raise
    # Keep compatibility links in archived directories readable after relocation.
    for relative, target in links.items():
        archived = saved/relative
        candidate = history/target.relative_to(directory) if target.is_relative_to(directory) else target
        if candidate.exists():
            target = candidate
        archived.unlink()
        archived.symlink_to(os.path.relpath(target, archived.parent))
