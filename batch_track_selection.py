"""Run legacy selection per FLOW file, optionally merging completed outputs.

Run one batch process per output directory. Parallel workers should use separate
output directories and feed their completed files to merge_track_outputs.py.
"""
import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py

from merge_track_outputs import merge_outputs, publish, signature


def write_manifest(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def valid_output(output, source, is2x2):
    """Resume only a complete output for unchanged input, code, and detector mode."""
    expected = signature(source)
    with h5py.File(output, 'r') as f:
        if f.attrs.get('schema_version') != '2.0' or f.attrs.get('status') != 'complete':
            raise ValueError(f'Not a completed schema-2 per-input output: {output}')
        if list(f['sources']) != ['0'] or f['sources/0'].attrs.get('status') != 'complete':
            raise ValueError(f'Invalid per-input source metadata: {output}')
        meta = f['sources/0/metadata'].attrs
        for key, value in [('input_file', expected['path']), ('input_size_bytes', expected['size']),
                           ('input_mtime_ns', expected['mtime_ns']), ('is2x2', is2x2)]:
            if meta.get(key) != value:
                raise ValueError(f'Resume mismatch ({key}): {output}')
        for name in ['track_selection_fsd.py', 'efield.py', 'track_output_hdf5.py']:
            digest = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            if meta.get(f'{name}_sha256') != digest:
                raise ValueError(f'Resume code changed ({name}): {output}; choose a new output directory')


def subprocess_runner(source, partial, is2x2, log):
    with log.open('w') as stream:
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('track_selection_fsd.py')),
                        str(source), str(partial), str(is2x2)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_batch(inputs, output_dir, is2x2=False, resume=False, merged_output=None, runner=None):
    inputs = [Path(path).resolve() for path in inputs]
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError('Provide at least one input; duplicate input paths are refused')
    for source in inputs:
        if not source.is_file():
            raise FileNotFoundError(source)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir/'batch_manifest.json'
    requested = [signature(source) for source in inputs]
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f'{manifest_path}; use --resume or a new directory')
        manifest = json.loads(manifest_path.read_text())
        if manifest['inputs'] != requested or manifest['is2x2'] != is2x2:
            raise ValueError('Resume input list/order, input size/mtime, or detector mode changed')
    else:
        manifest = dict(inputs=requested, is2x2=is2x2, status='incomplete', files=[])
    runner = runner or subprocess_runner
    outputs = []
    records = []
    for source in inputs:
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
        # Full basename + path hash avoids collisions across input directories.
        output = output_dir/f'{source.name}.{digest}.tracks.hdf5'
        if output in inputs:
            raise ValueError('Output overlaps a FLOW input')
        record = dict(input=str(source), output=str(output), status='pending')
        records.append(record)
        outputs.append(output)
    manifest['batch_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['files'] = records
    manifest['status'] = 'incomplete'
    write_manifest(manifest_path, manifest)
    for source, output, record in zip(inputs, outputs, records):
        partial = output.with_name(output.name + '.partial')
        log = output.with_name(output.name + '.log')
        try:
            if output.exists():
                if not resume:
                    raise FileExistsError(output)
                valid_output(output, source, is2x2)
                record['status'] = 'complete'
                record['reused'] = True
                # A crash between publishing and unlinking may leave a second link.
                if partial.exists() and os.path.samefile(partial, output):
                    partial.unlink()
            else:
                if partial.exists():
                    if not resume:
                        raise FileExistsError(partial)
                    try:
                        valid_output(partial, source, is2x2)
                    except (OSError, ValueError):
                        # Never continue a partial event: preserve it and rerun the
                        # entire FLOW file. Completed other files remain untouched.
                        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                        partial.rename(partial.with_name(partial.name + '.failed-' + stamp))
                    else:
                        publish(partial, output)
                if not output.exists():
                    record['status'] = 'running'
                    record['log'] = str(log)
                    write_manifest(manifest_path, manifest)
                    runner(source, partial, is2x2, log)
                    valid_output(partial, source, is2x2)
                    publish(partial, output)
                record['status'] = 'complete'
            write_manifest(manifest_path, manifest)
            print(f'Complete: {source.name}', flush=True)
        except Exception as exc:
            record['status'] = 'failed'
            record['error'] = str(exc)
            manifest['status'] = 'failed'
            write_manifest(manifest_path, manifest)
            raise
    if merged_output:
        manifest['merge_output'] = str(Path(merged_output).resolve())
        manifest['merge_status'] = 'running'
        write_manifest(manifest_path, manifest)
        try:
            merge_outputs(outputs, merged_output, resume=resume)
        except Exception as exc:
            manifest['status'] = 'failed'
            manifest['merge_status'] = 'failed'
            manifest['merge_error'] = str(exc)
            write_manifest(manifest_path, manifest)
            raise
        manifest['merge_status'] = 'complete'
    manifest['status'] = 'complete'
    write_manifest(manifest_path, manifest)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='*', help='FLOW files (shell-expanded paths)')
    parser.add_argument('--input-glob', action='append', default=[], help='quoted glob; repeatable')
    parser.add_argument('--input-list', help='one FLOW path per line; blank lines and # comments ignored')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--merged-output')
    parser.add_argument('--is2x2', action='store_true', help='default is FSD geometry')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    inputs = list(args.inputs)
    for pattern in args.input_glob:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f'No files match: {pattern}')
        inputs.extend(matches)
    if args.input_list:
        inputs.extend(line.strip() for line in Path(args.input_list).read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith('#'))
    # Duplicates are errors, including overlap between lists/globs.
    run_batch(inputs, args.output_dir, args.is2x2, args.resume, args.merged_output)
