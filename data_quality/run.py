"""CLI: python -m data_quality.run --config ... --output-dir ..."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import h5py
import numpy as np

from .flow import inspect_file, resolve_samples
from .metrics import DEFAULT_HISTOGRAMS, Sample, edges, fill_projections, spatial_edges

DEFAULT_SETTINGS = dict(hits_path='charge/calib_prompt_hits', charge_unit='FLOW Q units',
                        time_field='ts_pps', chunk_size=100000, event_batch_size=256,
                        max_event_links=2000000, adc_max=255,
                        channel_charge_bins=[-20, 200, 880], spatial_bins=120,
                        spatial_ranges={})


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    unknown = set(config) - {'samples', 'settings'}
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    supplied = config.get('settings', {})
    unknown = set(supplied) - (set(DEFAULT_SETTINGS) | {'histograms'})
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    settings = dict(DEFAULT_SETTINGS, **supplied)
    settings['histograms'] = dict(DEFAULT_HISTOGRAMS, **supplied.get('histograms', {}))
    unknown = set(settings['histograms']) - set(DEFAULT_HISTOGRAMS)
    if unknown:
        raise ValueError(f'Unknown histograms: {sorted(unknown)}')
    for spec in list(settings['histograms'].values())+[settings['channel_charge_bins']]:
        edges(spec)
    for key in ('chunk_size', 'event_batch_size', 'max_event_links', 'spatial_bins'):
        value = settings[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if settings['spatial_bins'] > 2000:
        raise ValueError('spatial_bins must be <= 2000 to bound 2D histogram memory')
    if not isinstance(settings['adc_max'], int) or settings['adc_max'] < 1:
        raise ValueError('adc_max must be a positive integer')
    if settings['time_field'] not in ('ts_pps', 't_drift'):
        raise ValueError('time_field must be ts_pps or t_drift')
    for axis, bounds in settings['spatial_ranges'].items():
        if axis not in 'xyz' or len(axis) != 1 or len(bounds) != 2:
            raise ValueError('spatial_ranges must map x/y/z to [low, high] in cm')
        edges([*bounds, settings['spatial_bins']])
    if not settings['hits_path'] or settings['hits_path'].endswith('/data'):
        raise ValueError('hits_path must be a FLOW group, e.g. charge/calib_prompt_hits')
    samples = resolve_samples(config['samples'], path.parent)
    return dict(samples=samples, settings=settings)


def preflight(config):
    inventories = []
    for sample in config['samples']:
        files = [inspect_file(path, config['settings']) for path in sample['files']]
        inventories.append(files)
        print(f'{sample["label"]}: {len(files)} files; '
              f'{sum(f["n_hits"] for f in files):,} hit rows; '
              f'{sum(f["n_events"] for f in files):,} event rows', flush=True)
        for info in files:
            print(f'  {info["path"]}\n    crs_ticks={info["tick_us"]} us; '
                  f'event refs={info["references_available"]}; '
                  f'packet fields={",".join(info["packet_fields"])}', flush=True)
            missing = set(('x', 'y', 'z', 't_drift', config['settings']['time_field']))-set(info['hit_fields'])
            if missing:
                print(f'    Missing optional hit fields: {sorted(missing)}', flush=True)
    return inventories


def run(config_path, output_dir=None, check_inputs=False):
    config = load_config(config_path)
    inventories = preflight(config)
    if check_inputs:
        return inventories
    if output_dir is None:
        raise ValueError('--output-dir is required unless --check-inputs is used')
    # Import matplotlib only after read-only input inspection succeeds.
    from .report import render, write_json
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(status='running', schema_version=1,
                    created_utc=datetime.now(timezone.utc).isoformat(), config=config,
                    python=platform.python_version(), numpy=np.__version__, h5py=h5py.__version__,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path(__file__).parent.glob('*.py'))},
                    definitions={
                        'hit_population': 'Every calibrated hit row once per input file; no track/channel/charge cuts.',
                        'event_population': 'Every event with valid forward hit references, including empty events.',
                        'adc_population': 'Every packet_type=0 row; not restricted to hit-linked packets.',
                        'duration': 'Max-min linked hit timestamp, not configured event-builder window.',
                        'delta_t': 'All adjacent time-sorted hits in each channel/event; no cross-event/file pairs.',
                        'normalization': 'Pooled histogram count divided by total entries, including nonfinite/overflow.',
                        'occupancy': 'Channel hit count divided by all sample calibrated-hit rows; not a rate.',
                        'channel_median': 'Median estimated from pooled charge-bin midpoints; overflow medians omitted.',
                        'timestamps': 'Integer differences before float conversion. CRS tick period read per file in us.',
                        'integrity_scope': 'Reference bounds/duplicate links and zero-time pairs audited. '
                                           'No exhaustive record deduplication or detection of copied acquisition data.',
                        'unknown_conditions': 'Both samples half nominal field per user; other operating/calibration '
                                              'conditions unverified. No lifetime, field, or live-time correction.',
                    })
    write_json(output/'summary.json', manifest)
    write_json(output/'config.resolved.json', config)
    try:
        samples = []
        for definition, files in zip(config['samples'], inventories):
            sample = Sample(definition, config['settings'])
            for info in files:
                sample.process(info)
            samples.append(sample)
        axes = spatial_edges(samples, config['settings'])
        for sample in samples:
            print(f'[{sample.name}] filling spatial projections', flush=True)
            fill_projections(sample, axes)
        # Detect changed inputs instead of reporting an internally inconsistent run.
        for files in inventories:
            for info in files:
                stat = Path(info['path']).stat()
                if (stat.st_size, stat.st_mtime_ns) != (info['size_bytes'], info['mtime_ns']):
                    raise RuntimeError(f'Input changed during analysis: {info["path"]}')
        print('Writing collective plots and report', flush=True)
        render(samples, axes, output, manifest)
        manifest.update(status='complete', elapsed_seconds=time.monotonic()-started)
        write_json(output/'summary.json', manifest)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                        elapsed_seconds=time.monotonic()-started)
        write_json(output/'summary.json', manifest)
        raise
    print(f'Complete: {output}/report.html\nPlot PDF: {output}/report.pdf', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, help='New output directory; existing directories are refused')
    parser.add_argument('--check-inputs', action='store_true', help='Inspect paths/schemas only; write nothing')
    args = parser.parse_args()
    if args.output_dir is None and not args.check_inputs:
        parser.error('--output-dir is required unless --check-inputs is used')
    try:
        run(args.config, args.output_dir, args.check_inputs)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f'Data quality failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
