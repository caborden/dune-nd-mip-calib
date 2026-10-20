"""Charge sharing in exactly-two-hit pixels, using the hit-timing population."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .hit_time_difference import resolve_inputs, summarize as summarize_timing
from .io import code_hashes, file_identity, read_json, safe_name, write_json

CATEGORIES = {'all': 'All finite-time pairs',
              'dt25_40': '25 ≤ Δt ≤ 40 CRS ticks',
              'dt_gt40': 'Δt > 40 CRS ticks'}
COMPONENTS = {'first': 'First hit', 'second': 'Second hit', 'total': 'Total Q₁ + Q₂'}
COLORS = {'first': '#0072B2', 'second': '#009E73', 'total': '#CC79A7'}


def summarize(path, edges, time_field='ts_pps'):
    """Stream once; all components share a category's pair-count denominator."""
    categories = {key: dict(instances=0, invalid_charge_pairs=0,
                           counts={c: np.zeros(len(edges)-1, dtype=np.int64) for c in COMPONENTS},
                           underflow=Counter(), overflow=Counter()) for key in CATEGORIES}

    def consume(pair, delta):
        # Stable order makes tied timestamps deterministic: original saved row first.
        pair = pair[np.argsort(pair[time_field], kind='stable')]
        q1, q2 = pair['Q'].astype(float)
        with np.errstate(over='ignore', invalid='ignore'):
            values = dict(first=q1, second=q2, total=q1+q2)
        valid = all(np.isfinite(q) for q in values.values())
        selected = ['all']
        if 25 <= delta <= 40:
            selected.append('dt25_40')
        elif delta > 40:
            selected.append('dt_gt40')
        for key in selected:
            category = categories[key]
            category['instances'] += 1
            if not valid:
                category['invalid_charge_pairs'] += 1
                continue
            for component, q in values.items():
                # Match numpy.histogram's inclusive final edge without a histogram per pair.
                index = np.searchsorted(edges, q, side='right') - 1
                if q == edges[-1]:
                    index = len(edges)-2
                if q < edges[0]:
                    category['underflow'][component] += 1
                elif q > edges[-1]:
                    category['overflow'][component] += 1
                else:
                    category['counts'][component][index] += 1

    timing = summarize_timing(path, np.array([0., 25., 40.]), time_field,
                              pair_consumer=consume, required_fields=['Q'])
    audit = {key: value for key, value in timing['audit'].items()
             if key not in {'pairs_above_range', 'pairs_in_histogram'}}
    return dict(audit=audit, crs_ticks=timing['crs_ticks'], categories=categories)


def fraction(category, component):
    counts = category['counts'][component]
    return counts / category['instances'] if category['instances'] else np.zeros_like(counts, dtype=float)


def render(output, samples, edges, overlay_data):
    def finish(fig, axes, name):
        for ax in np.asarray(axes).flat:
            ax.set(xlabel='Charge Q [ke−]', ylabel='Fraction of pairs per bin',
                   xlim=(edges[0], edges[-1]), ylim=(0, None))
            ax.grid(axis='y', alpha=.2)
            ax.legend(fontsize=8)
        fig.text(.5, .015, 'Each curve: bin count / all pairs in its timing category; out-of-range and invalid charges remain in N.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, .94))
        for ext in ('png', 'pdf'):
            fig.savefig(output/f'{name}.{ext}', dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = iter(['#0072B2', '#009E73', '#CC79A7', '#D55E00', '#56B4E9'])
    for sample in samples.values():
        color = '#E69F00' if sample['kind'] == 'mc' else next(palette, None)
        for ax, key in zip(axes, ['dt25_40', 'dt_gt40']):
            cat = sample['categories'][key]
            ax.stairs(fraction(cat, 'total'), edges, color=color,
                      linestyle='--' if sample['kind'] == 'mc' else '-',
                      label=f"{sample['label']}: N={cat['instances']:,}")
            ax.set_title(CATEGORIES[key])
    fig.suptitle('Total charge in pixels with exactly two hits')
    finish(fig, axes, 'total_charge_by_separation')

    # Include an uncut breakdown as well as both requested timing categories.
    for sample_key, sample in samples.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (key, title) in zip(axes, CATEGORIES.items()):
            cat = sample['categories'][key]
            for component, label in COMPONENTS.items():
                ax.stairs(fraction(cat, component), edges, color=COLORS[component], label=label)
            ax.set_title(f"{title}\nN={cat['instances']:,}")
        fig.suptitle(f"{sample['label']}: two-hit charge breakdown")
        finish(fig, axes, f'{sample_key}_charge_breakdown')

    mc_key = next(key for key, sample in samples.items() if sample['kind'] == 'mc')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, title) in zip(axes, CATEGORIES.items()):
        for sample_key in [overlay_data, mc_key]:
            sample = samples[sample_key]
            cat = sample['categories'][key]
            for component, label in COMPONENTS.items():
                ax.stairs(fraction(cat, component), edges, color=COLORS[component],
                          linestyle='--' if sample['kind'] == 'mc' else '-',
                          label=f"{sample['label']} {label} (N={cat['instances']:,})")
        ax.set_title(title)
    fig.suptitle('Two-hit charge: data (solid) and MC (dashed)')
    finish(fig, axes, f'{overlay_data}_mc_charge_overlay')


def run(config_path, root, run_id, bins=100, charge_min=0., charge_max=200.,
        time_field='ts_pps', overlay_data='data_prc16'):
    if bins < 1 or not np.isfinite([charge_min, charge_max]).all() or charge_min >= charge_max:
        raise ValueError('Require positive bins and finite increasing charge limits')
    if time_field not in {'ts_pps', 't_drift'}:
        raise ValueError('time_field must be ts_pps or t_drift')
    config = read_json(config_path)
    target, inputs = resolve_inputs(Path(root).resolve(), config, 'Data')
    if overlay_data not in {key for key, _, _, kind in inputs if kind == 'data'}:
        raise ValueError(f'Overlay data key {overlay_data!r} is not a data sample in the recipe')
    identities = {key: file_identity(path) for key, _, path, _ in inputs}
    edges = np.linspace(charge_min, charge_max, bins+1)
    directory = target/'analysis'/safe_name(run_id)
    directory.mkdir(parents=True, exist_ok=False)
    output = directory/'two_hit_charge'
    output.mkdir()
    settings = dict(time_field=time_field, charge_field='Q', charge_units='Q assumed ke−; no additional correction',
                    edges_charge=edges.tolist(), categories=CATEGORIES, overlay_data=overlay_data,
                    population='exactly two recorded hits per electronics pixel per saved event, including track-rejected hits; only events with accepted tracks are saved',
                    ordering='increasing timestamp; ties use saved row order; no rollover correction',
                    invalid='nonfinite times excluded; if either charge or their sum is nonfinite, exclude pair from all three charge histograms but retain in denominator',
                    normalization='bin count / all finite-time pairs in category, including invalid charges and under/overflow; no bin-width division')
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(), inputs=identities,
                    settings=settings, source_hashes=code_hashes(['analysis/two_hit_charge.py', 'analysis/hit_time_difference.py', 'analysis/io.py', 'analysis/run.py']))
    write_json(directory/'config.json', dict(recipe=config, settings=settings))
    write_json(directory/'manifest.json', manifest)
    start = time.monotonic()
    try:
        samples = {}
        for key, label, path, kind in inputs:
            sample = summarize(path, edges, time_field)
            sample.update(label=label, kind=kind)
            samples[key] = sample
            print(f"{label}: " + ', '.join(f"{c}: {v['instances']:,} pairs" for c, v in sample['categories'].items()), flush=True)
            for category_key, cat in sample['categories'].items():
                for component in COMPONENTS:
                    np.savetxt(output/f'{key}_{category_key}_{component}.csv',
                               np.column_stack([edges[:-1], edges[1:], cat['counts'][component], fraction(cat, component)]),
                               delimiter=',', header='bin_low_ke,bin_high_ke,count,fraction', comments='')
        periods = [sample['crs_ticks'] for sample in samples.values()]
        if not np.allclose(periods, periods[0], rtol=1e-10, atol=0):
            raise ValueError('Data and MC CRS tick periods differ')
        render(output, samples, edges, overlay_data)
        if identities != {key: file_identity(path) for key, _, path, _ in inputs}:
            raise ValueError('Input changed during analysis')
        # Keep all counts, including out-of-range audits, in the machine-readable record.
        for sample in samples.values():
            for cat in sample['categories'].values():
                cat['counts'] = {key: value.tolist() for key, value in cat['counts'].items()}
        manifest.update(status='complete', samples=samples, elapsed_seconds=time.monotonic()-start)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(directory/'manifest.json', manifest)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/analyses/hit_timing_prc_comparison.json')
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-id', default='two-hit-charge-001')
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--charge-min', type=float, default=0.)
    parser.add_argument('--charge-max', type=float, default=200.)
    parser.add_argument('--time-field', choices=['ts_pps', 't_drift'], default='ts_pps')
    parser.add_argument('--overlay-data', default='data_prc16', help='data key in the recipe for the MC overlay')
    args = vars(parser.parse_args())
    args['config_path'] = args.pop('config')
    print('Complete:', run(**args))


if __name__ == '__main__':
    main()
