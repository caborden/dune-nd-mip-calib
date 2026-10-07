"""Overlay data dQ/dx histograms and saved fits from completed analysis modules.

This lightweight comparison reads CSV/JSON products, not selection or segment
HDF5 files. Each input is a dqdx/ directory, labeled independently of data/MC kind.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .io import code_hashes, file_identity, read_json, safe_name, write_json
from pixel_dqdx.dqdx import langau


DEFAULT_COMPARISON = 'reflow-v3-feb2026-prc2_vs_prc8_vs_prc16'
COLORS = ['#E69F00', '#009E73', '#CC79A7', '#0072B2', '#D55E00', '#56B4E9']


def product_identity(path):
    return dict(file_identity(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def load_member(label, directory):
    directory = Path(directory).resolve()
    report_path = directory/'analysis.json'
    histogram_path = directory/'data_dqdx_histogram.csv'
    identities = [product_identity(path) for path in (report_path, histogram_path)]
    report = read_json(report_path)
    if report.get('status') != 'complete':
        raise ValueError(f'{label}: dQ/dx analysis is not complete: {report_path}')
    # The module report can remain complete after its parent marks it stale.
    parent_path = directory.parent/'manifest.json'
    if parent_path.exists():
        parent = read_json(parent_path)
        for name in ('segments', 'dqdx'):
            stage = parent.get('stages', {}).get(name)
            if stage is not None and stage.get('status') != 'complete':
                raise ValueError(f'{label}: parent run marks {name} as {stage.get("status")}')
        identities.append(product_identity(parent_path))
    fits = [fit for fit in report['fits'] if fit.get('sample_kind') == 'data']
    samples = [sample for sample in report['samples'] if sample.get('sample_kind') == 'data']
    if len(fits) != 1 or len(samples) != 1:
        raise ValueError(f'{label}: expected exactly one data sample and fit record')
    fit, sample, settings = fits[0], samples[0], report['settings']
    table = np.atleast_1d(np.genfromtxt(histogram_path, delimiter=',', names=True))
    if table.dtype.names != ('bin_low', 'bin_high', 'count'):
        raise ValueError(f'{label}: unsupported histogram columns')
    edges = np.linspace(*settings['hist_range'], settings['bins'] + 1)
    if (len(table) != len(edges)-1 or
            not np.allclose(table['bin_low'], edges[:-1], rtol=0, atol=1e-10) or
            not np.allclose(table['bin_high'], edges[1:], rtol=0, atol=1e-10)):
        raise ValueError(f'{label}: histogram edges disagree with saved settings')
    counts = table['count']
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError(f'{label}: histogram counts must be finite nonnegative integers')
    total = int(counts.sum())
    if total == 0 or total != fit['n_entries_in_histogram']:
        raise ValueError(f'{label}: empty histogram or count total disagrees with saved fit')
    if fit['status'] == 'fit':
        params = np.asarray(fit['params'], dtype=float)
        if (params.shape != (4,) or not np.isfinite(params).all() or
                not 0 < params[1] <= np.ptp(edges) or not 0 < params[2] <= np.ptp(edges) or
                params[3] < 0 or not np.isfinite([fit['mpv'], fit['chi2_red']]).all() or
                not np.isclose(params[0], fit['mpv'])):
            raise ValueError(f'{label}: invalid saved fit parameters')
        # Rendering a saved fit must use the same numerical function as its fit.
        name = 'pixel_dqdx/dqdx.py'
        if report.get('source_hashes', {}).get(name) != code_hashes([name])[name]:
            raise ValueError(f'{label}: fit-function source changed; regenerate the source dQ/dx analysis')
    elif fit['status'] != 'no_fit':
        raise ValueError(f'{label}: unsupported fit status: {fit["status"]}')
    return dict(label=label, directory=str(directory), products=identities,
                sample=sample, fit=fit, settings=settings,
                source_hashes=report.get('source_hashes', {}),
                edges=edges, counts=counts, fraction=counts/total)


def draw_overlay(members, output_dir, alpha):
    settings = members[0]['settings']
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    try:
        for index, member in enumerate(members):
            color = COLORS[index % len(COLORS)]
            fit, edges = member['fit'], member['edges']
            if fit['status'] == 'fit':
                label = (f"{member['label']}   Component MPV = {fit['mpv']:.2f} ke−/cm"
                         f"   $\\chi^2$/ndf = {fit['chi2_red']:.2f}")
            else:
                label = f"{member['label']}   Fit unavailable"
            ax.stairs(member['fraction'], edges, fill=True, color=color, alpha=alpha, label=label)
            ax.stairs(member['fraction'], edges, color=color, linewidth=1.2)
            if fit['status'] == 'fit':
                x = np.linspace(*settings['fit_range'], 250)
                scale = member['counts'].sum()
                ax.plot(x, langau(x, *fit['params'])/scale, color=color, linewidth=2)
        ax.set(title='FSD Cube data: periodic-reset dependence',
               xlabel='dQ/dx [ke−/cm]', ylabel='Fraction of track segments',
               xlim=settings['hist_range'], ylim=(0, None))
        ax.set_ylim(top=ax.get_ylim()[1]*1.22)
        ax.legend(loc='upper right', fontsize=9, framealpha=.95)
        ax.grid(axis='y', alpha=.18)
        low, high = settings['hist_range']
        fig.text(.5, .025,
                 f'Bin fractions sum to 1 over {low:g}–{high:g} ke−/cm; solid curves are saved fits.\n'
                 'Fit: upstream Moyal-like approximation convolved with Gaussian; MPV is the component parameter.',
                 ha='center', va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, .08, 1, 1))
        fig.savefig(output_dir/'dqdx_prc_overlay.pdf')
    finally:
        plt.close(fig)


def compare(root, member_paths, run_id, comparison_id=DEFAULT_COMPARISON, alpha=.5):
    root = Path(root).resolve()
    if not 0 < alpha < 1:
        raise ValueError('Fill alpha must be between zero and one')
    labels = [label for label, _ in member_paths]
    if len(labels) < 2 or len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError('Provide at least two members with unique nonempty labels')
    members = [load_member(label, root/Path(path)) for label, path in member_paths]
    if len({member['sample']['input'] for member in members}) != len(members):
        raise ValueError('Members reference the same data selection; use distinct samples')
    for member in members[1:]:
        if member['settings'] != members[0]['settings']:
            raise ValueError(f'{member["label"]}: segmentation/fit settings differ; regenerate matching source analyses')
        for name in ('analysis/segments.py', 'analysis/dqdx.py', 'pixel_dqdx/segments.py', 'pixel_dqdx/dqdx.py'):
            if (not member['source_hashes'].get(name) or
                    member['source_hashes'][name] != members[0]['source_hashes'].get(name)):
                raise ValueError(f'{member["label"]}: source analysis code differs for {name}')
    comparison_dir = root/'comparisons'/safe_name(comparison_id)
    directory = comparison_dir/'analysis'/safe_name(run_id)
    registry_path = comparison_dir/'comparison.json'
    registry = dict(id=comparison_id, kind='dqdx-overlay',
                    members=[dict(label=m['label'], sample_kind='data', selection_input=m['sample']['input'])
                             for m in members])
    if registry_path.exists() and read_json(registry_path) != registry:
        raise ValueError(f'Comparison members changed: choose a new comparison ID: {registry_path}')
    # Creating the run exclusively prevents concurrent or accidental overwrites.
    directory.mkdir(parents=True, exist_ok=False)
    if not registry_path.exists():
        write_json(registry_path, registry)
    manifest = dict(format='dqdx-overlay-v2', status='incomplete',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    normalization='fraction of in-range track segments per bin: count / total; fits divided by the same total',
                    alpha=alpha, settings=members[0]['settings'],
                    source_hashes=code_hashes(['analysis/compare_dqdx.py', 'pixel_dqdx/dqdx.py']),
                    members=[{key: value for key, value in m.items()
                              if key not in {'edges', 'counts', 'fraction'}} for m in members])
    write_json(directory/'manifest.json', manifest)
    write_json(directory/'config.json', dict(comparison_id=comparison_id, alpha=alpha,
               members=[dict(label=m['label'], dqdx_dir=m['directory']) for m in members]))
    output = directory/'dqdx_overlay'
    try:
        output.mkdir()
        draw_overlay(members, output, alpha)
        for m in members:
            for identity in m['products']:
                if product_identity(identity['path']) != identity:
                    raise ValueError(f'Source product changed during plotting: {identity["path"]}')
        manifest.update(status='complete', outputs=[product_identity(path) for path in sorted(output.iterdir())])
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(directory/'manifest.json', manifest)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='fsdcube workspace root')
    parser.add_argument('--member', action='append', required=True, metavar='LABEL=DQDX_DIR',
                        help='repeat per sample; directory absolute or relative to --root')
    parser.add_argument('--comparison-id', default=DEFAULT_COMPARISON)
    parser.add_argument('--run-id', required=True, help='new overlay run ID; existing runs are refused')
    parser.add_argument('--alpha', type=float, default=.5)
    args = parser.parse_args()
    members = []
    for member in args.member:
        label, separator, path = member.partition('=')
        if not separator or not path:
            parser.error('--member must have the form LABEL=DQDX_DIR')
        members.append((label, path))
    directory = compare(args.root, members, args.run_id, args.comparison_id, args.alpha)
    print(f'Complete: {directory}')


if __name__ == '__main__':
    main()
