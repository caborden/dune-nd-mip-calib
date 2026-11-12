"""Plot segment dQ/dx in five pixel-plane angle bins by five drift angle bins."""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from .compare_dqdx import fitted_widths
from .dqdx import fit_counts


ANGLE_EDGES = np.linspace(0, 90, 6)
CHARGE_EDGES = np.linspace(0, 90, 101)  # PRC2 dQ/dx histogram settings


def folded_angles(theta, phi, vertical):
    """Use the segment writer's theta from +y and phi from +z toward +x."""
    theta, phi = np.deg2rad(theta), np.deg2rad(phi)
    x = np.abs(np.sin(theta) * np.sin(phi))
    y = np.abs(np.cos(theta))
    z = np.abs(np.sin(theta) * np.cos(phi))
    transverse = np.hypot(y, z)
    beta = np.degrees(np.arctan2(z, y))
    alpha = np.degrees(np.arctan2(y if vertical == 'xy' else transverse, x))
    # At an axis pole the projected direction is undefined. Allow for the
    # floating-point residual when reconstructing exact cardinal directions.
    beta[transverse < 1e-12] = np.nan
    if vertical == 'xy':
        alpha[np.hypot(x, y) < 1e-12] = np.nan
    return np.round(alpha, 12), np.round(beta, 12)


def histograms(path, kind, vertical):
    counts = np.zeros((5, 5, len(CHARGE_EDGES) - 1), dtype=np.int64)
    selected = valid = 0
    with h5py.File(path, 'r') as source:
        if source.attrs.get('status') != 'complete':
            raise ValueError(f'Expected a completed segment table in {path}')
        sample_ids = [int(i) for i, g in source['samples'].items()
                      if g.attrs.get('sample_kind') == kind]
        if len(sample_ids) != 1:
            raise ValueError(f'Expected one {kind} sample in {path}, found {sample_ids}')
        table = source['segments/data']
        fields = table.fields(['sample_id', 'theta', 'phi', 'dqdx'])
        for start in range(0, len(table), 100_000):
            rows = fields[start:start + 100_000]
            rows = rows[rows['sample_id'] == sample_ids[0]]
            selected += len(rows)
            alpha, beta = folded_angles(rows['theta'], rows['phi'], vertical)
            q = rows['dqdx']
            keep = np.isfinite(alpha) & np.isfinite(beta) & np.isfinite(q)
            valid += np.count_nonzero(keep)
            hist, _ = np.histogramdd((alpha[keep], beta[keep], q[keep]),
                                     bins=(ANGLE_EDGES, ANGLE_EDGES, CHARGE_EDGES))
            counts += hist.astype(np.int64)
    return counts, selected, valid


def fit_annotation(fit):
    if fit is None:
        return 'Fit unavailable'
    widths = fitted_widths(dict(status='fit', params=fit['params']))
    lines = [f"MPV {fit['params'][0]:.1f}"]
    if widths['status'] == 'complete':
        lines.extend([f"FWHM {widths['fwhm']:.1f}",
                      f"FWHM/MPV {widths['fwhm_over_component_mpv']:.3f}"])
    else:
        lines.append('FWHM unavailable')
    return '\n'.join(lines)


def plot(counts, label, vertical, output, stats_threshold=200):
    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(14, 11))
    fig.subplots_adjust(left=.14, right=.98, top=.86, bottom=.10, wspace=.08, hspace=.15)
    fig.suptitle(f'{label}   |   Segment dQ/dx by track direction', fontsize=18, y=.975)
    fig.text(.56, .92, 'Angle from y toward z in the pixel plane', ha='center', fontsize=13)
    row_title = 'Angle from x in the x–y projection' if vertical == 'xy' else '3D angle from the x drift axis'
    fig.text(.024, .48, row_title, rotation=90, ha='center', va='center', fontsize=13)
    fig.text(.56, .052, 'Segment dQ/dx [ke⁻/cm]', ha='center', fontsize=13)
    fig.text(.14, .028, 'Height = % of displayed segments per 0.9 ke⁻/cm bin.  N = displayed segments.', fontsize=9)
    fig.text(.14, .010, f'N > {stats_threshold}: component MPV and full convolution FWHM [ke⁻/cm], red curve = fit over 20–50 ke⁻/cm.', fontsize=9)
    for i in range(5):
        angle_range = f'{int(ANGLE_EDGES[i])}–{int(ANGLE_EDGES[i+1])}°'
        axes[0, i].set_title(angle_range)
        axes[4-i, 0].set_ylabel(angle_range, rotation=0, ha='right', va='center', labelpad=12)
        for j in range(5):
            ax = axes[4-i, j]
            hist = counts[i, j]
            n = hist.sum()
            if n:
                ax.stairs(100 * hist / n, CHARGE_EDGES, color='steelblue', fill=True, alpha=.85)
            note = f'N = {n:,}'
            if n > stats_threshold:
                fit, _ = fit_counts(hist, CHARGE_EDGES, (20., 50.), min_fit_entries=100)
                if fit is not None:
                    ax.plot(fit['x_fit'], 100 * fit['fit_vals'] / n, color='red', lw=1.4)
                note += '\n' + fit_annotation(fit)
            ax.text(.98, .97, note, transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=.85))
            ax.grid(axis='y', alpha=.18)
    axes[0, 0].set(xlim=(0, 90), ylim=(0, None), xticks=[0, 30, 60, 90])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments', required=True, type=Path, help='completed shared/segments.hdf5')
    parser.add_argument('--sample', choices=['data', 'mc'], required=True)
    parser.add_argument('--vertical', choices=['xy', 'drift'], default='xy',
                        help='angle from x after projection onto x–y (xy), or full 3D angle from x (drift)')
    parser.add_argument('--label', help='display name for the selected sample')
    parser.add_argument('--output', type=Path, required=True, help='figure filename, e.g. grid.pdf (defaults to .pdf if no extension)')
    parser.add_argument('--stats-threshold', type=int, default=200,
                        help='annotate fit statistics when displayed N exceeds this value')
    args = parser.parse_args()
    if args.stats_threshold < 0:
        parser.error('--stats-threshold must be nonnegative')
    counts, selected, valid = histograms(args.segments, args.sample, args.vertical)
    if not counts.sum():
        parser.error('No segments fall within the displayed angle and dQ/dx ranges')
    if not args.output.suffix:
        args.output = args.output.with_suffix('.pdf')
    plot(counts, args.label or args.sample, args.vertical, args.output, args.stats_threshold)
    print(f'{args.output}: {counts.sum():,} shown of {selected:,} {args.sample} segments, '
          f'{selected-valid:,} undefined angles/nonfinite values, {valid-counts.sum():,} outside dQ/dx range')


if __name__ == '__main__':
    main()
