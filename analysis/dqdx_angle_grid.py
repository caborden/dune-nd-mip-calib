"""Plot segment dQ/dx in five pixel-plane angle bins by five drift angle bins."""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def plot(counts, label, vertical, output):
    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(14, 11))
    fig.subplots_adjust(left=.14, right=.98, top=.86, bottom=.10, wspace=.08, hspace=.15)
    fig.suptitle(f'{label}   |   Segment dQ/dx by track direction', fontsize=18, y=.975)
    fig.text(.56, .92, 'Angle from y toward z in the pixel plane', ha='center', fontsize=13)
    row_title = 'Angle from x in the x–y projection' if vertical == 'xy' else '3D angle from the x drift axis'
    fig.text(.024, .48, row_title, rotation=90, ha='center', va='center', fontsize=13)
    fig.text(.56, .042, 'Segment dQ/dx [ke⁻/cm]', ha='center', fontsize=13)
    fig.text(.14, .012, 'Height = % of displayed segments in each panel per 0.9 ke⁻/cm bin.  N = displayed segments.', fontsize=10)
    for i in range(5):
        angle_range = f'{int(ANGLE_EDGES[i])}–{int(ANGLE_EDGES[i+1])}°'
        axes[0, i].set_title(angle_range)
        axes[4-i, 0].set_ylabel(angle_range, rotation=0, ha='right', va='center', labelpad=12)
        for j in range(5):
            ax = axes[4-i, j]
            hist = counts[i, j]
            n = hist.sum()
            if n:
                ax.stairs(100 * hist / n, CHARGE_EDGES, color='#106c77', fill=True, alpha=.85)
            ax.text(.97, .91, f'N = {n:,}', transform=ax.transAxes, ha='right', va='top', fontsize=9)
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
    parser.add_argument('--output', type=Path, required=True, help='figure filename, e.g. grid.png or grid.pdf')
    args = parser.parse_args()
    counts, selected, valid = histograms(args.segments, args.sample, args.vertical)
    if not counts.sum():
        parser.error('No segments fall within the displayed angle and dQ/dx ranges')
    plot(counts, args.label or args.sample, args.vertical, args.output)
    print(f'{args.output}: {counts.sum():,} shown of {selected:,} {args.sample} segments, '
          f'{selected-valid:,} undefined angles/nonfinite values, {valid-counts.sum():,} outside dQ/dx range')


if __name__ == '__main__':
    main()
