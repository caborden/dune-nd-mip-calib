"""Register reset data/MC pairs after validating their completed selections."""
import argparse
from pathlib import Path

from analysis.io import read_json, write_json
from analysis.run import resolve_target


def prepare(root, resets=('prc2', 'prc8', 'prc16')):
    root = Path(root).resolve()
    records = {}
    for reset in resets:
        if reset not in {'prc2', 'prc8', 'prc16'}:
            raise ValueError(f'Unknown reset: {reset}')
        # Preserve the historical prc8 sample's canonical path.
        data_id = 'data-Reflow_FSDCube_v3.binary-cosmics-2026_02'
        if reset != 'prc8':
            data_id += '_' + reset
        members = {'data': data_id, 'mc': 'mc-FSDCubeSim_v1_prc256'}
        for kind, sample_id in members.items():
            _, samples = resolve_target(root, dict(kind='sample', id=sample_id,
                                                  selection_version='v1'))
            if samples[0][0] != kind:
                raise ValueError(f'Wrong sample kind for {sample_id}: expected {kind}')
            print(f'{reset} {kind}: {samples[0][2]}')
        name = f'reflow-v3-feb2026-{reset}_vs_fsdcube-sim-v1-prc256'
        path = root/'comparisons'/name/'comparison.json'
        record = dict(id=name, samples={kind: dict(id=sample_id, selection_version='v1')
                                       for kind, sample_id in members.items()})
        if path.exists() and read_json(path) != record:
            raise FileExistsError(f'Existing comparison registry differs: {path}')
        records[path] = record
    # Preflight all requested pairs before creating any registry.
    for path, record in records.items():
        if not path.exists():
            write_json(path, record)
        print(f'Ready: {path}')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--resets', nargs='+', choices=['prc2', 'prc8', 'prc16'],
                        default=['prc2', 'prc8', 'prc16'])
    args = parser.parse_args()
    prepare(args.root, args.resets)
