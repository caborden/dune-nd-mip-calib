"""Compare completed single-data runs with explicit sample labels."""
import argparse
from pathlib import Path
from .compare_dqdx import load_member, draw_overlay, product_identity, COLORS
from .hit_density import save_plot
from .selection_hits import draw as draw_hits
from .io import read_json, write_json, safe_name, code_hashes


def compare(root, members, comparison_id, run_id):
    root = Path(root).resolve()
    if len(members) < 2 or len({label for label, _ in members}) != len(members) or any(not label.strip() for label, _ in members):
        raise ValueError('Provide at least two unique nonempty labels')
    dqdx, density, hits, products = [], [], [], []
    for label, path in members:
        directory = root / path
        parent = read_json(directory/'manifest.json')
        for name in ['segments', 'dqdx', 'hit_density', 'selection_hits']:
            if parent['stages'].get(name, {}).get('status') != 'complete':
                raise ValueError(f'{label}: {name} is not complete')
        products.append(product_identity(directory/'manifest.json'))
        dqdx.append(load_member(label, directory/'dqdx'))
        for name, destination in [('hit_density', density), ('selection_hits', hits)]:
            report_path = directory/name/'analysis.json'
            products.append(product_identity(report_path))
            report = read_json(report_path)
            if report['status'] != 'complete' or len(report['samples']) != 1 or report['samples'][0]['sample_kind'] != 'data':
                raise ValueError(f'{label}: expected completed single-data {name}')
            destination.append(report)
    if len({m['sample']['input'] for m in dqdx}) != len(dqdx):
        raise ValueError('Comparison requires distinct selections')
    for i in range(1, len(members)):
        if dqdx[i]['settings'] != dqdx[0]['settings']:
            raise ValueError('dQ/dx settings differ')
        for name in ['analysis/segments.py', 'analysis/dqdx.py', 'pixel_dqdx/segments.py', 'pixel_dqdx/dqdx.py']:
            if not dqdx[0]['source_hashes'].get(name) or dqdx[i]['source_hashes'].get(name) != dqdx[0]['source_hashes'][name]:
                raise ValueError(f'dQ/dx source differs: {name}')
        for reports in [density, hits]:
            if reports[i]['settings'] != reports[0]['settings'] or reports[i]['source_hashes'] != reports[0]['source_hashes']:
                raise ValueError('Module settings/source differ; regenerate matching analyses')
        if density[i]['segment_settings'] != density[0]['segment_settings']:
            raise ValueError('Hit-density segment settings differ')
    directory = root/'comparisons'/safe_name(comparison_id)/'analysis'/safe_name(run_id)
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', members=[dict(label=l, run=str(root/p)) for l,p in members],
                  products=products + [p for m in dqdx for p in m['products']],
                  source_hashes=code_hashes(['analysis/compare_runs.py', 'analysis/compare_dqdx.py',
                                            'analysis/hit_density.py', 'analysis/selection_hits.py']))
    write_json(directory/'manifest.json', record)
    try:
        draw_overlay(dqdx, directory, .5, title='FSD Cube: data comparison')
        density_samples, hit_samples = [], []
        for i, (label, _) in enumerate(members):
            density_samples.append(dict(density[i]['samples'][0], label=label, color=COLORS[i % len(COLORS)]))
            hit_samples.append(dict(hits[i]['samples'][0], label=label, color=COLORS[i % len(COLORS)]))
        save_plot(directory, 'hit_density_overlay', density_samples, density[0]['settings'], density[0]['segment_settings'])
        draw_hits(hit_samples, directory)
        for identity in record['products']:
            if product_identity(identity['path']) != identity:
                raise ValueError('Source products changed during comparison')
        record.update(status='complete', outputs=[product_identity(p) for p in sorted(directory.iterdir()) if p.suffix in {'.pdf', '.png'}])
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(directory/'manifest.json', record)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--member', action='append', required=True, help='LABEL=sample analysis run directory')
    parser.add_argument('--comparison-id', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    members = []
    for value in args.member:
        label, separator, path = value.partition('=')
        if not separator or not path:
            parser.error('Expected --member LABEL=RUN_DIRECTORY')
        members.append((label, Path(path)))
    print(compare(args.root, members, args.comparison_id, args.run_id))


if __name__ == '__main__':
    main()
