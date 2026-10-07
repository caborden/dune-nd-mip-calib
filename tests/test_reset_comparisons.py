"""Verify reset registries resolve distinct data and shared MC selections."""
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

import h5py

from analysis.io import write_json
from analysis.run import resolve_target
from prepare_reset_comparisons import prepare


class ResetComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_id = 'data-Reflow_FSDCube_v3.binary-cosmics-2026_02'
        self.mc_id = 'mc-FSDCubeSim_v1_prc256'
        for sample_id in [self.data_id, self.data_id+'_prc2', self.data_id+'_prc16', self.mc_id]:
            directory = self.root/'samples'/sample_id
            directory.mkdir(parents=True)
            with h5py.File(directory/'selection.hdf5', 'w') as out:
                out.attrs.update(status='complete', schema_version='2.0')
            write_json(directory/'sample.json', dict(kind='mc' if sample_id == self.mc_id else 'data',
                selection_versions={'v1': {'path': 'selection.hdf5'}}))

    def test_pairs_and_recipes_resolve_shared_mc_without_sample_analysis(self):
        with redirect_stdout(io.StringIO()):
            records = prepare(self.root)
            self.assertEqual(prepare(self.root), records)
        mc_paths = []
        for reset in ('prc2', 'prc8', 'prc16'):
            config = json.loads((Path(__file__).resolve().parents[1]/'configs/analyses'/f'data_{reset}.json').read_text())
            directory, samples = resolve_target(self.root, config['target'])
            self.assertEqual(directory.parent, self.root/'comparisons')
            self.assertIn(f'-{reset}_vs_', directory.name)
            self.assertEqual([s[0] for s in samples], ['data', 'mc'])
            expected = self.data_id if reset == 'prc8' else self.data_id+'_'+reset
            self.assertEqual(samples[0][2].parent.name, expected)
            mc_paths.append(samples[1][2])
        self.assertEqual(len(set(mc_paths)), 1)
        self.assertFalse(list((self.root/'samples').glob('*/analysis')))

    def test_incomplete_selection_prevents_all_registry_creation(self):
        with h5py.File(self.root/'samples'/(self.data_id+'_prc16')/'selection.hdf5', 'r+') as out:
            out.attrs['status'] = 'incomplete'
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            prepare(self.root)
        self.assertFalse((self.root/'comparisons').exists())

    def test_conflicting_registry_is_preserved(self):
        path = self.root/'comparisons/reflow-v3-feb2026-prc2_vs_fsdcube-sim-v1-prc256/comparison.json'
        write_json(path, {'samples': {}})
        with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            prepare(self.root)
        self.assertEqual(json.loads(path.read_text()), {'samples': {}})


if __name__ == '__main__':
    unittest.main()
