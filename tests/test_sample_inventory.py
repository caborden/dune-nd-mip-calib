"""Inventory regressions for aliases, incomplete campaigns and concurrent updates."""
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py

from sample_inventory import collect, generate, refresh_after_selection


def selection(path, cut=29):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as source:
        source.attrs.update(status='complete', schema_version='2.0')
        source.create_dataset('tracks/data', shape=(3,))
        meta = source.create_group('sources/0/metadata')
        if cut is not None:
            meta.attrs['min_length_cm'] = cut


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'fsdcube'
        self.root.mkdir()

    def test_aliases_missing_files_and_metadata(self):
        data = self.root / 'data/tracks.hdf5'
        selection(data)
        for name in ('original', 'alias'):
            write(self.root / f'samples/{name}/sample.json', {
                'kind': 'data', 'selection_versions': {'v1': {'path': str(data)}}})
        legacy = self.root / 'samples/legacy/selection/pilot.hdf5'
        selection(legacy, cut=None)
        selection(self.root / 'data/parts/ignored.tracks.hdf5')
        missing = self.root / 'missing.hdf5'
        write(self.root / 'samples/missing/sample.json', {
            'kind': 'mc', 'selection_versions': {'v1': {'path': str(missing)}}})
        items, _, _ = collect(self.root, {})
        self.assertEqual(len(items), 3)
        records = {row['path']: row for row in items}
        self.assertEqual(len(records[str(data)]['aliases']), 2)
        self.assertEqual(records[str(data)]['tracks'], 3)
        self.assertEqual(records[str(data)]['cut'], '>29 cm')
        self.assertEqual(records[str(legacy)]['cut'], 'unknown')
        self.assertEqual(records[str(missing)]['status'], 'missing')

    def test_campaign_settings_and_persisted_discovery(self):
        jobs = self.root.parent / 'jobs'
        production = self.root.parent / 'production'
        mc = production / 'campaigns/mc/tracks/mc.tracks.hdf5'
        selection(mc)
        write(jobs / 'mc/summary.json', dict(status='complete', output=str(mc), total_tracks=3))
        (jobs / 'mc/detector-prc4.yaml').write_text('periodic_reset_cycles: 256\nadc_busy_delay: 8\n')
        write(jobs / 'pending/campaign.json', {})
        report = generate(self.root, jobs, production).read_text()
        self.assertIn('adc_busy_delay=8', report)
        self.assertIn('adc_hold_delay=unspecified/inherited', report)
        self.assertIn('no summary published', report)
        self.assertIn('Selections (1)', report)
        self.assertIn('mc.tracks.hdf5', generate(self.root).read_text())

    def test_concurrent_refreshes_keep_both_external_selections(self):
        paths = [self.root.parent / f'external{i}.hdf5' for i in range(2)]
        for path in paths:
            selection(path)
        context = multiprocessing.get_context('spawn')
        workers = [context.Process(target=generate, args=(self.root,),
                                   kwargs={'selections': [path]}) for path in paths]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=20)
            self.assertEqual(worker.exitcode, 0)
        report = (self.root / 'sample-inventory.md').read_text()
        for path in paths:
            self.assertIn(str(path), report)

    def test_completion_hook_and_nonfatal_failure(self):
        path = self.root / 'data/tracks.hdf5'
        selection(path)
        with patch.dict('os.environ', {}, clear=True):
            refresh_after_selection(path)
            self.assertTrue((self.root / 'sample-inventory.md').exists())
            with patch('sample_inventory.generate', side_effect=OSError('test failure')):
                refresh_after_selection(path)
            with patch('sample_inventory.generate') as generate_mock:
                refresh_after_selection(self.root / 'data/parts/part.tracks.hdf5')
                refresh_after_selection(self.root / 'batch/part.tracks.hdf5.partial')
                generate_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
