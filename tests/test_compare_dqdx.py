"""Data overlay normalization, saved-fit reuse, and incompatible-product checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis.compare_dqdx import compare, load_member, DEFAULT_COMPARISON
from analysis.config import make_settings
from analysis.io import code_hashes, read_json, write_json
from pixel_dqdx.dqdx import langau


class DqdxOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.members = []
        settings = make_settings()
        edges = np.linspace(0, 90, 101)
        for index, label in enumerate(('prc2', 'prc8', 'prc16')):
            directory = self.root/label/'dqdx'
            directory.mkdir(parents=True)
            params = [31.8 + index*1.5, 2.8, 5.3, 12000*(index+1)]
            counts = np.rint(langau((edges[:-1]+edges[1:])/2, *params))
            np.savetxt(directory/'data_dqdx_histogram.csv',
                       np.column_stack([edges[:-1], edges[1:], counts]),
                       delimiter=',', header='bin_low,bin_high,count', comments='')
            fit = dict(sample_kind='data', status='fit', params=params,
                       mpv=params[0], mpv_error=.1, chi2_red=2.5+index, ndf=30,
                       n_entries_in_histogram=int(counts.sum()))
            write_json(directory/'analysis.json', dict(status='complete', settings=settings,
                       samples=[dict(sample_kind='data', input=f'/selection/{label}.hdf5'),
                                dict(sample_kind='mc', input='/selection/mc.hdf5')],
                       fits=[fit, dict(sample_kind='mc', status='no_fit')],
                       source_hashes=code_hashes()))
            self.members.append((label, directory))

    def test_real_render_normalization_and_saved_fit_preservation(self):
        # No optimizer should run: legend and manifest must use the saved data fits.
        with patch('scipy.optimize.curve_fit', side_effect=AssertionError('Unexpected refit')):
            output = compare(self.root, self.members, 'run-001')
        self.assertEqual(output, self.root/'comparisons'/DEFAULT_COMPARISON/'analysis/run-001')
        manifest = read_json(output/'manifest.json')
        self.assertEqual(manifest['status'], 'complete')
        for label, directory in self.members:
            member = load_member(label, directory)
            self.assertAlmostEqual(np.sum(member['fraction']), 1)
            np.testing.assert_allclose(member['fraction'], member['counts']/member['counts'].sum())
        fits = [member['fit'] for member in manifest['members']]
        self.assertEqual([f['mpv'] for f in fits], [31.8, 33.3, 34.8])
        self.assertEqual([f['chi2_red'] for f in fits], [2.5, 3.5, 4.5])
        products = list((output/'dqdx_overlay').iterdir())
        self.assertEqual([p.name for p in products], ['dqdx_prc_overlay.pdf'])
        self.assertGreater(products[0].stat().st_size, 1000)
        self.assertEqual([Path(p['path']).name for p in manifest['outputs']],
                         ['dqdx_prc_overlay.pdf'])
        with self.assertRaises(FileExistsError):
            compare(self.root, self.members, 'run-001')

    def test_reject_mismatched_settings_and_stale_stage_before_writing(self):
        path = self.members[1][1]/'analysis.json'
        report = read_json(path)
        report['settings']['step_cm'] = 1.5
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            compare(self.root, self.members, 'bad-settings')
        self.assertFalse((self.root/'comparisons').exists())
        write_json(self.members[0][1].parent/'manifest.json',
                   dict(stages={'dqdx': {'status': 'stale'}}))
        with self.assertRaisesRegex(ValueError, 'stale'):
            load_member(*self.members[0])

    def test_reject_corrupt_histogram_and_duplicate_selection(self):
        path = self.members[0][1]/'analysis.json'
        report = read_json(path)
        report['fits'][0]['n_entries_in_histogram'] += 1
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, 'count total'):
            load_member(*self.members[0])
        with self.assertRaisesRegex(ValueError, 'same data selection'):
            compare(self.root, [('a', self.members[1][1]), ('b', self.members[1][1])], 'duplicates')

    def test_unavailable_fit_keeps_histogram_and_no_fit_label(self):
        path = self.members[0][1]/'analysis.json'
        report = read_json(path)
        report['fits'][0] = dict(sample_kind='data', status='no_fit', reason='insufficient entries',
                                n_entries_in_histogram=report['fits'][0]['n_entries_in_histogram'])
        write_json(path, report)
        member = load_member(*self.members[0])
        self.assertEqual(member['fit']['status'], 'no_fit')
        self.assertAlmostEqual(np.sum(member['fraction']), 1)


if __name__ == '__main__':
    unittest.main()
