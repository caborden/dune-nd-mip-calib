"""Whole-track resampling, endpoint binning, ratio coupling, and resume tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from analysis.io import read_json
from analysis.mpv_drift import (read_tracks, resampled_histograms, errors, bootstrap,
                                checkpoint, run)


def simple_fit(counts):
    centers = np.linspace(.45, 89.55, 100)
    if not counts.sum():
        return dict(mpv=None, fit_covariance_se=None, chi2_ndf=None, status='no_fit')
    return dict(mpv=float(np.average(centers, weights=counts)), fit_covariance_se=.1,
                chi2_ndf=1., status='fit')


class MPVDriftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root/'segments.hdf5'
        dtype = [('sample_id','i4'), ('file_id','i8'), ('source_event_row','i8'),
                 ('track_id','i8'), ('t_drift','f8'), ('dqdx','f8')]
        rows = []
        for sample in (0, 1):
            for track in range(4):
                for t in (0., 100., 400., 5000., 5001.):
                    rows.append((sample, 0, track, track, t, 30.+track*2+sample))
        with h5py.File(self.path, 'w') as f:
            f.attrs.update(status='complete', settings='{}')
            f.create_dataset('segments/data', data=np.array(rows, dtype=dtype))
            for sample in (0, 1):
                group = f.require_group(f'samples/{sample}')
                group.attrs.update(sample_kind='data' if sample == 0 else 'mc', input_file=f'/selection/{sample}')
                group.require_group('sources/0/metadata/run_info').attrs['crs_ticks'] = .1

    def test_read_endpoints_and_track_population(self):
        matrix, audit = read_tracks(self.path, 'data')
        self.assertEqual(matrix.shape, (4, 1500))
        self.assertEqual(audit['segments'], 20)
        counts = np.asarray(matrix.sum(axis=0)).reshape(15, 100)
        self.assertEqual(counts[0].sum(), 8)
        self.assertEqual(counts[1].sum(), 4)
        self.assertEqual(counts[-1].sum(), 4)
        self.assertEqual(audit['excluded_segments'], 4)

    def test_one_track_weight_shared_across_bins(self):
        values = np.zeros((3, 1500))
        values[:, 0] = [1, 2, 3]
        values[:, 100] = [2, 4, 6]
        matrix = csr_matrix(values)
        for seed in range(5):
            hist = resampled_histograms(matrix, np.random.default_rng(seed))
            self.assertEqual(hist[1, 0], 2*hist[0, 0])
        draws = np.array([[1, 1], [3, np.nan], [5, np.nan]])
        se, success = errors(draws, [True, True], .9)
        self.assertEqual(se[0], 2)  # std(ddof=1), not std/sqrt(replicates)
        self.assertTrue(np.isnan(se[1]))
        self.assertEqual(success.tolist(), [3, 1])

    def test_exact_checkpoint_restart_and_shared_mc_ratios(self):
        matrix, _ = read_tracks(self.path, 'data')
        with patch('analysis.mpv_drift.fit_mpv', side_effect=simple_fit):
            first = bootstrap(matrix, 4, 123, self.root/'first.npz')
            checkpoint(self.root/'resume.npz', first.copy(), 2)
            resumed = bootstrap(matrix, 4, 123, self.root/'resume.npz', True)
            np.testing.assert_equal(first, resumed)
            second = self.root/'second.hdf5'
            import shutil
            shutil.copy2(self.path, second)
            with h5py.File(second, 'r+') as f:
                f['samples/0'].attrs['input_file'] = '/selection/second'
            directory = run(self.root, [('prc2', self.path), ('prc8', second)], self.path,
                            'test', repetitions=4)
            report = read_json(directory/'manifest.json')
            self.assertEqual(report['status'], 'complete')
            output = directory/'mpv_drift'
            mc = np.load(output/'MC-bootstrap.npz')['mpv']
            for label in ('prc2', 'prc8'):
                data = np.load(output/f'{label}-bootstrap.npz')['mpv']
                ratios = np.load(output/f'{label}-ratio-bootstrap.npz')['mpv']
                np.testing.assert_allclose(ratios, data/mc, equal_nan=True)
            self.assertGreater((output/'mpv_drift.png').stat().st_size, 1000)
            with self.assertRaisesRegex(ValueError, 'Existing run'):
                run(self.root, [('prc2', self.path), ('prc8', second)], self.path,
                    'test', repetitions=5, resume=True)

    def test_tick_period_mismatch_rejected(self):
        with h5py.File(self.path, 'r+') as f:
            f['samples/1/sources/0/metadata/run_info'].attrs['crs_ticks'] = .2
        with self.assertRaisesRegex(ValueError, 'tick period differs'):
            run(self.root, [('prc2', self.path)], self.path, 'bad', repetitions=2)
        self.assertFalse((self.root/'comparisons').exists())


if __name__ == '__main__':
    unittest.main()
