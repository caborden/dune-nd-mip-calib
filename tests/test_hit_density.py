"""Analytic binning and correlation checks for drift-time hit density."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from analysis.hit_density import make_settings, summarize, ratios, plot_segments
from analysis.segments import SEGMENT_DTYPE


class HitDensityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root/'segments.hdf5'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows, kinds=(0,)):
        """Rows: sample, file, event, track, time, nhits, dx, group."""
        values = np.zeros(len(rows), dtype=SEGMENT_DTYPE)
        for i, row in enumerate(rows):
            for field, value in zip(['sample_id', 'file_id', 'source_event_row', 'track_id',
                                     't_drift', 'nhits', 'dx', 'group'], row):
                values[field][i] = value
        with h5py.File(self.path, 'w') as f:
            f.attrs['status'] = 'complete'
            f.attrs['settings'] = json.dumps(dict(segment_length_cm=3., step_cm=3.))
            f.create_dataset('segments/data', data=values)
            for identity in kinds:
                group = f.create_group(f'samples/{identity}')
                group.attrs['sample_kind'] = 'data' if identity == 0 else 'mc'
                group.attrs['input_file'] = f'input-{identity}.hdf5'

    def test_edges_segment_weighting_event_scoping_and_audits(self):
        # Track identity repeats across files and events; readout groups share a track.
        self.write([(0,0,0,7,0,4,2,b'io1'), (0,0,0,7,5,8,2,b'io2'),
                    (0,1,0,7,0,2,2,b'io1'), (0,0,0,7,10,10,2,b'io1'),
                    (0,1,0,7,20,6,2,b'io1'), (0,0,1,7,20,8,2,b'io1'),
                    (0,0,0,7,-1,9,2,b'io1'), (0,0,0,7,21,9,2,b'io1'),
                    (0,0,0,7,np.nan,9,2,b'io1')])
        settings = make_settings(drift_edges_ticks=[0,10,20], uncertainty='segment-sem')
        samples, _ = summarize(self.path, settings, chunk_rows=2)
        data = samples[0]
        self.assertEqual(data['n_tracks'], 3)
        self.assertEqual(data['audit'], dict(total_segments=9, included_segments=6,
                         below_range=1, above_range=1, invalid_segments=1))
        self.assertEqual([b['n_segments'] for b in data['bins']], [3,3])
        self.assertEqual([b['n_tracks'] for b in data['bins']], [2,3])
        self.assertAlmostEqual(data['bins'][0]['mean'], 14/3)
        self.assertEqual(data['bins'][1]['mean'], 8)
        self.assertEqual(data['bins'][0]['hit_count_distribution'],
                         dict(nhits=[2,4,8], counts=[1,1,1], mean_nhits=14/3, mode_nhits=[2,4,8]))
        self.assertEqual(data['bins'][1]['hit_count_distribution']['nhits'], [6,8,10])
        self.assertAlmostEqual(data['bins'][0]['error'], np.std([4,8,2], ddof=1)/np.sqrt(3))
        # Same raw counts; hits/cm divides by physical dx, not the nominal 3 cm.
        per_cm, _ = summarize(self.path, make_settings(drift_edges_ticks=[0,10,20],
                              observable='hits-per-cm', uncertainty='none'))
        self.assertAlmostEqual(per_cm[0]['bins'][0]['mean'], 7/3)
        self.assertIsNone(per_cm[0]['bins'][0]['error'])

    def test_whole_track_errors_determinism_and_single_track_bins(self):
        rows = [(0,0,0,0,5,2,3,b'io1')]*20 + [(0,0,0,1,5,8,3,b'io1')]*20
        rows += [(0,0,0,0,15,4,3,b'io1')]
        self.write(rows)
        settings = make_settings(drift_edges_ticks=[0,10,20,30], bootstrap_samples=2000)
        samples, _ = summarize(self.path, settings, chunk_rows=3)
        repeat, _ = summarize(self.path, settings, chunk_rows=100)
        self.assertEqual(samples, repeat)
        first = samples[0]['bins'][0]
        self.assertEqual(first['mean'], 5)
        self.assertGreater(first['error'], 1.8)
        self.assertLess(first['segment_sem'], .5)
        self.assertIsNone(samples[0]['bins'][1]['error'])  # one independent track
        self.assertEqual(samples[0]['bins'][1]['mean'], 4)
        self.assertIsNone(samples[0]['bins'][2]['mean'])  # empty bin stays empty
        threshold, _ = summarize(self.path, make_settings(drift_edges_ticks=[0,10,20,30],
                                  min_segments=2, uncertainty='none'))
        self.assertIsNone(threshold[0]['bins'][1]['mean'])

    def test_sample_streams_ratio_and_mc_only_outputs(self):
        rows = [(0,0,0,0,5,4,3,b'io1'), (0,0,0,1,5,8,3,b'io1'),
                (1,0,0,0,5,2,3,b'io1'), (1,0,0,1,5,4,3,b'io1')]
        self.write(rows, (0,1))
        settings = make_settings(drift_edges_ticks=[0,10,20], bootstrap_samples=200)
        paired, _ = summarize(self.path, settings)
        ratio = ratios(paired)
        self.assertEqual(ratio[0]['ratio'], 2)
        expected = np.hypot(paired[0]['bins'][0]['error']/3,
                            6*paired[1]['bins'][0]['error']/9)
        self.assertAlmostEqual(ratio[0]['error'], expected)
        self.assertIsNone(ratio[1]['ratio'])
        before = paired[1]
        self.write(rows[2:], (1,))
        mc, _ = summarize(self.path, settings)
        self.assertEqual(mc[0], before)
        result = plot_segments(self.path, self.root/'mc', settings)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue((self.root/'mc/mc_hit_density.pdf').exists())
        self.assertTrue((self.root/'mc/mc_hit_density.csv').exists())
        self.assertTrue((self.root/'mc/hit_count_distributions.png').exists())
        histogram = np.loadtxt(self.root/'mc/mc_hit_count_distributions.csv', delimiter=',', skiprows=1)
        np.testing.assert_allclose(histogram, [[0,10,2,1,.5], [0,10,4,1,.5]])
        self.assertFalse((self.root/'mc/data_mc_hit_density.png').exists())
        # No NaN tokens in JSON; missing observations/errors are null.
        saved = (self.root/'mc/analysis.json').read_text()
        self.assertNotIn('NaN', saved)

    def test_skewed_distributions_and_paired_grid(self):
        self.write([(0,0,0,0,5,n,3,b'io1') for n in [2,2,2,10]] +
                   [(1,0,0,0,5,n,3,b'io1') for n in [4,4,8]] +
                   [(0,0,0,0,5000,6,3,b'io1'), (1,0,0,0,5001,99,3,b'io1')], (0,1))
        result = plot_segments(self.path, self.root/'paired', make_settings(uncertainty='none'))
        data, mc = result['samples']
        self.assertEqual(data['bins'][0]['hit_count_distribution'],
                         dict(nhits=[2,10], counts=[3,1], mean_nhits=4., mode_nhits=[2]))
        self.assertEqual(mc['bins'][0]['hit_count_distribution']['mode_nhits'], [4])
        self.assertEqual(data['bins'][-1]['hit_count_distribution']['nhits'], [6])
        self.assertEqual(mc['bins'][-1]['hit_count_distribution']['nhits'], [])
        self.assertEqual(mc['audit']['above_range'], 1)
        self.assertTrue((self.root/'paired/hit_count_distributions.pdf').exists())
        rows = np.loadtxt(self.root/'paired/data_hit_count_distributions.csv', delimiter=',', skiprows=1)
        self.assertEqual(rows[rows[:,0] == 0,4].sum(), 1.)

    def test_empty_sample_invalid_dx_and_ratio_zero_handling(self):
        self.write([], (0,1))
        result = plot_segments(self.path, self.root/'empty', make_settings(bootstrap_samples=20))
        self.assertTrue(all(b['mean'] is None for s in result['samples'] for b in s['bins']))
        self.assertTrue((self.root/'empty/data_mc_hit_density_ratio.png').exists())
        self.write([(0,0,0,0,5,4,0,b'io1')])
        values, _ = summarize(self.path, make_settings(observable='hits-per-cm'))
        self.assertEqual(values[0]['audit']['invalid_segments'], 1)
        fake = [dict(sample_kind='data', bins=[dict(bin_low=0,bin_high=1,bin_center=.5,mean=0,error=1)]),
                dict(sample_kind='mc', bins=[dict(bin_low=0,bin_high=1,bin_center=.5,mean=2,error=.5)])]
        self.assertEqual(ratios(fake)[0]['error'], .5)
        fake[1]['bins'][0]['mean'] = 0
        self.assertIsNone(ratios(fake)[0]['ratio'])

    def test_invalid_configuration_and_incomplete_tables(self):
        for options in [dict(drift_edges_ticks=[0,0,10]), dict(drift_bins=0),
                        dict(drift_range_ticks=[5,0]), dict(bootstrap_samples=1),
                        dict(observable='pixels'), dict(uncertainty='unknown')]:
            with self.assertRaises(ValueError):
                make_settings(**options)
        self.write([], (0,))
        with h5py.File(self.path, 'r+') as f:
            f.attrs['status'] = 'incomplete'
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            summarize(self.path, make_settings())
