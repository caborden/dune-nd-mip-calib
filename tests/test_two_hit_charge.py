"""Timing boundaries, chronological charges, common normalization and rendering."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from analysis.io import read_json, write_json
from analysis.two_hit_charge import run, summarize, histogram_fwhm

DTYPE = [(name, 'i8') for name in ['io_group', 'io_channel', 'chip_id', 'channel_id',
                                  'file_id', 'source_event_row', 'selected_track_id']]
DTYPE += [('ts_pps', 'u8'), ('Q', 'f8')]


def fixture(path):
    # Each tuple is a separate pixel. First stored record is the later hit.
    pairs = [(25, 12, 3), (40, 18, 2), (41, 30, 4), (24, 6, 1),
             (0, 7, 8), (50, np.nan, 9), (45, 210, -2)]
    hits = np.zeros(17, dtype=DTYPE)
    for i, (delta, later, earlier) in enumerate(pairs):
        hits['channel_id'][2*i:2*i+2] = i
        hits['ts_pps'][2*i:2*i+2] = [2**60+delta, 2**60]
        hits['Q'][2*i:2*i+2] = [later, earlier]
    hits['selected_track_id'] = -1  # Track-rejected records still participate.
    hits['channel_id'][-3:] = 99  # Never turn a three-hit pixel into a pair.
    events = np.array([(0, 0, 0, len(hits))], dtype=[(n, 'i8') for n in
                      ['file_id', 'source_event_row', 'hit_start', 'hit_stop']])
    with h5py.File(path, 'w') as f:
        f.attrs.update(schema_version='2.0', status='complete')
        f.create_dataset('events/data', data=events)
        f.create_dataset('events/hits', data=hits)
        f.require_group('sources/0/metadata/run_info').attrs['crs_ticks'] = .1


class TwoHitChargeTests(unittest.TestCase):
    def test_fwhm_interpolation_and_clipping(self):
        self.assertAlmostEqual(histogram_fwhm([0, 2, 6, 2, 0], np.arange(6.)), 1.5)
        self.assertAlmostEqual(histogram_fwhm([0, 4, 4, 0], np.arange(5.)), 2.)
        self.assertIsNone(histogram_fwhm([4, 2, 0], np.arange(4.)))
        self.assertIsNone(histogram_fwhm([0, 2, 4], np.arange(4.)))
        self.assertIsNone(histogram_fwhm([0, 0], np.arange(3.)))

    def test_boundaries_order_and_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'selection.hdf5'
            fixture(path)
            result = summarize(path, np.arange(0., 201.))
            cats = result['categories']
            self.assertEqual(result['audit']['pixels_more_than_two_hits'], 1)
            self.assertEqual(cats['all']['instances'], 7)
            self.assertEqual(cats['dt25_40']['instances'], 2)
            self.assertEqual(cats['dt_gt40']['instances'], 3)
            np.testing.assert_array_equal(np.flatnonzero(cats['dt25_40']['counts']['first']), [2, 3])
            np.testing.assert_array_equal(np.flatnonzero(cats['dt25_40']['counts']['second']), [12, 18])
            np.testing.assert_array_equal(np.flatnonzero(cats['dt25_40']['counts']['total']), [15, 20])
            self.assertEqual(cats['all']['counts']['first'][7], 1)  # Stable tied order.
            tail = cats['dt_gt40']
            self.assertEqual(tail['invalid_charge_pairs'], 1)
            self.assertEqual(tail['underflow']['first'], 1)
            self.assertEqual(tail['overflow']['second'], 1)
            self.assertEqual(tail['overflow']['total'], 1)
            self.assertEqual(tail['counts']['total'].sum(), 1)
            self.assertEqual(tail['statistics']['total']['mean'], (34+208)/2)
            # Histogram final edge is inclusive.
            short = summarize(path, np.array([0., 10., 20.]))['categories']['dt25_40']
            np.testing.assert_array_equal(short['counts']['total'], [0, 2])

    def test_render_csv_manifest_and_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            refs = {}
            for key, kind in [('data_prc16', 'data'), ('mc', 'mc')]:
                sample = root/'samples'/key
                sample.mkdir(parents=True)
                fixture(sample/'selection.hdf5')
                write_json(sample/'sample.json', dict(kind=kind, selection_versions={'v1': {'path': 'selection.hdf5'}}))
                refs[key] = dict(id=key, kind=kind, label=key)
            config = root/'recipe.json'
            write_json(config, dict(target=dict(kind='comparison', id='test'), samples=refs))
            for name, flags, expected in [('default', {}, {'.pdf'}),
                    ('png', {'write_png': True}, {'.pdf', '.png'}),
                    ('csv', {'write_csv': True}, {'.pdf', '.csv'})]:
                directory = run(config, root, name, bins=20, **flags)
                self.assertEqual({p.suffix for p in (directory/'two_hit_charge').iterdir()}, expected)
            directory = run(config, root, 'test', bins=20, write_csv=True, write_png=True)
            manifest = read_json(directory/'manifest.json')
            self.assertEqual(manifest['status'], 'complete')
            cat = manifest['samples']['data_prc16']['categories']['dt_gt40']
            self.assertEqual(cat['instances'], 3)
            table = np.genfromtxt(directory/'two_hit_charge/data_prc16_dt_gt40_total.csv', delimiter=',', names=True)
            self.assertAlmostEqual(table['fraction'].sum(), 1/3)
            for stem in ['total_charge_by_separation', 'data_prc16_charge_breakdown',
                         'mc_charge_breakdown', 'data_prc16_mc_charge_overlay']:
                for ext in ['png', 'pdf']:
                    self.assertGreater((directory/'two_hit_charge'/f'{stem}.{ext}').stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                run(config, root, 'test', write_csv=True, write_png=True)
            with self.assertRaisesRegex(ValueError, 'Overlay data key'):
                run(config, root, 'bad', overlay_data='missing', write_csv=True, write_png=True)

    def test_missing_charge_and_nonfinite_times(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'selection.hdf5'
            fixture(path)
            with h5py.File(path, 'r+') as f:
                old = f['events/hits'][:]
                dtype = [(n, 'f8' if n == 'ts_pps' else old.dtype[n]) for n in old.dtype.names]
                hits = np.zeros(len(old), dtype=dtype)
                for n in old.dtype.names:
                    hits[n] = old[n]
                hits['ts_pps'][0] = np.nan
                del f['events/hits']
                f.create_dataset('events/hits', data=hits)
            result = summarize(path, np.arange(201.))
            self.assertEqual(result['audit']['pairs_nonfinite_time'], 1)
            self.assertEqual(result['categories']['all']['instances'], 6)
            with h5py.File(path, 'r+') as f:
                hits = f['events/hits'][:][[n for n in hits.dtype.names if n != 'Q']]
                del f['events/hits']
                f.create_dataset('events/hits', data=hits)
            with self.assertRaisesRegex(ValueError, 'missing hit fields'):
                summarize(path, np.arange(201.))


if __name__ == '__main__':
    unittest.main()
