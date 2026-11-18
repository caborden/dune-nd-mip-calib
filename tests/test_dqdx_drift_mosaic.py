import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from analysis.dqdx_drift_mosaic import histograms, run
from analysis.segments import SEGMENT_DTYPE


class MosaicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / 'segments.hdf5'
        with h5py.File(self.path, 'w') as f:
            f.attrs.update(status='complete', settings=json.dumps({'segment_length_cm': 3}))
            rows = []
            for sample_id, kind in enumerate(('data', 'mc')):
                sample = f.require_group(f'samples/{sample_id}')
                sample.attrs.update(sample_kind=kind, input_file=f'{kind}.hdf5')
                for source_id, period in enumerate((.1, .2)):
                    meta = sample.require_group(f'sources/{source_id}/metadata/run_info')
                    meta.attrs['crs_ticks'] = period
                    # Identical physical times with different clock periods.
                    for t, q in ((0, 30), (50, 35), (500, 90), (-1, 30), (501, 30),
                                 (100, -1), (100, 91), (np.nan, 30)):
                        row = np.zeros((), dtype=SEGMENT_DTYPE)
                        row['sample_id'], row['file_id'] = sample_id, source_id
                        row['t_drift'], row['dqdx'] = t / period, q
                        rows.append(row)
            f.create_dataset('segments/data', data=np.array(rows, dtype=SEGMENT_DTYPE))

    def tearDown(self):
        self.tmp.cleanup()

    def test_conversion_boundaries_and_audit(self):
        counts, audit = histograms(self.path, 'data')
        self.assertEqual(counts.sum(), 6)
        self.assertEqual(counts.sum(axis=1).tolist(), [2, 2, 0, 0, 0, 0, 0, 0, 0, 2])
        for name in ('nonfinite', 'drift_underflow', 'drift_overflow', 'dqdx_underflow', 'dqdx_overflow'):
            self.assertEqual(audit[name], 2)
        self.assertEqual(audit['total'], 16)

    def test_missing_clock_and_incomplete_rejected(self):
        with h5py.File(self.path, 'a') as f:
            del f['samples/0/sources/0/metadata/run_info'].attrs['crs_ticks']
        with self.assertRaisesRegex(ValueError, 'tick period'):
            histograms(self.path, 'data')
        with h5py.File(self.path, 'a') as f:
            f.attrs['status'] = 'incomplete'
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            histograms(self.path, 'mc')

    def test_plot_exports_empty_bins_and_no_overwrite(self):
        out = self.root / 'plot'
        result = run(self.path, self.path, out, label='PRC8', write_png=True, write_csv=True)
        self.assertEqual(result['status'], 'complete')
        for filename in ('dqdx_drift_mosaic.pdf', 'dqdx_drift_mosaic.png', 'analysis.json', 'histograms.csv'):
            self.assertTrue((out / filename).is_file())
        with self.assertRaises(FileExistsError):
            run(self.path, self.path, out)


if __name__ == '__main__':
    unittest.main()
