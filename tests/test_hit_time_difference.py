"""Two-hit multiplicity, event/source boundaries, and exact timestamp arithmetic."""
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch
import unittest

import h5py
import numpy as np

from analysis.hit_time_difference import pixel_deltas, summarize, run
from analysis.io import write_json, read_json

DTYPE = np.dtype([(n, 'i8') for n in ['io_group', 'io_channel', 'chip_id', 'channel_id',
                                    'file_id', 'source_event_row', 'selected_track_id']] + [('ts_pps', 'u8')])


class HitTimeDifferenceTests(unittest.TestCase):
    def test_exact_integer_delta_and_all_hit_multiplicity(self):
        hits = np.zeros(7, dtype=DTYPE)
        hits['channel_id'] = [1, 1, 2, 2, 2, 3, 3]
        hits['ts_pps'] = 2**60 + np.array([9, 2, 1, 2, 3, 4, 4], dtype='u8')
        hits['selected_track_id'] = [0, -1, 0, 0, -1, 0, 0]
        delta, audit = pixel_deltas(hits, 'ts_pps')
        np.testing.assert_array_equal(delta, [7, 0])
        self.assertEqual(audit['pairs_both_selected'], 1)
        self.assertEqual(audit['pixels_more_than_two_hits'], 1)
        hits['io_group'][-1] = 2
        delta, _ = pixel_deltas(hits, 'ts_pps')
        np.testing.assert_array_equal(delta, [7])

    def test_nonfinite_hits_do_not_change_multiplicity(self):
        dtype = np.dtype([(n, 'f8' if n == 'ts_pps' else DTYPE[n]) for n in DTYPE.names])
        hits = np.zeros(5, dtype=dtype)
        hits['channel_id'] = [1, 1, 1, 2, 2]
        hits['ts_pps'] = [0, 3, np.nan, 5, np.nan]
        delta, audit = pixel_deltas(hits, 'ts_pps')
        self.assertEqual(len(delta), 0)
        self.assertEqual(audit['pixels_more_than_two_hits'], 1)
        self.assertEqual(audit['pairs_nonfinite_time'], 1)

    def test_event_streaming_render_and_overflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = np.zeros(3, dtype=[(n, 'i8') for n in ['file_id', 'source_event_row', 'hit_start', 'hit_stop']])
            events['hit_start'] = [0, 4, 6]
            events['hit_stop'] = [4, 6, 8]
            events['file_id'] = [0, 0, 1]
            events['source_event_row'] = [0, 1, 0]
            hits = np.zeros(8, dtype=DTYPE)
            hits['file_id'] = [0, 0, 0, 0, 0, 0, 1, 1]
            hits['source_event_row'] = [0, 0, 0, 0, 1, 1, 0, 0]
            hits['channel_id'] = [0, 0, 1, 1, 0, 0, 0, 0]
            hits['ts_pps'] = [1, 4, 1, 4, 1, 6, 2, 22]
            for kind in ['data', 'mc']:
                path = root/'samples'/kind/'selection.hdf5'
                path.parent.mkdir(parents=True)
                with h5py.File(path, 'w') as f:
                    f.attrs.update(status='complete', schema_version='2.0')
                    f.create_dataset('events/data', data=events)
                    f.create_dataset('events/hits', data=hits)
                    for file_id in [0, 1]:
                        f.require_group(f'sources/{file_id}/metadata/run_info').attrs['crs_ticks'] = .1
                write_json(path.parent/'sample.json', dict(kind=kind, selection_versions={'v1': {'path': path.name}}))
            result = summarize(path, np.array([0., 5., 10.]))
            np.testing.assert_array_equal(result['counts'], [2, 1])
            self.assertEqual(result['audit']['pairs_above_range'], 1)
            write_json(root/'comparisons/pair/comparison.json', dict(samples={kind: {'id': kind} for kind in ['data','mc']}))
            config = root/'config.json'
            write_json(config, dict(target={'kind': 'comparison', 'id': 'pair'}))
            output = run(config, root, 'test', bins=2, max_delta_ticks=10)
            self.assertEqual(read_json(output/'manifest.json')['status'], 'complete')
            self.assertEqual(read_json(output/'manifest.json')['samples']['data']['normalization_instances'], 4)
            table = np.genfromtxt(output/'hit_time_difference/data_delta_t.csv', delimiter=',', names=True)
            # Four pixel/event instances in three events; overflow remains in N.
            # Width is five ticks, but there is no division by bin width.
            np.testing.assert_allclose(table['fraction'], [.5, .25])
            self.assertGreater((output/'hit_time_difference/hit_time_difference.png').stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                run(config, root, 'test')
            # Three distinct data inputs plus one MC, with different bin populations.
            refs = {}
            for reset, delta in [('prc2', 2), ('prc8', 6), ('prc16', 20)]:
                sample_dir = root/'samples'/reset
                shutil.copytree(root/'samples/data', sample_dir)
                with h5py.File(sample_dir/'selection.hdf5', 'r+') as f:
                    rows = f['events/hits'][:]
                    rows['ts_pps'][1] = rows['ts_pps'][0] + delta
                    f['events/hits'][:] = rows
                refs[reset] = dict(kind='data', id=reset, label='Data '+reset)
            refs['mc'] = dict(kind='mc', id='mc', label='MC prc256')
            combined = root/'combined.json'
            write_json(combined, dict(target=dict(kind='comparison', id='combined'), samples=refs))
            with patch('analysis.hit_time_difference.summarize', wraps=summarize) as scan:
                result_dir = run(combined, root, 'overlay', bins=2, max_delta_ticks=10)
                self.assertEqual(scan.call_count, 4)
                self.assertEqual(sum(call.args[0] == path.resolve() for call in scan.call_args_list), 1)
            manifest = read_json(result_dir/'manifest.json')
            self.assertEqual(set(manifest['samples']), set(refs))
            for key, expected in [('prc2', [.5, .25]), ('prc8', [.25, .5]),
                                  ('prc16', [.25, .25]), ('mc', [.5, .25])]:
                table = np.genfromtxt(result_dir/'hit_time_difference'/f'{key}_delta_t.csv',
                                     delimiter=',', names=True)
                np.testing.assert_allclose(table['fraction'], expected)
            self.assertEqual(read_json(root/'comparisons/combined/comparison.json')['samples'], refs)
            # A rerun with a fresh ID reuses the matching registry.
            run(combined, root, 'overlay-again', bins=2, max_delta_ticks=10)
            refs['prc2']['label'] = 'Changed label'
            write_json(combined, dict(target=dict(kind='comparison', id='combined'), samples=refs))
            with self.assertRaisesRegex(FileExistsError, 'registry differs'):
                run(combined, root, 'conflict')
            with h5py.File(path, 'r+') as f:
                f['events/data'][0] = (0, 0, 0, 5)
            with self.assertRaisesRegex(ValueError, 'crosses source'):
                summarize(path, np.array([0., 10.]))


if __name__ == '__main__':
    unittest.main()
