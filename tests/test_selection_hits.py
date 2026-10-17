import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from analysis.selection_hits import summarize, make_settings
from analysis.run import run
from analysis.io import write_json, read_json


class SelectionHitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample = self.root/'samples/test'
        self.sample.mkdir(parents=True)
        self.path = self.sample/'selection.hdf5'
        dtype = [(k, 'i8') for k in ['file_id','source_event_row','selected_track_id','io_group','io_channel','chip_id','channel_id']] + [('Q','f8')]
        rows = np.zeros(9, dtype=dtype)
        rows['Q'] = [1,2,3,100,101,-1,np.nan,5,6]
        rows['selected_track_id'] = [0,1,-1,2,2,2,2,2,2]
        rows['source_event_row'][3:] = 1
        rows['io_channel'][4] = 1
        rows['chip_id'][5] = 1
        rows['channel_id'][6] = 1
        rows['chip_id'][7] = -1
        rows['file_id'][8] = 1
        with h5py.File(self.path, 'w') as f:
            f.attrs.update(status='complete', schema_version='2.0')
            f.create_dataset('events/hits', data=rows)
            events = np.array([(0,0,0,3),(0,1,3,8),(1,1,8,9)], dtype=[(k,'i8') for k in ['file_id','source_event_row','hit_start','hit_stop']])
            f.create_dataset('events/data', data=events)
        write_json(self.sample/'sample.json', dict(kind='data',selection_versions={'v1':{'path':'selection.hdf5'}}))
        self.config = self.root/'config.json'
        write_json(self.config,dict(target={'kind':'sample','id':'test'},run_modules=['selection_hits']))

    def tearDown(self):
        self.temp.cleanup()

    def test_counts_and_scope(self):
        result = summarize(self.path, make_settings())
        self.assertEqual(result['multiplicity'], {1:5, 2:1})
        self.assertEqual(sum(result['charge_counts']), 5)
        self.assertEqual(result['audit'], dict(selected_hits=8,nonfinite_charge=1,charge_underflow=1,charge_overflow=1,invalid_pixel_hits=1,events=3))

    def test_runner_resume_and_rerun(self):
        directory = run(self.config, self.root, 'run-001')
        report = read_json(directory/'selection_hits/analysis.json')
        self.assertEqual(report['status'], 'complete')
        before = read_json(directory/'manifest.json')['stages']['selection_hits']['generation']
        run(self.config, self.root, 'run-001', resume=True)
        self.assertEqual(read_json(directory/'manifest.json')['stages']['selection_hits']['generation'], before)
        run(self.config, self.root, 'run-001', rerun=['selection_hits'])
        self.assertNotEqual(read_json(directory/'manifest.json')['stages']['selection_hits']['generation'], before)
        self.assertTrue((directory/'selection_hits/multiplicity.pdf').is_file())

    def test_missing_electronics_rejected(self):
        with h5py.File(self.path, 'r+') as f:
            del f['events/hits']
            f.create_dataset('events/hits', data=np.zeros(0,dtype=[('Q','f8')]))
        with self.assertRaises(ValueError):
            summarize(self.path, make_settings())
