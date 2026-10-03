"""Batch/resume and merge tests with repeated source IDs across FLOW files."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from batch_track_selection import run_batch
import merge_track_outputs as merger
import test_track_output_hdf5 as fixtures


class BatchMergeTests(unittest.TestCase):
    def setUp(self):
        # Reuse the FLOW fixture without inheriting/rerunning selector tests.
        self.fixture = fixtures.OutputTests()
        self.fixture.setUp()
        self.fixture.fixture()
        self.root = self.fixture.root.resolve()
        self.first = self.fixture.source.resolve()
        self.second = self.root/'second.h5'
        shutil.copyfile(self.first,self.second)
        # Different run metadata must remain distinct in merged output.
        with h5py.File(self.second,'r+') as f:
            f['run_info'].attrs['crs_ticks'] = 0.2
        self.selector = fixtures.import_selector()
        self.calls = []

    def tearDown(self):
        self.fixture.tearDown()

    def runner(self, source, output, is2x2, log):
        self.calls.append(str(source))
        with log.open('w') as stream, redirect_stdout(stream):
            self.selector.run(str(source),str(output),is2x2)

    def produce(self, optional_missing=False, second_empty=False):
        if optional_missing:
            with h5py.File(self.second,'r+') as f:
                del f['combined/t0']
                del f['charge/events/ref/combined/t0']
                del f['charge/packets']
                del f['charge/calib_prompt_hits/ref/charge/packets']
        if second_empty:
            with h5py.File(self.second,'r+') as f:
                hits = f['charge/calib_prompt_hits/data'][:]
                hits['x'] = np.nan
                f['charge/calib_prompt_hits/data'][:] = hits
        with redirect_stdout(io.StringIO()):
            return run_batch([self.first,self.second], self.root/'batch', runner=self.runner)

    def test_merge_offsets_file_ids_and_metadata(self):
        outputs = self.produce()
        merged = self.root/'merged.h5'
        merger.merge_outputs(outputs,merged,chunk_rows=17)
        with h5py.File(merged) as f:
            hits, events, tracks = f['events/hits'][:], f['events/data'][:], f['tracks/data'][:]
            n = len(hits)//2
            np.testing.assert_array_equal(tracks['track_id'],[0,1,2,3])
            np.testing.assert_array_equal(events['file_id'],[0,1])
            np.testing.assert_array_equal(events['hit_start'],[0,n])
            np.testing.assert_array_equal(events['hit_stop'],[n,2*n])
            self.assertEqual(set(hits['selected_track_id'][n:]),{-1,2,3})
            np.testing.assert_array_equal(hits['id'][:n],hits['id'][n:])
            np.testing.assert_array_equal(hits['source_hit_row'][:n],hits['source_hit_row'][n:])
            self.assertTrue((hits['file_id'][:n]==0).all())
            self.assertTrue((hits['file_id'][n:]==1).all())
            refs = f['events/hit_packet_refs'][:]
            self.assertTrue((refs['hit_row'][refs['file_id']==1]>=n).all())
            self.assertTrue((refs['packet_row'][refs['file_id']==1]>=n+1).all())
            self.assertEqual(f['sources/0/metadata/run_info'].attrs['crs_ticks'],.1)
            self.assertEqual(f['sources/1/metadata/run_info'].attrs['crs_ticks'],.2)
            self.assertEqual(f.attrs['status'],'complete')
        # Already completed batch+merge resumes without selecting or duplicating.
        with redirect_stdout(io.StringIO()):
            run_batch([self.first,self.second],self.root/'batch',resume=True,
                      merged_output=merged,runner=self.runner)
        self.assertEqual(len(self.calls),2)

    def test_empty_input_and_optional_table_availability(self):
        outputs = self.produce(optional_missing=True,second_empty=True)
        merged = self.root/'merged.h5'
        merger.merge_outputs(outputs,merged,chunk_rows=13)
        with h5py.File(merged) as f:
            self.assertEqual(len(f['sources']),2)
            self.assertFalse(f['sources/1'].attrs['events/packets_available'])
            self.assertEqual(len(f['tracks/data']),2)
            self.assertEqual(len(f['events/t0']),1)

    def test_selected_input_without_packets_merges_in_either_order(self):
        outputs = self.produce(optional_missing=True)
        for order, suffix in [(outputs,'forward'),(outputs[::-1],'reverse')]:
            merged = self.root/f'{suffix}.h5'
            merger.merge_outputs(order,merged,chunk_rows=7)
            with h5py.File(merged) as f:
                self.assertEqual(len(f['tracks/data']),4)
                self.assertEqual(len(f['events/hits']),484)
                self.assertEqual(len(f['events/packets']),243)
                missing_id = 1 if suffix=='forward' else 0
                self.assertFalse(f[f'sources/{missing_id}'].attrs['events/packets_available'])
                rows = f['events/hits'][:]
                self.assertTrue((rows['packet_link_count'][rows['file_id']==missing_id]==-1).all())

    def test_duplicate_and_incompatible_inputs_refused(self):
        outputs = self.produce()
        with self.assertRaisesRegex(ValueError,'Duplicate FLOW'):
            merger.merge_outputs([outputs[0],outputs[0]],self.root/'duplicate.h5')
        with h5py.File(outputs[1],'r+') as f:
            del f['events/hits']
            f.create_dataset('events/hits',shape=(0,),dtype=[('file_id','i8')])
        with self.assertRaisesRegex(ValueError,'Incompatible schema'):
            merger.merge_outputs(outputs,self.root/'bad.h5')
        self.assertFalse((self.root/'bad.h5.partial').exists())

    def test_merge_interruption_rolls_back_entire_input(self):
        outputs = self.produce()
        merged = self.root/'merged.h5'
        real_append = merger.append
        def interrupted(dataset,data):
            real_append(dataset,data)
            # Interrupt partway through second input after first was committed.
            if dataset.name=='/events/hits' and np.any(data['file_id']==1):
                raise RuntimeError('injected interruption')
        with patch.object(merger,'append',interrupted):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                merger.merge_outputs(outputs,merged,chunk_rows=19)
        self.assertFalse(merged.exists())
        with h5py.File(str(merged)+'.partial') as f:
            self.assertEqual(f.attrs['committed_inputs'],1)
            self.assertEqual(f.attrs['status'],'incomplete')
        merger.merge_outputs(outputs,merged,resume=True,chunk_rows=11)
        with h5py.File(merged) as f:
            self.assertEqual(len(f['events/data']),2)
            self.assertEqual(len(f['tracks/data']),4)
            self.assertEqual(len(f['events/hits']),484)
            self.assertEqual(len(f['events/hit_packet_refs']),486)

    def test_resume_recovers_after_commit_marker_before_root_counter(self):
        outputs = self.produce()
        merged = self.root/'merged.h5'
        merger.merge_outputs(outputs,merged)
        partial = Path(str(merged)+'.partial')
        merged.rename(partial)
        with h5py.File(partial,'r+') as f:
            f.attrs['status'] = 'incomplete'
            f.attrs['committed_inputs'] = 0
        merger.merge_outputs(outputs,merged,resume=True)
        with h5py.File(merged) as f:
            self.assertEqual(len(f['tracks/data']),4)
            self.assertEqual(len(f['events/hits']),484)
            self.assertEqual(f.attrs['committed_inputs'],2)

    def test_failed_batch_input_is_rerun_without_duplicate_completed_inputs(self):
        attempts = []
        def fail_second(source,output,is2x2,log):
            attempts.append(str(source))
            if source == self.second:
                with h5py.File(output,'w') as f:
                    f.attrs['status'] = 'incomplete'
                raise RuntimeError('interrupted selection')
            self.runner(source,output,is2x2,log)
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                run_batch([self.first,self.second],self.root/'batch',runner=fail_second)
            outputs = run_batch([self.first,self.second],self.root/'batch',resume=True,runner=self.runner)
        self.assertEqual(self.calls,[str(self.first),str(self.second)])
        self.assertEqual(len(outputs),2)
        self.assertEqual(len(list((self.root/'batch').glob('*.failed-*'))),1)
        manifest = json.loads((self.root/'batch/batch_manifest.json').read_text())
        self.assertEqual(manifest['status'],'complete')

    def test_resume_rejects_changed_source_and_changed_code(self):
        outputs = self.produce()
        with h5py.File(outputs[0],'r+') as f:
            f['sources/0/metadata'].attrs['efield.py_sha256'] = 'different'
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'code changed'):
            run_batch([self.first,self.second],self.root/'batch',resume=True,runner=self.runner)
        with h5py.File(self.second,'r+') as f:
            f['run_info'].attrs['changed'] = True
        with self.assertRaisesRegex(ValueError,'Resume input'):
            run_batch([self.first,self.second],self.root/'batch',resume=True,runner=self.runner)

    def test_batch_duplicate_paths_refused(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            run_batch([self.first,self.first],self.root/'batch',runner=self.runner)


if __name__ == '__main__':
    unittest.main()
