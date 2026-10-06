from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import h5py
import numpy as np
from scipy.stats import moyal

from analyze_track_dqdx import analyze, fit_counts, groups_for, automatic_bounds
from pixel_dqdx.segments import segment_track_hits
import test_track_output_hdf5 as fixtures


class DqdxTests(unittest.TestCase):
    def test_segment_charge_spacing_and_repeated_hit(self):
        t = np.arange(0.,10.,.25)
        xyz = np.column_stack([t, t/10, t/10])
        charge = np.full(len(t),8.)
        result = segment_track_hits(xyz,charge,t,3.,3.)
        self.assertEqual(len(result['dq']),3)
        # Interior inclusive upstream windows: 12 hits / 3.02985 cm.
        for nhits,dq in zip(result['nhits'],result['dq']):
            self.assertEqual(dq,8*nhits)
        np.testing.assert_allclose(result['dq']/result['dx'],8/(.25*np.sqrt(1.02)))
        # Every recorded selected hit contributes; no one-hit-per-pixel dedup.
        duplicated = segment_track_hits(np.vstack([xyz,xyz[3]]),np.append(charge,5),np.append(t,t[3]),3.,3.)
        self.assertEqual(duplicated['dq'][0],result['dq'][0]+5)

    def test_fit_recovers_upstream_component_mpv(self):
        rng = np.random.default_rng(7)
        values = moyal.rvs(loc=34,scale=3,size=8000,random_state=rng)+rng.normal(0,1,8000)
        counts,edges = np.histogram(values,bins=100,range=(0,90))
        result,reason = fit_counts(counts,edges,(20,50),100)
        self.assertIsNotNone(result,reason)
        self.assertAlmostEqual(result['params'][0],34,delta=1)
        self.assertGreater(result['chi2_red'],0)
        empty,reason = fit_counts(np.zeros(100),edges,(20,50),100)
        self.assertIsNone(empty)

    def test_merged_reader_excludes_rejected_charge_and_saves_two_panels(self):
        fixture = fixtures.OutputTests()
        fixture.setUp()
        try:
            fixture.fixture()
            selector = fixtures.import_selector()
            with redirect_stdout(io.StringIO()):
                selector.run(str(fixture.source),str(fixture.output),False)
            output_dir=fixture.root/'analysis'
            manifest = analyze(fixture.output,output_dir,mc=fixture.output,face_cuts="none")
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(len(manifest['samples']),2)
            self.assertTrue((output_dir/'data_mc_dqdx.png').exists())
            self.assertTrue((output_dir/'data_dqdx.pdf').exists())
            with h5py.File(output_dir/'segments.hdf5') as f:
                segments=f['segments/data'][:]
                self.assertEqual(set(segments['sample_id']),{0,1})
                # Accepted track uses charge=1; rejected same-pixel charge=3 excluded.
                np.testing.assert_allclose(segments['dq'],segments['nhits'])
                self.assertEqual(len(f['samples/0/sources']),1)
            with self.assertRaises(FileExistsError):
                analyze(fixture.output,output_dir)
        finally:
            fixture.tearDown()

    def test_mc_only_cli_labels_outputs_and_matches_data_only_numerics(self):
        fixture = fixtures.OutputTests()
        fixture.setUp()
        try:
            fixture.fixture()
            selector = fixtures.import_selector()
            with redirect_stdout(io.StringIO()):
                selector.run(str(fixture.source),str(fixture.output),False)
            mc_dir = fixture.root/'mc_only'
            command = [sys.executable,str(Path(__file__).resolve().parents[1]/'analyze_track_dqdx.py'),
                       '--mc',str(fixture.output),'--output-dir',str(mc_dir),'--face-cuts','none']
            completed = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertIn('FSD Cube Simulation Fit',completed.stdout)
            manifest = json.loads((mc_dir/'analysis.json').read_text())
            self.assertEqual(manifest['samples'][0]['sample_kind'],'mc')
            self.assertEqual(manifest['fits'][0]['title'],'FSD Cube Simulation Fit')
            self.assertEqual(manifest['fits'][0]['sample_id'],1)
            self.assertTrue((mc_dir/'mc_dqdx.png').exists())
            self.assertTrue((mc_dir/'mc_dqdx.pdf').exists())
            self.assertTrue((mc_dir/'mc_dqdx_histogram.csv').exists())
            self.assertFalse((mc_dir/'data_dqdx.png').exists())
            self.assertFalse((mc_dir/'data_mc_dqdx.png').exists())
            data_dir = fixture.root/'data_only'
            analyze(fixture.output,data_dir,face_cuts='none')
            with h5py.File(mc_dir/'segments.hdf5') as mc, h5py.File(data_dir/'segments.hdf5') as data:
                self.assertNotIn('samples/0',mc)
                self.assertEqual(mc['samples/1'].attrs['sample_kind'],'mc')
                values = mc['segments/data'][:]
                self.assertEqual(set(values['sample_id']),{1})
                np.testing.assert_array_equal(values['dqdx'],data['segments/data']['dqdx'])
            np.testing.assert_array_equal(np.loadtxt(mc_dir/'mc_dqdx_histogram.csv',delimiter=',',skiprows=1),
                                          np.loadtxt(data_dir/'data_dqdx_histogram.csv',delimiter=',',skiprows=1))
        finally:
            fixture.tearDown()

    def test_no_sample_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)/'unused'
            with self.assertRaisesRegex(ValueError,'at least one'):
                analyze(output_dir=output)
            command = [sys.executable,str(Path(__file__).resolve().parents[1]/'analyze_track_dqdx.py'),
                       '--output-dir',str(output)]
            completed = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,2)
            self.assertIn('at least one of --data or --mc',completed.stderr)
            self.assertFalse(output.exists())

    def test_explicit_geometry_and_source_identity(self):
        fixture = fixtures.OutputTests()
        fixture.setUp()
        try:
            fixture.fixture()
            selector=fixtures.import_selector()
            with redirect_stdout(io.StringIO()):
                selector.run(str(fixture.source),str(fixture.output),False)
            geometry={'groups':[{'name':'cube','io_groups':[1],
                                 'bounds':{'x':[-1,101],'y':[-1,250],'z':[-1,11]}}]}
            result=analyze(fixture.output,fixture.root/'fid',geometry=geometry,require_through_going=True)
            self.assertGreater(result['samples'][0]['audit']['segments'],0)
            with h5py.File(fixture.output,'r+') as f:
                event=f['events/data'][0]
                event['file_id']=5
                f['events/data'][0]=event
            with self.assertRaisesRegex(ValueError,'crosses'):
                analyze(fixture.output,fixture.root/'bad')
            saved=json.loads((fixture.root/'bad/analysis.json').read_text())
            self.assertEqual(saved['status'],'failed')
        finally:
            fixture.tearDown()

    def test_automatic_geometry_rejects_degenerate_bounds(self):
        fixture=fixtures.OutputTests()
        fixture.setUp()
        try:
            fixture.fixture()
            selector=fixtures.import_selector()
            with redirect_stdout(io.StringIO()):
                selector.run(str(fixture.source),str(fixture.output),False)
            with h5py.File(fixture.output,'r+') as f:
                bounds,reason=automatic_bounds(f,0,1.)
                self.assertIsNotNone(bounds)
                f['sources/0/metadata/geometry_info'].attrs['lar_detector_bounds']=[[48,-149,0],[48,149,48]]
                bounds,reason=automatic_bounds(f,0,1.)
                self.assertIsNone(bounds)
                self.assertIn('degenerate',reason)
            result=analyze(fixture.output,fixture.root/'fallback')
            self.assertGreater(result['samples'][0]['audit']['segments'],0)
        finally:
            fixture.tearDown()

    def test_invalid_geometry_and_missing_bounds_fail(self):
        hits=np.zeros(3,dtype=[('io_group','i8')])
        hits['io_group']=1
        geometry={'groups':[dict(name='bad',io_groups=[1],bounds=dict(x=[0,0],y=[0,1],z=[0,1]))]}
        with self.assertRaisesRegex(ValueError,'Invalid physical bounds'):
            groups_for(hits,0,geometry,'io-group')
        with self.assertRaisesRegex(ValueError,'requires --geometry-json'):
            analyze('unused','unused',require_through_going=True)


if __name__ == '__main__':
    unittest.main()
