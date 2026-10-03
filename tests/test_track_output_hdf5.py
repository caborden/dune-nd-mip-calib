"""Synthetic FLOW/output regression; h5flow is replaced only at its I/O boundary."""
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from track_output_hdf5 import TrackOutputWriter, linked_rows, take_rows


class Manager:
    def __init__(self, path, *args, **kwargs):
        self.f = h5py.File(path, 'r')

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.f[key]
        source, target, row = key
        rows = linked_rows(self.f, source, target, row)
        # Deliberately reverse the manager order; IDs are not dataset rows.
        return np.ma.array(take_rows(self.f[f'{target}/data'], rows[::-1]))[None, :]

    def close(self):
        self.f.close()


def import_selector():
    fake = types.ModuleType('h5flow')
    data = types.ModuleType('h5flow.data')
    data.H5FlowDataManager = Manager
    data.dereference = None
    fake.data = data
    plotly = types.ModuleType('plotly')
    graph = types.ModuleType('plotly.graph_objects')
    plotly.graph_objects = graph
    with patch.dict(sys.modules, {'h5flow': fake, 'h5flow.data': data,
                                  'plotly': plotly, 'plotly.graph_objects': graph}):
        return importlib.import_module('track_selection_fsd')


def add_links(f, source, target, lists):
    pairs, regions = [], []
    for row, targets in enumerate(lists):
        start = len(pairs)
        pairs.extend((row, int(t)) for t in targets)
        regions.append((start, len(pairs)))
    base = f'{source}/ref/{target}'
    f.create_dataset(f'{base}/ref', data=np.array(pairs, dtype='i8').reshape(-1, 2))
    f.create_dataset(f'{base}/ref_region', data=np.array(regions, dtype=[('start','i8'),('stop','i8')]))


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = import_selector()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.h5'
        self.output = self.root/'tracks.h5'

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, noisy=False, optional=True):
        dtype = [('id','u8'), ('x','f8'), ('y','f8'), ('z','f8'), ('Q','f8'),
                 ('Q_raw','f8'), ('t_drift','u8'), ('ts_pps','u8'),
                 ('io_group','u1'), ('io_channel','u1'), ('chip_id','u1'),
                 ('channel_id','u1'), ('is_disabled','?')]
        # Event 0 contains two separated straight tracks and rejected hits.
        n = 122 if noisy else 242
        hits = np.zeros(n, dtype=dtype)
        hits['id'] = 10000 + np.arange(n)*3
        hits['Q'], hits['Q_raw'] = 1, 2
        hits['t_drift'] = np.arange(n)
        hits['ts_pps'] = 2**60 + np.arange(n, dtype='u8')
        hits['io_group'], hits['io_channel'], hits['chip_id'] = 1, 2, 3
        hits['channel_id'] = np.arange(n) % 120
        t = np.linspace(0, 100, 120)
        for offset in ([0] if noisy else [0,120]):
            hits['x'][offset:offset+120] = t
            hits['y'][offset:offset+120] = t/10 + offset
            hits['z'][offset:offset+120] = t/10
        if noisy:
            hits['y'][:120] = 0.02*np.sin(np.arange(120))
        else:
            # Same address as hit 60, but too far from the axis.
            hits[-2] = hits[60]
            hits['id'][-2] = 10000 + (n-2)*3
            hits['y'][-2] += 2.5
            hits['Q'][-2] = 3
            hits['t_drift'][-2] += 10
        hits['x'][-1] = np.nan
        with h5py.File(self.source,'w') as f:
            f.create_dataset('charge/calib_prompt_hits/data', data=hits)
            f.create_dataset('charge/events/data', data=np.array([(789,n),(790,0)],dtype=[('id','u8'),('nhit','i8')]))
            add_links(f,'charge/events','charge/calib_prompt_hits',[np.arange(n),[]])
            f.create_group('geometry_info').attrs['lar_detector_bounds'] = [[-1,-1,-1],[101,250,11]]
            f.create_group('run_info').attrs['crs_ticks'] = 0.1
            f.create_dataset('charge/ext_trigs/data',data=np.array([(6,99)],dtype=[('iogroup','u1'),('ts','u8')]))
            add_links(f,'charge/events','charge/ext_trigs',[[0],[]])
            if optional:
                f.create_dataset('combined/t0/data', data=np.array([(1,12)],dtype=[('type','u1'),('ts','u8')]))
                add_links(f,'charge/events','combined/t0',[[0],[]])
                packets = np.zeros(n+1, dtype=[('id','u8'),('dataword','u1'),('valid_parity','u1'),('reset_sample_flag','u1')])
                packets['dataword'], packets['valid_parity'] = 8, 1
                packets['reset_sample_flag'][-1] = 1
                f.create_dataset('charge/packets/data',data=packets)
                lists = [[i] for i in range(n)]
                lists[60].append(n)
                add_links(f,'charge/calib_prompt_hits','charge/packets',lists)
        return hits

    def legacy_selected_ids(self, hits):
        # Independent expression of the original selection, including the
        # standardized final quality and its strict thresholds.
        hits = hits[~np.isnan(np.column_stack([hits[d] for d in 'xyz'])).any(axis=1)]
        xyz = np.column_stack([hits[d] for d in 'xyz'])
        labels = DBSCAN(eps=9.3,min_samples=1).fit_predict(xyz)
        result = set()
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            if len(idx) < 90:
                continue
            points = xyz[idx]
            fit = PCA(3).fit(points)
            var = fit.explained_variance_
            if np.sqrt((var[1]+var[2])/var[0]) > 0.07:
                continue
            keep = np.linalg.norm(np.cross(points-points.mean(axis=0),fit.components_[0]),axis=1) < 1.116
            idx, points = idx[keep], points[keep]
            if len(points)<90:
                continue
            a = PCA(1).fit(StandardScaler().fit_transform(points)).explained_variance_ratio_[0]
            low, high = points[np.argmin(points[:,2])], points[np.argmax(points[:,2])]
            if a > .973 and np.linalg.norm(high-low)>55:
                result.update(hits['id'][idx].tolist())
        return result

    def test_single_copy_two_views_and_original_selection(self):
        original = self.fixture()
        self.selector.run(str(self.source),str(self.output),False)
        with h5py.File(self.output) as f:
            self.assertEqual(f.attrs['status'],'complete')
            hits, events, tracks = f['events/hits'][:], f['events/data'][:], f['tracks/data'][:]
            self.assertEqual(len(hits),len(original))
            self.assertEqual(len(events),1)
            self.assertEqual(len(tracks),2)
            selected = hits[hits['selected_track_id']>=0]
            self.assertEqual(set(selected['id']), self.legacy_selected_ids(original))
            self.assertEqual(len(selected),240)
            np.testing.assert_array_equal(hits['source_hit_row'],np.arange(len(original)))
            np.testing.assert_array_equal(hits['ts_pps'],original['ts_pps'])
            self.assertTrue(np.isnan(hits['projected_xyz_cm'][-2:]).all())
            self.assertEqual((events['hit_start'][0],events['hit_stop'][0]),(0,len(original)))
            # Both views share one table: selected charge 1; full pixel charge 4.
            pixel = hits[(hits['channel_id']==60)&(hits['y']<100)]
            self.assertEqual(pixel['Q'].sum(),4)
            self.assertEqual(pixel['Q'][pixel['selected_track_id']>=0].sum(),1)
            physical = np.array([10,1,1])/np.sqrt(102)
            for track in tracks:
                self.assertAlmostEqual(abs(track['pca_dir_physical']@physical),1)
                self.assertAlmostEqual(abs(track['pca_dir_standardized']@np.ones(3)/np.sqrt(3)),1)
                self.assertEqual(track['n_selected_hits'],120)
            self.assertEqual(hits['packet_link_count'][60],2)
            self.assertEqual(len(f['events/packets']),len(original)+1)
            self.assertEqual(len(f['events/hit_packet_refs']),len(original)+1)
            refs = f['events/hit_packet_refs'][:]
            self.assertTrue((refs['hit_row']<len(hits)).all())
            self.assertTrue((refs['packet_row']<len(f['events/packets'])).all())
            self.assertEqual(len(f['events/t0']),1)

    def test_standardized_cut_is_still_used_empty_output(self):
        original = self.fixture(noisy=True)
        self.assertEqual(self.legacy_selected_ids(original),set())
        self.selector.run(str(self.source),str(self.output),False)
        with h5py.File(self.output) as f:
            self.assertEqual(len(f['tracks/data']),0)
            self.assertEqual(len(f['events/hits']),0)
            self.assertEqual(f.attrs['status'],'complete')

    def test_missing_optional_fields_and_links(self):
        self.fixture(optional=False)
        self.selector.run(str(self.source),str(self.output),False)
        with h5py.File(self.output) as f:
            self.assertFalse(f['events/packets'].attrs['available'])
            self.assertFalse(f['events/t0'].attrs['available'])
            self.assertTrue((f['events/hits']['packet_link_count']==-1).all())

    def test_multiple_event_offsets_and_track_ids(self):
        original = self.fixture()
        with h5py.File(self.source,'r+') as f:
            del f['charge/events/ref/charge/calib_prompt_hits']
            add_links(f,'charge/events','charge/calib_prompt_hits',
                      [np.arange(len(original)),np.arange(len(original))])
            f['charge/events/data'][1] = (790,len(original))
        self.selector.run(str(self.source),str(self.output),False)
        with h5py.File(self.output) as f:
            n = len(original)
            events, hits, tracks = f['events/data'][:], f['events/hits'][:], f['tracks/data'][:]
            np.testing.assert_array_equal(events['hit_start'],[0,n])
            np.testing.assert_array_equal(events['hit_stop'],[n,2*n])
            np.testing.assert_array_equal(tracks['track_id'],np.arange(4))
            self.assertEqual(set(hits['selected_track_id'][n:]),{-1,2,3})
            refs = f['events/hit_packet_refs'][:]
            self.assertEqual(len(refs),2*(n+1))
            self.assertTrue((refs['packet_row'][n+1:] >= n+1).all())

    def test_2x2_trigger_requirement_is_preserved(self):
        self.fixture()
        with h5py.File(self.source,'r+') as f:
            f['charge/ext_trigs/data'][0] = (5,99)
        self.selector.run(str(self.source),str(self.output),True)
        with h5py.File(self.output) as f:
            self.assertEqual(len(f['tracks/data']),0)
        second_output = self.root/'fsd.h5'
        self.selector.run(str(self.source),str(second_output),False)
        with h5py.File(second_output) as f:
            self.assertEqual(len(f['tracks/data']),2)

    def test_no_overwrite(self):
        self.fixture()
        self.output.write_text('preserve me')
        with self.assertRaises(FileExistsError):
            TrackOutputWriter(self.output,self.source,{})
        self.assertEqual(self.output.read_text(),'preserve me')

    def test_invalid_reference_marks_output_incomplete(self):
        self.fixture()
        with h5py.File(self.source,'r+') as f:
            f['charge/calib_prompt_hits/ref/charge/packets/ref'][0] = [0,99999]
        with self.assertRaisesRegex(ValueError,'Out-of-bounds'):
            self.selector.run(str(self.source),str(self.output),False)
        with h5py.File(self.output) as f:
            self.assertEqual(f.attrs['status'],'incomplete')


if __name__ == '__main__':
    unittest.main()
