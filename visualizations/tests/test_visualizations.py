import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import yaml

from visualizations.geometry import load_geometry, tile_pixels, repositioning_boxes, geometry_figure, pixel_figure
from visualizations.signals import (illustrative_electronics,load_bundle,save_bundle,validate_bundle,
    demo_bundle,signal_figure,attach_event_segments,read_channel_packets,packet_figure)
from visualizations.capture_larnd import make_capture_wrapper,pixel_positions


def fixture(root):
    base=Path(root)/'larndsim'
    for d in ['config','pixel_layouts','detector_properties']: (base/d).mkdir(parents=True)
    layout=dict(pixel_pitch=4.,drift_length=100.,
        chip_channel_to_position={11000:[0,0],11001:[1,0],11002:[0,1],11003:[1,1]},
        tile_positions={1:[100,200,300]},tile_orientations={1:[-1,1,-1]},
        tile_indeces={1:[1,1]},tile_chip_to_io={1:{11:1002}})
    prop=dict(tpc_offsets=[[1,2,3]],module_to_io_groups={1:[7]},module_to_tpcs={1:[0]},
              tile_map=[[[1]]],e_field=.25)
    config={'fsd_cube':dict(PIXEL_LAYOUT='test.yaml',DET_PROPERTIES='test.yaml')}
    for path,data in [('config/config.yaml',config),('pixel_layouts/test.yaml',layout),('detector_properties/test.yaml',prop)]:
        (base/path).write_text(yaml.safe_dump(data))
    return load_geometry(root)


class GeometryTests(unittest.TestCase):
    def test_units_orientation_and_channels(self):
        with TemporaryDirectory() as tmp:
            g=fixture(tmp);p=tile_pixels(g,1)
            np.testing.assert_allclose(g.tile_centers[1],[11,22,33])
            np.testing.assert_allclose(g.tpc_bounds[1][0],[1,11])
            np.testing.assert_allclose(p['xyz_cm'][0],[11,21.8,33.2])
            np.testing.assert_array_equal(p['pixel_id'],[1,0,3,2])
            np.testing.assert_array_equal(p['io_group'],[7]*4)
            np.testing.assert_array_equal(p['io_channel'],[2]*4)
            geometry_figure(g);pixel_figure(g)

    def test_capture_position_roundtrip(self):
        with TemporaryDirectory() as tmp:
            g=fixture(tmp);p=tile_pixels(g,1);d=runtime(g)
            xyz,addr=pixel_positions(d,p['pixel_id'])
            np.testing.assert_allclose(xyz,p['xyz_cm'])
            np.testing.assert_array_equal(addr[:,2],p['chip_id'])
            np.testing.assert_array_equal(addr[:,3],p['channel_id'])

    def test_all_harvest_windows_reach_same_destination(self):
        _,target,_=repositioning_boxes(0)
        for i in range(40):
            source,destination,shift=repositioning_boxes(i)
            np.testing.assert_allclose(source+shift[:,None],target,atol=1e-12)
            np.testing.assert_allclose(destination,target,atol=1e-12)
        with self.assertRaises(ValueError):repositioning_boxes(40)

    def test_local_production_layouts(self):
        repo=Path(__file__).resolve().parents[3]/'larnd-sim'
        if not repo.exists(): self.skipTest('Optional sibling larnd-sim checkout absent')
        fsd=load_geometry(repo,'fsd');cube=load_geometry(repo,'fsd_cube')
        self.assertEqual(len(fsd.tile_centers),40);self.assertEqual(len(cube.tile_centers),1)
        self.assertEqual(fsd.shape,(128,80));self.assertAlmostEqual(cube.pitch_cm,.372)
        self.assertEqual(fsd.layout['chip_channel_to_position'],cube.layout['chip_channel_to_position'])
        self.assertAlmostEqual(cube.summary()['drift_cm'],47.75)
        for g in [fsd,cube]:
            allids=[]
            for tile in g.tile_centers:
                p=tile_pixels(g,tile);allids.extend(p['pixel_id'])
                xyz,addr=pixel_positions(runtime(g),p['pixel_id'])
                np.testing.assert_allclose(xyz,p['xyz_cm'])
                np.testing.assert_array_equal(addr[:,3],p['channel_id'])
            self.assertEqual(len(allids),len(set(allids)))
        b=demo_bundle(cube);validate_bundle(b)
        for i in range(3): signal_figure(b,i)
        self.assertAlmostEqual(np.sum(b['current_e_per_us'][2])*.1,0.,places=8)


def runtime(g):
    shape=g.shape
    props=dict(TIME_SAMPLING=.1,CLOCK_CYCLE=.1,PERIODIC_RESET_CYCLES=512,
        ADC_HOLD_DELAY=20,ADC_BUSY_DELAY=9,RESET_CYCLES=1,GAIN=.0034,V_PEDESTAL=980,
        V_REF=1734,V_CM=0,ADC_COUNTS=1023,ADC_SCALE_FACTOR=2,ADC_OFFSET=0,
        BUFFER_RISETIME=.1,DISCRIMINATION_THRESHOLD=5000,LARPIX_VERSION=3,PACKET_TYPE=1)
    return SimpleNamespace(**props,N_PIXELS=(shape[0]*len(g.properties['tile_map'][0]),
        shape[1]*len(g.properties['tile_map'][0][0])),N_PIXELS_PER_TILE=shape,
        MODULE_TO_IO_GROUPS=g.properties['module_to_io_groups'],TILE_MAP=g.properties['tile_map'],
        TILE_POSITIONS=g.layout['tile_positions'],TILE_ORIENTATIONS=g.layout['tile_orientations'],
        TILE_CHIP_TO_IO=g.layout['tile_chip_to_io'],TPC_OFFSETS=np.asarray(g.properties['tpc_offsets'])[:,[2,1,0]],
        PIXEL_PITCH=g.pitch_cm,PIXEL_CONNECTION_DICT={tuple(v):(k//1000,k%1000) for k,v in g.layout['chip_channel_to_position'].items()})


class SignalTests(unittest.TestCase):
    def test_integrator_charge_and_threshold(self):
        c,at,av,r=illustrative_electronics(np.ones(10)*100,1,threshold=1000,reset_period_us=0,
            hold_us=0,busy_us=0,gain=1,pedestal=0,v_ref=2000,adc_counts=2000,adc_scale=1)
        np.testing.assert_allclose(c,np.arange(1,11)*100)
        np.testing.assert_allclose(at,[9]);np.testing.assert_allclose(av,[1000])
        c,at,av,r=illustrative_electronics(np.ones(10)*100,1,threshold=1000,reset_period_us=5,hold_us=0)
        self.assertEqual(len(at),0)
        np.testing.assert_allclose(c,[100,200,300,400,500,100,200,300,400,500])

    def test_capture_does_not_change_driver_or_rng(self):
        with TemporaryDirectory() as tmp:
            g=fixture(Path(tmp)/'repo');d=runtime(g);out=Path(tmp)/'out'
            calls=[];rng=np.random.default_rng(4)
            state=json.dumps(rng.bit_generator.state)
            currents=np.arange(12,dtype=float).reshape(4,3)
            original_currents=currents.copy()
            def driver(results_acc,unique_eventIDs,unique_pix,pixels_signals,max_signal_time):
                calls.append(1)
                results_acc['adc_tot'].append(np.array([[0,0],[9,0],[4,8],[3,0]]))
                results_acc['adc_tot_ticks'].append(np.array([[0,0],[.2,0],[.1,.2],[.3,0]]))
                return 'unchanged-return'
            acc={'adc_tot':[],'adc_tot_ticks':[]}
            w=make_capture_wrapper(driver,d,1,out,dict(detector='FSD Cube'),requested=[2],max_batches=1)
            self.assertEqual(w(acc,np.array([42]),np.arange(4),currents,.3),'unchanged-return')
            self.assertEqual(calls,[1]);np.testing.assert_array_equal(currents,original_currents)
            self.assertEqual(json.dumps(rng.bit_generator.state),state)
            b=load_bundle(next(out.glob('*.npz')))
            np.testing.assert_array_equal(b['current_e_per_us'],currents[[2]])
            np.testing.assert_array_equal(b['adc_value'],[4,8])
            np.testing.assert_array_equal(b['adc_channel_index'],[0,0])
            self.assertNotIn('csa_e',b);signal_figure(b)
            nonuniform=dict(b,time_us=np.array([0.,.1,.3]))
            with self.assertRaisesRegex(ValueError,'uniform'):validate_bundle(nonuniform)
            with self.assertRaises(FileExistsError):save_bundle(next(out.glob('*.npz')),b)
            w(acc,np.array([42]),np.arange(4),currents,.3)
            self.assertEqual(len(calls),2);self.assertEqual(w.capture_state['saved'],1)
            multi=make_capture_wrapper(driver,d,1,out,dict(detector='FSD Cube'))
            with self.assertRaisesRegex(RuntimeError,'one event'):
                multi(acc,np.array([42,43]),np.arange(4),currents,.3)
            limited=make_capture_wrapper(driver,d,1,out,dict(detector='FSD Cube'),max_samples=1)
            with self.assertRaisesRegex(RuntimeError,'samples'):
                limited(acc,np.array([42]),np.arange(4),currents,.3)
            self.assertEqual(len(calls),2)

    def test_packet_integer_timing_and_type(self):
        dtype=[(k,'u8') for k in ['packet_type','io_group','io_channel','chip_id','channel_id','timestamp','dataword']]
        rows=np.zeros(4,dtype=dtype)
        for k in ['io_group','io_channel','chip_id','channel_id']:rows[k]=1
        rows['packet_type']=[1,0,1,1];rows['timestamp']=2**60+np.arange(4,dtype='u8');rows['dataword']=7
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'packets.h5'
            with h5py.File(p,'w') as f:f['packets']=rows
            a=read_channel_packets(p,(1,1,1,1),data_packet_type=1,chunk_rows=1,limit=2)
            self.assertEqual(len(a),2);fig=packet_figure(a)
            np.testing.assert_allclose(fig.data[0].x,[0,.2])
            with self.assertRaisesRegex(ValueError,'decrease'):packet_figure(a[::-1])

    def test_event_context_is_bounded_and_preserves_xyz(self):
        with TemporaryDirectory() as tmp:
            g=fixture(Path(tmp)/'repo');d=runtime(g)
            def driver(results_acc,unique_eventIDs,unique_pix,pixels_signals,max_signal_time):
                results_acc['adc_tot'].append(np.zeros((1,1)));results_acc['adc_tot_ticks'].append(np.zeros((1,1)))
            w=make_capture_wrapper(driver,d,1,Path(tmp)/'trace',dict(detector='FSD Cube'))
            w({'adc_tot':[],'adc_tot_ticks':[]},np.array([42]),np.array([0]),np.zeros((1,4)),.4)
            b=load_bundle(next((Path(tmp)/'trace').glob('*.npz')))
            fields=['x_start','y_start','z_start','x_end','y_end','z_end']
            a=np.zeros(5,dtype=[('event_id','i8')]+[(k,'f4') for k in fields]);a['event_id']=[0,42,42,42,7]
            a['x_start']=1;a['z_end']=9
            p=Path(tmp)/'input.h5'
            with h5py.File(p,'w') as f:f['segments']=a
            b=attach_event_segments(b,p,max_segments=2,chunk_rows=2)
            self.assertTrue(validate_bundle(b)['context_truncated'])
            self.assertEqual(b['segments_xyz_cm'].shape,(2,2,3))
            np.testing.assert_allclose(b['segments_xyz_cm'][0],[[1,0,0],[0,0,9]])


if __name__=='__main__':unittest.main()
