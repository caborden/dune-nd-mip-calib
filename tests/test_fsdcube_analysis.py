"""Synthetic integration tests; no detector data or network required."""
import tempfile
from pathlib import Path
import unittest
import h5py
import numpy as np
import pandas as pd
from fsdcube_analysis.flow_io import read_event, ReferenceError
from fsdcube_analysis.charge_splitting import add_pixel_keys, summarize_pixels
from fsdcube_analysis.tracks import TrackConfig, associate_tracks
from fsdcube_analysis.scan import scan_files, load_results


def create_fixture(path):
    """Two populated events and an empty event, shuffled links, IDs != rows.

    First event has a straight 24-pixel track, a delayed same-pixel hit displaced
    from the seed axis, and a long-delay same-pixel hit that must not be recovered.
    Second event is isolated from the first despite reusing the electronics key.
    """
    n = 28
    dtype = [(c, "u8") for c in ["id", "ts_pps", "io_group", "io_channel", "chip_id", "channel_id"]]
    dtype += [(c, "f8") for c in ["x", "y", "z", "Q", "Q_raw", "t_drift"]] + [("is_disabled", "?")]
    hits = np.zeros(n, dtype=dtype)
    hits["id"] = np.arange(n) + 1000
    hits["io_group"] = 1; hits["io_channel"] = 1; hits["chip_id"] = 11
    hits["channel_id"][:24] = np.arange(24)
    hits["y"][:24] = np.arange(24) * 0.6
    hits["t_drift"][:24] = np.arange(24) * 5
    hits["Q"] = 10; hits["Q_raw"] = 10
    for j, t, x in [(24, 75, 0.7), (25, 500, 0.8)]:
        hits[j] = hits[8]; hits[j]["id"] = j + 1000
        hits[j]["t_drift"] = t; hits[j]["x"] = x
    hits[24]["Q"] = hits[24]["Q_raw"] = 4
    hits[25]["Q"] = hits[25]["Q_raw"] = 2
    hits[26] = hits[8]; hits[27] = hits[8]
    hits[26]["id"] = 1026; hits[27]["id"] = 1027
    hits[27]["t_drift"] += 20
    hits["ts_pps"] = hits["t_drift"].astype("u8") + 9000000
    packets = np.zeros(n + 3, dtype=[(c, "u8") for c in [
        "io_group", "io_channel", "chip_id", "channel_id", "packet_type", "timestamp",
        "dataword", "valid_parity", "trigger_type", "reset_sample_flag", "cds_flag"]])
    for c in ["io_group", "io_channel", "chip_id", "channel_id"]:
        packets[c][3:] = hits[c]
    packets["packet_type"][3:] = 1; packets["valid_parity"][3:] = 1
    packets["timestamp"][3:] = hits["ts_pps"]; packets["dataword"][3:] = 300
    with h5py.File(path, "w") as f:
        f.create_dataset("charge/calib_prompt_hits/data", data=hits)
        f.create_dataset("charge/packets/data", data=packets)
        f.create_dataset("charge/events/data", data=np.array([(100,26), (200,2), (300,0)],
                         dtype=[("id","u8"),("nhit","u4")]))
        f.create_dataset("combined/t0/data", data=np.array([(9,9000000.,0),(8,9000000.,0),(7,9000000.,0)],
                         dtype=[("id","u4"),("ts","f8"),("type","u1")]))
        def refs(source, target, links, regions):
            base = f"{source}/ref/{target}"
            f.create_dataset(base + "/ref", data=np.array(links,dtype="u4").reshape(-1,2))
            f.create_dataset(base + "/ref_region", data=np.array(regions,dtype=[("start","i4"),("stop","i4")]))
        refs("charge/events", "charge/calib_prompt_hits",
             [(0,i) for i in reversed(range(26))]+[(1,26),(1,27)], [(0,26),(26,28),(28,28)])
        refs("charge/calib_prompt_hits", "charge/packets", [(i,i+3) for i in range(n)], [(i,i+1) for i in range(n)])
        refs("charge/events", "combined/t0", [(i,i) for i in range(3)], [(i,i+1) for i in range(3)])
        f.require_group("run_info").attrs.update(crs_ticks=0.1,data_packet_type=1,rollover_ticks=10000000)
        f.require_group("geometry_info").attrs.update(network_agnostic=True,n_io_channels_per_tile=4)
        f["charge/calib_prompt_hits"].attrs["adc_droop_calibration"] = False


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "input.h5"
        create_fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def read(self, row=0):
        with h5py.File(self.path) as f:
            hits, audit = read_event(f,row,0)
            return add_pixel_keys(hits,f["geometry_info"].attrs,"network"), audit

    def test_references_and_summary(self):
        hits,audit=self.read()
        self.assertTrue(audit["nhit_matches"])
        self.assertEqual(hits.iloc[0]["id"],1000)
        self.assertEqual(hits.iloc[0]["packet_row"],3)
        _,pixels,gaps=summarize_pixels(hits)
        repeated=pixels.loc[pixels.multiplicity==3].iloc[0]
        self.assertEqual(repeated.Q_total,16)
        self.assertEqual(repeated.Q2,4)
        np.testing.assert_allclose(gaps.delta_t_us,[3.5,42.5])

    def test_network_alias_and_ties(self):
        hits,_=self.read(1)
        hits.loc[1,"io_channel"]=4
        attrs={"network_agnostic":True,"n_io_channels_per_tile":4}
        address=add_pixel_keys(hits,attrs,"address")
        self.assertEqual(len(summarize_pixels(address)[1]),2)
        hits.loc[1,"t_drift"]=hits.loc[0,"t_drift"]
        hits.loc[1,"t_drift_us"]=hits.loc[0,"t_drift_us"]
        network=add_pixel_keys(hits,attrs,"network")
        pixel=summarize_pixels(network)[1].iloc[0]
        self.assertTrue(pixel.sum_quality_ok)
        self.assertFalse(pixel.order_quality_ok)

    def test_bad_charge_retained(self):
        hits,_=self.read(1)
        hits.loc[0,"Q"]=-2
        self.assertEqual(summarize_pixels(hits)[1].iloc[0].Q_total,8)
        hits.loc[0,"Q"]=np.nan; hits.loc[0,"hit_quality_ok"]=False
        pixel=summarize_pixels(hits)[1].iloc[0]
        self.assertTrue(np.isnan(pixel.Q_total))
        self.assertFalse(pixel.sum_quality_ok)

    def test_recovery_limits_and_length(self):
        hits,_=self.read()
        associated,tracks=associate_tracks(hits)
        self.assertEqual(tracks.accepted.sum(),1)
        self.assertEqual(associated.loc[24,"association_role"],"recovered")
        self.assertEqual(associated.loc[25,"track_id"],-1)
        _,pixels,_=summarize_pixels(associated.loc[associated.track_id>=0],track=True)
        pair=pixels.loc[pixels.multiplicity==2].iloc[0]
        self.assertEqual(pair.Q_total,14)
        self.assertFalse(pair.association_complete)
        _,strict=associate_tracks(hits,TrackConfig(min_length_cm=55))
        self.assertEqual(strict.accepted.sum(),0)

    def test_competing_tracks_are_ambiguous(self):
        hits,_=self.read()
        hits=hits.iloc[:24].copy()
        hits["y"]=np.tile(np.arange(12)*0.3,2)
        hits["z"]=np.repeat([0.0,0.5],12)
        hits["t_drift_us"]=1.0
        out,tracks=associate_tracks(hits,TrackConfig(eps_cm=0.4,min_pixels=4,
            min_length_cm=1,axis_distance_cm=0.1,recovery_distance_cm=0.6))
        self.assertEqual(tracks.accepted.sum(),2)
        self.assertTrue((out.track_id==-2).all())

    def test_packet_quality_and_geometry_exclusions(self):
        with h5py.File(self.path,"r+") as f:
            ds=f["charge/packets/data"]
            p=ds[3];p["valid_parity"]=0;ds[3]=p
            p=ds[4];p["chip_id"]=99;ds[4]=p
            p=ds[5];p["packet_type"]=2;ds[5]=p
        hits,_=self.read()
        self.assertFalse(hits.loc[0,"hit_quality_ok"])
        self.assertFalse(hits.loc[1,"address_match"])
        self.assertFalse(hits.loc[2,"data_packet_ok"])
        hits.loc[24,"y"]+=0.1
        pixel=summarize_pixels(hits)[1].query("multiplicity == 3").iloc[0]
        self.assertFalse(pixel.geometry_ok)
        self.assertFalse(pixel.sum_quality_ok)

    def test_event_count_mismatch_does_not_select(self):
        with h5py.File(self.path,"r+") as f:
            d=f["charge/events/data"]; e=d[0];e["nhit"]=999;d[0]=e
        tables=scan_files([self.path],self.root/"mismatch",max_events=1)
        self.assertFalse(tables["events"].iloc[0].nhit_matches)
        self.assertFalse(tables["pixels"].sum_quality_ok.any())
        self.assertEqual(len(tables["track_hits"]),0)

    def test_two_files_empty_event_and_plotting(self):
        from fsdcube_analysis.plots import make_plots
        import matplotlib.pyplot as plt
        second=self.root/"second.h5"; create_fixture(second)
        tables=scan_files([self.path,second],self.root/"output",pixel_mode="network")
        self.assertEqual(len(tables["events"]),6)
        self.assertEqual(len(tables["pixels"]),50)
        loaded,manifest=load_results(self.root/"output")
        self.assertEqual(manifest["status"],"complete")
        for sample in ["event","track"]:
            figs,_,_=make_plots(loaded,self.root/"plots",sample=sample)
            self.assertEqual(len(figs),6)
        plt.close("all")
        with self.assertRaises(FileExistsError):
            scan_files([self.path],self.root/"output")

    def test_duplicate_reference_is_audited(self):
        with h5py.File(self.path,"r+") as f:
            refs=f["charge/events/ref/charge/calib_prompt_hits/ref"]
            refs[1]=refs[0]
        with self.assertRaises(ReferenceError):self.read()
        result=scan_files([self.path],self.root/"bad",max_events=1)
        self.assertEqual(result["events"].iloc[0].status,"reference_error")
        self.assertEqual(len(result["hits"]),0)

    def test_empty_range_plots_and_no_tracks(self):
        from fsdcube_analysis.plots import make_plots
        import matplotlib.pyplot as plt
        for start in [2,30]:
            tables=scan_files([self.path],self.root/f"empty{start}",start_event=start,tracks=False)
            make_plots(tables,sample="event")
            plt.close("all")
            make_plots(tables,sample="track")
            plt.close("all")
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
