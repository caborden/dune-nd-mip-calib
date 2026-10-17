"""Synthetic FLOW tests: pooling, references, exact timing, and report generation."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from .flow import ADDRESS, event_hits, inspect_file, resolve_samples
from .metrics import Histogram, Sample, positive_differences, timing
from .report import channel_ratios, draw_ratio, plt
from .run import DEFAULT_SETTINGS, DEFAULT_HISTOGRAMS, run, load_config


def settings(**changes):
    return dict(DEFAULT_SETTINGS, histograms=DEFAULT_HISTOGRAMS, **changes)


def fixture(path, repeats=1, tick=.1):
    hit_dtype = [('id', 'u8'), ('Q', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                 ('t_drift', 'f8'), ('ts_pps', 'u8'), ('is_disabled', '?')]
    hit_dtype += [(n, 'i4') for n in ADDRESS]
    hits = np.zeros(6*repeats, dtype=hit_dtype)
    for i in range(repeats):
        block = hits[i*6:(i+1)*6]
        block['id'] = np.arange(100, 106)  # Repeated IDs are deliberately not row indices.
        block['Q'] = [2, 4, 6, 8, 10, 12]
        block['ts_pps'] = np.uint64(2**60)+np.array([10, 12, 14, 15, 20, 21], dtype='u8')
        block['t_drift'] = [1, 2, 3, 4, 5, 6]
        block['channel_id'] = [0, 0, 1, 0, 0, 2]
        block['io_group'], block['io_channel'], block['chip_id'] = 1, 1, 3
        block['y'], block['z'] = block['channel_id']*.372, block['channel_id']*.372
        block['x'] = np.arange(6)
        block['is_disabled'][-1] = True
    events = np.zeros(3*repeats, dtype=[('id', 'u8'), ('unix_ts', 'u8')])
    events['id'] = 900  # Do not index the reference table with this ID.
    events['unix_ts'] = 1000+np.arange(len(events))
    refs, regions = [], []
    for i in range(repeats):
        base = len(refs)
        refs.extend([[3*i, 6*i+2], [3*i+1, 6*i+4], [3*i, 6*i],
                     [3*i, 6*i+1], [3*i+1, 6*i+3]])
        # Non-contiguous associations share enclosing regions, as h5flow permits.
        regions.extend([(base, base+5), (base+1, base+5), (base+5, base+5)])
    with h5py.File(path, 'w') as f:
        f.create_dataset('charge/calib_prompt_hits/data', data=hits)
        f.create_dataset('charge/events/data', data=events)
        f.create_dataset('charge/events/ref/charge/calib_prompt_hits/ref', data=np.array(refs, dtype='i8').reshape(-1, 2))
        f.create_dataset('charge/events/ref/charge/calib_prompt_hits/ref_region',
                         data=np.array(regions, dtype=[('start', 'u8'), ('stop', 'u8')]))
        f.require_group('run_info').attrs['crs_ticks'] = tick
        packets = np.array([(0, 3, 1), (0, 255, 0), (4, 255, 1)],
                           dtype=[('packet_type', 'i4'), ('dataword', 'i4'), ('valid_parity', 'i4')])
        f.create_dataset('charge/packets/data', data=packets)
    return path


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def sample(self, files, opts=None):
        opts = opts or settings()
        sample = Sample(dict(name='sample', label='Sample'), opts)
        for path in files:
            sample.process(inspect_file(str(path), opts))
        return sample

    def test_histogram_accounting(self):
        hist = Histogram(np.array([0., 1., 2.]))
        hist.add([-1, 0, 1, 2, 3, np.nan])
        self.assertEqual(hist.summary(), dict(total=6, nonfinite=1, underflow=1, overflow=1, in_range=3))
        np.testing.assert_array_equal(hist.counts, [1, 2])

    def test_ratio_extremes_marked_without_modifying_values(self):
        values = np.array([np.nan, 0., 1., 50., 17000.])
        original = values.copy()
        fig, ax = plt.subplots()
        try:
            audit = draw_ratio(ax, np.arange(5), values, np.array([0, 30, 80, 1, 2000]), settings(), 'C1')
            np.testing.assert_equal(values, original)
            self.assertEqual(ax.get_ylim(), (0., 2.))
            self.assertEqual(audit, dict(undefined=1, low_reference_count=1,
                                         below_display=0, above_display=2))
            drawn = np.concatenate([c.get_offsets() for c in ax.collections])
            self.assertEqual(len(drawn), 4)
            np.testing.assert_array_equal(sorted(drawn[:, 1]), [0, 1, 2, 2])
        finally:
            plt.close(fig)

    def test_full_sample_glob_spans_dates_but_not_siblings(self):
        root = self.root/'lt_prc16'
        selected = []
        for day in ('2026_02_25', '2026_02_27', '2026_02_28'):
            directory = root/('Reflow_FSDCube_v3.flow.binary-cosmics-'+day)
            directory.mkdir(parents=True)
            path = directory/('Reflow_FSDCube_v3.flow.binary-cosmics-'+day+'_run.FLOW.hdf5')
            path.touch()
            selected.append(str(path.resolve()))
        sibling = self.root/'lt_prc8'
        sibling.mkdir()
        (sibling/'unrelated.FLOW.hdf5').touch()
        pattern = str(root/'Reflow_FSDCube_v3.flow.binary-cosmics-2026_02_*'/'*.FLOW.hdf5')
        result = resolve_samples([dict(name='feb', files=[pattern])], self.root)
        self.assertEqual(result[0]['files'], selected)

    def test_invalid_ratio_settings_rejected(self):
        config = self.root/'bad-settings.json'
        for key, value in [('ratio_ylim', [0, 0]), ('ratio_ylim', [2, 3]),
                           ('ratio_min_reference_count', 0)]:
            config.write_text(json.dumps(dict(samples=[], settings={key: value})))
            with self.assertRaisesRegex(ValueError, key):
                load_config(config)

    def test_adc_schema_present_but_no_data_packets(self):
        path = fixture(self.root/'no-data-packets.h5')
        with h5py.File(path, 'r+') as f:
            packets = f['charge/packets/data'][:]
            packets['packet_type'] = 4
            f['charge/packets/data'][:] = packets
        result = self.sample([path])
        self.assertEqual(result.files[0]['packet_type_counts'], {'4': 3})
        self.assertEqual(len(result.coverage['adc']), 1)
        self.assertEqual(result.histograms['adc'].total, 0)
        self.assertTrue(any('no packet_type=0' in w for w in result.warnings))

    def test_pooling_is_invariant_to_file_partition(self):
        first = fixture(self.root/'first.h5')
        second = fixture(self.root/'second.h5', repeats=2)
        combined = fixture(self.root/'combined.h5', repeats=3)
        pooled, whole = self.sample([first, second]), self.sample([combined])
        for key in pooled.histograms:
            if key == 'adc':  # Fixture writes three packets per file, independent of repeats.
                continue
            np.testing.assert_array_equal(pooled.histograms[key].counts, whole.histograms[key].counts)
            self.assertEqual(pooled.histograms[key].summary(), whole.histograms[key].summary())
        self.assertEqual(list(pooled.channel_rows()), list(whole.channel_rows()))
        self.assertEqual(pooled.audit['events_analyzed'], 9)
        self.assertEqual(pooled.audit['hit_rows'], 18)
        self.assertEqual(pooled.histograms['delta_us'].total, 6)
        self.assertEqual(pooled.audit['empty_events'], 3)

    def test_references_use_rows_and_filter_enclosing_regions(self):
        path = fixture(self.root/'flow.h5')
        audit = Counter()
        with h5py.File(path) as f:
            events = list(event_hits(f, settings(), audit))
        self.assertEqual([list(h['Q']) for _, h in events], [[2, 4, 6], [8, 10], []])
        self.assertEqual(audit['events_analyzed'], 3)

    def test_malformed_references_and_duplicate_links(self):
        path = fixture(self.root/'bad.h5')
        base = 'charge/events/ref/charge/calib_prompt_hits'
        with h5py.File(path, 'r+') as f:
            f[base+'/ref'][3] = [0, 0]  # Duplicate link, not an extra recorded hit.
            f[base+'/ref'][1] = [1, 999]  # Invalid target: omit the whole event.
        result = self.sample([path])
        self.assertEqual(result.audit['duplicate_event_hit_links'], 1)
        self.assertEqual(result.audit['events_invalid_hit_references'], 1)
        self.assertEqual(result.audit['reference_rows_invalid_target'], 1)
        self.assertEqual(result.audit['events_analyzed'], 2)
        self.assertEqual(result.histograms['charge'].total, 6)  # Inclusive hits survive bad associations.

    def test_invalid_regions_and_memory_limit(self):
        path = fixture(self.root/'regions.h5')
        with h5py.File(path, 'r+') as f:
            f['charge/events/ref/charge/calib_prompt_hits/ref_region'][0] = (0, 999)
        opts = settings()
        opts['max_event_links'] = 2
        result = self.sample([path], opts)
        self.assertEqual(result.audit['events_invalid_reference_region'], 1)
        self.assertEqual(result.audit['events_over_link_limit'], 1)
        self.assertEqual(result.audit['events_analyzed'], 1)

    def test_exact_timing_and_all_multiplicities(self):
        path = fixture(self.root/'timing.h5')
        with h5py.File(path) as f:
            hits = f['charge/calib_prompt_hits/data'][:]
        # Four hits on channel 0 -> three adjacent differences, not all combinations.
        audit = Counter()
        mult, delta = timing(hits, 'ts_pps', .1, audit)
        np.testing.assert_array_equal(sorted(mult), [1, 1, 4])
        np.testing.assert_allclose(delta, [.2, .3, .5])
        np.testing.assert_allclose(positive_differences(np.array([-2**63, 2**63-1], dtype='i8')), [float(2**64-1)])
        hits['t_drift'][1] = np.nan
        _, delta = timing(hits, 't_drift', .1, audit)
        self.assertEqual(len(delta), 0)  # Do not bridge past the NaN in channel 0.

    def test_per_file_tick_conversion_and_packet_population(self):
        path = fixture(self.root/'ticks.h5', tick=.2)
        result = self.sample([path])
        self.assertEqual(result.histograms['adc'].total, 2)
        self.assertEqual(result.audit['adc_at_configured_max'], 1)
        self.assertEqual(result.audit['data_packets_invalid_parity'], 1)
        self.assertEqual(result.histograms['delta_us'].total, 2)
        expected = np.histogram([.4, 1.], result.histograms['delta_us'].edges)[0]
        np.testing.assert_array_equal(result.histograms['delta_us'].counts, expected)

    def test_missing_optional_inputs(self):
        path = fixture(self.root/'missing.h5')
        with h5py.File(path, 'r+') as f:
            del f['charge/events/ref']
            del f['charge/packets']
            del f['run_info'].attrs['crs_ticks']
        result = self.sample([path])
        self.assertEqual(result.histograms['charge'].total, 6)
        self.assertEqual(result.histograms['drift_us'].total, 0)
        self.assertEqual(result.audit['events_missing_references'], 3)
        self.assertEqual(len(result.coverage['adc']), 0)
        self.assertTrue(result.warnings)

    def test_medians_are_pooled_and_censored(self):
        first, second = fixture(self.root/'a.h5'), fixture(self.root/'b.h5', repeats=3)
        for path, q in ((first, 1), (second, 9)):
            with h5py.File(path, 'r+') as f:
                data = f['charge/calib_prompt_hits/data'][:]
                data['Q'] = q
                f['charge/calib_prompt_hits/data'][:] = data
        result = self.sample([first, second])
        for row in result.channel_rows():
            self.assertAlmostEqual(row['median_charge'], 9.125)
        result = self.sample([second])
        for channel in result.channels.values():
            channel['charge'].add([1e9]*100)
        self.assertTrue(all(r['median_status'] == 'median_rank_above_range' for r in result.channel_rows()))

    def test_repeated_files_and_aliases_refused(self):
        path = fixture(self.root/'source.h5')
        alias = self.root/'alias.h5'
        alias.symlink_to(path)
        with self.assertRaisesRegex(ValueError, 'Repeated input'):
            resolve_samples([dict(name='test', files=[str(path), str(alias)])], self.root)

    def test_nonfinite_event_charge_not_partially_summed(self):
        path = fixture(self.root/'nonfinite.h5')
        with h5py.File(path, 'r+') as f:
            data = f['charge/calib_prompt_hits/data'][:]
            data['Q'][0] = np.nan
            f['charge/calib_prompt_hits/data'][:] = data
        result = self.sample([path])
        self.assertEqual(result.histograms['charge'].nonfinite, 1)
        self.assertEqual(result.histograms['event_charge'].total, 2)
        self.assertEqual(result.audit['events_excluded_from_charge_nonfinite'], 1)

    def test_failed_run_keeps_failure_manifest(self):
        path = fixture(self.root/'failed.h5')
        config = self.root/'failure.json'
        config.write_text(json.dumps(dict(samples=[dict(name='sample', files=[str(path)])])))
        output = self.root/'failure'
        with patch('data_quality.run.Sample.process', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                run(config, output)
        summary = json.loads((output/'summary.json').read_text())
        self.assertEqual(summary['status'], 'failed')
        self.assertIn('test failure', summary['error'])

    def test_empty_single_sample_report(self):
        path = fixture(self.root/'empty.h5', repeats=0)
        with h5py.File(path, 'r+') as f:
            del f['charge/packets']
        config = self.root/'empty.json'
        config.write_text(json.dumps(dict(samples=[dict(name='empty', files=[str(path)])])))
        output = self.root/'empty-report'
        run(config, output)
        summary = json.loads((output/'summary.json').read_text())
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['samples'][0]['audit']['hit_rows'] if 'hit_rows' in summary['samples'][0]['audit'] else 0, 0)
        self.assertEqual(summary['samples'][0]['histograms']['charge']['total'], 0)

    def test_channel_ratio_missing_and_mismatched_coordinates(self):
        result = self.sample([fixture(self.root/'ratio.h5')])
        table = {tuple(r[n] for n in ADDRESS): r for r in result.channel_rows()}
        other = {key: dict(row) for key, row in table.items()}
        keys = list(other)
        del other[keys[0]]
        other[keys[1]]['y'] += 1
        rows = channel_ratios(table, other)
        self.assertEqual(rows[0]['ratio'], 0)
        self.assertFalse(rows[1]['position_compatible'])

    def test_end_to_end_report_preflight_and_no_overwrite(self):
        a, b = fixture(self.root/'feb.h5'), fixture(self.root/'oct.h5', repeats=2)
        config = self.root/'config.json'
        config.write_text(json.dumps(dict(samples=[dict(name='feb', files=[str(a)]),
                                                   dict(name='oct', files=[str(b)])])))
        output = self.root/'report'
        inventory = run(config, output, check_inputs=True)
        self.assertFalse(output.exists())
        self.assertEqual(len(inventory), 2)
        run(config, output)
        summary = json.loads((output/'summary.json').read_text())
        self.assertEqual(summary['status'], 'complete')
        for name in ('report.html', 'report.pdf', 'plots/charge.png', 'tables/feb_channels.csv'):
            self.assertGreater((output/name).stat().st_size, 100)
        rows = np.genfromtxt(output/'tables/oct_charge.csv', delimiter=',', names=True)
        np.testing.assert_allclose(rows['ratio_to_reference'][np.isfinite(rows['ratio_to_reference'])], 1)
        with self.assertRaises(FileExistsError):
            run(config, output)


if __name__ == '__main__':
    unittest.main()
