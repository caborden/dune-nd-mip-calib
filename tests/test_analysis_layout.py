"""Exercise migration preservation, module dependencies, cache checks and recovery."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from analysis.io import write_json, read_json
from analysis.run import run
from analyze_track_dqdx import analyze
from reorganize_outputs import (migrate, check, layout, DATA_ID, MC_ID,
                                COMPARISON_ID, SPECS)
import test_track_output_hdf5 as fixtures


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OutputTests()
        self.fixture.setUp()
        self.fixture.fixture()
        with redirect_stdout(io.StringIO()):
            fixtures.import_selector().run(str(self.fixture.source), str(self.fixture.output), False)
        self.root = self.fixture.root/'scratch'
        self.base = self.root/'fsdcube'
        for sample, _, legacy, filename, pilot in SPECS:
            directory = self.root/'fsdcube-charge-splitting'/legacy
            directory.mkdir(parents=True)
            shutil.copy2(self.fixture.output, directory/filename)
            shutil.copy2(self.fixture.output, directory/pilot)
            (directory/'inputs.txt').write_text('FLOW.hdf5\n')
            for part in ['full_parts', 'pilot_parts']:
                (directory/part).mkdir()
                shutil.copy2(self.fixture.output, directory/part/'one.hdf5')
                write_json(directory/part/'batch_manifest.json', dict(status='complete', inputs=[{}],
                    files=[dict(output='/historical/path/one.hdf5', status='complete')]))
        analyze(self.fixture.output, self.base/'dqdx_data_v1', face_cuts='none')
        analyze(self.fixture.output, self.base/'dqdx_data_mc_v1', mc=self.fixture.output, face_cuts='none')
        archive = self.root/'fsdcube-charge-splitting/FSDCubeSim_v1_prc256-20261003-191727'
        archive.mkdir()
        (archive/'inputs.txt').write_text('old list\n')
        self.config = self.fixture.root/'config.json'
        write_json(self.config, dict(target=dict(kind='comparison', id=COMPARISON_ID),
                                    segments=dict(face_cuts='none')))

    def tearDown(self):
        self.fixture.tearDown()

    def migrate(self):
        with redirect_stdout(io.StringIO()):
            return migrate(self.root, apply=True)

    def test_migration_preserves_bytes_links_and_repeated_application(self):
        with redirect_stdout(io.StringIO()):
            migrate(self.root)
        self.assertFalse((self.base/'samples').exists())
        record = self.migrate()
        self.assertEqual(record['status'], 'complete')
        for old, new in layout(self.root)[1]:
            self.assertTrue(old.is_symlink())
            self.assertEqual(old.resolve(), new.resolve())
        self.assertEqual(read_json(self.base/'comparisons'/COMPARISON_ID/'analysis/run-001/manifest.json')['format'],
                         'imported-analysis-v1')
        with redirect_stdout(io.StringIO()):
            check(self.root)
            self.assertEqual(migrate(self.root, apply=True)['preserved_files'], record['preserved_files'])
        # Detection includes historical logs/JSON/figures, not just HDF5 readability.
        (self.base/'dqdx_data_v1/analysis.json').write_text('changed')
        with self.assertRaises((ValueError, KeyError, json.JSONDecodeError)):
            check(self.root)

    def test_collision_and_incomplete_input_leave_original_tree_untouched(self):
        destination = self.base/'samples'/DATA_ID/'selection/v1'
        destination.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.migrate()
        self.assertFalse((self.root/'fsdcube-charge-splitting/run-20261003-151120').is_symlink())
        destination.rmdir()
        old = self.root/'fsdcube-charge-splitting/run-20261003-151120/full_parts/one.hdf5'
        with h5py.File(old, 'r+') as f:
            f.attrs['status'] = 'incomplete'
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.migrate()
        self.assertFalse((self.base/'relocation.json').exists())

    def test_interrupted_migration_can_continue(self):
        # Fail after the first move was journaled; all later operations are recoverable.
        from reorganize_outputs import write_json as real_write
        def interrupt(path, value):
            real_write(path, value)
            if Path(path).name == 'relocation.json' and len(value.get('completed_moves', [])) == 1:
                raise RuntimeError('simulated interruption')
        with patch('reorganize_outputs.write_json', side_effect=interrupt):
            with self.assertRaises(RuntimeError):
                self.migrate()
        self.assertEqual(self.migrate()['status'], 'complete')

    def test_runner_reuses_segments_adds_plots_and_matches_legacy(self):
        self.migrate()
        with redirect_stdout(io.StringIO()):
            directory = run(self.config, self.base, 'run-002', modules=['segments'])
        shared = directory/'shared/segments.hdf5'
        before = shared.stat().st_mtime_ns
        self.assertFalse((directory/'dqdx').exists())
        with redirect_stdout(io.StringIO()):
            run(self.config, self.base, 'run-002', modules=['dqdx'], resume=True)
            run(self.config, self.base, 'run-002', modules=['all'], resume=True)
        self.assertEqual(shared.stat().st_mtime_ns, before)
        self.assertTrue((directory/'dqdx/data_mc_dqdx.pdf').exists())
        self.assertTrue((directory/'hit_density/data_mc_hit_density.png').exists())
        self.assertEqual(read_json(directory/'manifest.json')['stages']['hit_density']['status'], 'complete')
        legacy = self.base/'dqdx_data_mc_v1'
        with h5py.File(shared) as new, h5py.File(legacy/'segments.hdf5') as old:
            np.testing.assert_array_equal(new['segments/data'][:], old['segments/data'][:])
        for kind in ['data', 'mc']:
            np.testing.assert_array_equal(np.loadtxt(directory/f'dqdx/{kind}_dqdx_histogram.csv', delimiter=',', skiprows=1),
                                          np.loadtxt(legacy/f'{kind}_dqdx_histogram.csv', delimiter=',', skiprows=1))
        manifest = (directory/'manifest.json').read_bytes()
        config = read_json(self.config)
        config['segments']['segment_length_cm'] = 4
        write_json(self.config, config)
        with self.assertRaisesRegex(ValueError, 'new run ID'):
            run(self.config, self.base, 'run-002', resume=True)
        self.assertEqual((directory/'manifest.json').read_bytes(), manifest)

    def test_failed_plot_retry_reuses_completed_segments(self):
        self.migrate()
        with patch('analysis.run.plot_segments', side_effect=RuntimeError('plot failure')):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'plot failure'):
                run(self.config, self.base, 'run-002')
        directory = self.base/'comparisons'/COMPARISON_ID/'analysis/run-002'
        record = read_json(directory/'manifest.json')
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['stages']['segments']['status'], 'complete')
        before = (directory/'shared/segments.hdf5').stat().st_mtime_ns
        with redirect_stdout(io.StringIO()):
            run(self.config, self.base, 'run-002', resume=True)
        self.assertEqual((directory/'shared/segments.hdf5').stat().st_mtime_ns, before)
        self.assertEqual(read_json(directory/'manifest.json')['status'], 'complete')

    def test_mc_only_unknown_modules_imported_runs_and_changed_outputs(self):
        self.migrate()
        write_json(self.config, dict(target=dict(kind='sample', id=MC_ID),
                                    modules=dict(segments=dict(segment_length_cm=4))))
        with self.assertRaisesRegex(ValueError, 'top-level segments'):
            run(self.config, self.base, 'misplaced-settings', modules=['segments'])
        self.assertFalse((self.base/'samples'/MC_ID/'analysis/misplaced-settings').exists())
        write_json(self.config, dict(target=dict(kind='sample', id=MC_ID), segments=dict(face_cuts='none')))
        with self.assertRaisesRegex(ValueError, 'Unknown module'):
            run(self.config, self.base, 'invalid', modules=['pixel_charge'])
        self.assertFalse((self.base/'samples'/MC_ID/'analysis/invalid').exists())
        with redirect_stdout(io.StringIO()):
            directory = run(self.config, self.base, 'run-002')
        self.assertTrue((directory/'dqdx/mc_dqdx.png').exists())
        self.assertFalse((directory/'dqdx/data_dqdx.png').exists())
        with redirect_stdout(io.StringIO()):
            density_dir = run(self.config, self.base, 'density-only', modules=['hit_density'])
        self.assertTrue((density_dir/'hit_density/mc_hit_density.png').exists())
        self.assertFalse((density_dir/'dqdx').exists())
        with self.assertRaises(FileExistsError):
            run(self.config, self.base, 'run-002')
        (directory/'dqdx/mc_dqdx.png').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            run(self.config, self.base, 'run-002', resume=True)
        write_json(self.config, dict(target=dict(kind='sample', id=DATA_ID)))
        with self.assertRaisesRegex(ValueError, 'historical'):
            run(self.config, self.base, 'run-001', resume=True)
