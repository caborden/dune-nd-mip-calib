import unittest
from pathlib import Path
from unittest.mock import patch
import test_compare_dqdx
from analysis.compare_runs import compare
from analysis.hit_density import make_settings
from analysis.io import read_json, write_json, code_hashes


class CompareRunsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_compare_dqdx.DqdxOverlayTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.members = []
        for i, (_, dqdx) in enumerate(self.fixture.members[:2]):
            directory = dqdx.parent
            label = ['Feb 2026 prc16 data', 'Oct 2026 prc16 data'][i]
            self.members.append((label, directory))
            write_json(directory/'manifest.json', dict(stages={name:dict(status='complete') for name in ['segments','dqdx','hit_density','selection_hits']}))
            write_json(directory/'hit_density/analysis.json',dict(status='complete',settings=make_settings(),
                source_hashes=code_hashes(['analysis/hit_density.py']),segment_settings={'segment_length_cm':3},
                samples=[dict(sample_kind='data',bins=[dict(bin_center=250,value=5+i,error=.1)])]))
            write_json(directory/'selection_hits/analysis.json',dict(status='complete',settings={'charge_bins':2,'charge_range':[0,100]},
                source_hashes=code_hashes(['analysis/selection_hits.py']),
                samples=[dict(sample_kind='data',charge_counts=[2,4],charge_edges=[0,50,100],multiplicity={'1':5,'2':2})]))

    def test_render_labels_and_preserve_source_settings(self):
        from analysis.compare_runs import save_plot
        with patch('analysis.compare_runs.save_plot', wraps=save_plot) as draw:
            directory = compare(self.root,self.members,'oct-vs-feb','run-001')
        labels = [s['label'] for s in draw.call_args.args[2]]
        self.assertEqual(labels,[l for l,_ in self.members])
        self.assertEqual(draw.call_args.args[3]['drift_range_ticks'],[0.,5000.])
        self.assertEqual(len(list(directory.glob('*.pdf'))),4)
        self.assertEqual(read_json(directory/'manifest.json')['status'],'complete')
        with self.assertRaises(FileExistsError):
            compare(self.root,self.members,'oct-vs-feb','run-001')

    def test_stale_and_incompatible_reports_rejected(self):
        path=self.members[1][1]/'hit_density/analysis.json'
        report=read_json(path)
        report['settings']['drift_bins']=18
        write_json(path,report)
        with self.assertRaisesRegex(ValueError,'settings/source differ'):
            compare(self.root,self.members,'bad','run-001')
        self.assertFalse((self.root/'comparisons').exists())
        path=self.members[1][1]/'manifest.json'
        report=read_json(path)
        report['stages']['selection_hits']['status']='stale'
        write_json(path,report)
        with self.assertRaisesRegex(ValueError,'not complete'):
            compare(self.root,self.members,'bad','run-001')
