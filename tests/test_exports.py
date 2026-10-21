"""Shared CLI export defaults and exact spelling supported across entry points."""
import argparse
import unittest

from analysis.exports import add_export_arguments, figure_extensions


class ExportTests(unittest.TestCase):
    def test_independent_flags_and_alias(self):
        parser = argparse.ArgumentParser()
        add_export_arguments(parser)
        self.assertEqual(vars(parser.parse_args([])), dict(write_csv=False, write_png=False))
        self.assertEqual(vars(parser.parse_args(['--write-csv'])), dict(write_csv=True, write_png=False))
        for flag in ['--write-png', '-write-png']:
            self.assertEqual(vars(parser.parse_args([flag])), dict(write_csv=False, write_png=True))
        self.assertEqual(figure_extensions(), ('pdf',))
        self.assertEqual(figure_extensions(True), ('pdf', 'png'))
