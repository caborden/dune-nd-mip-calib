"""Shared opt-in presentation exports; PDF and scientific records remain standard."""
import numpy as np


def add_export_arguments(parser):
    parser.add_argument('--write-csv', action='store_true', help='also export numerical CSV tables')
    parser.add_argument('--write-png', '-write-png', action='store_true',
                        help='also export PNG figures (PDF is always written)')


def figure_extensions(write_png=False):
    return ('pdf', 'png') if write_png else ('pdf',)


def save_csv(enabled, *args, **kwargs):
    if enabled:
        np.savetxt(*args, **kwargs)
