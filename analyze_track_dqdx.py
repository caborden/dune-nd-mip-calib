"""Segment merged selected tracks and plot the upstream dQ/dx histogram fit.

Only selected_track_id>=0 hits enter segmentation. Full-event rejected hits remain
in the source file for separate charge-splitting studies. No implicit TPC geometry
or lifetime correction is applied.
"""
import argparse
import json
from pathlib import Path
from analysis.dqdx import analyze, fit_counts, plot_panel
from analysis.segments import SEGMENT_DTYPE, groups_for, automatic_bounds, segment_sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',help='completed merged data track-selection HDF5')
    parser.add_argument('--mc',help='completed merged simulation track-selection HDF5')
    parser.add_argument('--output-dir',required=True,help='new directory for plots/tables')
    parser.add_argument('--segment-length-cm',type=float,default=3.)
    parser.add_argument('--step-cm',type=float,default=3.)
    parser.add_argument('--grouping',choices=['io-group','legacy-tpc'],default='io-group')
    parser.add_argument('--face-cuts',choices=['auto','none'],default='auto',
                        help='auto applies 1 cm outer-face cuts only for valid stored bounds; none disables automatic cuts')
    parser.add_argument('--geometry-json',help='explicit group bounds, optionally keyed by source file_id')
    parser.add_argument('--require-through-going',action='store_true')
    parser.add_argument('--face-margin-cm',type=float,default=1.)
    parser.add_argument('--bins',type=int,default=100)
    parser.add_argument('--hist-range',type=float,nargs=2,default=(0.,90.))
    parser.add_argument('--fit-range',type=float,nargs=2,default=(20.,50.))
    parser.add_argument('--min-fit-entries',type=int,default=100)
    args = vars(parser.parse_args())
    if args['data'] is None and args['mc'] is None:
        parser.error('provide at least one of --data or --mc')
    geometry_path = args.pop('geometry_json')
    if geometry_path:
        args['geometry'] = json.loads(Path(geometry_path).read_text())
    result = analyze(**args)
    for sample in result['samples']:
        print(sample['title'], 'automatic bounds usable for',sample['audit'].get('sources_with_usable_auto_bounds',0),
              'sources; unavailable for',sample['audit'].get('sources_without_usable_auto_bounds',0))
    for fit in result['fits']:
        print(fit['title'],fit['status'], 'MPV=',fit.get('mpv'), 'chi2/ndf=',fit.get('chi2_red'))
