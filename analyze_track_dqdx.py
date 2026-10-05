"""Segment merged selected tracks and plot the upstream dQ/dx histogram fit.

Only selected_track_id>=0 hits enter segmentation. Full-event rejected hits remain
in the source file for separate charge-splitting studies. No implicit TPC geometry
or lifetime correction is applied.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pixel_dqdx.segments import segment_track_hits, is_through_going, fiducialize_hits
from pixel_dqdx.dqdx import fit_histogram, langau

SEGMENT_DTYPE = np.dtype([
    ('sample_id', 'i4'), ('file_id', 'i8'), ('source_event_row', 'i8'), ('track_id', 'i8'),
    ('group', 'S64'), ('dq', 'f8'), ('dx', 'f8'), ('dqdx', 'f8'), ('nhits', 'i8'),
    ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('t_drift', 'f8'),
    ('theta', 'f8'), ('phi', 'f8'),
])


def groups_for(hits, file_id, geometry, grouping):
    config = geometry.get('sources', {}).get(str(file_id), geometry)
    if 'groups' in config:
        groups = config['groups']
        used = set()
        for group in groups:
            if not group.get('io_groups') or used.intersection(group['io_groups']):
                raise ValueError('Geometry groups must have disjoint, nonempty io_groups')
            used.update(group['io_groups'])
            for axis in 'xyz':
                bounds = group['bounds'][axis]
                if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                    raise ValueError(f'Invalid physical bounds: {group["name"]}/{axis}')
            if len(group['name'].encode()) > 64:
                raise ValueError('Group name is too long')
        unknown = set(hits['io_group'].tolist()) - used
        if unknown:
            raise ValueError(f'Geometry lacks IO groups {unknown} for file_id={file_id}')
        return groups
    if grouping == 'legacy-tpc':
        if not set(hits['io_group'].tolist()) <= {1, 2, 3, 4}:
            raise ValueError('Legacy TPC map only supports IO groups 1..4')
        return [dict(name='tpc0', io_groups=[1]), dict(name='tpc1', io_groups=[2,3,4])]
    return [dict(name=f'io_group_{int(io)}', io_groups=[int(io)]) for io in np.unique(hits['io_group'])]


def automatic_bounds(source, file_id, margin):
    """Use only finite, nondegenerate stored outer detector bounds in hit cm."""
    path = f'sources/{file_id}/metadata/geometry_info'
    if path not in source or 'lar_detector_bounds' not in source[path].attrs:
        return None, 'stored detector bounds unavailable'
    values = np.asarray(source[path].attrs['lar_detector_bounds'], dtype=float)
    if values.shape != (2,3) or not np.isfinite(values).all():
        return None, 'stored detector bounds have unsupported shape/nonfinite values'
    low, high = values.min(axis=0), values.max(axis=0)
    if np.any(high-low <= 2*margin):
        return None, 'stored detector bounds are degenerate or narrower than twice the margin'
    return {axis:[float(low[i]),float(high[i])] for i,axis in enumerate('xyz')}, 'stored outer detector bounds'


def segment_sample(source_path, out, sample_id, config):
    audit = Counter()
    with h5py.File(source_path, 'r') as source:
        if source.attrs.get('schema_version') != '2.0' or source.attrs.get('status') != 'complete':
            raise ValueError('Analysis requires a complete schema-2 track-selection output')
        provenance = out.require_group(f'samples/{sample_id}')
        provenance.attrs['input_file'] = str(Path(source_path).resolve())
        stat = Path(source_path).stat()
        provenance.attrs['input_size_bytes'], provenance.attrs['input_mtime_ns'] = stat.st_size, stat.st_mtime_ns
        source.copy(source['sources'], provenance, name='sources')
        auto_geometry = {}
        face_policy = {}
        for file_id in source['sources']:
            if config['face_cuts'] == 'auto':
                bounds, reason = automatic_bounds(source,file_id,config['face_margin_cm'])
                auto_geometry[int(file_id)] = bounds
                face_policy[file_id] = dict(bounds=bounds,reason=reason)
        provenance.attrs['automatic_face_policy'] = json.dumps(face_policy)
        audit['sources_with_usable_auto_bounds'] = sum(value['bounds'] is not None for value in face_policy.values())
        audit['sources_without_usable_auto_bounds'] = sum(value['bounds'] is None for value in face_policy.values())
        # Read the merged file by event ranges, not a full hit-table load.
        for start in range(0, len(source['events/data']), 1000):
            for event in source['events/data'][start:start+1000]:
                audit['events'] += 1
                hits = source['events/hits'][int(event['hit_start']):int(event['hit_stop'])]
                if np.any(hits['file_id'] != event['file_id']) or np.any(hits['source_event_row'] != event['source_event_row']):
                    raise ValueError('Event range crosses source-file/event identity')
                hits = hits[hits['selected_track_id'] >= 0]
                for track_id in np.unique(hits['selected_track_id']):
                    track = hits[hits['selected_track_id'] == track_id]
                    audit['selected_tracks'] += 1
                    for group in groups_for(track, int(event['file_id']), config['geometry'], config['grouping']):
                        part = track[np.isin(track['io_group'], group['io_groups'])]
                        if not len(part):
                            continue
                        audit['track_groups'] += 1
                        xyz = np.column_stack([part[d] for d in 'xyz'])
                        good = np.isfinite(xyz).all(axis=1) & np.isfinite(part['Q']) & np.isfinite(part['t_drift'])
                        audit['nonfinite_selected_hits_excluded'] += int((~good).sum())
                        part, xyz = part[good], xyz[good]
                        if len(part) < 3:
                            audit['too_few_hits'] += 1
                            continue
                        if 'bounds' not in group and auto_geometry.get(int(event['file_id'])) is not None:
                            group = dict(group,bounds=auto_geometry[int(event['file_id'])])
                        if 'bounds' in group:
                            if config['require_through_going'] and not is_through_going(xyz, group['bounds'], config['face_margin_cm']):
                                audit['not_through_going'] += 1
                                continue
                            xyz, keep = fiducialize_hits(xyz, group['bounds'], config['face_margin_cm'])
                            part = part[keep]
                            if len(part) < 3:
                                audit['too_few_fiducial_hits'] += 1
                                continue
                        elif config['require_through_going']:
                            raise ValueError('--require-through-going requires explicit geometry bounds')
                        result = segment_track_hits(xyz, part['Q'], part['t_drift'],
                                                    config['segment_length_cm'], config['step_cm'], int(track_id))
                        n = len(result.get('dq', []))
                        if not n:
                            audit['groups_without_segments'] += 1
                            continue
                        values = np.zeros(n, dtype=SEGMENT_DTYPE)
                        values['sample_id'], values['file_id'] = sample_id, event['file_id']
                        values['source_event_row'], values['track_id'] = event['source_event_row'], track_id
                        values['group'] = group['name'].encode()
                        for key in ['dq', 'dx', 'nhits', 'x', 'y', 'z', 't_drift', 'theta', 'phi']:
                            values[key] = result[key]
                        values['dqdx'] = values['dq']/values['dx']
                        dataset = out['segments/data']
                        offset = len(dataset)
                        dataset.resize(offset+n, axis=0)
                        dataset[offset:] = values
                        audit['segments'] += n
        provenance.attrs['audit'] = json.dumps(dict(audit))
    return dict(audit)


def fit_counts(counts, edges, fit_range, min_fit_entries):
    centers = .5*(edges[:-1]+edges[1:])
    mask = (centers > fit_range[0]) & (centers < fit_range[1])
    if mask.sum() <= 4 or counts[mask].sum() < min_fit_entries or np.count_nonzero(counts[mask]) < 5:
        return None, 'insufficient populated fit bins/entries'
    # Reuse upstream weighting, bounds, histogram and numerical convolution.
    return fit_histogram(None, langau, None, fit_range, (edges[0],edges[-1]),
                         nbins=len(counts), histogram=(counts,edges),
                         max_width=edges[-1]-edges[0]), None


def plot_panel(ax, counts, edges, title, fit_range, min_fit_entries):
    ax.stairs(counts, edges, fill=True, color='steelblue', label='Data' if 'Simulation' not in title else 'Simulation')
    result, failure = fit_counts(counts, edges, fit_range, min_fit_entries)
    summary = dict(title=title, n_entries_in_histogram=int(counts.sum()))
    if result is not None:
        ax.plot(result['x_fit'], result['fit_vals'], color='red', label='Fit')
        mpv = float(result['params'][0])
        text = f"$\\chi^2/ndf={result['chi2_red']:.2f}$\nComponent MPV = {mpv:.2f}"
        summary.update(status='fit', params=result['params'].tolist(), covariance=result['cov'].tolist(),
                       mpv=mpv, mpv_error=float(result['mpv_error']), chi2_red=float(result['chi2_red']),
                       ndf=len(result['x_fit'])-4)
    else:
        text = 'Fit unavailable'
        summary.update(status='no_fit', reason=failure or 'optimizer failed')
    ax.text(.95,.55,text,transform=ax.transAxes,ha='right',va='center',
            bbox=dict(boxstyle='round',facecolor='white',edgecolor='0.4'))
    ax.set(title=title,xlabel='dQ/dx [ke−/cm]',ylabel='Number of track segments',xlim=(edges[0],edges[-1]))
    ax.legend()
    return summary


def analyze(data=None, output_dir=None, mc=None, segment_length_cm=3., step_cm=3.,
            grouping='io-group', geometry=None, require_through_going=False, face_cuts='auto',
            face_margin_cm=1., bins=100, hist_range=(0.,90.), fit_range=(20.,50.), min_fit_entries=100):
    if data is None and mc is None:
        raise ValueError('Provide at least one of --data or --mc')
    if output_dir is None:
        raise ValueError('output_dir is required')
    if face_cuts not in {'auto','none'}:
        raise ValueError('face_cuts must be auto or none')
    if not (np.isfinite([segment_length_cm,step_cm,face_margin_cm,*hist_range,*fit_range]).all()
            and segment_length_cm > 0 and step_cm > 0 and face_margin_cm >= 0
            and bins > 4 and min_fit_entries > 0
            and hist_range[0] < fit_range[0] < fit_range[1] < hist_range[1]):
        raise ValueError('Invalid segment scales, histogram bins, or fit/histogram ranges')
    if require_through_going and not geometry:
        raise ValueError('--require-through-going requires --geometry-json')
    config = dict(segment_length_cm=segment_length_cm,step_cm=step_cm,grouping=grouping,
                  geometry=geometry or {},require_through_going=require_through_going,
                  face_margin_cm=face_margin_cm,face_cuts=face_cuts,bins=bins,hist_range=list(hist_range),
                  fit_range=list(fit_range),min_fit_entries=min_fit_entries,
                  fit_model='upstream Moyal-like Landau approximation convolved with Gaussian',
                  charge_field='Q',charge_units_assumed='ke−',hit_view='selected only',
                  boundary_policy='upstream inclusive endpoints; exact boundary hits can appear in adjacent windows')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    # Stable identities: data=0, MC=1, including single-sample runs.
    samples = ([('data','FSD Data Fit',data)] if data is not None else [])
    samples += ([('mc','FSD Simulation Fit',mc)] if mc is not None else [])
    segment_file = output_dir/'segments.hdf5'
    manifest = dict(status='incomplete',settings=config,samples=[],
                    upstream_commit='33cf2fe926afe416b31fe930f18dff80b9ab47d8',
                    generated_utc=datetime.now(timezone.utc).isoformat())
    for name in ['analyze_track_dqdx.py','pixel_dqdx/segments.py','pixel_dqdx/dqdx.py']:
        manifest.setdefault('source_hashes',{})[name] = hashlib.sha256(Path(__file__).parent.joinpath(name).read_bytes()).hexdigest()
    manifest_path = output_dir/'analysis.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        with h5py.File(segment_file,'x') as out:
            out.attrs['status'] = 'incomplete'
            out.attrs['settings'] = json.dumps(config)
            out.create_dataset('segments/data',shape=(0,),maxshape=(None,),dtype=SEGMENT_DTYPE,chunks=True,compression='gzip')
            for kind,title,path in samples:
                sample_id = 0 if kind == 'data' else 1
                audit = segment_sample(path,out,sample_id,config)
                out[f'samples/{sample_id}'].attrs['sample_kind'] = kind
                out[f'samples/{sample_id}'].attrs['title'] = title
                manifest['samples'].append(dict(sample_id=sample_id,sample_kind=kind,
                                               title=title,input=str(Path(path).resolve()),audit=audit))
            out.attrs['status'] = 'complete'
        edges = np.linspace(*hist_range,bins+1)
        counts = [np.zeros(bins,dtype=np.int64) for _ in samples]
        tails = [dict(below=0,above=0) for _ in samples]
        # Histogram bounded chunks rather than materializing all segments.
        with h5py.File(segment_file,'r') as f:
            ds = f['segments/data']
            for start in range(0,len(ds),100000):
                values = ds[start:start+100000]
                for index,(kind,_,_) in enumerate(samples):
                    sample_id = 0 if kind == 'data' else 1
                    dqdx = values['dqdx'][values['sample_id']==sample_id]
                    counts[index] += np.histogram(dqdx,bins=edges)[0]
                    tails[index]['below'] += int((dqdx<edges[0]).sum())
                    tails[index]['above'] += int((dqdx>edges[-1]).sum())
        fit_summaries = []
        for index,(kind,title,_) in enumerate(samples):
            fig,ax = plt.subplots(figsize=(8,5))
            result = plot_panel(ax,counts[index],edges,title,fit_range,min_fit_entries)
            result['sample_id'] = 0 if kind == 'data' else 1
            result['sample_kind'] = kind
            result['outside_histogram_range'] = tails[index]
            fit_summaries.append(result)
            fig.text(.5,.01,'Fit: upstream Moyal-like approximation × Gaussian convolution',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.035,1,1))
            stem = f'{kind}_dqdx'
            for ext in ['png','pdf']:
                fig.savefig(output_dir/f'{stem}.{ext}',dpi=180)
            plt.close(fig)
            np.savetxt(output_dir/f'{stem}_histogram.csv',
                       np.column_stack((edges[:-1],edges[1:],counts[index])),delimiter=',',
                       header='bin_low,bin_high,count',comments='')
        if len(samples)==2:
            fig,axes = plt.subplots(1,2,figsize=(14,5))
            # Replot stored fit results instead of refitting.
            for i,ax in enumerate(axes):
                ax.stairs(counts[i],edges,fill=True,color='steelblue',label='Data' if i==0 else 'Simulation')
                summary = fit_summaries[i]
                if summary['status']=='fit':
                    centers = .5*(edges[:-1]+edges[1:])
                    x = centers[(centers>fit_range[0])&(centers<fit_range[1])]
                    ax.plot(x,langau(x,*summary['params']),color='red',label='Fit')
                    text=f"$\\chi^2/ndf={summary['chi2_red']:.2f}$\nComponent MPV = {summary['mpv']:.2f}"
                else:
                    text='Fit unavailable'
                ax.text(.95,.55,text,transform=ax.transAxes,ha='right',bbox=dict(boxstyle='round',facecolor='white',edgecolor='0.4'))
                ax.set(title=samples[i][1],xlabel='dQ/dx [ke−/cm]',ylabel='Number of track segments',xlim=hist_range)
                ax.legend()
            fig.text(.5,.01,'Fit: upstream Moyal-like approximation × Gaussian convolution',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.035,1,1))
            for ext in ['png','pdf']:
                fig.savefig(output_dir/f'data_mc_dqdx.{ext}',dpi=180)
            plt.close(fig)
        manifest.update(status='complete',fits=fit_summaries)
    except Exception as exc:
        manifest.update(status='failed',error=str(exc))
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        raise
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


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
