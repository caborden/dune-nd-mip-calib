"""Segment merged selected tracks and plot the upstream dQ/dx histogram fit.

Only selected_track_id>=0 hits enter segmentation. Full-event rejected hits remain
in the source file for separate charge-splitting studies. No implicit TPC geometry
or lifetime correction is applied.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .segments import build_segments
from .config import make_settings
from .io import code_hashes
from pixel_dqdx.dqdx import fit_histogram, langau


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



def plot_segments(segment_file, output_dir, config):
    """dQ/dx module: consume shared segments, write only plots/fit products."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_range, bins = config['hist_range'], config['bins']
    fit_range, min_fit_entries = config['fit_range'], config['min_fit_entries']
    samples, records = [], []
    with h5py.File(segment_file, 'r') as f:
        if f.attrs.get('status') != 'complete':
            raise ValueError('Shared segments are incomplete')
        for sample_id in sorted(f['samples'], key=int):
            attrs = f[f'samples/{sample_id}'].attrs
            kind, title = attrs['sample_kind'], attrs['title']
            samples.append((kind, title, attrs['input_file']))
            records.append(dict(sample_id=int(sample_id), sample_kind=kind, title=title,
                                input=attrs['input_file'], audit=json.loads(attrs['audit'])))
    manifest = dict(status='incomplete', settings=config, samples=records,
                    segment_file=str(Path(segment_file).resolve()),
                    upstream_commit='33cf2fe926afe416b31fe930f18dff80b9ab47d8',
                    generated_utc=datetime.now(timezone.utc).isoformat(),
                    source_hashes=code_hashes())
    manifest_path = output_dir/'analysis.json'
    try:
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
        manifest.update(status='complete', fits=fit_summaries)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        raise
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def analyze(data=None, output_dir=None, mc=None, **settings):
    """Compatibility interface retaining the original standalone output layout."""
    if data is None and mc is None:
        raise ValueError('Provide at least one of --data or --mc')
    if output_dir is None:
        raise ValueError('output_dir is required')
    config = make_settings(**settings)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = ([('data', 'FSD Data Fit', data)] if data is not None else [])
    samples += ([('mc', 'FSD Simulation Fit', mc)] if mc is not None else [])
    try:
        build_segments(samples, output_dir/'segments.hdf5', config)
        return plot_segments(output_dir/'segments.hdf5', output_dir, config)
    except Exception as exc:
        if not (output_dir/'analysis.json').exists():
            (output_dir/'analysis.json').write_text(json.dumps(dict(status='failed',settings=config,error=str(exc)),indent=2)+'\n')
        raise
