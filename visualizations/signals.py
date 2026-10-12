"""Portable signal bundles and synchronized plots (no CUDA dependency).

Demo electronics are intentionally illustrative, not a CPU clone of larnd-sim.
Real capture bundles contain actual larnd-sim currents and pre-packet ADC results.
"""
from pathlib import Path
import json
import numpy as np


def validate_bundle(b):
    required=['time_us','current_e_per_us','pixel_id','pixel_xyz_cm','adc_channel_index',
              'adc_time_us','adc_value','metadata_json']
    missing=set(required)-set(b)
    if missing: raise ValueError(f'Missing bundle fields: {sorted(missing)}')
    t=np.asarray(b['time_us']); currents=np.asarray(b['current_e_per_us'])
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):
        raise ValueError('time_us must be finite and strictly increasing, with >=2 samples')
    if not np.allclose(np.diff(t),t[1]-t[0],rtol=1e-7,atol=1e-10):
        raise ValueError('time_us must have uniform sample spacing')
    if currents.ndim!=2 or currents.shape[1]!=len(t) or not np.isfinite(currents).all():
        raise ValueError('current_e_per_us must be finite (channels, samples)')
    n=currents.shape[0]
    if n<1 or np.shape(b['pixel_id'])!=(n,) or len(np.unique(b['pixel_id']))!=n:
        raise ValueError('Need one unique pixel ID per channel')
    if np.shape(b['pixel_xyz_cm'])!=(n,3): raise ValueError('pixel_xyz_cm must be (channels,3)')
    a=np.asarray(b['adc_channel_index'])
    if a.ndim!=1 or a.dtype.kind not in 'iu' or np.any((a<0)|(a>=n)):
        raise ValueError('ADC channel indices must index the captured channel table')
    if np.shape(b['adc_time_us'])!=a.shape or np.shape(b['adc_value'])!=a.shape:
        raise ValueError('ADC columns must have equal lengths')
    for key in ['pixel_xyz_cm','adc_time_us','adc_value']:
        if not np.isfinite(b[key]).all(): raise ValueError(f'{key} must be finite')
    if 'csa_e' in b and (np.shape(b['csa_e'])!=currents.shape or not np.isfinite(b['csa_e']).all()):
        raise ValueError('csa_e must be finite and match the current shape')
    if 'segments_xyz_cm' in b:
        seg=np.asarray(b['segments_xyz_cm'])
        if seg.ndim!=3 or seg.shape[1:]!=(2,3) or not np.isfinite(seg).all():
            raise ValueError('segments_xyz_cm must be finite (segments,2,3)')
    metadata=json.loads(str(np.asarray(b['metadata_json']).item()))
    if metadata.get('schema_version')!=1: raise ValueError('Unsupported signal bundle version')
    if metadata.get('kind') not in ['illustrative','larnd-sim-capture']:
        raise ValueError('Bundle kind must explicitly distinguish demonstration and simulation')
    return metadata


def save_bundle(path,b):
    validate_bundle(b)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Never overwrite an existing trace accidentally; npz stores no pickle objects.
    with path.open('xb') as stream: np.savez_compressed(stream,**b)
    return path.resolve()


def load_bundle(path):
    with np.load(path,allow_pickle=False) as f: b={k:f[k] for k in f.files}
    validate_bundle(b)
    return b


def illustrative_electronics(current,dt,threshold=5000.,reset_period_us=51.2,
                            reset_phase_us=0.,hold_us=2.3,busy_us=1.,gain=.0034,
                            pedestal=980.,v_ref=1734.375,v_cm=0.,adc_counts=1023,
                            adc_scale=2,adc_offset=0):
    """Noise-free integrate/threshold/hold/reset teaching model, in electrons/us.

    It omits buffer response, random reset phase/noise and larnd-sim kernel details.
    ADC transfer follows fee.digitize's formula; timestamps here are sample times.
    Periodic resets are deferred/skipped during hold and busy states in this model.
    """
    if dt<=0 or threshold<=0 or hold_us<0 or busy_us<0 or v_ref<=v_cm or adc_scale<=0:
        raise ValueError('Invalid electronics parameters')
    a=np.asarray(current,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all(): raise ValueError('Expected finite one-channel current')
    period=max(1,round(reset_period_us/dt)) if reset_period_us>0 else 0
    phase=round(reset_phase_us/dt)%period if period else 0
    q=0.; trace=np.zeros(len(a)); adct=[]; adcv=[]; resets=[]; sample=None; busy_until=-1
    hold=round(hold_us/dt)
    for i,value in enumerate(a):
        if i<busy_until:
            trace[i]=0
            continue
        if sample is None and period and i%period==phase:
            q=0.;resets.append(i*dt)
        q+=value*dt
        trace[i]=q
        if sample is None and q>=threshold: sample=i+hold
        if sample is not None and i>=sample:
            if q>=threshold:
                adc=np.floor(np.clip(max(q*gain+pedestal-v_cm,0)*adc_counts/((v_ref-v_cm)*adc_scale)+adc_offset,0,adc_counts-1))
                adct.append(i*dt);adcv.append(adc)
            resets.append(i*dt);q=0.;sample=None;busy_until=i+max(1,round(busy_us/dt))
    return trace,np.array(adct),np.array(adcv),np.array(resets)


def demo_bundle(g,reset_period_us=51.2,threshold_e=5000.,reset_phase_us=0.):
    """Analytic pulses plus a schematic muon, explicitly NOT detector predictions."""
    from .geometry import tile_pixels
    tile=next(iter(g.tile_centers));p=tile_pixels(g,tile); xyz=p['xyz_cm']
    center=g.tile_centers[tile]
    rows=[]
    for offset in [0,1,6]:
        target=center+np.array([0,0,offset*g.pitch_cm])
        rows.append(int(np.argmin(np.linalg.norm(xyz-target,axis=1))))
    if len(set(rows))!=3: raise ValueError('Demo needs three distinct pixels')
    dt=.1; t=np.arange(0,120,dt)
    pulse=np.exp(-.5*((t-60)/2.5)**2);pulse/=pulse.sum()*dt
    # d/dt Gaussian yields a bipolar response with zero discrete integral.
    bipolar=-(t-60)/8**2*np.exp(-.5*((t-60)/8)**2)
    bipolar-=bipolar.mean()
    bipolar/=np.sum(bipolar[bipolar>0])*dt
    currents=np.array([18000*pulse,6000*pulse+3500*bipolar,8000*bipolar])
    prop=g.properties
    params=dict(gain=prop.get('larpix_gain',.004),pedestal=prop.get('v_pedestal',580.),
                v_ref=prop.get('v_ref',1568.),v_cm=prop.get('v_cm',478.1),
                adc_counts=prop.get('adc_counts',256),adc_scale=prop.get('adc_scale_factor',1),
                adc_offset=prop.get('adc_offset',0),hold_us=(3+prop.get('adc_hold_delay',15))*.1,
                busy_us=1.,reset_period_us=reset_period_us,reset_phase_us=reset_phase_us,
                threshold=threshold_e)
    csas=[];adc_rows=[];adc_times=[];adc_values=[];reset_rows=[];reset_times=[]
    for i,current in enumerate(currents):
        c,at,av,rt=illustrative_electronics(current,dt,**params)
        csas.append(c);adc_rows.extend([i]*len(at));adc_times.extend(at);adc_values.extend(av)
        reset_rows.extend([i]*len(rt));reset_times.extend(rt)
    bounds=next(iter(g.tpc_bounds.values()))
    mid=np.mean(bounds,axis=1)
    track=np.array([mid+[-np.ptp(bounds[0])/2+.5,-6,-8],mid+[np.ptp(bounds[0])/2-.5,6,8]])
    b=dict(time_us=t,current_e_per_us=currents,pixel_id=p['pixel_id'][rows],pixel_xyz_cm=xyz[rows],
           adc_channel_index=np.array(adc_rows,dtype=int),adc_time_us=np.array(adc_times),adc_value=np.array(adc_values),
           csa_e=np.array(csas),threshold_e=np.full(3,threshold_e),reset_channel_index=np.array(reset_rows,dtype=int),
           reset_times_us=np.array(reset_times),segments_xyz_cm=track[None,:,:],
           labels=np.array(['collection-like','neighbor-like','induction-only-like']),
           metadata_json=np.array(json.dumps(dict(schema_version=1,kind='illustrative',detector=g.name,
               description='Analytic pulses; schematic track is not transported. Not a larnd-sim prediction.',
               electronics=params,time_reference='local illustrative time [us]',layout=str(g.layout_path)))))
    validate_bundle(b)
    return b


def signal_figure(b,channel=0):
    """One chosen pixel: context, current, CSA/integral, and ADC on linked time axes."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    meta=validate_bundle(b)
    if not 0<=channel<len(b['pixel_id']): raise IndexError('channel out of range')
    measured='csa_e' in b
    subtitle='Illustrative amplifier charge [e−]' if measured and meta['kind']=='illustrative' else 'Current integral [e−] (not amplifier history)'
    fig=make_subplots(rows=3,cols=2,specs=[[{'type':'scene','rowspan':3},{}],[None,{}],[None,{}]],
                     column_widths=[.42,.58],horizontal_spacing=.08,vertical_spacing=.12,
                     subplot_titles=['Track / captured pixel centers','Induced current',subtitle,'ADC readings'])
    xyz=np.asarray(b['pixel_xyz_cm']);colors=['#d1d5db']*len(xyz);colors[channel]='#dc2626'
    fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,2],z=xyz[:,1],mode='markers',
        marker=dict(size=6,color=colors),text=[f'pixel {v}' for v in b['pixel_id']],hoverinfo='text',name='Captured pixels'),row=1,col=1)
    if 'segments_xyz_cm' in b and len(b['segments_xyz_cm']):
        seg=np.asarray(b['segments_xyz_cm']);sep=np.full((len(seg),1,3),np.nan)
        a=np.concatenate([seg,sep],axis=1).reshape(-1,3)
        fig.add_trace(go.Scatter3d(x=a[:,0],y=a[:,2],z=a[:,1],mode='lines',line=dict(color='#334155',width=3),
                                 name='Schematic muon' if meta['kind']=='illustrative' else 'Input energy deposits'),row=1,col=1)
    t=b['time_us'];current=b['current_e_per_us'][channel]
    fig.add_trace(go.Scatter(x=t,y=current,name='Current',line_color='#2563eb'),row=1,col=2)
    q=b['csa_e'][channel] if measured else np.cumsum(current)*np.median(np.diff(t))
    fig.add_trace(go.Scatter(x=t,y=q,name='Amplifier (illustrative)' if measured else 'Unreset current integral',line_color='#059669'),row=2,col=2)
    if measured and 'threshold_e' in b:
        fig.add_trace(go.Scatter(x=[t[0],t[-1]],y=[float(b['threshold_e'][channel])]*2,
                                name='Threshold',line=dict(dash='dash',color='#d97706')),row=2,col=2)
    if measured and 'reset_times_us' in b:
        times=b['reset_times_us'][b['reset_channel_index']==channel]
        fig.add_trace(go.Scatter(x=times,y=np.zeros(len(times)),mode='markers',marker=dict(symbol='x',color='#d97706'),
                                 name='Model resets'),row=2,col=2)
    mask=b['adc_channel_index']==channel
    fig.add_trace(go.Scatter(x=b['adc_time_us'][mask],y=b['adc_value'][mask],mode='markers',
                            marker=dict(size=9,color='#7c3aed'),name='ADC (illustrative)' if meta['kind']=='illustrative' else 'larnd-sim ADC before packet serialization'),row=3,col=2)
    if not mask.any(): fig.add_annotation(text='No ADC readings on this pixel',xref='x3 domain',yref='y3 domain',x=.5,y=.5,showarrow=False)
    fig.update_xaxes(title_text='Local time [µs]',row=3,col=2)
    fig.update_xaxes(matches='x3',row=1,col=2);fig.update_xaxes(matches='x3',row=2,col=2)
    fig.update_yaxes(title_text='e− / µs',row=1,col=2)
    fig.update_yaxes(title_text='e−',row=2,col=2);fig.update_yaxes(title_text='ADC code',row=3,col=2)
    fig.update_layout(height=850,template='plotly_white',title=f"{meta['detector']} · pixel {b['pixel_id'][channel]} · {meta['kind']}",
        scene=dict(xaxis_title='x: drift [cm]',yaxis_title='z: transverse [cm]',zaxis_title='y: vertical [cm]',aspectmode='data'),
        legend=dict(orientation='h',y=-.1),margin=dict(t=90,b=100))
    return fig


def attach_event_segments(bundle,input_h5,max_segments=20000,chunk_rows=100000):
    """Attach original edep-HDF5 spatial context for the captured single event.

    Context includes all input segments in that event, not only this call's
    contributing segments. Reading and display are bounded; truncation is logged.
    """
    import h5py
    meta=validate_bundle(bundle)
    events=meta.get('event_ids',[])
    if len(events)!=1: raise ValueError('Need a captured single event ID')
    if min(max_segments,chunk_rows)<1: raise ValueError('Limits must be positive')
    selected=[];count=0;truncated=False
    with h5py.File(input_h5,'r') as f:
        ds=f['segments']
        fields=['event_id','x_start','y_start','z_start','x_end','y_end','z_end']
        for start in range(0,len(ds),chunk_rows):
            a=ds.fields(fields)[start:start+chunk_rows]
            a=a[a['event_id']==events[0]]
            if count+len(a)>max_segments: truncated=True
            a=a[:max_segments-count]
            if len(a):
                xyz=np.stack([np.column_stack([a[f'{axis}_{end}'] for axis in 'xyz'])
                              for end in ['start','end']],axis=1)
                selected.append(xyz);count+=len(a)
            if truncated: break
    if not selected: raise ValueError(f'No segments for event {events[0]} in {input_h5}')
    out=dict(bundle,segments_xyz_cm=np.concatenate(selected))
    meta.update(context_input=str(Path(input_h5).resolve()),context_truncated=truncated,
                context_description='Original edep XYZ cm; whole input event, not per-pixel backtracking')
    out['metadata_json']=np.array(json.dumps(meta));validate_bundle(out)
    return out


def read_channel_packets(path,address,*,data_packet_type,limit=5000,chunk_rows=100000):
    """Read bounded raw /packets for an exact nominal electronics address.

    No network-alias, sync, rollover or absolute-time correction is inferred.
    Raw integer timestamps remain integers until plotting.
    """
    import h5py
    if len(address)!=4 or min(limit,chunk_rows)<1: raise ValueError('Need four address components and positive limits')
    columns=['io_group','io_channel','chip_id','channel_id']
    output=[];n=0
    with h5py.File(path,'r') as f:
        ds=f['packets']
        for start in range(0,len(ds),chunk_rows):
            a=ds[start:start+chunk_rows];keep=a['packet_type']==data_packet_type
            for k,v in zip(columns,address): keep &= a[k]==v
            a=a[keep][:limit-n];output.append(a);n+=len(a)
            if n>=limit: break
        return np.concatenate(output) if output else np.empty(0,dtype=ds.dtype)


def packet_figure(packets,clock_cycle_us=.1):
    """Plot raw ADC words against relative hardware timestamps, separately from captures."""
    import plotly.graph_objects as go
    if not len(packets): raise ValueError('No matching packets')
    if clock_cycle_us<=0: raise ValueError('Clock period must be positive')
    ticks=[int(v) for v in packets['timestamp']]
    if any(b<a for a,b in zip(ticks,ticks[1:])):
        raise ValueError('Timestamps decrease: select a contiguous clock interval or correct rollover/sync explicitly.')
    t=np.array([v-ticks[0] for v in ticks],float)*clock_cycle_us
    fig=go.Figure(go.Scatter(x=t,y=packets['dataword'],mode='markers',
                             text=[str(v) for v in ticks],hovertemplate='relative t=%{x} µs<br>ADC=%{y}<br>raw tick=%{text}<extra></extra>'))
    fig.update_layout(template='plotly_white',title='Serialized packets · no alignment to diagnostic local time inferred',
                      xaxis_title=f'Time relative to first raw tick ({ticks[0]}) [µs]',yaxis_title='ADC dataword')
    return fig
