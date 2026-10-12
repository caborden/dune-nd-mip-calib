"""Opt-in diagnostic wrapper around an installed larnd-sim driver.

Run in the larnd-sim CUDA environment, not the lightweight notebook environment.
No source files are changed. This adapter targets do_digitize_and_update as found
in local larnd-sim 6106b83; incompatible drivers fail explicitly.
"""
import argparse
import functools
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def host(a):
    return a.get() if hasattr(a,'get') else np.asarray(a)


def selected_rows(ids,requested=None,max_channels=8):
    ids=host(ids).astype(np.int64)
    if requested is None: return np.arange(min(max_channels,len(ids)))
    return np.flatnonzero(np.isin(ids,requested))[:max_channels]


def pixel_positions(detector,ids):
    """Map runtime grid IDs back to physical tile centers in edep XYZ, cm.

    A single module is required so module-local IDs cannot be silently conflated.
    Tile geometry, not an assumed contiguous plane, supplies physical positions.
    """
    nx,ny=map(int,detector.N_PIXELS)
    tx,ty=map(int,detector.N_PIXELS_PER_TILE)
    result=[];addresses=[]
    groups=next(iter(detector.MODULE_TO_IO_GROUPS.values()))
    for pid in ids:
        ix=int(pid)%nx;iy=(int(pid)//nx)%ny;plane=int(pid)//(nx*ny)
        tile=detector.TILE_MAP[plane][ix//tx][iy//ty]
        center=np.asarray(detector.TILE_POSITIONS[tile],float)/10
        center+=np.asarray(detector.TPC_OFFSETS[0])[[2,1,0]]
        xyz=center+np.array([0,((iy%ty+.5)-ty/2)*detector.PIXEL_PITCH,
                              ((ix%tx+.5)-tx/2)*detector.PIXEL_PITCH])
        axes=detector.TILE_ORIENTATIONS[tile]
        gx=ix%tx if axes[2]>0 else tx-1-ix%tx
        gy=iy%ty if axes[1]>0 else ty-1-iy%ty
        chip,channel=detector.PIXEL_CONNECTION_DICT[(gx,gy)]
        route=detector.TILE_CHIP_TO_IO[tile][chip]
        addresses.append([groups[route//1000-1],route%1000,chip,channel])
        result.append(xyz)
    return np.asarray(result),np.asarray(addresses,dtype=np.int64)


def make_capture_wrapper(original,detector,unit_charge,output_dir,metadata,*,
                         requested=None,max_channels=8,max_batches=4,max_samples=2000000,
                         event_id=None):
    """Capture selected rows without modifying inputs, results, or RNG draws."""
    signature=inspect.signature(original)
    required={'results_acc','unique_eventIDs','unique_pix','pixels_signals','max_signal_time'}
    if not required<=set(signature.parameters):
        raise RuntimeError('Unsupported larnd-sim digitization interface; adapter needs updating.')
    if min(max_channels,max_batches,max_samples)<1: raise ValueError('Capture limits must be positive')
    out=Path(output_dir)
    state={'calls':0,'saved':0,'files':[]}

    @functools.wraps(original)
    def wrapped(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
        call=state['calls'];state['calls']+=1
        events=host(v['unique_eventIDs']).astype(np.int64)
        wanted=state['saved']<max_batches and (event_id is None or event_id in events)
        rows=selected_rows(v['unique_pix'],requested,max_channels) if wanted else np.array([],dtype=int)
        current=None
        if len(rows):
            if len(events)!=1:
                raise RuntimeError('Capture requires exactly one event per digitization call; reduce event batching.')
            if len(detector.MODULE_TO_IO_GROUPS)!=1:
                raise RuntimeError('Capture currently supports a single module (FSD/FSD Cube).')
            count=len(rows)*v['pixels_signals'].shape[1]
            if count>max_samples:
                raise RuntimeError(f'Capture would copy {count} samples; raise --max-samples or select fewer pixels.')
            # Copies only selected rows; no full GPU current tensor transfer.
            current=host(v['pixels_signals'][rows]).copy()/unit_charge
            ids=host(v['unique_pix'][rows]).astype(np.int64)
            xyz,addresses=pixel_positions(detector,ids)
        result=original(*args,**kwargs)
        if current is not None:
            from visualizations.signals import save_bundle
            acc=v['results_acc']
            adc=host(acc['adc_tot'][-1][rows]).copy()
            times=host(acc['adc_tot_ticks'][-1][rows]).copy()
            channel,slot=np.nonzero(adc>0)  # same data-reading condition as fee.save_packets
            params={k:float(getattr(detector,k)) for k in ['TIME_SAMPLING','CLOCK_CYCLE','PERIODIC_RESET_CYCLES',
                'ADC_HOLD_DELAY','ADC_BUSY_DELAY','RESET_CYCLES','GAIN','V_PEDESTAL','V_REF','V_CM',
                'ADC_COUNTS','ADC_SCALE_FACTOR','ADC_OFFSET','BUFFER_RISETIME','DISCRIMINATION_THRESHOLD',
                'LARPIX_VERSION','PACKET_TYPE']}
            meta=dict(metadata,schema_version=1,kind='larnd-sim-capture',call_index=call,
                event_ids=events.tolist(),electronics=params,
                time_reference='Digitization-call local microseconds; before event offset, clock quantization and rollover.',
                description='Actual current entering electronics and ADC output; no CSA/reset history captured.',
                selection='explicit pixel IDs' if requested is not None else 'first channels in this call; not classified by physics')
            b=dict(time_us=np.arange(current.shape[1])*detector.TIME_SAMPLING,current_e_per_us=current,
                pixel_id=ids,pixel_xyz_cm=xyz,addresses=addresses,
                adc_channel_index=channel.astype(np.int64),adc_time_us=times[channel,slot],
                adc_value=adc[channel,slot],metadata_json=np.array(json.dumps(meta)))
            path=out/f'trace_{call:05d}_event_{int(events[0])}.npz'
            save_bundle(path,b)
            state['saved']+=1;state['files'].append(str(path))
            print(f'[visualizations] Captured {len(rows)} pixels to {path}')
        return result
    wrapped.capture_state=state
    return wrapped


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--larnd-repo',type=Path,required=True)
    parser.add_argument('--trace-dir',type=Path,required=True)
    parser.add_argument('--detector-name',default='FSD Cube')
    parser.add_argument('--pixel-ids',help='Comma-separated simulation grid IDs (not electronics addresses)')
    parser.add_argument('--event-id',type=int)
    parser.add_argument('--max-channels',type=int,default=8)
    parser.add_argument('--max-batches',type=int,default=4)
    parser.add_argument('--max-samples',type=int,default=2000000)
    parser.add_argument('simulation_args',nargs=argparse.REMAINDER,help='After --, normal simulate_pixels.py arguments')
    args=parser.parse_args()
    if min(args.max_channels,args.max_batches,args.max_samples)<1: parser.error('Capture limits must be positive')
    argv=args.simulation_args
    if argv and argv[0]=='--': argv=argv[1:]
    if not argv: parser.error('Supply simulate_pixels.py arguments after --')
    repo=args.larnd_repo.expanduser().resolve();driver=repo/'cli/simulate_pixels.py'
    if not driver.is_file(): parser.error(f'Driver not found: {driver}')
    args.trace_dir=args.trace_dir.expanduser().resolve()
    if args.trace_dir.exists(): parser.error('Use a new trace directory to keep captures from different runs separate')
    args.trace_dir.mkdir(parents=True)
    sys.path.insert(0,str(repo));sys.path.insert(0,str(driver.parent))
    spec=importlib.util.spec_from_file_location('fsd_capture_driver',driver)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from larndsim.consts import detector
    from larndsim.consts.units import e
    if not hasattr(module,'do_digitize_and_update'): raise RuntimeError('Driver has no do_digitize_and_update hook')
    git=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],text=True,capture_output=True)
    requested=None if args.pixel_ids is None else np.array([int(x) for x in args.pixel_ids.split(',')])
    meta=dict(detector=args.detector_name,driver=str(driver),driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
              larnd_commit=git.stdout.strip(),command=argv,
              fee_sha256=hashlib.sha256((repo/'larndsim/fee.py').read_bytes()).hexdigest())
    wrapper=make_capture_wrapper(module.do_digitize_and_update,detector,e,args.trace_dir,meta,
        requested=requested,max_channels=args.max_channels,max_batches=args.max_batches,
        max_samples=args.max_samples,event_id=args.event_id)
    module.do_digitize_and_update=wrapper
    import fire
    try:
        fire.Fire(module.run_simulation,command=argv)
    finally:
        (args.trace_dir/'capture_manifest.json').write_text(json.dumps(dict(meta,**wrapper.capture_state),indent=2)+'\n')
    if not wrapper.capture_state['saved']:
        raise RuntimeError('Simulation returned but no matching traces were captured; check pixel/event selection.')


if __name__=='__main__': main()
