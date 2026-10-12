"""Read the single-module FSD layouts without importing larnd-sim/CUDA.

Public coordinates are edep-sim-style (x=drift, y=vertical, z=transverse), cm.
larnd-sim swaps x and z internally. Pixel marks are centers, not copper pads.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import itertools

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml


@dataclass
class Geometry:
    name: str
    layout_path: Path
    properties_path: Path
    layout: dict
    properties: dict
    pitch_cm: float
    shape: tuple
    tile_centers: dict
    tile_bounds: dict
    tpc_bounds: dict
    cathodes: dict

    def summary(self):
        return dict(detector=self.name, tiles=len(self.tile_centers),
                    pixels_per_tile=len(self.layout['chip_channel_to_position']),
                    total_pixel_positions=len(self.tile_centers)*len(self.layout['chip_channel_to_position']),
                    pitch_mm=self.pitch_cm*10, pixel_grid=self.shape,
                    drift_cm=float(np.ptp(next(iter(self.tpc_bounds.values()))[0])),
                    field_kV_cm=self.properties['e_field'],
                    layout_sha256=hashlib.sha256(self.layout_path.read_bytes()).hexdigest())


def load_geometry(repo, config='fsd_cube'):
    """Load fsd/fsd_cube or a matching single-module config with no YAML mutation."""
    root = Path(repo).expanduser().resolve() / 'larndsim'
    configs = yaml.safe_load((root/'config/config.yaml').read_text())
    cfg = configs[config]
    lp = root/'pixel_layouts'/cfg['PIXEL_LAYOUT']
    dp = root/'detector_properties'/cfg['DET_PROPERTIES']
    layout, prop = yaml.safe_load(lp.read_text()), yaml.safe_load(dp.read_text())
    offsets = np.asarray(prop['tpc_offsets'], dtype=float)
    if offsets.shape != (1, 3) or len(prop['module_to_io_groups']) != 1:
        raise ValueError('This viewer supports one module (FSD/FSD Cube), not multi-module layouts.')
    coords = np.asarray(list(layout['chip_channel_to_position'].values()), dtype=int)
    if np.any(coords.min(axis=0) != 0):
        raise ValueError('Expected zero-based pixel indices.')
    shape = tuple((coords.max(axis=0)+1).tolist())
    if len(set(map(tuple, coords))) != len(coords):
        raise ValueError('Duplicate pixel positions in layout.')
    pitch = float(layout['pixel_pitch'])/10
    half = np.array(shape)*pitch/2
    centers = {t: np.asarray(c, float)/10 + offsets[0] for t,c in layout['tile_positions'].items()}
    bounds = {t: np.array([[c[0],c[0]], [c[1]-half[1],c[1]+half[1]],
                           [c[2]-half[0],c[2]+half[0]]]) for t,c in centers.items()}
    drift = layout.get('drift_length')
    drift = float(drift)/10 if drift is not None else np.ptp([c[0] for c in centers.values()])/2
    if drift <= 0:
        raise ValueError('Single-anode layout needs an explicit positive drift_length.')
    tpcs, cathodes = {}, {}
    for tpc in sorted({v[0] for v in layout['tile_indeces'].values()}):
        tiles = [t for t,idx in layout['tile_indeces'].items() if idx[0] == tpc]
        signs = {layout['tile_orientations'][t][0] for t in tiles}
        if len(signs) != 1 or not signs <= {-1,1}:
            raise ValueError('Inconsistent cathode directions.')
        anodes = [centers[t][0] for t in tiles]
        if not np.allclose(anodes, anodes[0]):
            raise ValueError('Anode tiles must share a drift coordinate.')
        cathodes[tpc] = anodes[0] + next(iter(signs))*drift
        b = np.stack([bounds[t] for t in tiles])
        tpcs[tpc] = np.array([sorted([anodes[0],cathodes[tpc]]),
                             [b[:,1,0].min(),b[:,1,1].max()],
                             [b[:,2,0].min(),b[:,2,1].max()]])
    name = 'FSD Cube' if config.startswith('fsd_cube') else 'FSD'
    return Geometry(name,lp,dp,layout,prop,pitch,shape,centers,bounds,tpcs,cathodes)


def tile_pixels(g, tile):
    """Physical pixel centers plus the nominal YAML electronics route.

    pixel_id is the single-module larnd-sim grid ID used by fee.save_packets.
    Nominal routes do not resolve data-run network aliases or disabled channels.
    """
    if tile not in g.tile_centers:
        raise KeyError(f'Tile {tile} not in {g.name}')
    keys = np.asarray(list(g.layout['chip_channel_to_position']),dtype=np.int64)
    xy = np.asarray(list(g.layout['chip_channel_to_position'].values()),dtype=int)
    orient = g.layout['tile_orientations'][tile]
    local = ((xy+0.5)-np.array(g.shape)/2)*g.pitch_cm
    local *= [orient[2],orient[1]]
    center = g.tile_centers[tile]
    xyz = np.column_stack([np.full(len(keys),center[0]),center[1]+local[:,1],center[2]+local[:,0]])
    chips, channels = keys//1000, keys%1000
    routes = np.array([g.layout['tile_chip_to_io'][tile].get(int(c),-1) for c in chips])
    groups = next(iter(g.properties['module_to_io_groups'].values()))
    io_group = np.array([groups[r//1000-1] if r>0 else -1 for r in routes])
    # A tile's location in tile_map supplies its unrotated simulation grid offset.
    locations = [(p,i,j) for p,plane in enumerate(g.properties['tile_map'])
                 for i,col in enumerate(plane) for j,t in enumerate(col) if t == tile]
    if len(locations)!=1:
        raise ValueError('Tile must occur exactly once in tile_map.')
    plane,i,j = locations[0]
    dims = np.array(g.shape)*[len(g.properties['tile_map'][plane]),len(g.properties['tile_map'][plane][0])]
    grid = xy.copy()
    for axis,sign in enumerate([orient[2],orient[1]]):
        if sign<0: grid[:,axis] = g.shape[axis]-1-grid[:,axis]
    grid += np.array([i,j])*g.shape
    pid = grid[:,0]+dims[0]*(grid[:,1]+dims[1]*plane)
    return dict(xyz_cm=xyz, chip_id=chips, channel_id=channels, io_group=io_group,
                io_channel=np.where(routes>0,routes%1000,-1),pixel_id=pid,
                tile_id=np.full(len(keys),tile))


def _box(bounds, name, color, dash='solid'):
    v=np.array(list(itertools.product(*bounds)))
    points=[]
    for i,j in itertools.combinations(range(8),2):
        if (v[i]!=v[j]).sum()==1: points.extend([v[i],v[j],[np.nan]*3])
    a=np.array(points)
    return go.Scatter3d(x=a[:,0],y=a[:,2],z=a[:,1],mode='lines',name=name,
                        line=dict(color=color,width=3,dash=dash),hoverinfo='name')


def _plane(bounds, x, name, color, opacity):
    y,z=bounds[1],bounds[2]
    return go.Mesh3d(x=[x]*4,y=[z[0],z[1],z[1],z[0]],z=[y[0],y[0],y[1],y[1]],
                     i=[0,0],j=[1,2],k=[2,3],name=name,color=color,opacity=opacity,
                     hoverinfo='name',showlegend=False)


def geometry_figure(*geometries, overlay=None, opacity=0.3):
    """Rotatable geometry; y is drawn vertically, axes retain physical names."""
    fig=make_subplots(rows=1,cols=len(geometries),specs=[[{'type':'scene'}]*len(geometries)],
                      subplot_titles=[g.name for g in geometries])
    for col,g in enumerate(geometries,1):
        for tpc,b in g.tpc_bounds.items():
            fig.add_trace(_box(b,f'{g.name}: TPC {tpc}', '#64748b'),row=1,col=col)
            fig.add_trace(_plane(b,g.cathodes[tpc],f'Cathode {tpc}','#f59e0b',opacity),row=1,col=col)
            a=np.mean(b,axis=1); a[0]=g.cathodes[tpc]
            anode=next(g.tile_centers[t][0] for t,idx in g.layout['tile_indeces'].items() if idx[0]==tpc)
            fig.add_trace(go.Cone(x=[a[0]],y=[a[2]],z=[a[1]],u=[anode-a[0]],v=[0],w=[0],
                                 sizemode='absolute',sizeref=10,showscale=False,anchor='tail',
                                 name='Electron drift toward anode',hoverinfo='name',colorscale=[[0,'#2563eb'],[1,'#2563eb']]),row=1,col=col)
        for t,b in g.tile_bounds.items():
            fig.add_trace(_plane(b,b[0,0],f'{g.name}: anode tile {t}','#16a085',opacity),row=1,col=col)
            c=g.tile_centers[t]
            fig.add_trace(go.Scatter3d(x=[c[0]],y=[c[2]],z=[c[1]],mode='text',text=[str(t)],
                                     showlegend=False,hoverinfo='skip'),row=1,col=col)
        if overlay is not None:
            for b,n,c in overlay:
                fig.add_trace(_box(b,n,c,'dash'),row=1,col=col)
    fig.update_scenes(xaxis_title='x: drift [cm]',yaxis_title='z: transverse [cm]',
                      zaxis_title='y: vertical [cm]',aspectmode='data')
    fig.update_layout(height=650,template='plotly_white',title='Simulation readout geometry · tile numbers on anodes',
                      margin=dict(l=0,r=0,b=0,t=75),legend=dict(orientation='h'))
    return fig


def pixel_figure(g, tile=1, chip=None, color_by='chip_id'):
    p=tile_pixels(g,tile)
    keep=np.ones(len(p['chip_id']),dtype=bool) if chip is None else p['chip_id']==chip
    if not keep.any(): raise ValueError(f'No pixels for chip {chip}')
    fields=['tile_id','chip_id','channel_id','io_group','io_channel','pixel_id']
    custom=np.column_stack([p[k][keep] for k in fields])
    fig=go.Figure(go.Scattergl(x=p['xyz_cm'][keep,2],y=p['xyz_cm'][keep,1],mode='markers',
        marker=dict(size=4 if chip is None else 12,color=p[color_by][keep],colorscale='Turbo',
                    colorbar=dict(title=color_by)),customdata=custom,
        hovertemplate='z=%{x:.4f} cm, y=%{y:.4f} cm<br>tile=%{customdata[0]} chip=%{customdata[1]} channel=%{customdata[2]}<br>nominal IO=%{customdata[3]}-%{customdata[4]}<br>simulation pixel ID=%{customdata[5]}<extra></extra>'))
    fig.update_layout(title=f'{g.name} · tile {tile} · pixel centers ({g.pitch_cm*10:g} mm pitch)',
        xaxis_title='z: transverse [cm]',yaxis_title='y: vertical [cm]',height=600,template='plotly_white',
        yaxis=dict(scaleanchor='x',scaleratio=1))
    return fig


def repositioning_boxes(shift_index=0):
    """Exact partition constants from 2026_05_27.skim_more_showers.ipynb.

    These are harvesting windows, not the FSD Cube active-volume definition.
    """
    if not 0<=shift_index<40: raise ValueError('shift_index must be 0..39')
    dx,dy,dz=47.1085,29.96,47.8318
    iz=shift_index%2; iy=(shift_index//2)%10; ix=shift_index//20
    y0=149.7995-dy
    source=np.array([[-ix*dx,(1-ix)*dx],[y0-iy*dy,y0-(iy-1)*dy],[-iz*dz,(1-iz)*dz]])
    shift=np.array([ix*dx,iy*dy,iz*dz])
    return source,source+shift[:,None],shift


def export_html(fig, path):
    """Self-contained HTML; no CDN or Jupyter dependency when opening later."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.write_html(str(path),include_plotlyjs=True)
    return path.resolve()
