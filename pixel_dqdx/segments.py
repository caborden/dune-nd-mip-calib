import numpy as np
import os
from sklearn.decomposition import PCA
import collections
import h5py
bounds = {
    'x': (-46.788, 46.788),  # FSD
    'y': (-148.8, 148.8),
    'z': (-47.616, 47.616)
}
#these are for FSD
'''
tpc_bounds = ({
    'x': (-46.788, -0.635/2),  # TPC 0
    'y': (-148.8, 148.8),
    'z': (-47.616, 47.616)
}, {
    'x': (0.635/2, 46.788),  # TPC 1
    'y': (-148.8, 148.8),
    'z': (-47.616, 47.616)
})
'''
#these are for FSD-cube
tpc_bounds = ({
    'x': (0, 46.788),  # TPC 0
    'y': (0, 29.388),
    'z': (0, 47.616)
}, {
    'x': (0.635/2, 46.788),  # TPC 1
    'y': (-148.8, 148.8),
    'z': (-47.616, 47.616)
})

unit_vectors = {'x': np.array([1, 0, 0]),
                'y': np.array([0, 1, 0]),
                'z': np.array([0, 0, 1]),
                }

bins = {'x': np.linspace(-50, 50, 51),
        'y': np.linspace(-150, 150, 151),
        'z': np.linspace(-50, 50, 51)
       }
histkwargs = {'bins': (bins['x'], bins['y'], bins['z'])}
xBinCenters = 0.5*(bins['x'][1:] + bins['x'][:-1])
yBinCenters = 0.5*(bins['y'][1:] + bins['y'][:-1])
zBinCenters = 0.5*(bins['z'][1:] + bins['z'][:-1])
bin_width = 0.5*(bins['x'][1] - bins['x'][0])
bin_width2 = 0.5*(bins['y'][1] - bins['y'][0])

def init_hdf5_file(filename):
    with h5py.File(filename, "w") as f:
        for tpc in [0, 1]:
            grp = f.create_group(f"tpc{tpc}/segments")
            dt = h5py.string_dtype(encoding='utf-8')

            maxshape = (None,)  # allow unlimited growth

            grp.create_dataset("dq", shape=(0,), maxshape=(None,), dtype="f8", chunks=True)
            grp.create_dataset("dx", shape=(0,), maxshape=(None,), dtype="f8", chunks=True)
            grp.create_dataset("nhits", shape=(0,), maxshape=(None,), dtype="i4", chunks=True)
            grp.create_dataset("x", shape=(0,), maxshape=(None,), dtype="f4", chunks=True)
            grp.create_dataset("y", shape=(0,), maxshape=(None,), dtype="f4", chunks=True)
            grp.create_dataset("z", shape=(0,), maxshape=(None,), dtype="f4", chunks=True)
            grp.create_dataset("t_drift", shape=(0,), maxshape=(None,), dtype="f4", chunks=True)
            grp.create_dataset("theta", shape=(0,), maxshape=(None,), dtype="f4", chunks=True)
            grp.create_dataset("phi", shape=(0,), maxshape=(None,), dtype="f4", chunks=True)
            grp.create_dataset("track_id", shape=(0,), maxshape=(None,), dtype=dt, chunks=True)
            grp.create_dataset("is_through", shape=(0,), maxshape=(None,), dtype="i1", chunks=True)

def append_segs(filename, tpc_id, seg_dict):
    """
    seg_dict must contain numpy arrays of equal length:
        dq, dx, nhits, x_center, time_center,
        theta, phi, track_id, through_flag
    """

    with h5py.File(filename, "a") as f:
        grp = f[f"tpc{tpc_id}/segments"]

        n_new = len(seg_dict["dq"])
        if n_new == 0:
            return

        for key, arr in seg_dict.items():
            dset = grp[key]
            old_size = dset.shape[0]
            new_size = old_size + n_new
            dset.resize((new_size,))
            dset[old_size:new_size] = arr

def is_through_going(track_hits, bounds, margin=1.0):
    mins = track_hits.min(axis=0)
    maxs = track_hits.max(axis=0)


    touches = 0
    for i, axis in enumerate(['x', 'y', 'z']):
        low, high = bounds[axis]
        if mins[i] <= low + margin:
            touches += 1
        if maxs[i] >= high - margin:
            touches += 1

    return touches >= 2

def fiducialize_hits(track_hits, bounds, margin=1.0):
    mask = np.ones(len(track_hits), dtype=bool)

    for i, axis in enumerate(['x', 'y', 'z']):
        low, high = bounds[axis]
        mask &= (track_hits[:, i] >= low + margin)
        mask &= (track_hits[:, i] <= high - margin)

    return track_hits[mask], mask

def group_by_track(labels, reco, q, pca_dir, hid, iog, t_drift):
    track_hits = collections.defaultdict(list)
    track_qs   = collections.defaultdict(list)
    track_ios  = collections.defaultdict(list)
    track_times = collections.defaultdict(list)
    track_dirs = {}

    # group hits and charges by track label (hid is per-hit label!)
    for hit, charge, track_label, io, times in zip(reco, q, hid, iog, t_drift):
        track_hits[track_label].append(hit)
        track_qs[track_label].append(charge)
        track_ios[track_label].append(io)
        track_times[track_label].append(times)

    # attach direction per track from labels list (1 per track)
    for track_label, direction in zip(labels, pca_dir):
        track_dirs[track_label] = direction

    return track_hits, track_qs, track_dirs, track_ios, track_times

def quantize_and_reconstruct_charge_v3(float_datawords, target_lsb):
    # from fsd-cube.yaml
    v_cm = 0.0          # mV
    v_ped = 980.0       # mV
    gain_ke = 3.4       # mV / ke- (3.4e-3 mV/e- * 1000)

    #  derived from yaml
    # (1734.375 - 0.0) * 2 / 1023
    nominal_lsb = 3.39076  # mV per nominal float dataword

    # unrounded float datawords to analog voltage
    voltage_mv = (float_datawords * nominal_lsb) + v_cm - v_ped

    # quantize using target LSB
    quantized_datawords = np.floor((voltage_mv - v_cm + v_ped) / target_lsb)

    # Clip to bit depth
    quantized_datawords = np.clip(quantized_datawords, 0, 1023)

    # Back to charge (ke-)
    reco_voltage = (quantized_datawords * target_lsb) + v_cm - v_ped
    reco_charge_ke = reco_voltage / gain_ke

    return reco_charge_ke

def segment_track_hits(hits_track, q_track, times_track, segment_length=3.0,
                       step_size=3.0, track_id=0):
    """Upstream physical-PCA segmentation without geometry cuts or file I/O.

    Retains the upstream inclusive window endpoints and dx = projected span +
    median local spacing. Geometry selection belongs to the caller. Final
    incomplete windows are discarded, and dq>0/dx>0 are required as upstream.
    """
    if segment_length <= 0 or step_size <= 0:
        raise ValueError('Segment length and step must be positive')
    hits_track = np.asarray(hits_track, dtype=float)
    q_track, times_track = np.asarray(q_track), np.asarray(times_track)
    if hits_track.shape != (len(q_track), 3) or len(times_track) != len(q_track):
        raise ValueError('Hit, charge, and drift-time shapes do not align')
    result = collections.defaultdict(list)
    if len(hits_track) < 3:
        return result
    if not (np.isfinite(hits_track).all() and np.isfinite(q_track).all() and np.isfinite(times_track).all()):
        raise ValueError('Segmentation requires finite coordinates, charge, and times')
    pca = PCA(n_components=3)
    #X_train = hits_track
    #X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    pca.fit(hits_track)
    dir_vec = pca.components_[0]
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    center = np.mean(hits_track, axis=0)
    proj = (hits_track - center) @ dir_vec
    sort_idx = np.argsort(proj)

    times_sorted = times_track[sort_idx]
    hits_sorted = hits_track[sort_idx]
    q_sorted = q_track[sort_idx]

    if dir_vec[1] > 0:
        dir_vec = -dir_vec # flip vector so it always points down
    cos_theta = dir_vec @ unit_vectors['y']
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    phi_deg = np.degrees(np.arctan2(dir_vec[0], dir_vec[2])) % 360

    #conditions for drift analysis
    theta_cond = (175 < theta_deg < 180 or 0 < theta_deg < 5)
    phi_cond = (phi_deg < 5 or phi_deg > 355 or 175 < phi_deg < 185)



    proj = proj[sort_idx]
    total_length = proj[-1] - proj[0]

    window_start = proj[0]
    while window_start + segment_length <= proj[-1]:
        # Find indices within the current segment
        lo = np.searchsorted(proj, window_start, side="left")
        hi = np.searchsorted(proj, window_start + segment_length, side="right")
        nhits = hi-lo

        # Check if segment is valid
        if nhits >= 2: # At least 2 hits in the segment
            # Get hits and charges for the current segment
            seg_times = times_sorted[lo:hi]
            seg_hits = hits_sorted[lo:hi]
            seg_qs = q_sorted[lo:hi]
            # Compute dx
            pitch = np.median(np.diff(proj[lo:hi]))  # local hit spacing
            dx_segment = abs(proj[hi-1] - proj[lo]) + pitch # add one pitch for the two half-spacings

            # Compute dq
            dq_segment = np.sum(seg_qs)

            #compute median positions
            median_x = np.median(seg_hits[:, 0])
            median_y = np.median(seg_hits[:, 1])
            median_z = np.median(seg_hits[:, 2])
            median_t = np.median(seg_times)
            #check for segments in cathode.... hard coding it for now but should put in bounds!!!
            '''
            is_within_range = (seg_hits[:,0] >= -0.635/2) & (seg_hits[:,0] <= 0.635/2)
            if np.any(is_within_range):
                window_start += step_size
                continue
            '''
            if dx_segment > 0 and dq_segment > 0:
                result["dq"].append(dq_segment)
                result["dx"].append(dx_segment)
                result["nhits"].append(nhits)
                result["x"].append(median_x)
                result["y"].append(median_y)
                result["z"].append(median_z)
                result["t_drift"].append(median_t)
                result["theta"].append(theta_deg)
                result["phi"].append(phi_deg)
                result["track_id"].append(str(track_id))

        # Advance the window by the step size
        window_start += step_size

    return {key: np.asarray(value) for key, value in result.items()}

def run_dqdx_tpc(segment_length, step_size, d_face, reco, q, labels, hid, pca_dir, iog, t_drift, pickle_name):
    # d_face cuts hits a distance d (cm) from detector edge
    # if step_size = segment_length, trackis split into equal, non-overlapping chunks (nominal)
    n_selec = 0
    n_tracks = 0
    #y_fudge = 119.226 #for fsd-cube
    y_fudge = 0.0 #for fsd
    # Initialize angle-binned dq/dx storage
    seg_data = {
    0: collections.defaultdict(list),
    1: collections.defaultdict(list)
    }

    # Mapping io_groups to TPC indices (0 or 1)
    #tpc_map = {i: 1 if i <= 2 else 0 for i in range(1, 5)}
    tpc_map = {i: 1 if i >= 2 else 0 for i in range(1, 5)}

    #initialize output file
    init_hdf5_file(f"{pickle_name}.dqdx.hdf5")

    track_hits, track_qs, track_dirs, track_ios, track_times = group_by_track(labels, reco, q, pca_dir, hid, iog, t_drift)
    for track in labels:
        hits_all = np.array(track_hits[track])
        q_all = np.array(track_qs[track])
        ios_all = np.array((track_ios[track]))
        times_all = np.array(track_times[track])
        #scale y for fsd-cube
        hits_all[:,1] = hits_all[:,1] - y_fudge
        # Process each TPC separately
        for tpc_id in [0, 1]:
            # Mask hits belonging to the current TPC
            tpc_mask = np.array([tpc_map[io] == tpc_id for io in ios_all])
            hits_track = hits_all[tpc_mask]
            q_track = q_all[tpc_mask]
            times_track = times_all[tpc_mask]
            if len(hits_track) < 3:
                continue

            if is_through_going(hits_track, tpc_bounds[tpc_id], margin=d_face):
                hits_track, tg_mask = fiducialize_hits(hits_track, tpc_bounds[tpc_id], margin=d_face)
                q_track = q_track[tg_mask]
                times_track = times_track[tg_mask]
                if len(hits_track) < 3:
                    continue
            else:
                continue

            n_tracks += 1
            n_selec += 1
            result = segment_track_hits(hits_track, q_track, times_track,
                                        segment_length, step_size, track)
            for key, values in result.items():
                seg_data[tpc_id][key].extend(values)

    # Save to hdf5 (Iterate through TPCs)
    for tpc_id in [0, 1]:
        if len(seg_data[tpc_id]["dq"]) == 0:
            continue

        seg_dict = {k: np.array(v) for k, v in seg_data[tpc_id].items()}
        append_segs(f"{pickle_name}.dqdx.hdf5", tpc_id, seg_dict)

    return None

def load_and_merge(outDir, tpc_id):
    merged = {}
    files = os.listdir(outDir)
    for fpath in files:
        if not fpath.endswith('.dqdx.hdf5'):
            continue
        with h5py.File(outDir+fpath, "r") as f:
            grp = f[f"tpc{tpc_id}/segments"]
            #print(fpath)
            for key in grp.keys():
                data = grp[key][:]
                merged.setdefault(key, []).append(data)

    for key in merged:
        merged[key] = np.concatenate(merged[key])

    return merged

def coplanar(data):
    coplanar_bool = (
    (data['phi'] < 5) |
    (data['phi'] > 355) |
    ((data['phi'] > 175) & (data['phi'] < 185))
)
    return coplanar_bool

def run_segmentation(file_path, out_name):
    # file_path should be path to hdf5 track selection output file
    # .dqdx.hdf5 will be added at end of out_name
    # in run_dqdx_tpc so chill.
    with h5py.File(file_path, "r") as fin:
            labels_raw = np.char.decode(fin['tracks/label'][()], 'utf-8')
            reco_raw   = fin['hits/reco'][()]
            q_raw      = fin['hits/charge'][()]
            #q_adc = quantize_and_reconstruct_charge_v3(q_raw, lsb)
            axis       = fin['tracks/PCA_dir'][()]
            t_drift    = fin['hits/t_drift'][()]
            iog        = np.array(fin['hits/io_group']).astype(int)
            hid        = np.char.decode(fin['hits/track_id'][()], 'utf-8')

            run_dqdx_tpc(3, 3, 1, reco_raw, q_raw, labels_raw, hid, axis, iog, t_drift, out_name)
            print('Segmentation for '+str(file_path)+' finished.')
