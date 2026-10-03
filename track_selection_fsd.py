from efield import PCAs, length_track, str2bool
import numpy as np
import h5flow
import argparse
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import sys
from track_output_hdf5 import TrackOutputWriter

# Selection parameters
pixel_pitch = 0.372
min_size = 90
max_rel_spread = 0.07
max_axis_dist = 3*pixel_pitch
#max_break_dist = 15*pixel_pitch
max_break_dist = 25*pixel_pitch
d = 2 #Distance away from a TPC face
v = 0.973 #Explained variance minimum
b = 55 #cm

# Initialize DBSCAN
dbscan = DBSCAN(eps=max_break_dist, metric='euclidean', min_samples=1)

# Initialize PCA
pca = PCA(3)

def run(input_file, output_file, is2x2):
    settings = dict(pixel_pitch_cm=pixel_pitch, min_size=min_size,
                    max_relative_spread=max_rel_spread, max_axis_distance_cm=max_axis_dist,
                    dbscan_eps_cm=max_break_dist, dbscan_min_samples=1,
                    boundary_margin_cm=d, min_quality_standardized=v,
                    min_length_cm=b, is2x2=is2x2)
    f_manager = h5flow.data.H5FlowDataManager(input_file, 'r', mpi=False)
    try:
        with TrackOutputWriter(output_file, input_file, settings) as writer:
            next_track_id = 0
            events = f_manager["charge/events/data"]
            length_of_file = len(f_manager["charge/events/data"])
            #module_bounds = f_manager['geometry_info'].attrs['module_RO_bounds']
            lar_detector_bounds = f_manager['geometry_info'].attrs['lar_detector_bounds']
            print("Bounds!")
            print(is2x2)
            if(is2x2):
                x_boundaries = np.array([-63.931, -3.069, 3.069, 63.931])
                y_boundaries = np.array([-42 - 19.8543 - 0.027712, -42 + 103.8543 + 0.027712]) #0.027 term is to account for module 2
                z_boundaries = np.array([-64.3163,  -2.6837, 2.6837, 64.3163 + 0.027712]) #0.027 term is to account for module 2
            else:
                x_boundaries = lar_detector_bounds[:,0]
                y_boundaries = lar_detector_bounds[:,1]
                z_boundaries = lar_detector_bounds[:,2]
            min_boundaries = np.array([x_boundaries[0], y_boundaries[0], z_boundaries[0]])
            max_boundaries = np.array([x_boundaries[-1], y_boundaries[-1], z_boundaries[-1]])
            print(y_boundaries)
            print(min_boundaries)
            print(max_boundaries)
            ev = f_manager["charge/events/data"].shape[0]
            #ev = 5000
            len_label = 0
            print("Looping through events")
            for i_evt in range(ev):
                selected_tracks = []
                step = i_evt/10
                progress_str = f"Event {i_evt} out of {ev}."
                sys.stdout.write('\r' + progress_str)
                sys.stdout.flush()

                #load dataset, checking for emptiness or bad references
                try:
                    pev = f_manager["charge/events", "charge/calib_prompt_hits", i_evt]
                    if len(pev) == 0:
                        print(f"Skipping event {i_evt} (no hits)")
                        continue
                except IndexError:
                    print(f"Skipping event {i_evt} (invalid reference)")
                    continue
                trigs = f_manager["charge/events", "charge/ext_trigs", i_evt]
                trigio = trigs['iogroup'].data
                PromptHits_ev = pev[0]
                selected_track_id = np.full(len(PromptHits_ev), -1, dtype=np.int64)
                projected_xyz = np.full((len(PromptHits_ev), 3), np.nan)
                hits = np.column_stack([
                    PromptHits_ev['x'].data,
                    PromptHits_ev['y'].data,
                    PromptHits_ev['z'].data
                ])
                io = PromptHits_ev['io_group'].data
                qin = PromptHits_ev['Q'].data
                t_drift = PromptHits_ev['t_drift'].data
                valid_mask = ~np.isnan(hits).any(axis=1)
                event_hit_index = np.flatnonzero(valid_mask)
                hits = hits[valid_mask]
                qin = qin[valid_mask]
                io = io[valid_mask]
                t_drift = t_drift[valid_mask]

                #some conditionals to account for different detectors
                if is2x2:
                    if not np.any(np.isin(trigio,6)):
                        continue # for 2x2 data, skip event if there isn't a light trigger

                if len(hits) == 0:
                    continue  # no possible track; avoids DBSCAN on an empty array

                # Cluster hits in the image with DBSCAN
                clust_labels = dbscan.fit(hits).labels_
                len_label += len(clust_labels)

                # Loop over the clusters and apply some quality cuts
                for c in np.unique(clust_labels):
                    # Restrict points to the relevant cluster
                    index = np.where(clust_labels == c)[0]
                    cluster_event_index = event_hit_index[index]
                    points_c = hits[index]
                    q = qin[index]
                    t = t_drift[index]
                    iog = io[index]
                    if len(index) < min_size:
                        continue

                    # If the relative spread of the points w.r.t. the main axis is too large, skip
                    decomp = pca.fit(points_c)
                    axis = decomp.components_[0]
                    var = decomp.explained_variance_
                    rel_spread = np.sqrt((var[1] + var[2])/var[0])
                    if rel_spread > max_rel_spread:
                        continue

                    # Project all points on the principal axis, filter out those too far from the main axis
                    cent = np.mean(points_c, axis=0)
                    dists = np.linalg.norm(np.cross(points_c - cent, axis), axis=1)
                    axis_keep = dists < max_axis_dist
                    cluster_event_index = cluster_event_index[axis_keep]
                    points_c = points_c[axis_keep]
                    q = q[dists < max_axis_dist]
                    t = t[dists < max_axis_dist]
                    iog = iog[dists < max_axis_dist]
                    if len(points_c) < min_size:
                        continue

                    # PCA fit of line
                    a, p, output = PCAs(points_c)
                    l, start, end = length_track(points_c)

                    #if satisfies all criteria now, we output
                    if np.logical_and(a > v, l > b):
                        # Output-only physical fit: never used to change acceptance.
                        physical = PCA(3).fit(points_c)
                        physical_var = physical.explained_variance_
                        anode = (min_boundaries[0]-d < np.min(points_c[:,0]) < min_boundaries[0]+d
                                 and max_boundaries[0]-d < np.max(points_c[:,0]) < max_boundaries[0]+d)
                        selected_track_id[cluster_event_index] = next_track_id
                        projected_xyz[cluster_event_index] = np.asarray(output['true'])
                        selected_tracks.append(dict(
                            track_id=next_track_id, source_event_row=i_evt,
                            length_cm=l, start=start, end=end,
                            initial_center_cm=cent, initial_axis_physical=axis,
                            initial_quality_physical=decomp.explained_variance_ratio_[0],
                            initial_relative_spread=rel_spread,
                            pca_dir_standardized=output['v_dir'],
                            pca_quality_standardized=float(np.asarray(a).item()),
                            pca_variance_standardized=float(np.asarray(p).item()),
                            pca_center_physical_cm=physical.mean_,
                            pca_dir_physical=physical.components_[0],
                            pca_quality_physical=physical.explained_variance_ratio_[0],
                            pca_variance_physical_cm2=physical_var,
                            relative_spread_physical=np.sqrt((physical_var[1]+physical_var[2])/physical_var[0]),
                            n_cluster_hits=len(index), n_selected_hits=len(points_c), a2a=anode,
                        ))
                        next_track_id += 1

                if selected_tracks:
                    writer.append_event(i_evt, PromptHits_ev, selected_track_id, projected_xyz, selected_tracks)
            print('Done!')
    finally:
        close = getattr(f_manager, "close", None)
        if callable(close):
            close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Legacy track selection with charge-splitting HDF5 output.')
    parser.add_argument('input_file', type=str, help='input FLOW filename')
    parser.add_argument('output_file', type=str, help='new output HDF5 filename (must not exist)')
    parser.add_argument('is2x2', type=str2bool, help='2x2 (true) or FSD (false)')
    args = parser.parse_args()
    run(args.input_file, args.output_file, args.is2x2)
