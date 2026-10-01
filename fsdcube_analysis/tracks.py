"""Exploratory straight-track finding and conservative same-pixel recovery."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from .charge_splitting import PIXEL_KEY


@dataclass(frozen=True)
class TrackConfig:
    # Starting values for investigation, not validated FSD-cube physics cuts.
    eps_cm: float = 3.0
    min_pixels: int = 20
    min_length_cm: float = 10.0
    max_relative_spread: float = 0.15
    axis_distance_cm: float = 1.116
    recovery_us: float = 10.0
    recovery_distance_cm: float = 1.116
    endpoint_margin_cm: float = 1.116

    def __post_init__(self):
        if self.min_pixels < 3:
            raise ValueError("min_pixels must be at least 3")
        if any(not np.isfinite(v) or v <= 0 for k, v in asdict(self).items()
               if k != "min_pixels"):
            raise ValueError("Track scales/cuts must be finite and positive")


def fit_line(points):
    center = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points - center, full_matrices=False)
    axis = vectors[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis  # deterministic sign only, not muon direction
    projection = (points - center) @ axis
    residual = np.linalg.norm(np.cross(points - center, axis), axis=1)
    spread = np.linalg.norm(singular[1:]) / singular[0] if singular[0] > 0 else np.inf
    return center, axis, projection, residual, spread


def associate_tracks(hits, config=TrackConfig()):
    """Single-event input. Fit one earliest usable representative per pixel.

    Preserve all hits. Recover only on retained seed pixels, within a per-pixel
    time window, axis distance, and endpoint margin. Multi-track matches receive
    track_id=-2; unassociated hits receive -1. No detector-boundary cuts.
    """
    if len(hits[["file_id", "event_row"]].drop_duplicates()) > 1:
        raise ValueError("associate_tracks takes one event at a time")
    out = hits.copy().reset_index(drop=True)
    out["track_id"] = -1
    out["association_count"] = 0
    out["association_role"] = "unassociated"
    out["event_pixel_nhits"] = out.groupby(PIXEL_KEY)["hit_row"].transform("size")
    # Exclude entire inconsistent pixel groups from seeding/recovery.
    span = out.groupby(PIXEL_KEY)[["y", "z"]].transform("max") - out.groupby(PIXEL_KEY)[["y", "z"]].transform("min")
    usable = out["hit_quality_ok"] & (span <= 1e-4).all(axis=1)
    reps = out.loc[usable].sort_values(["t_drift", "hit_row"]).drop_duplicates(PIXEL_KEY)
    table_columns = ["file_id", "event_row", "track_id", "accepted", "reason", "n_seed_pixels", "length_cm"]
    if len(reps) < 3:
        return out, pd.DataFrame(columns=table_columns)
    # Avoid connecting distinct readout groups even if reconstructed coordinates overlap.
    records, choices, models = [], [[] for _ in range(len(out))], []
    for _, io_reps in reps.groupby("io_group"):
        labels = DBSCAN(eps=config.eps_cm, min_samples=1).fit_predict(io_reps[["x", "y", "z"]])
        for label in np.unique(labels):
            seed = io_reps.loc[labels == label]
            track_id = len(records)
            record = dict(file_id=int(out["file_id"].iloc[0]), event_row=int(out["event_row"].iloc[0]),
                          track_id=track_id, n_cluster_pixels=len(seed), accepted=False,
                          reason="min_pixels", n_seed_pixels=len(seed), length_cm=np.nan)
            records.append(record)
            if len(seed) < config.min_pixels:
                continue
            points = seed[["x", "y", "z"]].to_numpy()
            _, _, _, residual, _ = fit_line(points)
            seed = seed.loc[residual <= config.axis_distance_cm]
            record["n_seed_pixels"] = len(seed)
            if len(seed) < config.min_pixels:
                record["reason"] = "min_pixels_after_trim"
                continue
            center, axis, proj, residual, spread = fit_line(seed[["x", "y", "z"]].to_numpy())
            length = float(np.ptp(proj))
            record.update(length_cm=length, relative_spread=float(spread),
                          t0_type=int(out["t0_type"].iloc[0]))
            for name, vector in [("center", center), ("axis", axis)]:
                record.update({f"{name}_{d}": float(v) for d, v in zip("xyz", vector)})
            if spread > config.max_relative_spread:
                record["reason"] = "relative_spread"
                continue
            if length < config.min_length_cm:
                record["reason"] = "min_length"
                continue
            record.update(accepted=True, reason="accepted")
            models.append((track_id, int(seed["io_group"].iloc[0]), center, axis,
                           proj.min(), proj.max(), seed["t_drift_us"].min(), seed["t_drift_us"].max()))
            seed_rows = set(seed["hit_row"])
            # Join on exact canonical key, never on float coordinates.
            seed_time = seed[PIXEL_KEY + ["t_drift_us"]].rename(columns={"t_drift_us": "seed_time_us"})
            candidates = out.reset_index(names="output_row").merge(seed_time, on=PIXEL_KEY)
            xyz = candidates[["x", "y", "z"]].to_numpy()
            s = (xyz - center) @ axis
            distance = np.linalg.norm(np.cross(xyz - center, axis), axis=1)
            dt = np.abs(candidates["t_drift_us"] - candidates["seed_time_us"])
            recover = (candidates["hit_quality_ok"] & (dt <= config.recovery_us)
                       & (distance <= config.recovery_distance_cm)
                       & (s >= proj.min() - config.endpoint_margin_cm)
                       & (s <= proj.max() + config.endpoint_margin_cm))
            for row in candidates.loc[recover].itertuples():
                choices[row.output_row].append((track_id, "seed" if row.hit_row in seed_rows else "recovered"))
    # Ambiguity veto: a hit already associated by pixel may also lie near another
    # accepted line. Do not invent a second pixel association; flag the conflict.
    xyz = out[["x", "y", "z"]].to_numpy()
    for track_id, io_group, center, axis, low, high, t_low, t_high in models:
        s = (xyz - center) @ axis
        distance = np.linalg.norm(np.cross(xyz - center, axis), axis=1)
        plausible = ((out["io_group"] == io_group)
                     & (distance <= config.recovery_distance_cm)
                     & (s >= low - config.endpoint_margin_cm)
                     & (s <= high + config.endpoint_margin_cm)
                     & out["t_drift_us"].between(t_low - config.recovery_us, t_high + config.recovery_us))
        for i in np.flatnonzero(plausible):
            if choices[i] and all(t != track_id for t, _ in choices[i]):
                choices[i].append((track_id, "ambiguous"))
    for i, matches in enumerate(choices):
        out.at[i, "association_count"] = len(matches)
        if len(matches) == 1:
            out.at[i, "track_id"], out.at[i, "association_role"] = matches[0]
        elif len(matches) > 1:
            out.at[i, "track_id"] = -2
            out.at[i, "association_role"] = "ambiguous"
    return out, pd.DataFrame(records, columns=None if records else table_columns)
