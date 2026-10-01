"""Candidate pixel identities and observed charge summaries, not deposit truth."""
import numpy as np
import pandas as pd

EVENT_KEY = ["file_id", "event_row"]
PIXEL_ADDRESS = ["io_group", "pixel_io_channel", "chip_id", "channel_id"]
PIXEL_KEY = EVENT_KEY + PIXEL_ADDRESS


def add_pixel_keys(hits, geometry_attrs, mode="address"):
    """address is conservative; network mode is an explicitly provisional alias key."""
    if mode not in {"address", "network"}:
        raise ValueError("pixel mode must be address or network")
    out = hits.copy()
    if (out["io_channel"] < 1).any():
        raise ValueError("IO channels must be one-based")
    if mode == "network":
        if not bool(geometry_attrs.get("network_agnostic", False)):
            raise ValueError("Network grouping requires network_agnostic geometry")
        n = int(geometry_attrs["n_io_channels_per_tile"])
        if n <= 0:
            raise ValueError("n_io_channels_per_tile must be positive")
        out["pixel_io_channel"] = (out["io_channel"].astype("int64") - 1) // n
    else:
        out["pixel_io_channel"] = out["io_channel"].astype("int64")
    out["pixel_mode"] = mode
    return out


def summarize_pixels(hits, track=False, geometry_tolerance_cm=1e-4):
    """Return ranked long-form hits, per-pixel summaries, and adjacent-hit gaps.

    No charge sign cut. Q1/Q2/Q3 are conveniences; ranked hits preserve any order.
    Unresolved equal timestamps invalidate order_quality_ok, not the charge sum.
    Track input must contain uniquely associated hits with track_id >= 0.
    """
    key = PIXEL_KEY + (["track_id"] if track else [])
    ranked = hits.sort_values(key + ["t_drift", "hit_row"]).copy()
    grouped = ranked.groupby(key, sort=False)
    ranked["hit_number"] = grouped.cumcount() + 1
    ranked["delta_t_us"] = grouped["t_drift_us"].diff()
    columns = key + ["multiplicity", "Q_total", "Q1", "Q2", "Q3", "duration_us",
                     "sum_quality_ok", "order_quality_ok", "any_time_tie", "t0_type"]
    records = []
    for identity, group in ranked.groupby(key, sort=False):
        times = group["t_drift_us"].to_numpy()
        finite_time = bool(np.isfinite(times).all())
        ties = bool(group["t_drift"].dropna().duplicated().any())
        yz_span = group[["y", "z"]].max() - group[["y", "z"]].min()
        geometry_ok = bool(group["finite_position"].all() and (yz_span <= geometry_tolerance_cm).all())
        quality = bool(group["hit_quality_ok"].all() and geometry_ok)
        q = group["Q"].to_numpy()
        record = dict(zip(key, identity))
        record.update(multiplicity=len(group), Q_total=float(np.sum(q)),
                      Q1=float(q[0]), Q2=float(q[1]) if len(q) > 1 else np.nan,
                      Q3=float(q[2]) if len(q) > 2 else np.nan,
                      duration_us=float(times[-1] - times[0]) if finite_time else np.nan,
                      t_first_us=float(times[0]), t_last_us=float(times[-1]),
                      y_span_cm=float(yz_span["y"]), z_span_cm=float(yz_span["z"]),
                      n_io_channels=int(group["io_channel"].nunique()),
                      any_disabled=bool(group["is_disabled"].any()),
                      any_time_tie=ties, geometry_ok=geometry_ok,
                      sum_quality_ok=quality, order_quality_ok=quality and not ties,
                      t0_type=int(group["t0_type"].iloc[0]),
                      n_nonpositive_charge=int((q <= 0).sum()))
        if track:
            n_original = int(group["event_pixel_nhits"].iloc[0])
            record.update(event_pixel_nhits=n_original,
                          association_complete=len(group) == n_original,
                          n_recovered=int((group["association_role"] == "recovered").sum()))
        records.append(record)
    pixels = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    if not records:
        for column in key:
            pixels[column] = pd.Series(dtype=ranked[column].dtype)
    gaps = ranked.loc[ranked["hit_number"] > 1].copy()
    # Plotting must apply whole-pixel flags, not only the later hit's validity.
    gaps = gaps.merge(pixels[key + ["sum_quality_ok", "order_quality_ok"]], on=key, how="left")
    return ranked, pixels, gaps
