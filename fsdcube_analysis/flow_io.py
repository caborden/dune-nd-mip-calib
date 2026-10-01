"""Read FLOW references without assuming that IDs equal rows or tables align."""
import numpy as np
import pandas as pd

HITS = "charge/calib_prompt_hits"
ADDRESS = ["io_group", "io_channel", "chip_id", "channel_id"]
REQUIRED = ["id", "x", "y", "z", "Q", "Q_raw", "t_drift", "ts_pps",
            "is_disabled", *ADDRESS]


class ReferenceError(ValueError):
    """Malformed, duplicate, or out-of-bounds references."""


def linked_rows(f, source, target, source_row):
    """Direct forward references: region slices the link table, stop exclusive.

    Supports the source-column order used in the inspected FLOW production.
    Missing reference paths are errors, not empty associations.
    """
    base = f"{source}/ref/{target}"
    region = f[f"{base}/ref_region"][source_row]
    start, stop = int(region["start"]), int(region["stop"])
    refs = f[f"{base}/ref"]
    if not 0 <= start <= stop <= len(refs):
        raise ReferenceError(f"Invalid region {base}, row {source_row}: {start}:{stop}")
    links = refs[start:stop]
    rows = links[links[:, 0] == source_row, 1].astype(np.int64)
    if len(rows) != len(np.unique(rows)):
        raise ReferenceError(f"Duplicate links at {base}, row {source_row}")
    if np.any((rows < 0) | (rows >= len(f[f"{target}/data"]))):
        raise ReferenceError(f"Target outside dataset: {base}, row {source_row}")
    return rows


def take_rows(dataset, rows):
    """h5py-compatible arbitrary row order, preserving repeated requested rows."""
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return dataset[unique][inverse]


def read_event(f, event_row, file_id):
    """Return all event hits with provenance, packet checks, and an event audit."""
    missing = set(REQUIRED) - set(f[f"{HITS}/data"].dtype.names)
    if missing:
        raise ValueError(f"Missing calibrated-hit fields: {sorted(missing)}")
    event = f["charge/events/data"][event_row]
    rows = np.sort(linked_rows(f, "charge/events", HITS, event_row))
    hits = pd.DataFrame.from_records(take_rows(f[f"{HITS}/data"], rows))
    hits.insert(0, "hit_row", rows)
    hits.insert(0, "event_id", event["id"])
    hits.insert(0, "event_row", event_row)
    hits.insert(0, "file_id", file_id)
    tick_us = float(f["run_info"].attrs["crs_ticks"])
    if not np.isfinite(tick_us) or tick_us <= 0:
        raise ValueError("crs_ticks must be a positive period in microseconds")
    hits["tick_us"] = tick_us
    hits["t_drift_us"] = hits["t_drift"] * tick_us

    t0_rows = linked_rows(f, "charge/events", "combined/t0", event_row)
    t0 = take_rows(f["combined/t0/data"], t0_rows)
    # Do not choose silently among multiple t0 records.
    t0_type = int(t0["type"][0]) if len(t0) == 1 else -1
    t0_ts = float(t0["ts"][0]) if len(t0) == 1 else np.nan
    hits["t0_type"] = t0_type
    hits["t0_ts"] = t0_ts
    hits["t0_count"] = len(t0)

    packet_links = [linked_rows(f, HITS, "charge/packets", int(r)) for r in rows]
    hits["packet_link_count"] = [len(r) for r in packet_links]
    packet_rows = np.array([int(r[0]) if len(r) == 1 else -1 for r in packet_links], dtype=np.int64)
    hits["packet_row"] = packet_rows
    valid = packet_rows >= 0
    packet_fields = [*ADDRESS, "packet_type", "timestamp", "dataword", "valid_parity",
                     "trigger_type", "reset_sample_flag", "cds_flag"]
    packets = take_rows(f["charge/packets/data"], packet_rows[valid])
    for field in packet_fields:
        # Nullable integer columns retain exact timestamps/IDs with missing links.
        values = pd.Series(pd.array([pd.NA] * len(hits), dtype="UInt64"))
        values.loc[valid] = packets[field]
        hits[f"packet_{field}"] = values
    hits["packet_reused"] = valid & hits["packet_row"].duplicated(keep=False)
    hits["address_match"] = valid
    for field in ADDRESS:
        hits["address_match"] &= hits[field].eq(hits[f"packet_{field}"]).fillna(False).astype(bool)
    hits["parity_ok"] = hits["packet_valid_parity"].eq(1).fillna(False).astype(bool)
    data_type = int(f["run_info"].attrs["data_packet_type"])
    hits["data_packet_ok"] = hits["packet_packet_type"].eq(data_type).fillna(False).astype(bool)
    hits["finite_charge"] = np.isfinite(hits["Q"])
    hits["finite_time"] = np.isfinite(hits["t_drift"])
    hits["finite_position"] = np.isfinite(hits[["x", "y", "z"]]).all(axis=1)
    hits["hit_quality_ok"] = (
        hits["finite_charge"] & hits["finite_time"] & hits["finite_position"]
        & ~hits["is_disabled"] & hits["address_match"] & hits["parity_ok"]
        & hits["data_packet_ok"] & ~hits["packet_reused"]
    )
    audit = dict(file_id=file_id, event_row=event_row, event_id=int(event["id"]),
                 expected_nhit=int(event["nhit"]), n_hits=len(hits),
                 nhit_matches=len(hits) == int(event["nhit"]), t0_count=len(t0),
                 t0_type=t0_type, t0_ts=t0_ts, tick_us=tick_us,
                 n_bad_hits=int((~hits["hit_quality_ok"]).sum()),
                 n_nonpositive_charge=int((hits["Q"] <= 0).sum()),
                 status="ok", error="")
    return hits, audit
