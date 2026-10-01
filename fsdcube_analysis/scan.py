"""CLI/API for bounded, auditable FLOW scans. Run python -m fsdcube_analysis.scan."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import platform
from importlib.metadata import version
import h5py
import numpy as np
import pandas as pd
from . import __version__
from .flow_io import read_event, ReferenceError
from .charge_splitting import add_pixel_keys, summarize_pixels
from .tracks import TrackConfig, associate_tracks


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    raise TypeError(type(value).__name__)


def scan_files(inputs, output, *, start_event=0, max_events=500,
               pixel_mode="address", tracks=True, track_config=None, locking=True):
    """Scan max_events consecutive events PER FILE; retain failed/empty events.

    Tables are held in memory: this is a bounded pilot scanner. Use separate
    output directories for larger batches. No existing directory is overwritten.
    """
    if start_event < 0 or max_events <= 0:
        raise ValueError("start_event >= 0 and max_events > 0 required")
    paths = [Path(p).resolve(strict=True) for p in inputs]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Provide at least one input, with no duplicate paths")
    if pixel_mode not in {"address", "network"}:
        raise ValueError("Unknown pixel mode")
    config = track_config or TrackConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", version=__version__, python=platform.python_version(),
                    dependencies={name: version(name) for name in
                                  ["numpy", "pandas", "h5py", "scikit-learn", "matplotlib"]},
                    command=sys.argv, start_event=start_event, max_events_per_file=max_events,
                    pixel_mode=pixel_mode, tracks=tracks, track_config=asdict(config),
                    units=dict(Q="ke-", position="cm", t_drift="CRS ticks", delta_t="us"),
                    files=[], warnings=["Exploratory cuts; no validated muon identification.",
                    "t0 NONE: absolute drift location unverified; event boundaries can truncate charge.",
                    "Network pixel keys require geometry validation; no charge-sign cut."])
    try:
        repo = Path(__file__).resolve().parents[1]
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        manifest["git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    manifest["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob("*.py")}
    write_manifest = lambda: (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_value) + "\n")
    write_manifest()
    event_records, hit_frames, track_frames = [], [], []
    try:
        for file_id, path in enumerate(paths):
            with h5py.File(path, "r", locking=locking) as f:
                n_events = len(f["charge/events/data"])
                manifest["files"].append(dict(file_id=file_id, path=str(path), size_bytes=path.stat().st_size,
                    mtime_ns=path.stat().st_mtime_ns, total_events=n_events,
                    attrs={name: dict(f[name].attrs) for name in
                           ["run_info", "geometry_info", "charge/calib_prompt_hits", "combined/t0"]}))
                stop = min(n_events, start_event + max_events)
                for row in range(start_event, stop):
                    try:
                        hits, audit = read_event(f, row, file_id)
                    except ReferenceError as exc:
                        event = f["charge/events/data"][row]
                        event_records.append(dict(file_id=file_id, event_row=row, event_id=int(event["id"]),
                            expected_nhit=int(event["nhit"]), n_hits=np.nan, status="reference_error", error=str(exc)))
                        continue
                    hits = add_pixel_keys(hits, f["geometry_info"].attrs, pixel_mode)
                    # A mismatched event count or ambiguous/missing t0 is retained but not selected.
                    event_ok = audit["nhit_matches"] and audit["t0_count"] == 1
                    hits["hit_quality_ok"] &= event_ok
                    if tracks:
                        hits, candidates = associate_tracks(hits, config)
                        track_frames.append(candidates)
                        audit["n_accepted_tracks"] = int(candidates["accepted"].sum())
                        audit["n_ambiguous_hits"] = int((hits["track_id"] == -2).sum())
                    else:
                        hits["track_id"] = -1
                        hits["association_count"] = 0
                        hits["association_role"] = "not_run"
                        audit["n_accepted_tracks"] = 0
                        audit["n_ambiguous_hits"] = 0
                    audit["n_selected_quality_hits"] = int(hits["hit_quality_ok"].sum())
                    event_records.append(audit)
                    hit_frames.append(hits)
                    if (row - start_event + 1) % 100 == 0:
                        print(f"{path.name}: {row - start_event + 1}/{stop - start_event} events", flush=True)
        if not hit_frames:
            # Build an empty typed template even when every requested event failed or range is empty.
            from .flow_io import REQUIRED
            empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in REQUIRED})
            for c in ["file_id", "event_row", "hit_row", "pixel_io_channel", "t0_type", "track_id"]:
                empty[c] = pd.Series(dtype="int64")
            for c in ["hit_quality_ok", "finite_position"]:
                empty[c] = pd.Series(dtype="bool")
            empty["t_drift_us"] = pd.Series(dtype="float64")
            hit_frames = [empty]
        hits = pd.concat(hit_frames, ignore_index=True)
        ranked, pixels, gaps = summarize_pixels(hits)
        selected = hits.loc[hits["track_id"] >= 0]
        track_hits, track_pixels, track_gaps = summarize_pixels(selected, track=True)
        tables = dict(events=pd.DataFrame(event_records, columns=None if event_records else
                      ["file_id", "event_row", "status", "error", "n_hits"]),
                      hits=ranked, pixels=pixels, gaps=gaps,
                      tracks=pd.concat(track_frames, ignore_index=True) if track_frames else
                      pd.DataFrame(columns=["file_id", "event_row", "track_id", "accepted", "reason"]),
                      track_hits=track_hits, track_pixels=track_pixels, track_gaps=track_gaps)
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
        manifest.update(status="complete", rows={k: len(v) for k, v in tables.items()},
                        reference_error_events=sum(r["status"] != "ok" for r in event_records))
        write_manifest()
        return tables
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_manifest()
        raise


def load_results(directory):
    """Read complete scanner output, preserving nullable integer packet fields."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Scan did not complete; inspect manifest.json")
    tables = {}
    for name in manifest["rows"]:
        path = directory / f"{name}.csv"
        columns = pd.read_csv(path, nrows=0).columns
        dtype = {c: "UInt64" for c in columns if c.startswith("packet_") and c not in
                 {"packet_row", "packet_link_count", "packet_reused"}}
        tables[name] = pd.read_csv(path, dtype=dtype)
    return tables, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--start-event", type=int, default=0)
    p.add_argument("--max-events", type=int, default=500)
    p.add_argument("--pixel-mode", choices=["address", "network"], default="address")
    p.add_argument("--no-tracks", action="store_true")
    p.add_argument("--no-file-locking", action="store_true", help="Read-only workaround for filesystems without HDF5 locking")
    for key, value in asdict(TrackConfig()).items():
        p.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = vars(p.parse_args())
    config = TrackConfig(**{k: args.pop(k) for k in asdict(TrackConfig())})
    args["tracks"] = not args.pop("no_tracks")
    args["locking"] = not args.pop("no_file_locking")
    tables = scan_files(**args, track_config=config)
    print({k: len(v) for k, v in tables.items()})


if __name__ == "__main__":
    main()
