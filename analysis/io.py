"""Portable provenance and atomic JSON records for analysis runs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def code_hashes(names=None):
    if names is None:
        names = ['analysis/config.py', 'analysis/io.py', 'analysis/segments.py',
                 'analysis/dqdx.py', 'analysis/run.py', 'pixel_dqdx/segments.py', 'pixel_dqdx/dqdx.py']
    return {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in names}


def runtime_versions():
    return dict(python=sys.version, packages={name: importlib.metadata.version(name)
                for name in ['numpy', 'h5py', 'scipy', 'scikit-learn', 'matplotlib']})


def safe_name(name):
    if not isinstance(name, str) or not name or name in {'.', '..'} or '/' in name or '\\' in name:
        raise ValueError(f'Expected a single directory name: {name!r}')
    return name
