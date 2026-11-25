"""Small file-provenance helpers for current RF evaluations."""

import hashlib
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
