"""Download and read the pinned Dolly file. The revision and sha256 are fixed so every rebuild of
the traffic starts from the same bytes; a file that does not match is refused, never used."""
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
       f"{REVISION}/databricks-dolly-15k.jsonl")
LICENSE = "CC BY-SA 3.0"


class DollyChecksumMismatch(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str = SHA256) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise DollyChecksumMismatch(
            f"{path} has sha256 {actual}, expected {expected} (revision {REVISION}). "
            "Delete it and run the build again to download the pinned file.")


def ensure_downloaded(path: Path) -> Path:
    """Download the pinned file to `path` when it is absent, then verify it either way."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(path)
    verify(path)
    return path


def load_records(path: Path) -> list[dict]:
    """Every Dolly line as a record with its 0-based line index added."""
    with path.open(encoding="utf-8") as f:
        return [{"index": i, **json.loads(line)} for i, line in enumerate(f) if line.strip()]
