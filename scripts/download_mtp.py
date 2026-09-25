#!/usr/bin/env python3
"""Download the pinned native Q4 MTP adapter, using Python's standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "mlx-community/Qwen3.8-27B-MTP-4bit"
REVISION = "b643c01b6d3b094e325edb6ebd832e16c486c575"
FILES = {
    "config.json": (3804, "16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55"),
    "model.safetensors": (238934137, "76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6"),
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Qwen3.8-27B-MTP-4bit"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        args.output.mkdir(parents=True, exist_ok=True)
    for name, (size, sha) in FILES.items():
        path = args.output / name
        if path.exists():
            if path.stat().st_size != size or digest(path) != sha:
                raise SystemExit(f"Existing {path} does not match the pinned adapter; choose a new --output directory.")
            print(f"Verified {path}")
            continue
        if args.verify_only:
            raise SystemExit(f"Missing {path}")
        part = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", headers={"User-Agent": "qwen-metal-mtp-downloader"})
        total = 0
        hasher = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > size:
                    raise SystemExit(f"Unexpected size for {name}; incomplete file kept at {part}")
                output.write(chunk)
                hasher.update(chunk)
        if total != size or hasher.hexdigest() != sha:
            raise SystemExit(f"Checksum mismatch for {name}; incomplete file kept at {part}")
        part.rename(path)
        print(f"Downloaded and verified {path}")
    if not args.verify_only:
        manifest = {"repository": REPO, "revision": REVISION,
                    "files": {name: {"bytes": size, "sha256": sha} for name, (size, sha) in FILES.items()}}
        (args.output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
