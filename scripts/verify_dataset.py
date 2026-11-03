"""Verify the published 500-episode manifest (streams files; no GPU needed)."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args=p.parse_args()
    manifest=json.loads((args.root/"manifest.json").read_text())
    entries=manifest["files"]
    if len(entries)!=500:
        raise ValueError("Expected 500 episodes")
    total=0
    for item in entries:
        path=(args.root/item["path"]).resolve()
        if not path.is_relative_to(args.root.resolve()):
            raise ValueError("Unsafe manifest path")
        h=hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b""):
                h.update(chunk)
        if h.hexdigest()!=item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {item['path']}")
        total+=path.stat().st_size
    print(json.dumps(dict(episodes=len(entries), bytes=total, sha256_verified=True)))

if __name__=="__main__":
    main()
