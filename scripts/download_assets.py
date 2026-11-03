"""Download public assets outside the checkout; no embedded credentials."""
import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset", choices=["data", "wan", "robot", "all"])
    p.add_argument("--root", default=os.environ.get("WOVR_ASSET_ROOT", str(Path(__file__).resolve().parents[2] / "wovr-assets")))
    p.add_argument("--revision", default="main", help="Hub revision; resolved commit is saved in download-provenance.json")
    a = p.parse_args()
    root = Path(a.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    api = HfApi()
    for kind in (["data", "wan", "robot"] if a.asset == "all" else [a.asset]):
        repo, repo_type = {
            "data": ("jzndd/RoboTwin_adjust_bottle_For_ACWM_Training", "dataset"),
            "wan": ("Wan-AI/Wan2.2-TI2V-5B", "model"),
            "robot": ("TianxingChen/RoboTwin2.0", "dataset"),
        }[kind]
        revision = api.repo_info(repo, repo_type=repo_type, revision=a.revision).sha
        if kind == "robot":
            archive = hf_hub_download(repo, "embodiments.zip", repo_type=repo_type, revision=revision,
                                      cache_dir=root / "downloads")
            destination = root / "robot" / "aloha-agilex"
            count = 0
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    parts = Path(member.filename).parts
                    if "aloha-agilex" not in parts or member.is_dir():
                        continue
                    relative = Path(*parts[parts.index("aloha-agilex")+1:])
                    target = (destination / relative).resolve()
                    if not target.is_relative_to(destination.resolve()):
                        raise ValueError("Unsafe archive path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    count += 1
            if not count or not (destination / "config.yml").exists():
                raise RuntimeError("Upstream embodiment archive layout changed; see docs/assets.md")
        else:
            destination = root / ("datasets/adjust_bottle" if kind == "data" else "models/Wan2.2-TI2V-5B")
            snapshot_download(repo, repo_type=repo_type, revision=revision, local_dir=destination)
        record = dict(asset=kind, repo=repo, repo_type=repo_type, revision=revision, path=str(destination))
        (root / (kind + "-download-provenance.json")).write_text(json.dumps(record, indent=2)+"\n")
        records.append(record)
    print(json.dumps(records, indent=2))

if __name__ == "__main__":
    main()
