"""Download the public T1D checkpoints for all eight paper baselines.

Run this from a SURF staging job, not on a login node.  Every Hugging Face
snapshot is pinned to the revision returned by the Hub and recorded, together
with installed-file hashes, in a JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ALGORITHMS = {
    "CPO": "cpo",
    "CUP": "cup",
    "FOCOPS": "focops",
    "OnCRPO": "oncrpo",
    "PCPO": "pcpo",
    "PPOLag": "ppolag",
    "RCPO": "rcpo",
    "TRPOLag": "trpolag",
}
COHORTS = ("child", "adolescent", "adult")
SEEDS = (0, 1, 2)
EPOCH = 2441


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == sha256(source):
        return
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glucoalg-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(ALGORITHMS),
        default=tuple(ALGORITHMS),
    )
    args = parser.parse_args()

    glucoalg_dir = args.glucoalg_dir.resolve()
    download_dir = args.download_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    api = HfApi()
    manifest: dict[str, object] = {
        "created_at_utc": timestamp,
        "checkpoint_epoch": EPOCH,
        "repositories": {},
    }

    for algorithm in args.algorithms:
        slug = ALGORITHMS[algorithm]
        for cohort in COHORTS:
            repo_id = f"safe-diabetes-benchmark/safe-diabetes-t1d-{cohort}-{slug}"
            info = api.model_info(repo_id)
            snapshot_root = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=info.sha,
                    local_dir=download_dir / cohort / slug,
                    allow_patterns=[
                        "config/seed*/config.json",
                        f"checkpoints/seed*/epoch-{EPOCH}.pt",
                    ],
                )
            )
            repo_record: dict[str, object] = {"revision": info.sha, "seeds": {}}

            for seed in SEEDS:
                source_config = snapshot_root / "config" / f"seed{seed}" / "config.json"
                source_checkpoint = (
                    snapshot_root
                    / "checkpoints"
                    / f"seed{seed}"
                    / f"epoch-{EPOCH}.pt"
                )
                if not source_config.is_file() or not source_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Incomplete public checkpoint: {repo_id} seed{seed}"
                    )

                target_root = (
                    glucoalg_dir
                    / "saved_models"
                    / "t1d"
                    / cohort
                    / algorithm
                    / f"seed{seed}"
                )
                target_config = target_root / "config.json"
                target_checkpoint = target_root / "torch_save" / f"epoch-{EPOCH}.pt"
                install_file(source_config, target_config)
                install_file(source_checkpoint, target_checkpoint)

                repo_record["seeds"][str(seed)] = {
                    "config_sha256": sha256(target_config),
                    "checkpoint_sha256": sha256(target_checkpoint),
                }

            manifest["repositories"][repo_id] = repo_record

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
