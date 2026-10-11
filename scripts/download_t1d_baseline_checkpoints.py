"""Download the public T1D checkpoints for all eight paper baselines.

Run this from a SURF staging job, not on a login node.  Every Hugging Face
snapshot is pinned to the January author-v1 revision in the repository's
provenance file and recorded, together with installed-file hashes, in a JSON
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


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
DEFAULT_REVISIONS_FILE = (
    Path(__file__).resolve().parents[1] / "configs" / "author_v1_hf_revisions.json"
)
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


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


def load_policy_revisions(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    if data.get("checkpoint_epoch") != EPOCH:
        raise ValueError(
            f"{path}: expected checkpoint_epoch {EPOCH}, "
            f"got {data.get('checkpoint_epoch')!r}"
        )
    revisions = data.get("policy_repositories")
    if not isinstance(revisions, dict):
        raise ValueError(f"{path}: policy_repositories must be an object")
    for repo_id, revision in revisions.items():
        if not isinstance(repo_id, str) or not isinstance(revision, str):
            raise ValueError(f"{path}: repository IDs and revisions must be strings")
        if REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError(f"{path}: invalid revision for {repo_id}: {revision!r}")
    return revisions


def policy_paths(
    glucoalg_dir: Path, cohort: str, algorithm: str, seed: int
) -> tuple[Path, Path]:
    target_root = (
        glucoalg_dir
        / "saved_models"
        / "t1d"
        / cohort
        / algorithm
        / f"seed{seed}"
    )
    return (
        target_root / "config.json",
        target_root / "torch_save" / f"epoch-{EPOCH}.pt",
    )


def verify_installed(
    glucoalg_dir: Path,
    manifest_path: Path,
    algorithms: list[str],
    policy_revisions: dict[str, str],
) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("checkpoint_epoch") != EPOCH:
        raise RuntimeError(
            f"{manifest_path}: expected checkpoint_epoch {EPOCH}, "
            f"got {manifest.get('checkpoint_epoch')!r}"
        )
    if manifest.get("revision_policy") != "author-v1-january-2026":
        raise RuntimeError(
            f"{manifest_path}: checkpoint revision policy is not author-v1; "
            "rerun the checkpoint download job"
        )
    repositories = manifest.get("repositories")
    if not isinstance(repositories, dict):
        raise RuntimeError(f"{manifest_path}: repositories must be an object")

    for algorithm in algorithms:
        slug = ALGORITHMS[algorithm]
        for cohort in COHORTS:
            repo_id = f"safe-diabetes-benchmark/safe-diabetes-t1d-{cohort}-{slug}"
            expected_revision = policy_revisions.get(repo_id)
            if expected_revision is None:
                raise RuntimeError(f"No pinned author-v1 revision for {repo_id}")
            repo_record = repositories.get(repo_id)
            if not isinstance(repo_record, dict):
                raise RuntimeError(
                    f"{manifest_path}: missing {repo_id}; rerun the checkpoint download job"
                )
            if repo_record.get("revision") != expected_revision:
                raise RuntimeError(
                    f"{manifest_path}: {repo_id} uses revision "
                    f"{repo_record.get('revision')!r}, expected author-v1 "
                    f"{expected_revision}; rerun the checkpoint download job"
                )
            seeds = repo_record.get("seeds")
            if not isinstance(seeds, dict):
                raise RuntimeError(f"{manifest_path}: missing seed records for {repo_id}")

            for seed in SEEDS:
                seed_record = seeds.get(str(seed))
                if not isinstance(seed_record, dict):
                    raise RuntimeError(
                        f"{manifest_path}: missing {repo_id} seed{seed} record"
                    )
                config_path, checkpoint_path = policy_paths(
                    glucoalg_dir, cohort, algorithm, seed
                )
                expected_hashes = {
                    config_path: seed_record.get("config_sha256"),
                    checkpoint_path: seed_record.get("checkpoint_sha256"),
                }
                for installed_path, expected_hash in expected_hashes.items():
                    if not installed_path.is_file():
                        raise FileNotFoundError(
                            f"Missing installed author-v1 asset: {installed_path}"
                        )
                    actual_hash = sha256(installed_path)
                    if actual_hash != expected_hash:
                        raise RuntimeError(
                            f"{installed_path}: SHA-256 {actual_hash}, expected "
                            f"{expected_hash}; rerun the checkpoint download job"
                        )

    print("Verified installed author-v1 checkpoints for: " + " ".join(algorithms))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glucoalg-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--revisions-file",
        type=Path,
        default=DEFAULT_REVISIONS_FILE,
        help="Pinned author-v1 Hugging Face revisions.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify installed files and their pinned revisions without downloading.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(ALGORITHMS),
        default=tuple(ALGORITHMS),
    )
    args = parser.parse_args()

    glucoalg_dir = args.glucoalg_dir.resolve()
    policy_revisions = load_policy_revisions(args.revisions_file.resolve())
    algorithms = list(args.algorithms)
    if args.verify_only:
        verify_installed(
            glucoalg_dir,
            args.manifest.resolve(),
            algorithms,
            policy_revisions,
        )
        return
    if args.download_dir is None:
        parser.error("--download-dir is required unless --verify-only is used")

    download_dir = args.download_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: dict[str, object] = {
        "created_at_utc": timestamp,
        "checkpoint_epoch": EPOCH,
        "revision_policy": "author-v1-january-2026",
        "repositories": {},
    }

    for algorithm in algorithms:
        slug = ALGORITHMS[algorithm]
        for cohort in COHORTS:
            repo_id = f"safe-diabetes-benchmark/safe-diabetes-t1d-{cohort}-{slug}"
            revision = policy_revisions.get(repo_id)
            if revision is None:
                raise RuntimeError(f"No pinned author-v1 revision for {repo_id}")
            snapshot_root = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=download_dir / cohort / slug,
                    allow_patterns=[
                        "config/seed*/config.json",
                        f"checkpoints/seed*/epoch-{EPOCH}.pt",
                    ],
                )
            )
            repo_record: dict[str, object] = {"revision": revision, "seeds": {}}

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

                target_config, target_checkpoint = policy_paths(
                    glucoalg_dir, cohort, algorithm, seed
                )
                install_file(source_config, target_config)
                install_file(source_checkpoint, target_checkpoint)

                repo_record["seeds"][str(seed)] = {
                    "config_sha256": sha256(target_config),
                    "checkpoint_sha256": sha256(target_checkpoint),
                }

            manifest["repositories"][repo_id] = repo_record

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_installed(glucoalg_dir, args.manifest, algorithms, policy_revisions)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
