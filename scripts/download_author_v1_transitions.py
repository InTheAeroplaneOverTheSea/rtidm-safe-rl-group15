"""Download or verify the transition data published with the author-v1 paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REVISIONS_FILE = (
    Path(__file__).resolve().parents[1] / "configs" / "author_v1_hf_revisions.json"
)
MARKER_NAME = "author_v1_transition_manifest.json"
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pin(path: Path) -> tuple[str, str, int]:
    data = json.loads(path.read_text())
    dataset = data.get("transition_dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"{path}: transition_dataset must be an object")
    repo_id = dataset.get("repo_id")
    revision = dataset.get("revision")
    expected_count = dataset.get("expected_npz_files")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"{path}: invalid transition dataset repo_id")
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{path}: invalid transition dataset revision {revision!r}")
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError(f"{path}: invalid expected_npz_files {expected_count!r}")
    return repo_id, revision, expected_count


def transition_files(saved_files_dir: Path) -> list[Path]:
    return sorted((saved_files_dir / "env_transitions").rglob("*.npz"))


def verify(saved_files_dir: Path, repo_id: str, revision: str, expected_count: int) -> None:
    marker_path = saved_files_dir / MARKER_NAME
    marker = json.loads(marker_path.read_text())
    if marker.get("repo_id") != repo_id or marker.get("revision") != revision:
        raise RuntimeError(
            f"{marker_path}: transition data is not the pinned author-v1 revision; "
            "rerun jobs/setup_surf.slurm"
        )
    files = transition_files(saved_files_dir)
    if len(files) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} author-v1 transition files, found {len(files)}; "
            "rerun jobs/setup_surf.slurm"
        )
    recorded_files = marker.get("files")
    if not isinstance(recorded_files, dict) or len(recorded_files) != expected_count:
        raise RuntimeError(f"{marker_path}: incomplete transition file manifest")
    for path in files:
        relative = path.relative_to(saved_files_dir).as_posix()
        expected_hash = recorded_files.get(relative)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{path}: SHA-256 {actual_hash}, expected {expected_hash}; "
                "rerun jobs/setup_surf.slurm"
            )
    print(f"Verified {len(files)} author-v1 transition files at {revision}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glucoalg-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument(
        "--revisions-file",
        type=Path,
        default=DEFAULT_REVISIONS_FILE,
        help="Pinned author-v1 Hugging Face revisions.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the installed transition files without downloading.",
    )
    args = parser.parse_args()

    glucoalg_dir = args.glucoalg_dir.resolve()
    saved_files_dir = glucoalg_dir / "saved_files"
    repo_id, revision, expected_count = load_pin(args.revisions_file.resolve())
    if args.verify_only:
        verify(saved_files_dir, repo_id, revision, expected_count)
        return
    if args.download_dir is None:
        parser.error("--download-dir is required unless --verify-only is used")

    snapshot_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=args.download_dir.resolve(),
            allow_patterns=["env_transitions/**"],
        )
    )
    source = snapshot_root / "env_transitions"
    if not source.is_dir():
        raise FileNotFoundError(f"Pinned transition snapshot is incomplete: {source}")
    saved_files_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, saved_files_dir / "env_transitions", dirs_exist_ok=True)

    files = transition_files(saved_files_dir)
    if len(files) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} author-v1 transition files after install, "
            f"found {len(files)}"
        )
    marker = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "files": {
            path.relative_to(saved_files_dir).as_posix(): sha256(path) for path in files
        },
        "repo_id": repo_id,
        "revision": revision,
    }
    marker_path = saved_files_dir / MARKER_NAME
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    verify(saved_files_dir, repo_id, revision, expected_count)


if __name__ == "__main__":
    main()
