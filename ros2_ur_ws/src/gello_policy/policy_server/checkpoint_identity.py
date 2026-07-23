"""Deterministic identity for a local LeRobot checkpoint directory."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path


_MODEL_PATTERNS = (
    "model*.safetensors",
    "model*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
)
_PROCESSOR_NAMES = (
    "preprocessor.json",
    "postprocessor.json",
    "processor.json",
    "preprocessor",
    "postprocessor",
)
_PROCESSOR_ROOT_PATTERNS = (
    "*preprocessor*",
    "*postprocessor*",
)


@dataclass(frozen=True)
class CheckpointIdentity:
    policy_type: str
    revision: str
    files: tuple[tuple[str, str], ...]


def _selected_files(root: Path) -> list[Path]:
    selected: set[Path] = set()
    config = root / "config.json"
    if not config.is_file():
        raise ValueError(f"checkpoint is missing {config}")
    selected.add(config)
    for pattern in _MODEL_PATTERNS:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    if not any(path.name.startswith(("model", "pytorch_model")) for path in selected):
        raise ValueError("checkpoint has no model*.safetensors or pytorch_model*.bin weights")
    processor_files: dict[str, set[Path]] = {
        "preprocessor": set(),
        "postprocessor": set(),
    }
    for name in _PROCESSOR_NAMES:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"checkpoint identity does not accept symlink: {path}")
        if path.is_file():
            selected.add(path)
        elif path.is_dir():
            selected.update(child for child in path.rglob("*") if child.is_file())
    # LeRobot may save processor descriptors and tensor state as several root
    # files (for example *_processor.json and *_processor.safetensors).
    for kind, pattern in zip(processor_files, _PROCESSOR_ROOT_PATTERNS, strict=True):
        for path in root.glob(pattern):
            if path.is_symlink():
                raise ValueError(f"checkpoint identity does not accept symlink: {path}")
            if path.is_file():
                processor_files[kind].add(path)
                selected.add(path)
            elif path.is_dir():
                children = {child for child in path.rglob("*") if child.is_file()}
                processor_files[kind].update(children)
                selected.update(children)
    for path in selected:
        relative = path.relative_to(root).as_posix().lower()
        for kind in processor_files:
            if kind in relative:
                processor_files[kind].add(path)
    missing = [kind for kind, files in processor_files.items() if not files]
    if missing:
        raise ValueError(
            "checkpoint has no saved " + " or ".join(missing) + " files"
        )
    for path in selected:
        if path.is_symlink():
            raise ValueError(f"checkpoint identity does not accept symlink: {path}")
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def compute_checkpoint_identity(checkpoint: str | Path) -> CheckpointIdentity:
    unresolved_root = Path(checkpoint).expanduser()
    if unresolved_root.is_symlink():
        raise ValueError(
            f"checkpoint identity does not accept symlink: {unresolved_root}"
        )
    root = unresolved_root.resolve()
    if not root.is_dir():
        raise ValueError("checkpoint identity requires a local checkpoint directory")
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read checkpoint config.json: {exc}") from exc
    policy_type = str(config.get("type", "")).strip()
    if not policy_type:
        raise ValueError("checkpoint config.json has no non-empty 'type'")

    entries: list[tuple[str, str]] = []
    for path in _selected_files(root):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        entries.append((path.relative_to(root).as_posix(), digest.hexdigest()))

    manifest = hashlib.sha256()
    for relative, digest in entries:
        manifest.update(relative.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(digest.encode("ascii"))
        manifest.update(b"\n")
    return CheckpointIdentity(
        policy_type=policy_type,
        revision=f"sha256:{manifest.hexdigest()}",
        files=tuple(entries),
    )


def verify_expected_identity(
    identity: CheckpointIdentity,
    *,
    expected_policy_type: str = "",
    expected_revision: str = "",
) -> None:
    expected_policy_type = expected_policy_type.strip()
    expected_revision = expected_revision.strip()
    if not expected_policy_type:
        raise ValueError("expected policy type must be non-empty")
    if not expected_revision or expected_revision.lower() in {"unknown", "auto"}:
        raise ValueError("expected checkpoint revision must be a pinned sha256 manifest")
    if expected_policy_type and expected_policy_type != identity.policy_type:
        raise ValueError(
            f"checkpoint policy type {identity.policy_type!r} != expected "
            f"{expected_policy_type!r}"
        )
    if expected_revision.lower() != identity.revision:
        raise ValueError(
            f"checkpoint revision {identity.revision} != expected "
            f"{expected_revision.lower()}"
        )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the content-derived identity of a LeRobot checkpoint."
    )
    parser.add_argument("checkpoint", help="Local pretrained_model directory")
    args = parser.parse_args()
    identity = compute_checkpoint_identity(args.checkpoint)
    print(
        json.dumps(
            {
                "policy_type": identity.policy_type,
                "revision": identity.revision,
                "files": [
                    {"path": path, "sha256": digest}
                    for path, digest in identity.files
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
