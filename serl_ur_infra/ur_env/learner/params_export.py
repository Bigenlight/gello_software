"""Publish the learner's TRAINABLE parameters to disk for a local-inference peer.

WHY THIS EXISTS
---------------
The production actor's Step RPC blocks the 10 Hz control loop for ~156 ms of
*server* compute.  The local-inference design moves policy evaluation onto
laptop3 and leaves training, reward authority and replay on the GPU server, so
something has to carry parameters the other way -- server to laptop -- without
touching the wire protocol (``serl_ur_infra/proto/`` is frozen by design: a
proto change would force a ``SCHEMA_VERSION`` bump on a stack where protobuf
silently drops unknown fields).

A file is the whole transport.  The learner drops a blob into
``<run_root>/params_live/`` at every publish; the laptop pulls it over the ssh
channel that already exists.  Nothing in the gRPC handshake, the actor argv or
``validate_process_contract`` changes.

WHY TRAINABLE-ONLY
------------------
The agent's parameter tree is 32,006,980 B, of which 19,623,168 B is the frozen
ResNet-10 trunk -- **61 %** of every transfer, identical on every publish, and
already present on laptop3 as the SHA-verified
``third_party/hil-serl/examples/experiments/resnet10_params.pkl``.  Sending it
would waste 5.4 s of a 3.6 MB/s link per publish against a measured 5.77 s
inter-publish period; the link would be saturated by a constant.  So the trunk
subtree is pruned here and grafted back from the local pkl on the other side,
leaving 12,383,812 B on the wire.

Pruning is by KEY NAME (``pretrained_encoder``), not by the single hardcoded
path ``modules_actor/encoder/encoder_cam1/pretrained_encoder``: Flax stores a
shared submodule under its first user only, so *where* the one trunk copy lands
is a property of module construction order, not a contract.  A rebuild that
moved it would silently ship 19.6 MB of frozen weights under a path-based
prune.  The canonical path is still asserted -- an export that pruned nothing
is a structural surprise and degrades this sink rather than shipping a blob the
peer cannot graft.

WHY IT NEVER RAISES INTO TRAINING
---------------------------------
:meth:`ParamsExporter.export` is called from the learner thread, inside
``HILSERLLearner.train_once``, whose ``except`` clause converts *any* exception
into a permanent learner fault.  A full disk on the server would therefore end
training.  So this object follows the rule commit ``c86dc54`` set for the
metrics sink and ``ur_env/latency_profile.py`` for the latency sink: catch
everything, warn exactly once, self-degrade to disabled, and let training
continue with a stale blob.  A local-inference peer that stops seeing new
versions is a visible, recoverable condition; a dead learner is not.

For the same reason the constructor touches no filesystem: a typo'd path or a
read-only mount surfaces as one warning at the first export, never as a session
that refuses to start.

ATOMICITY
---------
The reader is a remote poller that may read at any instant, so every file
appears whole or not at all: content is written to a dotted temporary name and
``os.replace``d into place.  The temporary name deliberately does not match
``params_v*.msgpack``, so a reader globbing for blobs cannot even see a partial
one.  The blob is renamed BEFORE ``LATEST.json``, so a manifest never names a
file that is not there yet.

``fsync`` is deliberately omitted.  It would cost tens of milliseconds of
learner-thread time per publish to protect a file that is regenerated every
~6 s and has no value after a crash (a fresh learner writes a fresh run root).
``os.replace`` already gives the only guarantee a live reader needs.

IMPORTS
-------
Module scope is pure stdlib plus ``numpy``-free typing, so the actor venv --
which has neither jax nor flax -- can import, inspect and unit-test everything
except the real serializer.  ``flax.serialization.to_bytes`` is resolved lazily
on first use and is injectable for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Callable, Iterable, Mapping, Optional, Union
import uuid

__all__ = [
    "BLOB_GLOB",
    "DIRECTORY_NAME",
    "ENABLE_ENV_VAR",
    "FROZEN_TRUNK_EXPECTED_PATH",
    "FROZEN_TRUNK_LEAF_KEY",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "RETAINED_BLOBS",
    "ExportedParams",
    "ParamsExportError",
    "ParamsExporter",
    "blob_name",
    "blob_version",
    "existing_versions",
    "is_params_export_enabled",
    "prune_frozen_trunk",
    "read_manifest",
]

_LOGGER = logging.getLogger(__name__)

#: Opt-in switch, read once at :meth:`ParamsExporter.from_env`.  Forwarded to a
#: freshly started learner by ``ros2_ur_ws/run_hil_server.sh`` as an ENV VAR --
#: never as an argv token, because ``validate_process_contract`` compares the
#: learner's argv token by token and a new flag would make every exporting
#: learner fail its own reuse check.
ENABLE_ENV_VAR = "HIL_PARAMS_EXPORT"

#: Accepted truthy spellings, compared lowercased and stripped.  Same table as
#: ``ur_env/latency_profile.py`` so the two server opt-ins cannot disagree about
#: what ``HIL_..=on`` means.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Directory under the run root.  A sibling of ``logs/`` and ``checkpoints/``,
#: so the analyzer, the operator and the puller all find it the same way.
DIRECTORY_NAME = "params_live"

#: Manifest filename.  Fixed, because the remote poller fetches it by name
#: before it knows any version.
MANIFEST_NAME = "LATEST.json"

#: Manifest format version.  Bump when a field's MEANING changes; adding a
#: field does not require it (the reader treats unknown keys as ignorable).
MANIFEST_SCHEMA = 1

_BLOB_TEMPLATE = "params_v%08d.msgpack"
BLOB_GLOB = "params_v*.msgpack"
_BLOB_PREFIX = "params_v"
_BLOB_SUFFIX = ".msgpack"

#: Temporary names start with a dot and do NOT match :data:`BLOB_GLOB`.
_TMP_PREFIX = ".tmp-"

#: How many blobs survive.  Two, not one: a puller that has just decided to
#: fetch version N must still find N on disk after the learner has published
#: N+1.  Two is also the smallest number that makes that race impossible for a
#: single skip-to-newest poller.
RETAINED_BLOBS = 2

#: Flax stores a shared submodule under its FIRST user only, so the one frozen
#: ResNet-10 copy lives at ``modules_actor/encoder/encoder_cam1/
#: pretrained_encoder``.  Pruning matches this KEY at any depth -- see the
#: module docstring for why the path itself is not the contract.
FROZEN_TRUNK_LEAF_KEY = "pretrained_encoder"

#: The path the production agent is expected to carry it at.  Used only as an
#: assertion that pruning found the real thing.
FROZEN_TRUNK_EXPECTED_PATH = (
    "modules_actor",
    "encoder",
    "encoder_cam1",
    FROZEN_TRUNK_LEAF_KEY,
)


class ParamsExportError(RuntimeError):
    """An export could not be produced.  Never escapes :meth:`export`."""


def is_params_export_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """True iff ``HIL_PARAMS_EXPORT`` is set to a truthy spelling."""

    source = os.environ if env is None else env
    return str(source.get(ENABLE_ENV_VAR, "") or "").strip().lower() in _TRUTHY


def blob_name(version: int) -> str:
    """Filename for ``version``.  Zero-padded so lexical order is numeric."""

    return _BLOB_TEMPLATE % int(version)


def blob_version(name: str) -> Optional[int]:
    """Parse a blob filename back to its version, or ``None`` if it is not one."""

    text = str(name)
    if not text.startswith(_BLOB_PREFIX) or not text.endswith(_BLOB_SUFFIX):
        return None
    digits = text[len(_BLOB_PREFIX) : -len(_BLOB_SUFFIX)]
    if not digits.isdigit():
        return None
    return int(digits)


@dataclass(frozen=True)
class ExportedParams:
    """What one successful export produced.  Mirrors ``LATEST.json`` exactly."""

    version: int
    learner_step: int
    sha256: str
    bytes: int
    filename: str
    path: Path

    def manifest(self) -> dict:
        return {
            "version": self.version,
            "learner_step": self.learner_step,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "filename": self.filename,
            "schema": MANIFEST_SCHEMA,
        }


def prune_frozen_trunk(
    params: Any,
    *,
    key: str = FROZEN_TRUNK_LEAF_KEY,
) -> tuple[dict, tuple[tuple[str, ...], ...]]:
    """Return ``(trainable tree, pruned paths)`` -- every ``key`` node removed.

    Mappings are rebuilt as plain ``dict`` (``flax.serialization.to_bytes``
    accepts them and the peer restores onto its own tree, so the container type
    carries no information).  Array leaves are shared, never copied: this runs
    on the learner thread and the tree is 12 MB.
    """

    pruned: list[tuple[str, ...]] = []

    def walk(node: Any, path: tuple[str, ...]) -> Any:
        if not isinstance(node, Mapping):
            return node
        result: dict = {}
        for raw_key, value in node.items():
            name = str(raw_key)
            child = path + (name,)
            if name == key:
                pruned.append(child)
                continue
            result[name] = walk(value, child)
        return result

    if not isinstance(params, Mapping):
        raise ParamsExportError("parameter tree must be a mapping")
    return walk(params, ()), tuple(pruned)


def read_manifest(directory: Union[str, os.PathLike]) -> Optional[dict]:
    """Read ``LATEST.json`` from ``directory``.

    Returns ``None`` when it is absent or unreadable.  A malformed manifest is
    also ``None``: to a reader "no usable manifest" and "no manifest" are the
    same situation, and the writer is the only party that can fix either.
    """

    path = Path(os.path.expanduser(os.fspath(directory))) / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _default_serializer(tree: Any) -> bytes:
    """Serialize with flax.  Imported lazily: the actor venv has no flax."""

    from flax.serialization import to_bytes

    return to_bytes(tree)


class ParamsExporter:
    """Write the trainable half of the agent's parameters at every publish.

    Thread confinement is not assumed -- the learner thread calls
    :meth:`export` and a supervisor may read :attr:`last_export` -- so state is
    guarded by one lock.  Serialization happens under it, which is what stops
    two exports from interleaving their retention passes and deleting a blob
    the manifest still names.
    """

    def __init__(
        self,
        directory: Optional[Union[str, os.PathLike]] = None,
        *,
        enabled: bool = True,
        retain: int = RETAINED_BLOBS,
        serializer: Optional[Callable[[Any], bytes]] = None,
    ) -> None:
        # An exporter with nowhere to write is a disabled exporter, not an
        # error: this constructor must never be able to fail a session.
        self._enabled = bool(enabled) and directory is not None
        self._directory = (
            Path(os.path.expanduser(os.fspath(directory)))
            if directory is not None
            else None
        )
        if isinstance(retain, bool) or not isinstance(retain, int) or retain < 1:
            raise ValueError("retain must be a positive integer")
        self._retain = retain
        self._serializer = serializer or _default_serializer
        self._lock = threading.Lock()
        self._degraded = False
        self._warned = False
        self._last: Optional[ExportedParams] = None
        self._export_count = 0

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_env(
        cls,
        directory: Optional[Union[str, os.PathLike]] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
        retain: int = RETAINED_BLOBS,
        serializer: Optional[Callable[[Any], bytes]] = None,
    ) -> "ParamsExporter":
        """Build an exporter enabled iff ``HIL_PARAMS_EXPORT`` is truthy.

        A disabled exporter resolves no path and never touches the filesystem,
        so the default server behaviour is byte-identical to before this
        feature existed.
        """

        if not is_params_export_enabled(env):
            return cls(None, enabled=False, retain=retain, serializer=serializer)
        return cls(directory, enabled=True, retain=retain, serializer=serializer)

    # -- state -------------------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        """False when opted out or after an I/O failure degraded this sink."""

        return self._enabled

    @property
    def degraded(self) -> bool:
        """True when a failure turned a live exporter off mid-session."""

        return self._degraded

    @property
    def directory(self) -> Optional[Path]:
        """Destination directory, or ``None`` when disabled.  May not exist."""

        return self._directory

    @property
    def retain(self) -> int:
        return self._retain

    @property
    def last_export(self) -> Optional[ExportedParams]:
        """The most recent successful export, or ``None``."""

        return self._last

    @property
    def export_count(self) -> int:
        return self._export_count

    @property
    def manifest_path(self) -> Optional[Path]:
        return None if self._directory is None else self._directory / MANIFEST_NAME

    # -- use ---------------------------------------------------------------- #

    def export(
        self,
        params: Any,
        *,
        learner_step: int,
        version: int,
    ) -> Optional[ExportedParams]:
        """Write one blob + manifest.  Returns ``None`` when nothing was written.

        **This method never raises.**  It is called from the learner thread
        inside ``train_once``, where any exception becomes a permanent learner
        fault; see the module docstring.
        """

        with self._lock:
            if not self._enabled:
                return None
            try:
                return self._export_locked(
                    params, learner_step=int(learner_step), version=int(version)
                )
            except Exception as exc:  # noqa: BLE001 - the whole point
                self._degrade_locked(exc)
                return None

    # -- internals ---------------------------------------------------------- #

    def _export_locked(
        self, params: Any, *, learner_step: int, version: int
    ) -> Optional[ExportedParams]:
        if version < 0 or learner_step < 0:
            raise ParamsExportError("version and learner_step must be >= 0")
        previous = self._last
        if previous is not None and version < previous.version:
            # A version that went backwards is not an error here -- it is a
            # caller mistake this sink refuses to propagate, because the puller
            # skips to newest and a regressed manifest would pin it to old
            # weights forever.  Silent: the learner cannot act on it.
            return None

        tree, pruned = prune_frozen_trunk(params)
        if not pruned:
            raise ParamsExportError(
                "parameter tree carries no "
                f"{FROZEN_TRUNK_LEAF_KEY!r} subtree to prune; refusing to ship "
                "a blob the local-inference peer cannot graft a trunk into"
            )
        if FROZEN_TRUNK_EXPECTED_PATH not in pruned:
            # Not fatal -- the blob is still trainable-only -- but the peer's
            # graft targets this path, so say it once rather than let the
            # mismatch surface as a shape error on the other host.
            self._warn_locked(
                "frozen trunk is not at "
                f"{'/'.join(FROZEN_TRUNK_EXPECTED_PATH)}; pruned "
                f"{['/'.join(path) for path in pruned]} instead"
            )
        payload = self._serializer(tree)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ParamsExportError("serializer did not return bytes")
        payload = bytes(payload)

        directory = self._directory
        assert directory is not None  # guarded by self._enabled
        directory.mkdir(parents=True, exist_ok=True)
        name = blob_name(version)
        record = ExportedParams(
            version=version,
            learner_step=learner_step,
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
            filename=name,
            path=directory / name,
        )
        # Blob first, manifest second: a manifest must never name a file that
        # has not landed.  Both land whole or not at all.
        _atomic_write(record.path, payload)
        _atomic_write(
            directory / MANIFEST_NAME,
            (
                json.dumps(record.manifest(), sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8"),
        )
        self._last = record
        self._export_count += 1
        self._retain_locked(directory, keep_version=version)
        return record

    def _retain_locked(self, directory: Path, *, keep_version: int) -> None:
        """Delete all but the newest :attr:`retain` blobs.  Best effort.

        Retention failing is not a reason to stop exporting: the next write
        still succeeds, and the disk-full case degrades through the write path
        where it belongs.  So this warns once and returns rather than raising
        into :meth:`export`'s degrade handler.
        """

        try:
            versions = []
            for entry in directory.iterdir():
                parsed = blob_version(entry.name)
                if parsed is not None:
                    versions.append((parsed, entry))
            versions.sort(key=lambda item: item[0])
            for parsed, entry in versions[: max(0, len(versions) - self._retain)]:
                if parsed == keep_version:
                    continue
                try:
                    entry.unlink()
                except FileNotFoundError:
                    pass
        except Exception as exc:  # noqa: BLE001 - best effort by contract
            self._warn_locked(
                f"params export retention failed under {directory}: {exc!r}"
            )

    def _degrade_locked(self, exc: BaseException) -> None:
        self._degraded = True
        self._enabled = False
        self._warn_locked(
            f"disabling params export to {self._directory}: {exc!r}; "
            "training continues and the local-inference peer keeps its last "
            "loaded version"
        )

    def _warn_locked(self, message: str) -> None:
        # Exactly one warning per exporter, whatever fails and however often:
        # this runs every publish, and a repeating warning would bury the log
        # it is trying to make visible.
        if self._warned:
            return
        self._warned = True
        _LOGGER.warning("%s", message)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Create ``path`` with ``payload``; a reader sees it whole or not at all.

    The temporary carries the pid and a uuid so two writers -- an operator's
    stray process, a second learner pointed at the same root -- cannot collide
    on it, and its name cannot match :data:`BLOB_GLOB`.
    """

    tmp = path.parent / f"{_TMP_PREFIX}{path.name}.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(tmp, "wb") as stream:
            stream.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass
        raise


def existing_versions(directory: Union[str, os.PathLike]) -> tuple[int, ...]:
    """Sorted versions of the blobs currently on disk.  Diagnostics only."""

    root = Path(os.path.expanduser(os.fspath(directory)))
    try:
        entries: Iterable[Path] = list(root.iterdir())
    except Exception:
        return ()
    found = [
        parsed
        for parsed in (blob_version(entry.name) for entry in entries)
        if parsed is not None
    ]
    return tuple(sorted(found))
