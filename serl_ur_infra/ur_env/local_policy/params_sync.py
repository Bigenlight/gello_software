"""Pull freshly published policy parameters from the learner host to laptop3.

WHAT THIS IS
------------
The learner publishes a new policy every 50 learner steps (measured real
inter-publish wall clock p50 **5.77 s**) and, when ``HIL_PARAMS_EXPORT=1``,
writes each publication to ``<run_root>/params_live/`` as

    params_v%08d.msgpack   flax.serialization.to_bytes(<trainable-only tree>)
    LATEST.json            {"version", "learner_step", "sha256", "bytes",
                            "filename", "schema"}

This module is the laptop3 half: poll ``LATEST.json``, and when its ``version``
grows, fetch that one blob, prove it is intact, turn it back into a full
parameter tree, and hand the result to a swap callback.  The proxy's inference
path never blocks on any of it -- a poll that fails leaves the currently served
parameters exactly where they are.

WHY THE TRUNK IS NOT ON THE WIRE
--------------------------------
The full tree is 32,006,980 B, of which the frozen ResNet-10 trunk is
19,623,168 B -- 61% of every transfer, and it is the same bytes every time
because the trunk is frozen by construction (``resnetv1-10-frozen`` behind a
``stop_gradient``; ``FrozenResNet10TrunkExtractor`` exists precisely to prove
it never moves).  laptop3 already has that file, SHA-identical, at
``third_party/hil-serl/examples/experiments/resnet10_params.pkl``.  So the
export prunes it (12,383,812 B on the wire) and this module GRAFTS the local
copy back in.  At ~3.6 MB/s that is the difference between ~8.9 s and ~3.4 s
per publication, against a 5.77 s publication period -- i.e. the difference
between keeping up and falling permanently behind.

The graft reuses ``ur_env.learner.agent._load_resnet10_params_from_file``
(the same loader ``create_frozen_trunk_feature_agent`` uses) exactly ONCE, at
codec construction, so the pickle is parsed and SHA-verified by the code that
owns that format and never by a second copy of it here.  Per-swap grafting is
then a pure tree splice of that already-verified subtree: no pickle, no
hashing, no filesystem in the 5.77 s loop.

WHAT "REFUSE" MEANS HERE
------------------------
Every rejection path -- bad manifest, short read, SHA mismatch, undeserializable
blob, tree-shape drift, version regression -- ends the same way: warn, keep the
parameters already being served, try again next poll.  Stale-but-alive beats
correct-but-dead for the same reason the upstream actor conflates rather than
queues: the arm is moving, and a policy 5 s out of date is a policy, while no
policy is a fault.  The one exception is a version REGRESSION, which is loud
every warning interval because it means the server started a new lineage under
a running proxy -- the parameters and the replay stream have silently stopped
belonging to each other.

CLOCKS
------
Ordering and staleness use :func:`time.monotonic` (never wall clock, never
across hosts).  Phase durations go through
:class:`ur_env.latency_profile.LatencyProfiler` under role ``"paramsync"``, so
they cost nothing when ``HIL_LATENCY_PROFILE`` is unset.

IMPORTS
-------
Module scope is pure stdlib plus the stdlib-only latency profiler, so the actor
venv (no jax, no flax) can import this file.  jax/flax/numpy appear only inside
:class:`FrozenTrunkParamsCodec`, which only the proxy builds.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from ur_env.latency_profile import LatencyProfiler

__all__ = [
    "AppliedParams",
    "DEFAULT_LATENCY_PROFILE_DIR",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_SSH_CONTROL_DIR",
    "DEFAULT_SSH_HOST",
    "FROZEN_TRUNK_SUBTREE_KEY",
    "LATENCY_ROLE",
    "LATEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "NO_PARAMS_VERSION",
    "PARAMS_DIR_NAME",
    "POLL_INTERVAL_ENV_VAR",
    "SSH_CONTROL_DIR_ENV_VAR",
    "SSH_HOST_ENV_VAR",
    "FrozenTrunkParamsCodec",
    "LocalDirectoryFetcher",
    "ParamsFetcher",
    "ParamsHolder",
    "ParamsIntegrityError",
    "ParamsLoadError",
    "ParamsManifest",
    "ParamsManifestError",
    "ParamsNotFoundError",
    "ParamsSwapError",
    "ParamsSyncClient",
    "ParamsSyncError",
    "ParamsTransportError",
    "SshParamsFetcher",
    "blob_filename",
    "graft_frozen_trunk",
    "manifest_for_blob",
    "prune_frozen_trunk",
    "remote_params_dir",
    "resolve_poll_interval",
    "serialize_trainable_params",
]

_LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pinned contract constants
# --------------------------------------------------------------------------- #

#: Directory the learner exports into, under its run root.
PARAMS_DIR_NAME = "params_live"

#: Manifest filename inside that directory.
LATEST_FILENAME = "LATEST.json"

#: ``schema`` value this client understands.  A blob announcing anything else
#: is refused rather than guessed at -- a silently mis-parsed parameter tree is
#: exactly the failure this whole module exists to make impossible.
MANIFEST_SCHEMA_VERSION = 1

#: The one key whose subtree is pruned from the wire and grafted back locally.
#: In the production tree it occurs exactly once, at
#: ``modules_actor/encoder/encoder_cam1/pretrained_encoder`` (36 of 81 leaves;
#: flax parameter sharing means cam2 resolves to that same module and
#: contributes no second copy).  The prune/graft here is written by KEY NAME
#: rather than by that hardcoded path so a second shared trunk -- or a renamed
#: camera -- cannot silently ship 19 MB of frozen weights over a 3.6 MB/s link.
FROZEN_TRUNK_SUBTREE_KEY = "pretrained_encoder"

#: ``version`` reported by a holder that has never been given parameters.  It is
#: below every legal version (which is ``>= 0``), so the ordinary "apply only
#: when it grew" rule accepts the very first publication with no special case.
NO_PARAMS_VERSION = -1

#: Poll cadence.  2.0 s against a 5.77 s publication period means a new policy
#: is normally noticed within one period and never later than two.
DEFAULT_POLL_INTERVAL_S = 2.0
POLL_INTERVAL_ENV_VAR = "HIL_PARAMS_POLL_S"

#: Same host alias ``run_hil_server.sh`` uses, and the same env var name, so an
#: operator shell that already redirected the launcher redirects this too.
DEFAULT_SSH_HOST = "junhyeong_ai"
SSH_HOST_ENV_VAR = "HIL_SSH_HOST"
SSH_CONTROL_DIR_ENV_VAR = "HIL_PARAMS_SSH_CONTROL_DIR"

#: Latency profiler role for this component.
LATENCY_ROLE = "paramsync"

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Where opt-in latency profiling writes when the operator set no
#: ``HIL_LATENCY_PROFILE_DIR``.  Derived from this file's location, not written
#: out absolute, so a second checkout does not pour its samples into the
#: first one's directory -- the rule ``remote_actor.py`` already follows.
DEFAULT_LATENCY_PROFILE_DIR = str(
    _REPO_ROOT / "ros2_ur_ws" / "gello_logs" / "hil_latency"
)

#: Where the ssh ControlMaster socket lives.  Next to the other operator-scratch
#: directories under ``ros2_ur_ws/gello_logs/`` (git-ignored as a whole), so it
#: shares their lifetime and nobody has to remember a second cleanup location.
#: A unix socket path is capped near 104 bytes by the kernel, which this leaves
#: roughly 25 bytes of headroom under for a default-depth checkout; a deeper one
#: gets a warning, not a silent connection failure.
DEFAULT_SSH_CONTROL_DIR = str(
    _REPO_ROOT / "ros2_ur_ws" / "gello_logs" / "hil_params_sync"
)

_CONTROL_PATH_WARN_LENGTH = 100

#: Remote paths are built by us from an operator-supplied run root, then handed
#: to ``ssh``/``scp``, which may or may not expand them through a remote shell
#: depending on the OpenSSH version.  Rather than depend on which, only paths
#: that mean the same thing with and without shell expansion are allowed --
#: the identical rule ``run_hil_server.sh`` applies to ``HIL_REMOTE_DATA_ROOT``.
_SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]*$")

#: ``filename`` arrives from a remote file, so it is treated as untrusted input:
#: a bare name, no separators, no ``..``.  Without this a manifest could point
#: the fetcher at any path on either host.
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")

_PINNED_BLOB_NAME = re.compile(r"^params_v(\d+)\.msgpack$")

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def blob_filename(version: int) -> str:
    """Pinned blob name for ``version`` (``params_v%08d.msgpack``)."""

    _require_version(version, name="version")
    return f"params_v{int(version):08d}.msgpack"


def remote_params_dir(run_root: str) -> str:
    """``<run_root>/params_live`` -- the directory the learner exports into."""

    text = str(run_root).rstrip("/")
    if not text:
        raise ValueError("run_root must not be empty")
    return posixpath.join(text, PARAMS_DIR_NAME)


def resolve_poll_interval(
    value: Optional[float] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> float:
    """Explicit value > ``HIL_PARAMS_POLL_S`` > :data:`DEFAULT_POLL_INTERVAL_S`.

    An unparseable or non-positive env value warns and falls back rather than
    refusing to start: a typo in an operator's shell must not be able to keep
    the proxy from serving.
    """

    if value is not None:
        interval = float(value)
        if not interval > 0.0:
            raise ValueError("poll_interval_s must be positive")
        return interval
    source = os.environ if env is None else env
    text = str(source.get(POLL_INTERVAL_ENV_VAR, "") or "").strip()
    if not text:
        return DEFAULT_POLL_INTERVAL_S
    try:
        interval = float(text)
    except ValueError:
        interval = 0.0
    if not interval > 0.0:
        _LOGGER.warning(
            "ignoring %s=%r (must be a positive number); using %.3f s",
            POLL_INTERVAL_ENV_VAR,
            text,
            DEFAULT_POLL_INTERVAL_S,
        )
        return DEFAULT_POLL_INTERVAL_S
    return interval


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ParamsSyncError(RuntimeError):
    """Base class: this poll produced nothing usable; keep serving."""


class ParamsTransportError(ParamsSyncError):
    """ssh/scp (or the injected transport) failed."""


class ParamsNotFoundError(ParamsTransportError):
    """The remote file does not exist.

    Distinguished from a transport failure because it is the NORMAL state
    before the learner's first export, and repeating a warning for it every
    2 s would bury the warnings that mean something.
    """


class ParamsManifestError(ParamsSyncError):
    """``LATEST.json`` is missing a field, malformed, or a foreign schema."""


class ParamsIntegrityError(ParamsSyncError):
    """The blob's size or SHA-256 disagrees with its manifest."""


class ParamsLoadError(ParamsSyncError):
    """The blob deserialized, grafted or validated wrong."""


class ParamsSwapError(ParamsSyncError):
    """The swap callback refused (or failed on) an otherwise valid candidate.

    Its own class because the proxy's callback is where a candidate gets smoked
    on the real agent: "the weights arrived intact but will not serve" is a
    different operator story from "the weights did not arrive".
    """


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _require_version(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


@dataclass(frozen=True)
class ParamsManifest:
    """Parsed ``LATEST.json``.

    ``size_bytes`` is the JSON key ``"bytes"``; the field is renamed here only
    because a dataclass field called ``bytes`` shadows the builtin at every
    later reader's expense.
    """

    version: int
    learner_step: int
    sha256: str
    size_bytes: int
    filename: str
    schema: int = MANIFEST_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ParamsManifest":
        if not isinstance(payload, Mapping):
            raise ParamsManifestError("manifest must be a JSON object")
        try:
            schema = payload["schema"]
            version = payload["version"]
            learner_step = payload["learner_step"]
            sha256 = payload["sha256"]
            size_bytes = payload["bytes"]
            filename = payload["filename"]
        except KeyError as exc:
            raise ParamsManifestError(f"manifest is missing {exc.args[0]!r}") from exc

        if isinstance(schema, bool) or not isinstance(schema, int):
            raise ParamsManifestError("manifest schema must be an integer")
        if int(schema) != MANIFEST_SCHEMA_VERSION:
            raise ParamsManifestError(
                f"manifest schema {schema} is not the supported "
                f"{MANIFEST_SCHEMA_VERSION}"
            )
        try:
            version = _require_version(version, name="manifest version")
            learner_step = _require_version(
                learner_step, name="manifest learner_step"
            )
        except ValueError as exc:
            raise ParamsManifestError(str(exc)) from exc
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise ParamsManifestError("manifest bytes must be an integer")
        if size_bytes <= 0:
            raise ParamsManifestError("manifest bytes must be positive")
        if not isinstance(sha256, str) or not _SHA256_HEX.match(sha256):
            raise ParamsManifestError(
                "manifest sha256 must be 64 lowercase hex characters"
            )
        if not isinstance(filename, str) or not _SAFE_FILENAME.match(filename):
            raise ParamsManifestError(
                "manifest filename must be a bare name of [A-Za-z0-9._-]"
            )
        return cls(
            version=version,
            learner_step=learner_step,
            sha256=sha256,
            size_bytes=int(size_bytes),
            filename=filename,
            schema=MANIFEST_SCHEMA_VERSION,
        )

    @classmethod
    def from_json(cls, text: str) -> "ParamsManifest":
        try:
            payload = json.loads(text)
        except Exception as exc:
            raise ParamsManifestError(f"manifest is not valid JSON: {exc}") from exc
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "learner_step": self.learner_step,
            "sha256": self.sha256,
            "bytes": self.size_bytes,
            "filename": self.filename,
            "schema": self.schema,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True)

    def verify_blob(self, blob: bytes) -> None:
        """Raise :class:`ParamsIntegrityError` unless ``blob`` is this blob.

        Size is checked first: it is free, and a truncated transfer is the
        overwhelmingly likely failure over a link that has already been
        measured at 3.6 MB/s for a 12 MB payload.
        """

        actual_size = len(blob)
        if actual_size != self.size_bytes:
            raise ParamsIntegrityError(
                f"params v{self.version} size mismatch: expected "
                f"{self.size_bytes} B, got {actual_size} B"
            )
        actual_sha = hashlib.sha256(blob).hexdigest()
        if actual_sha != self.sha256:
            raise ParamsIntegrityError(
                f"params v{self.version} sha256 mismatch: expected "
                f"{self.sha256}, got {actual_sha}"
            )

    def filename_matches_version(self) -> bool:
        """Advisory: does the pinned name encode this manifest's version?

        Advisory on purpose.  The SHA is the integrity authority; a name that
        disagrees is worth a warning, not a refusal that would strand the whole
        feature behind a formatting choice.
        """

        match = _PINNED_BLOB_NAME.match(self.filename)
        return match is None or int(match.group(1)) == self.version


def manifest_for_blob(
    blob: bytes,
    *,
    version: int,
    learner_step: int,
    filename: Optional[str] = None,
) -> ParamsManifest:
    """Build the manifest that describes ``blob``.

    Lives here so the client's own tests can produce byte-exact exports without
    importing the learner-side exporter (which needs jax); the exporter and this
    helper are two implementations of one pinned contract, and the round trip
    through :meth:`ParamsManifest.verify_blob` is what keeps them honest.
    """

    return ParamsManifest(
        version=_require_version(version, name="version"),
        learner_step=_require_version(learner_step, name="learner_step"),
        sha256=hashlib.sha256(blob).hexdigest(),
        size_bytes=len(blob),
        filename=filename or blob_filename(version),
    )


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #


@runtime_checkable
class ParamsFetcher(Protocol):
    """Read-only access to the learner host's export directory.

    Two calls, because that is all the protocol needs: one small text read for
    the manifest and one bulk file read for the blob.  Keeping it this narrow is
    what lets every test in this module run against a local directory and never
    open a socket.
    """

    def fetch_text(self, remote_path: str) -> str:
        """Return the remote file's contents as text.

        Raise :class:`ParamsNotFoundError` when it does not exist and
        :class:`ParamsTransportError` for anything else.
        """

    def fetch_file(self, remote_path: str, local_path: Union[str, os.PathLike]) -> None:
        """Copy the remote file to ``local_path``, same error contract."""


def _validate_remote_path(remote_path: str) -> str:
    text = str(remote_path)
    if not _SAFE_REMOTE_PATH.match(text) or ".." in text.split("/"):
        raise ValueError(
            "remote path must be absolute, of [A-Za-z0-9._/-], and contain no "
            f"'..' segment (got: {remote_path!r})"
        )
    return text


class LocalDirectoryFetcher:
    """Serve the export from a directory on this machine.

    Used by every test in this module, and genuinely useful in production the
    day the learner and the proxy share a filesystem (an sshfs mount, or a
    single-host bring-up).  It applies the same remote-path rules as ssh so a
    path that works here cannot fail to work there.
    """

    def __init__(self, root: Union[str, os.PathLike]) -> None:
        self._root = Path(os.path.expanduser(os.fspath(root)))

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, remote_path: str) -> Path:
        text = str(remote_path)
        if text.startswith("/"):
            _validate_remote_path(text)
            return Path(text)
        # A relative path is interpreted against the configured root, which is
        # what makes a tmp_path-based test read like the real deployment.
        if ".." in text.split("/"):
            raise ValueError(f"remote path must contain no '..' segment: {text!r}")
        return self._root / text

    def fetch_text(self, remote_path: str) -> str:
        path = self._resolve(remote_path)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ParamsNotFoundError(f"{path} does not exist") from exc
        except OSError as exc:
            raise ParamsTransportError(f"could not read {path}: {exc}") from exc

    def fetch_file(
        self, remote_path: str, local_path: Union[str, os.PathLike]
    ) -> None:
        path = self._resolve(remote_path)
        try:
            shutil.copyfile(path, os.fspath(local_path))
        except FileNotFoundError as exc:
            raise ParamsNotFoundError(f"{path} does not exist") from exc
        except OSError as exc:
            raise ParamsTransportError(f"could not copy {path}: {exc}") from exc


_MISSING_FILE_MARKERS = (
    "no such file",
    "not a regular file",
    "no such directory",
)


class SshParamsFetcher:
    """Default transport: ssh/scp over one multiplexed connection.

    WHY ControlMaster IS NOT OPTIONAL
    ---------------------------------
    A fresh ssh handshake to the learner host measures **~0.44 s**.  At a 2.0 s
    poll that is 22% of the interval spent on TLS-equivalent setup, every
    interval, forever -- and it lands on the same laptop that is driving a
    UR7e.  ``ControlMaster=auto`` + ``ControlPersist=60`` collapses every poll
    after the first onto one already-open connection, and the 60 s persist is
    comfortably longer than the poll interval, so the connection is effectively
    permanent for the session without ever being unkillable.

    The constructor touches no filesystem and opens no connection (the rule
    ``latency_profile`` and ``bc_inference_log`` already follow): a typo'd host
    or an unwritable control directory must surface as one warning at the first
    poll, never as a session that refuses to start.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        *,
        control_dir: Optional[Union[str, os.PathLike]] = None,
        control_persist_s: int = 60,
        connect_timeout_s: int = 10,
        text_timeout_s: float = 20.0,
        file_timeout_s: float = 120.0,
        ssh_binary: str = "ssh",
        scp_binary: str = "scp",
        batch_mode: bool = True,
        runner: Optional[Callable[..., Any]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        source = os.environ if env is None else env
        self.host = str(
            host
            if host is not None
            else (source.get(SSH_HOST_ENV_VAR, "") or DEFAULT_SSH_HOST)
        ).strip()
        if not self.host:
            raise ValueError("ssh host must not be empty")
        if control_dir is None:
            control_dir = (
                str(source.get(SSH_CONTROL_DIR_ENV_VAR, "") or "").strip()
                or DEFAULT_SSH_CONTROL_DIR
            )
        self.control_dir = Path(os.path.expanduser(os.fspath(control_dir)))
        self.control_persist_s = int(control_persist_s)
        self.connect_timeout_s = int(connect_timeout_s)
        self.text_timeout_s = float(text_timeout_s)
        self.file_timeout_s = float(file_timeout_s)
        self.ssh_binary = str(ssh_binary)
        self.scp_binary = str(scp_binary)
        self.batch_mode = bool(batch_mode)
        self._runner = runner if runner is not None else subprocess.run
        self._prepared = False
        self._warned_long_control_path = False

    # -- command construction (pure; tests assert on it) -------------------- #

    @property
    def control_path(self) -> str:
        """``ControlPath`` template.  ``%r``/``%h`` are expanded by ssh."""

        return os.fspath(self.control_dir / "cm-%r@%h")

    def ssh_options(self) -> list[str]:
        options = [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={self.control_path}",
            "-o",
            f"ControlPersist={self.control_persist_s}",
            "-o",
            f"ConnectTimeout={self.connect_timeout_s}",
        ]
        if self.batch_mode:
            # No password prompt may ever block a poll thread on a robot laptop.
            options[:0] = ["-o", "BatchMode=yes"]
        return options

    def text_argv(self, remote_path: str) -> list[str]:
        path = _validate_remote_path(remote_path)
        return [
            self.ssh_binary,
            *self.ssh_options(),
            self.host,
            "--",
            "cat",
            "--",
            path,
        ]

    def file_argv(
        self, remote_path: str, local_path: Union[str, os.PathLike]
    ) -> list[str]:
        path = _validate_remote_path(remote_path)
        return [
            self.scp_binary,
            *self.ssh_options(),
            "-p",
            f"{self.host}:{path}",
            os.fspath(local_path),
        ]

    # -- use ---------------------------------------------------------------- #

    def fetch_text(self, remote_path: str) -> str:
        completed = self._run(
            self.text_argv(remote_path),
            timeout=self.text_timeout_s,
            what=f"read {remote_path}",
        )
        stdout = completed.stdout
        if isinstance(stdout, bytes):
            return stdout.decode("utf-8", errors="replace")
        return "" if stdout is None else str(stdout)

    def fetch_file(
        self, remote_path: str, local_path: Union[str, os.PathLike]
    ) -> None:
        self._run(
            self.file_argv(remote_path, local_path),
            timeout=self.file_timeout_s,
            what=f"copy {remote_path}",
        )

    # -- internals ---------------------------------------------------------- #

    def _prepare(self) -> None:
        if self._prepared:
            return
        try:
            self.control_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise ParamsTransportError(
                f"could not create ssh control directory {self.control_dir}: {exc}"
            ) from exc
        if (
            not self._warned_long_control_path
            and len(self.control_path) > _CONTROL_PATH_WARN_LENGTH
        ):
            # A unix socket path over the kernel limit makes ssh fall back to a
            # fresh handshake per poll, silently, at 0.44 s each.
            self._warned_long_control_path = True
            _LOGGER.warning(
                "ssh ControlPath is %d characters (%s); multiplexing may fail "
                "silently -- set %s to a shorter directory",
                len(self.control_path),
                self.control_path,
                SSH_CONTROL_DIR_ENV_VAR,
            )
        self._prepared = True

    def _run(self, argv: list[str], *, timeout: float, what: str) -> Any:
        self._prepare()
        try:
            completed = self._runner(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ParamsTransportError(
                f"ssh transport timed out after {timeout:g} s ({what})"
            ) from exc
        except OSError as exc:
            raise ParamsTransportError(
                f"ssh transport could not start ({what}): {exc}"
            ) from exc
        if int(getattr(completed, "returncode", 1) or 0) == 0:
            return completed
        stderr = getattr(completed, "stderr", b"") or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        lowered = str(stderr).lower()
        detail = " ".join(str(stderr).split())[:400]
        if any(marker in lowered for marker in _MISSING_FILE_MARKERS):
            # Heuristic, and deliberately so: ssh reports the REMOTE command's
            # exit status, so a missing file and a broken link both arrive as a
            # nonzero int.  The stderr text is the only discriminator available
            # without a second round trip, and getting it wrong costs a warning,
            # not correctness -- both branches keep the current parameters.
            raise ParamsNotFoundError(f"remote file missing ({what}): {detail}")
        raise ParamsTransportError(
            f"ssh transport failed rc={completed.returncode} ({what}): {detail}"
        )


# --------------------------------------------------------------------------- #
# Tree prune / graft / codec
# --------------------------------------------------------------------------- #


def _is_mapping(node: Any) -> bool:
    return isinstance(node, Mapping)


def prune_frozen_trunk(tree: Any) -> Any:
    """Drop every ``pretrained_encoder`` subtree; share all remaining leaves.

    This is the wire form: the learner's exporter produces exactly this and
    then ``flax.serialization.to_bytes`` it.  Container mappings are rebuilt as
    plain dicts; array leaves are referenced, never copied, so pruning a 32 MB
    tree allocates a few kilobytes of dicts.
    """

    if not _is_mapping(tree):
        raise ValueError("parameter tree must be a mapping")
    pruned: dict[str, Any] = {}
    for key, value in tree.items():
        if key == FROZEN_TRUNK_SUBTREE_KEY:
            continue
        pruned[key] = prune_frozen_trunk(value) if _is_mapping(value) else value
    return pruned


def graft_frozen_trunk(trainable: Any, reference: Any) -> Any:
    """Rebuild the full tree: ``trainable`` leaves + ``reference``'s trunk.

    ``reference`` supplies the SHAPE of the result as well as the trunk: every
    key it has must be present in ``trainable`` (unless it is the trunk), and a
    key ``trainable`` has and ``reference`` does not is a hard error rather than
    a passenger.  Structural drift therefore fails HERE, with the offending
    path named, instead of surfacing later as an opaque jax tree-structure
    mismatch.
    """

    return _graft(trainable, reference, ())


def _graft(trainable: Any, reference: Any, path: tuple[str, ...]) -> Any:
    if not _is_mapping(reference):
        raise ParamsLoadError(
            f"reference parameter tree has a non-mapping container at "
            f"{'/'.join(path) or '<root>'}"
        )
    if not _is_mapping(trainable):
        raise ParamsLoadError(
            f"candidate parameter tree is not a mapping at "
            f"{'/'.join(path) or '<root>'}"
        )
    grafted: dict[str, Any] = {}
    for key, reference_value in reference.items():
        child = path + (str(key),)
        if key == FROZEN_TRUNK_SUBTREE_KEY:
            if key in trainable:
                raise ParamsLoadError(
                    f"candidate still carries the frozen trunk at "
                    f"{'/'.join(child)}; the export must prune it"
                )
            grafted[key] = reference_value
            continue
        if key not in trainable:
            raise ParamsLoadError(
                f"candidate is missing {'/'.join(child)}"
            )
        candidate_value = trainable[key]
        grafted[key] = (
            _graft(candidate_value, reference_value, child)
            if _is_mapping(reference_value)
            else candidate_value
        )
    extra = sorted(set(map(str, trainable)) - set(map(str, reference)))
    if extra:
        raise ParamsLoadError(
            f"candidate has unexpected keys at {'/'.join(path) or '<root>'}: "
            f"{extra}"
        )
    return grafted


def serialize_trainable_params(params: Any) -> bytes:
    """Prune the trunk and serialize -- the exact bytes the learner writes.

    Needs flax.  Kept here (rather than only on the learner side) so this
    module's own tests can build byte-exact exports, and so the encode and
    decode halves of the pinned contract are readable next to each other.
    """

    from flax import serialization

    return serialization.to_bytes(prune_frozen_trunk(params))


class FrozenTrunkParamsCodec:
    """Turn one exported blob back into a full, validated parameter tree.

    Construction is where the expensive, once-per-session work happens: build
    the reference tree, SHA-verify the ResNet-10 asset, and (via
    :meth:`from_agent`) re-apply it through the learner's own loader so the
    reference trunk provably equals the bytes on disk.  ``load`` is then pure
    tree work -- deserialize, splice, validate -- which is what keeps a 5.77 s
    publication period comfortable.
    """

    def __init__(
        self,
        reference_params: Any,
        *,
        resnet_sha256: Optional[str] = None,
    ) -> None:
        if not _is_mapping(reference_params):
            raise ValueError("reference_params must be a mapping")
        self._reference = reference_params
        self._template = prune_frozen_trunk(reference_params)
        self.resnet_sha256 = resnet_sha256

    @classmethod
    def from_agent(
        cls,
        agent: Any,
        *,
        image_keys: Iterable[str] = ("cam1", "cam2"),
        resnet_source_path: Optional[Union[str, os.PathLike]] = None,
        resnet_cache_path: Optional[Union[str, os.PathLike]] = None,
    ) -> "FrozenTrunkParamsCodec":
        """Build from a freshly created agent, pinning its trunk to the asset.

        The graft source is the agent's own trunk subtree AFTER the learner's
        loader has written the verified ``resnet10_params.pkl`` into it.  That
        is a deliberate ordering: the pickle is parsed exactly once, by the code
        that owns that format (``ur_env.learner.agent``), and every later swap
        splices an already-verified subtree instead of re-reading 19 MB.
        """

        from ur_env.learner.agent import (
            _load_resnet10_params_from_file,
            ensure_resnet10_cache,
            file_sha256,
        )

        cache = ensure_resnet10_cache(
            source_path=resnet_source_path, cache_path=resnet_cache_path
        )
        pinned = _load_resnet10_params_from_file(agent, tuple(image_keys), cache)
        return cls(pinned.state.params, resnet_sha256=file_sha256(cache))

    @property
    def reference_params(self) -> Any:
        """The full tree every candidate is validated against."""

        return self._reference

    @property
    def template(self) -> Any:
        """Trainable-only tree used as the ``from_bytes`` target."""

        return self._template

    def load(self, blob: bytes) -> Any:
        """Deserialize, graft the local trunk, validate.  Raises on any doubt."""

        from flax import serialization

        try:
            trainable = serialization.from_bytes(self._template, blob)
        except Exception as exc:
            raise ParamsLoadError(
                f"could not deserialize params blob: {type(exc).__name__}: {exc}"
            ) from exc
        full = graft_frozen_trunk(trainable, self._reference)
        self.validate(full)
        return full

    def validate(self, params: Any) -> None:
        """Tree structure + leaf shapes/dtypes identical to the reference.

        Delegates to ``ur_env.learner.policy.validate_parameter_tree`` -- the
        same check the learner's own publisher runs before it serves a snapshot,
        so laptop3 cannot accept a tree the server would have rejected.  It also
        rejects non-finite leaves, which is the shape a partially-written export
        takes once it has survived the SHA (it cannot, but defence in depth here
        costs one pass over 32 MB every 5.77 s).
        """

        from ur_env.learner.policy import (
            PolicyValidationError,
            validate_parameter_tree,
        )

        try:
            validate_parameter_tree(params, self._reference)
        except PolicyValidationError as exc:
            raise ParamsLoadError(f"params failed validation: {exc}") from exc

    __call__ = load


# --------------------------------------------------------------------------- #
# Holder
# --------------------------------------------------------------------------- #


class AppliedParams(NamedTuple):
    """What :meth:`ParamsHolder.current` hands to the inference path."""

    params: Any
    version: int
    applied_monotonic: float


class ParamsHolder:
    """The one object the inference path reads parameters from.

    Deliberately three members wide.  It holds an immutable reference and
    swaps it; it does not validate trees, fetch anything, or know what a
    version means beyond "must not go backwards".  The inference path can
    therefore call :meth:`current` at loop rate and pay one uncontended lock
    acquisition, and a swap can never leave a reader looking at half a tree --
    the reference is replaced, never mutated.

    The monotonicity check is not redundant with the client's: this is what
    makes the pinned "policy_version never decreases within a session" rule
    hold for ANY writer, including a future one that has its own poll loop.
    """

    __slots__ = ("_lock", "_params", "_version", "_applied_monotonic")

    def __init__(
        self, params: Any = None, version: int = NO_PARAMS_VERSION
    ) -> None:
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("version must be an integer")
        if version < NO_PARAMS_VERSION:
            raise ValueError(f"version must be >= {NO_PARAMS_VERSION}")
        if params is None and version != NO_PARAMS_VERSION:
            raise ValueError("a versioned holder must be given params")
        self._lock = threading.Lock()
        self._params = params
        self._version = int(version)
        self._applied_monotonic = time.monotonic()

    @property
    def version(self) -> int:
        """Currently served version, or :data:`NO_PARAMS_VERSION`."""

        with self._lock:
            return self._version

    def current(self) -> AppliedParams:
        """``(params, version, applied_monotonic)`` -- one consistent triple.

        ``applied_monotonic`` is :func:`time.monotonic` at the swap, so a caller
        gets ``params_age_s`` as ``time.monotonic() - applied_monotonic``
        without a second lock acquisition or a second clock read here.
        """

        with self._lock:
            return AppliedParams(
                self._params, self._version, self._applied_monotonic
            )

    def swap(self, params: Any, version: int) -> None:
        """Publish ``params`` as ``version``.  Refuses to go backwards."""

        version = _require_version(version, name="version")
        if params is None:
            raise ValueError("params must not be None")
        now = time.monotonic()
        with self._lock:
            if version < self._version:
                raise ValueError(
                    f"refusing to swap params backwards: serving v{self._version}, "
                    f"offered v{version}"
                )
            self._params = params
            self._version = version
            self._applied_monotonic = now


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class _WarnState:
    __slots__ = ("last", "suppressed")

    def __init__(self, last: float) -> None:
        self.last = last
        self.suppressed = 0


#: How often a repeating warning is allowed to reach the log.  This runs every
#: 2 s; an unrated warning would produce 1,800 identical lines an hour and bury
#: the session's real events.
DEFAULT_WARN_INTERVAL_S = 60.0


class ParamsSyncClient:
    """Poll the learner's export directory and hot-swap what it finds.

    One poll is: read ``LATEST.json``; if its ``version`` did not grow, stop.
    Otherwise fetch THAT blob only -- never the ones in between, because a
    parameter set has no history worth replaying and the newest is strictly
    better -- verify size and SHA, deserialize + graft + validate, then hand the
    candidate to the swap callback.

    Nothing here can end a session.  :meth:`poll_once` converts every failure
    into a rate-limited warning and a ``False`` return, and the background
    thread additionally catches anything :meth:`poll_once` could not have
    anticipated.  The parameters already being served survive all of it.
    """

    def __init__(
        self,
        fetcher: ParamsFetcher,
        remote_dir: str,
        deserializer: Callable[[bytes], Any],
        *,
        holder: Optional[ParamsHolder] = None,
        swap_callback: Optional[Callable[[Any, int], None]] = None,
        poll_interval_s: Optional[float] = None,
        profiler: Optional[LatencyProfiler] = None,
        scratch_dir: Optional[Union[str, os.PathLike]] = None,
        warn_interval_s: float = DEFAULT_WARN_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not callable(getattr(fetcher, "fetch_text", None)) or not callable(
            getattr(fetcher, "fetch_file", None)
        ):
            raise TypeError("fetcher must provide fetch_text and fetch_file")
        if not callable(deserializer):
            raise TypeError("deserializer must be callable")
        self._fetcher = fetcher
        self._remote_dir = str(remote_dir).rstrip("/")
        if not self._remote_dir:
            raise ValueError("remote_dir must not be empty")
        self._deserializer = deserializer
        self._holder = holder if holder is not None else ParamsHolder()
        # The default sink is the holder.  A caller that supplies its own
        # callback OWNS publication -- the proxy uses that seam to smoke the
        # candidate on the real agent before letting the inference path see it,
        # and a callback that raises leaves the applied version unchanged, so
        # the next poll retries the same blob.
        self._swap_callback = (
            swap_callback if swap_callback is not None else self._holder.swap
        )
        self._poll_interval_s = resolve_poll_interval(poll_interval_s, env=env)
        self._profiler = (
            profiler
            if profiler is not None
            else LatencyProfiler.from_env(
                LATENCY_ROLE, DEFAULT_LATENCY_PROFILE_DIR, env=env
            )
        )
        self._scratch_dir = (
            Path(os.path.expanduser(os.fspath(scratch_dir)))
            if scratch_dir is not None
            else None
        )
        self._warn_interval_s = float(warn_interval_s)
        self._clock = clock
        self._warn_state: dict[str, _WarnState] = {}
        self._lock = threading.Lock()
        self._applied_version = self._holder.version
        self._applied_manifest_at: Optional[float] = None
        self._applied_learner_step: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reported_absent = False
        self._stats = {
            "polls": 0,
            "applied": 0,
            "unchanged": 0,
            "absent": 0,
            "transport_errors": 0,
            "manifest_errors": 0,
            "integrity_errors": 0,
            "load_errors": 0,
            "swap_errors": 0,
            "unexpected_errors": 0,
            "regressions": 0,
        }

    # -- introspection ------------------------------------------------------ #

    @property
    def holder(self) -> ParamsHolder:
        return self._holder

    @property
    def remote_dir(self) -> str:
        return self._remote_dir

    @property
    def manifest_path(self) -> str:
        return posixpath.join(self._remote_dir, LATEST_FILENAME)

    @property
    def poll_interval_s(self) -> float:
        return self._poll_interval_s

    @property
    def applied_version(self) -> int:
        with self._lock:
            return self._applied_version

    @property
    def applied_learner_step(self) -> Optional[int]:
        with self._lock:
            return self._applied_learner_step

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def staleness_s(self) -> Optional[float]:
        """Seconds since ``LATEST.json`` announced the version now applied.

        ``None`` before the first successful apply.  This is the pinned
        ``staleness_s`` field: it answers "how old is the policy the arm is
        being driven by", which is the question a 5.77 s publication period and
        a 3.4 s transfer make interesting.
        """

        with self._lock:
            announced = self._applied_manifest_at
        if announced is None:
            return None
        return max(0.0, self._clock() - announced)

    # -- one poll ----------------------------------------------------------- #

    def poll_once(self) -> bool:
        """Run one poll cycle.  Returns True iff a new version was applied.

        Never raises: this is called from a background thread whose death would
        silently freeze the policy at whatever version it happened to hold.
        """

        record = self._profiler.record()
        started = self._clock()
        self._bump("polls")
        applied = False
        try:
            applied = self._poll(record, started)
        except ParamsNotFoundError as exc:
            self._bump("absent")
            self._note_absent(exc)
        except ParamsTransportError as exc:
            self._bump("transport_errors")
            self._warn("transport", f"params sync transport failed: {exc}")
        except ParamsManifestError as exc:
            self._bump("manifest_errors")
            self._warn("manifest", f"params sync manifest rejected: {exc}")
        except ParamsIntegrityError as exc:
            self._bump("integrity_errors")
            self._warn(
                "integrity",
                f"params sync REFUSED a corrupt blob (keeping v"
                f"{self.applied_version}): {exc}",
            )
        except ParamsSwapError as exc:
            self._bump("swap_errors")
            self._warn(
                "swap",
                f"params sync could not publish (keeping v"
                f"{self.applied_version}): {exc}",
            )
        except ParamsLoadError as exc:
            self._bump("load_errors")
            self._warn(
                "load",
                f"params sync REFUSED an unusable blob (keeping v"
                f"{self.applied_version}): {exc}",
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._bump("unexpected_errors")
            self._warn(
                "unexpected",
                f"params sync poll failed unexpectedly "
                f"({type(exc).__name__}): {exc}",
            )
        finally:
            self._commit(record, applied)
        return applied

    def _poll(self, record: Any, started: float) -> bool:
        with record.phase("poll"):
            text = self._fetcher.fetch_text(self.manifest_path)
        manifest = ParamsManifest.from_json(text)
        if self._reported_absent:
            self._reported_absent = False
            _LOGGER.info(
                "params sync found %s again (v%d)",
                self.manifest_path,
                manifest.version,
            )
        if not manifest.filename_matches_version():
            self._warn(
                "filename",
                f"params manifest names {manifest.filename!r} for version "
                f"{manifest.version}; trusting the sha256, not the name",
            )

        applied_version = self.applied_version
        if manifest.version == applied_version:
            self._bump("unchanged")
            return False
        if manifest.version < applied_version:
            # Loud, and NOT a transient: the only way a published version goes
            # backwards is a new learner lineage under a running proxy, which
            # means the replay stream and the policy have stopped belonging to
            # each other.  Refusing keeps serving a policy whose transitions the
            # old lineage understood.
            self._bump("regressions")
            self._warn(
                "regression",
                f"params sync REFUSING a version regression: serving v"
                f"{applied_version}, learner announced v{manifest.version} "
                f"(learner_step {manifest.learner_step}).  A new lineage under "
                f"a running proxy: restart the proxy to follow it.",
            )
            return False

        blob = self._fetch_blob(record, manifest)
        manifest.verify_blob(blob)
        with record.phase("load"):
            params = self._deserializer(blob)
        if params is None:
            raise ParamsLoadError("deserializer returned None")
        with record.phase("swap"):
            try:
                self._swap_callback(params, manifest.version)
            except Exception as exc:
                raise ParamsSwapError(
                    f"swap callback rejected v{manifest.version} "
                    f"({type(exc).__name__}): {exc}"
                ) from exc
        with self._lock:
            self._applied_version = manifest.version
            self._applied_learner_step = manifest.learner_step
            self._applied_manifest_at = started
            self._stats["applied"] += 1
        _LOGGER.info(
            "params sync applied v%d (learner_step %d, %d B, %.3f s old)",
            manifest.version,
            manifest.learner_step,
            manifest.size_bytes,
            max(0.0, self._clock() - started),
        )
        return True

    def _fetch_blob(self, record: Any, manifest: ParamsManifest) -> bytes:
        """Download to a temp file, read it, delete it.

        Through a file rather than a pipe because ``scp`` is the transport and
        it writes files; the blob is 12 MB, so the extra copy is microseconds
        against a ~3.4 s transfer, and the temp file is unlinked even when the
        transfer fails half-way.
        """

        remote_path = posixpath.join(self._remote_dir, manifest.filename)
        directory = self._scratch_dir
        if directory is not None:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ParamsTransportError(
                    f"could not create scratch directory {directory}: {exc}"
                ) from exc
        handle, temp_name = tempfile.mkstemp(
            prefix="hil-params-", suffix=".msgpack", dir=directory
        )
        os.close(handle)
        try:
            with record.phase("fetch"):
                self._fetcher.fetch_file(remote_path, temp_name)
            try:
                with open(temp_name, "rb") as stream:
                    return stream.read()
            except OSError as exc:
                raise ParamsTransportError(
                    f"could not read fetched blob {temp_name}: {exc}"
                ) from exc
        finally:
            try:
                os.unlink(temp_name)
            except OSError:
                pass

    def _commit(self, record: Any, applied: bool) -> None:
        if not getattr(record, "enabled", False):
            return
        record.set("version", self.applied_version)
        record.set("applied", bool(applied))
        staleness = self.staleness_s()
        if staleness is not None:
            record.set("staleness_s", round(staleness, 3))
        learner_step = self.applied_learner_step
        if learner_step is not None:
            record.set("learner_step", learner_step)
        record.commit()

    # -- thread ------------------------------------------------------------- #

    def start(self) -> None:
        """Start the background poll thread.  Idempotent while it is alive."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run, name="hil-params-sync", daemon=True
            )
            self._thread = thread
        thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the thread to stop and wait for it.  Safe if never started.

        A thread that has not returned within ``timeout`` is left alone with a
        warning rather than waited on forever: it is a daemon blocked in a
        bounded subprocess call, so process exit is not at risk, and blocking
        shutdown here would hold up the operator's Ctrl-C.
        """

        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            _LOGGER.warning(
                "params sync thread did not stop within %.1f s; leaving it "
                "(daemon)",
                timeout,
            )
            return
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except BaseException as exc:  # pragma: no cover - poll_once catches
                # poll_once already swallows everything; this exists so a bug
                # in the swallowing itself still cannot take the thread down.
                self._warn(
                    "thread",
                    f"params sync poll raised out of poll_once "
                    f"({type(exc).__name__}): {exc}",
                )
            self._stop_event.wait(self._poll_interval_s)

    def close(self) -> None:
        """Stop polling and flush the latency sink."""

        self.stop()
        self._profiler.close()

    def __enter__(self) -> "ParamsSyncClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    # -- logging ------------------------------------------------------------ #

    def _bump(self, key: str) -> None:
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + 1

    def _note_absent(self, exc: Exception) -> None:
        """A missing export is the NORMAL state before the learner's first one.

        Said once at INFO, then only at DEBUG, and said again at INFO when it
        comes back.  Anything louder would train the operator to ignore this
        module's output during the first minute of every session.
        """

        if not self._reported_absent:
            self._reported_absent = True
            _LOGGER.info(
                "params sync waiting for %s (%s); the learner exports only "
                "with HIL_PARAMS_EXPORT=1",
                self.manifest_path,
                exc,
            )
        else:
            _LOGGER.debug("params sync still waiting for %s", self.manifest_path)

    def _warn(self, key: str, message: str) -> None:
        now = self._clock()
        with self._lock:
            state = self._warn_state.get(key)
            if state is not None and (now - state.last) < self._warn_interval_s:
                state.suppressed += 1
                return
            suppressed = 0 if state is None else state.suppressed
            self._warn_state[key] = _WarnState(now)
        suffix = (
            f" ({suppressed} identical warning(s) suppressed in the last "
            f"{self._warn_interval_s:g} s)"
            if suppressed
            else ""
        )
        _LOGGER.warning("%s%s", message, suffix)
