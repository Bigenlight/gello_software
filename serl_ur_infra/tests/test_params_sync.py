"""Contract tests for ``ur_env.local_policy.params_sync``.

TWO TIERS, AND WHY
------------------
Most of this file runs in the ACTOR venv, which has no jax and no flax: the
poll/refuse state machine is the part that must never fail, and it is pure
stdlib once the deserializer is injected.  The tests that need real
``flax.serialization`` bytes -- the prune/graft/validate codec -- are gated with
``pytest.importorskip("flax")`` and run under ``/home/laptop3/venvs/hilserl`` or
``gello-local-policy``.

NOTHING HERE OPENS A SOCKET.  The ssh transport is exercised through an injected
``runner``, and every end-to-end path uses ``LocalDirectoryFetcher`` against a
``tmp_path`` laid out exactly like ``<run_root>/params_live/``.

Actor venv (jax tests skip)::

    cd /home/laptop3/gello_software
    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
      -p no:cacheprovider serl_ur_infra/tests/test_params_sync.py

jax/flax venv (everything runs)::

    ... /home/laptop3/venvs/hilserl/bin/python -m pytest -q ...

The real-agent round trip additionally needs ``RUN_HIL_SERL_ACTUAL_PARAMS_SYNC=1``
because building the production SAC agent costs ~8 s.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pytest


_INFRA = Path(__file__).resolve().parents[1]
_REPO = _INFRA.parent
sys.path.insert(0, str(_INFRA))

from ur_env.latency_profile import LatencyProfiler  # noqa: E402
from ur_env.local_policy import params_sync  # noqa: E402
from ur_env.local_policy.params_sync import (  # noqa: E402
    CONTROL_PATH_BASENAME,
    CONTROL_PATH_SUFFIX_ALLOWANCE,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_SSH_CONTROL_DIR,
    FROZEN_TRUNK_SUBTREE_KEY,
    LATENCY_ROLE,
    LATEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    MAX_CONTROL_PATH_LENGTH,
    MAX_UNIX_SOCKET_PATH,
    NO_PARAMS_VERSION,
    PARAMS_DIR_NAME,
    POLL_INTERVAL_ENV_VAR,
    SSH_CONTROL_DIR_ENV_VAR,
    UNRESOLVED_EXPANSION_FLOOR,
    AppliedParams,
    LocalDirectoryFetcher,
    ParamsFetcher,
    ParamsHolder,
    ParamsIntegrityError,
    ParamsLoadError,
    ParamsManifest,
    ParamsManifestError,
    ParamsNotFoundError,
    ParamsSyncClient,
    ParamsTransportError,
    RemoteIdentity,
    SshParamsFetcher,
    blob_filename,
    expand_control_path,
    fallback_control_dir,
    graft_frozen_trunk,
    manifest_for_blob,
    measure_control_path,
    prune_frozen_trunk,
    remote_params_dir,
    resolve_poll_interval,
    resolve_remote_identity,
)


#: The identity every test in this file sees unless it says otherwise.  It is
#: SHORT on purpose: without it, ``SshParamsFetcher`` would shell out to a real
#: ``ssh -G`` (breaking "no process is ever spawned") and would resolve the real
#: learner alias to ``junhyeong@166.104.146.29``, which on top of a pytest
#: ``tmp_path`` overruns the 90-character ControlPath budget and would send half
#: this file's fetchers down the fallback path for reasons unrelated to what
#: they assert.
_TEST_IDENTITY = RemoteIdentity(user="u", hostname="h", port="22", resolved=True)


@pytest.fixture(autouse=True)
def _pinned_ssh_identity(monkeypatch):
    """No test here may consult the machine's real ssh config."""

    monkeypatch.setattr(
        params_sync,
        "resolve_remote_identity",
        lambda host, **kwargs: _TEST_IDENTITY,
    )
    params_sync.clear_remote_identity_cache()
    yield
    params_sync.clear_remote_identity_cache()


# --------------------------------------------------------------------------- #
# helpers -- a fake export directory in agent A's pinned format
# --------------------------------------------------------------------------- #


def _fake_blob(version: int) -> bytes:
    """A deterministic, version-dependent payload standing in for msgpack."""

    return b"PARAMS-v%08d-" % version + bytes((version * 7 + i) % 251 for i in range(64))


def _publish(
    export_dir: Path,
    version: int,
    *,
    learner_step: int | None = None,
    blob: bytes | None = None,
    corrupt_sha: bool = False,
    corrupt_size: bool = False,
    filename: str | None = None,
    write_blob: bool = True,
) -> ParamsManifest:
    """Write one export exactly the way the learner does (blob then manifest)."""

    export_dir.mkdir(parents=True, exist_ok=True)
    payload = _fake_blob(version) if blob is None else blob
    step = version * 50 if learner_step is None else learner_step
    manifest = manifest_for_blob(
        payload, version=version, learner_step=step, filename=filename
    )
    if corrupt_sha:
        manifest = ParamsManifest(
            version=manifest.version,
            learner_step=manifest.learner_step,
            sha256="0" * 64,
            size_bytes=manifest.size_bytes,
            filename=manifest.filename,
        )
    if corrupt_size:
        manifest = ParamsManifest(
            version=manifest.version,
            learner_step=manifest.learner_step,
            sha256=manifest.sha256,
            size_bytes=manifest.size_bytes + 1,
            filename=manifest.filename,
        )
    if write_blob:
        (export_dir / manifest.filename).write_bytes(payload)
    (export_dir / LATEST_FILENAME).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


class _CountingFetcher:
    """Wrap a fetcher and record every call, so "newest only" is provable."""

    def __init__(self, inner: ParamsFetcher) -> None:
        self._inner = inner
        self.text_calls: list[str] = []
        self.file_calls: list[str] = []

    def fetch_text(self, remote_path: str) -> str:
        self.text_calls.append(remote_path)
        return self._inner.fetch_text(remote_path)

    def fetch_file(self, remote_path: str, local_path) -> None:
        self.file_calls.append(remote_path)
        self._inner.fetch_file(remote_path, local_path)


class _ExplodingFetcher:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def fetch_text(self, remote_path: str) -> str:
        self.calls += 1
        raise self._exc

    def fetch_file(self, remote_path: str, local_path) -> None:  # pragma: no cover
        raise self._exc


class _FakeClock:
    """Monotonic clock the test drives by hand, for warning rate limiting."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _identity_deserializer(blob: bytes):
    """Injected stand-in for the flax codec: the blob IS the parameter tree."""

    return {"payload": blob}


def _client(
    export_dir: Path,
    *,
    fetcher=None,
    deserializer=_identity_deserializer,
    **kwargs,
) -> ParamsSyncClient:
    return ParamsSyncClient(
        fetcher if fetcher is not None else LocalDirectoryFetcher(export_dir),
        str(export_dir),
        deserializer,
        env={},
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def test_manifest_round_trips_the_pinned_json_keys():
    blob = b"x" * 11
    manifest = manifest_for_blob(blob, version=3, learner_step=150)

    payload = json.loads(manifest.to_json())
    assert payload == {
        "version": 3,
        "learner_step": 150,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": 11,
        "filename": "params_v00000003.msgpack",
        "schema": MANIFEST_SCHEMA_VERSION,
    }
    assert ParamsManifest.from_json(manifest.to_json()) == manifest
    manifest.verify_blob(blob)


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema": 2},
        {"version": -1},
        {"version": "3"},
        {"learner_step": None},
        {"sha256": "abc"},
        {"sha256": "A" * 64},
        {"bytes": 0},
        {"bytes": "11"},
        {"filename": "../../.ssh/id_rsa"},
        {"filename": "sub/dir/params.msgpack"},
    ],
)
def test_manifest_refuses_malformed_fields(mutation):
    payload = manifest_for_blob(b"x" * 11, version=3, learner_step=150).to_mapping()
    payload.update(mutation)

    with pytest.raises(ParamsManifestError):
        ParamsManifest.from_mapping(payload)


def test_manifest_refuses_missing_key_and_non_json():
    payload = manifest_for_blob(b"x" * 11, version=3, learner_step=150).to_mapping()
    payload.pop("sha256")
    with pytest.raises(ParamsManifestError, match="sha256"):
        ParamsManifest.from_mapping(payload)
    with pytest.raises(ParamsManifestError, match="valid JSON"):
        ParamsManifest.from_json("{not json")


def test_manifest_verify_blob_names_size_and_sha_separately():
    manifest = manifest_for_blob(b"x" * 11, version=1, learner_step=50)

    with pytest.raises(ParamsIntegrityError, match="size mismatch"):
        manifest.verify_blob(b"x" * 10)
    with pytest.raises(ParamsIntegrityError, match="sha256 mismatch"):
        manifest.verify_blob(b"y" * 11)


def test_pinned_names_and_paths():
    assert blob_filename(0) == "params_v00000000.msgpack"
    assert blob_filename(1234) == "params_v00001234.msgpack"
    assert remote_params_dir("/home/junhyeong/hil-serl-data/runs/r1") == (
        f"/home/junhyeong/hil-serl-data/runs/r1/{PARAMS_DIR_NAME}"
    )
    assert remote_params_dir("/a/b/") == f"/a/b/{PARAMS_DIR_NAME}"


def test_filename_version_mismatch_is_advisory_not_fatal():
    manifest = manifest_for_blob(
        b"x" * 11, version=3, learner_step=150, filename="params_v00000009.msgpack"
    )
    assert manifest.filename_matches_version() is False
    # A name outside the pinned pattern makes no claim at all.
    other = manifest_for_blob(
        b"x" * 11, version=3, learner_step=150, filename="whatever.bin"
    )
    assert other.filename_matches_version() is True


# --------------------------------------------------------------------------- #
# holder
# --------------------------------------------------------------------------- #


def test_holder_starts_empty_and_reports_the_sentinel_version():
    holder = ParamsHolder()

    current = holder.current()
    assert isinstance(current, AppliedParams)
    params, version, applied_at = current
    assert params is None
    assert version == NO_PARAMS_VERSION == -1
    assert holder.version == NO_PARAMS_VERSION
    assert applied_at <= time.monotonic()


def test_holder_swap_publishes_reference_and_advances_applied_time():
    holder = ParamsHolder()
    tree = {"a": 1}

    before = holder.current().applied_monotonic
    time.sleep(0.002)
    holder.swap(tree, 7)

    params, version, applied_at = holder.current()
    assert params is tree
    assert version == 7 == holder.version
    assert applied_at > before


def test_holder_refuses_a_version_regression_and_keeps_serving():
    holder = ParamsHolder()
    holder.swap({"a": 1}, 5)

    with pytest.raises(ValueError, match="backwards"):
        holder.swap({"a": 2}, 4)

    params, version, _ = holder.current()
    assert version == 5
    assert params == {"a": 1}
    # Equal is a legal re-publish; strictly less is not.
    holder.swap({"a": 3}, 5)
    assert holder.current().params == {"a": 3}


def test_holder_rejects_bad_arguments():
    holder = ParamsHolder()
    with pytest.raises(ValueError):
        holder.swap({"a": 1}, -1)
    with pytest.raises(ValueError):
        holder.swap({"a": 1}, True)
    with pytest.raises(ValueError):
        holder.swap(None, 1)
    with pytest.raises(ValueError):
        ParamsHolder({"a": 1}, version=-2)
    with pytest.raises(ValueError):
        ParamsHolder(None, version=3)


def test_concurrent_current_reads_never_see_a_torn_swap():
    """The triple must always be self-consistent under a hammering reader.

    The invariant is not "readers see the newest" -- it is that the version a
    reader gets always belongs to the params it got.  A tree read while another
    thread swaps must never be a mix of two publications.
    """

    holder = ParamsHolder()
    holder.swap({"version": 0}, 0)
    stop = threading.Event()
    seen: list[tuple[int, int]] = []
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                params, version, applied_at = holder.current()
                seen.append((params["version"], version))
                assert applied_at > 0.0
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for thread in readers:
        thread.start()
    try:
        for version in range(1, 400):
            holder.swap({"version": version}, version)
    finally:
        stop.set()
        for thread in readers:
            thread.join(timeout=10.0)

    assert not errors
    assert seen, "readers never observed the holder"
    assert all(inner == outer for inner, outer in seen)
    assert max(outer for _, outer in seen) >= 1
    assert holder.version == 399


# --------------------------------------------------------------------------- #
# poll interval
# --------------------------------------------------------------------------- #


def test_poll_interval_precedence_and_bad_env_falls_back(caplog):
    assert resolve_poll_interval(env={}) == DEFAULT_POLL_INTERVAL_S == 2.0
    assert resolve_poll_interval(env={POLL_INTERVAL_ENV_VAR: "0.25"}) == 0.25
    assert resolve_poll_interval(5.0, env={POLL_INTERVAL_ENV_VAR: "0.25"}) == 5.0
    with caplog.at_level(logging.WARNING):
        assert resolve_poll_interval(env={POLL_INTERVAL_ENV_VAR: "nope"}) == 2.0
        assert resolve_poll_interval(env={POLL_INTERVAL_ENV_VAR: "-1"}) == 2.0
    assert len(caplog.records) == 2
    with pytest.raises(ValueError):
        resolve_poll_interval(0.0, env={})


def test_client_reads_the_poll_interval_from_the_environment(tmp_path):
    client = ParamsSyncClient(
        LocalDirectoryFetcher(tmp_path),
        str(tmp_path),
        _identity_deserializer,
        env={POLL_INTERVAL_ENV_VAR: "0.05"},
    )
    assert client.poll_interval_s == 0.05


# --------------------------------------------------------------------------- #
# client -- happy path and refusals (jax-free: injected deserializer)
# --------------------------------------------------------------------------- #


def test_happy_path_applies_a_new_version_once(tmp_path):
    export = tmp_path / PARAMS_DIR_NAME
    manifest = _publish(export, 0, learner_step=0)
    fetcher = _CountingFetcher(LocalDirectoryFetcher(export))
    client = _client(export, fetcher=fetcher)

    assert client.applied_version == NO_PARAMS_VERSION
    assert client.poll_once() is True

    params, version, _ = client.holder.current()
    assert version == 0
    assert params == {"payload": _fake_blob(0)}
    assert client.applied_version == 0
    assert client.applied_learner_step == 0
    assert fetcher.file_calls == [str(export / manifest.filename)]

    # A second poll with nothing new must not refetch the blob.
    assert client.poll_once() is False
    assert len(fetcher.file_calls) == 1
    assert client.stats["applied"] == 1
    assert client.stats["unchanged"] == 1


def test_skip_to_newest_fetches_only_the_newest_blob(tmp_path):
    """Three publications land between two polls; only the last is fetched.

    This is the retention interaction too: the learner keeps the last 2 blobs,
    so v3's file is deleted by the time we look.  A client that walked the
    versions in order would fail on a file that is *supposed* to be gone.
    """

    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 3)
    _publish(export, 4)
    (export / blob_filename(3)).unlink()  # retention: keep the last 2
    _publish(export, 5)
    (export / blob_filename(4)).unlink()

    fetcher = _CountingFetcher(LocalDirectoryFetcher(export))
    client = _client(export, fetcher=fetcher)

    assert client.poll_once() is True
    assert client.holder.version == 5
    assert fetcher.file_calls == [str(export / blob_filename(5))]
    assert client.holder.current().params == {"payload": _fake_blob(5)}


def test_sha_mismatch_refuses_and_keeps_the_current_params(tmp_path, caplog):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 1)
    client = _client(export)
    assert client.poll_once() is True

    _publish(export, 2, corrupt_sha=True)
    with caplog.at_level(logging.WARNING):
        assert client.poll_once() is False

    assert client.holder.version == 1
    assert client.holder.current().params == {"payload": _fake_blob(1)}
    assert client.applied_version == 1
    assert client.stats["integrity_errors"] == 1
    assert any("sha256 mismatch" in record.getMessage() for record in caplog.records)
    # The refusal is not sticky: a good v2 later still applies.
    _publish(export, 2)
    assert client.poll_once() is True
    assert client.holder.version == 2


def test_short_blob_refuses_on_size_before_hashing(tmp_path):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 1, corrupt_size=True)
    client = _client(export)

    assert client.poll_once() is False
    assert client.holder.version == NO_PARAMS_VERSION
    assert client.stats["integrity_errors"] == 1


def test_corrupt_blob_that_the_deserializer_rejects_is_refused(tmp_path, caplog):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 1)
    client = _client(export)
    assert client.poll_once() is True

    def explode(blob: bytes):
        raise ParamsLoadError("msgpack: unpack failed")

    _publish(export, 2)
    client._deserializer = explode  # the codec is the injected seam
    with caplog.at_level(logging.WARNING):
        assert client.poll_once() is False

    assert client.holder.version == 1
    assert client.stats["load_errors"] == 1
    assert any("REFUSED" in record.getMessage() for record in caplog.records)


def test_a_deserializer_returning_none_is_refused(tmp_path):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 1)
    client = _client(export, deserializer=lambda blob: None)

    assert client.poll_once() is False
    assert client.holder.version == NO_PARAMS_VERSION
    assert client.stats["load_errors"] == 1


def test_missing_latest_retries_quietly(tmp_path, caplog):
    export = tmp_path / PARAMS_DIR_NAME
    export.mkdir(parents=True)
    client = _client(export)

    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            assert client.poll_once() is False

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], "a not-yet-exported params dir must not warn"
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1, "say it once, then stay quiet"
    assert client.stats["absent"] == 5
    assert client.holder.version == NO_PARAMS_VERSION

    # And it says so once when the export finally appears.
    _publish(export, 0, learner_step=0)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert client.poll_once() is True
    assert any("found" in record.getMessage() for record in caplog.records)


def test_malformed_latest_json_warns_and_keeps_current(tmp_path, caplog):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 1)
    client = _client(export)
    assert client.poll_once() is True

    (export / LATEST_FILENAME).write_text("{ truncated", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert client.poll_once() is False

    assert client.holder.version == 1
    assert client.stats["manifest_errors"] == 1
    assert any("manifest" in record.getMessage() for record in caplog.records)


def test_version_regression_is_refused_loudly(tmp_path, caplog):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 9)
    client = _client(export)
    assert client.poll_once() is True

    # A fresh lineage restarts version numbering under a running proxy.
    _publish(export, 1, learner_step=50)
    with caplog.at_level(logging.WARNING):
        assert client.poll_once() is False

    assert client.holder.version == 9
    assert client.holder.current().params == {"payload": _fake_blob(9)}
    assert client.applied_version == 9
    assert client.stats["regressions"] == 1
    message = " ".join(record.getMessage() for record in caplog.records)
    assert "REFUSING a version regression" in message
    assert "v9" in message and "v1" in message


def test_transport_failure_warning_is_rate_limited(tmp_path, caplog):
    clock = _FakeClock()
    fetcher = _ExplodingFetcher(ParamsTransportError("ssh: connection refused"))
    client = _client(
        tmp_path, fetcher=fetcher, warn_interval_s=60.0, clock=clock
    )

    with caplog.at_level(logging.WARNING):
        for _ in range(30):
            assert client.poll_once() is False
            clock.now += 2.0  # the default poll cadence

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert fetcher.calls == 30
    assert client.stats["transport_errors"] == 30
    # 30 polls span 58 s of fake time: one warning at t=0, one at t>=60.
    assert len(warnings) == 1
    assert "connection refused" in warnings[0].getMessage()

    caplog.clear()
    clock.now += 120.0
    with caplog.at_level(logging.WARNING):
        assert client.poll_once() is False
    assert len(caplog.records) == 1
    assert "suppressed" in caplog.records[0].getMessage()


def test_a_swap_callback_that_raises_leaves_the_version_unapplied(tmp_path):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 4)
    seen: list[int] = []

    def veto(params, version):
        seen.append(version)
        raise RuntimeError("smoke failed")

    client = _client(export, swap_callback=veto)
    assert client.poll_once() is False
    assert client.applied_version == NO_PARAMS_VERSION
    assert client.holder.version == NO_PARAMS_VERSION
    assert client.stats["swap_errors"] == 1

    # Unapplied means the next poll retries the same blob, not skips it.
    assert client.poll_once() is False
    assert seen == [4, 4]


def test_a_custom_swap_callback_owns_publication(tmp_path):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 2)
    holder = ParamsHolder()
    calls: list[tuple[dict, int]] = []

    def publish(params, version):
        calls.append((params, version))
        holder.swap(params, version)

    client = _client(export, holder=holder, swap_callback=publish)
    assert client.poll_once() is True
    assert [version for _, version in calls] == [2]
    assert holder.version == 2


# --------------------------------------------------------------------------- #
# client -- background thread
# --------------------------------------------------------------------------- #


def test_background_thread_applies_and_stops_cleanly(tmp_path):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 1)
    client = _client(export, poll_interval_s=0.01)

    client.start()
    try:
        deadline = time.monotonic() + 10.0
        while client.holder.version != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.holder.version == 1

        _publish(export, 2)
        deadline = time.monotonic() + 10.0
        while client.holder.version != 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.holder.version == 2
        assert client.running is True
    finally:
        client.stop(timeout=10.0)
    assert client.running is False
    assert client.stats["applied"] == 2


def test_background_thread_survives_a_transport_that_always_fails(tmp_path):
    """The thread is the last thing allowed to die: it holds the policy alive."""

    fetcher = _ExplodingFetcher(OSError("nic on fire"))
    client = _client(tmp_path, fetcher=fetcher, poll_interval_s=0.01)

    client.start()
    try:
        deadline = time.monotonic() + 10.0
        while fetcher.calls < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fetcher.calls >= 5
        assert client.running is True
    finally:
        client.stop(timeout=10.0)
    assert client.running is False


def test_start_is_idempotent_and_stop_is_safe_without_start(tmp_path):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 0, learner_step=0)
    client = _client(export, poll_interval_s=0.01)

    client.stop(timeout=1.0)  # never started
    client.start()
    thread = client._thread
    client.start()
    assert client._thread is thread
    client.close()
    assert client.running is False


def test_context_manager_starts_and_closes(tmp_path):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 0, learner_step=0)
    client = _client(export, poll_interval_s=0.01)

    with client as entered:
        assert entered is client
        deadline = time.monotonic() + 10.0
        while client.holder.version != 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert client.running is False
    assert client.holder.version == 0


# --------------------------------------------------------------------------- #
# latency records
# --------------------------------------------------------------------------- #


def test_paramsync_latency_record_carries_the_pinned_fields(tmp_path, monkeypatch):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 6, learner_step=300)
    monkeypatch.setenv("HIL_LATENCY_PROFILE", "1")
    out = tmp_path / "latency"
    profiler = LatencyProfiler.from_env(LATENCY_ROLE, out)
    client = _client(export, profiler=profiler)

    assert client.poll_once() is True
    assert client.poll_once() is False  # unchanged: still emits a sample
    profiler.close()

    lines = [
        json.loads(line)
        for line in profiler.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    applied, unchanged = lines
    assert applied["role"] == LATENCY_ROLE == "paramsync"
    for key in ("poll_ms", "fetch_ms", "load_ms", "swap_ms"):
        assert key in applied, key
        assert applied[key] >= 0.0
    assert applied["version"] == 6
    assert applied["learner_step"] == 300
    assert applied["applied"] is True
    assert applied["staleness_s"] >= 0.0

    # A poll that changed nothing still reports the age of what is served.
    assert unchanged["applied"] is False
    assert unchanged["version"] == 6
    assert "poll_ms" in unchanged
    assert "fetch_ms" not in unchanged
    assert unchanged["staleness_s"] >= applied["staleness_s"]


def test_staleness_tracks_the_manifest_observation_not_the_swap(tmp_path):
    clock = _FakeClock()
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 1)
    client = _client(export, clock=clock)

    assert client.staleness_s() is None
    assert client.poll_once() is True
    assert client.staleness_s() == 0.0

    clock.now += 7.5
    assert client.staleness_s() == 7.5

    # Applying a newer version resets the age to the moment IT was announced.
    _publish(export, 2)
    assert client.poll_once() is True
    assert client.staleness_s() == 0.0
    clock.now += 1.25
    assert client.staleness_s() == 1.25


def test_profiling_disabled_writes_nothing(tmp_path, monkeypatch):
    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 1)
    monkeypatch.delenv("HIL_LATENCY_PROFILE", raising=False)
    client = _client(export)

    assert client.poll_once() is True
    assert client._profiler.enabled is False
    assert list(tmp_path.glob("**/*.jsonl")) == []


# --------------------------------------------------------------------------- #
# transports
# --------------------------------------------------------------------------- #


def test_local_fetcher_maps_missing_files_to_not_found(tmp_path):
    fetcher = LocalDirectoryFetcher(tmp_path)
    with pytest.raises(ParamsNotFoundError):
        fetcher.fetch_text(str(tmp_path / "nope.json"))
    with pytest.raises(ParamsNotFoundError):
        fetcher.fetch_file(str(tmp_path / "nope.bin"), tmp_path / "out.bin")
    (tmp_path / "a.json").write_text("hello", encoding="utf-8")
    assert fetcher.fetch_text("a.json") == "hello"


def test_local_fetcher_refuses_traversal():
    fetcher = LocalDirectoryFetcher("/tmp")
    with pytest.raises(ValueError):
        fetcher.fetch_text("../etc/passwd")
    with pytest.raises(ValueError):
        fetcher.fetch_text("/home/../etc/passwd")


class _FakeSsh:
    """Record argv; return a canned result.  No process is ever spawned."""

    def __init__(self, returncode=0, stdout=b"", stderr=b"", raises=None) -> None:
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._raises = raises

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        self.kwargs.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return subprocess.CompletedProcess(
            argv, self._returncode, self._stdout, self._stderr
        )


def test_ssh_fetcher_multiplexes_and_never_prompts(tmp_path):
    runner = _FakeSsh(stdout=b'{"version": 1}')
    fetcher = SshParamsFetcher(
        "junhyeong_ai", control_dir=tmp_path / "cm", runner=runner, env={}
    )

    assert fetcher.fetch_text("/home/junhyeong/hil-serl-data/runs/r1/LATEST.json") == (
        '{"version": 1}'
    )

    argv = runner.calls[0]
    assert argv[0] == "ssh"
    assert argv[-3:] == ["cat", "--", "/home/junhyeong/hil-serl-data/runs/r1/LATEST.json"]
    assert "junhyeong_ai" in argv
    assert "ControlMaster=auto" in argv
    assert f"ControlPath={os.fspath(tmp_path / 'cm' / 'cm-%r@%h')}" in argv
    assert "ControlPersist=60" in argv
    assert "BatchMode=yes" in argv
    assert "ConnectTimeout=10" in argv
    assert runner.kwargs[0]["stdin"] is subprocess.DEVNULL
    assert runner.kwargs[0]["capture_output"] is True
    assert runner.kwargs[0]["timeout"] == fetcher.text_timeout_s
    # The control directory is created lazily, on first use, and privately.
    assert (tmp_path / "cm").is_dir()
    assert (tmp_path / "cm").stat().st_mode & 0o077 == 0


def test_ssh_fetcher_constructor_touches_nothing(tmp_path):
    control = tmp_path / "never"
    SshParamsFetcher("h", control_dir=control, runner=_FakeSsh(), env={})
    assert not control.exists()


def test_ssh_fetcher_scp_reuses_the_same_control_socket(tmp_path):
    runner = _FakeSsh()
    fetcher = SshParamsFetcher(
        "junhyeong_ai", control_dir=tmp_path / "cm", runner=runner, env={}
    )

    fetcher.fetch_file("/data/runs/r1/params_live/params_v00000001.msgpack", tmp_path / "b")

    argv = runner.calls[0]
    assert argv[0] == "scp"
    assert f"ControlPath={os.fspath(tmp_path / 'cm' / 'cm-%r@%h')}" in argv
    assert argv[-2] == (
        "junhyeong_ai:/data/runs/r1/params_live/params_v00000001.msgpack"
    )
    assert argv[-1] == os.fspath(tmp_path / "b")


def test_ssh_fetcher_reads_host_and_control_dir_from_env(tmp_path):
    fetcher = SshParamsFetcher(
        env={"HIL_SSH_HOST": "other_host", "HIL_PARAMS_SSH_CONTROL_DIR": str(tmp_path)},
        runner=_FakeSsh(),
    )
    assert fetcher.host == "other_host"
    assert fetcher.control_path == os.fspath(tmp_path / "cm-%r@%h")

    default = SshParamsFetcher(env={}, runner=_FakeSsh())
    assert default.host == "junhyeong_ai"


def test_ssh_fetcher_rejects_unsafe_remote_paths(tmp_path):
    fetcher = SshParamsFetcher("h", control_dir=tmp_path, runner=_FakeSsh(), env={})
    for bad in (
        "relative/path",
        "/data/$(rm -rf ~)",
        "/data/a b",
        "/data/../../etc/passwd",
        "/data/x;reboot",
    ):
        with pytest.raises(ValueError):
            fetcher.fetch_text(bad)


def test_ssh_fetcher_distinguishes_missing_file_from_failure(tmp_path):
    missing = SshParamsFetcher(
        "h",
        control_dir=tmp_path,
        runner=_FakeSsh(
            returncode=1, stderr=b"cat: /x/LATEST.json: No such file or directory"
        ),
        env={},
    )
    with pytest.raises(ParamsNotFoundError):
        missing.fetch_text("/x/LATEST.json")

    broken = SshParamsFetcher(
        "h",
        control_dir=tmp_path,
        runner=_FakeSsh(returncode=255, stderr=b"ssh: connect to host h port 22: refused"),
        env={},
    )
    with pytest.raises(ParamsTransportError, match="rc=255"):
        broken.fetch_text("/x/LATEST.json")


def test_ssh_fetcher_maps_timeout_and_exec_failure(tmp_path):
    timed_out = SshParamsFetcher(
        "h",
        control_dir=tmp_path,
        runner=_FakeSsh(raises=subprocess.TimeoutExpired("ssh", 20.0)),
        env={},
    )
    with pytest.raises(ParamsTransportError, match="timed out"):
        timed_out.fetch_text("/x/LATEST.json")

    no_binary = SshParamsFetcher(
        "h",
        control_dir=tmp_path,
        runner=_FakeSsh(raises=FileNotFoundError("ssh")),
        env={},
    )
    with pytest.raises(ParamsTransportError, match="could not start"):
        no_binary.fetch_text("/x/LATEST.json")


def test_ssh_fetcher_warns_once_about_an_oversized_control_path(tmp_path, caplog):
    deep = tmp_path / ("d" * 120)
    fetcher = SshParamsFetcher("h", control_dir=deep, runner=_FakeSsh(), env={})
    with caplog.at_level(logging.WARNING):
        fetcher.fetch_text("/x/a.json")
        fetcher.fetch_text("/x/a.json")
    warnings = [r for r in caplog.records if "ControlPath" in r.getMessage()]
    assert len(warnings) == 1


# --------------------------------------------------------------------------- #
# ControlPath length: the socket ssh binds, not the string we configure
#
# The regression these pin: the default control directory lived in the checkout,
# the guard measured the unexpanded TEMPLATE against a limit that ignored ssh's
# mkstemp suffix, and so every poll died with rc=255 "unix_listener: path too
# long" while the guard said nothing and the proxy blamed HIL_PARAMS_EXPORT.
# --------------------------------------------------------------------------- #


#: The two learner hosts this deployment actually has, as ``ssh -G`` resolves
#: them (measured 2026-08-10).  The aliases are 12 and 4 characters; what ssh
#: puts in the socket name is 24 and 23 -- which is the entire trap.
_REAL_IDENTITIES = {
    "junhyeong_ai": RemoteIdentity("junhyeong", "166.104.146.29", "22"),
    "kanu": RemoteIdentity("junhyeong", "166.104.35.33", "22"),
}

#: The control directory the default used to be, verbatim.
_OLD_DEFAULT_CONTROL_DIR = str(
    _REPO / "ros2_ur_ws" / "gello_logs" / "hil_params_sync"
)


def _dir_with_expanded_length(base: Path, length: int) -> Path:
    """A directory under ``base`` whose ControlPath expands to exactly ``length``."""

    for pad in range(1, 400):
        candidate = base / ("p" * pad)
        expanded = expand_control_path(
            os.fspath(candidate / CONTROL_PATH_BASENAME), _TEST_IDENTITY, host="alias"
        )
        if len(expanded) == length:
            return candidate
    raise AssertionError(f"no directory under {base} expands to {length}")


def test_control_path_budget_is_the_measured_kernel_boundary():
    # sun_path is 108 bytes including the NUL, so 107 characters can be bound.
    assert MAX_UNIX_SOCKET_PATH == 107
    # ssh binds "<ControlPath>.XXXXXXXXXXXXXXXX" -- a dot plus 16 mkstemp
    # characters -- before renaming it into place.
    assert CONTROL_PATH_SUFFIX_ALLOWANCE == 17
    # Verified against the real host on 2026-08-10: 90 binds, 91 fails with
    # unix_listener: path "...cm-junhyeong@166.104.146.29.lQJfUMMSrFNtxTyL"
    # too long for Unix domain socket.
    assert MAX_CONTROL_PATH_LENGTH == 90
    assert MAX_CONTROL_PATH_LENGTH + CONTROL_PATH_SUFFIX_ALLOWANCE == (
        MAX_UNIX_SOCKET_PATH
    )


def test_the_old_guard_measured_the_template_and_the_new_one_measures_the_socket():
    """The exact arithmetic of the outage, reproduced from both directions."""

    identity = _REAL_IDENTITIES["junhyeong_ai"]
    template = os.path.join(_OLD_DEFAULT_CONTROL_DIR, CONTROL_PATH_BASENAME)
    expanded, measured = measure_control_path(
        template, identity, host="junhyeong_ai"
    )

    # What the old guard looked at: a template that fits comfortably ...
    assert len(template) == 75
    assert len(template) <= 100  # the old _CONTROL_PATH_WARN_LENGTH -- silent
    # ... and what ssh actually binds, which does not.
    assert expanded.endswith("/cm-junhyeong@166.104.146.29")
    assert len(expanded) == 94
    assert measured == 94
    assert measured + CONTROL_PATH_SUFFIX_ALLOWANCE == 111 > MAX_UNIX_SOCKET_PATH
    assert measured > MAX_CONTROL_PATH_LENGTH

    # Expansion is not a rounding error: it is 19 characters wider than the
    # template, and every one of them is invisible in the alias an operator
    # types.
    assert len(expanded) - len(template) == 19


def test_default_control_dir_fits_both_real_learner_hosts():
    assert DEFAULT_SSH_CONTROL_DIR == os.path.expanduser("~/.ssh/hil_cm")
    template = os.path.join(DEFAULT_SSH_CONTROL_DIR, CONTROL_PATH_BASENAME)
    for host, identity in _REAL_IDENTITIES.items():
        expanded, measured = measure_control_path(template, identity, host=host)
        assert measured <= MAX_CONTROL_PATH_LENGTH, (host, expanded, measured)
        # Not "fits" by a character: fits with room for a longer $HOME or a
        # future host, so this cannot silently re-break.
        assert MAX_CONTROL_PATH_LENGTH - measured >= 30, (host, measured)
        assert measured + CONTROL_PATH_SUFFIX_ALLOWANCE <= MAX_UNIX_SOCKET_PATH


def test_expand_control_path_expands_the_tokens_ssh_expands():
    identity = RemoteIdentity("junhyeong", "166.104.146.29", "2222")
    assert (
        expand_control_path("/d/cm-%r@%h:%p", identity, host="alias")
        == "/d/cm-junhyeong@166.104.146.29:2222"
    )
    # %n is the alias as typed; %% is a literal percent.
    assert expand_control_path("/d/%n%%", identity, host="alias") == "/d/alias%"
    # %C is ssh's SHA-1 hash: 40 characters wide, and the width is what matters.
    assert len(expand_control_path("/d/%C", identity, host="alias")) == len("/d/") + 40
    # An unknown token stays put rather than vanishing -- dropping it would
    # under-measure, which is the failure direction that caused the outage.
    assert expand_control_path("/d/%Z", identity, host="alias") == "/d/%Z"


def test_ninety_is_accepted_and_ninety_one_falls_back(tmp_path, caplog):
    """The boundary, from both sides, at the exact character."""

    fallback = tmp_path / "fb"
    ok_dir = _dir_with_expanded_length(tmp_path, MAX_CONTROL_PATH_LENGTH)
    over_dir = _dir_with_expanded_length(tmp_path, MAX_CONTROL_PATH_LENGTH + 1)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(params_sync, "fallback_control_dir", lambda: fallback)

        fits = SshParamsFetcher("alias", control_dir=ok_dir, runner=_FakeSsh(), env={})
        with caplog.at_level(logging.WARNING):
            fits.fetch_text("/x/a.json")
        assert fits.control_dir == ok_dir
        assert len(fits.expanded_control_path) == 90
        assert not [r for r in caplog.records if "ControlPath" in r.getMessage()]

        caplog.clear()
        over = SshParamsFetcher("alias", control_dir=over_dir, runner=_FakeSsh(), env={})
        with caplog.at_level(logging.WARNING):
            over.fetch_text("/x/a.json")
        assert over.control_dir == fallback
        messages = [r.getMessage() for r in caplog.records if "ControlPath" in r.getMessage()]
        assert len(messages) == 1
        # The warning names the length that was actually measured, not the
        # template length -- the thing the old warning could never have said.
        assert "91 characters" in messages[0]
        assert str(fallback) in messages[0]
        assert SSH_CONTROL_DIR_ENV_VAR in messages[0]


def test_an_oversized_control_path_moves_the_socket_instead_of_failing(tmp_path):
    fallback = tmp_path / "fb"
    runner = _FakeSsh(stdout=b"{}")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(params_sync, "fallback_control_dir", lambda: fallback)
        fetcher = SshParamsFetcher(
            "alias", control_dir=tmp_path / ("d" * 150), runner=runner, env={}
        )
        fetcher.fetch_text("/x/a.json")

    # The socket moved, the directory is private, and multiplexing survived --
    # the poll never gets the chance to come back rc=255.
    assert fetcher.control_dir == fallback
    assert fallback.is_dir()
    assert fallback.stat().st_mode & 0o077 == 0
    assert fetcher.multiplexing_enabled
    argv = runner.calls[0]
    assert f"ControlPath={os.fspath(fallback / CONTROL_PATH_BASENAME)}" in argv
    assert "ControlMaster=auto" in argv
    # And the first poll already used it: no argv anywhere names the old dir.
    assert not any("d" * 150 in arg for arg in argv)


def test_the_real_fallback_dir_is_short_and_per_user():
    fallback = fallback_control_dir()
    assert fallback.parent == Path(tempfile.gettempdir())
    assert fallback.name == f"hil_cm_{os.geteuid()}"
    for host, identity in _REAL_IDENTITIES.items():
        _, measured = measure_control_path(
            os.fspath(fallback / CONTROL_PATH_BASENAME), identity, host=host
        )
        assert measured <= MAX_CONTROL_PATH_LENGTH, (host, measured)


def test_an_unwritable_control_dir_moves_the_socket_too(tmp_path):
    """Same answer for the other way a socket directory can be unusable."""

    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory", encoding="utf-8")
    fallback = tmp_path / "fb"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(params_sync, "fallback_control_dir", lambda: fallback)
        fetcher = SshParamsFetcher(
            "alias", control_dir=blocker / "cm", runner=_FakeSsh(), env={}
        )
        # It used to raise ParamsTransportError here, every poll, forever.
        fetcher.fetch_text("/x/a.json")
    assert fetcher.control_dir == fallback
    assert fetcher.multiplexing_enabled


def test_multiplexing_switches_off_when_even_the_fallback_is_unusable(tmp_path, caplog):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory", encoding="utf-8")
    runner = _FakeSsh(stdout=b"{}")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(params_sync, "fallback_control_dir", lambda: blocker / "fb")
        fetcher = SshParamsFetcher(
            "alias", control_dir=tmp_path / ("d" * 150), runner=runner, env={}
        )
        with caplog.at_level(logging.WARNING):
            assert fetcher.fetch_text("/x/a.json") == "{}"

    # Degraded, not dead: no ControlPath is offered at all, so ssh cannot fail
    # to bind one, and the fetch still returned the manifest.
    assert not fetcher.multiplexing_enabled
    argv = runner.calls[0]
    assert "ControlMaster=no" in argv
    assert not any(arg.startswith("ControlPath=") for arg in argv)
    assert "BatchMode=yes" in argv
    assert len([r for r in caplog.records if "disabling ControlMaster" in r.getMessage()]) == 1


def test_env_override_wins_verbatim_when_it_fits(tmp_path):
    short = tmp_path / "cm"
    fetcher = SshParamsFetcher(
        env={
            "HIL_SSH_HOST": "alias",
            SSH_CONTROL_DIR_ENV_VAR: str(short),
        },
        runner=_FakeSsh(),
    )
    fetcher.fetch_text("/x/a.json")
    assert fetcher.control_dir == short
    assert fetcher.control_path == os.fspath(short / CONTROL_PATH_BASENAME)


def test_env_override_that_is_too_long_still_falls_back(tmp_path, caplog):
    """An operator's own directory gets the same treatment, loudly.

    Honouring it verbatim would be honouring a request for rc=255 on every
    poll.  The warning names the env var, so the operator can see whose
    setting moved and where to fix it.
    """

    fallback = tmp_path / "fb"
    too_long = tmp_path / ("o" * 150)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(params_sync, "fallback_control_dir", lambda: fallback)
        fetcher = SshParamsFetcher(
            env={"HIL_SSH_HOST": "alias", SSH_CONTROL_DIR_ENV_VAR: str(too_long)},
            runner=_FakeSsh(),
        )
        with caplog.at_level(logging.WARNING):
            fetcher.fetch_text("/x/a.json")

    assert fetcher.control_dir == fallback
    assert fetcher.multiplexing_enabled
    assert not (too_long).exists()
    message = [r.getMessage() for r in caplog.records if "ControlPath" in r.getMessage()]
    assert len(message) == 1
    assert SSH_CONTROL_DIR_ENV_VAR in message[0]


def test_resolve_remote_identity_reads_ssh_dash_g(tmp_path):
    runner = _FakeSsh(
        stdout=b"host junhyeong_ai\nuser junhyeong\nhostname 166.104.146.29\nport 22\n"
    )
    identity = resolve_remote_identity(
        "junhyeong_ai", runner=runner, use_cache=False
    )
    assert identity == RemoteIdentity("junhyeong", "166.104.146.29", "22", True)
    assert runner.calls[0] == ["ssh", "-G", "--", "junhyeong_ai"]
    assert runner.kwargs[0]["stdin"] is subprocess.DEVNULL


@pytest.mark.parametrize(
    "runner",
    [
        _FakeSsh(raises=FileNotFoundError("ssh")),
        _FakeSsh(raises=subprocess.TimeoutExpired("ssh", 5.0)),
        _FakeSsh(returncode=255, stderr=b"nope"),
        _FakeSsh(stdout=b"garbage without a hostname\n"),
    ],
)
def test_resolve_remote_identity_never_raises(runner):
    identity = resolve_remote_identity("some_alias", runner=runner, use_cache=False)
    assert identity.resolved is False
    assert identity.hostname == "some_alias"


def test_an_unresolved_identity_is_never_credited_with_being_narrow():
    """A guess must not talk the guard into approving a path that will not bind.

    The guess is built from the alias and the LOCAL username, so it is usually
    shorter than the truth -- and an optimistic guard is precisely how the
    original outage got through.
    """

    guess = RemoteIdentity("a", "b", "22", resolved=False)
    template = "/d/" + CONTROL_PATH_BASENAME
    expanded, measured = measure_control_path(template, guess, host="b")
    assert expanded == "/d/cm-a@b"
    # Measured as if %r@%h were as wide as the widest identity we really have.
    assert measured == len(expanded) + (UNRESOLVED_EXPANSION_FLOOR - len("a@b"))
    assert UNRESOLVED_EXPANSION_FLOOR == len("junhyeong@166.104.146.29")

    # A resolved identity of the same width is measured at face value, and a
    # template with no identity token is never padded.
    assert measure_control_path(template, RemoteIdentity("a", "b", "22"), host="b")[1] == (
        len(expanded)
    )
    assert measure_control_path("/d/cm", guess, host="b")[1] == len("/d/cm")


def test_the_identity_is_resolved_once_per_fetcher(tmp_path):
    calls = []

    def _resolver(host, **kwargs):
        calls.append(host)
        return _TEST_IDENTITY

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(params_sync, "resolve_remote_identity", _resolver)
        fetcher = SshParamsFetcher(
            "alias", control_dir=tmp_path / "cm", runner=_FakeSsh(), env={}
        )
        # The constructor runs no subprocess: measurement is first-poll work.
        assert calls == []
        fetcher.fetch_text("/x/a.json")
        fetcher.fetch_text("/x/a.json")
        fetcher.fetch_file("/x/b.bin", tmp_path / "b")
    assert calls == ["alias"]


def test_an_injected_identity_skips_resolution_entirely(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            params_sync,
            "resolve_remote_identity",
            lambda host, **kwargs: pytest.fail("must not resolve"),
        )
        fetcher = SshParamsFetcher(
            "alias",
            control_dir=tmp_path / "cm",
            identity=_REAL_IDENTITIES["kanu"],
            runner=_FakeSsh(),
            env={},
        )
        assert fetcher.expanded_control_path.endswith("cm-junhyeong@166.104.35.33")


# --------------------------------------------------------------------------- #
# prune / graft (structure only -- no serialization, so no flax needed)
# --------------------------------------------------------------------------- #


def _reference_tree() -> dict:
    """A miniature of the production tree: one shared trunk, several heads."""

    return {
        "modules_actor": {
            "encoder": {
                "encoder_cam1": {
                    FROZEN_TRUNK_SUBTREE_KEY: {
                        "conv_init": {"kernel": np.full((2, 2), 9.0, dtype=np.float32)},
                        "norm": {"scale": np.full((2,), 8.0, dtype=np.float32)},
                    },
                    "Dense_0": {"kernel": np.zeros((2, 3), dtype=np.float32)},
                },
                "encoder_cam2": {
                    "Dense_0": {"kernel": np.zeros((2, 3), dtype=np.float32)}
                },
            },
            "Dense_0": {"bias": np.zeros((3,), dtype=np.float32)},
        },
        "modules_temperature": {
            "lagrange": {"value": np.zeros((), dtype=np.float32)}
        },
    }


def test_prune_removes_the_trunk_and_shares_every_other_leaf():
    reference = _reference_tree()
    pruned = prune_frozen_trunk(reference)

    cam1 = pruned["modules_actor"]["encoder"]["encoder_cam1"]
    assert FROZEN_TRUNK_SUBTREE_KEY not in cam1
    assert set(cam1) == {"Dense_0"}
    # Containers are rebuilt; leaves are the SAME objects (a 32 MB tree must not
    # be copied to prune 61% of it).
    assert cam1 is not reference["modules_actor"]["encoder"]["encoder_cam1"]
    assert (
        cam1["Dense_0"]["kernel"]
        is reference["modules_actor"]["encoder"]["encoder_cam1"]["Dense_0"]["kernel"]
    )
    assert pruned["modules_temperature"]["lagrange"]["value"] is (
        reference["modules_temperature"]["lagrange"]["value"]
    )


def test_graft_restores_the_local_trunk_and_takes_everything_else_from_the_wire():
    reference = _reference_tree()
    pruned = prune_frozen_trunk(reference)
    # The wire tree carries updated trainable weights.
    pruned["modules_actor"]["encoder"]["encoder_cam1"]["Dense_0"]["kernel"] = np.ones(
        (2, 3), dtype=np.float32
    )

    grafted = graft_frozen_trunk(pruned, reference)

    trunk = grafted["modules_actor"]["encoder"]["encoder_cam1"][FROZEN_TRUNK_SUBTREE_KEY]
    assert trunk is reference["modules_actor"]["encoder"]["encoder_cam1"][
        FROZEN_TRUNK_SUBTREE_KEY
    ]
    assert np.array_equal(trunk["conv_init"]["kernel"], np.full((2, 2), 9.0))
    assert np.array_equal(
        grafted["modules_actor"]["encoder"]["encoder_cam1"]["Dense_0"]["kernel"],
        np.ones((2, 3), dtype=np.float32),
    )


def test_graft_names_the_path_of_any_structural_drift():
    reference = _reference_tree()
    pruned = prune_frozen_trunk(reference)

    missing = prune_frozen_trunk(reference)
    del missing["modules_temperature"]
    with pytest.raises(ParamsLoadError, match="missing modules_temperature"):
        graft_frozen_trunk(missing, reference)

    extra = prune_frozen_trunk(reference)
    extra["modules_surprise"] = {"kernel": np.zeros((1,), dtype=np.float32)}
    with pytest.raises(ParamsLoadError, match="unexpected keys"):
        graft_frozen_trunk(extra, reference)

    unpruned = prune_frozen_trunk(reference)
    unpruned["modules_actor"]["encoder"]["encoder_cam1"][FROZEN_TRUNK_SUBTREE_KEY] = {}
    with pytest.raises(ParamsLoadError, match="must prune it"):
        graft_frozen_trunk(unpruned, reference)

    assert graft_frozen_trunk(pruned, reference) is not None


# --------------------------------------------------------------------------- #
# codec -- needs real flax serialization
# --------------------------------------------------------------------------- #


def _codec_module():
    pytest.importorskip("flax", reason="the params codec needs flax + jax")
    pytest.importorskip("jax", reason="the params codec needs flax + jax")
    from ur_env.local_policy import params_sync

    return params_sync


def test_codec_round_trips_a_real_flax_blob_and_grafts_the_local_trunk():
    module = _codec_module()
    reference = _reference_tree()
    published = _reference_tree()
    published["modules_actor"]["encoder"]["encoder_cam1"]["Dense_0"]["kernel"] = (
        np.full((2, 3), 3.5, dtype=np.float32)
    )
    # The learner's trunk is irrelevant: it is never serialized.
    published["modules_actor"]["encoder"]["encoder_cam1"][FROZEN_TRUNK_SUBTREE_KEY][
        "norm"
    ]["scale"] = np.full((2,), -1.0, dtype=np.float32)

    blob = module.serialize_trainable_params(published)
    codec = module.FrozenTrunkParamsCodec(reference)
    restored = codec.load(blob)

    assert np.array_equal(
        restored["modules_actor"]["encoder"]["encoder_cam1"]["Dense_0"]["kernel"],
        np.full((2, 3), 3.5, dtype=np.float32),
    )
    trunk = restored["modules_actor"]["encoder"]["encoder_cam1"][
        FROZEN_TRUNK_SUBTREE_KEY
    ]
    assert np.array_equal(trunk["norm"]["scale"], np.full((2,), 8.0))


def _blob_keys(blob: bytes) -> list[str]:
    from flax import serialization

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from walk(value, f"{prefix}/{key}")
        else:
            yield prefix

    return list(walk(serialization.msgpack_restore(blob)))


def test_serialized_blob_omits_the_trunk_entirely():
    """The whole point of the wire format: the frozen 61% is not on it."""

    module = _codec_module()
    from flax import serialization

    reference = _reference_tree()

    blob = module.serialize_trainable_params(reference)

    assert all(FROZEN_TRUNK_SUBTREE_KEY not in key for key in _blob_keys(blob))
    assert len(blob) < len(serialization.to_bytes(reference))


def test_codec_refuses_a_blob_whose_leaf_shape_drifted():
    module = _codec_module()
    reference = _reference_tree()
    drifted = _reference_tree()
    drifted["modules_actor"]["encoder"]["encoder_cam1"]["Dense_0"]["kernel"] = np.zeros(
        (2, 4), dtype=np.float32
    )

    codec = module.FrozenTrunkParamsCodec(reference)
    with pytest.raises(ParamsLoadError, match="validation"):
        codec.load(module.serialize_trainable_params(drifted))


def test_codec_refuses_a_blob_with_a_different_key_set():
    module = _codec_module()
    reference = _reference_tree()
    renamed = _reference_tree()
    renamed["modules_actor"]["encoder"]["encoder_cam3"] = renamed[
        "modules_actor"
    ]["encoder"].pop("encoder_cam2")

    codec = module.FrozenTrunkParamsCodec(reference)
    with pytest.raises(ParamsLoadError, match="deserialize"):
        codec.load(module.serialize_trainable_params(renamed))


def test_codec_refuses_random_bytes():
    module = _codec_module()
    codec = module.FrozenTrunkParamsCodec(_reference_tree())

    with pytest.raises(ParamsLoadError, match="deserialize"):
        codec.load(b"\x00\x01\x02not-msgpack")


def test_codec_refuses_a_non_finite_leaf():
    module = _codec_module()
    reference = _reference_tree()
    poisoned = _reference_tree()
    poisoned["modules_actor"]["Dense_0"]["bias"] = np.array(
        [0.0, np.nan, 0.0], dtype=np.float32
    )

    codec = module.FrozenTrunkParamsCodec(reference)
    with pytest.raises(ParamsLoadError, match="non-finite"):
        codec.load(module.serialize_trainable_params(poisoned))


def test_client_end_to_end_with_the_real_codec(tmp_path):
    """The whole path: publish real bytes, poll, graft, validate, swap."""

    module = _codec_module()
    reference = _reference_tree()
    export = tmp_path / PARAMS_DIR_NAME

    published = _reference_tree()
    published["modules_actor"]["Dense_0"]["bias"] = np.full(
        (3,), 2.0, dtype=np.float32
    )
    blob = module.serialize_trainable_params(published)
    _publish(export, 1, blob=blob)

    codec = module.FrozenTrunkParamsCodec(reference)
    client = _client(export, deserializer=codec.load)

    assert client.poll_once() is True
    params, version, _ = client.holder.current()
    assert version == 1
    assert np.array_equal(
        params["modules_actor"]["Dense_0"]["bias"], np.full((3,), 2.0)
    )
    assert params["modules_actor"]["encoder"]["encoder_cam1"][
        FROZEN_TRUNK_SUBTREE_KEY
    ] is reference["modules_actor"]["encoder"]["encoder_cam1"][
        FROZEN_TRUNK_SUBTREE_KEY
    ]

    # A corrupt v2 must not disturb the v1 that is being served.
    _publish(export, 2, blob=b"\x00" * 32)
    assert client.poll_once() is False
    assert client.holder.version == 1
    assert np.array_equal(
        client.holder.current().params["modules_actor"]["Dense_0"]["bias"],
        np.full((3,), 2.0),
    )


# --------------------------------------------------------------------------- #
# opt-in: the production agent, the real ResNet-10 asset, real sizes
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.environ.get("RUN_HIL_SERL_ACTUAL_PARAMS_SYNC") != "1",
    reason="set RUN_HIL_SERL_ACTUAL_PARAMS_SYNC=1 to build the real SAC agent",
)
def test_actual_agent_blob_is_trunk_free_and_restores_bit_exactly(tmp_path):
    module = _codec_module()
    import jax

    from ur_env.learner.frozen_trunk import create_frozen_trunk_feature_agent

    agent = create_frozen_trunk_feature_agent(
        resnet_source_path=(
            _REPO / "third_party" / "hil-serl" / "examples" / "experiments"
            / "resnet10_params.pkl"
        ),
        resnet_cache_path=tmp_path / "verified-resnet10.pkl",
    )
    codec = module.FrozenTrunkParamsCodec.from_agent(
        agent,
        resnet_source_path=(
            _REPO / "third_party" / "hil-serl" / "examples" / "experiments"
            / "resnet10_params.pkl"
        ),
        resnet_cache_path=tmp_path / "verified-resnet10.pkl",
    )

    blob = module.serialize_trainable_params(agent.state.params)
    # The measured production numbers: 32,006,980 B total, 19,623,168 B trunk.
    assert len(blob) < 32_006_980 - 19_000_000

    export = tmp_path / PARAMS_DIR_NAME
    _publish(export, 3, blob=blob)
    client = _client(export, deserializer=codec.load)
    assert client.poll_once() is True

    restored = client.holder.current().params
    equal = jax.tree_util.tree_all(
        jax.tree_util.tree_map(np.array_equal, restored, agent.state.params)
    )
    assert bool(equal)
