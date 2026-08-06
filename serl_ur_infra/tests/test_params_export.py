"""The learner's trainable-params export: what lands on disk, and what never does.

This sink exists so laptop3 can run policy inference locally while the GPU
server keeps training, reward authority and replay.  Three properties decide
whether that is safe, and each one is a section below:

1. **It is trainable-only.**  61 % of the parameter tree is the frozen
   ResNet-10 trunk, byte-identical on every publish and already SHA-verified on
   laptop3.  Shipping it would saturate a 3.6 MB/s link with a constant.  The
   real-agent test measures the prune against the numbers the feasibility study
   established (32,006,980 B total / 19,623,168 B trunk / 12,383,812 B
   remainder) rather than trusting a shape check.
2. **It cannot end training.**  ``export`` runs on the learner thread inside
   ``train_once``, whose ``except`` turns any exception into a permanent learner
   fault.  A full disk on the server must cost a warning, not a session.
3. **A reader never sees half a file.**  The consumer is a remote poller that
   reads at arbitrary instants, so the ordering test below observes the
   directory *from inside* ``os.replace`` and asserts what was visible then.

Most of this runs in the actor venv, which has NEITHER jax NOR flax: the module
keeps flax behind a lazy, injectable serializer precisely so the parts that can
silently go wrong -- pruning, atomicity, retention, degradation -- are pinned on
an interpreter where an import error cannot hide behind a skip.  The real
serialization/round-trip tests import flax and skip cleanly without it; the
full production agent is behind ``RUN_HIL_SERL_ACTUAL_FEATURE_AGENT=1`` like
every other test that builds it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_INFRA, ".."))
sys.path.insert(0, _INFRA)
sys.path.insert(0, _HERE)

from ur_env.learner.params_export import (  # noqa: E402
    BLOB_GLOB,
    DIRECTORY_NAME,
    ENABLE_ENV_VAR,
    FROZEN_TRUNK_EXPECTED_PATH,
    FROZEN_TRUNK_LEAF_KEY,
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    RETAINED_BLOBS,
    ParamsExporter,
    blob_name,
    existing_versions,
    is_params_export_enabled,
    prune_frozen_trunk,
    read_manifest,
)


# ---------------------------------------------------------------------------
# A tree shaped like the production one, small enough to have no jax in it.
# ---------------------------------------------------------------------------


def _trunk() -> dict:
    return {
        "conv_init": {"kernel": np.arange(12, dtype=np.float32).reshape(3, 4)},
        "norm_init": {"scale": np.ones(4, dtype=np.float32)},
    }


def _fake_params() -> dict:
    """Mirror the production layout: ONE trunk, under encoder_cam1 only."""

    return {
        "modules_actor": {
            "encoder": {
                "encoder_cam1": {
                    FROZEN_TRUNK_LEAF_KEY: _trunk(),
                    "SpatialLearnedEmbeddings_0": {
                        "kernel": np.full((2, 3), 0.5, dtype=np.float32)
                    },
                },
                "encoder_cam2": {
                    "SpatialLearnedEmbeddings_0": {
                        "kernel": np.full((2, 3), 0.25, dtype=np.float32)
                    }
                },
                "Dense_0": {"kernel": np.zeros((3, 3), dtype=np.float32)},
            },
            "network": {"Dense_0": {"bias": np.zeros(3, dtype=np.float32)}},
        },
        "modules_critic": {"network": {"Dense_0": {"bias": np.ones(3, np.float32)}}},
        "modules_temperature": {"lagrange": np.float32(0.01)},
    }


def _fake_serializer(tree) -> bytes:
    """Deterministic stand-in for ``flax.serialization.to_bytes``.

    Only ordering and content matter to the tests that use it; the real
    serializer is exercised by the flax-gated tests further down.
    """

    def walk(node, prefix: str, out: list[bytes]) -> None:
        if isinstance(node, dict):
            for key in node:
                walk(node[key], f"{prefix}/{key}", out)
            return
        array = np.asarray(node)
        out.append(prefix.encode("utf-8"))
        out.append(str(array.dtype).encode("utf-8"))
        out.append(array.tobytes())

    chunks: list[bytes] = []
    walk(tree, "", chunks)
    return b"\x00".join(chunks)


def _exporter(tmp_path: Path, **overrides) -> ParamsExporter:
    options = dict(serializer=_fake_serializer)
    options.update(overrides)
    return ParamsExporter(tmp_path / DIRECTORY_NAME, **options)


# ---------------------------------------------------------------------------
# 1. Trainable-only: what the prune removes and what it must not.
# ---------------------------------------------------------------------------


def test_prune_removes_the_trunk_and_keeps_every_sibling():
    tree, pruned = prune_frozen_trunk(_fake_params())

    assert pruned == (FROZEN_TRUNK_EXPECTED_PATH,)
    cam1 = tree["modules_actor"]["encoder"]["encoder_cam1"]
    assert FROZEN_TRUNK_LEAF_KEY not in cam1
    assert set(cam1) == {"SpatialLearnedEmbeddings_0"}
    # Everything the learner actually trains survives, including the head that
    # lives in the same dict as the trunk.
    assert set(tree) == {"modules_actor", "modules_critic", "modules_temperature"}
    assert set(tree["modules_actor"]["encoder"]) == {
        "encoder_cam1",
        "encoder_cam2",
        "Dense_0",
    }


def test_prune_finds_the_trunk_wherever_flax_put_it():
    """The key is the contract, not the path.

    Flax stores a shared submodule under its FIRST user, so a rebuild that
    changed construction order would move the one trunk copy.  A path-based
    prune would then ship 19.6 MB of frozen weights with no error anywhere.
    """

    params = {"modules_critic": {"encoder": {"encoder_cam2": {
        FROZEN_TRUNK_LEAF_KEY: _trunk(),
        "Dense_0": {"kernel": np.zeros((2, 2), np.float32)},
    }}}}
    tree, pruned = prune_frozen_trunk(params)

    assert pruned == (("modules_critic", "encoder", "encoder_cam2",
                       FROZEN_TRUNK_LEAF_KEY),)
    assert set(tree["modules_critic"]["encoder"]["encoder_cam2"]) == {"Dense_0"}


def test_prune_shares_leaves_rather_than_copying_twelve_megabytes():
    params = _fake_params()
    tree, _ = prune_frozen_trunk(params)

    assert (
        tree["modules_actor"]["network"]["Dense_0"]["bias"]
        is params["modules_actor"]["network"]["Dense_0"]["bias"]
    )


def test_a_tree_without_a_trunk_is_refused_rather_than_shipped(tmp_path, caplog):
    """A blob the peer cannot graft a trunk into is worse than no blob."""

    exporter = _exporter(tmp_path)
    with caplog.at_level("WARNING"):
        assert exporter.export({"a": np.zeros(3)}, learner_step=0, version=0) is None

    assert exporter.degraded and not exporter.enabled
    assert not (tmp_path / DIRECTORY_NAME).exists()
    assert any("pretrained_encoder" in record.message for record in caplog.records)


def test_a_relocated_trunk_warns_but_still_exports(tmp_path, caplog):
    params = {"modules_critic": {FROZEN_TRUNK_LEAF_KEY: _trunk(),
                                 "Dense_0": {"b": np.zeros(2, np.float32)}}}
    exporter = _exporter(tmp_path)
    with caplog.at_level("WARNING"):
        record = exporter.export(params, learner_step=0, version=0)

    assert record is not None and record.path.is_file()
    assert not exporter.degraded
    assert any("frozen trunk is not at" in item.message for item in caplog.records)


# ---------------------------------------------------------------------------
# 2. The manifest: the only thing a poller reads before it knows a version.
# ---------------------------------------------------------------------------


def test_manifest_names_the_blob_and_its_sha(tmp_path):
    exporter = _exporter(tmp_path)
    record = exporter.export(_fake_params(), learner_step=250, version=5)

    assert record is not None
    directory = tmp_path / DIRECTORY_NAME
    assert record.path == directory / "params_v00000005.msgpack"
    payload = record.path.read_bytes()

    manifest = read_manifest(directory)
    assert manifest == {
        "version": 5,
        "learner_step": 250,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "filename": "params_v00000005.msgpack",
        "schema": MANIFEST_SCHEMA,
    }
    assert manifest["sha256"] == record.sha256
    assert manifest["bytes"] == record.bytes == len(payload)
    assert exporter.last_export == record
    assert exporter.export_count == 1


def test_blob_name_is_zero_padded_so_lexical_order_is_numeric():
    assert blob_name(0) == "params_v00000000.msgpack"
    assert sorted([blob_name(9), blob_name(10)]) == [blob_name(9), blob_name(10)]


def test_read_manifest_treats_missing_and_corrupt_alike(tmp_path):
    assert read_manifest(tmp_path) is None
    (tmp_path / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    assert read_manifest(tmp_path) is None
    (tmp_path / MANIFEST_NAME).write_text("[1, 2]", encoding="utf-8")
    assert read_manifest(tmp_path) is None


# ---------------------------------------------------------------------------
# 3. Atomicity: observed from inside the rename, not inferred from the result.
# ---------------------------------------------------------------------------


def test_a_reader_never_sees_a_partial_blob_or_a_manifest_without_one(
    tmp_path, monkeypatch
):
    directory = tmp_path / DIRECTORY_NAME
    observations: list[tuple[str, list[str], bool]] = []
    real_replace = os.replace

    def observing_replace(source, destination):
        # What a poller globbing for blobs would see at this instant, plus
        # whether the manifest it might read already names a file that exists.
        blobs = sorted(item.name for item in directory.glob(BLOB_GLOB))
        manifest = read_manifest(directory)
        named_exists = bool(
            manifest and (directory / manifest["filename"]).is_file()
        )
        observations.append((Path(destination).name, blobs, named_exists))
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", observing_replace)
    exporter = _exporter(tmp_path)
    record = exporter.export(_fake_params(), learner_step=50, version=1)
    assert record is not None

    assert [item[0] for item in observations] == [
        "params_v00000001.msgpack",
        MANIFEST_NAME,
    ]
    # Before the blob rename the directory holds no blob at all -- the
    # temporary is dotted and cannot match the glob a reader uses.
    assert observations[0][1] == []
    # The manifest is written only after the blob it names has landed whole.
    assert observations[1][1] == ["params_v00000001.msgpack"]
    assert not observations[0][2]

    manifest = read_manifest(directory)
    assert manifest is not None
    assert (directory / manifest["filename"]).is_file()
    # No temporary survives a successful export.
    assert sorted(item.name for item in directory.iterdir()) == [
        MANIFEST_NAME,
        "params_v00000001.msgpack",
    ]


def test_a_failed_write_leaves_no_temporary_behind(tmp_path, monkeypatch):
    directory = tmp_path / DIRECTORY_NAME
    exporter = _exporter(tmp_path)
    assert exporter.export(_fake_params(), learner_step=0, version=0) is not None

    def exploding_replace(source, destination):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", exploding_replace)
    assert exporter.export(_fake_params(), learner_step=50, version=1) is None
    assert exporter.degraded
    assert sorted(item.name for item in directory.iterdir()) == [
        MANIFEST_NAME,
        "params_v00000000.msgpack",
    ]


# ---------------------------------------------------------------------------
# 4. Retention, monotonicity.
# ---------------------------------------------------------------------------


def test_retention_keeps_the_last_two_versions(tmp_path):
    exporter = _exporter(tmp_path)
    for version in range(5):
        exporter.export(_fake_params(), learner_step=50 * version, version=version)

    directory = tmp_path / DIRECTORY_NAME
    assert RETAINED_BLOBS == 2
    assert existing_versions(directory) == (3, 4)
    assert read_manifest(directory)["version"] == 4
    # Two, not one: a poller that has decided to fetch N must still find N on
    # disk after the learner has published N+1.
    assert (directory / blob_name(3)).is_file()


def test_retention_is_configurable_and_never_deletes_the_newest(tmp_path):
    exporter = _exporter(tmp_path, retain=1)
    for version in range(3):
        exporter.export(_fake_params(), learner_step=50 * version, version=version)
    assert existing_versions(tmp_path / DIRECTORY_NAME) == (2,)

    with pytest.raises(ValueError, match="retain"):
        ParamsExporter(tmp_path, retain=0)


def test_a_version_that_went_backwards_is_ignored_not_published(tmp_path):
    exporter = _exporter(tmp_path)
    exporter.export(_fake_params(), learner_step=100, version=2)

    assert exporter.export(_fake_params(), learner_step=50, version=1) is None
    directory = tmp_path / DIRECTORY_NAME
    assert read_manifest(directory)["version"] == 2
    assert existing_versions(directory) == (2,)
    # Refusing a regression is not a failure: the sink stays live.
    assert exporter.enabled and not exporter.degraded


# ---------------------------------------------------------------------------
# 5. Disabled and degraded: the two states that must cost nothing.
# ---------------------------------------------------------------------------


def test_disabled_exporter_touches_no_filesystem(tmp_path):
    exporter = ParamsExporter.from_env(
        tmp_path / DIRECTORY_NAME, env={}, serializer=_fake_serializer
    )

    assert not exporter.enabled
    assert exporter.directory is None and exporter.manifest_path is None
    assert exporter.export(_fake_params(), learner_step=0, version=0) is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_env_gate_accepts_the_same_spellings_as_the_latency_profiler(value):
    assert is_params_export_enabled({ENABLE_ENV_VAR: value})


@pytest.mark.parametrize("value", ["", "0", "no", "off", "maybe"])
def test_env_gate_rejects_everything_else(value):
    assert not is_params_export_enabled({ENABLE_ENV_VAR: value})


def test_from_env_reads_the_process_environment_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    assert not ParamsExporter.from_env(tmp_path).enabled
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    exporter = ParamsExporter.from_env(tmp_path)
    assert exporter.enabled and exporter.directory == tmp_path


def test_the_constructor_touches_no_filesystem(tmp_path):
    """A typo'd path must surface as one warning, never as a refused session."""

    ParamsExporter(tmp_path / "nested" / DIRECTORY_NAME, serializer=_fake_serializer)
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_destination_degrades_once_and_never_raises(tmp_path, caplog):
    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not a directory", encoding="utf-8")
    exporter = ParamsExporter(blocked / DIRECTORY_NAME, serializer=_fake_serializer)

    with caplog.at_level("WARNING"):
        for version in range(3):
            assert (
                exporter.export(_fake_params(), learner_step=0, version=version)
                is None
            )

    assert exporter.degraded and not exporter.enabled
    warnings = [item for item in caplog.records if item.levelname == "WARNING"]
    assert len(warnings) == 1, [item.message for item in warnings]
    assert "disabling params export" in warnings[0].message


def test_a_serializer_that_explodes_degrades_the_sink_not_the_caller(tmp_path):
    def boom(tree):
        raise RuntimeError("device transfer failed")

    exporter = ParamsExporter(tmp_path / DIRECTORY_NAME, serializer=boom)
    assert exporter.export(_fake_params(), learner_step=0, version=0) is None
    assert exporter.degraded
    assert not (tmp_path / DIRECTORY_NAME).exists()


def test_a_serializer_returning_non_bytes_is_refused(tmp_path):
    exporter = ParamsExporter(
        tmp_path / DIRECTORY_NAME, serializer=lambda tree: "not bytes"
    )
    assert exporter.export(_fake_params(), learner_step=0, version=0) is None
    assert exporter.degraded


# ---------------------------------------------------------------------------
# 6. The learner hook: at every publish, and once at ready.
# ---------------------------------------------------------------------------


class _RecordingExporter:
    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def export(self, params, *, learner_step, version):
        self.calls.append((learner_step, version))
        return None


def _jax_learner(exporter=None):
    """A real ``HILSERLLearner`` over a toy agent whose tree HAS a trunk.

    The toy agent from ``test_learner_policy_checkpoint`` publishes a one-leaf
    tree, which the exporter would rightly refuse; this one keeps the
    production shape (a ``pretrained_encoder`` subtree the update never
    touches, plus a trained leaf) so the hook is exercised against something
    the exporter would actually accept.
    """

    from dataclasses import dataclass, replace

    import flax
    import jax
    import jax.numpy as jnp
    from flax.core import freeze

    from test_learner_policy_checkpoint import _sample_action, _sampler
    from ur_env.learner import (
        HILSERLLearner,
        LearnerConfig,
        VersionedPolicyRuntime,
    )

    @flax.struct.dataclass
    class _State:
        params: object
        target_params: object
        opt_state: object
        rng: object
        step: int

    def _params(value):
        return freeze(
            {
                "weight": jnp.array([value], dtype=jnp.float32),
                "modules_actor": {
                    "encoder": {
                        "encoder_cam1": {
                            FROZEN_TRUNK_LEAF_KEY: {
                                "kernel": jnp.ones((2, 2), dtype=jnp.float32)
                            },
                            "Dense_0": {
                                "kernel": jnp.full((2, 2), value, dtype=jnp.float32)
                            },
                        }
                    }
                },
            }
        )

    @dataclass(frozen=True)
    class _Agent:
        state: _State

        def replace(self, **updates):
            return replace(self, **updates)

        def update(self, batch, *, networks_to_update):
            del batch, networks_to_update
            next_step = int(self.state.step) + 1
            value = float(self.state.params["weight"][0]) + 0.1
            params = _params(value)
            return (
                self.replace(
                    state=self.state.replace(
                        params=params,
                        target_params=params,
                        opt_state=freeze({"moment": jnp.zeros(1, jnp.float32)}),
                        rng=jax.random.fold_in(self.state.rng, next_step),
                        step=next_step,
                    )
                ),
                {"loss": jnp.float32(value)},
            )

    initial = _params(0.0)
    agent = _Agent(
        _State(
            params=initial,
            target_params=initial,
            opt_state=freeze({"moment": jnp.zeros(1, jnp.float32)}),
            rng=jax.random.PRNGKey(9),
            step=0,
        )
    )
    config = LearnerConfig(
        batch_size=4, training_starts=4, publish_period=1, checkpoint_period=1_000
    )
    return HILSERLLearner(
        agent=agent,
        sampler=_sampler(),
        publisher=VersionedPolicyRuntime(agent, sample_action=_sample_action),
        config=config,
        params_exporter=exporter,
    )


def test_learner_exports_at_every_publish_and_at_attach(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    recorder = _RecordingExporter()
    learner = _jax_learner()

    assert learner.attach_params_exporter(recorder) is None
    # Version 0 before a single gradient step: the peer needs a blob to load at
    # ITS startup, and the first publish is 50 steps away on an empty replay.
    assert recorder.calls == [(0, 0)]

    learner.train_once()
    learner.train_once()
    assert recorder.calls == [(0, 0), (1, 1), (2, 2)]
    assert learner.policy_version == 2


def test_learner_without_an_exporter_behaves_exactly_as_before(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    learner = _jax_learner()
    result = learner.train_once()
    assert result.published and result.policy_version == 1


def test_a_real_export_reaches_disk_from_the_publish_site(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    exporter = ParamsExporter(tmp_path / DIRECTORY_NAME)
    learner = _jax_learner(exporter=exporter)
    learner.attach_params_exporter(exporter)
    learner.train_once()

    directory = tmp_path / DIRECTORY_NAME
    assert existing_versions(directory) == (0, 1)
    manifest = read_manifest(directory)
    assert manifest["version"] == 1 and manifest["learner_step"] == 1
    blob = (directory / manifest["filename"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == manifest["sha256"]
    # The trunk is not in there: 2x2 float32 of ones would show up verbatim.
    assert np.ones((2, 2), dtype=np.float32).tobytes() not in blob


def test_a_degraded_exporter_does_not_fault_the_learner(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    class _Broken:
        def export(self, params, *, learner_step, version):
            raise AssertionError("ParamsExporter.export must never raise")

    # A real exporter pointed at an unwritable place: it degrades, the learner
    # keeps training, and train_once still reports a published version.
    (tmp_path / "nope").write_text("file", encoding="utf-8")
    exporter = ParamsExporter(
        tmp_path / "nope" / DIRECTORY_NAME, serializer=_fake_serializer
    )
    learner = _jax_learner(exporter=exporter)
    result = learner.train_once()

    assert result.published
    assert exporter.degraded
    assert learner.fault is None

    with pytest.raises(TypeError, match="export"):
        learner.attach_params_exporter(object())
    # The contract the hook relies on, stated where the hook is: a raising
    # exporter WOULD fault the learner, which is why ParamsExporter catches.
    learner._params_exporter = _Broken()
    with pytest.raises(Exception):
        learner.train_once()
    assert learner.fault is not None


def test_exported_blob_round_trips_onto_a_matching_subtree(tmp_path):
    """flax must be able to put the blob back where it came from."""

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp
    from flax.core import freeze
    from flax.serialization import from_bytes

    params = freeze(
        {
            "modules_actor": {
                "encoder": {
                    "encoder_cam1": {
                        FROZEN_TRUNK_LEAF_KEY: {
                            "kernel": jnp.full((2, 2), 7.0, dtype=jnp.float32)
                        },
                        "Dense_0": {
                            "kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
                        },
                    }
                }
            },
            "modules_temperature": {"lagrange": jnp.float32(0.5)},
        }
    )
    exporter = ParamsExporter(tmp_path / DIRECTORY_NAME)
    record = exporter.export(params, learner_step=50, version=1)
    assert record is not None

    # The peer restores onto ITS OWN freshly built tree, trunk already pruned.
    target, pruned = prune_frozen_trunk(
        jax.tree_util.tree_map(jnp.zeros_like, params)
    )
    assert pruned == (FROZEN_TRUNK_EXPECTED_PATH,)
    restored = from_bytes(target, record.path.read_bytes())

    np.testing.assert_array_equal(
        np.asarray(restored["modules_actor"]["encoder"]["encoder_cam1"]["Dense_0"][
            "kernel"
        ]),
        np.asarray(params["modules_actor"]["encoder"]["encoder_cam1"]["Dense_0"][
            "kernel"
        ]),
    )
    assert float(np.asarray(restored["modules_temperature"]["lagrange"])) == 0.5
    assert FROZEN_TRUNK_LEAF_KEY not in (
        restored["modules_actor"]["encoder"]["encoder_cam1"]
    )
    assert hashlib.sha256(record.path.read_bytes()).hexdigest() == record.sha256


# ---------------------------------------------------------------------------
# 7. The real agent.  Opt-in: building it costs a resnet10 load.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_HIL_SERL_ACTUAL_FEATURE_AGENT") != "1",
    reason="set RUN_HIL_SERL_ACTUAL_FEATURE_AGENT=1 to measure the real prune",
)
def test_real_agent_prune_matches_the_measured_byte_budget(tmp_path):
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")

    from ur_env.learner import LearnerConfig, create_frozen_trunk_feature_agent

    agent = create_frozen_trunk_feature_agent(
        config=LearnerConfig(),
        resnet_source_path=(
            Path(_REPO)
            / "third_party"
            / "hil-serl"
            / "examples"
            / "experiments"
            / "resnet10_params.pkl"
        ),
        resnet_cache_path=tmp_path / "verified-resnet10.pkl",
        validate_versions=True,
    )
    params = agent.state.params

    def nbytes(tree) -> int:
        return sum(
            int(np.asarray(jax.device_get(leaf)).nbytes)
            for leaf in jax.tree_util.tree_leaves(tree)
        )

    trunk = params
    for key in FROZEN_TRUNK_EXPECTED_PATH:
        trunk = trunk[key]

    total_bytes = nbytes(params)
    trunk_bytes = nbytes(trunk)
    tree, pruned = prune_frozen_trunk(params)
    trainable_bytes = nbytes(tree)

    # The numbers the local-inference design was sized against.
    assert total_bytes == 32_006_980
    assert trunk_bytes == 19_623_168
    assert trainable_bytes == 12_383_812
    assert trainable_bytes == total_bytes - trunk_bytes
    # Exactly ONE trunk copy: cam2 shares cam1's, and the critic/grasp-critic
    # share the actor's encoder.
    assert pruned == (FROZEN_TRUNK_EXPECTED_PATH,)

    exporter = ParamsExporter(tmp_path / DIRECTORY_NAME)
    record = exporter.export(params, learner_step=0, version=0)
    assert record is not None
    print(
        f"\nreal-agent params export: total={total_bytes} trunk={trunk_bytes} "
        f"trainable={trainable_bytes} blob={record.bytes} "
        f"overhead={record.bytes - trainable_bytes}"
    )
    # msgpack adds key names and per-array headers; the payload must still be
    # dominated by the tensors, and must never contain the trunk.
    assert trainable_bytes <= record.bytes < trainable_bytes + 200_000
    assert record.bytes < trunk_bytes

    from flax.serialization import from_bytes

    restored = from_bytes(tree, record.path.read_bytes())
    assert set(restored) == set(tree)
    assert FROZEN_TRUNK_LEAF_KEY not in (
        restored["modules_actor"]["encoder"]["encoder_cam1"]
    )


# ---------------------------------------------------------------------------
# 8. The launcher: the env var has to arrive, and nothing else may move.
# ---------------------------------------------------------------------------
#
# run_hil_server.sh re-validates a RUNNING learner against the exact command it
# would have launched, comparing argv values through /proc.  Server-side opt-in
# therefore rides on the environment.  These tests expand the launcher's own
# blocks (helpers live in test_hil_server_wandb_mode.py) rather than restating
# what they should contain.


_SERVER_SH = Path(_REPO) / "ros2_ur_ws" / "run_hil_server.sh"
_SENTINEL = "__unset__"


def test_launcher_forwards_the_env_var_and_leaves_argv_byte_identical(tmp_path):
    from test_hil_server_wandb_mode import (
        _expand_launch_command,
        _rig,
        _split_env_prefix,
    )

    rig = _rig(tmp_path, wandb_mode="offline")
    assert rig["PARAMS_EXPORT"] == ""

    off_env, off_argv = _split_env_prefix(_expand_launch_command(tmp_path, rig))
    on_env, on_argv = _split_env_prefix(
        _expand_launch_command(tmp_path, dict(rig, PARAMS_EXPORT="1"))
    )

    # THE constraint: not one argv token added, removed, reordered or rewritten.
    assert on_argv == off_argv
    assert ENABLE_ENV_VAR not in off_env
    assert on_env == {**off_env, ENABLE_ENV_VAR: "1"}


def _positionals(result) -> list[str]:
    tokens = [
        token.decode()
        for token in result.ssh_log.read_bytes().split(b"\0")
        if token
    ]
    return tokens[tokens.index("--") + 1 :]


def test_params_export_alone_does_not_arrive_as_latency_profiling(tmp_path):
    """The failure this ordering guard exists for.

    ssh joins its command arguments into ONE remote command line, so an empty
    argument in the MIDDLE of the list silently disappears and shifts every
    later value left.  Without the sentinel, HIL_PARAMS_EXPORT=1 with latency
    profiling off would land in ${15} and switch on the wrong feature.
    """

    from test_hil_server_wandb_mode import _run_launcher

    unset_dir = tmp_path / "unset"
    params_dir = tmp_path / "params"
    both_dir = tmp_path / "both"
    for directory in (unset_dir, params_dir, both_dir):
        directory.mkdir()

    # Pinned, not incidental: the launcher's default run id is
    # `cube_in_cup_real_$(date -u +...%H%M%S)`, so two invocations that straddle
    # a second boundary disagree in positional 4 and this list comparison fails
    # for a reason that has nothing to do with the ordering guard.  Measured at
    # roughly one run in ten before it was pinned.
    fixed = {"HIL_RUN_ID": "cube_in_cup_real_20260101_000000"}

    unset = _run_launcher(unset_dir, "--check", **fixed)
    assert unset.returncode == 3, unset.stderr
    baseline = _positionals(unset)
    assert len(baseline) == 14

    params_only = _run_launcher(
        params_dir, "--check", HIL_PARAMS_EXPORT="1", **fixed
    )
    assert params_only.returncode == 3, params_only.stderr
    assert _positionals(params_only) == baseline + [_SENTINEL, "1"]

    both = _run_launcher(
        both_dir,
        "--check",
        HIL_PARAMS_EXPORT="1",
        HIL_LATENCY_PROFILE="1",
        **fixed,
    )
    assert both.returncode == 3, both.stderr
    assert _positionals(both) == baseline + ["1", "1"]


def test_launcher_refuses_a_params_export_value_that_would_resplit(tmp_path):
    from test_hil_server_wandb_mode import _run_launcher

    result = _run_launcher(
        tmp_path, "--check", HIL_PARAMS_EXPORT="1 ; touch /tmp/pwned-params"
    )

    assert result.returncode == 3, result.stderr
    assert "ignoring HIL_PARAMS_EXPORT" in result.stderr
    assert len(_positionals(result)) == 14
    assert not os.path.exists("/tmp/pwned-params")


def _remote_positional_prologue() -> str:
    """The remote script's argument-decoding header, verbatim from the file."""

    lines = _SERVER_SH.read_text(encoding="utf-8").splitlines()
    opener = next(
        index
        for index, line in enumerate(lines)
        if line.rstrip().endswith("<<'REMOTE_SCRIPT'")
    )
    start = lines.index('MODE="$1"', opener)
    end = next(
        index
        for index in range(start, len(lines))
        if lines[index].startswith("CLASSIFIER=")
    )
    return "\n".join(lines[start:end])


@pytest.mark.parametrize(
    "extra,expected",
    [
        ([], ("", "")),
        (["1"], ("1", "")),
        ([_SENTINEL, "1"], ("", "1")),
        (["1", "1"], ("1", "1")),
    ],
)
def test_the_remote_side_decodes_every_optional_combination(tmp_path, extra, expected):
    """Both halves of the sentinel contract, run as the shell actually runs it."""

    script = (
        "set -euo pipefail\n"
        + _remote_positional_prologue()
        + '\nprintf "%s\\n%s\\n" "$LATENCY_PROFILE" "$PARAMS_EXPORT"\n'
    )
    result = subprocess.run(
        [
            "bash",
            "-s",
            "--",
            "check",
            "0",
            "0",
            "run-id",
            "300",
            str(tmp_path),
            sys.executable,
            "50053",
            f"{tmp_path}/runs",
            "abc123",
            "branch",
            "0",
            str(tmp_path),
            "offline",
            *extra,
        ],
        input=script,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert tuple(result.stdout.split("\n")[:2]) == expected


def test_the_learner_entrypoint_derives_the_directory_from_the_run_root():
    """<run root>/logs/learner.jsonl -> <run root>/params_live/, one derivation."""

    source = (
        Path(_INFRA) / "scripts" / "run_rlpd_learner_server.py"
    ).read_text(encoding="utf-8")
    assert "ParamsExporter.from_env(" in source
    assert 'jsonl_path.parent.parent / PARAMS_EXPORT_DIRECTORY_NAME' in source
    assert "attach_params_exporter(params_exporter)" in source
    # No CLI flag: validate_process_contract compares argv token by token.
    assert "--params-export" not in source
    assert DIRECTORY_NAME == "params_live"


def test_the_manifest_shape_is_pinned_where_the_puller_will_look():
    """The puller reads exactly these keys; a rename is a cross-host break."""

    from ur_env.learner.params_export import ExportedParams

    manifest = ExportedParams(
        version=3,
        learner_step=150,
        sha256="ab" * 32,
        bytes=12_385_459,
        filename=blob_name(3),
        path=Path("."),
    ).manifest()

    assert manifest == {
        "version": 3,
        "learner_step": 150,
        "sha256": "ab" * 32,
        "bytes": 12_385_459,
        "filename": "params_v00000003.msgpack",
        "schema": MANIFEST_SCHEMA,
    }
    # It has to survive a JSON round trip unchanged: that is the wire format.
    assert json.loads(json.dumps(manifest)) == manifest
    assert ParamsExporter(None, enabled=False).manifest_path is None
