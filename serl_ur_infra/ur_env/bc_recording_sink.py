"""On-disk episode recorder used as the BC policy server's ``accept_data``.

WHAT THIS IS
------------
:class:`ActorSessionService` hands every accepted transition to one callback --
``accept_data(data, intervened)`` -- inside the Step RPC, before the ACK is
returned (``ur_env/actor_network.py:769``).  In the production learner that
callback inserts into the RLPD replay buffers.  A BC evaluation has no learner:
nothing trains, and the only reason to run the actor is to be able to look at
what the policy did afterwards.  So this sink writes the transitions down.

The data item is the canonical wrapper the actor built in
``ur_env/remote_actor.py::build_data`` and the finalizer returned unchanged::

    {"meta": {schema_version, run_id, actor_id, session_id, transition_id,
              env_step, timestamp_ns, policy_version, policy_action,
              intervened, auto_success, operator_success,
              policy_actions_synthetic},
     "transition": {episode_id, step_id, observation_id, next_observation_id,
                    actions, rewards, masks, dones, truncated,
                    observations, next_observations, success, ...}}

``run_id`` is authoritative in ``meta``; ``episode_id``/``step_id`` are
authoritative in ``transition``.  Episodes are pickled as a plain ``list`` of
those wrapper dicts, which is exactly what
``ur_env/learner/demo.py::load_demo_object`` accepts (it special-cases
``set(item) == {"meta", "transition"}``), so a recorded episode can be re-read
with the repo's own strict loader instead of a bespoke parser.

WHY THERE IS NO ``prime_observation`` HERE -- LOAD-BEARING ABSENCE
------------------------------------------------------------------
``_prime_replay_observation`` (``actor_network.py:1476``) probes the sink with
``getattr(self._accept_data, "prime_observation", None)`` and, when it finds a
callable, feeds the *encoded frozen-trunk features* to the policy instead of the
pixels.  This sink deliberately does **not** define that attribute, so the
service falls through to ``None`` and the BC agent receives the actor's raw
pixel observation -- which is what its dual-input encoder was trained on.
Adding ``prime_observation`` to this class would silently change inference
input.  Do not add it.

WHAT IS NOT RECORDED
--------------------
An episode is only written when its terminal transition arrives.  A run that
dies mid-episode loses that episode's pickle -- the buffered items are RAM only.
That is accepted: ``actions.jsonl`` still holds every action of the dead
episode, and the alternative (partial pickles rewritten per step) would either
overwrite files or multiply them.  Image tensors never enter the jsonl; they
would dwarf the file and it is meant to stay tail-able during a session.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import threading
from typing import Any, Mapping

import numpy as np


#: Name of the provenance file written into the record root on construction.
METADATA_FILENAME = "metadata.json"

#: Append-only, one compact JSON object per accepted transition.
ACTIONS_FILENAME = "actions.jsonl"

#: Protocol 4 is the floor the repo's demo pickles already use; it keeps the
#: artifacts readable by the pinned NumPy 1 / Python 3.8+ learner env.
PICKLE_PROTOCOL = 4

#: The 7-D EEF-delta action contract (6 pose deltas + gripper).
ACTION_DIM = 7


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    return bool(value)


def _path_component(value: Any, *, name: str) -> str:
    """Reject anything that would escape or collide with the record root.

    ``run_id`` and ``episode_id`` arrive over the wire and are used to build
    filesystem paths, so a separator or traversal segment must fail loudly here
    rather than write outside the root the operator chose.
    """

    text = str(value)
    if not text or text in (".", ".."):
        raise ValueError(f"{name} is not a usable path component: {text!r}")
    if os.sep in text or (os.altsep and os.altsep in text) or "\0" in text:
        raise ValueError(f"{name} must not contain a path separator: {text!r}")
    return text


def _episode_key(value: Any) -> Any:
    """Normalise so ``np.int64(3)`` and ``3`` are the same episode."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    return value


def _episode_filename(episode_id: Any) -> str:
    if isinstance(episode_id, int) and not isinstance(episode_id, bool):
        return f"episode_{episode_id:04d}.pkl"
    return f"episode_{_path_component(episode_id, name='episode_id')}.pkl"


class EpisodeRecordingSink:
    """Record every accepted transition; one pickle per completed episode.

    Thread-safe: gRPC may serve Step from several handler threads, so buffers,
    counters and both files are guarded by a single lock.

    NOTE: this class must never grow a ``prime_observation`` attribute -- see
    the module docstring.  Its absence is what keeps raw pixels flowing to the
    policy.
    """

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        artifact_sha256: str = "",
        model_id: str = "",
    ) -> None:
        root_path = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
        # The server entrypoint owns creation; a sink that mkdir'd its own root
        # would happily record into a typo'd path for a whole session.
        if not root_path.is_dir():
            raise ValueError(f"record root is not an existing directory: {root_path}")
        self.root = root_path
        self.artifact_sha256 = str(artifact_sha256)
        self.model_id = str(model_id)
        self._metadata_path = root_path / METADATA_FILENAME
        self._actions_path = root_path / ACTIONS_FILENAME
        self._lock = threading.Lock()
        self._episodes: dict[tuple[str, Any], list[Mapping[str, Any]]] = {}
        self.replay_count = 0
        self.intervention_count = 0
        self._write_metadata()

    def _write_metadata(self) -> None:
        payload = {
            "artifact_sha256": self.artifact_sha256,
            "model_id": self.model_id,
            "created_at_utc": _utc_now_text(),
        }
        try:
            with open(self._metadata_path, "x", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
        except FileExistsError:
            self._require_matching_metadata()

    def _require_matching_metadata(self) -> None:
        """A record root must never mix two policies."""

        try:
            with open(self._metadata_path, "r", encoding="utf-8") as stream:
                existing = json.load(stream)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{self._metadata_path} exists but is not readable JSON: {exc}"
            ) from exc
        if not isinstance(existing, dict):
            raise ValueError(
                f"{self._metadata_path} must contain a JSON object, "
                f"got {type(existing).__name__}"
            )
        for key, expected in (
            ("artifact_sha256", self.artifact_sha256),
            ("model_id", self.model_id),
        ):
            found = existing.get(key, "")
            if str(found) != expected:
                raise ValueError(
                    f"{self._metadata_path} was written for {key}={found!r}; "
                    f"this sink records {key}={expected!r}. Use a fresh record "
                    "root per policy."
                )

    def __call__(self, data: dict[str, Any], intervened: bool) -> None:
        meta = data["meta"]
        transition = data["transition"]
        run_id = _path_component(meta["run_id"], name="run_id")
        episode_id = _episode_key(transition["episode_id"])
        # The service derives this argument from ``meta["intervened"]`` before
        # calling; the argument stays authoritative so a custom finalizer that
        # rewrote meta cannot desynchronise the counter from the routing.
        intervened_flag = bool(intervened)
        record = {
            "ts": _utc_now_text(),
            "run_id": run_id,
            "episode_id": episode_id,
            "step_id": int(transition["step_id"]),
            "env_step": int(meta["env_step"]),
            "actions": self._action_floats(transition["actions"]),
            "intervened": intervened_flag,
            "dones": _as_bool(transition["dones"]),
            "truncated": _as_bool(transition["truncated"]),
            "success": _as_bool(transition.get("success", False)),
            "rewards": float(transition["rewards"]),
        }
        terminal = record["dones"] or record["truncated"]
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        key = (run_id, episode_id)

        with self._lock:
            # The item arrives already deep-copied (actor_network.py:769), so
            # holding the reference is safe and a second copy is pure cost.
            self._episodes.setdefault(key, []).append(data)
            self.replay_count += 1
            self.intervention_count += int(intervened_flag)
            with open(self._actions_path, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
            if terminal:
                self._write_episode(key)

    def _write_episode(self, key: tuple[str, Any]) -> None:
        """Caller holds the lock."""

        run_id, episode_id = key
        items = list(self._episodes[key])
        directory = self.root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / _episode_filename(episode_id)
        # "xb", never "wb": a repeated (run_id, episode_id) means the server saw
        # two episodes it cannot tell apart, and overwriting would destroy a
        # recording an operator can never reproduce.  The buffer is kept on
        # failure so nothing is lost before the exception reaches the RPC.
        with open(path, "xb") as stream:
            pickle.dump(items, stream, protocol=PICKLE_PROTOCOL)
            stream.flush()
        del self._episodes[key]

    @staticmethod
    def _action_floats(value: Any) -> list[float]:
        action = np.asarray(value, dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"transition actions must hold {ACTION_DIM} floats, "
                f"got shape {np.asarray(value).shape}"
            )
        return [float(item) for item in action]
