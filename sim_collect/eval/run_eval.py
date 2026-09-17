#!/usr/bin/env python
"""Closed-loop success-rate evaluation CLI (sim_collect/eval/DESIGN.md §2.3).

    MUJOCO_GL=glfw DISPLAY=:0 .venv/bin/python -m sim_collect.eval.run_eval \\
        --policy zmq://127.0.0.1:5593 --task "Put carrot in pot" --seeds 0-19 \\
        --max-steps 600 --dwell-s 1.0 --out sim_collect/eval/runs/<name> [--video] [--config ...]
    ... --policy scripted --seeds 0-19 --out ...          (F2's oracle; positive control)
    ... --policy replay --takes ros2_ur_ws/gello_logs/sim --out ...   (one episode per take)
    ... --policy zero --seeds 0-4 --out ...               (negative control, SR must be 0)

Outputs in --out: episodes.jsonl, summary.json (SR + Wilson 95 % CI, per-outcome counts,
mean steps, mean time-to-success, config sha256, git commit, policy meta), summary.md,
ep_<id>.h5 (125 Hz qpos/qvel/ctrl, replayable) and ep_<id>_cam{1,2}.mp4 with --video.
Exit code 0 always (SR is data, not a test). One MuJoCo world + one CameraRig per process,
episodes strictly sequential (memory guardrail).
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim_collect.eval.policies import PolicyRefused, ReplayPolicy, make_policy  # noqa: E402
from sim_collect.eval.world import EvalWorld, Outcome  # noqa: E402

OUTCOMES = (Outcome.SUCCESS, Outcome.TIMEOUT, Outcome.FAULT, Outcome.FAILURE)


def git_commit(root: str = _ROOT) -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=3.0).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True,
                               text=True, timeout=5.0).stdout.strip()
        return (h or "unknown") + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def parse_seeds(spec: Optional[str], default: Sequence[int]) -> List[int]:
    """'0-19' | '0,3,7' | '0-4,10' -> list of ints (default when None/empty)."""
    if not spec:
        return [int(s) for s in default]
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1) if not part.startswith("-") else ("-" + part[1:].split("-", 1)[0], part[1:].split("-", 1)[1])
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Dict[str, float]:
    if n <= 0:
        return {"lo": 0.0, "hi": 1.0}
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"lo": max(0.0, centre - half), "hi": min(1.0, centre + half)}


def list_takes(takes: str) -> List[str]:
    """A directory of take_* folders, one take folder, or a comma list of either."""
    out: List[str] = []
    for item in str(takes).split(","):
        item = item.strip()
        if not item:
            continue
        p = item if os.path.isabs(item) else os.path.join(_ROOT, item)
        if os.path.isfile(os.path.join(p, "vectors.h5")):
            out.append(p)
        elif os.path.isdir(p):
            out.extend(sorted(os.path.join(p, d) for d in os.listdir(p)
                              if os.path.isfile(os.path.join(p, d, "vectors.h5"))))
        else:
            raise FileNotFoundError(f"--takes {item!r}: not a take directory or a directory of takes")
    return out


# --------------------------------------------------------------------------- #
# One episode                                                                   #
# --------------------------------------------------------------------------- #
def run_episode(world: EvalWorld, policy, ep_id: str, seed: Optional[int], max_steps: int, out_dir: Optional[str],
                video: bool = False, save_state: bool = True, verbose: bool = True) -> Dict[str, Any]:
    t_wall = time.time()
    rec: Dict[str, Any] = {"episode": ep_id, "seed": seed, "outcome": None, "n_steps": 0, "t_success_s": None,
                           "t_first_ok_s": None, "detail": "", "failure": None, "fault": None,
                           "policy": dict(getattr(policy, "meta", {}) or {}), "wall_s": 0.0}
    info: Dict[str, Any] = {}
    try:
        info = policy.reset(world.info()) or {}
    except PolicyRefused:
        raise                                   # a checkpoint this harness cannot drive: abort the run
    except Exception as exc:  # noqa: BLE001  (a RESET timeout / ok:false is a fault)
        rec.update(outcome=Outcome.FAULT, fault=f"reset: {type(exc).__name__}: {exc}", wall_s=time.time() - t_wall)
        if verbose:
            print(f"[run_eval] {ep_id}: FAULT at reset: {rec['fault']}", flush=True)
        return rec
    # RESET may add episode-specific metadata (notably IFQL reset_counter/log_dir)
    # to policy.meta.  Snapshot it *after* RESET: taking this copy before RESET makes
    # every episode point at the previous episode's policy log.
    rec["policy"] = copy.deepcopy(dict(getattr(policy, "meta", {}) or {}))
    reset_reply = rec["policy"].get("reset_reply")
    if isinstance(reset_reply, dict):
        rec["policy_reset"] = {
            k: copy.deepcopy(reset_reply[k])
            for k in ("reset_counter", "seed", "torch_seed", "log_dir")
            if k in reset_reply
        }
    need_images = bool(getattr(policy, "needs_images", True)) or video
    obs = world.reset(seed=seed, layout_override=info.get("layout_override"), q0=info.get("q0"),
                      video_dir=out_dir, video_tag=f"ep_{ep_id}", images=need_images)
    rec["layout"] = world.layout
    rec["reset_note"] = world.reset_note
    steps = int(max_steps)
    if info.get("max_steps_hint") is not None:
        steps = min(steps, int(info["max_steps_hint"]))
    outcome = Outcome.TIMEOUT
    n = 0
    while n < steps:
        try:
            action = policy.act(obs)
        except Exception as exc:  # noqa: BLE001  (a raising policy is a fault; the run continues)
            outcome = Outcome.FAULT
            rec["fault"] = f"act: {type(exc).__name__}: {exc}"
            break
        if action is None:
            outcome = Outcome.FAULT
            rec["fault"] = str(getattr(policy, "last_error", None) or "policy returned None")
            break
        try:
            world.apply(action)
        except ValueError as exc:
            outcome = Outcome.FAULT
            rec["fault"] = f"apply: {exc}"
            break
        r = world.step()
        n += 1
        if r["done"]:
            outcome = r["outcome"]
            break
        obs = world.observe(images=need_images)
    rec.update(outcome=outcome, n_steps=n, t_success_s=world.t_success, t_first_ok_s=world.t_first_ok,
               detail=world.task_detail, failure=world.failure, t_end_s=world.t,
               n_clamped_limit=world.n_clamped_limit, n_clamped_dev=world.n_clamped_dev,
               clamp_hits=json.loads(json.dumps(world.clamp_hits)),
               final_state=[float(v) for v in world.data.qpos[:6]] + [world.grip_pos()],
               objects={nm: {"pos": world.object_pose(nm)[0].tolist(), "quat_wxyz": world.object_pose(nm)[1].tolist()}
                        for nm in world.obj_frame_ids})
    if hasattr(policy, "stats"):
        try:
            rec["policy_stats"] = policy.stats()
        except Exception:  # noqa: BLE001
            pass
    if out_dir and save_state:
        rec["state_h5"] = os.path.relpath(world.save_episode(os.path.join(out_dir, f"ep_{ep_id}.h5"),
                                                             {"episode": ep_id, "outcome": outcome,
                                                              "policy_reset": rec.get("policy_reset")}), _ROOT)
    if video and getattr(world, "video_paths", None):
        rec["video"] = {k: os.path.relpath(v, _ROOT) for k, v in world.video_paths.items()}
        world._close_video()
    rec["wall_s"] = time.time() - t_wall
    if verbose:
        ts = f" t_success={rec['t_success_s']:.2f}s" if rec["t_success_s"] is not None else ""
        extra = f" fault={rec['fault']}" if rec.get("fault") else (f" failure={rec['failure']}" if rec.get("failure") else "")
        ch = rec["clamp_hits"]
        clamps = f" clamps env={ch['envelope']} dev={ch['max_dev']} grip={ch['grip']}" if (ch["envelope"] or ch["max_dev"] or ch["grip"]) else ""
        print(f"[run_eval] {ep_id}: {outcome:8s} steps={n:4d}{ts}{extra}{clamps} | {rec['detail']} | {rec['wall_s']:.1f}s", flush=True)
    return rec


# --------------------------------------------------------------------------- #
# Summary                                                                       #
# --------------------------------------------------------------------------- #
def summarize(episodes: List[Dict[str, Any]], *, policy_spec: str, world: Optional[EvalWorld], args: Dict[str, Any],
              wall_s: float, policy_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    n = len(episodes)
    counts = {o: sum(1 for e in episodes if e["outcome"] == o) for o in OUTCOMES}
    k = counts[Outcome.SUCCESS]
    ts = [e["t_success_s"] for e in episodes if e.get("t_success_s") is not None]
    steps = [e["n_steps"] for e in episodes]
    succ_steps = [e["n_steps"] for e in episodes if e["outcome"] == Outcome.SUCCESS]
    hits = [e.get("clamp_hits") or {} for e in episodes]
    clamp_totals = {
        "steps": sum(h.get("steps", 0) for h in hits),
        "envelope": sum(h.get("envelope", 0) for h in hits), "max_dev": sum(h.get("max_dev", 0) for h in hits),
        "grip": sum(h.get("grip", 0) for h in hits),
        "envelope_per_joint": [sum(h.get("envelope_per_joint", [0] * 6)[i] for h in hits) for i in range(6)],
        "max_dev_per_joint": [sum(h.get("max_dev_per_joint", [0] * 6)[i] for h in hits) for i in range(6)],
        "episodes_with_envelope_hits": sum(1 for h in hits if h.get("envelope", 0)),
        "episodes_with_max_dev_hits": sum(1 for h in hits if h.get("max_dev", 0)),
    }
    return {
        "policy": policy_spec, "policy_meta": policy_meta or (episodes[0].get("policy") if episodes else {}),
        "n_episodes": n, "n_success": k, "success_rate": (k / n) if n else None, "wilson_95": wilson_ci(k, n),
        "counts": counts, "mean_steps": (sum(steps) / n) if n else None,
        "mean_steps_success": (sum(succ_steps) / len(succ_steps)) if succ_steps else None,
        "mean_time_to_success_s": (sum(ts) / len(ts)) if ts else None,
        "clamp_hits": clamp_totals,
        "faults": [{"episode": e["episode"], "fault": e["fault"]} for e in episodes if e.get("fault")],
        "failures": [{"episode": e["episode"], "failure": e["failure"]} for e in episodes if e.get("failure")],
        "config_path": (os.path.relpath(world.config_path, _ROOT) if world else args.get("config")),
        "config_sha256": world.config_sha256 if world else None, "xml_sha256": world.xml_sha256 if world else None,
        "eval_config": world.ev if world else None, "git_commit": git_commit(), "args": args,
        "started_at": args.get("started_at"), "wall_s": wall_s, "episodes_file": "episodes.jsonl",
    }


def summary_markdown(s: Dict[str, Any], episodes: List[Dict[str, Any]]) -> str:
    ci = s["wilson_95"]
    sr = s["success_rate"]
    lines = [f"# eval summary — policy `{s['policy']}`", "",
             f"- episodes: **{s['n_episodes']}**, successes: **{s['n_success']}**, "
             f"SR = **{(sr * 100 if sr is not None else float('nan')):.1f} %** "
             f"(Wilson 95 % CI {ci['lo'] * 100:.1f}–{ci['hi'] * 100:.1f} %)",
             f"- counts: " + ", ".join(f"{k} {v}" for k, v in s["counts"].items()),
             f"- mean steps {s['mean_steps']:.1f}" if s["mean_steps"] is not None else "- mean steps n/a",
             (f"- mean time-to-success {s['mean_time_to_success_s']:.2f} s" if s["mean_time_to_success_s"] is not None
              else "- mean time-to-success n/a"),
             (f"- clamp hits (steps on which a clamp bound): envelope {s['clamp_hits']['envelope']} / max_dev "
              f"{s['clamp_hits']['max_dev']} / grip {s['clamp_hits']['grip']} of {s['clamp_hits']['steps']} steps; "
              f"envelope per joint {s['clamp_hits']['envelope_per_joint']}; episodes with envelope hits "
              f"{s['clamp_hits']['episodes_with_envelope_hits']}/{s['n_episodes']}"),
             f"- envelope: {s['eval_config'].get('envelope_source', 'unknown') if s.get('eval_config') else 'unknown'}",
             f"- envelope lo {s['eval_config']['joint_limits_lo'] if s.get('eval_config') else '?'} hi "
             f"{s['eval_config']['joint_limits_hi'] if s.get('eval_config') else '?'} (max_dev_rad "
             f"{s['eval_config']['max_dev_rad'] if s.get('eval_config') else '?'})",
             f"- config `{s['config_path']}` sha256 `{(s['config_sha256'] or '')[:12]}` · git `{s['git_commit']}` · wall {s['wall_s']:.0f} s",
             f"- policy meta: `{json.dumps(s.get('policy_meta') or {}, default=str)[:400]}`", "",
             "| episode | seed | outcome | steps | t_success (s) | clamps env/dev/grip | detail |", "|---|---|---|---|---|---|---|"]
    for e in episodes:
        ts = f"{e['t_success_s']:.2f}" if e.get("t_success_s") is not None else ""
        det = e.get("fault") or e.get("failure") or e.get("detail") or ""
        ch = e.get("clamp_hits") or {}
        cl = f"{ch.get('envelope', 0)}/{ch.get('max_dev', 0)}/{ch.get('grip', 0)}"
        lines.append(f"| {e['episode']} | {e.get('seed')} | {e['outcome']} | {e['n_steps']} | {ts} | {cl} | {det} |")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: str, episodes: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as f:
        for e in episodes:
            f.write(json.dumps(e, default=str) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary_markdown(summary, episodes))


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="zmq://host:port | replay | scripted | zero")
    ap.add_argument("--takes", default=None, help="replay: directory of take_* folders (or one take, or a comma list)")
    ap.add_argument("--seeds", default=None, help="e.g. 0-19 or 0,3,7 (default: yaml eval.seeds)")
    ap.add_argument("--max-steps", type=int, default=None, help="default: yaml eval.max_steps (600)")
    ap.add_argument("--dwell-s", type=float, default=None, help="default: yaml eval.dwell_s (1.0)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--video", action="store_true", help="write ep_<id>_cam1.mp4 / _cam2.mp4 (mp4v, 30 fps)")
    ap.add_argument("--config", default="sim_collect/configs/carrot_in_pot_sim.yaml")
    ap.add_argument("--task", default="Put carrot in pot", help="task string (recorded; FM servers read it server-side)")
    ap.add_argument("--timeout-s", type=float, default=None,
                    help="ZMQ REQ timeout (a timeout is a fault). Default = the real client's per-type value "
                         "(act 0.5 s, diffusion/fm 0.6 s) inferred from --policy-type or the port 5591/5592/5593")
    ap.add_argument("--policy-type", choices=["act", "diffusion", "fm"], default=None,
                    help="server type behind zmq:// (sets the default timeout; recorded in policy_meta)")
    ap.add_argument("--no-state", action="store_true", help="skip the per-episode h5 state logs")
    ap.add_argument("--no-envelope", action="store_true",
                    help="disable clamp (1) (yaml eval.joint_limits = the REAL dataset's envelope, which the sim "
                         "demos leave on 24 %% of steps); only the model joint range + max_dev_rad remain")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "glfw")
    started = _dt.datetime.now().isoformat(timespec="seconds")
    t_wall = time.time()
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    world = EvalWorld(args.config, video=args.video)
    if args.dwell_s is not None:
        world.dwell_s = float(args.dwell_s)
    if args.no_envelope:
        world.set_envelope(None, None)
    max_steps = int(args.max_steps) if args.max_steps is not None else int(world.ev.get("max_steps", 600))

    if args.policy == "replay":
        if not args.takes:
            ap.error("--policy replay needs --takes")
        jobs = [(os.path.basename(t.rstrip("/")), i, t) for i, t in enumerate(list_takes(args.takes))]
        if not jobs:
            ap.error(f"no takes found under {args.takes}")
    else:
        seeds = parse_seeds(args.seeds, world.ev.get("seeds", list(range(20))))
        jobs = [(str(s), s, None) for s in seeds]

    policy = None if args.policy == "replay" else make_policy(args.policy, task=args.task, timeout_s=args.timeout_s,
                                                              world=world, policy_type=args.policy_type)
    policy_meta = dict(getattr(policy, "meta", {}) or {}) if policy is not None else {"policy": "replay", "takes": args.takes}
    print(f"[run_eval] policy={args.policy} episodes={len(jobs)} max_steps={max_steps} dwell={world.dwell_s}s "
          f"video={args.video} out={os.path.relpath(out_dir, _ROOT)}", flush=True)
    episodes: List[Dict[str, Any]] = []
    run_args = {**vars(args), "started_at": started, "max_steps": max_steps}
    try:
        for ep_id, seed, take in jobs:
            try:
                pol = ReplayPolicy(take, policy_hz=world.policy_hz) if take is not None else policy
                rec = run_episode(world, pol, ep_id, seed, max_steps, out_dir, video=args.video,
                                  save_state=not args.no_state, verbose=not args.quiet)
            except PolicyRefused as exc:
                # an EEF / wrong-dimension checkpoint: abort the run instead of producing N faults
                print(f"[run_eval] ABORT: {exc}", file=sys.stderr, flush=True)
                write_outputs(out_dir, episodes, summarize(episodes, policy_spec=args.policy, world=world,
                                                           args={**run_args, "aborted": str(exc)},
                                                           wall_s=time.time() - t_wall, policy_meta=policy_meta))
                return 2
            if take is not None:
                rec["take"] = os.path.relpath(take, _ROOT)
            episodes.append(rec)
            if policy is not None:
                policy_meta = dict(getattr(policy, "meta", {}) or {})   # RESET adds the server's v2 fields
            # keep the run readable while it is still going
            write_outputs(out_dir, episodes, summarize(episodes, policy_spec=args.policy, world=world, args=run_args,
                                                       wall_s=time.time() - t_wall, policy_meta=policy_meta))
    finally:
        world.close()
        if policy is not None and hasattr(policy, "close"):
            policy.close()
    s = summarize(episodes, policy_spec=args.policy, world=world, args=run_args,
                  wall_s=time.time() - t_wall, policy_meta=policy_meta)
    write_outputs(out_dir, episodes, s)
    ci = s["wilson_95"]
    print(f"[run_eval] SR {s['n_success']}/{s['n_episodes']} = {(s['success_rate'] or 0) * 100:.1f} % "
          f"(Wilson 95 % {ci['lo'] * 100:.1f}–{ci['hi'] * 100:.1f} %) counts={s['counts']} "
          f"wall={s['wall_s']:.0f}s -> {os.path.relpath(out_dir, _ROOT)}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
