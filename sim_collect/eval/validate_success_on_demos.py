#!/usr/bin/env python
"""Validate the success predicate + dwell latch against the recorded human demos
(sim_collect/eval/DESIGN.md §2.4 — "replay the dataset in MuJoCo and the method will
show itself").

    MUJOCO_GL=glfw DISPLAY=:0 .venv/bin/python -m sim_collect.eval.validate_success_on_demos \\
        --takes ros2_ur_ws/gello_logs/sim [--dwell-s 1.0] [--out sim_collect/eval/runs/demo_validation.json]
        [--no-dynamic] [--limit N]

1. KINEMATIC replay: every `/sim_mj_state` row (125 Hz) of each take is written into the
   take's own recorded scene (`replay_take.load_scene/load_model/set_row`) and run through
   `TaskEvaluator` (grip_cmd = ctrl[6]/255) with the dwell latch. Requirements: NOT success
   at t=0 and success (latched) by the end, for every take; prints time-to-success and the
   recorder's `task_success_at_stop` flag next to it.
2. DYNAMIC replay: `ReplayPolicy` re-executes each take's recorded `command` + grip_cmd from
   the recorded initial layout / arm pose through `EvalWorld` (the deploy clamps + bridge
   upsampler + physics). Physics divergence is expected — the number is reported honestly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "ros2_ur_ws", "src", "ur_gello_bringup")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "glfw")


def kinematic_validate(take_dir: str, dwell_s: float = 1.0) -> Dict[str, Any]:
    """Replay the take's recorded state rows through TaskEvaluator with the dwell latch."""
    import h5py
    import mujoco
    from sim_collect.scene import SceneConfig, build_scene
    from sim_collect.task import TaskConfig, TaskEvaluator
    from sim_collect.tools import replay_take as rt

    take = os.path.basename(take_dir.rstrip("/"))
    scene = rt.load_scene(take_dir)
    cfg = SceneConfig.from_dict(dict(scene.config))
    meta = build_scene(cfg, scene.layout or None).meta          # names only (frame_body, sites)
    model = rt.load_model(scene)                                  # the RECORDED xml
    data = mujoco.MjData(model)
    task = TaskEvaluator(model, TaskConfig.from_scene(cfg.task, meta))
    st = rt.load_state(take_dir)
    with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
        g = f["gripper"]
        grip_t, grip_cmd = g["t_rel_s"][:], g["grip_cmd"][:]
        sim_meta = json.loads(f.attrs["sim_meta"]) if "sim_meta" in f.attrs else {}
    n = len(st.t_rel_s)
    ok_since: Optional[float] = None
    first_ok: Optional[float] = None
    first_success: Optional[float] = None
    latched = False
    ok_at_t0 = False
    ok_at_end = False
    detail_end = ""
    n_ok = 0
    t0 = float(st.t_rel_s[0])
    for i in range(n):
        rt.set_row(model, data, st, i)
        gc = st.ctrl[i, 6] / 255.0 if (st.ctrl.shape[1] >= 7 and np.isfinite(st.ctrl[i, 6])) else \
            float(grip_cmd[int(np.argmin(np.abs(grip_t - st.t_rel_s[i])))])
        ok, detail = task.evaluate(data, float(gc))
        t = float(st.t_rel_s[i]) - t0
        if i == 0:
            ok_at_t0 = ok
        if ok:
            n_ok += 1
            if ok_since is None:
                ok_since = t
                if first_ok is None:
                    first_ok = t
            if not latched and t - ok_since >= dwell_s - 1e-9:
                latched = True
                first_success = t
        else:
            ok_since = None
        if i == n - 1:
            ok_at_end, detail_end = ok, detail
    return {"take": take, "n_rows": n, "duration_s": float(st.t_rel_s[-1]) - t0, "success_at_t0": bool(ok_at_t0),
            "first_ok_t": first_ok, "first_success_t": first_success, "latched": latched,
            "predicate_at_end": bool(ok_at_end), "success_at_end": bool(latched and ok_at_end),
            "fraction_rows_ok": n_ok / max(n, 1), "detail_end": detail_end,
            "recorder_flag": sim_meta.get("task_success_at_stop"), "rebuilt_matches": scene.rebuilt_matches,
            "notes": scene.notes}


def dynamic_replay(world, take_dir: str, max_steps: int = 600, out_dir: Optional[str] = None) -> Dict[str, Any]:
    from sim_collect.eval.policies import ReplayPolicy
    from sim_collect.eval.run_eval import run_episode
    pol = ReplayPolicy(take_dir, policy_hz=world.policy_hz)
    rec = run_episode(world, pol, os.path.basename(take_dir.rstrip("/")), None, max_steps, out_dir, verbose=False)
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--takes", default="ros2_ur_ws/gello_logs/sim")
    ap.add_argument("--dwell-s", type=float, default=1.0)
    ap.add_argument("--config", default="sim_collect/configs/carrot_in_pot_sim.yaml")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default="sim_collect/eval/runs/demo_validation.json")
    ap.add_argument("--no-dynamic", action="store_true", help="kinematic validation only")
    ap.add_argument("--limit", type=int, default=None, help="only the first N takes")
    args = ap.parse_args(argv)
    from sim_collect.eval.run_eval import list_takes
    takes = list_takes(args.takes)
    if args.limit:
        takes = takes[:args.limit]
    print(f"[validate] {len(takes)} takes under {args.takes}, dwell {args.dwell_s}s", flush=True)

    kin: List[Dict[str, Any]] = []
    t0 = time.time()
    for td in takes:
        r = kinematic_validate(td, args.dwell_s)
        kin.append(r)
        fs = f"{r['first_success_t']:.2f}" if r["first_success_t"] is not None else "  -  "
        fo = f"{r['first_ok_t']:.2f}" if r["first_ok_t"] is not None else "  -  "
        print(f"  {r['take']:28s} rows {r['n_rows']:5d} dur {r['duration_s']:6.2f}s | t0 {'SUCC' if r['success_at_t0'] else 'no  '} "
              f"| first_ok {fo} | latched {fs} | end {'SUCC' if r['success_at_end'] else 'FAIL'} "
              f"| recorder {r['recorder_flag']} | xml_match {r['rebuilt_matches']} | {r['detail_end']}", flush=True)
    n = len(kin)
    n_end = sum(1 for r in kin if r["success_at_end"])
    n_t0 = sum(1 for r in kin if r["success_at_t0"])
    n_flag = sum(1 for r in kin if r["recorder_flag"])
    print(f"[validate] KINEMATIC: success_at_end {n_end}/{n}, success_at_t0 {n_t0}/{n} (must be 0), "
          f"recorder task_success_at_stop {n_flag}/{n} ({time.time() - t0:.0f}s)", flush=True)

    dyn: List[Dict[str, Any]] = []
    if not args.no_dynamic:
        from sim_collect.eval.run_eval import summarize
        from sim_collect.eval.world import EvalWorld
        world = EvalWorld(args.config, video=False)
        world.dwell_s = float(args.dwell_s)
        t1 = time.time()
        try:
            for td in takes:
                rec = dynamic_replay(world, td, args.max_steps, out_dir=None)
                dyn.append(rec)
                ts = f"{rec['t_success_s']:.2f}s" if rec.get("t_success_s") is not None else "  -  "
                print(f"  {rec['episode']:28s} {rec['outcome']:8s} steps {rec['n_steps']:4d} t_success {ts} | "
                      f"{rec.get('fault') or rec.get('failure') or rec['detail']}", flush=True)
        finally:
            world.close()
        s = summarize(dyn, policy_spec="replay", world=world, args=vars(args), wall_s=time.time() - t1)
        print(f"[validate] DYNAMIC replay: {s['n_success']}/{s['n_episodes']} successes "
              f"(Wilson 95 % {s['wilson_95']['lo'] * 100:.0f}–{s['wilson_95']['hi'] * 100:.0f} %) counts={s['counts']} "
              f"({time.time() - t1:.0f}s)", flush=True)
    else:
        s = None

    out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"takes_dir": args.takes, "dwell_s": args.dwell_s, "n_takes": n,
                   "kinematic": {"success_at_end": n_end, "success_at_t0": n_t0, "recorder_flag_true": n_flag, "per_take": kin},
                   "dynamic": ({"summary": s, "per_take": dyn} if s else None)}, f, indent=2, default=str)
    print(f"[validate] wrote {os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
