#!/usr/bin/env python3
"""[C] sim_collect operator GUI (tkinter) — DESIGN.md §2.3.

This window is the ONLY thing the operator touches while collecting takes. It owns no
physics, no cameras and no files: it is a thin, never-blocking client of the two REP
servers of §2.1/§2.2 plus the 10 Hz preview PUB of §2.2.

    sim_rep      (A, sim_main.py) : get_status engage disengage set_pos_scale
                                    set_control_mode reset_scene home
                                    gripper_pause gripper_resume
    capture_rep  (B, capture.py)  : get_status start_take stop_take
                                    discard_last_take snapshot
    preview_pub  (B, topic "preview") : {"cam1": jpeg, "cam2": jpeg, "recording": bool, ...}

WORDING AND BEHAVIOUR ARE COPIED FROM THE TWO REAL OPERATOR GUIs so the user's hands
already know this window:

  * ``ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/gello_eef_gui_node.py`` — one big
    colour-coded primary toggle that is the WHOLE mouse button (ENGAGE green = go,
    DISENGAGE blue = stop/hold, armed-confirm orange), ENGAGE behind a two-click confirm
    inside a 3 s window because it starts motion while DISENGAGE is single-click because
    it is safe, the pos_scale "DPI" slider 10..100 → 0.10..1.00 whose value is PENDING
    until an anchor-reset moment (A안: committed right before an engage, never mid-stroke),
    ``Gripper PAUSE`` / ``Gripper Resume`` with resume behind its own confirm, and the same
    state colours/sentences (_STATE_STYLE below).
  * ``ros2_ur_ws/src/gello_recorder/gello_recorder/gello_recorder_gui.py`` — the recording
    bar: start/stop, ``Take: N``, ``● RECORDING take N`` / ``PREVIEW``, ``Elapsed: x.xs``.

THREADING / BLOCKING. Tk is single-threaded and nothing here may sit on a socket. Two
different timeouts do that work:

  * command clients use ``--sim-timeout-ms`` (a click may legitimately take a while — an
    engage runs the G2–G8 gates server-side) and are called synchronously, so one click on
    a dead server costs at most that once;
  * the 5 Hz ``get_status`` pollers use their own SHORT-timeout clients
    (``--status-timeout-ms``, 120 ms) and, the moment a poll times out, that server's poll
    drops to 1 Hz and the status bar says "DISCONNECTED". So a dead or restarting server
    costs ~120 ms per second, not 5 × per second, and the GUI keeps retrying forever.
    A REQ socket that timed out is recreated by ``ipc.Client``, and ZMQ reconnects on its
    own, so a sim restart heals with no operator action (we re-sync the slider and the
    control-mode radio on the disconnected→connected edge).

Everything the servers reply with lands in the log pane at the bottom, which is the only
place a ``msg`` string ever goes (status-poll messages are de-duplicated so 5 Hz polling
cannot flood it).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tkinter as tk
from typing import Any, Dict, Optional

import cv2
import numpy as np

if __package__ in (None, ""):  # allow `python sim_collect/gui.py`
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim_collect import ipc  # noqa: E402

# --------------------------------------------------------------------------- #
# Chrome (copied from gello_eef_gui_node so the two windows read the same)     #
# --------------------------------------------------------------------------- #
_GREEN = "#2e7d32"      # ENGAGE = go
_GREEN_HI = "#1b5e20"
_BLUE = "#1565c0"       # DISENGAGE = stop / hold
_BLUE_HI = "#0d47a1"
_ORANGE = "#ef6c00"     # armed "click again"
_ORANGE_HI = "#e65100"
_SLATE = "#546e7a"      # neutral secondary
_SLATE_HI = "#37474f"
_RED = "#cc3333"        # hazard (Gripper PAUSE)
_RED_HI = "#a02020"
_GRAY = "#888888"
_DISABLED_BG = "#cfd8dc"
_DISABLED_FG = "#7a8a94"
_BG = "#eceff1"

#: eef_state → (colour, one-line meaning). Same vocabulary as the real EEF GUI.
_STATE_STYLE = {
    "HOLD": ("#dd8800", "HOLD — armed, arm frozen"),
    "ENGAGED": ("#22aa22", "ENGAGED — tracking"),
    "DISENGAGED": (_GRAY, "DISENGAGED — clutched up, arm holds"),
    "JOINT_BOOTSTRAP": ("#3366cc", "JOINT PASSTHROUGH — arm mirrors leader"),
}

#: how long an armed two-click confirm stays armed (ms) — same 3 s as the real GUI.
CONFIRM_MS = 3000
#: status poll period while the server answers / while it does not (ms).
POLL_MS = 200
POLL_SLOW_MS = 1000
#: preview repaint period (ms) — the publisher runs at 10 Hz, we never go faster.
PREVIEW_MS = 100
#: hard auto-repeat backstop for the keyboard shortcuts (s).
KEY_DEBOUNCE_S = 0.12
#: log pane ring size.
LOG_LINES = 200
#: preview pane width (the publisher sends 320×180).
PREVIEW_W = 320


def jpeg_to_ppm(buf: bytes, max_w: int = PREVIEW_W) -> Optional[bytes]:
    """Decode a JPEG to the PPM byte string ``tk.PhotoImage(data=...)`` eats.

    tkinter has no JPEG support and this venv has no PIL, so cv2 does the decode and we
    hand Tk a raw P6 (binary RGB) blob. Returns None if the buffer is not decodable.
    """
    if not buf:
        return None
    bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        return None
    if bgr.shape[1] > max_w:
        h = max(1, int(round(bgr.shape[0] * max_w / bgr.shape[1])))
        bgr = cv2.resize(bgr, (max_w, h), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return b"P6 %d %d 255\n" % (w, h) + rgb.tobytes()


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


class SimCollectGui:
    """The whole window. Construct, then call :meth:`run` (or drive it from tests)."""

    def __init__(
        self,
        master: Optional[tk.Misc] = None,
        sim_timeout_ms: int = 1500,
        status_timeout_ms: int = 120,
        title: str = "sim_collect — GELLO ▸ MuJoCo 데이터 수집",
        autostart: bool = True,
    ) -> None:
        self.root = tk.Tk() if master is None else master
        self._owns_root = master is None
        try:
            self.root.title(title)
        except tk.TclError:
            pass
        self.root.configure(bg=_BG)

        # Two client pairs on purpose: commands may take ~a second, polls must not.
        self.sim = ipc.Client("sim_rep", timeout_ms=sim_timeout_ms)
        self.cap = ipc.Client("capture_rep", timeout_ms=sim_timeout_ms)
        self.sim_poll = ipc.Client("sim_rep", timeout_ms=status_timeout_ms)
        self.cap_poll = ipc.Client("capture_rep", timeout_ms=status_timeout_ms)
        self.preview = ipc.Subscriber("preview_pub", "preview", conflate=True)

        self.sim_status: Dict[str, Any] = {}
        self.cap_status: Dict[str, Any] = {}
        self.sim_connected = False
        self.cap_connected = False
        self.preview_msg: Dict[str, Any] = {}

        self._armed: Dict[str, Optional[str]] = {}   # key → after-id of the disarm timer
        self._after_ids: Dict[str, Optional[str]] = {}
        self._photo: Dict[str, Optional[tk.PhotoImage]] = {"cam1": None, "cam2": None}
        self._last_poll_msg: Dict[str, str] = {}
        self._last_problems: list = []
        self._closed = False

        self._build_ui()
        self._bind_keys()
        self.refresh()
        if autostart:
            self.start_polling()

    # ------------------------------------------------------------------ UI --
    def _button(self, parent, text, command, bg=_SLATE, hi=_SLATE_HI, big=False, **kw):
        return tk.Button(
            parent, text=text, command=command, bg=bg, fg="white",
            activebackground=hi, activeforeground="white",
            disabledforeground=_DISABLED_FG, relief="raised", bd=2,
            font=("DejaVu Sans", 16 if big else 11, "bold"),
            padx=10, pady=12 if big else 4, takefocus=False,
            highlightthickness=0, **kw)

    def _build_ui(self) -> None:
        root = self.root

        # --- status bar --------------------------------------------------
        bar = tk.Frame(root, bg=_BG)
        bar.pack(fill="x", padx=8, pady=(8, 2))
        self.sim_conn_var = tk.StringVar(value="sim: DISCONNECTED")
        self.cap_conn_var = tk.StringVar(value="capture: DISCONNECTED")
        self.sim_conn_lbl = tk.Label(bar, textvariable=self.sim_conn_var, bg=_GRAY,
                                     fg="white", font=("DejaVu Sans", 10, "bold"), padx=6)
        self.cap_conn_lbl = tk.Label(bar, textvariable=self.cap_conn_var, bg=_GRAY,
                                     fg="white", font=("DejaVu Sans", 10, "bold"), padx=6)
        self.sim_conn_lbl.pack(side="left")
        self.cap_conn_lbl.pack(side="left", padx=(6, 12))

        self.info_var = tk.StringVar(value="mode — · state — · sigma_min — · reason —")
        tk.Label(bar, textvariable=self.info_var, bg=_BG, anchor="w",
                 font=("DejaVu Sans", 10)).pack(side="left", fill="x", expand=True)

        self.task_var = tk.StringVar(value="TASK —")
        self.task_lbl = tk.Label(bar, textvariable=self.task_var, bg=_GRAY, fg="white",
                                 font=("DejaVu Sans", 10, "bold"), padx=6)
        self.task_lbl.pack(side="right")

        # --- second status row: the two servers' own health counters ---------
        bar2 = tk.Frame(root, bg=_BG)
        bar2.pack(fill="x", padx=8, pady=(0, 2))
        self.sim_health_var = tk.StringVar(value="sim —")
        tk.Label(bar2, textvariable=self.sim_health_var, bg=_BG, anchor="w",
                 font=("DejaVu Sans", 9)).pack(side="left")
        self.cap_health_var = tk.StringVar(value="capture —")
        tk.Label(bar2, textvariable=self.cap_health_var, bg=_BG, anchor="w",
                 font=("DejaVu Sans", 9)).pack(side="left", padx=(12, 0))
        # Drop counters live in their own label so they can go RED on their own.
        self.drops_var = tk.StringVar(value="")
        self.drops_lbl = tk.Label(bar2, textvariable=self.drops_var, bg=_BG, fg="#2e7d32",
                                  font=("DejaVu Sans", 9, "bold"))
        self.drops_lbl.pack(side="right")

        # --- big state banner --------------------------------------------
        self.state_var = tk.StringVar(value="no signal")
        self.state_lbl = tk.Label(root, textvariable=self.state_var, bg=_GRAY, fg="white",
                                  font=("DejaVu Sans", 18, "bold"), pady=12)
        self.state_lbl.pack(fill="x", padx=8, pady=2)

        # --- primary toggle ------------------------------------------------
        self.primary = self._button(root, "ENGAGE", self.on_primary, _GREEN, _GREEN_HI, big=True)
        self.primary.pack(fill="x", padx=8, pady=4)

        # --- gripper row ---------------------------------------------------
        grow = tk.Frame(root, bg=_BG)
        grow.pack(fill="x", padx=8, pady=2)
        tk.Label(grow, text="Gripper:", bg=_BG, font=("DejaVu Sans", 11, "bold")).pack(side="left")
        self.grip_pause_btn = self._button(grow, "Gripper PAUSE", self.on_gripper_pause, _RED, _RED_HI)
        self.grip_pause_btn.pack(side="left", padx=4)
        self.grip_resume_btn = self._button(grow, "Gripper Resume", self.on_gripper_resume)
        self.grip_resume_btn.pack(side="left", padx=4)

        # --- pos_scale slider (A안: pending until the next engage) ----------
        sbox = tk.LabelFrame(root, text="Sensitivity — pos_scale (DPI). 회전은 1:1 (스케일 안 함)",
                             bg=_BG, font=("DejaVu Sans", 10, "bold"))
        sbox.pack(fill="x", padx=8, pady=4)
        self.scale_var = tk.IntVar(value=100)
        self.scale = tk.Scale(sbox, from_=10, to=100, orient="horizontal", resolution=1,
                              variable=self.scale_var, showvalue=False, bg=_BG,
                              command=lambda _v: self._update_scale_label(),
                              troughcolor="#cfd8dc", highlightthickness=0, takefocus=False)
        self.scale.pack(fill="x", padx=6)
        self.scale.bind("<ButtonRelease-1>", lambda _e: self.on_pos_scale_commit())
        self.scale_lbl_var = tk.StringVar(value="")
        tk.Label(sbox, textvariable=self.scale_lbl_var, bg=_BG,
                 font=("DejaVu Sans", 10, "bold")).pack(anchor="w", padx=6)
        self._update_scale_label()

        # --- control mode + scene row --------------------------------------
        mrow = tk.Frame(root, bg=_BG)
        mrow.pack(fill="x", padx=8, pady=2)
        tk.Label(mrow, text="control_mode:", bg=_BG, font=("DejaVu Sans", 11, "bold")).pack(side="left")
        self.mode_var = tk.StringVar(value="eef")
        self.mode_buttons = {}
        for mode in ("eef", "joint"):
            rb = tk.Radiobutton(mrow, text=mode, value=mode, variable=self.mode_var, bg=_BG,
                                command=lambda m=mode: self.on_set_mode(m),
                                font=("DejaVu Sans", 11), takefocus=False,
                                highlightthickness=0, selectcolor="#ffffff")
            rb.pack(side="left", padx=2)
            self.mode_buttons[mode] = rb
        tk.Label(mrow, text="  (engage 중에는 잠김)", bg=_BG, fg="#666666",
                 font=("DejaVu Sans", 9, "italic")).pack(side="left")

        self.home_btn = self._button(mrow, "HOME", self.on_home)
        self.home_btn.pack(side="right", padx=4)
        self.reset_btn = self._button(mrow, "RESET SCENE", self.on_reset_scene)
        self.reset_btn.pack(side="right", padx=4)
        self.seed_var = tk.StringVar(value="")
        self.seed_entry = tk.Entry(mrow, textvariable=self.seed_var, width=8,
                                   font=("DejaVu Sans", 10))
        self.seed_entry.pack(side="right")
        # Clicking the entry steals focus and would otherwise kill every shortcut for the
        # rest of the session: Return and Escape hand focus back to the window (Escape
        # also clears what was typed).
        self.seed_entry.bind("<Return>", lambda e: self._blur_entry())
        self.seed_entry.bind("<KP_Enter>", lambda e: self._blur_entry())
        self.seed_entry.bind("<Escape>", lambda e: self._blur_entry(clear=True))
        tk.Label(mrow, text="seed:", bg=_BG, font=("DejaVu Sans", 10)).pack(side="right", padx=(8, 2))

        # --- recording block -------------------------------------------------
        rbox = tk.LabelFrame(root, text="녹화 (recording)", bg=_BG,
                             font=("DejaVu Sans", 10, "bold"))
        rbox.pack(fill="x", padx=8, pady=4)
        rrow = tk.Frame(rbox, bg=_BG)
        rrow.pack(fill="x", padx=4, pady=2)
        self.start_btn = self._button(rrow, "START TAKE", self.on_start_take, _GREEN, _GREEN_HI)
        self.start_btn.pack(side="left", padx=2)
        self.stop_btn = self._button(rrow, "STOP TAKE", self.on_stop_take, _BLUE, _BLUE_HI)
        self.stop_btn.pack(side="left", padx=2)
        self.discard_btn = self._button(rrow, "DISCARD LAST", self.on_discard_last)
        self.discard_btn.pack(side="left", padx=2)
        self.take_var = tk.StringVar(value="Take: 0")
        tk.Label(rrow, textvariable=self.take_var, bg=_BG,
                 font=("DejaVu Sans", 12, "bold")).pack(side="left", padx=10)
        self.rec_var = tk.StringVar(value="PREVIEW")
        self.rec_lbl = tk.Label(rrow, textvariable=self.rec_var, bg=_BG, fg="#333333",
                                font=("DejaVu Sans", 12, "bold"))
        self.rec_lbl.pack(side="left", padx=6)
        self.elapsed_var = tk.StringVar(value="")
        tk.Label(rrow, textvariable=self.elapsed_var, bg=_BG,
                 font=("DejaVu Sans", 11)).pack(side="left", padx=6)
        self.counts_var = tk.StringVar(value="frames cam1 0 / cam2 0 · rows 0")
        tk.Label(rbox, textvariable=self.counts_var, bg=_BG, anchor="w",
                 font=("DejaVu Sans", 10)).pack(fill="x", padx=6)
        self.rows_var = tk.StringVar(value="streams: —")
        tk.Label(rbox, textvariable=self.rows_var, bg=_BG, anchor="w", fg="#555555",
                 font=("DejaVu Sans", 9)).pack(fill="x", padx=6)
        self.take_dir_var = tk.StringVar(value="take dir: —")
        tk.Label(rbox, textvariable=self.take_dir_var, bg=_BG, anchor="w", fg="#555555",
                 font=("DejaVu Sans", 9)).pack(fill="x", padx=6)

        # --- camera previews --------------------------------------------------
        prow = tk.Frame(root, bg=_BG)
        prow.pack(fill="x", padx=8, pady=4)
        self.cam_labels = {}
        self.cam_status = {}
        for cam, human in (("cam1", "cam1 — scene"), ("cam2", "cam2 — wrist")):
            box = tk.LabelFrame(prow, text=human, bg=_BG, font=("DejaVu Sans", 10, "bold"))
            box.pack(side="left", padx=4)
            lbl = tk.Label(box, text="no signal", width=44, height=11, bg="#263238",
                           fg="white", font=("DejaVu Sans", 10))
            lbl.pack()
            self.cam_labels[cam] = lbl
            var = tk.StringVar(value="no signal")
            tk.Label(box, textvariable=var, bg=_BG, font=("DejaVu Sans", 9)).pack(anchor="w")
            self.cam_status[cam] = var

        # --- log pane ---------------------------------------------------------
        lbox = tk.LabelFrame(root, text="log (서버 응답)", bg=_BG, font=("DejaVu Sans", 10, "bold"))
        lbox.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self.log_text = tk.Text(lbox, height=8, wrap="none", state="disabled",
                                font=("DejaVu Sans Mono", 9), bg="#ffffff")
        sb = tk.Scrollbar(lbox, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        hint = ("keys: space = ENGAGE/DISENGAGE · r = START/STOP TAKE · "
                "n = RESET SCENE · h = HOME")
        tk.Label(root, text=hint, bg=_BG, fg="#555555",
                 font=("DejaVu Sans", 9, "italic")).pack(anchor="w", padx=10, pady=(0, 6))

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    #: shortcut → action name, action
    _SHORTCUTS = (("space", "primary"), ("r", "take"), ("n", "reset"), ("h", "home"))

    def _bind_keys(self) -> None:
        """Bind the four shortcuts, once per key, HELD-KEY SAFE.

        X11 auto-repeat turns a held key into a stream of KeyPress events (and, without
        detectable auto-repeat, KeyPress/KeyRelease pairs). Holding <space> would then be
        engage → disengage → engage… So a press only fires when the key is not already
        down, a release only clears that flag after a 40 ms grace period (an auto-repeat's
        release is immediately followed by its next press, which cancels the clear), and a
        hard KEY_DEBOUNCE_S backstop covers the rest. The backstop is deliberately shorter
        than a human double-tap (X11 repeats every ~30 ms; people tap at >150 ms) so the
        two-click ENGAGE confirm is never silently swallowed.
        """
        actions = {"primary": self.on_primary, "take": self.on_toggle_take,
                   "reset": self.on_reset_scene, "home": self.on_home}
        self._key_down: Dict[str, bool] = {}
        self._key_release_job: Dict[str, Optional[str]] = {}
        self._key_last: Dict[str, float] = {}
        for key, name in self._SHORTCUTS:
            fn = actions[name]
            for spec in ({"space": ("space",)}.get(key, (key, key.upper()))):
                self.root.bind(f"<KeyPress-{spec}>",
                               lambda e, k=key, f=fn: self._on_key_press(k, f))
                self.root.bind(f"<KeyRelease-{spec}>", lambda e, k=key: self._on_key_release(k))

    def _on_key_press(self, key: str, fn) -> str:
        job = self._key_release_job.pop(key, None)
        if job is not None:                     # an auto-repeat, not a fresh press
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        if self._key_down.get(key):
            return "break"
        self._key_down[key] = True
        now = time.monotonic()
        if now - self._key_last.get(key, 0.0) < KEY_DEBOUNCE_S:
            return "break"
        self._key_last[key] = now
        return self._key(fn)

    def _on_key_release(self, key: str) -> str:
        def clear():
            self._key_release_job.pop(key, None)
            self._key_down[key] = False
        try:
            self._key_release_job[key] = self.root.after(40, clear)
        except tk.TclError:
            clear()
        return "break"

    def _blur_entry(self, clear: bool = False) -> str:
        if clear:
            self.seed_var.set("")
        self.root.focus_set()
        return "break"

    def _key(self, fn) -> str:
        """Run a shortcut unless the operator is typing into the seed entry."""
        try:
            if isinstance(self.root.focus_get(), tk.Entry):
                return ""
        except (tk.TclError, KeyError):
            pass
        fn()
        return "break"

    # ------------------------------------------------------------- logging --
    def log(self, msg: str) -> None:
        if not msg:
            return
        line = f"{time.strftime('%H:%M:%S')}  {msg}\n"
        try:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line)
            n = int(self.log_text.index("end-1c").split(".")[0])
            if n > LOG_LINES:
                self.log_text.delete("1.0", f"{n - LOG_LINES}.0")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        except tk.TclError:
            pass

    def log_lines(self) -> list:
        """The log pane's current contents, for tests."""
        try:
            return [l for l in self.log_text.get("1.0", "end-1c").split("\n") if l]
        except tk.TclError:
            return []

    # ------------------------------------------------------------ IPC glue --
    def _call(self, client: ipc.Client, cmd: str, **kw) -> Dict[str, Any]:
        """Synchronous command call, bounded by the client's timeout. Logs the reply."""
        rep = client.call(cmd, **kw)
        who = "sim" if client in (self.sim, self.sim_poll) else "capture"
        msg = rep.get("msg") or ("ok" if rep.get("ok") else "failed")
        self.log(f"[{who}] {cmd}: {msg}")
        if rep.get("timeout"):
            self._set_connected(who, False)
        return rep

    def call_sim(self, cmd: str, **kw) -> Dict[str, Any]:
        rep = self._call(self.sim, cmd, **kw)
        self.poll_sim()          # a command changes state: re-read it at once
        self.refresh()
        return rep

    def call_capture(self, cmd: str, **kw) -> Dict[str, Any]:
        rep = self._call(self.cap, cmd, **kw)
        self.poll_capture()
        self.refresh()
        return rep

    def _set_connected(self, who: str, ok: bool) -> None:
        was = self.sim_connected if who == "sim" else self.cap_connected
        if who == "sim":
            self.sim_connected = ok
        else:
            self.cap_connected = ok
        if ok and not was:
            self.log(f"[{who}] connected")
            if who == "sim":
                self._sync_from_sim()

    def _sync_from_sim(self) -> None:
        """Adopt the server's own pos_scale on a (re)connect — handles sim restarts."""
        ps = self.sim_status.get("pos_scale")
        if isinstance(ps, (int, float)) and 0.0 < float(ps) <= 1.0:
            self.scale_var.set(int(round(float(ps) * 100)))
            self._update_scale_label()

    def poll_sim(self) -> Dict[str, Any]:
        rep = self.sim_poll.call("get_status")
        if rep.get("ok"):
            self.sim_status = rep
            self._set_connected("sim", True)
            msg = rep.get("msg")
            if msg and self._last_poll_msg.get("sim") != msg:
                self._last_poll_msg["sim"] = msg
                self.log(f"[sim] {msg}")
        else:
            self._set_connected("sim", False)
        return rep

    def poll_capture(self) -> Dict[str, Any]:
        rep = self.cap_poll.call("get_status")
        if rep.get("ok"):
            self.cap_status = rep
            self._set_connected("capture", True)
            msg = rep.get("msg")
            if msg and self._last_poll_msg.get("capture") != msg:
                self._last_poll_msg["capture"] = msg
                self.log(f"[capture] {msg}")
        else:
            self._set_connected("capture", False)
        return rep

    # ----------------------------------------------------------- scheduling --
    def start_polling(self) -> None:
        self._schedule("sim", POLL_MS, self._tick_sim)
        self._schedule("cap", POLL_MS // 2, self._tick_capture)   # staggered
        self._schedule("prev", PREVIEW_MS, self._tick_preview)

    def _schedule(self, key: str, ms: int, fn) -> None:
        if self._closed:
            return
        try:
            self._after_ids[key] = self.root.after(ms, fn)
        except tk.TclError:
            self._closed = True

    def _tick_sim(self) -> None:
        if self._closed:
            return
        self.poll_sim()
        self.refresh()
        self._schedule("sim", POLL_MS if self.sim_connected else POLL_SLOW_MS, self._tick_sim)

    def _tick_capture(self) -> None:
        if self._closed:
            return
        self.poll_capture()
        self.refresh()
        self._schedule("cap", POLL_MS if self.cap_connected else POLL_SLOW_MS, self._tick_capture)

    def _tick_preview(self) -> None:
        if self._closed:
            return
        self.pump_preview()
        self._schedule("prev", PREVIEW_MS, self._tick_preview)

    def pump_preview(self) -> bool:
        """Drain the preview SUB and repaint. Returns True if a new frame arrived."""
        msg = self.preview.latest()
        if msg is None:
            return False
        self.preview_msg = msg
        for cam in ("cam1", "cam2"):
            ppm = jpeg_to_ppm(msg.get(cam) or b"")
            if ppm is None:
                self.cam_status[cam].set("no signal")
                continue
            try:
                photo = tk.PhotoImage(master=self.root, data=ppm)
            except tk.TclError:
                self.cam_status[cam].set("decode failed")
                continue
            self._photo[cam] = photo          # keep a reference or Tk drops the image
            self.cam_labels[cam].configure(image=photo, text="", width=0, height=0)
            frames = (msg.get("frames") or {}).get(cam)
            self.cam_status[cam].set(
                f"{photo.width()}×{photo.height()}"
                + (f" · frames {frames}" if frames is not None else ""))
        return True

    # -------------------------------------------------- two-click confirm --
    def _armed_click(self, key: str, action) -> bool:
        """First call arms (orange, 3 s); second fires. Returns True if it fired."""
        if self._armed.get(key) is not None:
            self._disarm(key)
            action()
            return True
        self._armed[key] = self.root.after(CONFIRM_MS, lambda: self._disarm(key, refresh=True))
        self.refresh()
        return False

    def _disarm(self, key: str, refresh: bool = False) -> None:
        aid = self._armed.pop(key, None)
        if aid is not None:
            try:
                self.root.after_cancel(aid)
            except tk.TclError:
                pass
        if refresh:
            self.refresh()

    def is_armed(self, key: str) -> bool:
        return self._armed.get(key) is not None

    # ------------------------------------------------------------- actions --
    @property
    def engaged(self) -> bool:
        st = self.sim_status
        if "engaged" in st:
            return bool(st.get("engaged"))
        return str(st.get("eef_state", "")).upper() == "ENGAGED"

    @property
    def recording(self) -> bool:
        if "recording" in self.cap_status:
            return bool(self.cap_status.get("recording"))
        return bool(self.preview_msg.get("recording"))

    def on_primary(self) -> None:
        """DISENGAGE is single-click (safe); ENGAGE needs two clicks in 3 s (motion)."""
        if self.engaged:
            self._disarm("primary")
            self.call_sim("disengage")
            return
        self._armed_click("primary", self._do_engage)

    def _do_engage(self) -> None:
        # A안: the slider is PENDING until an anchor-reset moment. Commit it right
        # before the engage so the new reference uses it, never mid-stroke.
        self.on_pos_scale_commit(quiet=True)
        self.call_sim("engage")

    def on_pos_scale_commit(self, quiet: bool = False) -> Dict[str, Any]:
        value = round(self.scale_var.get() / 100.0, 2)
        rep = self._call(self.sim, "set_pos_scale", value=value)
        if not quiet:
            self.poll_sim()
            self.refresh()
        return rep

    def on_set_mode(self, mode: str) -> Optional[Dict[str, Any]]:
        if self.engaged:
            self.log("[gui] control_mode 변경 거부 — ENGAGED 중에는 바꿀 수 없다 (DISENGAGE 먼저)")
            self.mode_var.set(str(self.sim_status.get("control_mode", mode)))
            return None
        return self.call_sim("set_control_mode", mode=mode)

    def on_gripper_pause(self) -> Dict[str, Any]:
        return self.call_sim("gripper_pause")

    def on_gripper_resume(self) -> None:
        # Resume seeds/ramps the gripper to the leader → confirm, like the real GUI.
        self._armed_click("grip_resume", lambda: self.call_sim("gripper_resume"))

    def on_reset_scene(self) -> Optional[Dict[str, Any]]:
        if self.recording:
            self.log("[gui] RESET SCENE 거부 — 녹화 중이다. STOP TAKE 먼저.")
            return None
        seed_txt = self.seed_var.get().strip()
        kw: Dict[str, Any] = {}
        if seed_txt:
            try:
                kw["seed"] = int(seed_txt)
            except ValueError:
                self.log(f"[gui] seed {seed_txt!r} 는 정수가 아니다 — 무시하고 랜덤 시드로 간다")
        return self.call_sim("reset_scene", **kw)

    def on_home(self) -> Dict[str, Any]:
        return self.call_sim("home")

    def on_start_take(self) -> Optional[Dict[str, Any]]:
        if self.recording:
            self.log("[gui] 이미 녹화 중이다")
            return None
        return self.call_capture("start_take")

    def on_stop_take(self) -> Optional[Dict[str, Any]]:
        if not self.recording:
            self.log("[gui] 녹화 중이 아니다")
            return None
        return self.call_capture("stop_take")

    def on_toggle_take(self) -> None:
        (self.on_stop_take if self.recording else self.on_start_take)()

    def on_discard_last(self) -> None:
        if self.recording:
            self.log("[gui] DISCARD LAST 거부 — 녹화 중이다. STOP TAKE 먼저.")
            return
        self._armed_click("discard", lambda: self.call_capture("discard_last_take"))

    # ------------------------------------------------------------- refresh --
    def refresh(self) -> None:
        """Repaint every widget from the latest snapshots. Cheap, idempotent, no IPC."""
        try:
            self._refresh_conn()
            self._refresh_state()
            self._refresh_buttons()
            self._refresh_recording()
            self._update_scale_label()
        except tk.TclError:
            self._closed = True

    def _refresh_conn(self) -> None:
        for who, var, lbl, ok in (
            ("sim", self.sim_conn_var, self.sim_conn_lbl, self.sim_connected),
            ("capture", self.cap_conn_var, self.cap_conn_lbl, self.cap_connected),
        ):
            var.set(f"{who}: {'CONNECTED' if ok else 'DISCONNECTED'}")
            lbl.configure(bg="#22aa22" if ok else _RED)

        st = self.sim_status
        info = st.get("eef_info") or {}
        # sim_main's get_status carries the reason inside eef_info (controller.info());
        # last_gate/last_gate_detail is the "why was my engage refused" pair.
        reason = (info.get("reject_reason") or info.get("auto_reason")
                  or st.get("reject_reason") or st.get("auto_reason"))
        gate, gate_detail = info.get("last_gate"), info.get("last_gate_detail")
        if not reason and gate:
            reason = f"{gate}: {gate_detail}" if gate_detail else str(gate)
        sigma = info.get("sigma_min", st.get("sigma_min"))
        self.info_var.set(
            f"mode {st.get('control_mode', '—')} · state {st.get('eef_state', '—')} · "
            f"pos_scale {_fmt(st.get('pos_scale'), 2)} · sigma_min {_fmt(sigma)} · "
            f"reason {reason or '—'}")

        self._refresh_sim_health(st, info)
        self._refresh_capture_health()

        # task: sim_main sends a task_success BOOL (get_status); the state message's
        # {"success", "detail"} dict only reaches us via a future key, so accept both.
        task = st.get("task") if isinstance(st.get("task"), dict) else {}
        success = st.get("task_success", task.get("success"))
        detail = st.get("task_detail") or task.get("detail") or ""
        if not self.sim_connected:
            self.task_var.set("TASK —")
            self.task_lbl.configure(bg=_GRAY)
        elif success:
            self.task_var.set(("TASK SUCCESS — " + detail).strip(" —-"))
            self.task_lbl.configure(bg="#22aa22")
        else:
            self.task_var.set(f"task: {detail or 'not yet'}")
            self.task_lbl.configure(bg=_SLATE)

    def _refresh_sim_health(self, st: Dict[str, Any], info: Dict[str, Any]) -> None:
        if not self.sim_connected:
            self.sim_health_var.set("sim —")
            return
        bits = [f"tick {st.get('tick', '—')}"]
        rt = st.get("rt_ratio")
        if rt is not None:
            bits.append(f"rt {float(rt):.2f}×")
        if st.get("leader_fake"):
            bits.append("LEADER=FAKE")
        elif "leader_ok" in st:
            bits.append("leader ok" if st["leader_ok"] else "LEADER STALE")
        if st.get("gripper_paused"):
            bits.append("GRIPPER PAUSED")
        elif st.get("gripper_ramping"):
            bits.append("gripper ramping")
        if st.get("gripper_mode"):
            bits.append(f"grip {st['gripper_mode']}")
        if st.get("layout_seed") is not None:
            bits.append(f"seed {st['layout_seed']}")
        if info.get("soft_start_active"):
            bits.append("soft-start")
        if st.get("viewer") is False:
            bits.append("no viewer")
        note = st.get("startup_note")
        if note:
            bits.append(str(note))
        self.sim_health_var.set("sim: " + " · ".join(bits))

    def _refresh_capture_health(self) -> None:
        """Mirror capture.CaptureApp.status(): scene/workers/sim_alive, the per-camera
        render stats, and — the important one — the recorder's OWN `problems` verdict."""
        cs = self.cap_status
        if not self.cap_connected:
            self.cap_health_var.set("capture —")
            self.drops_var.set("")
            return
        cap = cs.get("capture") or {}
        bits = ["scene ready" if cs.get("scene_ready") else "SCENE NOT READY"]
        workers = cs.get("workers") or {}
        if workers:
            dead = [c for c, alive in workers.items() if not alive]
            bits.append("workers ok" if not dead else "WORKERS DEAD: " + ",".join(sorted(dead)))
        if "sim_alive" in cs:
            bits.append("sim alive" if cs["sim_alive"] else "NO SIM STATE")
        for cam, r in sorted((cs.get("render") or {}).items()):
            if isinstance(r, dict):
                bits.append(f"{cam} {_fmt(r.get('hz'), 1)} Hz / {_fmt(r.get('render_ms'), 1)} ms")
        if cap.get("achieved_fps"):
            got = "/".join(_fmt(v, 1) for _, v in sorted(cap["achieved_fps"].items()))
            nominal = _fmt(cap.get("nominal_fps"), 0)
            bits.append(f"fps {got} of {nominal}" + ("" if cap.get("fps_ok", True) else " SLOW"))
        self.cap_health_var.set("capture: " + " · ".join(bits))

        # The recorder counts its own anomalies and names them in `problems` (empty =
        # clean). That verdict outranks our arithmetic, so it wins the label.
        problems = [str(p) for p in (cs.get("problems") or [])]
        if problems and problems != self._last_problems:
            for p in problems:
                self.log(f"[capture] PROBLEM: {p}")
        self._last_problems = problems

        counts = cs.get("counts") if isinstance(cs.get("counts"), dict) else {}
        rows = cs.get("rows") if isinstance(cs.get("rows"), dict) else {}
        drops: Dict[str, int] = {}
        for key in ("state_dropped", "frames_dropped", "write_errors", "missed_ticks",
                    "tick_restarts", "worker_queue_drops"):
            v = counts.get(key, rows.get(key, cs.get(key)))
            if isinstance(v, (int, float)):
                drops[key] = int(v)
        if isinstance(cs.get("bad_msgs"), int):
            drops["bad_msgs"] = cs["bad_msgs"]
        for cam, r in sorted((cs.get("render") or {}).items()):
            if isinstance(r, dict):
                for key in ("queue_drops", "bad_state", "dropped"):
                    if r.get(key):
                        drops[f"{cam}_{key}"] = int(r[key])
        bad = {k: v for k, v in drops.items() if v}
        if problems:
            self.drops_var.set(f"PROBLEMS ({len(problems)}): " + " · ".join(problems)[:120])
            self.drops_lbl.configure(fg=_RED)
        elif bad:
            self.drops_var.set("DROPS " + " · ".join(f"{k} {v}" for k, v in sorted(bad.items())))
            self.drops_lbl.configure(fg=_RED)
        elif not cap.get("fps_ok", True):
            self.drops_var.set("render below nominal fps")
            self.drops_lbl.configure(fg="#dd8800")
        else:
            self.drops_var.set("drops 0" if drops else "")
            self.drops_lbl.configure(fg="#2e7d32")

    def _refresh_state(self) -> None:
        if not self.sim_connected:
            self.state_var.set("no signal — sim_rep 응답 없음 (재시도 중)")
            self.state_lbl.configure(bg=_GRAY)
            return
        name = str(self.sim_status.get("eef_state", "") or "")
        colour, text = _STATE_STYLE.get(name.upper(), (_GRAY, f"{name or 'unknown'}"))
        self.state_lbl.configure(bg=colour)
        self.state_var.set(text)

    def _style(self, btn, bg, hi, enabled: bool) -> None:
        btn.configure(state="normal" if enabled else "disabled",
                      bg=bg if enabled else _DISABLED_BG,
                      activebackground=hi, fg="white" if enabled else _DISABLED_FG)

    def _refresh_buttons(self) -> None:
        live = self.sim_connected
        if self.engaged:
            self._disarm("primary")
            self.primary.configure(text="DISENGAGE (clutch up)")
            self._style(self.primary, _BLUE, _BLUE_HI, live)
        elif self.is_armed("primary"):
            self.primary.configure(text="CONFIRM ENGAGE (3 s) — 팔이 움직인다")
            self._style(self.primary, _ORANGE, _ORANGE_HI, live)
        else:
            self.primary.configure(text="ENGAGE  (press to start following)")
            self._style(self.primary, _GREEN, _GREEN_HI, live)

        self._style(self.grip_pause_btn, _RED, _RED_HI, live)
        if self.is_armed("grip_resume"):
            self.grip_resume_btn.configure(text="Click AGAIN to resume gripper")
            self._style(self.grip_resume_btn, _ORANGE, _ORANGE_HI, live)
        else:
            self.grip_resume_btn.configure(text="Gripper Resume")
            self._style(self.grip_resume_btn, _SLATE, _SLATE_HI, live)

        for mode, rb in self.mode_buttons.items():
            rb.configure(state="normal" if (live and not self.engaged) else "disabled")
        srv_mode = self.sim_status.get("control_mode")
        if srv_mode in self.mode_buttons and srv_mode != self.mode_var.get():
            self.mode_var.set(srv_mode)          # server is the authority (sim restarts)

        self._style(self.reset_btn, _SLATE, _SLATE_HI, live and not self.recording)
        self._style(self.home_btn, _SLATE, _SLATE_HI, live)

    def _refresh_recording(self) -> None:
        cs = self.cap_status
        live = self.cap_connected
        rec = self.recording
        self._style(self.start_btn, _GREEN, _GREEN_HI, live and not rec)
        self._style(self.stop_btn, _BLUE, _BLUE_HI, live and rec)
        if self.is_armed("discard"):
            self.discard_btn.configure(text="Click AGAIN to DISCARD")
            self._style(self.discard_btn, _ORANGE, _ORANGE_HI, live and not rec)
        else:
            self.discard_btn.configure(text="DISCARD LAST")
            self._style(self.discard_btn, _SLATE, _SLATE_HI, live and not rec)

        # take_index is the recorder's own per-process counter (recorder.status()).
        n = cs.get("take_index", self.preview_msg.get("take_index", 0))
        self.take_var.set(f"Take: {n} (recording)" if rec else f"Take: {n}")
        if rec:
            self.rec_var.set(f"● RECORDING take {n}")
            self.rec_lbl.configure(fg=_RED)
            dur = cs.get("duration_s", self.preview_msg.get("duration_s")) or 0.0
            self.elapsed_var.set(f"Elapsed: {float(dur):.1f}s")
        else:
            self.rec_var.set("PREVIEW" if live else "—")
            self.rec_lbl.configure(fg="#333333")
            self.elapsed_var.set("")

        frames = cs.get("frames") or (self.preview_msg.get("frames") or {})
        depth = cs.get("depth_frames") or {}
        rows = cs.get("rows", self.preview_msg.get("rows"))
        counts = rows if isinstance(rows, dict) else {}
        rec_counts = cs.get("counts") if isinstance(cs.get("counts"), dict) else {}
        state_msgs = cs.get("state_msgs", rec_counts.get("state_msgs"))
        # `rows` is a DICT of table→row count in the real recorder (recorder.status()),
        # and it also carries the recorder's own counters — show the tables, sum them,
        # and never assume a scalar.
        # `rows` is RecordingSession._counts: one counter per recorded STREAM plus the
        # frame counters (already shown above), never a scalar.
        streams = {k: v for k, v in counts.items()
                   if isinstance(v, int) and not k.endswith("_frames")
                   and k not in ("state_msgs", "state_dropped", "frames_dropped")}
        total = sum(streams.values()) if streams else (rows if isinstance(rows, int) else 0)
        self.counts_var.set(
            f"frames cam1 {frames.get('cam1', 0)} / cam2 {frames.get('cam2', 0)}"
            + (f" · depth {depth.get('cam1', 0)}/{depth.get('cam2', 0)}" if depth else "")
            + f" · rows Σ{total}"
            + (f" · state msgs {state_msgs}" if state_msgs is not None else ""))
        self.rows_var.set(
            "streams: " + (" · ".join(f"{k} {v}" for k, v in sorted(streams.items())) if streams else "—"))

        take_dir = (cs.get("take_dir") or cs.get("last_take_dir")
                    or self.preview_msg.get("take_dir") or "")
        root_dir = cs.get("root")
        self.take_dir_var.set(f"take dir: {take_dir or '—'}"
                              + (f"   (root {root_dir})" if root_dir else ""))

    def _update_scale_label(self) -> None:
        pending = self.scale_var.get() / 100.0
        active = self.sim_status.get("pos_scale")
        tail = "" if active is None else f" · active {float(active):.2f}"
        if self.engaged:
            self.scale_lbl_var.set(
                f"pending {pending:.2f} → applies on next engage (ENGAGED 중에는 안 밀어 넣는다){tail}")
        else:
            self.scale_lbl_var.set(f"pending {pending:.2f} — 다음 engage 에 커밋된다{tail}")

    # --------------------------------------------------------------- lifecycle --
    def pump(self, seconds: float = 0.2) -> None:
        """Run the Tk loop for `seconds` without blocking a test. Tests use this."""
        end = time.time() + seconds
        while time.time() < end:
            try:
                self.root.update()
            except tk.TclError:
                return
            time.sleep(0.01)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for aid in list(self._after_ids.values()) + list(self._armed.values()):
            if aid:
                try:
                    self.root.after_cancel(aid)
                except tk.TclError:
                    pass
        for c in (self.sim, self.cap, self.sim_poll, self.cap_poll):
            c.close()
        self.preview.close()
        if self._owns_root:
            try:
                self.root.destroy()
            except tk.TclError:
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="sim_collect operator GUI (DESIGN.md §2.3)")
    p.add_argument("--sim-timeout-ms", type=int, default=1500,
                   help="REQ timeout for operator COMMANDS (engage runs gates server-side)")
    p.add_argument("--status-timeout-ms", type=int, default=120,
                   help="REQ timeout for the 5 Hz get_status polls (keeps the Tk loop free)")
    p.add_argument("--title", default="sim_collect — GELLO ▸ MuJoCo 데이터 수집")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    app = SimCollectGui(sim_timeout_ms=args.sim_timeout_ms,
                        status_timeout_ms=args.status_timeout_ms,
                        title=args.title)
    # Non-fatal reachability probe; the GUI keeps retrying either way.
    probe = ipc.Client("sim_rep", timeout_ms=300)
    app.log("[gui] sim_rep " + ("up" if ipc.wait_for(probe, 1.0) else "down — 재시도 중"))
    probe.close()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
