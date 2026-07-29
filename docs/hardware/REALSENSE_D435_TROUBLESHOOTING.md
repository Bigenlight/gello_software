# Intel RealSense D435 — Intermittent "Frame didn't arrive" Timeouts

## Symptoms
- Device enumerates fine (`lsusb`, `rs-enumerate-devices`) but streaming intermittently fails with `RuntimeError: Frame didn't arrive within 5000/8000` (pyrealsense2 `wait_for_frames()` or the ROS2 `realsense2_camera` node).
- Color-only streams usually succeed; combined **depth+color** streams fail more often (~2/3 of attempts).
- Separate sub-symptom: device sometimes drops into USB recovery/DFU mode — `lsusb` shows `8086:0adb "RS400 Device"` instead of the normal `8086:0b07 "RealSense D435"`. Triggered by `hardware_reset()` or a client crashing mid-stream. Fixed by a physical USB replug (unrelated to the quirks fix below).

## Root Cause
A stray modprobe file `/etc/modprobe.d/uvcvideo-actioncam.conf` (added for an unrelated USB action camera) set `options uvcvideo quirks=0x12`. The `uvcvideo` kernel driver is **shared by all UVC cameras**, so this applies globally — including the RealSense.

`quirks=0x12` = `UVC_QUIRK_PROBE_MINMAX (0x02) | UVC_QUIRK_IGNORE_SELECTOR_UNIT (0x10)`, which alters UVC format probe/commit negotiation. Kernel logs showed `uvcvideo 4-1:1.1: Unknown video format ...` and `4-1:1.4: Unknown video format ...` on the RealSense depth/IR interfaces — the forced quirk interacting with these already-unrecognized format GUIDs caused the intermittent negotiation/frame-delivery failures.

## Diagnosis
```bash
cat /sys/module/uvcvideo/parameters/quirks        # nonzero (not 4294967295 = 0xFFFFFFFF "unset") => quirks forced globally
grep -r quirks /etc/modprobe.d/                    # find the offending .conf file
journalctl -k | grep -i "unknown video format"     # confirm RealSense interfaces hit unrecognized formats
```

## Fix
```bash
sudo mv /etc/modprobe.d/<offending-file>.conf /etc/modprobe.d/<offending-file>.conf.disabled
sudo rmmod uvcvideo
sudo modprobe uvcvideo
```
Verify `cat /sys/module/uvcvideo/parameters/quirks` returns to the default sentinel (`4294967295`), then re-test. After the fix: 5/5 consecutive depth+color pipeline starts and a 20s continuous stream with zero stalls (vs. failing ~2/3 before).

## Notes
- **udev over modprobe:** if the action camera (or any device) genuinely needs a quirk, scope it to that device via a udev rule matching its vendor/product ID rather than a global `options uvcvideo` line, so it can't affect the RealSense.
- **RT kernel (secondary factor):** this machine runs a custom PREEMPT_RT kernel (`6.8.2-rt11`); RT scheduling can affect USB isochronous transfer timing. Contributing, not the main fix.
- **DFU mode:** if `lsusb` shows `8086:0adb` instead of `8086:0b07`, that's the recovery-mode issue — physically replug the USB cable; unrelated to quirks.

---

# A different failure: the camera never comes up and *nothing tells you why*

This one is not a streaming problem. The device is healthy; it was asked for by a
serial number that does not exist.

## Symptom

Everything looks like a dead camera, from every angle except one:

- `realsense2_camera` starts normally and **does not exit**. It stays up and retries forever.
- `ros2 topic info /camN/camN/color/image_raw/compressed` reports **Publisher count: 1**.
- `ros2 topic hz` on that topic prints nothing.
- Every consumer (viewer, recorder, policy leader, `ur7e_env.get_im()`) reports
  **"no frame" / stale frame**, never "wrong serial".

**Nothing in the stack distinguishes an unplugged camera from a misconfigured one.**
The single place the truth appears is the camera node's own launch log:

```
[cam1.cam1]: Device with serial number 147122072740 was found.
[cam1.cam1]: Device with serial number 243222072700 was found.
[ERROR] [cam1.cam1]: The requested device with serial number 151623020789 is NOT found. Will Try again.
```

(Real capture, `/tmp/launch_cameras_20260728_162820/cam1_launch.log`. Note the node
*enumerated both connected cameras* and still could not satisfy the request.)

## Root cause: each D400 device carries TWO different serial numbers

This is the trap. One physical camera reports one serial to librealsense and a
**different** serial to the USB stack:

| Identifier | Where you see it | Value on this rig |
|---|---|---|
| **Module serial** (`RS2_CAMERA_INFO_SERIAL_NUMBER`) | `rs-enumerate-devices -s`, `pyrealsense2`, and **what `serial_no:=` matches** | D435 `147122072740`, D435IF `243222072700` |
| **ASIC serial** (= "Firmware Update Id") | USB string descriptor → `lsusb -v`, `/sys/bus/usb/devices/*/serial`, `journalctl -k` | D435 `151623020789`, D435IF `322743060038` |

Verify that these are the *same two devices*, not two pairs, by matching the physical port:

```bash
rs-enumerate-devices | grep -E "Name|Serial Number|Asic|Physical Port"
for d in /sys/bus/usb/devices/*/serial; do
  s=$(cat "$d" 2>/dev/null); case "$s" in 1471*|2432*|1516*|3227*)
    echo "$(dirname "$d") -> $s ($(cat "$(dirname "$d")"/product 2>/dev/null))";; esac
done
```

Verified 2026-07-29: `4-4.1` is the plain D435 — module `147122072740`, ASIC `151623020789`;
`4-4.3` is the D435IF — module `243222072700`, ASIC `322743060038`. **One pair of cameras,
two identifier namespaces.**

> ### How this bit us
> Reading the kernel journal (which only ever shows ASIC serials) led to the conclusion
> that the cameras had been physically swapped, and the ASIC serials were pasted into the
> launch scripts and runbooks as `serial_no`. They can never match, so the cameras silently
> never came up — see the log above. **A serial harvested from `lsusb`/sysfs/`journalctl`
> is the wrong number for `serial_no`.** Only `rs-enumerate-devices -s` / `pyrealsense2`
> give the value the ROS node matches.

## Fix: do not hardcode serials at all

`launch_cameras.sh` and `run_recorder.sh` source `ros2_ur_ws/_resolve_camera_serials.sh`
and resolve against the live USB bus before launching anything;
`gello_recorder_gui.py` has equivalent Python-side logic. Behaviour:

| Situation | What happens |
|---|---|
| Both configured serials are connected | passed through unchanged, **silently** |
| Otherwise, one plain + one D435i/IF | assigned by model class (**plain D435 → cam1 SCENE, D435i/IF → cam2 WRIST**) with a `WARN auto-selected by model class:` line |
| Otherwise, both the same model class | falls back to sorted serial order + `WARN VERIFY THE PANES BEFORE RECORDING` |
| Fewer than 2 devices connected | **hard error**: lists what *is* connected, prints the override syntax, exits 1 |
| `pyrealsense2` not importable | non-fatal `SKIP`; configured serials are used as-is |

Enumeration takes **no streaming lock**, so it is safe to run while another process
already holds the cameras. Override only if you have a specific reason:

```bash
CAM1_SERIAL=<module-serial> CAM2_SERIAL=<module-serial> ./launch_cameras.sh
```

> ⚠️ **Known wart (2026-07-29):** the configured defaults in `launch_cameras.sh`,
> `run_recorder.sh` and `gello_recorder_gui.py` are the **ASIC** serials
> (`151623020789` / `322743060038`), so on this rig the "silent passthrough" row above
> never happens — every launch takes the model-class `WARN` path. It resolves to the
> right cameras today only because the two units are different model classes. If
> `pyrealsense2` were ever missing, the `SKIP` row would hand those unmatched serials
> straight to `realsense2_camera` and both cameras would silently never come up.

## Related

- `docs/testing/06_SENSORS.md` §1 — cam1/cam2 role contract and the arm-jog check for which unit is on the wrist.
- `ros2_ur_ws/_resolve_camera_serials.sh` — the helper itself, with the full behaviour description in its header.
