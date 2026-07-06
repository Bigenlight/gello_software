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
