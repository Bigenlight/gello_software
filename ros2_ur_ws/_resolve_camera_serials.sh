# Shared RealSense serial auto-resolution helper.
#
# SOURCE this file (do NOT execute it directly) from a script that has already
# set CAM1_SERIAL and CAM2_SERIAL to their configured/default values, then call
# `resolve_serials`. On return it has REASSIGNED the caller's CAM1_SERIAL /
# CAM2_SERIAL shell variables in place -- that only works because the caller
# sources this file into its own shell rather than executing it as a
# subprocess.
#
# Behaviour (identical to the resolve_serials() this was factored out of --
# see launch_cameras.sh's git history, commit "fix(cameras): resolve
# RealSense serials against the live USB bus", for the incident this guards
# against):
#   - if BOTH configured serials are currently enumerated -> passes them
#     through unchanged, silently (nothing printed);
#   - otherwise assigns by model class (device name containing "d435i"
#     case-insensitively -> cam2, plain -> cam1), printing a WARN;
#   - if both connected devices are the SAME model class the mapping is
#     ambiguous -> falls back to sorted serial order, with a stronger WARN
#     telling the operator to verify the camera panes before recording;
#   - fewer than 2 connected devices is a hard error: prints what IS
#     connected plus the manual-override syntax, then exits the calling
#     script with status 1;
#   - pyrealsense2 not importable -> the heredoc exits 3 internally, and
#     resolve_serials() downgrades that to a non-fatal SKIP (the configured
#     serials are used as-is; the realsense2_camera node does its own serial
#     lookup regardless).
#
# Enumeration (`rs.context().query_devices()`) takes no streaming lock, so
# this is safe to run even while another process already holds the cameras
# (it just re-reports the same devices).
resolve_serials() {
    local resolved
    resolved="$(CAM1_SERIAL="$CAM1_SERIAL" CAM2_SERIAL="$CAM2_SERIAL" python3 - <<'PY'
import os, sys

try:
    import pyrealsense2 as rs
except ImportError:
    print("SKIP pyrealsense2 not importable; using configured serials as-is", file=sys.stderr)
    raise SystemExit(3)

want1, want2 = os.environ["CAM1_SERIAL"], os.environ["CAM2_SERIAL"]
devs = [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
        for d in rs.context().query_devices()]

if len(devs) < 2:
    for name, serial in devs:
        print(f"    connected: {name}  {serial}", file=sys.stderr)
    print(f"ERROR only {len(devs)} RealSense device(s) found, need 2", file=sys.stderr)
    raise SystemExit(1)

present = {serial for _, serial in devs}
if want1 in present and want2 in present:
    print(f"{want1} {want2}")
    raise SystemExit(0)

# Configured pair is not what is plugged in.  Assign by model class instead:
# the IMU/IF variants report "D435I..."; the plain unit reports "D435".
imu = [s for n, s in devs if "d435i" in n.lower()]
plain = [s for n, s in devs if "d435i" not in n.lower()]
for name, serial in devs:
    print(f"    connected: {name}  {serial}", file=sys.stderr)
print(f"WARN configured serials ({want1}, {want2}) are not both connected", file=sys.stderr)

if len(plain) == 1 and len(imu) == 1:
    print(f"WARN auto-selected by model class: cam1={plain[0]} (plain D435), "
          f"cam2={imu[0]} (D435IF/i)", file=sys.stderr)
    print(f"{plain[0]} {imu[0]}")
    raise SystemExit(0)

# Ambiguous: same model class on both mounts.  Order is arbitrary, so say so.
ordered = sorted(s for _, s in devs)[:2]
print(f"WARN model classes are ambiguous; falling back to serial sort order: "
      f"cam1={ordered[0]} cam2={ordered[1]}", file=sys.stderr)
print("WARN VERIFY THE PANES BEFORE RECORDING -- cam1/cam2 may be swapped", file=sys.stderr)
print(f"{ordered[0]} {ordered[1]}")
PY
)" || {
        local rc=$?
        # rc 3 = pyrealsense2 missing: not fatal, the realsense node does its own
        # lookup.  Anything else means we genuinely cannot bring 2 cameras up.
        if [ "$rc" != "3" ]; then
            echo "###" >&2
            echo "### Camera resolution failed. Check that both RealSense cameras are" >&2
            echo "### plugged into USB 3 ports, then rerun.  Override explicitly with:" >&2
            echo "###   CAM1_SERIAL=<serial> CAM2_SERIAL=<serial> $0" >&2
            exit 1
        fi
        resolved=""
    }
    if [ -n "$resolved" ]; then
        CAM1_SERIAL="$(echo "$resolved" | cut -d' ' -f1)"
        CAM2_SERIAL="$(echo "$resolved" | cut -d' ' -f2)"
    fi
}
