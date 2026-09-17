# Shared RealSense serial auto-resolution helper.
#
# SOURCE this file (do NOT execute it directly) from a script that has already
# set CAM1_SERIAL and CAM2_SERIAL to their configured/default values, then call
# `resolve_serials`. On return it has REASSIGNED the caller's CAM1_SERIAL /
# CAM2_SERIAL shell variables in place -- that only works because the caller
# sources this file into its own shell rather than executing it as a
# subprocess.
#
# Behaviour:
#   - reads the USB devices with the Jazzy-provided rs-enumerate-devices tool,
#     which is available with ros-jazzy-librealsense2 even when pyrealsense2 is
#     not installed for the active Python interpreter;
#   - if BOTH configured serials are currently enumerated -> passes them
#     through unchanged, preserving the physical cam1/cam2 assignment;
#   - otherwise fails before a camera node is launched. Two D435 bodies cannot
#     be safely assigned by model class or USB enumeration order;
#   - CAMERA_ALLOW_SERIAL_FALLBACK=1 restores the legacy model-class / sorted
#     serial fallback for temporary non-production rigs. It always warns.
#
# Enumeration (`rs.context().query_devices()`) takes no streaming lock, so
# this is safe to run even while another process already holds the cameras
# (it just re-reports the same devices).
resolve_serials() {
    local enumerator="${REALSENSE_ENUMERATOR:-rs-enumerate-devices}"
    local devices name serial line field value have_cam1=false have_cam2=false
    local -a names=() serials=() plain=() imu=()

    if [ "$CAM1_SERIAL" = "$CAM2_SERIAL" ]; then
        echo "### ERROR: cam1 and cam2 must use distinct serial numbers." >&2
        return 1
    fi

    if ! command -v "$enumerator" >/dev/null 2>&1; then
        echo "### ERROR: RealSense enumerator is unavailable: ${enumerator}" >&2
        echo "### Source /opt/ros/jazzy/setup.bash, or install ros-jazzy-librealsense2." >&2
        return 1
    fi

    if ! devices="$($enumerator 2>/dev/null)"; then
        echo "### ERROR: ${enumerator} could not enumerate USB RealSense devices." >&2
        echo "### Reconnect both cameras to USB 3, then run: ${enumerator} -s" >&2
        return 1
    fi

    name=""
    serial=""
    while IFS= read -r line || [ -n "$line" ]; do
        field="${line%%:*}"
        field="${field#${field%%[![:space:]]*}}"
        field="${field%${field##*[![:space:]]}}"
        value="${line#*:}"
        value="${value#${value%%[![:space:]]*}}"
        case "$field" in
            "Device info")
                if [ -n "$name" ] && [ -n "$serial" ]; then
                    names+=("$name")
                    serials+=("$serial")
                fi
                name=""
                serial=""
                ;;
            Name)
                name="$value"
                ;;
            "Serial Number")
                serial="$value"
                ;;
        esac
    done <<< "$devices"
    if [ -n "$name" ] && [ -n "$serial" ]; then
        names+=("$name")
        serials+=("$serial")
    fi

    if [ "${#serials[@]}" -lt 2 ]; then
        echo "### ERROR: only ${#serials[@]} RealSense device(s) found; need 2." >&2
        [ -n "$devices" ] && printf '%s\n' "$devices" >&2
        echo "### Expected cam1=${CAM1_SERIAL}, cam2=${CAM2_SERIAL}. Reconnect both to USB 3, then run: ${enumerator} -s" >&2
        return 1
    fi

    for serial in "${serials[@]}"; do
        [ "$serial" = "$CAM1_SERIAL" ] && have_cam1=true
        [ "$serial" = "$CAM2_SERIAL" ] && have_cam2=true
    done
    if [ "$have_cam1" = true ] && [ "$have_cam2" = true ]; then
        return 0
    fi

    echo "### ERROR: expected RealSense pair is not fully connected; refusing an unsafe remap." >&2
    echo "### Expected: cam1=${CAM1_SERIAL} (scene), cam2=${CAM2_SERIAL} (wrist)" >&2
    echo "### Detected:" >&2
    for ((i = 0; i < ${#serials[@]}; i++)); do
        printf '###   %s  %s\n' "${names[$i]}" "${serials[$i]}" >&2
    done
    echo "### Reconnect the expected cameras, then rerun: ${enumerator} -s" >&2

    case "${CAMERA_ALLOW_SERIAL_FALLBACK:-0}" in
        1|true|TRUE|yes|YES|on|ON) ;;
        *)
            echo "### For a temporary non-production rig only, set CAMERA_ALLOW_SERIAL_FALLBACK=1." >&2
            return 1
            ;;
    esac

    for ((i = 0; i < ${#serials[@]}; i++)); do
        if [[ "${names[$i],,}" == *d435i* ]]; then
            imu+=("${serials[$i]}")
        else
            plain+=("${serials[$i]}")
        fi
    done
    if [ "${#plain[@]}" -eq 1 ] && [ "${#imu[@]}" -eq 1 ]; then
        CAM1_SERIAL="${plain[0]}"
        CAM2_SERIAL="${imu[0]}"
        echo "### WARN: legacy fallback selected by model class: cam1=${CAM1_SERIAL}, cam2=${CAM2_SERIAL}" >&2
        return 0
    fi

    if [ "${#serials[@]}" -ne 2 ]; then
        echo "### ERROR: ambiguous fallback requires exactly two cameras; set explicit serials." >&2
        return 1
    fi
    mapfile -t serials < <(printf '%s\n' "${serials[@]}" | sort)
    CAM1_SERIAL="${serials[0]}"
    CAM2_SERIAL="${serials[1]}"
    echo "### WARN: legacy fallback selected by serial sort: cam1=${CAM1_SERIAL}, cam2=${CAM2_SERIAL}" >&2
    echo "### WARN: verify scene/wrist panes before recording or deployment." >&2
}
