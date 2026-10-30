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
- Every consumer (viewer, recorder, and any policy/RL client on a research branch) reports
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

## 현재 동작: 설정된 시리얼을 live 버스와 **엄격히** 대조한다

`launch_cameras.sh`와 `run_recorder.sh`는 카메라 노드를 띄우기 전에
`ros2_ur_ws/_resolve_camera_serials.sh`를 source해 설정된 `CAM1_SERIAL`/`CAM2_SERIAL`을
`rs-enumerate-devices`(ros-jazzy-librealsense2에 포함, `pyrealsense2` 불필요) 출력과 대조한다.
`gello_recorder_gui.py`도 같은 규칙을 Python 쪽에 갖고 있다.

| 상황 | 동작 |
|---|---|
| 설정된 두 시리얼이 **둘 다** 연결돼 있음 | 그대로 사용(물리적 cam1/cam2 배정 보존) |
| 그 외 | 카메라 노드를 띄우기 **전에 실패**한다. 두 D435를 모델 클래스나 USB 열거 순서로 안전하게 배정할 수 없기 때문이다 |
| `CAMERA_ALLOW_SERIAL_FALLBACK=1` | 옛 모델 클래스/정렬 순서 fallback을 켠다(임시 리그용, 항상 WARN) |

기본값은 **module serial**이다(2026-10 기준 cam1 `143322071682`, cam2 `143322072540`, 둘 다 plain D435 —
위 표의 2026-07-29 값은 그 이전 카메라 쌍의 기록이다). 기본값의 위치는 `launch_cameras.sh`/`run_recorder.sh`
머리말을 볼 것. 카메라를 바꾸면 기본값을 고치거나 환경변수로 덮어쓴다:

```bash
CAM1_SERIAL=<module-serial> CAM2_SERIAL=<module-serial> ./launch_cameras.sh
./launch_cameras.sh --check     # 스트림을 열지 않고 시리얼 매핑만 검사
```

열거는 스트리밍 락을 잡지 않으므로 다른 프로세스가 카메라를 쓰는 중에도 안전하다.

---

# cam1 / cam2 역할 계약과 손목 확인

| | cam1 | cam2 |
|---|---|---|
| 역할 | **SCENE** — 삼각대 고정, 3인칭 | **WRIST** — 그리퍼에 강체 장착, 팔과 함께 움직인다 |
| 토픽 | `/cam1/cam1/color/image_raw/compressed` | `/cam2/cam2/color/image_raw/compressed` |
| 프로파일 | `1280x720x30` | 동일 |

매핑이 뒤집혀도 **에러는 나지 않는다** — 녹화 데이터와 그 데이터로 학습한 정책이 조용히 망가진다.
그래서 `launch_cameras.sh`는 뷰어(좌 = cam1, 우 = cam2)를 **기본으로 띄운다**(`VIEW=false`로 끔).
판정은 팔을 한 번 움직여 보는 것이다:

- **좌측 창(cam1)**: 팔을 움직여도 배경이 고정돼 있다.
- **우측 창(cam2)**: 팔을 움직이면 **화면 전체가 흐르고**, 그리퍼 손가락은 **같은 픽셀에 남는다.**
  "화면에 물체가 안 보인다"는 조준 문제가 아니라 팔이 다른 곳을 보고 있을 뿐일 수 있다.

> `launch_cameras.sh` 머리말 주석의 `cam2 = CLOSE-UP (workspace)` 류 문구는 낡은 표현이다 —
> cam2는 손목 카메라다.

손으로 한 대만 띄울 때는 시리얼을 **따옴표로 감싼다**. `ros2 launch`는 전부 숫자인 인자를 정수로
바꾸고, `serial_no`는 STRING 파라미터라 노드가 즉사한다:

```bash
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=cam1 camera_namespace:=cam1 "serial_no:='<module-serial>'" \
    "rgb_camera.color_profile:='1280x720x30'"
```

## 카메라 판정 체크리스트

- [ ] `./launch_cameras.sh`가 30초 안에 두 스트림 ~30 Hz 도달 (`wait_for_stream`)
- [ ] 뷰어 좌 = cam1 SCENE(배경 고정) / 우 = cam2 WRIST(배경이 흐르고 손가락은 제자리)
- [ ] `lsusb -t`에서 두 카메라 모두 `5000M`(USB3). `480M`이면 USB2로 폴백된 것 — 케이블/포트 교체
- [ ] `cat /sys/module/uvcvideo/parameters/quirks` = `4294967295`
- [ ] 유량은 `ros2 topic hz`로 판정한다. RealSense 퍼블리셔는 RELIABLE/TRANSIENT_LOCAL이라
      늦게 붙은 구독자는 **캐시된 마지막 프레임**을 즉시 받는다 — `echo --once` 성공은
      카메라가 살아 있다는 증거가 아니다.

> "두 대가 같은 USB 허브에 있으면 문제"라는 말은 **가설**이다. 리포 안에 이를 뒷받침하는 근거는
> 없고, 확인된 근본 원인은 위 `uvcvideo` quirks였다. 별도 버스로 옮겨 재현이 사라지면 그때 기록한다.

## Related

- `ros2_ur_ws/_resolve_camera_serials.sh` — 시리얼 해석 헬퍼. 전체 동작은 파일 머리말에 있다.
- `ros2_ur_ws/launch_cameras.sh` — 카메라 2대 + 뷰어. `ENABLE_DEPTH`/`ALIGN_DEPTH` 등 옵션은 머리말.
- `ros2_ur_ws/camera_viewer.py` — 단독 2-카메라 뷰어.
