# Jazzy 포팅 핸드오프 — 2026-09-16 (토큰 소진으로 중단)

이 파일은 **작업 중단 시점의 상태 스냅샷**이다. 다음 세션은 이 문서만 읽고 이어간다.
브랜치 `feat/gello-ur7e-jazzy-24.04`, 워크트리 `/home/junhyeong/gello_software_jazzy`.

> 최신 상태: [VALIDATION_20260916.md](VALIDATION_20260916.md),
> 드라이버 검토: [DRIVER_3_8_REVIEW.md](DRIVER_3_8_REVIEW.md)를 먼저 읽을 것.
> 아래 §4c의 캘리브레이션 유실과 gravity 컨트롤러 가설은 실제 설치본 검토로
> 반증됐다. 캘리브레이션은 하위 launch context로 전달되며 회귀 테스트 2개 통과.
> 기본 활성 목록에는 gravity 컨트롤러가 없고 friction 컨트롤러는 관절 위치
> 인터페이스를 점유하지 않는다. 3개 ROS 패키지의 첫 colcon build도 통과했다.
> 사용자가 누락된 apt 의존성 4개를 설치한 뒤 `./build_ur7e.sh`까지 정상 완료했고,
> 전체 ROS 패키지 테스트는 **928 passed, 0 skipped**였다 (Python 3.12.3).
> §4 이하의 완료/미완료 표시는 최초 Claude 중단 시점의 기록이다.

## 0. 한 줄 요약

**GPU 서버 `junhyeong`(Ubuntu 24.04)에서 UR7e + RealSense ×2 + GELLO를 ROS 2 Jazzy로 직접 돌리기 위해
`feat/sim-data-collection`을 포크해 Humble→Jazzy 포팅 중.** 코드 수정은 대부분 끝났고
**커밋 전, `colcon build` 전**이다. ROS 2 Jazzy + UR/RealSense/Dynamixel 패키지 설치는 **완료**됐다.

## 1. 왜 이 방향인가 (결정 근거)

- IFQL 추론 서버(`~/carrot_ifql/code/.../vision_carrot/ifql_server.py`)는 ROS 없이 ZMQ만 쓴다.
  이 PC에서 실가중치·실프레임으로 **refill 중앙값 8.77 ms(BoN K=32) / 8.71 ms(BC)**, 800 ms 큐 대비
  duty cycle 1.1 % 확인. BoN 추가 비용 +0.04 ms. 콜드 스타트 3.0 s(bon) — 첫 에피소드 전 warmup 필수.
  체크포인트(`k0.9_s0`)의 253프레임 중 252에서 critic Q spread < 1e-3으로 보고됐다.
  작은 Q 차이만으로 BoN이 랜덤이거나 BC와 같은 정책이라고 결론낼 수 없다.
  후보 순위의 안정성과 실제 성공률 비교는 별도 검증 대상이다.
  원시 데이터: 세션 스크래치패드 `bench_ifql_real.json`.
- 사용자가 Docker Humble 대신 **네이티브 Jazzy 포팅**을 택했다. Sonnet 4명의 읽기 전용 조사 결과가 근거:
  - noble용 `ros-jazzy-*` 바이너리 **전부 존재**, 결손 0. 리포 파이썬 124개 파일 py3.12 `compileall -W error::SyntaxWarning` **통과**.
  - 노드 코드는 이미 배포판 중립 — `policy_leader_node.py` 주석 "Verified to construct under ROS2 Jazzy",
    `use_mock_hardware`/`use_fake_hardware` 이중 전달이 이미 들어 있음, `SwitchController`는 살아남는 필드만 사용.
  - 실질 파손은 `kinematics_params_file` 하나 (아래 §4).

## 2. 베이스 브랜치 선택

`origin/feat/sim-data-collection`(10cf3b9)을 베이스로 했다. 이유: IFQL 실기 배포 파일
(`run_ur7e_ifql_real.sh`, `ifql_deploy.yaml`, port 5595, `ifql/setup_ifql_workspace.sh`)이 **그 브랜치에만** 있고,
현재 작업 브랜치 `feat/gello-ur7e-humble-22.04`를 **완전히 포함**한다(ancestor 확인, 50커밋 앞섬).

기존 체크아웃 `/home/junhyeong/gello_software`는 `feat/gello-ur7e-humble-22.04`에 WIP 커밋 `7d5e9e2`
(offline IFQL learner 스택, **로컬 전용, 미푸시**)가 있고 그대로 보존됐다. 두 워크트리는 같은 `.git`을 공유한다.

## 3. 이 PC 하드웨어 실측 (2026-09-16)

| 항목 | 상태 |
| --- | --- |
| GELLO | FTDI FT232H `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0` → ttyUSB0. **코드 기본값과 시리얼까지 동일.** 유저 dialout ✅. latency_timer 16(리포는 요구 안 함) |
| RealSense | **cam1(SCENE)만 연결** — serial_number 143322071682 / ASIC 143623022572, Bus 002 USB 3.0 5 Gbps. cam2(143322072540)는 미연결 |
| udev | `99-realsense-libusb.rules` **설치됨**(install 스크립트). 설치 전엔 usbfs 노드가 root:root였음 → **카메라 재연결 필요** |
| 이더넷 | `enp12s0` 1개뿐. 마지막 확인 시 carrier=0(케이블 옮기는 중). 프로필 "Wired connection 1"에 랩 IP 166.104.146.29/24만 있음. **192.168.10.x 아직 없음** |
| 로봇 | 192.168.10.11 ping 실패(IP 없어서 당연). Robotiq은 별도 USB 아님 — UR 드라이버 tool_communication의 `/tmp/ttyUR` 경유 |
| 커널 | 6.17 non-RT. UR7e 경로는 RT 요구 없음(리포에서 RT 언급은 Franka만) |
| GUI | X11 세션(`DISPLAY=:1`), Wayland 아님 → Qt 우회 불필요 |
| RMW | 리포가 RMW를 고정하지 않음. Humble·Jazzy 모두 기본 fastrtps → 무변경 |
| Python | 시스템 `/usr/bin/python3` = 3.12.3. ⚠️ **conda base(3.13) 자동 활성화가 `python3`를 가린다** — 아래 §6 |
| 설치 완료 | ros-jazzy-desktop 0.11.0, **ur-robot-driver 3.8.0**, realsense2-camera 4.58.4, librealsense2 2.58.4, dynamixel-sdk 4.0.3, colcon, rosdep |

## 4. 코드 변경 현황 (미커밋, `git status` 36 modified + 2 untracked)

### 4a. 완료 — 스크립트 층 (Sonnet 2명, 보고 완료)
- **모든 `ros2_ur_ws/*.sh`(31개)**: `source /opt/ros/humble/setup.bash` → `GELLO_ROS_DISTRO="${GELLO_ROS_DISTRO:-jazzy}"` +
  `source "/opt/ros/${GELLO_ROS_DISTRO}/setup.bash"`. `$ROS_DISTRO`를 안 쓴 이유: 그건 setup.bash가 **설정하는** 변수라
  sourcing 전엔 신뢰 불가.
- `run_ur7e_ifql_real.sh`: `IFQL_PY` 해석 체인 `$IFQL_PY` → `~/carrot_ifql/.venv-svf/bin/python` → **`/home/junhyeong/miniconda3/envs/il/bin/python`** → 실패.
  안전 프리플라이트(`real_lead6` norm_stats 가드, sampler 라인, 5595 점유 검사)는 **바이트 동일 유지**.
- `build_ur7e.sh`: 3개 패키지 전부 빌드(`GELLO_BUILD_PACKAGES`로 재정의), conda 가드, `--symlink-install`.
- `run_hil_actor.sh`: site-packages ABI 태그 `python3.10` 하드코딩 → 동적. `ACTOR_VENV` 기본값 `/home/laptop3/venvs/gello-hil-actor`는
  **이 PC에 없음** — 오버라이드 가능, 주석만 추가.
- `launch_cameras.sh`의 `rs_launch.py` 인자 6개는 realsense2_camera **4.58.4 태그 소스 대조** — 전부 유효, 무변경.
- `_resolve_camera_serials.sh`: 카메라 2대 미만이면 **의도적으로 exit 1** (설계). cam2 꽂기 전엔 `launch_cameras.sh`가 거절한다.

### 4b. 부분 완료 — 파이썬/패키징 (Sonnet, 429로 중단됐지만 파일은 저장됨)
- `gello_recorder/home_move_ros.py`, `gello_policy/setup.py`, `policy_leader_node.py`, `remote_diffusion_pb2_grpc.py` 주석,
  `gello_recorder/README.md` 수정됨. `setup_jazzy/NOTES_grpcio.md` 작성됨(noble grpcio 1.51.1 vs 핀 1.74.0 — 미검증, 폴백은
  `pip install --user grpcio==1.74.0`).
- ⚠️ **리뷰 필요**: `git diff ros2_ur_ws/src/gello_recorder/gello_recorder/home_move_ros.py` (+20/-?)가 무엇을 바꿨는지 확인하고
  `SwitchController` 필드가 `activate_controllers`/`deactivate_controllers`인지 볼 것.

### 4c. **미완료 — 다음 세션이 해야 할 것**
1. **`gello_policy/launch/ur7e_diffusion_real.launch.py` — `kinematics_params_file` 조용한 유실.**
   Jazzy `ur_control.launch.py`는 이 인자를 **선언하지 않는다**(로봇 description 생성이 `description_launchfile`
   → `ur_rsp.launch.py`로 이사, `ur_control`은 `robot_ip`/`ur_type`만 전달). 우리 런치 L169-187·L~302가 넘기는 값은
   **에러 없이 무시**된다. 이 세션에서 `CALIB`은 비어 있고 리포에 캘리브 YAML도 없어 기본 `default_kinematics.yaml`로
   떨어지므로 **지금은 무해**하고, IFQL은 관절 절대각 명령이라 정책엔 영향 없음. 다만 누가 `CALIB=`를 주면 조용히 거짓말한다.
   해결: `gello_policy/launch/`에 `ur_rsp.launch.py`를 include하고 `kinematics_params_file`을 전달하는 얇은 래퍼 →
   `description_launchfile:=`로 지정. 또는 최소한 LogInfo로 크게 알릴 것. Agent(a3ddfa06)는 이 작업 중 429로 죽었고 **파일 변경 없음**.
2. **`ur_gello_bringup/` 감사 미완** (Agent a1656f29, 429). 특히 **Jazzy 드라이버가 새로 활성화하는 `gravity_update_controller`가
   `gello_move_to_start_node.py`의 strict(=2) 컨트롤러 스위치(SJTC→FPC)를 깨는지** — upstream `ur_controllers.yaml`(jazzy)에서
   그 컨트롤러가 command interface를 점유하는지 확인 필요. 점유하면 스위치 요청에서 같이 deactivate해야 한다.
   `ur7e_gello_real.launch.py`에 humble 참조 남아 있음(미수정).
3. **`run_mock_rviz.sh`** 는 여전히 `use_fake_hardware:=true`를 넘긴다(Humble 이름). Jazzy에선 `use_mock_hardware:=true`. 결정 필요.
4. **`ur_control_fake_safe.launch.py`**(gello_policy): importlib으로 upstream `launch_setup(context, *args, **kwargs)`를 호출 —
   Jazzy 시그니처는 `launch_setup(context)`. 실제 호출은 인자 없이 되므로 호환되지만, 이 워크어라운드의 전제(urscript_interface
   무조건 기동)는 upstream에서 이미 `UnlessCondition(use_mock_hardware)`로 막혀 있어 **불필요할 가능성 높음**. 실기 경로 아님(mock 전용).
5. **드라이버 2.13.2 → 3.8.0 거동 리스크 메모** (Opus a4cad3f4, 429로 죽음, 산출물 없음). 특히 `fpc-switch-probe.md`
   (`ros2_ur_ws/gello_logs/experiments/`)의 분석 — SJTC→FPC 전환 시 `URPositionHardwareInterface`가 command interface를
   측정 위치로 재시딩하는지 — 를 3.8.0 `hardware_interface.cpp`에 대해 다시 해야 한다. **실기 전 최우선.**
6. **IFQL 서버 실기동 스모크** (Agent ac94e8e9, 429). `ifql_server.py`를 `il` 인터프리터로 포트 **5695**(5591-5595 금지)에 띄워
   실프레임 왕복 지연을 재고, `setup_jazzy/ifql_server_smoke.sh`를 남기는 일. 미착수. 주의: `FeatureHeads.__init__`이 r18_ss 모드에서도
   dinov2 torch.hub 다운로드를 시도할 수 있음 — 서버 CLI 플래그가 이를 피하는지 확인.
7. **문서**: `docs/ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md`(작성됨, 미검토), `setup_jazzy/network_and_urcap.md`(작성 중 중단 — 내용 확인 필요).
8. **`colcon build`** — 아직 한 번도 안 돌렸다. `cd ros2_ur_ws && ./build_ur7e.sh` (conda 가드 내장).
9. **전체 diff 리뷰 후 커밋** (푸시는 사용자 승인 후에만).

## 5. 로봇 연결 — 사용자(sudo/펜던트)만 할 수 있는 것

```bash
# NIC 하나뿐. 케이블이 로봇으로 가면 랩 유선망은 끊긴다(wifi wlp13s0는 유지).
sudo nmcli con mod "Wired connection 1" +ipv4.addresses 192.168.10.100/24
sudo nmcli con up "Wired connection 1"
ping -c3 192.168.10.11
nc -zv 192.168.10.11 29999
nc -zv 192.168.10.11 30004
nc -zv 192.168.10.11 30001
```
🛑 **펜던트의 External Control 프로그램에 적힌 "제어 PC IP"(reverse 연결, port 50001)** 는 리포 어디에도 숫자가 없다 —
laptop3가 쓰던 192.168.10.x는 펜던트에만 있다. **(a)** 이 PC에 그 IP를 그대로 주거나 **(b)** 펜던트에서 IP를 이 PC로 고치고
**Update program**. PolyScope 5(`.urcap`) vs X(`.urcapx`)는 비호환. 드라이버 3.x의 최소 URCap 버전은 **미확인**.

## 6. 함정 — 반드시 알 것

- **conda base(3.13)가 `python3`를 가린다.** Jazzy rclpy/colcon은 시스템 3.12용. ROS 터미널마다
  `conda deactivate`(또는 PATH에서 miniconda 제거) 후 `source /opt/ros/jazzy/setup.bash`. 안 하면 rclpy ImportError.
  `install_jazzy.sh`·`build_ur7e.sh`는 이 가드를 내장했다.
- **`il` 환경은 절대 변경 금지**(IFQL 추론 검증 완료 상태). 포트 **5591-5595 바인딩 금지**(실배포 예약).
- `FMRL_CAM1_CROP`/`FMRL_CAM1_MODE` 환경변수 **설정 금지**(cam1 feature가 2048-D로 바뀌어 2055-D 계약 파괴).
- `~/carrot_ifql/` 읽기 전용.
- 기존 체크아웃 `/home/junhyeong/gello_software`는 CLAUDE.md에 "다른 사람 트리"로 적혀 있으나 이 세션은 사용자가 거기서 직접 열어 승인함.

## 7. 산출물 위치

- 코드: `/home/junhyeong/gello_software_jazzy` (미커밋)
- `ros2_ur_ws/setup_jazzy/`: `install_jazzy.sh`(실행 완료), `network_and_urcap.md`, `NOTES_grpcio.md`, 이 파일
- `docs/ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md`
- 벤치 핵심 파일 보존본: `ros2_ur_ws/log/jazzy_port_20260916/claude_benchmark/`
  (`bench_ifql_real.py|json`, `zmq_bench_client.py`, `lerobot_frames.py`). git에는 넣지 않는다.
- 세션 스크래치패드(`/tmp/claude-1000/-home-junhyeong-gello-software/ef161f25-.../scratchpad/`): `bench_ifql_real.py|json`,
  `zmq_bench_client.py`, `lerobot_frames.py`, `episode0_extracted/`(실프레임 253장), `simdc/`(sim-data-collection export),
  `jazzy_assess/`(Packages 인덱스, install 초안). **재부팅 시 사라질 수 있음.**
- 모델/데이터: `~/carrot_ifql/hf/ifql_real_lead6_k0.9_s0/`, `~/carrot_ifql/data/carrot_in_pot_lerobot_v3/`(RGB만, 17,088 프레임)
