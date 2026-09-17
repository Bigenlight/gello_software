# Jazzy 포팅 후속 검증 — 2026-09-16

워크트리: `/home/junhyeong/gello_software_jazzy`
브랜치: `feat/gello-ur7e-jazzy-24.04` (base `10cf3b9`).
Claude의 기존 변경을 보존하고 Codex가 미완료 항목을 검토했다.
작은 모델 `gpt-5.6-luna`가 mock 런치, 문서, IFQL 스모크를 분담했다.

## 검증 결과

### 2026-09-17 GELLO 녹화 / EEF 추가 검증

- 작은 모델 두 개가 녹화·실행 스크립트를 분담하고 부모가 EEF ROS 통합 검증과 변경 검토를 수행했다.
- 최종 전체 테스트 **931 passed**, 기존 protobuf 경고 3개. 3패키지 재빌드 2.03초, rosdep check PASS.
- EEF GenericSystem 스모크 PASS: STRICT 전환, 특이점 거부, 서로 다른 리더/로봇 자세에서
  zero-jump engage/re-engage, reclutch, delta 추종, disengage 후 hold.
- EEF GUI offscreen 생성·refresh·시작 자세 YAML 로딩 PASS. 실제 HOME 이동은 실행하지 않았다.
- 합성 녹화 산출물 직접 재열기: RGB 각 39프레임 MP4 + vectors.h5;
  별도 depth 시험 각 29프레임 MP4 + depth.h5. 실제 장비 녹화가 아니다.
- recorder의 Jazzy ExternalShutdownException 처리, checkout 기준 저장 경로,
  실행 래퍼 6개의 시스템 Python 우선순위를 보완했다. conda il 및 제어 제한은 불변.
- 산출물은 `log/jazzy_port_20260916/recorder_smoke/`, EEF 로그는 같은 폴더의
  `eef_mock_smoke.log`, 재현은 `setup_jazzy/eef_mock_smoke.py`.
- 운영 순서는 [EEF 수집 가이드](../../docs/ros2/GELLO_EEF_COLLECTION_JAZZY.md).
  아래 928개 결과는 추가 변경 전 기록이며 최신 전체 수와 합산하지 않는다.

| 항목 | 결과 | 범위 |
| --- | --- | --- |
| ./build_ur7e.sh | PASS, 3 packages finished | 의존성 검사 포함; gello_recorder, ur_gello_bringup, gello_policy |
| bringup pytest | 675 passed | 시스템 Python 3.12.3, ROS Jazzy + workspace overlay |
| policy/recorder 선택 pytest | 120 passed, 1 skipped | 아래 재현 명령, 같은 인터프리터 |
| 캘리브레이션 전달 | 2 passed | 위 120개에 포함; 두 런치의 최종 URDF에 별도 calibration hash 존재 |
| mock STRICT SJTC→FPC | PASS, ok=True | GenericSystem, domain 173, loopback; friction controller active 유지 |
| 전체 3패키지 pytest | **928 passed, 0 skipped**, 3 warnings | 의존성 설치 후 Python 3.12.3, 8.30 s |
| rosdep check | PASS | All system dependencies have been satisfied |
| IFQL 설정 policy 노드 생성 | PASS, HOLD | domain 174, loopback 5695; executor spin/제어 명령 없음 |
| Python 문법 | PASS | 3패키지 코드·launch, Python 3.12 SyntaxWarning을 error로 처리 |
| 셸 문법 / diff 공백 | PASS | bash -n / git diff --check |
| policy·GELLO 실기 launch --show-args | PASS | 노드 실행 없음 |
| IFQL 호환 서버 ZMQ smoke | PASS, 48 ACT | 실제 체크포인트/프레임, K=32, 127.0.0.1:5695 |

ROS는 시스템 Python 3.12 ABI를 사용했다. IFQL은 기존 conda `il`을 사용한다.
누락된 시스템 패키지 4개는 사용자가 sudo로 설치한 뒤 확인했다.
conda `il`은 변경하지 않았다. 위 선택 테스트 수는 전체 928개에 포함되므로 합산하지 않는다.
경고 3개는 배포판 protobuf의 Python deprecation 경고이며 테스트 실패가 아니다.

## 수정·검토

- 세 Python 패키지는 `<build_type>ament_python</build_type>`와 함께
  `<buildtool_depend>ament_python</buildtool_depend>`를 유지한다. Jazzy에서 세 패키지를
  다시 빌드해 manifest와 install-space 생성이 정상임을 확인했다. policy의 ZMQ 의존성도 명시했다.
- 설치 스크립트에 누락된 4개 Python apt 패키지를 추가하고 존재하지 않던
  `build_jazzy.sh` 안내를 `../build_ur7e.sh`로 고쳤다.
- mock 런처의 Jazzy argument를 `use_mock_hardware`로 수정했다.
- `kinematics_params_file` 유실은 재현되지 않았다. 추가 래퍼 없이 회귀 테스트로
  현재 launch context 전달을 검증했다.
- `gravity_update_controller` 충돌 가설도 설치본과 달랐다.
  상세 근거와 실기 검증 한계는 [DRIVER_3_8_REVIEW.md](DRIVER_3_8_REVIEW.md).
- 네트워크 문서의 PC/로봇 포트 방향과 링크를 수정했다. 로봇 Primary는 30001,
  드라이버 PC의 reverse/script sender는 50001/50002다.
- Q spread가 작다는 사실만으로 BoN이 BC와 동일하거나 랜덤이라고 단정하던 문구를 철회했다.

## IFQL 서버 검증

이 PC에는 `params_100000.infer.pkl`만 있고 전체 학습 checkpoint가 없다.
원본 서버는 전체 checkpoint 파일/optimizer state를 요구하며, r18_ss를 쓰는 경우에도
DINO 모델을 초기화한다. `ifql_server_compat.py`를 추가해 실제 inference params를
모델 template의 key/shape/dtype과 대조해 복원하고, 캐시된 ResNet18만 로드한다.
원본 외부 코드는 수정하지 않았다. 따라서 결과는 **호환 어댑터를 통한 서버 검증**이다.

실제 런처도 `IFQL_COMPAT=auto`에서 infer-only 파일을 발견하면 같은 어댑터를 사용한다.
`IFQL_COMPAT=0`은 기존 전체 checkpoint 경로, `1`은 명시적 어댑터 선택이다.
서버 로그 기본 위치는 이제 workspace `log/ifql/`이며 모델 디렉터리 밖이다.

서버 왕복 재검증(22:46~22:47 KST):

- RESET 응답: 실제 `.infer.pkl` 경로, checkpoint revision **100000**, r18_ss,
  D_a=2055, action_dim=7, queue=24, K=32, random_init=false.
- checkpoint 내부 optimizer step은 100001이며 파일 revision 100000을 덮어쓰지 않는다.
- 48 ACT 왕복: p50 **0.06 ms**, p95 **0.14 ms**, 최대 **23.57 ms**.
- 새 추론 청크를 만든 두 요청: **9.26 / 23.57 ms**, 중앙값 **16.42 ms**.
  나머지는 queue hit다. 두 refill만으로 일반 성능 p99나 제어 루프 주기를 주장하지 않는다.
- startup **3.2 s**, 첫 warmup **3051.6 ms**, 세 번째 warmup **17.2 ms**.
- SIGTERM에서 서버와 임시 staging을 정리하고 포트 5695를 해제한다.

초기 작은 모델 보고의 2표본 refill p50은 nearest-rank 최솟값이었다.
최종 스모크는 `statistics.median`으로 수정했다. 이전 8.77 ms 벤치는 직접 plan 호출이며
이번 값은 서버 왕복이므로 동일 실험으로 섞지 않는다.

재현:

```bash
cd /home/junhyeong/gello_software_jazzy
bash ros2_ur_ws/setup_jazzy/ifql_server_smoke.sh
```

스크립트는 `conda run --no-capture-output -n il`을 사용하며,
보존한 실제 프레임을 기본값으로 읽는다. 다른 추출 프레임은 `IFQL_SMOKE_FRAME_DIR`로 지정한다.
로그: `ros2_ur_ws/log/ifql_smoke/ifql_server_smoke_20260916_224657_110094.log`.

종료 경로도 수정 후 22:48~22:49에 다시 검증했다. conda 부모만 기다리지 않고
프로세스 그룹 전체에 종료 유예를 주도록 고쳤다. 48 ACT / 2 refill / **errors 0**,
종료 STATS 기록, 포트 5695 해제, `ifql_compat_*` 임시 디렉터리 잔존 0을 확인했다.
이 실행의 refill 왕복은 9.24 / 24.89 ms(중앙값 17.07 ms)였다.
로그: `ros2_ur_ws/log/ifql_smoke/ifql_server_smoke_20260916_224849_110869.log`.

## 재현 명령

```bash
cd /home/junhyeong/gello_software_jazzy/ros2_ur_ws
source /opt/ros/jazzy/setup.bash
/usr/bin/colcon build --symlink-install --packages-select ur_gello_bringup gello_policy gello_recorder
source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q src/ur_gello_bringup/test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q \
  src/gello_policy/test/test_joint_angles.py \
  src/gello_policy/test/test_jazzy_description_contract.py \
  src/gello_recorder/test/test_home_move.py \
  src/gello_recorder/test/test_spin_health.py \
  src/gello_recorder/test/test_take_delete.py \
  src/gello_recorder/test/test_reward_classifier_helpers.py \
  src/gello_recorder/test/test_realsense_argv_depth.py
```

## 환경 의존성 완료 / 남은 실기 항목

최초 전체 테스트는 protobuf/h5py 미설치로 collection에서 중단됐다.
이후 사용자가 아래 명령을 실행했고 `./build_ur7e.sh`와 전체 suite가 모두 통과했다.

```bash
sudo apt-get install -y python3-grpcio python3-protobuf python3-h5py python3-zmq
```

설치 확인: grpcio apt 1.51.1-4.1build5, protobuf apt 3.21.12-8.2ubuntu0.3
(Python module 4.21.12), h5py 3.10.0-1ubuntu3, zmq 24.0.1-5build1.
IFQL YAML을 적용한 실제 `PolicyLeaderNode`도 HOLD로 생성·종료했다.
이 확인은 원격 gRPC 서버와의 상호 운용 테스트는 아니다.

전체 suite 재현 (앞 절의 ROS/overlay sourcing 후):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen /usr/bin/python3 -m pytest -q \
  src/ur_gello_bringup/test src/gello_policy/test src/gello_recorder/test
```

로봇 NIC은 재확인 시에도 DOWN이며 192.168.10.x 주소가 없었다.
의존성 설치 후 연결 재점검에서도 `enp12s0`는 NO-CARRIER이고,
`ip route get 192.168.10.11`은 Wi-Fi gateway 192.168.0.1로 향했다.
ROS 환경을 source한 `rs-enumerate-devices -s`는 `No device detected`를 출력했고
`lsusb`에도 RealSense가 없었다 (기존 cam1 한 대 연결 스냅샷에서 달라짐).
GELLO FTBEO6QK는 `/dev/ttyUSB0`로 확인됐고 사용자 dialout 권한과 비점유 상태를 확인했다.
시스템 Python의 선택적 `pyrealsense2` binding은 없다. 자동 시리얼 해석은 기존 설계대로
SKIP 후 설정값을 사용하며, ROS camera driver 설치와 Python binding 설치를 혼동하던
진단 문구만 고쳤다. 장치 열거는 설치된 C++ CLI로 확인했다.
실물 UR7e/그리퍼 동작 검증, 두 카메라 동시 스트리밍, 펜던트 IP·버전 확인은 남아 있다.
mock 성공은 실제 UR hardware interface 재시딩을 측정한 결과가 아니다.
실기 실행과 커밋/푸시는 수행하지 않았다.

## 2026-09-17 최근 7일 원격 코드 통합

원격 10개 브랜치에서 2026-09-10~17 커밋을 다시 fetch해 중복 제거했다. 대상은 58개이며,
기존 Jazzy HEAD에 없던 런타임 변경은 `origin/feat/sim-data-collection`의 마지막 6개였다.
`origin/test/kanu-learner-fallback`의 별도 SHA들은 기존 Jazzy 커밋과 패치 동등하거나 기능적으로
이미 반영돼 있었다.

마지막 6개에서 orange IFQL task profile, exact norm-stats/PX guard, dry-run, 진단 영상 렌더러,
RESET 뒤 policy metadata snapshot을 가져왔다. 실기 런처는 Jazzy Python 3.12 sourcing,
`il` fallback, `IFQL_COMPAT`, `.infer.pkl` checkpoint와 workspace 로그 경로를 보존했다.
`checkpoint_path`도 호환 모드가 선택한 실제 `IFQL_CHECKPOINT`를 launch에 전달한다.

검증 결과:

- ROS 단위 테스트: **934 passed** (`carrot|orange` dry-run 2개와 invalid-task 거부 포함)
- MuJoCo 수집·평가 테스트: **287 passed, 8 skipped**
- Jazzy `gello_policy`, `gello_recorder`, `ur_gello_bringup`: clean source 환경에서 build PASS
- 진단 영상: conda `gello-sim`의 ffmpeg 8.0.1/libx264로 1280×720, H.264,
  yuv420p, 266 frame 재인코딩·전수 decode PASS
- shell syntax, package XML parse, `git diff --check`: PASS

진단 영상의 최종 H.264 encode에는 PATH의 `ffmpeg`가 필요하다. 이 PC의 시스템 apt 설치는
sudo 비밀번호 입력 때문에 자동 설치하지 못했고, 수정 가능한 `gello-sim` conda 환경에 설치했다.
새 머신용 `install_jazzy.sh`에는 apt `ffmpeg`를 추가했다. 실제 UR7e, gripper, RealSense,
물리 GELLO는 이 통합 검증에서 열지 않았다.

## 로컬 산출물

`ros2_ur_ws/log/jazzy_port_20260916/`에 mock 로그와 Claude 벤치 핵심 파일을
보존했다. 실제 추출 프레임 253개(31 MB)도 `claude_benchmark/episode0_extracted/`에
복사했다. 이 디렉터리는 git ignored이며 `/tmp` 원본이 없어져도 유지된다.
