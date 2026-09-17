# sim_collect Ubuntu 24.04 / ROS 2 Jazzy 새 머신 이전

이 문서는 `/home/junhyeong/gello_software_jazzy`의 MuJoCo GELLO 데이터 수집 스택을 새 머신에
재현하는 절차다. 기존 laptop3의 `.venv`, `DISPLAY=:0`, 소프트웨어 GLFW 측정은
[`README.md`](README.md)에 역사 기록으로 남긴다. 이 문서의 `gello-sim` 환경과 검증 결과를
laptop3에 소급 적용하지 않는다.

## 범위

- 순수 시뮬레이션 프로세스: `sim_collect.sim_main`, `sim_collect.capture`, `sim_collect.gui`
- 물리 GELLO를 쓰지 않는 마이그레이션 검증: `--fake-leader`
- 실제 UR 로봇, HIL, ROS 2 hardware launch는 범위 밖
- 정책 서버의 Torch/LeRobot 환경은 범위 밖
- 수집 IPC는 기본 Unix socket 또는 TCP 6701/6702/6711/6712이며, 정책 포트 5591~5595를 쓰지 않는다

## 새 머신 기준선

2026-09-17에 확인한 기준선은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| OS / ROS | Ubuntu 24.04 / ROS 2 Jazzy |
| checkout | `/home/junhyeong/gello_software_jazzy` |
| Python | Conda `gello-sim`, Python 3.11 |
| GPU | NVIDIA RTX 5070 Ti |
| driver | 590.48.01 |
| interactive display | `DISPLAY=:1` |
| interactive backend | GLFW |
| headless backend | EGL |

`MUJOCO_GL=egl`에서 240×320 RGB 렌더 결과 `(240, 320, 3)`과 `DISPLAY=:1`의 Tk 창 기동은
각각 통과했다. 이 둘은 렌더·GUI 기반 점검이고, 실제 GELLO 텔레옵 또는 실제 로봇 검증은 아니다.

## 1. 서브모듈과 에셋

```bash
cd /home/junhyeong/gello_software_jazzy
git submodule update --init third_party/mujoco_menagerie third_party/DynamixelSDK
git submodule status third_party/mujoco_menagerie third_party/DynamixelSDK
```

두 서브모듈은 저장소가 지정한 커밋이어야 한다. 새 checkout에서는 바닥 texture PNG, object PNG,
`bread` collision STL도 실제로 존재하는지 확인한다. 이 파일이 없으면 `test_assets.py`와 씬 생성이
실패하며, Python 패키지를 다시 설치해도 해결되지 않는다.

이번 이전에서는 Git 제외 규칙 때문에 누락된 12개 파일을 upstream에서 복구하고 해당 에셋만
Git 제외 대상에서 해제했다. 이전 PC의 바이너리가 없어 바이트 단위 동일성은 확인하지 못했다.
출처 커밋과 SHA-256은 [`assets/RECOVERY_2026-09-17.md`](assets/RECOVERY_2026-09-17.md)에 있다.
특히 바닥 texture가 없으면 기존 씬 생성기가 checker로 대체하므로, smoke 도구는 이 대체를 거부한다.

## 2. 전용 Conda 환경

```bash
sudo apt install ffmpeg
conda create -n gello-sim python=3.11 pip tk
conda run -n gello-sim python -m pip install -r sim_collect/requirements-sim.txt
conda run -n gello-sim python -m pip install -e . -e third_party/DynamixelSDK/python
```

시스템 sudo를 쓸 수 없으면 `conda install -n gello-sim -c conda-forge ffmpeg`로 같은 실행 파일과
`libx264` encoder를 환경 안에 설치할 수 있다.

`requirements-sim.txt`은 MuJoCo, capture, asset 도구와 테스트의 재현 핀이다. Pillow도 object/texture
검증과 변환 도구가 직접 import하므로 포함한다. IFQL 진단 영상 렌더러의 마지막 H.264/yuv420p
인코딩은 시스템 `ffmpeg`의 `libx264`를 사용하며, 없으면 추측성 codec으로 대체하지 않고 실패한다.

런처의 Python 선택 순서는 고정되어 있다.

1. `SIM_COLLECT_PY`
2. 현재 활성화된 `gello-sim`
3. `conda run -n gello-sim`으로 찾은 해당 환경의 실제 Python
4. 기존 laptop3 호환용 `.venv/bin/python`

자동 선택과 무관하게 실행 파일을 고정하려면 다음처럼 지정한다.

```bash
SIM_COLLECT_PY="$CONDA_PREFIX/bin/python" ./sim_collect/run_sim_collect.sh --fake-leader
```

## 3. 인터랙티브 실행

```bash
cd /home/junhyeong/gello_software_jazzy
conda activate gello-sim
DISPLAY=:1 ./sim_collect/run_sim_collect.sh --fake-leader
```

인터랙티브 모드는 `MUJOCO_GL`이 비어 있으면 `glfw`를 선택하고 MuJoCo viewer와 Tk GUI를 띄운다.
호출자가 `MUJOCO_GL`을 지정하면 런처는 덮어쓰지 않는다. 새 머신에서 실제 GELLO를 연결하기 전에는
`--fake-leader`를 유지한다.

## 4. 헤드리스 실행과 durable smoke

`--headless`는 sim에 `--no-viewer`를 전달하고 GUI를 시작하지 않는다. `MUJOCO_GL`이 비어 있으면
`egl`을 선택하며, 명시한 backend는 그대로 보존한다. GUI 없이 녹화를 제어하려면 IPC client가 필요하다.

```bash
SIM_COLLECT_PY="$CONDA_PREFIX/bin/python" \
SIM_COLLECT_IPC=tcp \
MUJOCO_GL=egl \
./sim_collect/run_sim_collect.sh --headless --fake-leader --depth
```

전체 수집 경로는 opt-in smoke 도구로 검증한다. 출력 디렉터리는 실행 전에 존재하면 안 된다.

```bash
conda run -n gello-sim python -m sim_collect.tools.smoke_collect \
  --output /tmp/gello-sim-smoke
```

이 도구는 TCP 6701/6702/6711/6712가 모두 비어 있는지 먼저 확인한 뒤, 정상 `stop_take` take와
SIGTERM 종료 중 finalization되는 take를 각각 만든다. HDF5, RGB MP4, depth PNG를 다시 열어 검증하고,
런처 종료 코드 143과 결과를 `summary.json`에 남긴다. 5591~5595는 열지 않는다.

런처는 sim, capture, GUI를 각각 독립 process group으로 시작한다. 종료 시 capture leader를 먼저
TERM하고 finalization을 기다린 뒤 sim/GUI leader를 TERM하며, 강제 종료는 기록해 둔 세 group에만 한다.
호출자나 같은 terminal의 unrelated Python process는 cleanup 대상이 아니다.

## 5. 테스트

```bash
cd /home/junhyeong/gello_software_jazzy
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl DISPLAY=:1 \
  conda run -n gello-sim python -m pytest -q -p no:cacheprovider sim_collect/tests
```

런처 단위 테스트 `sim_collect/tests/test_launcher.py`는 가짜 Python 프로세스를 사용해 인자, 상대/공백
경로, backend 보존, GUI 생략, capture-first cleanup, SIGTERM 143을 확인한다. MuJoCo, serial hardware,
GUI, TCP endpoint를 열지 않는다.

## 6. 정책 환경과 분리

`gello-sim`은 MuJoCo 수집과 policy endpoint probe용 환경이다. 실제 local ACT/Diffusion/FM 서버는
기존 `ros2_ur_ws/act_venv`, remote 서버는 `--remote-venv`가 가리키는 별도 Torch/LeRobot 환경을
계속 쓴다. `sim_collect/eval/serve_policy.sh`의 probe만 런처와 같은 순서로 Python을 찾으며, 실제
policy serving interpreter, 원격 명령, 포트 기본값은 바뀌지 않는다.

## 7. 완료 판정

- 두 서브모듈이 pinned commit에 있음
- texture/object/bread 에셋이 checkout에 있음
- `gello-sim` import 및 asset 테스트 통과
- `MUJOCO_GL=egl` offscreen render 통과
- `DISPLAY=:1` Tk 통과
- mocked launcher 테스트 통과
- opt-in `smoke_collect`의 두 take와 `summary.json` 통과

실제 GELLO 전원·시리얼·텔레옵은 이 마이그레이션 검증 뒤 별도 조작자 검증으로 남긴다.

### 2026-09-17 실측 결과

- 전체 `sim_collect/tests`: **259 passed, 8 skipped**, 49.34초.
  Skip은 과거 sim demo 6건, 실기 reference take 1건, 별도 actor 환경 1건이 없는 경우다.
- 이후 SIGINT 회귀 케이스를 추가한 launcher 단독 테스트: **11 passed**, 9.18초.
  SIGINT 130 / SIGTERM 143, capture-first 종료, 독립 프로세스 그룹,
  호출자 그룹의 무관한 Python 프로세스 보존을 확인했다.
- GLFW viewer + Tk GUI + 두 카메라 worker: fake GELLO/EEF로 실제 기동·종료 통과.
- EGL renderer는 `NVIDIA GeForce RTX 5070 Ti/PCIe/SSE2`로 확인했다.
- 최종 smoke 첫 take는 정상 STOP: 카메라별 RGB **120프레임**, depth **120프레임**.
- 두 번째 take는 녹화 중 SIGTERM: 카메라별 RGB **136프레임**, depth **136프레임**.
  두 take 모두 카메라별 **30.0 fps**, `problems=[]`; 모든 MP4 프레임 디코드,
  depth 첫/마지막 PNG 디코드, HDF5 벡터·MuJoCo 상태·종료 메타데이터 재열기 통과.
- 두 take의 기록된 씬 재구성: XML 일치, 에셋 29개, 첫/마지막 상태의
  물체 위치 재구성 오차는 두 take 모두 출력 정밀도에서 **0.00 mm**.
- 로그·환경 freeze·최종 take·`summary.json`:
  `ros2_ur_ws/log/jazzy_port_20260917/` 아래에 보존했다.

가짜 리더 기반 검증이며 실제 GELLO 시리얼은 열지 않았다. 기존 Conda `il` 환경,
실제 로봇, 정책 포트 5591~5595는 건드리지 않았다. 커밋은 하지 않았다.

## 8. 실제 GELLO로 시뮬 영상 수집

2026-09-17 검증 시 GELLO 시리얼 장치는 연결되어 있지 않았다. USB와 5 V 외부 전원을 연결하고
ROS GELLO publisher나 다른 시리얼 점유 프로그램을 먼저 종료한다. 기존 busy-port 거부 검사와
passive 동작은 유지하며, 실제 UR 로봇은 필요하지 않다.

```bash
cd /home/junhyeong/gello_software_jazzy
conda activate gello-sim
DISPLAY=:1 MUJOCO_GL=glfw ./sim_collect/run_sim_collect.sh --control-mode eef
```

GUI에서 ENGAGE를 두 번 눌러 승인한 뒤 `START TAKE` / `STOP TAKE`로 녹화한다.
기본 저장 위치는 `ros2_ur_ws/gello_logs/sim/`이며 `cam1.mp4`, `cam2.mp4`, `vectors.h5`가 생긴다.
깊이도 필요하면 실행 명령에 `--depth`를 추가한다. 장비 없이 연습할 때만 `--fake-leader`를 추가한다.
DISENGAGE는 팔 추종 해제이며 그리퍼는 별도 `Gripper PAUSE`로 멈춘다.
