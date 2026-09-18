# Real policy eval quickstart (5070)

현재 5070에 stage된 checkpoint를 안전 launcher + HDF5/MP4 recorder로 실행하는 진입점임.

```bash
cd ~/gello_software_jazzy/ros2_ur_ws
./run_real_policy.sh --list
```

## 0. 먼저 카메라

별도 터미널에서 아래를 계속 켜둠.

```bash
cd ~/gello_software_jazzy/ros2_ur_ws
./launch_cameras.sh
```

## 1. 로봇 없이 preflight

```bash
./run_real_policy.sh ifql-min-carrot --dry-run
./run_real_policy.sh svf-carrot --dry-run
./run_real_policy.sh dsrl-carrot --dry-run
./run_real_policy.sh dsrl-orange --dry-run
```

`--dry-run`은 checkpoint/norm stats/deploy yaml/hash/port/renderer/h5py/ffmpeg만 검증하고 로봇·server·ROS를 시작하지 않음.

## 2. 실제 실행

```bash
# Flow BC K=1
HEADLESS=true ./run_real_policy.sh ifql-bc-carrot

# IFQL critic + BoN32
HEADLESS=true ./run_real_policy.sh ifql-mean-carrot
HEADLESS=true ./run_real_policy.sh ifql-min-carrot

# 핵심 method
HEADLESS=true ./run_real_policy.sh svf-carrot

# real-demo DSRL
HEADLESS=true ./run_real_policy.sh dsrl-carrot
HEADLESS=true ./run_real_policy.sh dsrl-orange
```

server가 warmup되고 arm이 start pose에서 HOLD한 뒤 viewer의 START를 눌러야 policy가 움직임. task가 바뀌면 물체 배치와 start pose도 해당 profile에 맞춰야 함.

## 3. 기록 위치

매 launch는 새 디렉터리를 만들고 기본으로 HDF5+MP4를 함께 남김.

```text
log/ifql/real_YYYYMMDD/<profile>/run_<UTC timestamp>_pid<PID>/
├── launch_manifest.json
├── server_stdout.log
├── hdf5/*.h5
└── policy_render.mp4
```

IFQL mean/min은 **같은 학습 checkpoint**를 사용함. 차이는 BoN32 후보를 점수화할 때 twin critic head를 각각 `mean`/`min`으로 줄이는 방식뿐이며, `launch_manifest.json`의 `policy.q_agg`에 기록됨.

## 4. 종료·판정

- 종료: 실행 터미널에서 `Ctrl-C`; 필요하면 pendant E-STOP.
- stale port 확인: `ss -ltnp | grep -E ':5595|:5596'`
- 결과를 유효 trial로 세기 전 `launch_manifest.json`이 `finalized`인지, HDF5/MP4 hash가 있는지, server log에 fault가 없는지 확인함.
- DSRL은 아직 real rollout 성능이 확인되지 않은 checkpoint임. 첫 실행은 workspace를 넓게 비우고 1회 smoke부터 진행함.
