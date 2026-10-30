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
./run_real_policy.sh common-bc-carrot --dry-run
./run_real_policy.sh common-ifql-carrot --dry-run
./run_real_policy.sh svf-carrot --dry-run
./run_real_policy.sh dsrl-carrot --dry-run
./run_real_policy.sh dsrl-orange --dry-run
```

`--dry-run`은 checkpoint/norm stats/deploy yaml/hash/port를 검증하고 로봇·server·ROS를 시작하지 않음. recording이 켜진 profile은 renderer/h5py/ffmpeg도 검증함.

새 common encoder + 새 lighted carrot data로 학습한 SVF는 recorded frame까지 아래처럼 확인함.

```bash
./setup_jazzy/svf_common_encoder_smoke.sh
```

이 smoke는 **로봇·ROS·실시간 카메라를 시작하지 않음**. 저장된 cam1/cam2 JPEG와 joint state 48개를 localhost server에 보내서 encoder package hash/shape, checkpoint restore, 2055-D actor observation, 7-D finite action, latency를 확인함. 로그는 `log/svf_common_encoder_smoke/` 아래에 보존됨.

이번 세션의 새 common encoder/lighted carrot **Base Flow와 IFQL을 함께** 확인하는 명령은 아래임.

```bash
./setup_jazzy/common_lighted_ifql_smoke.sh
```

두 profile은 같은 `params_50000.pkl`, 같은 frozen actor, 같은 common encoder와 56-D per-position z-score action contract를 사용함. `common-bc-carrot`은 frozen actor에서 K=1만 생성하고 critic 선택을 하지 않는 Base Flow이며, `common-ifql-carrot`은 같은 actor에서 32개를 생성해 twin-Q mean으로 best-of-32를 선택함. smoke 입력도 예전 benchmark가 아니라 새 lighted carrot validation episode 56에서 임시로 만든 recorded JPEG/state pair를 사용함.

UR joint의 $2\pi$ 동치성도 adapter에서 처리함. 관측은 학습 데이터의 joint branch로 canonicalize한 뒤 proprio z-score를 적용하고, 생성된 raw joint command는 현재 로봇 joint와 가장 가까운 branch로 되돌린 뒤 30 Hz queue에 넣음.

## 2. 실제 실행

```bash
# Flow BC K=1
HEADLESS=true ./run_real_policy.sh ifql-bc-carrot

# IFQL critic + BoN32
HEADLESS=true ./run_real_policy.sh ifql-mean-carrot
HEADLESS=true ./run_real_policy.sh ifql-min-carrot

# 새 lighted-carrot common encoder: 공정한 같은-actor 비교
HEADLESS=true ./run_real_policy.sh common-bc-carrot
HEADLESS=true ./run_real_policy.sh common-ifql-carrot

# 핵심 method
HEADLESS=true ./run_real_policy.sh svf-carrot

# real-demo DSRL
HEADLESS=true ./run_real_policy.sh dsrl-carrot
HEADLESS=true ./run_real_policy.sh dsrl-orange
```

`svf-carrot`은 `run_ur7e_svf_carrot_new.sh`로 고정되어 있으며 아래 artifact를 사용함.

새 common Base Flow/IFQL profile은 기존 IFQL profile을 덮지 않으며 아래 배포 후보에 고정됨.

```text
/home/junhyeong/ifql_common_lighted_20260919/carrot_seed0/params_50000.pkl
├── common-bc-carrot   sampler=bc, K=1
└── common-ifql-carrot sampler=bon, K=32, q_agg=mean
```

```text
/home/junhyeong/svf_real_carrot_commonenc_20260919/
├── common encoder: /home/junhyeong/common_encoder_lighted_20260919/carrot_seed0
└── SVF: campaign/train/exp/qflow_rc/svf_sim/
         new_commonenc_c3_k06_g025_c144_clip2_p1_s0_100k/params_100000.pkl
```

즉 실물 실행 순서는 `./launch_cameras.sh`를 별도 터미널에 먼저 띄운 뒤, 실행 터미널에서 `HEADLESS=true ./run_real_policy.sh svf-carrot`임. launcher를 직접 우회해서 server/ROS를 따로 띄우지 말 것.

server가 warmup되고 arm이 start pose에서 HOLD한 뒤 viewer의 START를 눌러야 policy가 움직임. task가 바뀌면 물체 배치와 start pose도 해당 profile에 맞춰야 함.

기록 구간은 viewer의 **START부터 HOLD까지**임. HOLD는 로봇을 먼저 즉시 정지시키고, 별도 background RESET으로 직전 HDF5를 finalize하고 MP4를 렌더링함. finalize가 끝나기 전에 START를 다시 누르면 episode 혼합을 막기 위해 잠시 거절되므로, 로그의 `HOLD episode finalize complete`를 확인한 뒤 다음 trial을 시작함.

## 3. 기록 위치

IFQL/DSRL launch는 새 디렉터리를 만들고 기본으로 HDF5+MP4를 함께 남김. 현재 `svf-carrot`의 common-encoder adapter는 recording 전용 CLI를 받지 않아서 `REAL_EVAL_RECORDING=0`으로 고정됨. 따라서 SVF는 server episode log만 남고, HDF5+MP4가 필요하면 adapter recording 계약을 먼저 이식해야 함.

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
