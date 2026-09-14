# GELLO → UR7e "Put carrot in pot" EEF-delta 정책 실기 배포 — 런북 (REAL)

> ⚠️ **이 문서는 `carrot_in_pot` 데이터셋으로 학습한 ACT / Flow-Matching 정책을 **카르테시안(EEF) 델타 액션 공간**으로 실제 UR7e에서 자율 구동하기 위한 실기 런북입니다.**
> 대상은 **로봇 PC** (실제 UR7e + Robotiq 2F-85 + RealSense 2대가 연결된 머신)입니다.
> 기존 **ACT 배포**([`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md))·**FM 배포**([`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md))와 **같은 ROS2 패키지·같은 브리지·같은 핸드셰이크·같은 그리퍼 스택**을 씁니다. 바뀐 것은 **정책이 무엇을 내보내는가** 하나뿐입니다: 절대 관절각 7-D 대신 **카르테시안 델타 7-D**.
>
> # 🔴 현재 상태 — **이 브랜치의 그 무엇도 로봇을 움직인 적이 없다**
>
> 이 배포 경로(`action_space: eef_delta`)는 **실기에서 단 한 번도 실행되지 않았습니다.**
> 학습된 체크포인트도 **아직 존재하지 않습니다.** 아래 §9의 "실기 미검증" 목록을 **먼저** 읽으십시오.
> 이 문서는 "지금 이대로 돌려도 된다"는 문서가 **아니라**, 그 순서를 적어 둔 문서입니다.
>
> - **설계 계약 (정본)**: [`ros2_ur_ws/src/gello_policy/CARROT_EEF_DESIGN.md`](../../ros2_ur_ws/src/gello_policy/CARROT_EEF_DESIGN.md) — state/action 정의, deploy 시맨틱, 담당 분해
> - **📌 0.9 s 명령 지연 노트**: [`CARROT_EEF_LAG_NOTE_KO.md`](./CARROT_EEF_LAG_NOTE_KO.md) — 이 데이터셋의 녹화 `command` 스트림이 실제 관절보다 **~0.9 s 앞서 있다.** 그래서 **액션은 `command`가 아니라 달성된(achieved) 움직임에서 정의**되고, 관절 envelope도 관측 관절만으로 만들어졌다. 배포 전에 반드시 읽을 것
> - **형제 문서 (실기 검증된 joint 배포)**: [`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md)
> - **형제 문서 (FM 서버/타임아웃 배치의 원본)**: [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md)

## 요약 (TL;DR)

| 항목 | 값 |
|---|---|
| 대상 로봇 | UR7e + Robotiq 2F-85, ROS2 **Humble** |
| 태스크 문자열 | **`Put carrot in pot`** (FM은 CLIP text 조건부라 **글자 그대로** 넣어야 함) |
| 액션 공간 | **`eef_delta_v1`** — `[dx, dy, dz, drx, dry, drz, grip_cmd]` (base frame **델타**) |
| 관측 | `state` **16-D** = `[q1..q6, tcp_x, tcp_y, tcp_z, r11,r21,r31,r12,r22,r32, grip_pos]` + cam1/cam2 |
| 체크포인트 | **아직 없음.** 학습 후 이름 예정: ACT `Bigenlight/act_carrot_in_pot_eef`, FM `Bigenlight/flow_matching_carrot_in_pot_eef` (**둘 다 placeholder 이름**) |
| params 파일 | ACT `config/carrot_eef_act.yaml`, FM `config/carrot_eef_fm.yaml` (`ACT_PARAMS_FILE` / `FM_PARAMS_FILE`로 선택) |
| 한계값 출처 | `config/carrot_eef_limits.json` (54 take / 18,557 frame 실측, `scripts/dataset/make_carrot_eef_limits.py` 생성) |
| 시작 자세 | `[-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]` rad (**de-staled**, §8), 그리퍼 OPEN(0.0118), TCP ≈ (0.529, 0.119, 0.225) |
| IK | 폐형해 **branch 2**로 고정 (18,557 프레임 전부 branch 2, off-branch 0건) |
| 안전 | **카르테시안 3단 클램프 → IK → 기존 관절 클램프 2단 → 브리지 slew 0.625 rad/s.** 기존 관절 스택은 **한 겹도 제거되지 않았다** (§6) |
| 핸드셰이크 거부 | `~/start_execution`은 서버 RESET 응답이 `action_space == eef_delta_v1` **이고** `state_dim == 16`, `action_dim == 7`이어야만 arm한다. joint 체크포인트는 **절대** eef 모드를 구동할 수 없다 |
| **실기 상태** | 🔴 **전부 미검증.** 체크포인트 없음, 팔 구동 없음, 레이턴시 미측정 — §9 |

### 목차

1. [joint 모드와 무엇이 다른가](#1-joint-모드와-무엇이-다른가)
2. [하드웨어 / 환경](#2-하드웨어--환경)
3. [데이터셋 변환 (laptop3)](#3-데이터셋-변환-laptop3)
4. [kanu에서 학습](#4-kanu에서-학습)
5. [체크포인트 가져오기](#5-체크포인트-가져오기)
6. [서버 단독 smoke + 레이턴시 확인 (팔 구동 전 필수)](#6-서버-단독-smoke--레이턴시-확인-팔-구동-전-필수)
7. [배포 실행하기](#7-배포-실행하기)
8. [안전 모델 — 클램프 전체 사슬](#8-안전-모델--클램프-전체-사슬)
9. [실기 미검증 목록 (정직하게)](#9-실기-미검증-목록-정직하게)
10. [트러블슈팅](#10-트러블슈팅)

---

## 1. joint 모드와 무엇이 다른가

ACT/Diffusion/FM joint 배포는 정책이 **가야 할 관절각 그 자체**를 내보냈다. EEF 모드는 정책이 **이번 틱에 TCP를 얼마나 움직일지**를 내보낸다. 그 차이가 만드는 변경은 정확히 다섯 가지다.

| | joint 모드 (기존) | **eef_delta 모드 (이 문서)** |
|---|---|---|
| `state` | **7-D** `[q1..q6, grip_pos]` | **16-D** `[q1..q6, tcp_xyz, 6-D 회전, grip_pos]` |
| `action` | **7-D 절대 관절각** `[cmd1..cmd6, grip_cmd]` | **7-D 카르테시안 델타** `[dx,dy,dz,drx,dry,drz, grip_cmd]` |
| 액션 해석 | 그 틱의 목표 자세 | **누적**된다 — 내부 `target` 포즈에 매 틱 ⊕ |
| 안전 클램프 | 관절 envelope + live 대비 `max_dev_rad` | **위 둘 + 카르테시안 step/lead/box 3단이 앞에 추가** |
| 체크포인트 오배치 | 조용히 이상하게 움직임 | **핸드셰이크가 arming을 거부** |

### 1.1 회전이 왜 6-D인가

`state`의 회전 6개 값은 `R_tcp`의 **첫 두 열(column)** 이다. 오일러각/rotvec가 아니다. carrot TCP 자세는 단위행렬에서 약 175° 떨어져 있어 rotvec로 표현하면 **경계에서 부호가 뒤집히며(wrap) 연속성이 깨진다.** 6-D 표현은 그 지점이 없다. 학습·배포·검증이 전부
[`gello_policy/eef_space.py`](../../ros2_ur_ws/src/gello_policy/gello_policy/eef_space.py)의 `rot_to_6d` / `rot_from_6d`를 쓴다 — **직접 다시 구현하지 말 것.**

### 1.2 chunk 누적 (이게 가장 헷갈리는 부분)

ACT/FM은 **한 번의 관측으로 액션 청크 전체를 예측**하고, 그 뒤 여러 틱 동안 큐에서 꺼내 쓴다(ACT `n_action_steps=30`, FM `24`). 그래서 노드는 이렇게 동작한다:

```
서버 응답에 chunk_start=true  (= 네트워크가 이번 틱에 다시 돌았다)
    또는 arming 직후 첫 틱
        → anchor = TCP(live_q),  target = anchor          ← 여기서만 live 자세를 읽는다
매 틱
        → target = target ⊕ Δ                              ← live가 아니라 target에 더한다
```

**왜 live에 더하지 않는가:** 정책은 자기가 본 관측에서 출발하는 궤적을 예측한다. 팔은 브리지 slew(0.625 rad/s) 때문에 항상 조금 뒤처져 있으므로, 매 틱 **뒤처진 live 자세**에 델타를 더하면 움직임이 매 틱 조금씩 깎여 궤적이 **점점 줄어든다**. 그래서 target에 누적한다.

**그 대가가 lead 클램프다.** 누적하면 팔이 멈춰 있어도 target은 계속 앞으로 간다. `eef_max_lead_m` / `eef_max_lead_rad`가 그것을 막는다(§8).

### 1.3 IK 브랜치 고정

`~/start_execution` 시점에 `branch_id(start_pose)`를 **한 번** 계산해 에피소드 내내 그 폐형해 브랜치로만 IK를 푼다. carrot 데이터셋은 18,557 프레임 **전부 branch 2**이고 off-branch가 0건이므로, 브랜치가 튀는 것은 "다른 해로 갔다"가 아니라 **뭔가 잘못됐다**는 신호다. IK가 해를 못 찾으면 그 틱은 **live 자세를 그대로 발행(hold)** 하고 카운트하며, `eef_ik_fail_fault_n`(10) 연속이면 **FAULT**다.

### 1.4 핸드셰이크 거부 — 체크포인트 오배치가 조용히 실패하지 않는다

`~/start_execution`은 서버에 RESET을 보내고 응답을 검사한다:

```json
{"ok": true, "action_space": "eef_delta_v1", "state_dim": 16, "action_dim": 7,
 "policy_type": "act", "checkpoint": "/path/..."}
```

셋 중 하나라도 yaml의 `expected_action_space` / 16 / 7과 다르면 **arming을 거부**한다. joint 체크포인트를 eef yaml로, 혹은 그 반대로 띄우는 사고는 **팔이 움직이기 전에** 잡힌다. 이것이 이 모드에서 추가된 **유일한 새 게이트**이며, 나머지 안전 장치는 전부 기존 것 그대로다.

## 2. 하드웨어 / 환경

joint 배포와 **완전히 동일**하다. 새로 준비할 것이 없다.

- **팔** UR7e / **그리퍼** Robotiq 2F-85 (Modbus, 드라이버 socat 브리지 `/tmp/ttyUR` 공유)
- **카메라** RealSense 2대, **장치 시리얼**로 바인딩. cam1 = D435 `147122072740` (SCENE, 삼각대), cam2 = D435if `243222072700` (WRIST). ⚠️ `journalctl`에 보이는 `151623020789`/`322743060038`은 **ASIC 시리얼**이라 `serial_no:=`로 **절대 해석되지 않는다.** 자세한 함정은 [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md) §2 정정 박스. 실사용은 시리얼을 직접 적지 말고 `./launch_cameras.sh`에 맡길 것
- **ROS2 Humble** (rclpy py3.10) + **py3.12 추론 venv** `ros2_ur_ws/act_venv`
- **colcon 빌드**: `gello_policy` **와** `ur_gello_bringup` **둘 다** 필요하다 (런치가 `ur_gello_bringup` 노드를 띄운다). 빌드는 **yaml을 share로 설치하기 위해서도** 필요하다:

  ```bash
  cd /home/laptop3/gello_software_humble/ros2_ur_ws
  set +u && source /opt/ros/humble/setup.bash && set -u
  colcon build --packages-select ur_gello_bringup gello_policy --symlink-install
  source install/setup.bash
  # 두 yaml이 설치됐는지 확인:
  ls install/gello_policy/share/gello_policy/config/carrot_eef_*.yaml
  ```

## 3. 데이터셋 변환 (laptop3)

원본 take: `/home/laptop3/youngwoong_ws/Put_carrot_in_pot/take_*` (54개, EEF-teleop 모드, 30 fps). **depth는 쓰지 않는다.**

```bash
cd /home/laptop3/gello_software_humble
WT=$PWD
PY=/home/laptop3/youngwoong_ws/lr_env/bin/python        # py3.12, lerobot 0.6.1
export PYTHONPATH="$WT/ros2_ur_ws/src/ur_gello_bringup:$WT/ros2_ur_ws/src/gello_policy"

# (1) 변환: state 16 / action 7, cam1 마스터 클럭 30 fps 리샘플링
$PY scripts/dataset/convert_carrot_to_lerobot_eef.py

# (2) 독립 검증 — 변환기를 import하지 않고 h5 + ur_kin에서 state/action을 다시 유도해 대조
$PY scripts/dataset/validate_carrot_eef_conversion.py
```

> **⚠️ 액션은 `command`에서 오지 않는다.** 이 EEF-mode 세션의 녹화 `command`는 실제 관절보다 **~0.9 s 앞서 있다**(τ=0에서 중앙값 61 mm 오차, τ=+0.9 s에서 1 mm). 그래서 액션은 **달성된 관절**로부터 `Δ_t = TCP(q_{t+1}) ⊖ TCP(q_t)`로 정의된다. 원인 분석은 [`CARROT_EEF_LAG_NOTE_KO.md`](./CARROT_EEF_LAG_NOTE_KO.md).

> **한계값 재생성.** 배포 클램프는 별도 스크립트가 만든다. 데이터셋이 바뀌면 **반드시 다시 돌려** yaml을 갱신할 것:
> ```bash
> $PY scripts/dataset/make_carrot_eef_limits.py      # -> ros2_ur_ws/src/gello_policy/config/carrot_eef_limits.json
> ```
> 이 JSON의 `generated_at` / `git_commit`이 yaml 주석에 적혀 있다. **둘이 어긋나면 yaml이 낡은 것이다.**

## 4. kanu에서 학습

학습 번들은 [`scripts/training/carrot_eef/`](../../scripts/training/carrot_eef/)에 있고, 절차·환경변수 정본은 그 안의 **`README_KO.md`** 다. 여기서는 순서와 **지켜야 할 제약**만 적는다.

```bash
cd /home/laptop3/gello_software_humble/scripts/training/carrot_eef

./sync_to_kanu.sh --dry-run     # 먼저 dry-run으로 무엇이 전송되는지 본다
./sync_to_kanu.sh               # 실제 전송 (kanu:~/workspace/youngwoong/carrot_eef/ 밖으로는 절대 쓰지 않는다)

# --- 여기부터 kanu ---
ssh kanu
cd ~/workspace/youngwoong/carrot_eef/scripts/training/carrot_eef
nvidia-smi                       # 지금 비어 있는 GPU를 고른다. 기본값 없음 — GPU_ID는 필수
./preflight.sh                   # GPU 유휴 / 데이터셋 feature가 계약대로인지 / 디스크 >= 20 GB / lerobot-train 동작
DRY_RUN=1 GPU_ID=<n> ./train_act_carrot_eef.sh    # 명령만 출력, 아무것도 실행하지 않음
```

🛑 **반드시 지킬 것:**

- **GPU는 최대 2개.** kanu는 8× A4000을 여러 사람이 공유한다. `GPU_ID`는 **기본값이 없고** 안 주면 거부한다 — GPU 0을 자동으로 잡지 않는다.
- **디스크.** kanu는 98% 차 있다(여유 ~51 GB). `~/workspace/youngwoong/carrot_eef/` **밖에는 아무것도 쓰지 말 것.**
- **venv를 새로 만들지 말 것.** `LEROBOT_VENV`가 가리키는 기존 cube 환경(torch 2.11 cu128 + lerobot 0.6.1 + CLIP 캐시)을 재사용한다.
- **학습 시작은 사람이 한다.** 에이전트/스크립트가 자동으로 시작하지 않는다. `launch_detached.sh`로 띄우고 `runtime/` 아래 pid·로그를 확인한다.
- held-out은 **마지막 6 에피소드**(`--dataset.eval_split=0.111`)이며, `eval_offline_eef.py`가 같은 6개를 본다 — 학습 중 loss와 오프라인 open-loop 지표가 **같은 미학습 데이터**를 본다.

학습이 끝나면 **먼저 오프라인 open-loop 평가로 체크포인트를 고른다**(position MAE m / rotation MAE rad / gripper accuracy):

```bash
./eval_checkpoints.sh            # 또는 eval_offline_eef.py 를 직접
```

## 5. 체크포인트 가져오기

고른 체크포인트를 로봇 PC의 `ros2_ur_ws/src/gello_policy/checkpoints/<이름>` 아래로 가져온다(`--checkpoint`에 넘길 것은 `config.json` + `model.safetensors` + pre/post-processor json이 있는 **그 디렉터리 자체**다).

```bash
cd /home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy
# kanu에서 직접:
rsync -av kanu:~/workspace/youngwoong/carrot_eef/outputs/act_carrot_eef/checkpoints/<step>/pretrained_model/ \
    checkpoints/act_carrot_in_pot_eef/
# 또는 HF에 올렸다면:
# act_venv/bin/hf download Bigenlight/act_carrot_in_pot_eef --local-dir checkpoints/act_carrot_in_pot_eef
```

> ⚠️ `checkpoints/`는 메인 checkout으로 가는 **심링크**다. 경로를 바꾸지 말고, 그 안에 있는 기존 banana/cube 체크포인트를 **덮어쓰지 말 것.**

## 6. 서버 단독 smoke + 레이턴시 확인 (팔 구동 전 필수)

**팔에 붙이기 전에 서버 half만 먼저 띄운다.** 여기서 확인할 것은 두 가지다: (a) 핸드셰이크가 `action_space`/`state_dim`/`action_dim`을 제대로 보고하는가, (b) 추론 1회가 리더의 `act_timeout_s` 예산 안에 드는가.

```bash
cd /home/laptop3/gello_software_humble/ros2_ur_ws

# ACT (포트 5591)
ACT_CHECKPOINT=$PWD/src/gello_policy/checkpoints/act_carrot_in_pot_eef \
ACT_ACTION_SPACE=eef_delta_v1 ACT_DEVICE=cuda \
    ./src/gello_policy/scripts/run_act_server.sh

# FM (포트 5593). task 문자열은 학습 문구 그대로.
FM_CHECKPOINT=$PWD/src/gello_policy/checkpoints/fm_carrot_in_pot_eef \
FM_ACTION_SPACE=eef_delta_v1 FM_TASK="Put carrot in pot" \
FM_DEVICE=cuda FM_NUM_INTEGRATION_STEPS=10 \
    ./src/gello_policy/scripts/run_fm_server.sh
```

확인 항목:

- 기동 로그에 **`state_dim=16`, `action_dim=7`, `action_space=eef_delta_v1`** 이 찍히는가. 서버는 이것이 체크포인트와 모순되면(예: state_dim 7인데 `--action-space eef_delta_v1`) **거부하고 죽는다.** 그게 정상 동작이다.
- **`warmup done in Xs`** 가 뜨는가.
- FM이라면 이후 **`net refill`** 간격이 `act_timeout_s`(0.6 s) 안인가. 넘으면 **워치독을 넓히지 말고** `FM_NUM_INTEGRATION_STEPS`를 줄인다(예: 5). 근거는 [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md) §6.

## 7. 배포 실행하기

**먼저 카메라를 띄운다.** EXECUTE는 매 틱 fresh cam1/cam2를 요구하고, 없으면 `~/start_execution`이 거부하거나 in-EXECUTE 워치독이 FAULT를 낸다.

```bash
cd /home/laptop3/gello_software_humble/ros2_ur_ws
./launch_cameras.sh      # 별도 터미널. VIEW=false 로 뷰어 없이도 가능
```

### ACT

```bash
cd /home/laptop3/gello_software_humble/ros2_ur_ws
HEADLESS=true \
ACT_PARAMS_FILE=$PWD/src/gello_policy/config/carrot_eef_act.yaml \
ACT_ACTION_SPACE=eef_delta_v1 \
ACT_CHECKPOINT=$PWD/src/gello_policy/checkpoints/act_carrot_in_pot_eef \
    ./run_ur7e_act_real.sh
```

### Flow-Matching

```bash
cd /home/laptop3/gello_software_humble/ros2_ur_ws
HEADLESS=true \
FM_PARAMS_FILE=$PWD/src/gello_policy/config/carrot_eef_fm.yaml \
FM_ACTION_SPACE=eef_delta_v1 \
FM_TASK="Put carrot in pot" \
FM_CHECKPOINT=$PWD/src/gello_policy/checkpoints/fm_carrot_in_pot_eef \
    ./run_ur7e_fm_real.sh
```

> **`HEADLESS=true`는 Method B** — 펜던트 Play 없이 드라이버가 URScript를 직접 보낸다. **로봇이 REMOTE 모드**여야 한다. 펜던트에서 External Control을 직접 Play하려면 `HEADLESS`를 빼면 된다(Method A).

> 🛑 **세 값이 전부 일치해야 한다.** `*_PARAMS_FILE`(노드 쪽 `action_space`) · `*_ACTION_SPACE`(서버 쪽) · **체크포인트의 state_dim**. 하나라도 어긋나면 서버가 기동을 거부하거나 `~/start_execution`이 arming을 거부한다. 배너에 세 값이 전부 출력되므로 **시작 전에 눈으로 확인할 것.**
> 🪤 **가장 흔한 실수: `*_PARAMS_FILE`만 바꾸고 `*_ACTION_SPACE`를 빼먹는 것.** 그러면 서버는 기본값 `joint`로 뜨고, 리더는 eef를 기대하므로 **arming이 거부된다.** (거부는 올바른 실패다 — 팔은 움직이지 않는다.)

### 시작 handshake 타임라인

1. **t=0s** — UR 드라이버 + `scaled_joint_trajectory_controller` ACTIVE, `forward_position_controller` INACTIVE.
2. **t≈6s** — `policy_leader_node`가 **HOLD**로 기동. 매 틱 고정 `start_pose` `[-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]`를 **완전히 정지된 상태로** 발행하고 그리퍼는 OPEN(0.0118)을 유지한다. **서버는 아직 쿼리하지 않는다.**
3. **t≈8s** — `gello_move_to_start`가 그 HELD 자세로 chase → 수렴하면 STRICT 컨트롤러 전환 → 브리지 `~/resume`.
4. **팔이 start_pose에 파킹한다. 여기까지 자율 모션은 0이다.**
5. **오퍼레이터가 명시적으로 시작한다:**

   ```bash
   ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
   ```

   이 호출이 하는 일: (a) live 자세가 `start_pose`에서 **0.1 rad 이내**인지 확인, (b) 서버에 RESET을 보내고 **`action_space`/`state_dim`/`action_dim` 계약을 검사**(§1.4 — 틀리면 여기서 거부), (c) `start_branch = branch_id(start_pose)` 계산, (d) EXECUTE 진입.

6. **일시정지:**

   ```bash
   ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger
   ```

   현재 live 자세로 HOLD 전환 → 그 자리에서 정지. 그리퍼는 마지막 명령을 유지한다. 재-arm은 다시 `start_pose` ±0.1 rad을 요구한다.

### FAULT 리커버리

FAULT는 노드가 `/gello/joint_states` 발행을 **완전히 중단**하는 상태다 → 브리지 staleness 워치독이 팔을 세운다. **마지막 타깃을 계속 재발행하지 않는다**(fail-silent). eef 모드에서 FAULT를 내는 조건:

| 원인 | 로그 단서 |
|---|---|
| ZMQ 타임아웃 / 서버 에러 / `{"ok":false}` | `act_timeout_s` 초과 |
| 관측 stale (joint_states, grip_pos, cam1, cam2 중 하나) | `obs_timeout_s` 초과 |
| **IK 연속 실패** (eef 전용) | `eef_ik_fail_fault_n`(10) 연속 — 그 전까지는 매 틱 live 자세 hold + throttled WARN |

복구 절차:

1. **원인을 먼저 확인한다.** 특히 IK 실패가 원인이면 팔이 이미 데이터 분포 밖으로 나갔을 가능성이 높다 — 그대로 재-arm하지 말 것.
2. `~/hold`로 확실히 HOLD에 둔다.
3. 팔을 `start_pose` 근처로 되돌린다. 가장 안전한 방법은 **Ctrl-C 후 런처를 다시 실행**하는 것이다 — move-to-start 핸드셰이크가 다시 돈다.
   > ⚠️ 재-chase 모션은 **장애물을 모른다(collision-unaware).** 경로상에 사람/물체가 없는지 확인하고, E-STOP에 손을 둔 채로 진행할 것.
4. `~/start_execution`을 다시 호출한다. 게이트(0.1 rad + 핸드셰이크 계약)가 처음부터 다시 걸린다 — **캐시되는 것은 없다.**

## 8. 안전 모델 — 클램프 전체 사슬

정책이 무엇을 내보내든, 명령이 팔에 도달하기까지 **여섯 겹**을 통과한다. 위 세 겹이 이번에 추가된 카르테시안 층이고, **아래 세 겹은 joint 배포에서 한 줄도 바뀌지 않았다.**

```
정책 출력 Δ (7-D)
  │
  ├─(1) 틱당 step 캡        clamp_delta(Δ, eef_max_step_m, eef_max_step_rad)     ← norm 비례 축소
  │                          target = target ⊕ Δ                                  (chunk_start에서 re-anchor)
  ├─(2) lead 캡             |p_target − p_live| > eef_max_lead_m 이면 축소         ← norm 비례 축소
  ├─(3) 워크스페이스 박스    clamp_position_box(p_target, workspace_lo/hi)
  │
  ├──── ik_tcp_nearest(target, live_q, branch=start_branch) ──► 실패시 live 발행(hold) + 카운트
  │
  ├─(4) 관절 envelope       joint_limits_lo/hi (1.2× 데이터셋 범위)
  ├─(5) live 대비 편차       max_dev_rad (0.5 rad)
  └─(6) 브리지 slew          max_step_rad 0.0025 rad @ 250 Hz = 0.625 rad/s        ← 절대 상한
```

🛑 **(1)~(3)은 전부 norm 비례 축소이고 축별 clip이 아니다.** 축별로 자르면 명령된 **방향이 꺾인다**(`[2.0, 0.5]` → `[1.0, 0.5]`). 회전이 개입하면 더 나빠진다. 각 클램프는 throttled WARN을 남긴다.

### 실측 기반 값 (`config/carrot_eef_limits.json`, 54 take / 18,557 frame)

| 파라미터 | 값 | 어디서 왔나 |
|---|---|---|
| `eef_max_step_m` | **0.0140** m | 1-frame 달성 `\|dp\|` **최댓값 9.016 mm**의 1.5배, 올림 (p50 2.329 / p99 6.572 mm, n=17,049, **de-staled + 선형보간**) — 🪤 아래 주의 |
| `eef_max_step_rad` | **0.0500** rad | 1-frame `\|drot\|` 최댓값 0.0282 rad의 1.5배, 올림 |
| `eef_max_lead_m` | **0.0700** m | 30-frame(1 s) `\|dp\|` **중앙값 64.10 mm**, 0.01 올림 |
| `eef_max_lead_rad` | **0.1000** rad | 30-frame `\|drot\|` **중앙값 0.0958 rad**, 0.01 올림. ⚠️ 설계 문서 §3의 기본값 0.30이 **아니다** — 실측 중앙값이 0.096이라 0.30은 3배 느슨하다 |
| `workspace_lo/hi` | `[0.3846, -0.2445, 0.0073]` / `[0.7391, 0.4193, 0.3362]` | 관측 TCP 범위(0.0001 m 격자로 바깥쪽 양자화) ± 0.03 m. **단 z 하한만 −0.01 m** — 테이블 아래는 복구 가능한 이탈이 아니다 |
| `joint_limits_lo/hi` | `[-3.8049, …]` / `[-2.5787, …]` | 1.2× 대칭 확장. ⚠️ **관측 관절만** 사용 (녹화 `command`는 0.9 s 앞서 있어 팔이 실제로 취한 적 없는 자세다) |
| `eef_ik_fail_fault_n` | **10** | 연속 IK 실패 → FAULT |

🪤 **`eef_max_step_m` 14 mm는 "nearest"가 아니라 "보간" 기준이다.** 위 1-frame 통계는 **A1 변환기가 데이터셋 액션을 만드는 방식 그대로**(de-staled 격자 `cam1_t + tau_q`에서 30 Hz **선형보간**) 리샘플링한 계열이다. **nearest-timestamp로 다시 유도하지 말 것** — 관절 캐시가 ~66.7 Hz에 dt가 흔들려(중앙값 15.8 ms, p99 25.9 ms) nearest는 2~3 샘플 간격에 스냅되고, `tau_q = 0.89 s`에서는 **8.8%의 프레임이 같은 샘플을 반복(움직임 0)하고 24%가 3칸을 건너뛰는** 나쁜 위상에 떨어진다. 그러면 단일 프레임 최댓값이 11.5 mm로 부풀어 이 클램프가 **0.018까지 올라가지만, 팔은 실제로 그렇게 빠르게 움직인 적이 없다.** (그 계열은 JSON에 `frame_1_destaled_nearest`로 감사용으로만 남아 있다.)

🔗 **`start_pose`와 `eef_max_step_m`은 독립 노브가 아니다.** 둘 다 변환기의 리샘플링 레시피에서 나온다. **A1이 리샘플링을 또 바꾸면 `make_carrot_eef_limits.py`를 다시 돌리고 두 값을 함께 재동기화할 것.** 손으로 튜닝하지 말 것.

📌 **기준이 섞여 있는 것은 의도적이다.** `start_pose`와 1-frame 통계는 **de-staled** 기준이고, **관절 envelope · 워크스페이스 박스 · 30-frame 통계는 변환기의 `tau=0` 기준**이다. 상수 시프트는 min/max를 움직이지 못하는 데다 de-staling은 관절 기록 끝에서 **마지막 7.8% 프레임을 버리므로**, `tau=0` 쪽이 **더 넓은(보수적인)** 집합이다.

🛑 **banana 관절 envelope를 재사용하지 말 것.** banana의 `shoulder_pan`은 `+2.40..+3.67`인데 carrot의 pan은 **−3.19**다. 그대로 쓰면 **모든 명령이 +2.40으로 클립**되어 ~5.6 rad 어긋난다. 그래서 carrot yaml이 별도로 존재한다.

🛑 **`eef_max_lead_*`를 "반응이 느리다"는 이유로 올리지 말 것.** 이것은 반응성 노브가 아니라 **누적이 폭주하지 않게 막는 유일한 장치**다. 팔이 slew·protective stop·마찰로 멈춰 있는 동안 target은 계속 전진하므로, 이 값이 곧 "팔이 멈춘 걸 눈치채기까지 target이 앞서 갈 수 있는 최대 거리"다.

### 그 외 (joint 배포와 동일)

- **속도 상한**: 브리지 `max_step_rad`는 **무변경**이며 정책이 뭘 내든 0.625 rad/s를 넘지 못한다. **올리지 말 것.**
- **워치독 순서**: ACT `act_timeout_s 0.5` / `obs_timeout_s 0.5` / staleness `0.5`, FM `0.6 < 0.7 < 0.8`. FM 쪽 순서 불변식의 근거는 [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md) §6 — **넓히지 말고 Euler 스텝을 줄일 것.**
- **그리퍼 단일 writer**: `gello_gripper_bridge`는 띄우지 않는다. `policy_leader_node`가 `/robotiq_gripper/command_percent`의 유일한 publisher이며, `action[6]`을 0..1로 clip해 발행한다.
- **E-STOP**: 어떤 소프트웨어 클램프도 물리 안전을 대체하지 않는다. 펜던트 E-STOP을 항상 손 닿는 곳에. 그리퍼는 연결 시 auto-calibrate(open/close sweep)하므로 **손가락을 치울 것.**

## 9. 실기 미검증 목록 (정직하게)

> 🔴 **이 브랜치의 어떤 것도 로봇을 움직인 적이 없다.** 아래는 "아직 안 한 것"이지 "아마 될 것"이 아니다.

**아직 존재하지도 않는 것:**

- **학습된 EEF 체크포인트** — ACT도 FM도 없다. `expected_model_id`의 `Bigenlight/act_carrot_in_pot_eef` / `Bigenlight/flow_matching_carrot_in_pot_eef`는 **placeholder 이름**이며 그 리포는 존재하지 않는다.
- **오프라인 open-loop 지표** — position/rotation MAE, gripper accuracy 전부 미측정. 체크포인트 선택 근거가 아직 없다.

**존재하지만 실기에서 확인 안 된 것:**

- **팔 구동** — `action_space: eef_delta` 경로로 실물 UR7e를 움직인 적이 **0회**.
- **핸드셰이크 거부** — 계약 검사가 실제 서버 응답에 대해 동작하는지 실기 미확인(단위 테스트만).
- **chunk 누적 / re-anchor** — 실제 정책의 `chunk_start` 타이밍에 대해 미확인.
- **카르테시안 클램프 3종** — 실기에서 한 번도 발동한 적이 없다. 특히 **lead 클램프**는 실제 팔의 지연 특성과 만나 본 적이 없다.
- **IK 브랜치 고정** — 데이터에서는 18,557 프레임 전부 branch 2였지만, **정책이 만들어 낸 자세**에서 브랜치가 유지되는지는 다른 문제다.
- **추론 레이턴시** — 이 로봇 PC의 GPU에서 EEF 체크포인트의 refill p99가 예산 안인지 미측정 (§6).
- **로봇 PC colcon build** — 워크트리에서의 빌드는 통과했으나 로봇 PC 실환경 빌드는 별도.
- **0.9 s 지연이 배포에도 나타나는가** — 녹화 세션의 지연 원인이 `policy_leader_node`가 브리지를 joint 모드로 몰 때도 재현되는지는 **미해결**이다. [`CARROT_EEF_LAG_NOTE_KO.md`](./CARROT_EEF_LAG_NOTE_KO.md) 참조. 재현된다면 lead 클램프가 정상 동작 중에 상시 발동할 수 있다.

**첫 실기 세션 순서 (제안):** (1) colcon build → (2) 데이터셋 변환 + 검증 → (3) kanu 학습 + 오프라인 eval로 체크포인트 선택 → (4) 체크포인트 fetch → (5) **서버 단독** smoke + 레이턴시 (§6) → (6) 카메라/그리퍼 라이브 확인 → (7) 핸드셰이크까지만 돌려 파킹 확인 → (8) **감독 하에** `~/start_execution`, E-STOP에 손을 둔 채.

## 10. 트러블슈팅

- **`~/start_execution`이 계속 거부된다.** 세 가지 중 하나다. (a) live 자세가 `start_pose`에서 0.1 rad 넘게 떨어짐 — 핸드셰이크 수렴 로그 확인. (b) fresh 관측 셋(카메라 2대 + 그리퍼 위치)이 불완전 — `ros2 topic hz`로 확인. (c) **핸드셰이크 계약 불일치** — 로그에 서버가 보고한 `action_space`/`state_dim`/`action_dim`이 찍힌다. 대개 `*_ACTION_SPACE`를 안 준 것이다(§7 🪤).
- **서버가 기동하자마자 죽는다 (action space 모순).** `--action-space eef_delta_v1`인데 체크포인트 `state_dim`이 7이거나 그 반대다. **의도된 거부다** — 체크포인트를 고치지 말고 어느 쪽이 틀렸는지 확인할 것.
- **팔이 데이터 분포보다 느리게/조금씩만 움직인다.** lead 클램프가 상시 발동 중일 가능성. throttled WARN 빈도를 본다. 원인이 브리지 slew라면 **`eef_max_lead_m`을 올리는 게 아니라** 왜 팔이 못 따라가는지를 먼저 본다(§8 🛑).
- **IK 실패 WARN이 계속 뜬다.** 팔이 워크스페이스 가장자리이거나 특이점 근처, 또는 브랜치가 튄 상황이다. 10회 연속이면 FAULT로 간다. 그 전에 `~/hold`로 세우고 §7 리커버리를 따르는 편이 낫다.
- **yaml을 고쳤는데 반영이 안 된다.** `--symlink-install`로 빌드했으면 share 복사본이 곧 src 파일이지만, 일반 `colcon build`면 **스냅샷**이다. 다시 빌드하거나 `*_PARAMS_FILE`을 src 경로로 직접 지정할 것. 배너에 실제 사용 경로가 출력된다.
- **FM: refill이 느려 FAULT-loop.** `FM_NUM_INTEGRATION_STEPS`를 줄인다(예: 5). **워치독을 넓히지 말 것** — 자세한 근거는 [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md) §9.
- **카메라가 조용히 안 뜬다 (노드는 살아 있는데 프레임 0).** 십중팔구 **ASIC 시리얼**로 바인딩한 것이다. `ros2 topic info`는 publisher 1을 보고하므로 `ros2 topic hz`로만 구분된다. `launch_cameras.sh`를 쓸 것.
- **포트 충돌.** ACT 5591 / Diffusion 5592 / FM 5593. FM 런처는 시작 전에 5593 점유를 **거부**한다(stale 서버가 팔을 몰지 못하도록). `pkill -f fm_server.py ; ss -ltnp | grep :5593`.

---

## 관련 문서

- [`CARROT_EEF_DESIGN.md`](../../ros2_ur_ws/src/gello_policy/CARROT_EEF_DESIGN.md) — **설계 계약 정본.** state/action 정의(§2), deploy 시맨틱(§3), ZMQ 프로토콜 v2(§4)
- [`CARROT_EEF_LAG_NOTE_KO.md`](./CARROT_EEF_LAG_NOTE_KO.md) — 녹화 `command`가 왜 0.9 s 앞서는가, 배포에서도 재현되는가
- [`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md) — **실기 검증된** joint ACT 배포. Method A/B, 4-터미널 워크플로우, recorder, 리커버리의 원본
- [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md) — FM 서버·Euler-ODE·타임아웃 배치의 원본
- [`scripts/training/carrot_eef/README_KO.md`](../../scripts/training/carrot_eef/README_KO.md) — kanu 학습 절차 정본
- [`gello_policy/README.md`](../../ros2_ur_ws/src/gello_policy/README.md) — 패키지 개요, 노드/토픽 레퍼런스, 빌드
- [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md) — 이 데이터셋을 만든 레코더
- [`GELLO_UR7E_EEF_MODE.md`](./GELLO_UR7E_EEF_MODE.md) — 사람이 GELLO로 EEF 텔레옵하는 경로 (이 데이터셋이 기록된 모드)
