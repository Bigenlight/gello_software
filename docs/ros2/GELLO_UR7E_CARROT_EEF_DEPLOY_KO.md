# GELLO → UR7e "Put carrot in pot" EEF-delta 정책 실기 배포 — 런북 (REAL)

> ⚠️ **이 문서는 `carrot_in_pot` 데이터셋으로 학습한 ACT / Flow-Matching 정책을 **카르테시안(EEF) 델타 액션 공간**으로 실제 UR7e에서 자율 구동하기 위한 실기 런북입니다.**
> 대상은 **로봇 PC** (실제 UR7e + Robotiq 2F-85 + RealSense 2대가 연결된 머신)입니다.
> 기존 **ACT 배포**([`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md))·**FM 배포**([`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md))와 **같은 ROS2 패키지·같은 브리지·같은 핸드셰이크·같은 그리퍼 스택**을 씁니다. 바뀐 것은 **정책이 무엇을 내보내는가** 하나뿐입니다: 절대 관절각 7-D 대신 **카르테시안 델타 7-D**.
>
> # 🔴 현재 상태 — **이 브랜치의 그 무엇도 로봇을 움직인 적이 없다**
>
> 이 배포 경로(`action_space: eef_delta`)는 **실기에서 단 한 번도 실행되지 않았습니다.**
> **체크포인트는 2026-09-15 현재 kanu에 존재합니다** — ACT는 50k까지 학습이 끝났고(`010000`~`050000`
> 5개) 오프라인 open-loop 평가도 끝났으며, FM은 아직 학습 중입니다(§4·§9). **그러나 그중
> 무엇도 로봇을 움직인 적이 없고 로봇 PC로 내려온 적도 없습니다.**
> 아래 §9의 "실기 미검증" 목록을 **먼저** 읽으십시오.
> 이 문서는 "지금 이대로 돌려도 된다"는 문서가 **아니라**, 그 순서를 적어 둔 문서입니다.
>
> - **설계 계약 (정본)**: [`ros2_ur_ws/src/gello_policy/CARROT_EEF_DESIGN.md`](../../ros2_ur_ws/src/gello_policy/CARROT_EEF_DESIGN.md) — state/action 정의, deploy 시맨틱, 담당 분해
> - **📌 0.9 s 기록 지연 노트**: [`CARROT_EEF_LAG_NOTE_KO.md`](./CARROT_EEF_LAG_NOTE_KO.md) — 이 데이터셋의 녹화 `ur_joint_states`가 나머지 스트림보다 **~0.9 s 늦게 스탬프돼 있다.** **팔이 늦은 게 아니라 레코더가 늦게 찍은 것이다**(단일 spin 스레드가 depth 기록에 굶어 KEEP_LAST 큐가 항상 차 있었다 — 나이 = `QoS depth ÷ 발행률`). 카메라·GELLO·그리퍼는 신선하고(0.022 s), 밀린 것은 100 Hz 이상으로 발행되는 로봇 상태 테이블뿐이다. 그래서 **액션은 `command`가 아니라 달성된(achieved) 움직임에서 정의**되고(차분에서 상수 오프셋이 소거된다), 관절 envelope도 관측 관절만으로 만들어졌다. 배포 전에 반드시 읽을 것
> - **형제 문서 (실기 검증된 joint 배포)**: [`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md)
> - **형제 문서 (FM 서버/타임아웃 배치의 원본)**: [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md)

## 요약 (TL;DR)

| 항목 | 값 |
|---|---|
| 대상 로봇 | UR7e + Robotiq 2F-85, ROS2 **Humble** |
| 태스크 문자열 | **`Put carrot in pot`** (FM은 CLIP text 조건부라 **글자 그대로** 넣어야 함) |
| 액션 공간 | **`eef_delta_v1`** — `[dx, dy, dz, drx, dry, drz, grip_cmd]` (base frame **델타**) |
| 관측 | `state` **16-D** = `[q1..q6, tcp_x, tcp_y, tcp_z, r11,r21,r31,r12,r22,r32, grip_pos]` + cam1/cam2 |
| 체크포인트 | **kanu에만 있다** (`kanu:~/workspace/youngwoong/carrot_eef/outputs/{act,fm}_carrot_eef/checkpoints/<step>/pretrained_model`). 로봇 PC에는 아직 없다 — §5. yaml의 `expected_model_id`(`Bigenlight/act_carrot_in_pot_eef` / `Bigenlight/flow_matching_carrot_in_pot_eef`)는 **HF에 올린 적 없는 placeholder 이름**이고 ZMQ 경로에서는 강제되지 않는다 |
| params 파일 | ACT `config/carrot_eef_act.yaml`, FM `config/carrot_eef_fm.yaml` (`ACT_PARAMS_FILE` / `FM_PARAMS_FILE`로 선택) |
| 한계값 출처 | `config/carrot_eef_limits.json` (54 take / 18,557 frame 실측, `scripts/dataset/make_carrot_eef_limits.py` 생성) |
| 시작 자세 | `[-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]` rad (**de-staled**, §8), 그리퍼 OPEN(0.0118), TCP ≈ (0.529, 0.119, 0.225) |
| IK | 폐형해 **branch 2**로 고정 (18,557 프레임 전부 branch 2, off-branch 0건) |
| 안전 | **카르테시안 3단 클램프 → IK → 기존 관절 클램프 2단 → 브리지 slew 0.625 rad/s.** 기존 관절 스택은 **한 겹도 제거되지 않았다** (§6) |
| 핸드셰이크 거부 | `~/start_execution`은 서버 RESET 응답이 `action_space == eef_delta_v1` **이고** `state_dim == 16`, `action_dim == 7`이어야만 arm한다. joint 체크포인트는 **절대** eef 모드를 구동할 수 없다 |
| **실기 상태** | 🔴 **전부 미검증.** 팔 구동 0회, 로봇 PC 빌드·레이턴시 미측정, 클램프·핸드셰이크 전부 단위 테스트만 — §9 |

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

  🪤 **colcon이 `act_venv`를 훑다가 `RuntimeError: Failed to determine Python package name …
  numpy/_core/tests/examples/…` 트레이스백을 두 번 찍는다. 빌드 실패가 아니다.**
  `act_venv`는 이 워크트리의 **심링크**(메인 checkout의 py3.12 venv)라 `COLCON_IGNORE`를
  넣을 수 없다 — 메인 checkout은 건드리지 않는다. `Finished <<< gello_policy` 와
  `Summary: 2 packages finished` 가 보이면 정상이다(같은 함정이
  [`GELLO_UR7E_EEF_MODE.md`](./GELLO_UR7E_EEF_MODE.md) §2 "빌드 & 배포"에도 적혀 있다).

## 3. 데이터셋 변환 (laptop3)

원본 take: `/home/laptop3/youngwoong_ws/Put_carrot_in_pot/take_*` (54개, EEF-teleop 모드, 30 fps). **depth는 쓰지 않는다.**

```bash
cd /home/laptop3/gello_software_humble
WT=$PWD
PY=/home/laptop3/youngwoong_ws/lr_env/bin/python        # py3.12, lerobot 0.6.1
export PYTHONPATH="$WT/ros2_ur_ws/src/ur_gello_bringup:$WT/ros2_ur_ws/src/gello_policy"

# (1) 변환: state 16 / action 7, cam1 마스터 클럭 30 fps 리샘플링
#     --data/--out은 기본값이 아니다(--out은 필수, --data 기본값은 상대경로 "Put_carrot_in_pot").
#     --drop-stale-tail은 기본이 아니지만 **채택된 변환본의 필수 플래그**다 (아래 🛑).
$PY scripts/dataset/convert_carrot_to_lerobot_eef.py \
    --data /home/laptop3/youngwoong_ws/Put_carrot_in_pot \
    --out  /home/laptop3/youngwoong_ws/carrot_in_pot_eef_lerobot \
    --drop-stale-tail

# (2) 독립 검증 — 변환기를 import하지 않고 h5 + ur_kin에서 state/action을 다시 유도해 대조
#     --raw/--lerobot 기본값은 위 두 절대경로 그대로라 생략해도 되지만,
#     --drop-stale-tail은 **변환기와 똑같이** 줘야 한다(프레임 수 계약이 달라진다).
$PY scripts/dataset/validate_carrot_eef_conversion.py --drop-stale-tail
```

> 🛑 **`--drop-stale-tail`을 빼면 조용히 다른(폐기된) 데이터셋이 나온다.** de-staling
> 뒤 take 끝의 ~27 프레임에는 대응하는 관절 샘플이 없어, 플래그가 없으면 마지막 샘플에
> **고정(frozen tail)** 되고 유한차분 액션이 그것을 **0 delta의 연속**으로 만든다 — 정책이
> "마지막에는 멈춘다"를 배운다. kanu의 `preflight.sh`가 `meta/source_takes.json`의
> `n_tail_dropped`로 이것을 잡아 거부한다(탈출구 `ALLOW_STALE_TAIL=1`).
> 검증기도 같은 플래그가 없으면 **`episodes declare dropped tail frames but
> --drop-stale-tail was not passed`로 FAIL한다**(실측). 채택된 변환본의 전수 검증 결과는
> `scripts/dataset/carrot_eef_validation_2026-09-14.json`(`verdict: PASS`, 54 take /
> 17,085 frame)에 있다.
> 📌 **변환은 이미 끝나 있다** — `/home/laptop3/youngwoong_ws/carrot_in_pot_eef_lerobot`
> (`total_frames 17085`, `total_episodes 54`, `fps 30`, `codebase_version v3.0`)이 그것이다.
> 변환기는 `--out`이 이미 있으면 `[!] output dir already exists`로 **거부한다** —
> 다시 만들 이유가 없으면 이 절은 건너뛰고 §4로 간다.

> **⚠️ 액션은 `command`에서 오지 않는다.** 이 EEF-mode 세션에서 녹화 `command`와 녹화
> `ur_joint_states`는 **~0.9 s 어긋나 있다**(τ=0에서 중앙값 61 mm 오차, τ=+0.9 s에서 1 mm).
> **원인은 로봇도 브리지도 아니고 레코더다** — GUI 레코더의 단일 spin 스레드가 9월에 추가된
> depth 기록(≈5.3 MB/s)에 굶어 라운드 주파수가 98–100 Hz → 60–69 Hz로 내려갔고, **발행률이
> 그보다 높은 토픽만** KEEP_LAST 큐에 갇혀 `나이 = QoS depth ÷ 발행률`만큼 늦게 스탬프됐다
> (`/joint_states` depth 100 ÷ ~101 Hz ≈ **0.99 s**, `tcp_pose`/`wrench` depth 50 ÷ ~101 Hz ≈
> **0.50 s**). 30 Hz인 **카메라는 0.022 s밖에 안 낡았고**(depth 헤더 스탬프로 절대 측정),
> GELLO·그리퍼도 신선하다. 즉 **늦은 것은 관절 테이블 하나뿐이고, One-Euro·슬루·ROS 전송·
> 서보 추종 지연과는 무관하다**(잔차가 0.5 mm까지 떨어지는 것 자체가 "형상 보존 = 순수 지연"의
> 증거다). 그래서 액션은 **달성된 관절**로부터 `Δ_t = TCP(q_{t+1}) ⊖ TCP(q_t)`로 정의된다 —
> 두 프레임이 같은 (늦은) 테이블에서 오므로 상수 시프트가 차분에서 소거되고, 남는 문제는
> **이미지 ↔ 상태 정렬** 하나뿐이라 변환기가 그것만 `tau_q ≈ 0.90 s`로 되돌린다.
> 전체 근거·반증 실험은 [`CARROT_EEF_LAG_NOTE_KO.md`](./CARROT_EEF_LAG_NOTE_KO.md).

> **한계값 재생성.** 배포 클램프는 별도 스크립트가 만든다. 데이터셋이 바뀌면 **반드시 다시 돌려** yaml을 갱신할 것:
> ```bash
> $PY scripts/dataset/make_carrot_eef_limits.py      # -> ros2_ur_ws/src/gello_policy/config/carrot_eef_limits.json
> ```
> 이 JSON의 `generated_at` / `git_commit`이 yaml 주석에 적혀 있다. **둘이 어긋나면 yaml이 낡은 것이다.**

## 4. kanu에서 학습

학습 번들은 [`scripts/training/carrot_eef/`](../../scripts/training/carrot_eef/)에 있고, 절차·환경변수 정본은 그 안의 **`README_KO.md`** 다. 여기서는 순서와 **지켜야 할 제약**만 적는다.

> 📌 **2026-09-15 현재 이 절은 이미 끝나 있다.** 번들·데이터셋은 kanu에 동기화돼 있고
> (`./sync_to_kanu.sh --dry-run`이 전송 0바이트를 보고한다), ACT는 50,000 step 학습이
> 끝났으며 FM만 돌고 있다. **다시 sync/학습할 필요가 없으면 §5로 간다.** 아래는 재현·
> 재학습용 절차다. ⚠️ `launch_detached.sh`는 `OUTPUT_DIR`이 이미 있으면 거부하므로,
> 재학습하려면 기존 `outputs/<job>/`를 치우거나 `OUTPUT_DIR`을 새로 줘야 한다.

```bash
cd /home/laptop3/gello_software_humble/scripts/training/carrot_eef

./sync_to_kanu.sh --dry-run     # 먼저 dry-run으로 무엇이 전송되는지 본다
./sync_to_kanu.sh               # 실제 전송 (kanu:~/workspace/youngwoong/carrot_eef/ 밖으로는 절대 쓰지 않는다)

# --- 여기부터 kanu ---
ssh kanu
cd ~/workspace/youngwoong/carrot_eef/training     # ⚠️ 번들은 여기로 간다 (scripts/training/... 아님)
nvidia-smi                       # 지금 비어 있는 GPU를 고른다. 기본값 없음 — GPU_ID는 필수
GPU_ID=<n> ./preflight.sh act    # GPU 유휴 / 데이터셋 feature·tail-drop 계약 / 디스크 >= 20 GB / lerobot-train 동작
DRY_RUN=1 GPU_ID=<n> ./train_act_carrot_eef.sh    # 명령만 출력, 아무것도 실행하지 않음
```

> 🪤 **`GPU_ID` 없이 `./preflight.sh`만 치면 GPU 게이트가 통째로 SKIP된다**
> (`SKIP [1] GPU_ID unset -- schema/venv checks only, no GPU reservation`).
> 스키마·venv만 보고 "PASS"가 찍히므로, 남이 쓰는 GPU를 그대로 잡을 수 있다.
> 잡 인자(`act`|`fm`)를 빼면 `both`로 두 정책 클래스를 다 확인한다 — 그건 무해하다.

🛑 **반드시 지킬 것:**

- **GPU는 최대 2개.** kanu는 8× A4000을 여러 사람이 공유한다. `GPU_ID`는 **기본값이 없고** 안 주면 거부한다 — GPU 0을 자동으로 잡지 않는다.
- **디스크.** kanu는 98% 차 있다(여유 ~51 GB). `~/workspace/youngwoong/carrot_eef/` **밖에는 아무것도 쓰지 말 것.**
- **venv를 새로 만들지 말 것.** `LEROBOT_VENV`가 가리키는 기존 cube 환경(torch 2.11 cu128 + lerobot 0.6.1 + CLIP 캐시)을 재사용한다.
- **학습 시작은 사람이 한다.** 에이전트/스크립트가 자동으로 시작하지 않는다. `launch_detached.sh`로 띄우고 `runtime/` 아래 pid·로그를 확인한다.
- held-out은 **마지막 6 에피소드**(`--dataset.eval_split=0.111`)이며, `eval_offline_eef.py`가 같은 6개를 본다 — 학습 중 loss와 오프라인 open-loop 지표가 **같은 미학습 데이터**를 본다.

학습이 끝나면 **먼저 오프라인 open-loop 평가로 체크포인트를 고른다**(position MAE m / rotation MAE rad / gripper accuracy):

```bash
# 잡 인자는 필수다(없으면 usage로 거부). 학습 중인 GPU가 아닌 다른 유휴 GPU를 쓴다.
GPU_ID=<c> ./eval_checkpoints.sh act
GPU_ID=<c> ./eval_checkpoints.sh fm
DEVICE=cpu ./eval_checkpoints.sh fm        # GPU가 없을 때 (느림)
```

결과는 `evaluations/<job>/summary.csv`(+ `<step>/{metrics.json,episodes.csv}`)에 쌓이고,
이미 `metrics.json`이 있는 step은 건너뛴다.

> ⚠️ **이 지표는 open-loop다** — 매 프레임 정답 관측을 다시 먹여 주므로 오차가 누적되지
> 않는다. **성공률이 아니라 배포 성능의 상한**에 가깝다. 체크포인트 *선택*에는 쓸 수 있어도
> "이 정책이 된다"의 근거는 아니다.

📌 **2026-09-15 현재 실측(kanu, ACT 5개 체크포인트, held-out 48–53 / 2,270 frame):**
`position_mae_m` 0.00085–0.00094 m, `rotation_mae_rad` 0.0048–0.0051, `gripper_accuracy`
0.919–0.923으로 **10k~50k가 사실상 평평하다**(학습 중 eval_loss도 0.61–0.66에서 평평했다).
즉 이 지표만으로는 체크포인트를 고를 근거가 약하다 — 실기에서 비교하기 전까지는
**마지막 것(`050000`)을 쓰는 것이 기본**이고, 그 선택에 특별한 증거는 없다.

## 5. 체크포인트 가져오기

고른 체크포인트를 로봇 PC의 `ros2_ur_ws/src/gello_policy/checkpoints/<이름>` 아래로 가져온다(`--checkpoint`에 넘길 것은 `config.json` + `model.safetensors` + pre/post-processor json이 있는 **그 디렉터리 자체**다).

```bash
cd /home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy
# kanu에서 직접 (`~`는 따옴표 없이 `kanu:` 뒤에 두면 **원격**에서 풀린다 — 로컬 변수에
# `CKPT=~/workspace/...`처럼 담으면 /home/laptop3/... 로 풀려 실패한다):
rsync -av --partial \
    kanu:~/workspace/youngwoong/carrot_eef/outputs/act_carrot_eef/checkpoints/050000/pretrained_model/ \
    checkpoints/carrot_eef_act_50k/
# FM은 (학습이 끝난 뒤):
# rsync -av --partial \
#     kanu:~/workspace/youngwoong/carrot_eef/outputs/fm_carrot_eef/checkpoints/100000/pretrained_model/ \
#     checkpoints/carrot_eef_fm_100k/
ls checkpoints/carrot_eef_act_50k   # config.json model.safetensors policy_pre/postprocessor.json
```

> 📛 **디렉터리 이름은 자유다** — 서버에 넘기는 것은 경로이고, `expected_model_id`는 ZMQ
> 경로에서 강제되지 않는다. 다만 학습 번들
> ([`README_KO.md`](../../scripts/training/carrot_eef/README_KO.md) §6)과 이 문서가
> **`carrot_eef_act_50k` / `carrot_eef_fm_100k`** 로 통일돼 있으니 그대로 쓰는 편이 낫다.
> 아래 §6·§7 명령은 전부 이 이름을 쓴다.

> ⚠️ `checkpoints/`는 메인 checkout(`/home/laptop3/gello_software/...`)으로 가는 **심링크**다.
> 경로를 바꾸지 말고, 그 안에 있는 기존 banana 체크포인트
> (`act_banana_in_pot` / `diffusion_banana_in_pot_joint` / `flow_matching_banana_in_pot_joint`)를
> **덮어쓰지 말 것.** 새 디렉터리를 추가하는 것뿐이다.

## 6. 서버 단독 smoke + 레이턴시 확인 (팔 구동 전 필수)

**팔에 붙이기 전에 서버 half만 먼저 띄운다.** 여기서 확인할 것은 두 가지다: (a) 핸드셰이크가 `action_space`/`state_dim`/`action_dim`을 제대로 보고하는가, (b) 추론 1회가 리더의 `act_timeout_s` 예산 안에 드는가.

```bash
cd /home/laptop3/gello_software_humble/ros2_ur_ws

# ACT (포트 5591)
ACT_CHECKPOINT=$PWD/src/gello_policy/checkpoints/carrot_eef_act_50k \
ACT_ACTION_SPACE=eef_delta_v1 ACT_DEVICE=cuda \
    ./src/gello_policy/scripts/run_act_server.sh

# FM (포트 5593). task 문자열은 학습 문구 그대로.
# (FM_TASK는 생략해도 된다 — fm_server.py가 action_space=eef_delta_v1이면 "Put carrot in pot"을
#  기본값으로 고른다. 명시하면 그 기본값을 덮어쓸 뿐이다. FM_NUM_INTEGRATION_STEPS도 셸
#  런처 기본값이 이미 10이다.)
# 🪤 프롬프트는 **yaml에 없다.** policy_leader_node가 `task`를 declare하지 않으므로
#    carrot_eef_fm.yaml에 `task:`를 적어도 rclpy가 저장만 하고 아무도 읽지 않는다
#    (2026-09-15 R2/F7로 그 줄을 주석으로 내렸다). 프롬프트를 바꾸려면 **여기 FM_TASK**를
#    바꾼다. 텍스트 조건부 모델이라 프롬프트가 틀려도 에러 없이 성능만 떨어진다.
FM_CHECKPOINT=$PWD/src/gello_policy/checkpoints/carrot_eef_fm_100k \
FM_ACTION_SPACE=eef_delta_v1 FM_TASK="Put carrot in pot" \
FM_DEVICE=cuda FM_NUM_INTEGRATION_STEPS=10 \
    ./src/gello_policy/scripts/run_fm_server.sh
```

> 🛑 **smoke가 끝나면 이 서버를 반드시 Ctrl-C로 내려라 — 특히 ACT.** §7의
> `run_ur7e_act_real.sh`는 **자기 서버를 새로 띄우는데**, 5591이 이미 점유돼 있으면 새 서버는
> bind에 실패해 죽고 런처의 헬스체크는 **살아 있는 옛 서버의 LISTEN을 보고 통과한다** →
> 리더가 **§6에서 띄운 stale 서버**에 물린 채 팔을 몬다. FM 런처에는 이 사고를 막는 포트 점유
> 거부가 들어 있지만 **ACT 런처에는 없다**(실측: `run_ur7e_fm_real.sh`에만 `ss -ltn` 가드가
> 있다). 확인: `ss -ltnp | grep -E ':(5591|5593)'`.

확인 항목:

- 기동 로그에 **`state_dim=16`, `action_dim=7`, `action_space=eef_delta_v1`** 이 찍히는가. 서버는 이것이 체크포인트와 모순되면(예: state_dim 7인데 `--action-space eef_delta_v1`) **거부하고 죽는다.** 그게 정상 동작이다.
- **`warmup done in Xs`** 가 뜨는가.
- FM이라면 이후 **`net refill #N (act call M, horizon=24)`** 줄 **사이의 벽시계 간격**이 `act_timeout_s`(0.6 s) 안인가. ⚠️ 그 줄에는 소요 시간이 **찍히지 않는다** — 연속한 두 줄의 도착 시각을 직접 재야 한다(예: `… | ts -s '%.S'`, 또는 `ts`가 없으면 `awk '{print systime(), $0}'`). 넘으면 **워치독을 넓히지 말고** `FM_NUM_INTEGRATION_STEPS`를 줄인다(예: 5). 근거는 [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md) §6.
- 🪤 **ACT 서버에는 per-tick 레이턴시 로그가 없다.** `act_server.py`가 찍는 것은
  `loading checkpoint` / `action_space=… state_dim=… action_dim=…` / `warmup done in Xs` /
  `ready in Xs` / `REP bound at …` 뿐이다. 즉 ACT 쪽에서 §6이 실제로 증명하는 것은
  **핸드셰이크와 기동뿐**이고, 추론 지연은 **EXECUTE에 들어가서 리더가 FAULT를 내는지로만**
  드러난다(`act_timeout_s` 0.5 s 초과 → FAULT → 팔 정지). 첫 구동은 그것을 감수하고
  **E-STOP에 손을 둔 채** 시작할 것. FM 전용/ACT 전용 벤치마크 스크립트는 아직 없다
  (`scripts/benchmark_diffusion_latency.py`는 `DiffusionPolicy`를 로드하므로 쓸 수 없다).

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
ACT_CHECKPOINT=$PWD/src/gello_policy/checkpoints/carrot_eef_act_50k \
    ./run_ur7e_act_real.sh
```

### Flow-Matching

```bash
cd /home/laptop3/gello_software_humble/ros2_ur_ws
HEADLESS=true \
FM_PARAMS_FILE=$PWD/src/gello_policy/config/carrot_eef_fm.yaml \
FM_ACTION_SPACE=eef_delta_v1 \
FM_TASK="Put carrot in pot" \
FM_CHECKPOINT=$PWD/src/gello_policy/checkpoints/carrot_eef_fm_100k \
    ./run_ur7e_fm_real.sh
```

> **`HEADLESS=true`는 Method B** — 펜던트 Play 없이 드라이버가 URScript를 직접 보낸다. **로봇이 REMOTE 모드**여야 한다. 펜던트에서 External Control을 직접 Play하려면 `HEADLESS`를 빼면 된다(Method A).

> 📍 **무장한 자리가 곧 궤적이 재생되는 자리다.** 액션은 **상대 델타**이고 노드는 무장 시점의
> `TCP(live_q)`를 앵커로 잡아 거기서부터 누적한다. 그래서 팔이 어디에 서 있든 **같은 모양의
> 궤적**이 나오지만 **위치가 통째로 평행이동한다.** A8 실측: yaml `start_pose`에서 무장한 경우와
> 에피소드 자신의 frame-0 자세에서 무장한 경우 사이에 **55 mm의 강체 평행이동**이 생겼고(형상
> 오차는 0.033 µm), 즉 궤적은 완벽히 같고 통째로 55 mm 옆에서 재생됐다. **당근과 냄비가 그만큼
> 어긋나면 태스크는 실패한다.** `~/start_execution`의 0.1 rad 게이트는 "`start_pose` 근처"만
> 보장하지 그 안에서의 55 mm급 차이를 막지 않는다 — **핸드셰이크가 완전히 수렴한 뒤에 무장할 것.**

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

   이 호출이 하는 일: (a) **fresh 관측 셋**(joint_states · grip_pos · cam1 · cam2)이 전부 있고 `obs_timeout_s` 안인지 확인, (b) live 자세가 `start_pose`에서 **0.1 rad 이내**인지 확인(관절 wrap을 고려한 주기적 비교), (c) 서버에 RESET을 보내고 **`action_space`/`state_dim`/`action_dim` 계약을 검사**(§1.4 — 틀리면 여기서 거부), (d) EEF executor의 anchor/target을 **현재 live 자세로 리셋**, (e) EXECUTE 진입.
   `start_branch = branch_id(start_pose)`는 `start_pose`만의 함수라 **노드 기동 때 한 번** 계산되고 프로세스 수명 동안 고정이다(arming마다 다시 계산되지 않는다 — 결과는 같다).

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

🛑 **(1)과 (2)는 norm 비례 축소이고 축별 clip이 아니다.** 축별로 자르면 명령된 **방향이
꺾인다**(`[2.0, 0.5]` → `[1.0, 0.5]`). 회전이 개입하면 더 나빠진다.
⚠️ **(3) 워크스페이스 박스만은 축별 clip이다** — `clamp_position_box`는 `min/max`의 축별
클립이고(`eef_space.py`), **박스 제약은 원리상 norm 비례로 표현할 수 없다.** 그래서 박스에
닿으면 타깃이 **벽을 따라 미끄러지고 방향이 꺾인다.** 이것은 결함이 아니라 박스의 성질이며,
그래서 박스는 "정상 동작 중 닿는 한계"가 아니라 **마지막 OOD 차단선**으로 잡혀 있다
(관측 TCP 범위 ±0.03 m). 박스 clip이 상시 발동하면 그건 클램프 튜닝 문제가 아니라
**정책이 분포 밖으로 나가고 있다**는 신호다.
다섯 클램프 전부 한 줄짜리 throttled WARN(1 s)으로 합쳐 찍힌다:
`EEF CLAMP engaged: eef_step_m (…), eef_lead_m (…), workspace_box` 형태.

🛑 **(4) 관절 envelope가 EEF 모드에서 물리면 그건 "클립"이 아니라 제어 상실이다 (2026-09-15, R2/F1).**
joint 배포에서는 (4)가 **관절 타깃**을 자르므로 결과가 여전히 유효한 관절 자세다. EEF 모드에서는
**IK 해**를 자르는데, **축별로 잘린 IK 해는 더 이상 IK 해가 아니다** — 실행된 TCP가 방금 (1)~(3)이
보증한 워크스페이스 박스 **밖으로 나간다.** 박스는 애초에 **target 자세**만 구속했지 브리지로
내려가는 관절 벡터를 구속한 적이 없기 때문이다. 오프라인 실측(R2, +x OOD 주행, 완전추종 팔):
**명령된 target z는 상수인데 툴이 168 mm 아래로 미끄러져** 테이블 쪽으로 내려갔고 최종 오차가
96 mm였다. 그때 FAULT는 나지 않았다.
**그래서 이제 EEF 모드에서 envelope clip이 발생한 틱은 live 자세를 HOLD하고**(팔이 그 자리에
선다) **연속 `eef_ik_fail_fault_n`(10)회면 FAULT**다:
`FAULT: EEF target hit the joint envelope on 10 consecutive ticks …`.
깨끗한 틱 하나, `~/hold`, 재무장이 streak을 0으로 되돌린다. **joint 배포는 무변경이다** —
거기서는 잘린 타깃이 그대로 발행된다.
틱마다의 WARN: `EEF joint-envelope clip (streak N/10); holding the live pose this tick.`

ℹ️ **IK 브랜치/관절한계 이탈은 WARN만 한다 (R2/F5).** `ik_tcp_nearest`는 고정 브랜치에 해가 없으면
**전체 브랜치 열거로 폴백**하고, 관절한계 필터에 살아남는 후보가 하나도 없으면 **한계 밖 해를
그대로 돌려준다.** 두 사실이 이제 노드까지 올라와 throttled WARN으로 찍힌다
(`EEF IK left the pinned branch 2->3 (in_limits=False)`). **hold도 FAULT도 하지 않는다** —
박스 안 729점 격자에서 off-branch 0 / out-of-limit 0이라 실측상 도달 불가능한 경로이고, 실제로
일어나면 바로 위 (4) 경로가 잡는다. 이 WARN이 뜨면 **실기를 세우고 원인을 보라.**

ℹ️ **`chunk_start` 이상도 WARN만 한다 (R2/F8).** 200틱(≈6.7 s) 동안 `chunk_start`가 한 번도 안 오면
(= 앵커 하나로 무한 누적, lead 클램프만이 유일한 상한), 또는 **60틱 연속으로 계속 true면**
(= 매 틱 뒤처진 live 자세에 재앵커 → 모든 스텝이 줄어듦, 실측 추종오차 p50 169 mm vs 정상 15 mm)
각각 throttled WARN이 뜬다. **FAULT는 아니다** — 안전 문제가 아니라 태스크 실패다.

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
- **워치독 순서**: ACT `act_timeout_s 0.5` / `obs_timeout_s 0.5` / **staleness `0.8`**, FM `0.6 < 0.7 < 0.8`.
  📌 **2026-09-15(R2/F6): carrot ACT의 브리지 staleness를 0.5 → 0.8로 올렸다.** 노드는 단일
  스레드라 ZMQ 왕복 동안 `/gello/joint_states` 발행이 통째로 멈춘다. `act_timeout_s`와
  staleness가 **둘 다 0.5면 여유가 0**이라, 0.45초짜리 응답은 노드를 FAULT시키지 않으면서
  브리지 워치독은 이미 넘겨 버린다 → 재시드 + 0.7 s soft-start 램프, 즉 **노드 쪽 로그가 하나도
  없는 눈에 보이는 stutter**가 나고 그 동안 EEF target은 계속 누적된다. 0.8은 FM 파일과 같은
  여유이며 순서(`act_timeout_s` 0.5 < staleness 0.8)는 그대로다 — **실제 장애에서는 여전히
  리더가 먼저 FAULT하고, 팔을 세우는 것은 여전히 브리지 워치독이다.** banana joint
  (`act_deploy.yaml`)는 **건드리지 않았다**(0.5 그대로). FM 쪽 순서 불변식의 근거는 [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md) §6 — **넓히지 말고 Euler 스텝을 줄일 것.**
- **그리퍼 단일 writer**: `gello_gripper_bridge`는 띄우지 않는다. `policy_leader_node`가 `/robotiq_gripper/command_percent`의 유일한 publisher이며, `action[6]`을 0..1로 clip해 발행한다.
- **E-STOP**: 어떤 소프트웨어 클램프도 물리 안전을 대체하지 않는다. 펜던트 E-STOP을 항상 손 닿는 곳에. 그리퍼는 연결 시 auto-calibrate(open/close sweep)하므로 **손가락을 치울 것.**

## 9. 실기 미검증 목록 (정직하게)

> 🔴 **이 브랜치의 어떤 것도 로봇을 움직인 적이 없다.** 아래는 "아직 안 한 것"이지 "아마 될 것"이 아니다.

**아직 존재하지 않는 것 (2026-09-15 기준):**

- **FM 체크포인트(완주본)** — FM은 아직 학습 중이다(~27k / 100,000). 중간 체크포인트
  `010000`/`020000`만 kanu에 있다.
- **HF 리포** — `expected_model_id`의 `Bigenlight/act_carrot_in_pot_eef` /
  `Bigenlight/flow_matching_carrot_in_pot_eef`는 **placeholder 이름**이고 그 리포는 존재하지
  않는다. 체크포인트는 kanu에서 rsync로 가져온다(§5). **ZMQ 경로는 이 필드를 강제하지 않는다**
  — 실제 게이트는 §1.4의 핸드셰이크(`action_space`/16/7)뿐이다.
- **로봇 PC의 체크포인트 사본** — `ros2_ur_ws/src/gello_policy/checkpoints/` 아래에는 아직
  banana 3종만 있다.
- **closed-loop 성공률** — 어떤 정책도 rollout으로 평가된 적이 없다.

**존재하지만, 근거가 약한 것:**

- **체크포인트 선택 근거** — ACT 오프라인 open-loop 평가는 끝났지만 10k~50k가 사실상 평평하다
  (§4의 실측 표). 즉 "50k가 제일 낫다"는 증거가 없고 그냥 마지막 것이다. 그리고 open-loop
  지표는 오차가 누적되지 않으므로 **성공률의 상한**에 가깝다.

**존재하지만 실기에서 확인 안 된 것:**

- **팔 구동** — `action_space: eef_delta` 경로로 실물 UR7e를 움직인 적이 **0회**. 이 브랜치의
  **어떤 커밋도 로봇을 움직인 적이 없다.**
- **핸드셰이크 거부** — 계약 검사가 실제 서버 응답에 대해 동작하는지 실기 미확인. 단위
  테스트와 **서버 측 거부**(banana 체크포인트 + `--action-space eef_delta_v1` → `exit 4`,
  CPU에서 실측)까지만 확인됐다. 노드 쪽 거부(`~/start_execution`)는 코드 리뷰 + 단위 테스트뿐.
- **chunk 누적 / re-anchor** — 실제 정책의 `chunk_start` 타이밍에 대해 미확인.
- **카르테시안 클램프 3종** — 실기에서 한 번도 발동한 적이 없다. 특히 **lead 클램프**는 실제 팔의 지연 특성과 만나 본 적이 없다.
- **IK 브랜치 고정** — 데이터에서는 18,557 프레임 전부 branch 2였지만, **정책이 만들어 낸 자세**에서 브랜치가 유지되는지는 다른 문제다.
- **추론 레이턴시** — 이 로봇 PC의 GPU에서 EEF 체크포인트의 refill p99가 예산 안인지 미측정
  (§6). **ACT는 per-tick 로그 자체가 없어** EXECUTE에 들어가야만 드러난다.
- **그리퍼** — `action[6]`(0..1)이 실제 2F-85를 EEF 모드에서 여닫는 것을 본 적이 없다.
  `policy_leader_node`가 `/robotiq_gripper/command_percent`의 유일한 publisher라는 것은
  joint 배포에서 검증됐지만, EEF 체크포인트가 내는 값으로는 아직이다.
- **로봇 PC colcon build** — 워크트리에서의 빌드는 통과했으나(2026-09-15 재확인) 로봇 PC 실환경 빌드는 별도.
- **정책이 학습된 `eef_space.py`와 배포의 `eef_space.py`가 같은지** — 데이터셋은
  `meta/eef_space.json`에 `eef_space_sha256`을 담고 있지만 **배포 경로는 그것을 검사하지 않는다.**
  16/7 차원만 맞으면 통과한다.

**해결됨(이전 판에서는 "미해결"이었다):**

- **0.9 s 지연이 배포에도 나타나는가 → 나타나지 않는다.** 원인이 **레코더의 구독 큐**이고
  배포 루프에는 그 레코더가 없다([`CARROT_EEF_LAG_NOTE_KO.md`](./CARROT_EEF_LAG_NOTE_KO.md) §5).
  배포에 실재하는 지연은 **리더→명령 ~0.25 s**(One-Euro 1 Hz ≈ 0.16 s + 슬루 + 30 Hz 샘플링;
  7월 joint 세션 0.253 s와 9월 EEF 세션 0.260 s가 같아 아티팩트가 아님이 확인됐다)와
  **슬루 캡 0.625 rad/s** 둘뿐이다.
  ⚠️ **다만 그 0.25 s가 lead 클램프의 여유를 거의 다 먹는다**: 정상 상태에서
  `|p_target − p_live| ≈ v × 0.25 s`이므로 데이터셋 p50 속도(0.073 m/s)에서 **18 mm**,
  거버너 천장(0.16 m/s)에서 **40 mm**다. `eef_max_lead_m = 0.07`에 대해 **여유가 1.75배뿐**이다
  (그 노트는 0.06 기준으로 1.5배라고 적었다). 빠른 이송 구간에서 lead 클램프가 상시 물릴 수
  있고, 그러면 정책이 스스로 감속하는 것처럼 보인다 — §10의 해당 항목을 먼저 볼 것.
  **재발 조건은 남아 있다**: `/joint_states`를 깊은 KEEP_LAST 큐로 구독하면서 콜백을
  100 Hz 미만으로 처리하는 노드는 무엇이든 같은 무증상 지연을 얻는다.

**첫 실기 세션 순서 (제안).** ✅ = 2026-09-15 기준 이미 끝난 것.

1. ✅ 데이터셋 변환 + 독립 검증 (§3, `verdict: PASS`)
2. ✅ kanu 학습 — ACT 50k 완료 / FM 진행 중 (§4)
3. ✅ ACT 오프라인 open-loop eval (§4) — 단 10k~50k가 평평해서 선택 근거는 약하다
4. **로봇 PC colcon build** (§2)
5. **체크포인트 fetch** (§5)
6. **서버 단독 smoke** (§6) — 핸드셰이크 3개 필드 확인, 끝나면 **서버를 반드시 내린다**
7. 카메라/그리퍼 라이브 확인 (`ros2 topic hz`)
8. 런처를 띄워 **핸드셰이크까지만** — 팔이 `start_pose`에 파킹하는 것을 확인하고 멈춘다
9. **감독 하에** `~/start_execution`, E-STOP에 손을 둔 채. 첫 몇 틱은 `EEF CLAMP engaged` /
   `EEF IK failed` / `EEF joint-envelope clip` WARN을 보면서 진행한다.
10. 📏 **`lead_rot_norm`을 기록할 것 (R2/F4 — 여유가 가장 적은 값이다).** 클로즈드루프 시뮬
   실측에서 **위치** lead는 캡의 30 %까지만 갔지만 **회전** lead는 τ=0.04 s에서 63 %,
   τ=0.08 s에서 87 %까지 올라갔다. 즉 실제로 먼저 물리는 것은 `eef_max_lead_m`(0.07)이 아니라
   **`eef_max_lead_rad`(0.10)**이고, 증상이 미묘하다 — 스쿱 동작에서 툴이 **끝까지 안 돌고**
   throttled `EEF CLAMP engaged: eef_lead_rad` WARN만 뜬다. **p99 > 0.08이면 `eef_max_lead_rad`를
   0.15로 올린다** (설계 §3 기본값 0.30의 절반이고, 실측 30-frame p99 0.379보다 여전히 훨씬
   아래다). 값은 executor `info` dict의 `lead_rot_norm` / `lead_pos_norm` / `clamped_*` /
   `ik_on_branch` / `ik_in_limits`와 관절 envelope clip 발생률을 같이 남긴다.

## 10. 트러블슈팅

- **`~/start_execution`이 계속 거부된다.** 응답 문자열이 어느 게이트인지 그대로 말해 준다(§7의 (a)~(c) 순서와 같다). (a) `refused: no live /joint_states yet` — 드라이버/컨트롤러가 아직 안 올라옴. (b) `refused: observations missing/stale [...]` — fresh 관측 셋(카메라 2대 + 그리퍼 위치 + joint_states)이 불완전하거나 `obs_timeout_s`를 넘었다. `ros2 topic hz`로 확인. (c) `refused: live pose not within 0.1 rad of start_pose (worst joint j: …)` — 핸드셰이크 수렴 로그 확인. (d) **핸드셰이크 계약 불일치** — 로그에 서버가 보고한 `action_space`/`state_dim`/`action_dim`이 찍힌다. 대개 `*_ACTION_SPACE`를 안 준 것이다(§7 🪤).
- **서버가 기동하자마자 죽는다 (action space 모순).** `--action-space eef_delta_v1`인데 체크포인트 `state_dim`이 7이거나 그 반대다. **의도된 거부다** — 체크포인트를 고치지 말고 어느 쪽이 틀렸는지 확인할 것.
- **팔이 데이터 분포보다 느리게/조금씩만 움직인다.** lead 클램프가 상시 발동 중일 가능성. throttled WARN 빈도를 본다. 원인이 브리지 slew라면 **`eef_max_lead_m`을 올리는 게 아니라** 왜 팔이 못 따라가는지를 먼저 본다(§8 🛑).
- **IK 실패 WARN이 계속 뜬다.** 팔이 워크스페이스 가장자리이거나 특이점 근처, 또는 브랜치가 튄 상황이다. 10회 연속이면 FAULT로 간다. 그 전에 `~/hold`로 세우고 §7 리커버리를 따르는 편이 낫다.
- **yaml을 고쳤는데 반영이 안 된다.** `--symlink-install`로 빌드했으면 share 복사본이 곧 src 파일이지만, 일반 `colcon build`면 **스냅샷**이다. 다시 빌드하거나 `*_PARAMS_FILE`을 src 경로로 직접 지정할 것. 배너에 실제 사용 경로가 출력된다.
- **FM: refill이 느려 FAULT-loop.** `FM_NUM_INTEGRATION_STEPS`를 줄인다(예: 5). **워치독을 넓히지 말 것** — 자세한 근거는 [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md) §9.
- **카메라가 조용히 안 뜬다 (노드는 살아 있는데 프레임 0).** 십중팔구 **ASIC 시리얼**로 바인딩한 것이다. `ros2 topic info`는 publisher 1을 보고하므로 `ros2 topic hz`로만 구분된다. `launch_cameras.sh`를 쓸 것.
- **포트 충돌.** ACT 5591 / Diffusion 5592 / FM 5593. FM 런처는 시작 전에 5593 점유를 **거부**한다(stale 서버가 팔을 몰지 못하도록). `pkill -f fm_server.py ; ss -ltnp | grep :5593`.
- ✅ **ACT 런처에도 그 포트 가드가 생겼다 (2026-09-15, R2/F2).** 예전에는
  `run_ur7e_act_real.sh`가 5591이 점유돼 있어도 그냥 서버를 띄웠고, 새 서버는 bind 실패로
  죽지만 헬스체크 루프는 **stale 서버의 LISTEN을 보고 통과**해서 리더가 옛 서버에 물렸다.
  §6의 smoke 서버를 내리지 않은 채 §7로 넘어가면 정확히 이 일이 생겼다. 이제 세 런처
  (ACT/FM/diffusion) 전부 **점유된 포트에서 시작을 거부**한다:
  `ERROR: port 5591 is already in use (a stale ACT/FM/diffusion server?)`.
  🪤 **핸드셰이크는 이걸 못 잡는다** — `action_space`/`state_dim`/`action_dim`만 확인하고
  **어느 체크포인트인지는 보지 않으므로** stale EEF 서버는 세 필드를 완벽히 통과한다.
  그래서 같은 수정에서 **arming 로그에 서버가 보고한 `checkpoint=`를 찍게 했다** —
  "무엇이 팔을 몰았는가"를 사후에 답할 수 있는 유일한 기록이다. 그래도 시작 전에
  `ss -ltnp | grep -E ':(5591|5593)'`로 비어 있는지 눈으로 확인하는 습관은 유지할 것.

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
