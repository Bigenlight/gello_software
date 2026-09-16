# GELLO → UR7e IFQL Policy 실기 배포 — 런북 (REAL, carrot-in-pot)

> ✅ **첫 실물 세션 2026-09-16(laptop3, RTX 3060).** `launch_cameras.sh` → `run_ur7e_ifql_real.sh`
> (bc) → move-to-start → START/HOLD/GO TO START 전 경로가 실제 UR7e에서 돌았다: 서버 GPU refill
> p50 18 ms, 30 Hz 액션 발행, 클램프 WARN 0, GO TO START 0.345 rad 복귀 후 START 재수락.
> **성공률(bc vs bon32 paired 표)은 아직 없다** — §5 프로토콜이 다음 단계다.
> 그날 한 번 로봇이 POWER_OFF로 떨어져(원인 미확인; 드라이버는 끄지 않는다) T2 재기동으로
> 복구했다 — 종료 시 `ur_ros2_control_node`가 futex에 걸리면 TERM/KILL로 정리해야 `ros2 launch`가
> 빠져나온다.

## 요약 (TL;DR)

| 항목 | 값 |
|---|---|
| 태스크 | **carrot-in-pot** — 당근을 집어 냄비에 넣고 그리퍼 open 상태로 **1 s dwell** |
| 모델 | **IFQL** = flow-matching BC actor + IQL critic. 배포 시 **best-of-N**(BoN, K=32)으로 critic이 후보를 고름. `--sampler bc`(K=1)면 **같은 체크포인트가 순수 BC**가 된다 → bc vs bon paired 비교의 근거 |
| 관측 | frozen **ResNet18 spatial-softmax** feature cam1(1024) + cam2(1024) + proprio 7 = **2055-D**. 이미지는 1280×720 JPEG → 224×224 INTER_AREA squash(crop 없음) |
| 액션 | 56-D 청크 = **8 knot × 7**(q1..q6 절대 관절 rad + grip 0..1, 10 Hz stride-3) → 서버가 **24-frame 30 Hz 큐**로 펼침 → ACT마다 1 frame pop, 큐가 비면 재계획(**≈0.8 s**마다 refill). 와이어의 7-D per-tick 계약은 ACT/Diffusion/FM과 **동일** |
| 체크포인트 | HF **`Bigenlight/carrot-in-pot-ifql`** → run `ifql_real_lead6_k0.9_s0`, step **100000** (family `real_lead6`: action_lead=6, κ=0.9, seed 0) |
| norm_stats | **`norm_stats_r18_ss_real_lead6.json`** — 런처가 서버 로그에서 basename에 `real_lead6`가 있는지 **강제 확인**, 아니면 즉시 종료 |
| 학습 데이터 | 실물 carrot 54 take / 18,557 frame (2026-09-14, HF `Bigenlight/carrot_in_pot_lerobot_v3`) — **54개 전부 성공 확인**(2026-09-15) |
| 두 프로세스 | IFQL 서버(`.venv-svf`, torch+JAX, `ifql_server.py`) ⇄ ZMQ **127.0.0.1:5595** ⇄ `policy_leader_node`(Humble py3.10). 런치는 정책-무관 `ur7e_diffusion_real.launch.py` 재사용 |
| 안전 | 브리지 slew **0.625 rad/s**(불변) + 리더의 1.2× envelope 클램프 + live 대비 0.5 rad 클램프. 타임아웃 **0.8 < 0.9 < 1.0**(FM 0.6/0.7/0.8에서 한 번 넓힘; 더 넓히지 말 것) |
| 포트 | **5595** 단독 (ACT 5591 / Diffusion 5592 / FM 5593 / SVF 5594) |
| 실기 상태 | **첫 실물 세션 2026-09-16** — bc 정책이 실제로 팔을 몰았다(성공률 표는 아직). GPU는 §7 |

---

## 1. 무엇을 배포하나

- **정책**: IFQL(Implicit Flow Q-Learning). actor는 BC flow-matching으로만 학습되고 critic은 IQL
  expectile로 학습된다. Q는 **배포 시점에만** BoN rejection sampling으로 개입한다 — 그래서 `bc`와
  `bon`은 **같은 가중치**를 두 방식으로 샘플링하는 것이며, 두 조건의 차이가 곧 critic의 기여다.
- **모델 카드 / 스펙**: `~/carrot_ifql/hf/cards/MODEL_CARD.md`, `SPEC_stage1_frozen_real_lead6.md`,
  `FIELD_GLOSSARY.md`. 체크포인트 인덱스는 `~/carrot_ifql/hf/models.json`(25 run, 3 family).
- **`lead6`가 무엇인가**: 실물 teleop 로그는 `action[i] ≈ state[i]`(lag 0)인데 sim command는 state보다
  ~6 frame(0.2 s) 앞선다. `lead6`은 joints 1–6의 knot을 `i+3k+6` frame에서 gather해 실물 청크의
  시간 컨벤션을 sim과 맞춘 재빌드다(그리퍼는 lead 0). **norm_stats는 lead6 전용 파일을 써야 하고
  lead0 파일과 action box가 같아 envelope 클램프로는 잡히지 않는다** — 런처의 로그 검사가 유일한
  방어선이다.

## 2. 어디에 있나 (`~/carrot_ifql/`, 리포 밖)

```
~/carrot_ifql/
  hf/                                 HF Bigenlight/carrot-in-pot-ifql 스냅샷
    models.json                         run 인덱스
    ifql_real_lead6_k0.9_s0/            flags.json · params_100000.pkl · norm_stats_r18_ss_real_lead6.json · serve_meta.json · RUN.md
    ifql_real_k0.9_s0/                  lead0 형제 — 쓰지 말 것 (norm_stats_r18_ss_real.json)
    cards/                              MODEL_CARD · SPEC_* · FIELD_GLOSSARY
  code/carrot_ifql_code_<date>/       코드 타르볼 (런처가 find로 ifql_server.py를 찾는다; D_c-null 패치 적용본)
    vision_carrot/ifql_server.py        ZMQ REP 서버 (+ action_queue · episode_logger · REALROBOT_RUNBOOK_IFQL.md)
    qflow_svf_merged/                   agent 코드 (Q/agents/ifql.py, utils.flax_utils.restore_agent)
  .venv-svf/                          uv py3.11, jax 0.6.2 cuda12 + torch 2.14+cu126 (7.5 GB)
  .cache/torch/                       TORCH_HOME — ResNet18 IMAGENET1K_V1 + torch.hub(dinov2) 오프라인 캐시
  eval_runs/real_<YYYYMMDD>/          런처가 만드는 서버 로그: ifql_lead6_<bon32|bc>/<HHMMSS>/
                                        ep_NNNN.npz · refill_stats.jsonl · server_stdout.log
```

**다른 PC에서 재현:** 위 트리 전체를 스크립트 하나가 만든다 —
[`ros2_ur_ws/ifql/setup_ifql_workspace.sh`](../../ros2_ur_ws/ifql/setup_ifql_workspace.sh)
(HF 선별 다운로드 → 타르볼 sha256·추출 → **D_c-null 패치 적용**(`ifql/ifql_server_Dc_null.patch`)
→ `uv venv` py3.11 + CUDA 휠(`ifql/requirements-svf-infer.txt`) → ResNet18·dinov2 hub 캐시 →
`--help` smoke). 전제: `uv`, `hf auth login`(private repo 읽기 권한), 인터넷, ~8 GB.
`IFQL_ROOT`(기본 `$HOME/carrot_ifql`)를 바꾸면 런처에도 같은 값을 준다.

이 리포 쪽 파일:

| 파일 | 역할 |
|---|---|
| [`ros2_ur_ws/src/gello_policy/config/ifql_deploy.yaml`](../../ros2_ur_ws/src/gello_policy/config/ifql_deploy.yaml) | `fm_deploy.yaml` 복제 + IFQL 값. **carrot start_pose / 1.2× envelope / 5595 / 0.8·0.9·1.0**. 숫자 출처는 파일 주석에 전부 있다 |
| [`ros2_ur_ws/run_ur7e_ifql_real.sh`](../../ros2_ur_ws/run_ur7e_ifql_real.sh) | 서버 기동 → norm_stats·sampler 로그 검사 → 포트 대기 → `ros2 launch`(FM 런처 구조 그대로) |
| [`ros2_ur_ws/ifql/`](../../ros2_ur_ws/ifql/) | `setup_ifql_workspace.sh` · `requirements-svf-infer.txt`(서빙 subset, cu126 핀) · `ifql_server_Dc_null.patch` |
| 이 문서 | |

> **빌드 없이 돈다.** `ifql_deploy.yaml`은 `install/`에 없다(이 PC의 `install/`은 데이터 수집이 쓰고
> 있어 `colcon build` 금지). 런처는 `params_file:=`에 **src 절대경로**를 넘긴다 — 런치 파일은
> 경로를 그대로 받으므로 설치본이 필요 없다. 런치 파일 자체(`ur7e_diffusion_real.launch.py`)와
> `policy_leader_node`는 이미 설치돼 있고 src와 동일함을 확인했다(2026-09-15 diff).

## 3. 절차

**T1 — 카메라 (먼저, 자기 터미널에서)**

```bash
cd ~/gello_software/ros2_ur_ws
./launch_cameras.sh          # cam1=SCENE(삼각대) 왼쪽, cam2=WRIST 오른쪽. 뷰어에 START / HOLD 버튼이 있다
ros2 topic hz /cam1/cam1/color/image_raw/compressed   # ~30 Hz (topic info의 publisher 수로 판정하지 말 것)
ros2 topic hz /cam2/cam2/color/image_raw/compressed
```

**T2 — 서버 + ROS (한 스크립트)**

```bash
cd ~/gello_software/ros2_ur_ws
HEADLESS=true ./run_ur7e_ifql_real.sh                 # bon, K=32 (기본). 로봇은 REMOTE 모드
IFQL_SAMPLER=bc HEADLESS=true ./run_ur7e_ifql_real.sh # bc 조건 (K=1)
IFQL_NUM_SAMPLES=16 HEADLESS=true ./run_ur7e_ifql_real.sh   # refill이 300 ms를 넘을 때
```

스크립트가 하는 일 순서: 포트 5595 점유 시 거부 → `.venv-svf` python으로 `ifql_server.py
--run-dir … --step 100000 --sampler … --num-samples … --port 5595 --budget-s 0.6 --norm-stats …
--log-dir …` 백그라운드 기동(`TORCH_HOME`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `HF_HUB_OFFLINE=1`)
→ 서버 로그의 `norm_stats: <path>` basename에 `real_lead6`, `sampler: kind=<요청값> K=<요청값>`
확인(아니면 서버를 죽이고 종료) → 포트 listen 대기(서버는 warmup **뒤에** bind하므로 listen =
warmup 통과; 최대 `IFQL_WARMUP_TIMEOUT_S`=120 s) → `ros2 launch gello_policy
ur7e_diffusion_real.launch.py robot_ip:=… start_mode:=gello params_file:=<src 절대경로>
act_host:=127.0.0.1 act_port:=5595 headless_mode:=…`. Ctrl-C 한 번에 서버와 런치를 함께 정리한다.

- `HEADLESS` 미설정이면 FM과 같은 Method A(펜던트 External Control **Play**). 이 조작자는 평소
  REMOTE 모드로 `HEADLESS=true`를 쓴다.
- 핸드셰이크가 팔을 **carrot start pose**(§4)로 보내 파킹한다. 그때까지 자율 모션은 없다.
- **START**: 뷰어의 START 버튼, 또는
  `ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger`.
  라이브 자세가 start_pose ±0.1 rad 안이고 4개 관측이 전부 fresh해야 받는다.
  **HOLD**: 뷰어 버튼 또는 `~/hold` — 그 자리에서 정지(그리퍼는 마지막 명령 유지).
- 샘플러 전환은 **T2를 Ctrl-C 하고 다른 `IFQL_SAMPLER`로 재실행**. 재실행 전 `ss -ltnp | grep :5595`가
  비어 있어야 한다(런처가 어차피 거부한다).

**T3 — 기록 (START 누르기 전에)**

```bash
cd ~/gello_software/ros2_ur_ws
./run_recorder.sh            # CAMS=true 금지
```

`gello_logs/session_<stamp>/vectors.h5`(synchronized / ur_joint_states / command / gripper / wrench /
tcp_pose) **+ cam1.mp4 / cam2.mp4**. **`CAMS=true`를 주면 안 되는 이유**: 그 플래그는 레코더가 자기
`rs_launch.py` 두 개를 **같은 시리얼로 한 번 더** 띄우게 하는 것뿐이라(`run_recorder.sh`의 `CAMS`
블록), T1이 이미 잡고 있는 USB 장치와 충돌한다. 레코더 노드는 `/cam1/cam1/color/image_raw/compressed`
를 **무조건** 구독하므로(`gello_ur_recorder_node.py` `cam1_topic` 기본값 + 구독) T1의 카메라로
mp4는 그대로 나온다.

서버 쪽 기록은 `--log-dir`가 자동으로 한다: `ep_<reset_counter:04d>.npz`(state · 두 JPEG · 실행
action · chunk id/step · refill ms, 에피소드당 ~90 MB), `refill_stats.jsonl`(refill마다 `encode_ms`,
`sample_ms`, `refill_ms`, `q_chosen`, `q_mean`, `q_std`, `q_spread`, `K`), 종료 시 stdout `STATS` 한 줄.
런처는 **기동마다 새 `<HHMMSS>/` 하위 디렉터리**를 만든다 — 서버가 매 기동 `ep_0001.npz`부터 다시
세고 `os.replace`로 덮어쓰기 때문에(`episode_logger.py` `path_for`), 디렉터리를 공유하면 이전 run의
에피소드가 사라진다.

## 4. 이 배포에서 바뀐 숫자 (전부 `ifql_deploy.yaml` 주석에 출처 있음)

| 파라미터 | 값 | 출처 |
|---|---|---|
| `start_pose` | `[-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]` | 실물 54 take **frame-0 pose 평균(de-staled, cam1_t[0]+0.89 s)**, `carrot_eef_limits.json` `start_pose`. **FM yaml의 banana 값(j0=+3.106)은 부호부터 다르다** — 남기면 move-to-start가 엉뚱한 자세로 간다 |
| `start_gripper` | `0.0` | 모든 take가 open에서 시작(측정 0.0118 = open 끝) |
| `joint_limits_lo/hi` | `[-3.8051,-1.8059,1.2622,-2.6921,-1.9311,-4.6439]` / `[-2.5783,-0.8822,2.1445,-1.4393,-1.1990,-1.8221]` | 관측 관절 범위의 1.2× envelope, HF `carrot_real_limits.json`(핸드오프). 로컬 `carrot_eef_limits.json` 파생값과 4째 자리(최대 0.0027 rad) 차이 — HF 파일 도착 후 재대조 |
| `act_port` | `5595` | |
| `act_timeout_s` / `obs_timeout_s` / `staleness_timeout_s` | **0.8 / 0.9 / 1.0** | FM 0.6/0.7/0.8에서 BoN K=32·CPU 폴백 대비 한 번 넓힘. **부등식 유지, 더 넓히지 말 것** — refill이 느리면 K를 줄인다 |
| `max_dev_rad` · `max_step_rad` · `auto_start_on_stream` | 0.5 · 0.0025 · false | 무변경 |

`start_mode`는 **`gello`만** 허용한다. 이 모드에서 `gello_move_to_start`는 `/gello/joint_states`를
쫓는데, HOLD 중인 리더가 거기 발행하는 것이 바로 yaml의 `policy_leader_node.start_pose`다 — 출처가
하나다. 런치 인자 `start_pose:=`는 **로그용**이라(리더 노드에 전달되지 않음) 런처가 넘기지 않는다.
`init_align`은 사람 GELLO 정렬 게이트라 synthetic 리더로는 통과할 수 없어 런처가 거부한다.

클램프 순서(`policy_leader_node._tick_execute`): (a) envelope clip → (b) live ±0.5 rad clip →
그리퍼 [0,1] → 브리지 0.625 rad/s. envelope은 서버의 denormalize box(`norm_stats` action lo/hi)를
감싸므로 **정상이라면 (a)는 거의 안 걸린다** — `SAFETY CLAMP engaged: joint_limits (OOD)` WARN이
계속 뜨면 sim norm_stats가 물린 것이다(lead0 실물 파일은 box가 같아 여기서 안 잡힌다).

## 5. 실험 프로토콜 (bc vs bon32, paired)

1. **배치 20개를 사전 고정**하고 번호·사진·테이프 마킹(당근 위치·각도, 냄비 위치).
2. 배치마다 **bc 1회 + bon32 1회**(같은 배치). **순서는 배치마다 교대**(A: bc→bon, B: bon→bc) —
   물체가 미세하게 밀리는 순서 효과 상쇄.
3. 매 trial: 팔은 런치의 move-to-start로 start_pose(재실행 또는 HOLD 후 재-arm) → 물체 배치 →
   T3 레코더 → START → **최대 20 s** 관찰 → 판정 → HOLD.
4. **성공** = 당근이 냄비 안 + 그리퍼 open 상태 **1 s dwell**(sim harness와 같은 기준). stopwatch로
   time-to-success.
5. **fault(타임아웃·obs stale·ok:false)가 난 trial은 폐기하고 같은 배치를 재시도**(sim 규칙과 동일). 왜
   났는지는 적는다.
6. 총 40 에피소드 → 2×2 표 → **McNemar**(불일치 칸 b, c만 사용). n=20 paired는 검출력이 약하다 —
   차이가 작으면 "차이 없음"이 아니라 **"이 n으로는 모름"**이라고 쓴다.
7. 부수 지표: 성공 trial의 평균 **time-to-success**, `refill_stats.jsonl`의 **refill p95**, **`q_spread`
   중앙값**(critic이 후보를 실제로 구분하는지), 클램프 WARN 개수.
8. trial 표 한 줄: `trial# | sampler | placement | outcome | t_success | 실패 모드 | reset_counter | refill p95 | clamp WARN`.

## 6. 해결된 blocker

- **start_pose 교체** — banana 값 대신 실물 carrot frame-0 pose(§4). 런북이 "실물 54 take로 다시
  뽑아서 박을 것"이라 했던 항목.
- **carrot envelope** — 런북의 "이 PC엔 `carrot_eef_limits.json`이 없다"는 해소됐다(로컬 파일 +
  HF 파생값 둘 다 있고, yaml은 HF 값).
- **54 take 전부 성공 확인**(2026-09-15) — 학습 데이터에 실패 시연이 섞이지 않았다.
- 런처: 포트 가드, norm_stats/sampler 로그 검사, per-launch 로그 디렉터리, `init_align` 거부.
- **서버 기동 즉사 2건 (2026-09-16 smoke에서 발견·해결)** —
  (a) `ifql_server.py:127` `NormStats.__init__`이 실물 norm_stats의 `"D_c": null`에서
  `int(None)` TypeError로 죽는다(핸드오프 `hf_release/README.md`·`serve_meta.json`
  `fallback_reason`에 "실물 전에 고칠 것"으로 기록돼 있던 알려진 결함). **로컬 타르볼 사본
  `~/carrot_ifql/code/carrot_ifql_code_20260915/vision_carrot/ifql_server.py`에 한 줄 패치**
  (`int(d.get("D_c") or (self.D_a + Z_PRIV_DIM))` → `D_c=2076`; `use_critic_obs=False`라 예제 배열
  크기 외 미사용), 원본은 `ifql_server.py.orig`. 타르볼을 다시 풀면 **패치가 사라진다.**
  (b) `QFLOW_DIR`이 없으면 서버가 학습 PC의 하드코딩 경로(`/home/theo_lab/...`)를 찾다가
  `ModuleNotFoundError: agents` — 런처가 `<snapshot>/qflow_svf_merged`를 자동 해석해 넘긴다
  (핸드오프의 `qflow_svf/`는 실제로 `qflow_svf_merged/`). `FMRL_CAM1_CROP/MODE`도 런처가 unset.

## 7. 남은 것 / 미검증

- **실기 0회.** 첫 세션 순서: `.venv-svf` + `TORCH_HOME` 캐시 확인 → 서버 단독(`--no-serve`
  또는 런처를 띄우고 warmup 로그만 보기) → 카메라 hz → 그리퍼 → 감독하에 START.
- ✅ **GPU (2026-09-16 해결)** — laptop3의 문제는 RT 커널이 아니라 `nvidia-dkms-595`가
  half-configured(`iF`)로 남아 어느 커널에도 모듈이 빌드되지 않은 것이었다. generic 커널로
  부팅한 뒤 `sudo dkms install nvidia/595.91.07 -k $(uname -r) && sudo dpkg --configure -a &&
  sudo modprobe nvidia`로 해결(재부팅 불필요). RTX 3060 실측 refill **p50 18 ms**(CPU 45 ms).
  런처는 여전히 `nvidia-smi` 실패 시 `--device cpu`로 폴백하고 배너에 알린다; CPU에서 warmup
  refill이 300 ms를 넘으면 **`IFQL_NUM_SAMPLES=16`** — 타임아웃은 늘리지 않는다.
- 🪤 **노트북 suspend/resume 뒤 CUDA가 죽는다** (2026-09-16 16:27 실측): resume 후 새 프로세스의
  `cuInit`이 torch·jax 모두 `CUDA unknown error`. 기존 서버 프로세스는 살아 보여도 믿지 말 것.
  복구는 nvidia 모듈 재로드(`sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe
  nvidia_uvm`, X가 잡고 있으면 실패) 또는 재부팅. 예방은 `sudo systemctl enable nvidia-suspend
  nvidia-resume nvidia-hibernate`. 세션 중 뚜껑을 닫지 말 것.
- **재계획 주기는 1.25 Hz로 고정**(24프레임 chunk, `action_queue.py` 상수 `QUEUE_LEN=24`) —
  실행은 30 Hz. GPU 한계가 아니라 학습 chunk 구조(8 knot × stride 3)이고 sim 평가도 같은
  조건. 더 자주 재계획하려면 서버 큐 상수를 바꿔야 하고 sim과 조건이 갈리므로 학습 세션과 상의.
- **proprio ~200 ms 미래 정렬 이슈** — `lead6`은 action만 shift하고 proprio는 그대로다(SPEC). 실기에서
  이상 거동(첫 청크에서 튐, 정지 자세에서 드리프트)이 보이면 **proprio 제외 ablation**을 먼저 본다.
  구체적 스위치는 런북 `REALROBOT_RUNBOOK_IFQL.md` 참조.
- **`FMRL_DETERMINISTIC=1`** — 서버 env. cudnn deterministic을 켜 bon 측정을 재현 가능하게 한다
  (약간 느려짐). 비교 실험에서 켤지 정한다.
- ✅ `.venv-svf`(uv py3.11, jax 0.6.2 cuda12 + torch 2.14+cu126, 7.5 GB)와 `.cache/torch`
  (ResNet18 + **dinov2 hub 캐시** — `FeatureHeads`가 r18_ss에서도 dinov2를 무조건 로드한다)는
  2026-09-16에 준비됐다. `requirements-svf-infer.txt`는 `hf_release/requirements-infer.txt`(CPU jax,
  torch 미고정)가 아니라 `vision_carrot/env/requirements.txt`의 cu126 핀에서 서빙 subset만 추린 것.
- ✅ **CPU smoke 실측 (2026-09-16, GPU 없음, 실물 take_01 frame 0 JPEG q92 1280×720)** —
  기동 6.2 s, RSS 1.6 GiB. bon32: refill p50 **53 ms** / max 143 ms(torch ResNet18 CPU 지터,
  `encode_ms` 129), 캐시 틱 2 ms; bc: refill 43 ms. envelope 위반 0, 첫 knot |Δq| ≤ 0.05 rad
  (클램프 0.5의 1/10), grip ∈ [0, 0.03]. `act_timeout_s 0.8` 대비 5배 이상 여유 → **K=32 유지**.
  RESET 응답에 계약 외 필드(`policy_type`·`state_dim 7`·`action_dim 7`·`n_action_steps 24`·`chunk`
  등)가 있지만 클라이언트는 `ok`만 보므로 무해. 산출물 `~/carrot_ifql/eval_runs/smoke_20260916_001204/`.
  ⚠️ 이 프레임(시작 자세)에서는 K=32 후보의 **Q spread가 ~1e-4로 사실상 0**(argmax가 랜덤하게
  흔들림) — 즉 시작 프레임에서 bon32 ≈ bc. 실물 bc-vs-bon 해석 시 참고.
- 클로즈드루프 성공률: 없음. §5가 첫 측정이다.

## 관련 문서

- [`GELLO_UR7E_FM_DEPLOY.md`](./GELLO_UR7E_FM_DEPLOY.md) — 두 프로세스 구조, 핸드셰이크 타임라인, 안전 모델 전문
- [`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md) — 실기 검증된 뼈대
- [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md) — 학습 데이터를 만든 레코더, 타임스탬프 아티팩트
- [`../../ros2_ur_ws/src/gello_policy/README.md`](../../ros2_ur_ws/src/gello_policy/README.md) — 패키지 개요
- `~/carrot_ifql/code/<snapshot>/vision_carrot/REALROBOT_RUNBOOK_IFQL.md` — 서버 CLI·로그·프로토콜 정본 (리포 밖)
