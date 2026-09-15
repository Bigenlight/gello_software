# sim_collect/eval — MuJoCo 안에서 정책 성공률(SR) 재기 (조작자 런북)

**학습된 carrot-in-pot 정책을 `sim_collect` MuJoCo 씬에서 닫힌 루프로 돌려 성공률을 잰다.**
정책은 **실기 배포와 똑같은 ZMQ 프로토콜**로 부른다 — 그래서 kanu GPU에 올린 체크포인트를
laptop3의 MuJoCo가 그대로 평가한다. 로봇은 쓰지 않는다.

> 이 문서는 **조작 절차서**다. 설계 계약(왜 이렇게 만들었나, 파일 소유자, 정확한 물리/클램프
> 계약)은 [`DESIGN.md`](DESIGN.md)에 있다.
> 데이터 수집 쪽은 [`../README.md`](../README.md)다. 실기 HIL 세션(`run_hil_*.sh` 3-CLI)과는
> **완전히 다른 스택이다** — 섞지 말 것.

> ✅ 아래 숫자는 2026-09-15 이 PC에서의 실측이다(zero/scripted/replay/운동학 재생). 학습된 정책의 SR은 아직 없다 — 실제 실행
> 결과가 나오면 통합자가 채운다. **추정값을 넣지 말 것.**

---

> ⚠️ **seed 선택 (2026-09-15 발견).** 시연 22개는 RESET마다 seed가 0,1,2,…로 자동 증가하며 녹화됐으므로
> **seed 0~19는 학습 배치와 정확히 같다**(당근 위치 오차 0.0 cm). 그래서 기본 seed는 **100~119**(held-out)다.
> seed 0~19로 잰 SR은 "학습 배치 재현" 지표이지 일반화 지표가 아니다.

## 0. SR이 무슨 뜻이고, 무슨 뜻이 아닌가 ← 먼저 읽어라

| | |
| --- | --- |
| **SR** | 성공 에피소드 ÷ 판정된 에피소드. 성공 판정은 `sim_collect/task.py::TaskEvaluator` — 당근 원점이 냄비 입구 실린더 안 + `\|v_z\| < 0.05` + `grip_cmd < 0.3`이 **`dwell_s`(기본 1.0 s) 동안 연속으로** 유지될 때 latch된다 |
| **시드** | 기본 20개(0–19). 시드가 씬 배치(당근/냄비 위치·요)를 정한다. 같은 시드 = 같은 장면 |
| **신뢰구간** | Wilson 95 %. `report.py`가 계산한다 |
| **분모** | `run_eval`의 SR 분모는 **돌린 에피소드 전부**(timeout·fault·failure 포함)다. `report.py`는 `episodes.jsonl`에서 outcome을 읽을 수 없는 행을 `unknown`으로 따로 세고 **분모에서 뺀다** — 정상 run에서는 두 값이 같고, 다르면 행이 깨진 것이다 |
| **잡기 불필요** | 판정은 당근의 **위치·정지·그리퍼 열림**만 본다. 열린 그리퍼로 당근을 냄비 안으로 **밀어 넣어도 성공**이다 — 설계상 그렇다(잡기 여부는 `grip_pos`·영상으로 따로 본다) |

🛑 **20 시드에서 CI는 넓다.** SR 0.5일 때 대략 **[0.30, 0.70]** — 폭이 0.4다.
즉 **10/20과 13/20은 이 실험으로 구별되지 않는다.** 두 체크포인트의 구간이 겹치면
"비슷하다"가 아니라 **"이 run으로는 구별 못 했다"**라고 써야 한다.

🛑 **SR은 실기 성능이 아니다.** 아래 §8의 시뮬-실기 격차를 읽기 전에는 이 숫자를 실기 성능의
예측치로 인용하지 말 것. 이 하네스가 재는 것은 **"정책이 이 시뮬 장면에서 이 태스크를 푸는가"**다.

🛑 **timeout·fault도 실패다.** RPC 타임아웃(§9)은 `fault`로 계산된다 — 실기 배포가 그렇게
취급하기 때문이다(`policy_leader_node`의 FAULT). `run_eval`의 기본 `--timeout-s`는 **0.6 s**로
실기와 같다. 그래서 **느린 CPU 서버로 잰 SR은 정책이 아니라 지연을 재고 있을 수 있다.**

---

## 1. 정책 4종 — 각각이 무엇을 증명하나

`run_eval --policy <...>`로 고른다. 모델 없이도 하네스를 검증할 수 있게 셋이 딸려 있다.

| 정책 | 무엇인가 | 무엇을 증명하나 |
| --- | --- | --- |
| **`zmq://127.0.0.1:<port>`** | 실기와 같은 ZMQ 프로토콜로 act/diffusion/fm 서버를 부른다 | **재고 싶은 그것.** 체크포인트의 SR |
| **`scripted`** | ground-truth 물체 자세에서 IK로 푸는 오라클(집기→옮기기→놓기) | **양성 대조.** 태스크가 이 씬에서 **풀리는가**, 성공 판정이 실제로 발동하는가. 목표 SR ≥ 15/20 |
| **`replay`** | 녹화된 take의 `command` 스트림을 30 Hz로 재생 | **사람 시범이 이 물리로 재현되는가.** 물리 발산은 예상된 것이고, 몇 개가 성공하는지 **정직하게 보고**한다 |
| **`zero`** | 시작 자세를 유지 (아무것도 안 한다) | **음성 대조.** SR이 **반드시 0**이어야 한다. 0이 아니면 성공 판정이 새고 있는 것이다 |

📌 **새 체크포인트를 재기 전에 `zero`와 `scripted`를 먼저 돌려라.** 5분이면 되고, "SR 0"이
정책 탓인지 하네스 탓인지를 가른다.

```bash
cd /home/laptop3/gello_software
export MUJOCO_GL=glfw DISPLAY=:0
.venv/bin/python -m sim_collect.eval.run_eval --policy zero     --seeds 0-4  --out sim_collect/eval/runs/zero_check
.venv/bin/python -m sim_collect.eval.run_eval --policy scripted --seeds 0-19 --out sim_collect/eval/runs/oracle
.venv/bin/python -m sim_collect.eval.run_eval --policy replay --takes ros2_ur_ws/gello_logs/sim --out sim_collect/eval/runs/replay
```

| 기준선 | 값 |
| --- | --- |
| `zero` SR | **0/5** (seed 0~4, 전부 timeout, fault 0) — 2026-09-15 실측 |
| `scripted` SR | **20/20** (seed 0~19; 20~39도 20/20), 229~256 스텝, 클램프 위반 0 — 2026-09-15 실측 |
| `replay` (22 take) | **18/22**, fault 0, 클램프 위반 0. 실패: take_08 냄비 안이지만 600스텝 상한에서 아직 움직임(900스텝이면 성공), take_09·take_17·take_21 들어 올린 뒤 냄비 밖에 떨어뜨림(개방 루프 물리 발산) |

---

## 2. (a) 체크포인트를 서빙한다

서버는 **실기에서 쓰는 바로 그 서버**(`policy_server/{act,fm,diffusion}_server.py`)다.
`serve_policy.sh`가 그것을 carrot 파라미터로 띄우고, kanu에 띄운 경우 **터널까지 들고 있는다.**

```bash
cd /home/laptop3/gello_software
./sim_collect/eval/serve_policy.sh --help
```

### 타입별 기본값

| `--type` | 포트 | `--n-action-steps` | `--task` | 비고 |
| --- | --- | --- | --- | --- |
| `act` | 5591 | 30 | **무시** (flag 자체가 없다) | |
| `diffusion` | 5592 | 32 | **무시** | 서버 기본 DDIM 10 step |
| `fm` | 5593 | 24 | **`"Put carrot in pot"`** ← CLIP 조건. 데이터셋 문자열과 **한 글자라도 다르면 안 된다** | Euler 기본 10 step |

🛑 **`--task`는 FM 전용이다.** `act_server.py` / `diffusion_server.py`에는 `--task` 플래그가
없어서 넘기면 **exit 2**로 죽는다. 이 스크립트는 FM이 아닌 타입에 `--task`가 오면 **경고만 찍고
전달하지 않는다.**

### 2-1. 로컬 (laptop3 CPU) — **느리다**

```bash
./sim_collect/eval/serve_policy.sh --type act \
    --checkpoint /path/to/outputs/<run>/checkpoints/last/pretrained_model
```

- venv는 `ros2_ur_ws/act_venv` (py3.12, torch + lerobot 0.6.1). 스크립트가 알아서 고른다.
- **`--device`의 기본값이 `cpu`다.** laptop3에는 이 스택이 쓸 수 있는 torch GPU가 없다.
  `--device cuda`를 주면 서버가 **조용히 CPU로 떨어지지 않고 exit 3으로 거부한다**(의도된 설계).
- 🛑 **CPU forward가 `run_eval --timeout-s`(기본 0.6 s)를 넘으면 그 스텝은 `fault` = 실패
  에피소드다.** 로컬 CPU로 SR을 믿으려면 **그 값부터 올려라**(예 `--timeout-s 10`). 안 그러면
  정책이 아니라 laptop3의 CPU를 재게 된다. ⚠️ **올린 타임아웃으로 잰 SR은 실기 계약(0.6 s)과
  다른 실험이다** — 보고할 때 그 사실을 같이 적는다.
- 🛑 **RAM.** 체크포인트 + torch는 GB 단위다. 띄우기 전에 `free -g`. (2026-09-15 이 박스는
  15 GB 중 ~7 GB 여유였다.)
- 이 터미널이 서버다. **Ctrl-C가 서버를 끈다.**

### 2-2. kanu (GPU) + 터널 ← **권장 경로**

```bash
# 처음 한 번: 서버 코드를 kanu의 내 작업 공간으로 복사 (--sync)
./sim_collect/eval/serve_policy.sh --type fm --remote kanu --gpus 7 --sync \
    --checkpoint /home/junhyeong/workspace/youngwoong/carrot_eef/outputs/<run>/checkpoints/last/pretrained_model
```

일어나는 일, 순서대로:

1. `rsync`로 `ros2_ur_ws/src/gello_policy/policy_server/` →
   `kanu:~/workspace/youngwoong/sim_eval_policy_server/policy_server/` (`--sync`일 때만)
2. kanu 프리플라이트: venv python 존재, 서버 파일 존재, **체크포인트 디렉터리 존재**,
   python 의존성(`zmq torch lerobot numpy`, FM은 `transformers` 추가), GPU 여유, 포트 여유
3. `nohup env CUDA_VISIBLE_DEVICES=<gpus> ... python <type>_server.py --host 127.0.0.1 --port <port> ...`
   — **127.0.0.1에만 바인드한다**(외부 노출 없음). 로그는 `~/logs/sim_eval_<type>_<port>_<UTC>.log`,
   PID는 `~/logs/sim_eval_<type>_<port>.pid`
4. `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:<port>:127.0.0.1:<port> kanu`
5. **RESET 왕복이 성공할 때까지 대기**(기본 600 s. FM/diffusion 콜드 로드는 분 단위다)

🛑 **kanu는 남의 작업이 도는 공용 박스다.** 그래서:

- 이 스크립트는 **자기가 만든 PID 파일의 pid만** 죽인다. 죽이기 전에 `/proc/<pid>/cmdline`이
  **그 타입의 서버 + 그 포트**임을 확인하고, 아니면 **거부하고 아무것도 건드리지 않는다.**
  `pkill`·`killall`은 코드에 없다(회귀 테스트가 막고 있다).
- 기본 `--gpus`는 `7`이다. **요청한 GPU에 이미 1000 MiB 이상 쓰이고 있으면 기동을 거부한다.**
  먼저 `ssh kanu nvidia-smi`로 빈 GPU를 보고 `--gpus N`을 골라라.
  내 작업인 걸 아는 경우에만 `SIM_EVAL_ALLOW_BUSY_GPU=1`.
- **아무것도 설치하지 않는다.** 패키지가 없으면 `pip install` **줄을 출력만** 한다.

터미널 유지 규약:

```
Ctrl-C  → 터널만 닫힌다. kanu의 서버는 계속 산다.
정지    → ./sim_collect/eval/serve_policy.sh --type fm --remote kanu --stop
```

### 2-3. 먼저 보기만 하기 (`--check`) / 소켓만 찔러 보기 (`--probe`)

```bash
# 무엇을 실행할지 전부 출력하고 끝. ssh도 rsync도 하지 않는다.
./sim_collect/eval/serve_policy.sh --type fm --remote kanu --gpus 7 --sync --check \
    --checkpoint /remote/path/pretrained_model

# 이미 떠 있는 엔드포인트에 RESET 한 번 (exit 0이면 살아 있다)
./sim_collect/eval/serve_policy.sh --port 5593 --probe
```

### 2-4. 환경변수

| 변수 | 뜻 |
| --- | --- |
| `SIM_EVAL_STATE_DIR` | 터널 PID 파일 위치 (기본 `${XDG_RUNTIME_DIR:-/tmp}/sim_collect_eval`) |
| `SIM_EVAL_REMOTE_LOGDIR` | 원격 로그/PID 디렉터리 (기본 `~/logs`) |
| `SIM_EVAL_REMOTE_GPUS` | `--gpus` 기본값 |
| `SIM_EVAL_ALLOW_BUSY_GPU` | `1`이면 사용 중인 GPU에도 기동 |
| `SIM_EVAL_HF_OFFLINE` | FM 전용, 기본 `1`. `run_fm_server.sh`와 같은 이유로 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`을 켠다. ⚠️ **kanu에 HF 캐시가 없어 CLIP `from_pretrained`가 죽으면 `0`으로 다시 띄워라** |

---

## 3. (b) 평가를 돌린다

**다른 터미널에서** (서버 터미널은 터널을 들고 있어야 한다):

```bash
cd /home/laptop3/gello_software
MUJOCO_GL=glfw DISPLAY=:0 .venv/bin/python -m sim_collect.eval.run_eval \
    --policy zmq://127.0.0.1:5593 --task "Put carrot in pot" \
    --seeds 0-19 --max-steps 600 --dwell-s 1.0 --timeout-s 0.6 \
    --out sim_collect/eval/runs/fm_<checkpoint>_<날짜>
```

자주 쓰는 플래그 (`--help`가 정본):

| 플래그 | 기본값 / 뜻 |
| --- | --- |
| `--policy` | `zmq://host:port` \| `scripted` \| `replay` \| `zero` |
| `--seeds` | `0-19` 또는 `0,3,7` (미지정이면 yaml `eval.seeds`) |
| `--takes` | `--policy replay`일 때 take 디렉터리 |
| `--max-steps` / `--dwell-s` | 600 / 1.0 (yaml `eval.*`가 기본) |
| `--timeout-s` | **0.6** — ZMQ REQ 타임아웃. **넘으면 `fault`** |
| `--video` | `ep_<id>_cam1.mp4` / `_cam2.mp4` (mp4v, 30 fps) |
| `--no-state` | 에피소드별 h5 상태 로그 생략 |
| `--config` | `sim_collect/configs/carrot_in_pot_sim.yaml` |

| 항목 | 값 |
| --- | --- |
| 인터프리터 | **`.venv/bin/python` 하나뿐** (mujoco 3.10). `act_venv`에는 mujoco가 없다 |
| 화면 | **헤드리스**로 돈다(뷰어 없음). 렌더는 `MUJOCO_GL=glfw` + `DISPLAY=:0`이 필요하다 — 이 PC에서 EGL/OSMESA는 죽는다 |
| 페이싱 | **없다(lockstep).** 느린 정책은 run을 길게 만들 뿐 결과를 바꾸지 않는다 — 단 RPC 타임아웃은 예외(§0) |
| 종료 코드 | **항상 0.** SR은 데이터지 테스트가 아니다 |

⚠️ **뷰어가 떠 있는 상태로 돌리지 말 것.** `mujoco.Renderer`와 `mujoco.viewer`는 한 프로세스에
같이 살 수 없다. `sim_collect` 수집 스택(`run_sim_collect.sh`)이 돌고 있으면 먼저 끈다.

---

## 4. (c) 결과를 읽는다

`--out` 디렉터리에 남는 것 (DESIGN §2.3):

| 파일 | 내용 |
| --- | --- |
| `episodes.jsonl` | 에피소드 1줄: `seed`, `outcome`, `n_steps`, `t_success_s`, `detail`, 정책 메타, wall time |
| `summary.json` | SR, Wilson CI, outcome별 개수, config sha, git commit, 체크포인트 id |
| `summary.md` | 위를 표로 |
| `ep_<id>.h5` | 그 에피소드의 `qpos/qvel/ctrl` (125 Hz) — **재생용**. `--no-state`면 없다 |
| `ep_<id>_cam1.mp4` · `_cam2.mp4` | `--video`를 줬을 때만 |

`outcome` 값의 뜻:

`outcome`은 **정확히 네 개**다 (`sim_collect/eval/world.py::Outcome`):

| 값 | 뜻 | 다음에 할 일 |
| --- | --- | --- |
| `success` | dwell latch까지 성립 | — |
| `timeout` | `--max-steps`(기본 600 = 20 s @30 Hz)까지 못 풀었다 | 영상/재생으로 **어디서 막혔는지** 본다 |
| `fault` | 정책이 응답을 못 줬다(RESET/ACT 타임아웃·`ok:false`·잘못된 액션) — 사유는 `fault` 필드 | **모델이 아니라 인프라 문제일 수 있다** → §9-2 |
| `failure` | 물리적 실패(당근 이탈, 관절 한계, NaN 등) — 사유는 `failure` 필드 | `failure`/`detail`을 읽는다 |

🛑 **`fault`가 여러 개면 SR을 인용하기 전에 원인을 먼저 없애라.** fault는 정책의 실력이 아니라
서버/터널/타임아웃 이야기다.

---

## 5. (d) 실패한 에피소드를 다시 본다

에피소드마다 `sim_mj_state` 형식(`qpos/qvel/ctrl`)이 남으므로 녹화 take와 **같은 도구로** 재생한다.

```bash
# 영상이 있으면 먼저 그걸 본다 (가장 빠르다)
mpv sim_collect/eval/runs/<run>/ep_07_cam1.mp4

# 상태 재생 (replay_take.py의 load_scene/load_model/load_state/set_row API)
.venv/bin/python sim_collect/tools/replay_take.py --help
```

📌 실패를 볼 때의 체크리스트:

1. **잡기까지 갔나?** 못 갔으면 인지(시각) 문제다 → §8.
2. **잡고 나서 놓쳤나?** 그리퍼 명령/접촉 문제다.
3. **냄비 위에서 놓았는데 성공이 안 떴나?** dwell latch가 안 걸린 것 — 당근이 튕겨 나갔거나
   `|v_z| < 0.05`를 못 만족했을 수 있다. `detail`을 읽어라.
4. **팔이 엉뚱한 쪽으로 갔나?** §7의 −π 브랜치 함정.

---

## 6. (e) 체크포인트 두 개를 비교한다

```bash
.venv/bin/python -m sim_collect.eval.report \
    sim_collect/eval/runs/act_40k sim_collect/eval/runs/act_80k \
    --label 40k --label 80k --compare \
    --out sim_collect/eval/runs/compare_act_40k_80k
```

나오는 것: `report.md`(표 5절) + `episodes.csv` + `per_seed.csv` + `runs.csv`.
인자 없이 하나만 줘도 되고, 기본은 markdown을 stdout에 찍는다.

| `report.py`가 하는 것 | 왜 |
| --- | --- |
| `episodes.jsonl`에서 **SR을 다시 계산**한다 | `summary.json`은 편의 사본이다. 원증거에서 다시 센다 |
| 판정 불가 행을 **분모에서 뺀다** | 깨진 줄 하나가 SR을 조용히 깎으면 안 된다. 개수는 보고서에 찍힌다 |
| **시드별 표**를 만든다 | 빈 칸(안 돌린 시드)과 실패를 **구별해서** 보여 준다 |
| **McNemar 정확검정**(짝지은 검정) | 시드가 고정이라 A와 B가 **같은 장면**을 봤다. 짝지은 자료에 독립 비율 검정을 쓰면 안 된다 |

🛑 **`p`는 판결이 아니다.** 20 시드에서 유의하려면 **불일치 시드가 한쪽으로 6개**는 나와야
0.05에 닿는다(정확히 `2/2⁶ = 0.03125`). 그 아래는 "구별 못 했다"다.
그리고 `A only` / `B only` 시드 번호가 보고서에 찍히므로 **Δ를 믿기 전에 그 시드들을 재생해서
눈으로 확인하라.**

---

## 7. 결정성과 함정

### 7-1. 무엇이 재현되고 무엇이 안 되나

| | 재현되나 |
| --- | --- |
| 씬 배치 | ✅ 시드가 고정한다 (`scene.sample_layout(cfg, seed, attempt)`) |
| 물리 | ✅ 같은 (시드, config, 정책 출력)이면 같은 궤적 |
| **ACT** | ✅ 결정적 |
| **Diffusion / FM** | ❌ **샘플링이 확률적이다.** 같은 체크포인트·같은 시드도 다른 에피소드가 나올 수 있다 |

→ diffusion/FM의 SR 차이를 논할 때는 **이 잡음이 CI 안에 이미 들어 있다**는 점을 기억하라.
서버가 torch 시드를 노출하지 않으면 보고서에 그 사실을 적는다(지금은 노출하지 않는다).

### 7-2. 🪤 −π shoulder_pan 브랜치

**당근 데이터는 `shoulder_pan`의 −π 브랜치 위에 있다.**

| | 값 |
| --- | --- |
| 정본 파일 | `/home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy/config/carrot_eef_limits.json` |
| `start_pose` | `[-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]` |
| J1 한계 | `[-3.8049, -2.5787]` |
| 시뮬 home | `[-3.302, -1.563, 1.607, -1.523, -1.615, -3.118]` ← **같은 브랜치다** |

🛑 **바나나(+π) 값을 쓰지 말 것.** 같은 팔 자세를 +2π 떨어진 관절 값으로도 표현할 수 있어서
**틀린 브랜치를 써도 에러가 안 난다** — 팔이 반대로 한 바퀴 돌 뿐이다. 관절 한계 위반이나
"왜 이렇게 크게 도나"로 나타난다.

---


### 결과 기록 — sim ACT (2026-09-15, CPU 서빙, n_action_steps 30, 600스텝 상한, dwell 1 s)

| 체크포인트 | 학습 배치(seed 0~19) | held-out(seed 100~119) | held-out 실패 중 "접근 실패" |
| --- | --- | --- | --- |
| ACT 10k | 11/20 | 7/20 | 8 |
| ACT 20k | 11/20 | 7/20 | 6 |
| ACT 30k | 12/20 | 6/20 | 11 |
| ACT 40k | – | 7/20 | 12 |
| ACT 50k | – | **10/20** | 8 |
| Diffusion 10k | – | 0/10 | 10 |
| Diffusion 20k | – | 1/10 | 8 |
| Diffusion 30k | – | **12/20** | 8 |
| Diffusion 40k | – | 10/20 | – |
| Diffusion 50k | – | 7/20 | – |
| FM 10k (Euler 10스텝, CPU) | – | 4/20 | – |
| Diffusion 60k | – | 9/20 | – |
| Diffusion 70k | – | **13/20** | – |
| Diffusion 80k | – | 8/20 | – |
| Diffusion 90k | – | 8/20 | – |
| Diffusion 100k (last) | – | 11/20 | – |
| FM 20k (Euler 10스텝, CPU) | – | 7/20 | – |
| FM 30k (Euler 10스텝, CPU) | – | 4/20 | – |
| FM 40k (Euler 10스텝, CPU) | – | 6/20 | – |
| FM 50k (Euler 10스텝, CPU) | – | **9/20** | – |
| FM 60k (Euler 10스텝, CPU) | – | 7/20 | – |
| FM 70k (Euler 10스텝, CPU) | – | **11/20** | – |
| FM 80k (Euler 10스텝, CPU) | – | **11/20** | – |
| FM 90k (Euler 10스텝, CPU) | – | 10/20 | – |

**60-seed 최종 비교(seed 100~159)**: Diffusion 70k **33/60 = 55 %** · FM 80k **33/60 = 55 %** (Wilson 95 % 42~67 %) · ACT 50k 25/60 = 42 % (30~55 %).

학습 데이터: `Bigenlight/carrot_in_pot_sim_lerobot_v3`(22 에피소드). fault·봉투/max_dev 클램프는 전 구간 0.
20 seed의 Wilson 구간은 ±20 %라 체크포인트 간 차이는 통계적으로 구분되지 않는다. 지배적 실패는
"당근에 접근을 못 함(r > 0.3 m)"으로, 시연 배치 범위(당근 x 0.40~0.49, y 0.12~0.24) 밖 일반화 한계로 보인다.

## 8. 🛑 시뮬 ≠ 실기 — 시각 격차

정책은 **실제 카메라 이미지**로 학습됐다. 이 하네스는 **MuJoCo 렌더**를 보낸다.

| | 실기 | 시뮬 |
| --- | --- | --- |
| 배경 | 실제 테이블 | **텍스처 바닥(z=0)** |
| 로봇 외형 | UR7e | 기구학은 UR7e, **메시는 menagerie UR5e** |
| 조명·노이즈·모션블러 | 실제 | 렌더 |
| 물체 | 실제 당근/냄비 | 시뮬 메시 |

→ **낮은 SR이 "정책이 나쁘다"를 뜻하지 않는다.** 도메인 갭일 수 있다. 그래서 §1의
`scripted`(양성 대조)와 §1의 `replay`가 같이 있는 것이다: 오라클이 15/20을 내는데 정책이
0/20이면 **태스크는 풀리는데 정책이 이 픽셀을 못 읽는 것**이다.

→ 반대로 **높은 SR도 실기 성공을 보장하지 않는다.** 이 하네스는 **상대 비교**(체크포인트 A vs
B, 같은 시드)에 훨씬 더 정직하다.

---

### 8.5 원격(kanu GPU) 서빙 실측 — 이 회선에서는 로컬 CPU보다 느리다 (2026-09-15)

`serve_policy.sh --remote kanu --gpus 0`은 정상 동작했지만(rsync·기동·터널·프로브 4 s), laptop3↔kanu 회선이
약 5 MB/s라 매 30 Hz 스텝마다 1280×720 JPEG 두 장(≈200 KB)을 터널로 보내는 비용이 GPU 추론 이득을 지웠다:
Diffusion 에피소드당 **134 s**(로컬 CPU 80~100 s), 실기 타임아웃 0.6 s에서 청크 생성 시점마다 **fault**.
결론: 이 회선에서는 **로컬 CPU + `--timeout-s 30`** 이 맞다. 원격 서빙은 회선이 빠른 곳(같은 서버 안, 또는
sim 자체를 GPU 서버에서 돌릴 때)에서만 유리하다.

## 9. 검증과 트러블슈팅

### 9-1. 데이터셋 자체 검증 (성공 판정이 맞는지)

성공 판정이 **사람 시범 22개를 실제로 성공으로 읽는지** 먼저 확인한다.
이게 "데이터셋을 MuJoCo에서 재생해 보면 방법이 드러난다"는 조작자 요구 그 자체다.

```bash
MUJOCO_GL=glfw DISPLAY=:0 .venv/bin/python -m sim_collect.eval.validate_success_on_demos \
    --takes ros2_ur_ws/gello_logs/sim [--dwell-s 1.0] [--no-dynamic] [--limit N] \
    [--out sim_collect/eval/runs/demo_validation.json]
```

`--no-dynamic`은 운동학 재생만 하고 `ReplayPolicy`(동적 재생)를 건너뛴다 — 빠른 확인용.

기대:

| 검사 | 기대 |
| --- | --- |
| t=0에서 성공? | **전부 아니오** |
| 끝까지 갔을 때 성공? | **22/22 예** |
| 판정이 처음 발동한 시각 vs 레코더의 `task_success_at_stop` | 표로 출력 |
| `ReplayPolicy`(동적 재생) 성공 수 | 물리 발산 때문에 22보다 적을 수 있다 — **정직하게 보고한다** |

| 결과 | 값 |
| --- | --- |
| 운동학 재생 22 take | **22/22 성공, t=0에서 성공 0/22**, 레코더 플래그와 22/22 일치, xml 재빌드 일치 22/22, 성공 latch 시각 9.7~26.2 s |
| 동적 `replay` 22 take | **18/22** (소프트스타트 램프를 실기처럼 경과된 상태로 시작한 뒤 값. 이전 19/22; 실기 봉투(`carrot_eef_limits.json`)를 쓰면 16/22, 봉투 클램프가 24 % 스텝에서 물림 → sim 시연에서 도출한 봉투가 기본) |

⚠️ 22개 take의 `sim_meta.layout_seed`는 **0으로 잘못 기록돼 있다.** 초기 배치는
`sim_object_poses` 0행 / `sim_mj_state` 0행에서 읽는다 (DESIGN §1.4).

### 9-2. 트러블슈팅

| 증상 | 원인 / 조치 |
| --- | --- |
| `outcome: fault`가 우수수 | 정책 응답이 타임아웃이거나 `ok:false`. **로컬 CPU 서버면 거의 확실히 타임아웃이다** → kanu로 옮기거나 `--timeout-s`를 올린다(⚠️ 올린 값을 보고서에 반드시 적을 것 — 실기 계약은 0.6 s다). 사유는 `episodes.jsonl`의 `fault` 필드. 서버 로그: `ssh kanu tail -f ~/logs/sim_eval_<type>_<port>_*.log` |
| 첫 RESET이 안 돌아온다 (`SIM_EVAL_PROBE=timeout`) | 모델 로딩 중일 수 있다(FM 콜드 로드는 분 단위). `--wait-s`를 올리고, 그래도 안 되면 원격 로그를 본다. **서버는 자동으로 죽지 않는다** — `--stop`으로 직접 정리 |
| `SIM_EVAL_PROBE=refused` | 서버는 살아 있는데 RESET이 `ok:false`다. 체크포인트가 잘못됐거나 로드 실패. 대기해도 안 낫는다 |
| 터널이 바로 죽는다 | 로컬 포트가 이미 점유됐다(`ExitOnForwardFailure=yes`가 붙어 있어 조용히 실패하지 않는다). 다른 세션이 5591–5593을 쓰고 있는지 확인하거나 `--local-port`로 옮긴다. ⚠️ **터널이 죽어도 kanu의 서버는 산다** |
| kanu 기동이 `missing: zmq`로 거부 | 스크립트는 설치하지 않는다. 출력된 `pip install pyzmq` 줄을 **직접** 실행한다 |
| FM이 CLIP `from_pretrained`에서 죽는다 | kanu에 HF 캐시가 없다. `SIM_EVAL_HF_OFFLINE=0`으로 다시 기동 |
| GPU 거부 (`already has NNN MiB in use`) | 남의 작업이다. `ssh kanu nvidia-smi`로 빈 GPU를 골라 `--gpus N` |
| `EEF checkpoint` 거부 메시지 | RESET 응답의 `state_dim`/`action_dim`이 7이 아니다(EEF 체크포인트는 16). **v1 하네스는 joint 7/7만 지원한다** — `ZmqPolicy`가 첫 RESET에서 거부한다. 억지로 우회하지 말 것 |
| 렌더가 안 된다 / EGL 오류 | `MUJOCO_GL=glfw` + `DISPLAY=:0`가 필요하다. 이 PC에서 EGL/OSMESA는 죽는다 |
| `mujoco.Renderer` 관련 크래시 | 뷰어가 떠 있는 프로세스와 섞였다. `run_sim_collect.sh` 스택을 끄고 다시 돌린다 |
| SR이 0인데 `zero`도 0, `scripted`도 0 | 하네스/씬 문제다. 정책 이야기가 아니다. `validate_success_on_demos`부터 돌려라 |
| `zero`의 SR이 0이 아니다 | **성공 판정이 새고 있다.** 다른 어떤 숫자도 인용하지 말 것 |

---

## 10. 파일

| 파일 | 소유 | 하는 일 |
| --- | --- | --- |
| `world.py` · `policies.py` · `run_eval.py` · `validate_success_on_demos.py` | F1 | 헤드리스 월드(실기 클램프·250 Hz 업샘플러 그대로), 정책 4종, CLI, 데모 검증 |
| `scripted_policy.py` · `ik.py` | F2 | 오라클(양성 대조) + `ur_kin` 래퍼 |
| `serve_policy.sh` · `report.py` · 이 문서 | O1 | 정책 서버·터널, 집계 보고서 |
| `DESIGN.md` | 통합자 | 설계 계약 |

그 밖에 `eval/`에 있는 모듈은 F1/F2의 것이다 — 이 문서를 고치는 사람은 **그 파일들을 편집하지
않는다.** 스텁 서버는 `sim_collect/tests/stub_policy_server.py`(F1)이며 CLI로도 띄울 수 있다:
`.venv/bin/python -m sim_collect.tests.stub_policy_server --port 5599`.

테스트:

```bash
cd /home/laptop3/gello_software
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  sim_collect/tests/test_eval_*.py
```
