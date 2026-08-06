# 정책 평가(BC / FM) 스택 — 정본 (한국어)

> **이 파일이 BC·FM 실기 평가의 진입점이다.** 여기서 "무엇을·왜"를 잡고, 실제 타이핑은
> 런북([`BC_DEPLOY_KO.md`](BC_DEPLOY_KO.md) / [`FM_DEPLOY_KO.md`](FM_DEPLOY_KO.md))에서,
> 코드·기록 스키마는 [`POLICY_EVAL_CODE_MAP_KO.md`](POLICY_EVAL_CODE_MAP_KO.md)에서 본다.
>
> **상태: BC·FM 둘 다 실기 UR7e에서 검증 완료(2026-07-31 ~ 08-02).**

---

## 1. 이게 뭐고, 뭐가 아닌가

외부 담당자가 서버에서 학습한 **정책 2종을 실기 UR7e에서 "얼마나 잘 하는지" 재는 스택**이다.

| | BC | FM |
|---|---|---|
| artifact 형식 | `hil-serl-bc-init` v1 | `hil-serl-jax-flow-matching` v1 |
| 정체 | production SAC hybrid 에이전트의 `modules_actor` + `modules_grasp_critic` 서브트리를 이식(graft) | JAX rectified flow — horizon **16** 액션 청크를 **Euler 8-step**으로 샘플링, **첫 액션만 실행**하고 매 스텝 재계획 |
| 결정성 | 항상 argmax → 결정적 | **확률적**(같은 장면에서도 액션이 조금씩 다르다 — 정상) |
| 원격 포트 | **50054** | **50055** |

**맞는 것**: 로봇·카메라·GELLO·GUI·데드맨·preflight·actor·gRPC 계약이 **production HIL 세션과
100 % 동일**하다. laptop3 스크립트는 **한 줄도 고치지 않는다** — 바뀌는 것은 환경변수 핀 3종과
터널의 반대쪽 끝뿐이다.

**아닌 것**:
- **학습이 전혀 없다.** 서버는 **고정 가중치 추론 + 기록** 전용이다. gradient도 replay도 없다.
- **reward classifier를 쓰지 않는다.** `--no-classifier-sidecar`로 끈다. 성공 라벨은
  **오직 GUI `MARK SUCCESS`**(MANUAL)다.
- **production RLPD learner(`:50053`)와 무관하다.** 포트가 달라 아예 연결되지 않는다.
  learner는 평가 내내 살아 있어도 되고, 꺼져 있어도 평가에 아무 영향이 없다.
  두 런처는 50053을 **거부**하도록 코딩돼 있어 구조적으로 learner를 건드릴 수 없다.

---

## 2. 아키텍처

```
laptop3 (production 스크립트 무수정 — 환경변수 핀 override만)
  T2  ./run_hil_hardware.sh                 UR7e + Robotiq + passive GELLO
  T3  <핀 3종> ./run_hil_session.sh --no-classifier-sidecar
                                            cameras + GUI + preflight + actor
  T4  ./run_bc_rollout_recorder.sh          (선택) 순수 구독자 — BC/FM 공용
        │
        actor ──gRPC──► 127.0.0.1:50153     ◄── 로컬 터널 입구는 하나뿐이다
                              │
  T1  ./run_bc_server.sh ──ssh 터널──►  junhyeong_ai:50054   run_bc_policy_server.py
      ./run_fm_server.sh ──ssh 터널──►  junhyeong_ai:50055   run_fm_policy_server.py
                                        junhyeong_ai:50053   RLPD learner (불가침)
```

actor는 예나 지금이나 `127.0.0.1:50153`만 본다. **T1 한 개만 열 수 있다** —
production `run_hil_server.sh`·BC·FM 셋이 같은 로컬 포트를 쓴다.

---

## 3. 세팅 체크리스트 (T1 누르기 전)

| # | 확인 | 방법 / 조치 |
|---|---|---|
| 1 | **laptop3 ↔ 서버 리포 HEAD 일치** | 런처가 검사하고 불일치면 거부한다(보고만 하고 고치지 않는다). 조치: `git -C /home/laptop3/gello_software push` 뒤 `ssh junhyeong_ai 'git -C ~/gello_software_runtime pull --ff-only'`. 탈출구 `BC_ACCEPT_HEAD_MISMATCH=1` / `FM_ACCEPT_HEAD_MISMATCH=1` |
| 2 | **artifact 존재** | 서버 `~/hil-serl-data/diagnostics/` 아래 `*.bc-init` / `*.fm-init` 디렉터리. manifest sha 검증은 **서버가 자동으로** 한다 |
| 3 | **로컬 50153이 비어 있음** | `ss -ltn \| grep 50153` → 아무 줄도 안 나와야 한다. 나오면 다른 T1 터미널에서 `Ctrl-C`(터널만 닫힌다 — 서버 learner는 산다) |
| 4 | 로봇·카메라·GELLO 준비 | production과 동일 → [`docs/testing/09_HIL_ACTOR_RUNBOOK.md`](../docs/testing/09_HIL_ACTOR_RUNBOOK.md) |

**핀 3종** (T3와 dry handshake에 **똑같이** 붙인다 — 값을 임의로 고치지 말 것. 잘못된 정책을
실기에서 돌리는 것을 막는 게이트다):

```bash
# BC
EXPECTED_MODEL_ID=bc-cube-in-cup-raw0731-bcinit-v1 EXPECTED_REWARD_AUTHORITY=local EXPECTED_REWARD_MODEL_ID=operator-manual-success-v1
# FM
EXPECTED_MODEL_ID=fm-cube-in-cup-raw0731-h16-euler8-v1 EXPECTED_REWARD_AUTHORITY=local EXPECTED_REWARD_MODEL_ID=operator-manual-success-v1
```

---

## 4. 사용 요약표

모든 셸 명령은 `cd /home/laptop3/gello_software/ros2_ur_ws` 기준이다(분석기만 예외).

| 목적 | 명령 | 성공 표식 | 상세 |
|---|---|---|---|
| **BC T1** 서버+터널 | `./run_bc_server.sh` | `BC_SERVER_RESULT=started` + `[bc-server] ready …` + 프롬프트 안 돌아옴 | [BC §4.1](BC_DEPLOY_KO.md) |
| **FM T1** 서버+터널 | `./run_fm_server.sh` | `FM_SERVER_RESULT=started` + `[fm-server] ready …` + 프롬프트 안 돌아옴 | [FM §2.1](FM_DEPLOY_KO.md) |
| **T2** 하드웨어 (BC/FM 공통) | `./run_hil_hardware.sh` | READY 배너 + 세 토픽 rate 충족 | [BC §4.3](BC_DEPLOY_KO.md) |
| **T3** 세션 (핀 3종 + sidecar off) | `<핀 3종> ./run_hil_session.sh --no-classifier-sidecar` | preflight `[1]`~`[11]` 통과 → GUI `WAIT_SCENE_READY` | [BC §4.4](BC_DEPLOY_KO.md) |
| **dry handshake** (로봇 안 씀, 권장) | `<핀 3종> ./run_hil_actor.sh --fake-env --no-classifier-sidecar` | 핸드셰이크 오류 없이 `BeginEpisode`까지 가고 종료. 센서 WARN은 정상 | [BC §4.2](BC_DEPLOY_KO.md) |
| **T4** 로봇측 녹화 (선택, BC/FM 공용) | `./run_bc_rollout_recorder.sh` | `### BC ROLLOUT RECORDER` + `ROLLOUT_DIR ->` 경로. **Ctrl-C로만 종료** | [BC §4.5](BC_DEPLOY_KO.md) |
| **step timing — T1** 서버측 분해 (선택) | `HIL_STEP_TIMING=1 ./run_bc_server.sh` (FM은 `./run_fm_server.sh`) | ready 라인에 **`step_timing=1` 토큰**이 붙는다 | [BC §4.6](BC_DEPLOY_KO.md) · [FM §5](FM_DEPLOY_KO.md) |
| **step timing — T3** 액터측 분해 (선택) | `HIL_STEP_TIMING=1 <핀 3종> ./run_hil_session.sh --no-classifier-sidecar` | actor 기동 로그에 `[actor] step timing ON -> <경로>` 한 줄 | [BC §4.6](BC_DEPLOY_KO.md) · [FM §5](FM_DEPLOY_KO.md) |
| **분석** (담당자, 세션 후) | 아래 블록 | `<served>/analysis/rollout_report.md` + `report.json` 생성 | [코드 지도](POLICY_EVAL_CODE_MAP_KO.md) |

```bash
cd /home/laptop3/gello_software
/home/laptop3/venvs/gello-hil-actor/bin/python serl_ur_infra/scripts/analyze_bc_rollout.py \
  --served <서버에서 받아온 served 디렉터리> \
  --robot  ros2_ur_ws/gello_logs/bc_rollouts/rollout_<타임스탬프>
```

`--robot`은 선택이다(없으면 서버 기록만으로 리포트를 만든다). step timing을 켰다면
`--actor-timing <액터 jsonl>`을 더한다 — 서버 `timing.jsonl`은 `--served` 디렉터리에서 **자동으로
찾는다.** 그 밖의 옵션은 `--out` / `--episode <id>` / `--deep-dive-rows N` / `--print`.

**artifact 교체 — T1만 내렸다 올리면 되고 코드 수정은 0이다.**

```bash
BC_ARTIFACT_DIR=<서버 절대경로> ./run_bc_server.sh
FM_ARTIFACT_DIR=<서버 절대경로> ./run_fm_server.sh
FM_WHICH=final ./run_fm_server.sh     # 기본 best, final도 서빙 가능
```

---

## 5. 기록물 (어디에 뭐가 남나)

| 위치 | 내용 |
|---|---|
| 서버 `~/hil-serl-data/bc_eval/bc_eval_<ts>/served/`<br>서버 `~/hil-serl-data/fm_eval/fm_eval_<ts>/served/` | episode별 pickle + `actions.jsonl` + `inference.jsonl`. **T4 없이도 항상 생성된다** |
| 서버 `…/<ts>/server.log` | 서버 기동·ready·추론 로그. 실패 시 런처가 마지막 40줄을 찍어 준다 |
| laptop3 `ros2_ur_ws/gello_logs/bc_rollouts/rollout_<ts>/` | `robot/vectors.h5`(native + synchronized 테이블) · `robot/cam1.mp4`/`cam2.mp4` · `robot/metadata.json` · `status.jsonl`(세션 상태 타임라인) |
| 서버 `…/served/timing.jsonl` | 서버측 스텝 latency 분해 — Step RPC당 1줄 + BeginEpisode당 1줄. **T1을 `HIL_STEP_TIMING=1`로 띄웠을 때만** 생성된다(기본 OFF) |
| laptop3 `ros2_ur_ws/gello_logs/step_timing/actor_step_timing_<ts>.jsonl` | 액터측 스텝 latency 분해 — Step RPC를 완료한 루프 반복당 1줄. **T3에 `HIL_STEP_TIMING=1`을 붙였을 때만** 생성된다(`--step-timing-path`로 경로 지정 가능) |

⚠️ **`inference.jsonl`은 `084ec1a` 이후에 기동한 서버에서만 나온다.** 그 전에 뜬 서버 프로세스를
재사용하면 이 파일이 없다.

**필드 단위 스키마·정렬 기준·코드 구조는 여기서 중복하지 않는다** →
[`POLICY_EVAL_CODE_MAP_KO.md`](POLICY_EVAL_CODE_MAP_KO.md).

---

## 6. 검증된 수치

| 항목 | BC | FM |
|---|---|---|
| holdout MSE (우리 재현 / 담당자 보고) | **0.0325 / 0.0325** | first-action **0.0375 / 0.0472** |
| translation cosine | 0.486 | **0.637** |
| rotation cosine | — | 0.166 |
| gripper 정확도 | 94 % | **96.7 %** |
| 서버 추론 지연 | **~6 ms** | **~72 ms** (jit 후) |
| 계약 위반 (holdout) | 0 | 0 |

🪤 **FM jit 교훈**: eager 실행일 때 Euler 8-step이 op 단위로 디스패치되어 **한 액션에 502 ms**가
걸렸다(제어 주기 ~100 ms). 관측 shape이 계약으로 고정이라 **한 번 trace하면 세션 전체를 커버**한다 →
`0c0f094`가 청크 샘플러를 jit했고 기동 smoke가 ready 광고 **전에** 컴파일을 흡수한다.

**테스트 기준선(`4a66706`)**: **968 passed / 17 skipped / 1 xfailed** (actor venv)
+ jax opt-in 게이트 **25건**(`RUN_HIL_SERL_ACTUAL_BC=1` / `RUN_HIL_SERL_ACTUAL_FM=1`, hilserl venv).

---

## 7. 함정·한계

| 항목 | 내용 |
|---|---|
| **T4 mp4** | `Ctrl-C`(graceful)로 끝낼 때만 moov atom이 써져 **재생 가능**해진다. 강제 종료하면 h5는 남아도 영상은 못 연다. 그리고 카메라 warmup 3 s 이전 프레임은 버려지므로 **아주 짧은 rollout은 영상이 아예 없다** |
| **FM 액션 변동** | 확률적 정책이라 같은 장면에서 액션이 매번 다르다. **고장이 아니다** |
| **로컬 터널은 하나** | `50153`을 production/BC/FM 셋이 공유한다. 다른 T1을 먼저 `Ctrl-C`로 닫아야 한다. 서버 쪽 포트는 50053/50054/50055로 갈려 있어 충돌하지 않는다 |
| **controller proof** | `31d6567`부터 3회 재시도(2 s 간격) + "둘 다 inactive인데 RESET 자세는 정상"인 benign 조합에서 소스 컨트롤러 1회 재활성화. 놉: `HIL_PREPOSITION_PROOF_RETRIES` / `HIL_PREPOSITION_PROOF_RETRY_DELAY_S` / `HIL_PREPOSITION_AUTOACTIVATE` |
| **learner와 완전 분리** | learner가 꺼져 있어도 평가는 정상 동작한다. 반대로 평가는 learner의 replay·lineage에 **아무것도 남기지 않는다** |
| **성공 라벨** | MANUAL `MARK SUCCESS`가 유일하다. 판정 문장을 세션 전에 한 줄로 못 박고 끝까지 바꾸지 않는다(BC/FM A/B의 전제) |
| **T1 Ctrl-C** | 터널 + **자기가 띄운 그 정책 서버**를 함께 내린다. 세션 도중에 누르면 actor가 다음 RPC에서 죽는다. 종료 순서는 **T3 → T2 → T1** |
| **step timing 기본 OFF** | 켜지 않으면 아무 파일도 안 남고 평소 실행에 오버헤드도 없다. T1과 T3를 **각각** 켜야 하며, 한쪽만 켜면 그쪽 분해만 나온다 — **둘 다 켜야 `wire_ms`(순수 통신)가 계산된다** |
| **서버 쪽은 T1 재기동 필요** | 코드가 서버 runtime에 pull된 뒤 **T1을 새로 띄워야** `timing.jsonl`이 나온다. 살아 있는 옛 서버 프로세스를 그대로 쓰면 안 남는다(`inference.jsonl` 때와 같은 함정) |
| **`env_step_ms`는 sleep 포함** | 액터 기록의 `env_step_ms`에는 `env.step`의 **~100 ms 자체 페이싱 sleep**이 들어 있다. "환경 처리 비용"으로 읽으면 안 된다 |

---

## 8. 문서 지도

| 문서 | 언제 |
|---|---|
| **이 파일** | 전체 그림, 무엇을 실행할지 고를 때 (5분) |
| [`BC_DEPLOY_KO.md`](BC_DEPLOY_KO.md) | **BC 세션을 실제로 돌릴 때.** 단계별 절차·성공 표식·실패 대응·평가 프로토콜·금지사항 |
| [`FM_DEPLOY_KO.md`](FM_DEPLOY_KO.md) | **FM 세션을 돌릴 때.** BC 런북의 **델타**다 — BC를 먼저 읽는다 |
| [`POLICY_EVAL_CODE_MAP_KO.md`](POLICY_EVAL_CODE_MAP_KO.md) | 코드가 어떻게 얽혀 있는지, 기록 파일의 필드가 무엇인지 (엔지니어용) |
| [`../docs/testing/09_HIL_ACTOR_RUNBOOK.md`](../docs/testing/09_HIL_ACTOR_RUNBOOK.md) | 하드웨어·preflight·controller·실패 대응의 **정본**. 평가 스택은 이것을 그대로 쓴다 |
| [`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](DATA_AND_MODELS_JUNHYEONG_AI_KO.md) | 서버 호스트·GPU·데이터/모델 경로의 정본 |
