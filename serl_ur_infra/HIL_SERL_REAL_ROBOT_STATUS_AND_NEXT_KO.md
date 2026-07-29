# 실물 UR7e HIL-SERL 현재 상태와 다음 단계

> 기준: **2026-07-29 KST**, 첫 실제 production-model actor run 직후
>
> 실행 코드 기준선: **`ca19652` 이상**. 이후 `f109656`까지는 준비 상태를 기록한 문서
> descendant다. 다음 실행 전에는 laptop3와 Kanu의 **실행 코드 HEAD**를 다시 대조한다.
>
> 이 문서의 목적은 다음 세션이 과거의 "아직 actor를 실기에서 돌리지 않았다"는 상태에서
> 다시 시작하지 않도록, **마지막 실제 성공·현재 한계·다음 구현 방향·필요 CLI**를 한곳에
> 고정하는 것이다. Kanu learner 전체 옵션의 정본은
> [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md), laptop actor의 상세 정본은
> [09_HIL_ACTOR_RUNBOOK.md](../docs/testing/09_HIL_ACTOR_RUNBOOK.md)다.

---

## 1. 한 줄 결론

**실물 HIL-SERL 원형은 end-to-end로 구동됐다.** 실제 UR7e에서 policy/GELLO 제어권 전환,
실제 transition 전송, Kanu의 RLPD online update, 새 policy publish, 그리고 actor 예외 후
controller 자동 복귀까지 확인했다.

다만 **연속 운용은 아직 PASS가 아니다.** replay가 201개 쌓인 시점에 학습과 추론이 같은
GPU에서 경쟁하면서 `Step RPC`가 0.6초 deadline을 넘었다. 또한 현재 launcher는 최초
commissioning을 위해 시작 시 `ENGAGED`를 강제하므로, 최종 목표인 **policy-first HIL**의
운영 UX와는 다르다.

따라서 이 run은 다음처럼 부른다.

> **실물 production-model HIL-SERL first E2E smoke: 핵심 원형 PASS, 지속 운용/운영 UX PARTIAL**

---

## 2. 현재 시스템 구성

```text
laptop3                                             Kanu
  UR7e + ROS2 Humble                                  SAC policy inference
  GELLO EEF leader                                    RLPD online learner
  RealSense cam1(scene), cam2(wrist)   gRPC/SSH       reward classifier
  Robotiq 2F-85                       <---------->     replay/intervention pools
  deadman/intervention GUI
```

- laptop3는 센서 수집, 실시간 robot command, GELLO intervention을 담당한다.
- Kanu는 policy inference, reward classifier, RLPD learner를 담당한다.
- Kanu server는 `127.0.0.1:50053`, laptop3는 SSH local forwarding
  `127.0.0.1:50153 -> Kanu 127.0.0.1:50053`을 사용한다.
- 제어 루프는 10 Hz다. actor의 현재 `Step RPC` deadline은 0.6초다.
- reward 권위는 Kanu의 classifier다. laptop의 환경 reward는 이 배치에서 사용하지 않는다.

---

## 3. 2026-07-29 첫 실제 E2E 결과

### 3.1 laptop3에서 확인된 순서

1. `run_hil_preposition.sh`가 UR7e를 `RESET_JOINTS` 허용 범위 안에 놓고 proof marker를 만들었다.
2. `run_hil_actor.sh --dry-preflight --arm --deadman topic`이 다음을 모두 통과했다.
   - `/joint_states` 약 100 Hz
   - `/gello/joint_states` 약 30 Hz
   - cam1/cam2 약 30 Hz
   - gripper state 약 5 Hz
   - `scaled_joint_trajectory_controller=active`
   - `forward_position_controller=inactive`
   - command publisher 0개
   - valid preposition proof + live RESET pose
   - fresh `ENGAGED` heartbeat 3개
3. 실제 actor에서 STJC -> FPC strict handoff가 성공했다.
4. actor가 Kanu production model과 protocol v2로 연결됐다.
5. 사용자가 확인한 실제 제어권은 다음과 같았다.
   - `ENGAGE`: UR7e가 GELLO EEF 움직임을 따라감
   - `DISENGAGE`: UR7e가 자율 policy처럼 움직임
6. 이후 `Step RPC failed (DEADLINE_EXCEEDED)`로 actor가 rc=1 종료됐다.
7. armed launcher가 actor publisher 소멸을 기다린 뒤 FPC -> STJC로 자동 복귀했다.
   `controller cleanup PASS`가 실제 controller_manager에서 확인됐다.

Qt의 `QFontDatabase` 경고는 font asset 경고이며 RPC timeout의 원인이 아니다.
종료 시 `QObject::killTimer` 경고도 OpenCV/Qt GUI thread의 비정상 종료 정리 경고다.

### 3.2 Kanu에 남은 정량 증거

증거 root:

```text
/home/junhyeong/hil-serl-data/runs/cube_in_cup_real_20260729_120225
```

| 항목 | 최종 확인값 | 의미 |
| --- | ---: | --- |
| accepted online replay | **201** | 실제 로봇 transition이 Kanu ingress에 들어감 |
| intervention replay | **153** | GELLO가 실제 action을 대체한 transition |
| policy-executed transition | **48** | `201 - 153`; DISENGAGED 상태에서 policy가 실제 제어 |
| intervention ratio | **76.12%** | 이번 run은 commissioning 성격상 사람 제어 비중이 큼 |
| offline demo | **2,037** | 사람 승인 canonical demo pool |
| learner step | **102** | online RLPD update가 실제 수행됨 |
| gradient step | **204** | CTA 2:1에 따라 learner step당 2 gradient step |
| policy version | **2** | step 50에서 v1, step 100에서 v2 publish |
| learner step, actor 동시 구간 | 중앙값 약 **1.123 s** | inference/ingress와 learner가 함께 돌 때의 실측 |
| learner step, actor 종료 후 | 중앙값 약 **0.457 s** | learner 단독 steady-state 실측 |
| 첫 policy publish 경계 | **5.474 s** | learner step 50의 일회성 validation/smoke stall |

이 수치는 단순 연결 성공 이상의 증거다. 다음 전체 경로가 실제로 이어졌다.

```text
실물 관측 -> Kanu inference -> 로봇/GELLO action 실행 -> transition ACK
          -> replay/intervention routing -> RLPD update -> policy v1/v2 publish
```

policy v1/v2가 learner-side runtime에 publish된 것은 증명됐지만, timeout 전에 v1 action이
actor로 돌아와 실제 로봇에서 실행됐다는 per-RPC 증거는 없다. **publish 성공과 robot 수신
성공을 같은 항목으로 승격하지 않는다.**

`learner_step=102`인 이유는 `training_starts=100`에서 replay 201개가 들어왔기 때문이다.
현재 계약에서는 `201 - 100 + 1 = 102` update가 허용된다.

### 3.3 현재 살아 있는 process는 고정 사실이 아니다

문서 작성 시점에는 Kanu learner PID `159159`와 port `50053`이 아직 살아 있었다. actor가
죽은 뒤 learner가 backlog를 step 102까지 따라잡은 상태다. 다음 세션에서 PID가 같다고
가정하지 말고 아래 명령으로 다시 확인한다.

```bash
ssh kanu 'pgrep -af "run_rlpd_learner_server.py" || echo "no learner"'
ssh kanu 'ss -ltnp 2>/dev/null | grep ":50053" || echo "50053 unbound"'
ssh kanu 'tail -n 5 /home/junhyeong/hil-serl-data/runs/cube_in_cup_real_20260729_120225/logs/learner.jsonl'
```

이 lineage는 target step 5,000 전에 멈춰 있으므로 production checkpoint가 없다. learner를
종료하면 **RAM-only online replay와 현재 optimizer state는 복구할 수 없고**, JSONL과 run
directory만 남는다. 종료가 필요하면 새 server를 먼저 띄우지 말고 기존 process에 SIGINT를
한 번 보내 정상 종료한다. 임의의 `SIGKILL`이나 같은 checkpoint root의 두 번째 learner는 금지한다.

---

## 4. HIL 제어권 의미: 사용자의 관찰이 맞다

현재 핵심 wrapper의 의미는 다음과 같다.

| GUI 상태 | 실제 실행 action | transition 기록 |
| --- | --- | --- |
| `DISENGAGED` | Kanu policy action | `intervened=0` |
| `ENGAGED` + fresh GELLO | GELLO EEF action | `intervened=1`, policy action도 counterfactual로 보존 |
| `ENGAGED` + GELLO stale | HOLD | intervention으로 기록 |
| deadman heartbeat stale | actor fail-stop | policy로 자동 복귀하지 않음 |

따라서 사용자가 본 “ENGAGE하면 GELLO를 따라오고, 해제하면 자율적으로 움직인다”는 현상은
의도한 HIL action routing이 실물에서 작동했다는 직접 증거다.

### 4.1 지금 launcher와 최종 목표의 차이

핵심 wrapper는 이미 policy-first를 지원하지만 `run_hil_actor.sh --arm`은 controller switch
전에 `ENGAGED` heartbeat 3개를 강제한다. 초기 untrained policy의 첫 action이
`max_abs=0.99894`로 측정됐기 때문에 최초 commissioning에서 사용한 안전 gate다.

현재 실행 가능한 절차는 다음과 같다.

```text
ENGAGED로 actor 시작 -> controller handoff 완료 -> operator가 DISENGAGE
                      -> policy 제어 -> 필요할 때 다시 ENGAGE
```

목표로 하는 정상 실험 절차는 다음이다.

```text
fresh heartbeat + DISENGAGED로 시작 -> policy가 먼저 수행
                                    -> 필요할 때 사람 ENGAGE
                                    -> 놓으면 policy로 복귀
```

즉 **HIL 알고리즘의 action routing은 맞고, startup launcher UX만 commissioning 모드에
고정돼 있다.** 이 둘을 같은 문제로 취급하지 않는다.

---

## 5. Reward classifier와 episode 동작의 현재 사실

### 5.1 classifier는 켜져 있었다

실행 로그의 다음 줄은 sidecar scheduler가 활성화됐다는 뜻이다.

```text
classifier sidecar: every 5 steps when TCP speed < 0.05 m/s
(every step once p >= 0.05)
```

- 기본적으로 10 Hz 제어 루프의 5 step마다, 즉 최대 약 2 Hz로 채점한다.
- TCP speed가 0.05 m/s 미만일 때만 무크롭 cam1/cam2 JPEG sidecar를 보낸다.
- `p >= 0.05`가 나오면 매 step 채점으로 escalation한다.
- 현재 success threshold는 0.2, `success_confirmations=1`이다.
- success이면 server가 `reward=1`, `done=true`, `success=true`를 반환한다.

다만 이번 run의 영구 JSONL에는 per-transition `classifier_probability`가 남지 않는다.
learner metric의 batch reward에는 offline demo가 섞이므로 그것만 보고 online classifier가
몇 번 성공했다고 역산하면 안 된다. 따라서 이번 실기에서 증명된 것은 **sidecar/classifier
경로가 활성화된 production server로 transition이 들어갔다**까지이며, probability와 verdict의
시각적 정합은 아직 별도 검증이 필요하다.

### 5.2 현재 GUI에 보이지 않는 것

actor는 server outcome에서 이미 다음 값을 받는다.

- `classifier_evaluated`
- `classifier_probability`
- `classifier_threshold`
- `success`, `done`, `truncated`

하지만 현재 deadman GUI에는 이 값들이 연결돼 있지 않다. 운영자는 현재 GUI만 보고는
“이번 frame이 채점됐는지”, “성공 확률이 얼마인지”, “왜 episode가 끝났는지”를 알 수 없다.

### 5.3 현재 episode reset은 자동이지만 대기하지 않는다

`cube_in_cup`의 `MAX_EPISODE_LENGTH`는 100 step, 즉 10 Hz에서 10초다. episode는 다음 두
경로로 terminal이 된다.

- remote classifier success
- local 100-step episode limit

terminal ACK를 받은 actor는 `env.reset()`을 호출한다. `UR7eEnv.reset()`은 로봇을
`RESET_JOINTS`로 보내고, 그 직후 새 `BeginEpisode`를 호출한다.

```text
현재: SUCCESS 또는 100-step terminal -> HOME -> 즉시 다음 episode
```

사람이 실제 cube/cup scene을 다시 놓을 때까지 기다리는 상태와 GUI의 `Start/Resume` 버튼은
아직 없다. 현재 classifier도 “성공” 확률을 출력하는 모델이지 별도의 “실패 classifier”가
아니다. success가 아니면 100-step limit까지 진행되는 것이 현재의 failure/timeout에 가까운
의미다.

---

## 6. 무엇이 PASS이고 무엇이 아직 아닌가

| 기능 | 상태 | 판정 범위 |
| --- | --- | --- |
| 실제 sensor/canonical observation | **PASS** | actor preflight와 실제 loop에서 cam1/cam2/state 사용 |
| preposition proof + controller handoff | **PASS** | 실제 STJC -> FPC |
| policy가 실제 UR7e 제어 | **PASS** | 48 non-intervention transition + 사용자 육안 관찰 |
| GELLO intervention | **PASS** | 153 transition + 사용자 육안 관찰 |
| online replay/intervention routing | **PASS** | Kanu 201/153 |
| online RLPD update | **PASS** | learner 102 / gradient 204 |
| 새 policy publish | **PASS(learner-side)** | version 1, 2; actor 수신은 미확인 |
| 예외 후 controller 복귀 | **PASS** | FPC -> STJC cleanup 실기 확인 |
| startup JIT timeout 해소 | **PASS** | transition 100을 지나 steady update까지 진행 |
| classifier sidecar 활성화 | **PASS(배선)** | production sidecar 설정으로 실제 transition 수신 |
| online classifier verdict 육안/로그 검증 | **미완료** | GUI와 per-step 영구 로그가 없음 |
| policy-first startup UX | **미완료** | launcher가 시작 ENGAGED를 강제 |
| scene reset WAIT/Resume | **미구현** | reset 직후 다음 episode 시작 |
| 장시간 10 Hz 연속 운용 | **FAIL/PARTIAL** | replay 201 부근에서 0.6 s deadline 초과 |
| checkpoint/resume 실물 검증 | **미완료** | learner step 5,000 미도달 |

---

## 7. 남은 RPC timeout의 해석

이 timeout은 이전의 “transition 100에서 처음 JIT compile이 수십 초 걸린 문제”와 다르다.
startup update warm-up 덕분에 learner가 transition 100을 통과했고 step 102까지 정상 학습했다.

이번 run의 startup warm-up은 약 `45.89 s -> 38.28 s -> 386 ms`였고, 실제 첫 learner
step도 758 ms여서 과거의 약 46초 cold compile은 재발하지 않았다. 대신 actor와 learner가
동시에 돌던 learner step 1~46의 중앙값은 약 1.123초였고, actor가 사라진 뒤 step 51~102는
약 457 ms였다. 첫 publish인 learner step 50은 5.474초가 걸렸고, update 구성요소 약 634 ms를
제외한 약 4.84초가 최초 publish validation/smoke 경계에 쓰였다. 두 번째 publish는 약 456 ms였다.

replay는 publish 경계 전후로 200에서 201이 된 뒤 더 늘지 않았다. server는 transition을
exactly-once로 accept한 뒤 다음 action을 추론하므로, actor가 응답 deadline으로 종료돼도 마지막
transition 201이 server에 남을 수 있다. 현재 증거는 단순 네트워크 단절보다 **첫 publish의
일회성 stall + co-located 학습/추론 contention**을 강하게 가리킨다.

현재 로그만으로 timeout request가 다음 중 어느 phase에서 막혔는지는 확정할 수 없다.

- policy inference queue wait
- classifier sidecar inference
- online frozen-trunk encoding
- replay finalize/insert
- learner GPU update와의 scheduler contention

따라서 timeout을 단순히 0.6초보다 크게 늘려 숨기지 않는다. delayed action을 로봇이 뒤늦게
실행하는 것은 실시간 제어 문제를 해결하지 않는다.

---

## 8. 다음 개발 방향과 우선순위

### P0. 이번 run을 보존하고 실험 계보를 구분한다

- `cube_in_cup_real_20260729_120225`는 **first E2E smoke evidence**로 보존한다.
- 76.1% intervention이고 timeout으로 끝났으므로 최종 production learning 결과로 승격하지 않는다.
- 다음 코드 변경 후에는 새 run root를 만들고, 이 RAM-only lineage를 최종 실험과 섞지 않는다.
- 기존 learner가 살아 있는 동안 Kanu checkout을 pull하거나 같은 port/root에 다른 learner를
  띄우지 않는다.

### P1. steady-state RPC latency의 phase를 먼저 계측한다

최소한 다음 시간을 같은 `transition_id`와 함께 남긴다.

```text
RPC receive/queue wait
classifier decode + inference
online trunk encode
policy inference
transition finalize + replay insert
response serialization
learner sampling / critic update / full update
```

그 뒤 가장 작은 scheduling 변경으로 actor inference를 우선한다. 후보는 learner update 사이에
RPC를 처리할 수 있도록 pacing/lock 범위를 줄이는 것이며, 필요하면 Kanu의 다른 GPU/process로
inference를 분리한다. **계측 없이 UTD, timeout, classifier cadence를 동시에 바꾸지 않는다.**

Acceptance는 “평균”이 아니라 live actor의 plain/sidecar `Step RPC` p99가 deadline 안에 있고,
최소 수백 step 동안 timeout이 없는 것이다.

### P2. commissioning startup과 정상 policy-first startup을 분리한다

필요한 의미는 두 개뿐이다.

1. `commissioning`: 지금처럼 ENGAGED로 시작해 사람이 첫 action을 소유한다.
2. `policy-first`: fresh heartbeat를 요구하되 DISENGAGED 상태로 시작하고 policy가 첫 action을 소유한다.

deadman heartbeat stale 시 fail-stop, GELLO stale 시 HOLD, controller proof 같은 기존 동작은
그대로 유지한다. 최종 기본값을 무엇으로 둘지는 초기 policy/action scale 검증 후 정하되,
정상 HIL 실험 자체는 policy-first여야 한다.

### P3. operator episode state machine과 GUI를 만든다

목표 상태는 다음처럼 단순하게 유지한다.

```text
WAIT_SCENE_READY
   -- operator START --> POLICY_RUNNING

POLICY_RUNNING
   -- ENGAGE ---------> HUMAN_INTERVENTION
   -- SUCCESS --------> HOMING -> WAIT_SCENE_READY
   -- TIME_LIMIT -----> HOMING -> WAIT_SCENE_READY

HUMAN_INTERVENTION
   -- DISENGAGE ------> POLICY_RUNNING
   -- SUCCESS --------> HOMING -> WAIT_SCENE_READY

any state
   -- heartbeat/RPC fault --> FAULT + controller cleanup
```

GUI에 필요한 최소 표시는 다음이다.

- control owner: `POLICY / HUMAN / HOLD / FAULT`
- current episode/step
- classifier evaluated 여부, `p(success)`, threshold
- terminal reason: `SUCCESS / TIME_LIMIT / FAULT`
- `START/RESUME` 버튼

actor는 이미 server outcome을 갖고 있으므로 새 classifier를 만들 필요는 없다. actor의 outcome을
GUI가 읽을 수 있게 노출하고, `env.reset()` 뒤 `BeginEpisode` 전에 operator gate를 두는 것이
핵심이다.

### P4. classifier 실기 정합을 눈으로 검증한다

- success/failed scene에서 GUI probability를 확인한다.
- server outcome과 standalone viewer가 같은 장면에서 대체로 일치하는지 본다.
- cam1을 팔이 가리는 `take_21` 유형은 sidecar로 해결되지 않는다. false negative가 반복되면
  threshold를 무작정 낮추지 말고 cam1 배치를 먼저 조정한다.
- per-transition classifier verdict를 JSONL 또는 bounded actor artifact에 남겨 사후 검증 가능하게 한다.

### P5. 그 다음에 bounded production run과 checkpoint를 검증한다

- 짧은 실기용 `max_steps`/정상 종료 CLI는 편의 기능으로 추가할 수 있지만 P1~P3보다 낮은 우선순위다.
- 지속 운용이 안정된 뒤 새 lineage에서 learner step 5,000까지 진행한다.
- checkpoint 생성, 정상 shutdown, `--resume-latest`, 새 process의 policy version 복구를 검증한다.

---

## 9. 다음 세션용 CLI

아래는 **현재 코드 그대로 재현하는 CLI**다. P1~P3가 구현되기 전까지 actor 시작은 여전히
ENGAGED gate를 요구하며, controller handoff 뒤 operator가 DISENGAGE해야 policy가 제어한다.

### 9.1 정상 운용: 터미널 세 개

세 terminal 모두 같은 디렉터리에서 시작한다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
```

Terminal 1 — Kanu learner를 검증해 재사용하거나 없으면 시작하고, SSH tunnel을 소유한다.

```bash
./run_hil_server.sh
```

Terminal 2 — UR7e driver, Robotiq gripper, passive GELLO reader를 한 번에 소유한다.

```bash
./run_hil_hardware.sh
```

Terminal 3 — 두 카메라, HIL GUI, preposition, armed-readiness preflight, 실제 actor를 순서대로
실행한다.

```bash
./run_hil_session.sh
```

순서는 **1 → 2 → 3**이다. 이 세 wrapper는 기존 검증된 개별 명령을 묶을 뿐 controller proof,
토픽 rate probe, deadman gate, interactive `GO`를 건너뛰지 않는다.

### 9.2 Terminal 1의 소유 범위

`run_hil_server.sh`는 Kanu에서 다음을 fail-closed로 확인한다.

- learner entrypoint가 0개이거나 정확히 1개인지(2개 이상이면 거절)
- classifier/demo/ResNet SHA와 schema/model/reward 및 production 필수 CLI/env 계약이 정확한지
- 기존 learner가 health-ready인지
- 새 learner가 필요하면 GPU가 비어 있고 `MemAvailable`이 예상 feature ring + demo + reserve
  합계 이상인지

정확히 일치하는 learner가 이미 있으면 **그 process를 재사용**하고, 없으면 기본 physical GPU 5에
새 run root를 만들어 detached learner를 시작한다. 그 뒤 laptop의
`127.0.0.1:50153 -> Kanu 127.0.0.1:50053` tunnel과 실제 gRPC read-only probe를 연다.

```bash
# learner/계약 상태만 읽고 tunnel은 열지 않는다.
./run_hil_server.sh --check

# 기존 learner가 없을 때 사용할 GPU/run 이름을 지정한다.
./run_hil_server.sh --gpu 6 --run-id cube_in_cup_real_YYYYMMDD_HHMMSS

# 기존 healthy/initializing learner가 있으면 재사용하지 않고 거절한다.
./run_hil_server.sh --new-lineage --gpu 6 --run-id cube_in_cup_real_YYYYMMDD_HHMMSS
```

Terminal 1의 `Ctrl-C`는 **자기가 만든 SSH tunnel만 닫는다.** Kanu learner는 server-owned라
계속 살아 있다. learner까지 끝낼 때만 PID를 다시 눈으로 확인하고 Kanu에서 `SIGINT`를 한 번
보낸다. target 5,000 전 checkpoint가 없는 learner를 내리면 RAM replay/optimizer state는
복구되지 않는다.

### 9.3 Terminal 2의 재기동 의미

`run_hil_hardware.sh`는 중복 wrapper나 기존 UR/GELLO/gripper owner가 있으면 먼저 거절한다.
그 뒤 headless/no-RViz UR7e(STJC 초기 active, tool communication 24 V),
`/tmp/ttyUR`의 Robotiq, read-only GELLO publisher를 각각 독립 process group으로 띄우고 다음을
확인한다.

- `/joint_states >= 50 Hz`
- `/robotiq_gripper/position_percent >= 2 Hz`
- `/gello/joint_states >= 15 Hz`

명령을 먼저 확인하고 하드웨어에는 접촉하지 않으려면 다음을 쓴다.

```bash
./run_hil_hardware.sh --dry-run
```

충돌/protective stop/연결 해제 뒤에는 로봇 fault와 주변 원인을 해소하고 Terminal 3을 먼저
종료한다. Terminal 2에서 `Ctrl-C` 후 `[cleanup] complete`를 확인한 다음 같은 명령을 다시
실행한다. 카메라와 learner는 이 재기동에 포함되지 않는다.

### 9.4 Terminal 3의 interactive 절차

`run_hil_session.sh`는 다음 순서로 진행한다.

```text
cam1/cam2 READY -> HIL GUI -> preposition(필요하면 operator GO)
-> operator ENGAGED 확인 -> read-only armed preflight -> 실제 actor
```

현재 commissioning launcher는 controller handoff 전에 fresh `ENGAGED` heartbeat를 요구한다.
프롬프트가 나오면 GUI를 `ENGAGED`로 만들고 GELLO를 고정한 뒤 Enter를 누른다. actor가 뜬 뒤
policy 동작을 보려면 `DISENGAGE`, 개입하려면 `ENGAGE`, policy에 돌려주려면 다시
`DISENGAGE`한다. **DISENGAGE는 정지 명령이 아니라 policy 제어 복귀다.**

자주 쓰는 진단 옵션은 다음뿐이다.

```bash
# 첫 smoke처럼 classifier sidecar를 매 step 평가하는 지연 진단용
./run_hil_session.sh --classifier-sidecar-interval 1

# camera viewer만 생략
VIEW=false ./run_hil_session.sh

# 실제 actor/controller switch/preposition 없이 camera+GUI+read-only probe
./run_hil_session.sh --no-arm

# 아무 process도 띄우지 않고 조립될 명령만 출력
./run_hil_session.sh --no-arm --plan
```

actor가 정상 종료하거나 RPC/preflight 오류가 나거나 Terminal 3에서 `Ctrl-C`하면 이 wrapper가
자기가 띄운 camera/GUI를 정리한다. armed actor가 시작된 뒤에는 하위 `run_hil_actor.sh`가
FPC -> STJC controller 복귀를 책임진다. 종료 로그에서 다음을 확인한다.

```text
controller cleanup: strict switch forward_position_controller -> scaled_joint_trajectory_controller
controller cleanup PASS: scaled_joint_trajectory_controller=active, forward_position_controller=inactive
```

### 9.5 장애 진단과 learner 관찰

세 wrapper 내부의 T0~T6 개별 명령은
[09_HIL_ACTOR_RUNBOOK.md](../docs/testing/09_HIL_ACTOR_RUNBOOK.md)에 장애 진단용으로 보존한다.
정상 세션에서는 다시 여러 terminal로 풀어 실행하지 않는다.

Kanu의 현재 process/run root는 Terminal 1 READY 배너가 출력한다. 실행 중에는 그 run root의
`logs/learner.jsonl`에서 다음 event를 본다.

- `learner_process_ready`
- 첫 `learner_update` (`learner_step=1`, `gradient_step=2`)
- `policy_published` (`learner_step=50`, `policy_version=1`)
- `rlpd_learner_worker_fault` / `rlpd_learner_actor_service_fault` 부재

하드웨어 복귀가 의심되면 Terminal 2가 살아 있는 상태에서 별도 진단 shell로 확인한다.

```bash
source /opt/ros/humble/setup.bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash
ros2 control list_controllers
ros2 topic info -v /forward_position_controller/commands
```

---

## 10. 다음 세션이 기억해야 할 핵심 다섯 줄

1. 실제 HIL-SERL 원형은 이미 성공했다. “actor 실기 미실행” 단계로 돌아가지 않는다.
2. `ENGAGE=GELLO`, `DISENGAGE=policy`가 실물에서 확인됐다.
3. Kanu는 실제 transition 201개로 learner step 102, policy version 2까지 갔다.
4. 현재 blocker는 startup JIT가 아니라 **첫 publish stall + 동시 학습/추론 중 Step RPC 0.6초 timeout**이다.
5. 다음 제품 방향은 **policy-first startup + classifier/episode GUI + scene-reset WAIT/Resume**다.
