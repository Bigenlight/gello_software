# 실물 UR7e HIL-SERL 현재 상태와 다음 단계

> 기준: **2026-07-29 KST**, 첫 실제 production-model actor run 직후
> · **2026-07-30 KST**, 개입 손맛(intervention feel) 실기 검증 (§3A)
>
> 실행 코드 기준선: **`ca19652` 이상**. 이후 `f109656`까지는 준비 상태를 기록한 문서
> descendant다. §3A의 서브스텝 경로는 **`4197f5b` 이상**에서만 존재한다.
> 다음 실행 전에는 laptop3와 Kanu의 **실행 코드 HEAD**를 다시 대조한다.
>
> ⚠️ **두 날짜의 숫자를 섞지 마라.** §3은 07-29 E2E(learner·gRPC 포함), §3A는 07-30 개입
> 경로 격리 검증(learner·gRPC 없음)이다. 리그가 다르므로 한 표에 나란히 놓으면 안 된다.
> §6도 같은 이유로 07-29 표(§6)와 07-30 표(§6.1)를 분리해 두었다.
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

**추가 (2026-07-30).** 그 run에서 조작자가 보고한 "개입 중 팔이 빳빳하다"는 별개 문제였고,
개입 경로를 격리한 실기 검증까지 끝났다 → **§3A**. 이것은 §8의 P0~P5 중 무엇도 닫지 않았다
(개입 손맛은 애초에 P 목록에 없었고 `08_OPEN_GAPS.md` **G24**로 신규 등재된 항목이다).
대신 **P1을 더 뾰족하게** 만들었다 — 개입 부드러움의 나머지 절반이 G21에 종속됨이
정량화됐다.

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
- **`4197f5b` 이후(2026-07-30): 개입 중에는 그 10 Hz 창 *안에서* 관절 타깃이 30 Hz로
  갱신된다.** 전이 생성률은 그대로 10 Hz(1 스텝 = 1 transition)이고 정책 경로는
  bit-identical이다. 액추에이터 층만 바뀐 것이다 — §3A.
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

## 3A. 2026-07-30 개입 손맛(intervention feel) 실기 검증

> **절 번호를 유지하려고 `3A`로 넣었다.** 다른 문서가 이 파일의 절 번호로 링크한다
> (`HIL_SERL_KANU_RUNBOOK_KO.md:83` → §9). 4번 이후를 밀지 말 것.
>
> 아래 숫자는 **전부 2026-07-30 측정**이다. §3(07-29)의 숫자와 같은 표에 놓지 않는다 —
> 07-29 run에는 이 코드가 아예 없었다.

실행 코드: 커밋 **`4197f5b`**. 원인 규명과 오프라인 실측 5개 표는
[`04_HIL_INTERVENTION.md`](../docs/testing/04_HIL_INTERVENTION.md) **§9**, 갭 대장은
[`08_OPEN_GAPS.md`](../docs/testing/08_OPEN_GAPS.md) **G24**에 있다. 이 절은 **실기 판정만**
기록한다.

### 3A.1 무엇이 바뀌었나

원인은 게인도 필터도 가속도 제한도 아니라 **타깃 갱신 주기**였다. `env.step()`이 100 ms 창에서
관절 타깃을 한 번만 세팅하고 나머지를 잤기 때문에, 250 Hz 업샘플러가 매 창마다
"가속 → 도달 → 제동 → 정지"를 반복했다.

조치는 그 sleep을 **창 안에서 30 Hz로 리더를 다시 읽는 서브스텝 페이싱**으로 바꾼 것이다
(`ur_env/envs/ur7e_env.py:537-629` `_drive_intervention_substeps`). 지켜진 것:

- **env 스텝 주기 10 Hz와 "1 스텝 = 1 transition"은 그대로다.** 창 길이를 안 건드린다.
- 창당 총 변위는 `InterventionBudget`이 `ACTION_SCALE` **1스텝**으로 묶는다
  (`ur_env/envs/leader_stream.py:354` 클래스). 저장 액션 불변식이 살아 있는 이유다.
- 정책 경로는 `driver is None`으로 갈라져 **bit-identical**이다
  (`ur7e_env.py:492-498`, `:650-652`). 이번 변경은 **개입 경로 전용**이다.
- 30 Hz는 임의값이 아니라 리더 publish rate이자 백엔드 리더 캐시가 실제로 갱신되는 rate다
  (`config.py:183-188` `INTERVENTION`, 근거는 그 위 주석).

> 위 줄번호는 **`4197f5b` 기준**이다. 그 커밋이 `wrappers.py`/`ur7e_env.py`의 줄을 크게
> 밀었으니, 낡은 문서의 줄번호를 그대로 믿지 말고 심볼 이름으로 다시 찾아라.

### 3A.2 리그 — 3-CLI 운영 경로가 **아니다**

`tests/run_real_hil.py`로 검증했고 **learner도 gRPC 서버도 띄우지 않았다.** 정책이 zero
고정이라 G21(RPC 지연)이 끼어들 수 없고 **개입 경로만 격리**된다.

| | |
| --- | --- |
| 로봇 | UR7e 드라이버를 `initial_joint_controller:=forward_position_controller`로 **직접** 기동 |
| 리더 | `gello_publisher` (`/gello/joint_states`) |
| 데드맨 | `ros2_ur_ws/run_hil_gui.sh` (`/hil/deadman`, topic 데드맨) |
| 정책 | zero 고정 — 학습 정책·Kanu·gRPC 전부 없음 |
| 그리퍼·카메라 | OFF (러너 기본값) |

⚠️ **`run_hil_hardware.sh`를 그대로 쓰면 안 된다** — STJC로 띄운다. 이 러너는 **FPC가 active**
여야 하고 컨트롤러 전환 로직이 없다. 정확한 명령은 **§9.6**.

### 3A.3 실기 결과 (2026-07-30, 실제 UR7e)

세 run 모두 개입 구간 1개, `--deadman topic`, 그리퍼/카메라 OFF다.

| # | 시각 | run | 개입 스텝 | frame-map |
| --- | --- | --- | ---: | --- |
| 1 | 11:00 | DRY `--scale 0.5` | 144 | **FAIL(당시 판정)** → 지금 코드로 재채점하면 **SKIP** |
| 2 | 11:11 | DRY `--scale 1.0` | 272 | **PASS** 잔차 0.016 / alpha 1.005 / 표본 141 (포화 131 제외) |
| 3 | 11:16 | ARMED `--scale 1.0 --max-steps 150` | 120 | **PASS** 잔차 0.130 / alpha 0.983 / 표본 51 (포화 69 제외) |

**run 2와 run 3은 자동 채점 전체 PASS다.** 항목별로:

| 항목 | run 1 (DRY 0.5) | run 2 (DRY 1.0) | run 3 (ARMED 1.0) |
| --- | --- | --- | --- |
| anchor-latch | PASS `0.000e+00` m | PASS `0.000e+00` m | PASS `0.000e+00` m |
| gain-latch | PASS `0.000e+00` | PASS | PASS |
| action-exec `dp_ratio` 중앙값 | **1.000** (표본 138) | **1.000** (272) | **1.000** (120) |
| held-rate | **0.0 %** | **0.0 %** | **0.0 %** |
| `substeps` | 개입 144창 **전부 2** | 272창 **전부 2** | 120창 **전부 2** |
| `governed` | **0** | **0** | **0** |
| 포화 `BUDGET_EXHAUSTED` | 86/144 = **59.7 %** | 131/272 = 48.2 % | 69/120 = 57.5 % |
| 스텝 주기 중앙값 | 101 ms | 101 ms | 101 ms |
| 조작자 주관 확인 | — | — | **"손맛 양호"** |

읽는 법:

- **`substeps=2`가 전부** = 창당 타깃 **3회** 갱신(`_apply_action`의 첫 타깃 + 서브스텝 2회).
  30 Hz·100 ms 창의 설계값이고, `substeps=0`인 창이 **하나도 없다** = 서브스텝 경로가 실제로
  돌았다는 증거다.
- **`governed=0`이 전부** = 창 예산(scale 1.0에서 0.0125 m)이 governor 캡(0.0150 m)보다
  타이트해서 **예산이 먼저 묶는다.** G24 리뷰 노트가 예상한 그대로다. 같은 커밋이 처음 드러낸
  대각 이동 상시 절삭(2축 0.849배, 3축 0.693배)은 **이 경로에서는 관측되지 않았다.**
- **포화 ≠ HOLD.** `BUDGET_EXHAUSTED`가 48~60 %인데 `held`는 0 %다. 예산이 소진된 창도
  **예산만큼은 정확히 실행**했고, `dp_ratio` 중앙값 1.000이 그 증거다 — 즉
  **"저장 액션 == 실행 액션" 불변식이 포화 상태에서도 성립한다.**
- **스텝 주기 101 ms** = 서브스텝 페이싱이 10 Hz 창을 늘리지 않았다.
- 조작자 확인은 **주관 보고이고 계측이 아니다.** 계측된 것은 위의 숫자들뿐이다.

#### run 1의 FAIL은 시스템 결함이 아니라 측정 아티팩트였다

그 run에서 리더가 앵커에서 최대 **74.68 cm** 나갔고 로봇 명령은 **23.27 cm**만 나갔다.
`--scale 0.5`의 예산이 6.25 cm/s인데 사람이 그보다 훨씬 빨리 움직였고, 표본의 **59.7 %**가
포화됐다. 포화 창에서는 명령이 리더의 순간 델타 방향이 아니라 **누적 오차 방향**으로 가고,
사람이 나갔다 되돌아오면 L(리더 누적)-R(로봇 누적) 관계가 직선이 아니라 히스테리시스 루프가
된다. 판정식의 `alpha`는 래그의 **크기**는 흡수하지만 **방향 발산**은 흡수하지 못한다.

| 같은 CSV, 표본만 다르게 | alpha | 상대잔차 | 판정 |
| --- | ---: | ---: | --- |
| 전체 144표본 | 0.227 | 0.776 | FAIL |
| 비포화 58표본 | 0.992 | 0.058 | (기준 0.15 안쪽) |

그래서 `4197f5b`가 frame-map 판정에서 포화 표본을 제외한다.

📌 **이 수정은 FAIL을 PASS로 바꾼 것이 아니라 SKIP으로 바꿨다.** 같은 CSV를 지금 코드로
재채점하면(2026-07-30 확인) 비포화 58표본이 개수 게이트(20)는 넘지만 축 여기가
**1.4/3.6/1.4 cm**로 축당 2 cm 게이트에 미달해 **SKIP**이다. PASS는 run 2·3이 **다시 천천히
움직여서** 받은 것이다. 조작 지침은 §9.6에 있다.

> CSV 3개는 리포 밖(에이전트 작업 tmp)에 있다 — `/home/laptop3/.claude/jobs/5dfbc13c/tmp/`의
> `hil_dryrun.csv`(run 1) / `hil_dry2.csv`(run 2) / `hil_armed1.csv`(run 3).
> **사라질 수 있다.** 재현 절차가 §9.6에 있으므로 없어지면 다시 만든다.

### 3A.4 오프라인 회귀

```bash
cd /home/laptop3/gello_software/serl_ur_infra
env -u PYTHONPATH \
PYTHONPATH="/home/laptop3/gello_software/ros2_ur_ws/install/ur_gello_bringup/lib/python3.10/site-packages:/home/laptop3/gello_software/third_party/hil-serl/serl_launcher" \
/home/laptop3/venvs/gello-hil-actor/bin/python -m pytest tests -q -p no:anyio
```

📌 실측(2026-07-30): **`579 passed, 11 skipped`**. 07-29 기준선 **497**에서 신규 **82**개
(`test_leader_stream.py` 28 / `test_governor_dt.py` 38 / `test_intervention_substeps.py` 16).
**passed 수를 볼 것** — `serl_launcher`가 PYTHONPATH에서 빠지면 조용히 줄고 skip 사유가
거짓말을 한다.

### 3A.5 이 작업이 닫지 **않은** 것

1. **개입 부드러움의 나머지 절반은 G21 종속이다.** 서브스텝은 100 ms 창 **안**만 채운다.
   창 **사이**(gRPC 왕복)에는 타깃 갱신이 없고, 그 간격이 `target_stale_s = 0.30`을 넘기면
   업샘플러가 **정상 안전 동작으로** HOLD한다. 오프라인 실측(리더 등속 0.15 rad/s):
   주기 **0.700 s**(첫 publish 5.474 s급 stall)에서 관절 완전정지·HOLD **51.5 %**, 리더 속도
   추종 **66 %**. `target_stale_s`를 올려 없애는 것은 `04` §9.3(b)가 실측으로 금지한다
   (정지 17.5 % → 54.5 %). **→ §8 P1이 유일한 경로다.**
2. **actor(gRPC) 경로에서의 서브스텝은 미검증이다.** 07-30 검증은 정책 zero + learner 없음
   이었다. `scripts/run_remote_rlpd_actor.py`로 서브스텝 경로를 돌린 실기 run은 **아직 없다.**
3. **mock RViz 개입 루프는 여전히 미실행**(`04` §3). 검증이 오프라인 → 실기로 바로 갔다.
4. **조작자는 `substeps`/`governed`/포화를 볼 수 없다.** CSV에만 있다 → §8 P3.
5. 07-29의 미완료 항목(classifier verdict GUI, policy-first startup, scene reset WAIT,
   장시간 연속 운용)은 이 작업과 무관하게 **그대로**다.

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

아래 표는 **2026-07-29 첫 E2E** 기준이다. 2026-07-30 개입 손맛 판정은 리그가 다르므로
(learner·gRPC 없음) **§6.1에 따로** 둔다 — 두 날짜를 한 표에 섞지 않는다.

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

### 6.1 2026-07-30 개입 손맛 판정 (§3A)

리그: `tests/run_real_hil.py`, 정책 zero, **learner·gRPC 없음**, FPC 직접 기동.
근거 숫자는 전부 §3A.3이고 **§6 표의 07-29 숫자와 섞지 않는다.**

| 기능 | 상태 | 판정 범위 (전부 2026-07-30) |
| --- | --- | --- |
| 개입 중 창 안 30 Hz 리더 재추종(서브스텝) | **PASS(개입 경로 격리)** | DRY 1.0 / ARMED 1.0 자동 채점 전체 PASS. `substeps=2`가 개입 창 전부 |
| 개입 앵커·gain 래치 | **PASS** | 세 run 모두 구간 내 변동 `0.000e+00` |
| 개입 좌표계 = 단위행렬 | **PASS** | 잔차 0.016(DRY 1.0, 표본 141) / 0.130(ARMED 1.0, 표본 51). 포화 표본 제외 판정 |
| "저장 액션 == 실행 액션" (포화 포함) | **PASS** | 세 run 모두 `dp_ratio` 중앙값 1.000, held 0 %. 포화 창도 예산만큼 정확히 실행 |
| 서브스텝이 10 Hz 창을 늘리지 않음 | **PASS** | 스텝 주기 중앙값 101 ms |
| 예산이 governor보다 먼저 묶음 | **PASS** | `governed=0`이 개입 창 전부 (예산 0.0125 m < 캡 0.0150 m) |
| 조작자 손맛 | **PASS(주관 보고)** | ARMED 1.0에서 "손맛 양호". **계측이 아니다** |
| 오프라인 회귀 | **PASS** | `579 passed / 11 skipped` (신규 82) |
| actor(gRPC) 경로의 서브스텝 | **미검증** | 07-30 리그에 learner/gRPC가 없다. `run_remote_rlpd_actor.py`로는 아직 안 돌렸다 |
| mock RViz 개입 루프 | **미실행** | `04` §3 — 오프라인에서 실기로 바로 갔다 |
| 창 **사이**(RPC 구간) 부드러움 | **FAIL / G21 종속** | 주기 0.700 s에서 HOLD 51.5 %, 리더 속도 추종 66 % (오프라인 실측, `04` §9 표 1) |
| `substeps`/`governed`/포화의 운영 가시성 | **미구현** | CSV에만 있다 → §8 P3 |

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

> **2026-07-30 재검토 결과: P0~P5 중 닫힌 항목은 없다.** 이날의 개입 손맛 작업(§3A)이 닫은
> 것은 P 목록에 없던 별개 갭(`08_OPEN_GAPS.md` **G24**)이다. 순서도 그대로 유지한다.
> 바뀐 것은 두 가지다.
>
> - **P1이 더 뾰족해졌다.** 개입 부드러움의 나머지 절반이 G21에 종속됨이 정량화됐고,
>   그래서 P1은 이제 "actor가 죽지 않게 한다"에 더해 **"조작자 손에 직접 느껴지는 비용"**
>   까지 근거로 갖는다. 여전히 최우선이다.
> - **P3에 표시 항목이 늘었다** — `substeps` / `governed` / 포화(`BUDGET_EXHAUSTED`)는
>   이제 `info`에 있지만 조작자는 볼 수 없다.
>
> 반대로 **"개입이 빳빳하다"는 항목은 P 목록에서 찾지 말 것** — 애초에 없었다. 그 항목의
> 실기 판정은 §6.1이고, 남은 부분은 P1 안으로 흡수됐다.

### P0. 이번 run을 보존하고 실험 계보를 구분한다

- `cube_in_cup_real_20260729_120225`는 **first E2E smoke evidence**로 보존한다.
- 76.1% intervention이고 timeout으로 끝났으므로 최종 production learning 결과로 승격하지 않는다.
- 다음 코드 변경 후에는 새 run root를 만들고, 이 RAM-only lineage를 최종 실험과 섞지 않는다.
- 기존 learner가 살아 있는 동안 Kanu checkout을 pull하거나 같은 port/root에 다른 learner를
  띄우지 않는다.
- **2026-07-30 개입 검증(§3A)은 새 lineage가 아니다** — learner가 없었으므로 Kanu run root도,
  transition도 만들지 않았다. 그 CSV 3개를 07-29 run root의 증거와 같은 묶음으로 취급하지 않는다.

### P1. steady-state RPC latency의 phase를 먼저 계측한다 (여전히 최우선)

**2026-07-30 갱신: 이 항목의 근거가 하나 더 늘었다.** G21은 actor를 죽이는 문제로만 적혀
있었지만, actor가 **살아 있는 동안에도 조작자의 손에 직접 나타난다.** RPC 왕복이 `env.step`
**밖**에 있어서 250 Hz 업샘플러가 보는 타깃 갱신 주기가 `100 ms + RPC`가 되고, §3A의 서브스텝은
100 ms 창 **안**만 채우므로 **창 사이 구간은 구조적으로 못 덮는다.**

| 타깃 주기 | 유래 | 관절 완전정지·HOLD | 리더 속도 추종 |
| --- | --- | ---: | ---: |
| 0.158 s | RPC p50 58 ms | 32.5 % | 100 % |
| 0.350 s | learner 경합 (step 중앙값 약 1.12 s의 일부) | 17.5 % | 99 % |
| **0.700 s** | **첫 publish 5.474 s급 stall** | **51.5 %** | **66 %** |

(2026-07-30 오프라인 실측, 리더 등속 0.15 rad/s. 전체 5개 표와 재현 스크립트는
[`04_HIL_INTERVENTION.md`](../docs/testing/04_HIL_INTERVENTION.md) §9.)

⚠️ 첫 행의 `RPC p50 58 ms`는 G16의 **열화된 링크**(13 Mbit/s) 기준이다. 07-29 실측 빠른 링크
(p50 약 12~22 ms)에서는 주기가 약 0.112~0.122 s다. 반면 **0.350 s와 0.700 s 행은 링크가 아니라
learner 경합과 첫 publish stall에서 오므로 네트워크를 개선해도 남는다** — 그게 P1이 P1인 이유다.

이 HOLD는
`target_stale_s = 0.30`을 넘긴 **정상 안전 동작**이므로 필터·타깃 rate 상향·외삽 어느 것으로도
없앨 수 없고, `target_stale_s`를 올려 없애는 것은 `04` §9.3(b)가 실측으로 금지한다.
**즉 P1이 유일한 경로다.**

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

**2026-07-30 추가 acceptance:** 위 phase 로그에 **연속된 두 `env.step` 사이의 실측 간격**
(= 업샘플러가 보는 타깃 갱신 주기)을 함께 남긴다. RPC p99만으로는 조작자 비용을 못 본다 —
위 표의 판정 축이 바로 그 주기다. 개입 중 `intervention_substeps`와 `reject_reason`
(`BUDGET_EXHAUSTED` / stale HOLD)을 같이 남기면 그 창이 왜 끊겼는지가 사후에 구분된다.

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

**2026-07-30 추가 후보 — 개입 품질 표시.** `4197f5b` 이후 `info`에 다음이 들어 있지만 조작자는
볼 수 없다. 실기에서는 CSV를 사후에 열어야만 확인됐다(§3A.3).

- `intervention_substeps` — 이번 창에서 타깃이 몇 번 갱신됐나. **0이면 서브스텝 경로가 아예
  돌지 않은 것**이므로 "왜 갑자기 빳빳해졌나"에 즉답이 된다.
- `governed` / `governed_scale` — governor rate cap이 요청을 깎았나(대각 이동에서 특히).
- `reject_reason == BUDGET_EXHAUSTED` — 창 예산 소진(= 리더가 예산보다 빠름). **HOLD가 아니다.**
  실기에서 개입 창의 48~60 %가 여기 걸렸는데 조작자에게는 아무 신호도 없었다.

우선순위는 위 5개보다 낮다. 다만 이 셋은 **새로 계산할 것이 없고** `info`에서 그대로 읽어
쓰기만 하면 된다.

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
- **2026-07-30 추가:** 그 bounded run에서 **actor(gRPC) 경로의 개입 서브스텝을 함께 재확인한다.**
  §3A는 정책 zero + learner 없음으로 검증했으므로 `run_remote_rlpd_actor.py` 경로의
  실기 판정은 아직 없다. P1 계측을 켜고 도는 run이면 별도 세션 없이 같이 확인된다 —
  개입 창의 `intervention_substeps`가 2로 유지되는지, `dp_ratio`가 1.0 부근인지만 본다.

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

### 9.6 개입 손맛 검증 절차 (§3A 재현) — **위 3-CLI와 다른 경로다**

> 🛑 **§9.1~9.5의 3-CLI 워크플로우를 여기에 섞지 마라.** `tests/run_real_hil.py`는
> **FPC가 active인 상태**를 요구하고 **컨트롤러 전환 로직이 없다.** 반면
> `run_hil_hardware.sh`는 **STJC**로 띄운다(§9.3). STJC 리그에 이 러너를 붙이면:
> DRY는 `!! 구독자가 없다` 경고만 찍고 **그대로 진행하며**(발행 자체를 안 하니 CSV는 정상적으로
> 채워진다), `--arm`은 preflight에서 **거부**된다. 최악은 그 사이 —
> **구독자 수가 0이 아니어도 FPC가 active라는 보장은 없다.** 그 경우 명령이 아무데도 가지 않고
> 러너는 조용히 성공한 것처럼 보인다. 그래서 T1에서 `ros2 control list_controllers`로
> **직접 확인**한다.
>
> learner도 gRPC 서버도 **띄우지 않는다.** 정책이 zero 고정이라 G21이 끼어들지 않는다.

**터미널 4개.** 모두 다음 선행이 필요하다.

```bash
source /opt/ros/humble/setup.bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash
```

Terminal 1 — UR7e 드라이버를 **FPC로 직접** 기동한다(teleop 브리지 없이).

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur7e robot_ip:=192.168.10.11 \
    initial_joint_controller:=forward_position_controller \
    headless_mode:=true launch_rviz:=false
```

```bash
# 확인 (별도 shell). FPC가 active여야 한다.
ros2 control list_controllers | grep forward_position_controller
ros2 topic info -v /forward_position_controller/commands   # Publisher 0 / Subscription 1
```

`headless_mode:=true`는 펜던트 **REMOTE** 모드를 요구한다. FPC는 `ForwardCommandController`라
**첫 `commands` 메시지가 올 때까지 아무것도 쓰지 않으므로** active로 띄워 놔도 팔은 가만히 있다.

Terminal 2 — GELLO 리더만 발행한다(로봇 브리지 없이).

```bash
GELLO_REPO_ROOT=$HOME/gello_software \
ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \
  $HOME/gello_software/ros2_ur_ws/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml
```

Terminal 3 — 데드맨 GUI.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws && ./run_hil_gui.sh
```

러너를 띄우기 전에 GUI는 **DISENGAGED**여야 한다 — ENGAGED 상태면 러너가 preflight에서
`데드맨이 이미 ENGAGE 상태다`로 거부한다. 반대로 러너가 도는 중에 GUI를 닫으면 0.5 s 뒤
stale 예외로 러너가 **종료**되고, 그 틱에 정책 fallback 명령은 나가지 않는다.
**이것은 §9.4의 commissioning gate(시작 시 `ENGAGED` 3개 강제)와 반대다** — 여기서는
DISENGAGED로 시작한다.

Terminal 4 — 러너. **시스템 `python3`가 맞다**(이 러너는 gRPC를 import하지 않는다).
`--arm`은 정확히 `ARM`을 타이핑해야 진행되고 그 뒤 3초 카운트다운이 있다.

```bash
cd /home/laptop3/gello_software/serl_ur_infra

# 1) DRY — 로봇 안 움직임. CSV/좌표계/앵커/서브스텝을 먼저 본다.
python3 tests/run_real_hil.py --scale 1.0

# 2) ARMED — 팔이 실제로 움직인다. E-STOP 손에.
python3 tests/run_real_hil.py --arm --scale 1.0 --max-steps 150
```

> ⚠️ **`PYTHONPATH`를 덮어쓰지 마라.** 위 두 `source`면 충분하다. `PYTHONPATH="..."`로
> 덮어쓰면 rclpy가 사라져 `RuntimeError: rclpy not available`로 죽는다 —
> **2026-07-30에 실제로 이걸로 한 번 실패했다.** 뭔가를 더해야 하면 반드시
> `PYTHONPATH="...:$PYTHONPATH"`로 **이어붙인다.** (`-m pytest`용 `env -u PYTHONPATH ...`
> 명령을 여기 복사해 오는 것이 바로 그 실패 경로다.)

**조작 지침 — 이걸 안 지키면 frame-map이 SKIP/FAIL로 나온다.** `X → Y → Z` **한 축씩**
5~10 cm를 **초당 3~5 cm로 천천히**, 축을 섞지 말고 움직인다.

- 창 예산은 `ACTION_SCALE[0] * HZ * scale` = **scale 1.0에서 12.5 cm/s**, 0.5에서 6.25 cm/s다.
  이보다 빨리 움직이면 그 창은 `BUDGET_EXHAUSTED`로 포화되고, 포화 표본은 frame-map 판정에서
  **제외된다**(§3A.3). 빠르게 크게 흔들면 판정 표본만 사라진다 — run 1이 정확히 그래서 SKIP이다.
- 판정 게이트는 비포화 표본 **20개 이상** + **축당 여기 2 cm 초과**다. 한 축씩 5~10 cm면 넉넉하다.
- 채점표(anchor-latch / gain-latch / frame-map / action-exec / held-rate)와 각 컬럼의 의미는
  `tests/run_real_hil.py` 파일 상단 docstring이 정본이다.

종료 후 볼 것: 콘솔 요약의 **전체 PASS**, `substeps`가 개입 창에서 **2**인지, `held-rate` 0 %,
`dp_ratio` 중앙값 1.0 부근. `--csv`로 경로를 주면 같은 이름의 `.meta.json`에 3층 설정 전체가
함께 남는다.

---

## 10. 다음 세션이 기억해야 할 핵심 여섯 줄

1. 실제 HIL-SERL 원형은 이미 성공했다. “actor 실기 미실행” 단계로 돌아가지 않는다.
2. `ENGAGE=GELLO`, `DISENGAGE=policy`가 실물에서 확인됐다.
3. Kanu는 실제 transition 201개로 learner step 102, policy version 2까지 갔다.
4. 현재 blocker는 startup JIT가 아니라 **첫 publish stall + 동시 학습/추론 중 Step RPC 0.6초 timeout**이다.
5. 다음 제품 방향은 **policy-first startup + classifier/episode GUI + scene-reset WAIT/Resume**다.
6. 개입 손맛은 **2026-07-30에 실기 PASS**다(§3A, 커밋 `4197f5b`) — 단 정책 zero·gRPC 없는
   **격리 리그**였고, 창 **사이**(RPC 구간)의 부드러움은 여전히 **G21 종속**이다. 그래서
   **P0~P5는 하나도 닫히지 않았고 P1만 더 뾰족해졌다.**
