# 실물 UR7e HIL-SERL 현재 상태와 다음 단계

> 기준: **2026-07-29 KST**, 첫 실제 production-model actor run
> · **2026-07-30 KST**, 개입 손맛 실기 검증과 schema 3 MANUAL learner 재기동 (§3A, §3B)
>
> laptop3 기준선: **이 문서를 포함한 최신 branch tip**. §3A의 30 Hz
> 개입 서브스텝은 `4197f5b`, 매 세션 Enter/GO 제거는 `e86edd5`에서 들어왔다. Kanu의 현재 schema 3 stage는
> `/home/junhyeong/gello_software_hil_schema3_stage_20260730` @ `c9c30c3e…`이고, 안정 링크
> `/home/junhyeong/gello_software_hil_current`가 그 checkout을 가리킨다. 다음 실행 전에는
> laptop3와 Kanu의 **실행 코드 HEAD와 실제 process argv를 다시 대조한다.**
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

다만 **장시간 연속 운용은 아직 PASS가 아니다.** 07-29 run은 replay 201개 부근에서 당시
`Step RPC` 0.6초 deadline을 넘었다. 현 wrapper는 각 RPC deadline을 **1.5초**, 요청 생성부터
응답 수락까지의 최대 age를 **2.0초**로 제한하지만, 이것은 늦은 action을 거절하는 경계이지
10 Hz 제어가 매번 100 ms 안에 끝난다는 보장은 아니다.

운영 UX는 이제 policy-first state machine으로 정리됐다. controller handoff까지만 fresh
`ENGAGED` heartbeat 3개를 요구하고, HOME 뒤 GUI의 `START / NEXT ITERATION`이 deadman을
해제한 다음 episode를 시작한다. terminal 뒤에는 바로 HOME으로 가지 않고 조작자 승인을 기다린다.

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
- transition/control 목표는 10 Hz다. 현재 각 RPC deadline은 **1.5초**, 응답 최대 age는
  **2.0초**다. 둘 다 fail boundary이며 10 Hz 달성 증명은 아니다.
- **`4197f5b` 이후(2026-07-30): 개입 중에는 그 10 Hz 창 *안에서* 관절 타깃이 30 Hz로
  갱신된다.** 전이 생성률은 그대로 10 Hz(1 스텝 = 1 transition)이고 정책 경로는
  bit-identical이다. 액추에이터 층만 바뀐 것이다 — §3A.
- reward 권위는 Kanu다. MANUAL에서는 classifier를 계속 계산·표시·기록하되 조작자의
  `MARK SUCCESS`만 성공 terminal/reward를 승인한다. AUTO에서는 classifier의 엄격한
  `p > 0.5`가 성공을 승인한다. laptop의 환경 reward는 이 배치에서 사용하지 않는다.

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

### 3.3 07-29 종료 직후 process 기록 (현재값 아님)

07-29 문서 작성 시점에는 Kanu learner PID `159159`와 port `50053`이 아직 살아 있었다. actor가
죽은 뒤 learner가 backlog를 step 102까지 따라잡은 상태다. 다음 세션에서 PID가 같다고
가정하지 않는다. 이 process는 §3B의 schema 3 learner와 다르다. 현재값은 아래 명령 또는
`run_hil_server.sh --check`로 다시 확인한다.

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

📌 최신 실측(2026-07-30): **`614 passed, 11 skipped, 1 xfailed`**. 여기에는 개입
서브스텝뿐 아니라 operator session/actor episode gate 회귀도 포함된다.
**passed 수를 볼 것** — `serl_launcher`가 PYTHONPATH에서 빠지면 조용히 줄고 skip 사유가
거짓말을 한다.

GUI ROS package는 clean build 뒤 **`461 passed`**이며, 격리 `ROS_DOMAIN_ID`에서 actor가
첫 WAIT를 보낸 뒤 GUI를 늦게 띄워도 0.5 s 재발행을 수신하고 `/hil/scene_ready` Trigger가
정확히 한 번 승인되는 DDS 왕복까지 통과했다. 이는 로봇 없는 통합 검증이며 실물
success→HOME→WAIT 연속 동작을 PASS로 승격시키지는 않는다.

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
5. 07-29의 미완료 항목 중 classifier verdict GUI, policy-first episode 시작, scene reset
   WAIT/Resume는 **코드·headless/DDS 검증까지 구현됐다**. 실물 success 연속 동작과 장시간
   연속 운용은 아직 남았다(§8 P2~P5).

## 3B. 2026-07-30 schema 3 MANUAL learner 현재 스냅샷

이 절은 2026-07-30 약 19:55 KST의 읽기 전용 관측이다. PID와 카운터는 불변값이 아니므로
다음 세션에서는 `run_hil_server.sh --check`와 READY 배너를 다시 본다.

| 항목 | 관측값 |
| --- | --- |
| learner | PID `1112465`, Kanu `127.0.0.1:50053` |
| checkout | `/home/junhyeong/gello_software_hil_schema3_stage_20260730` @ `c9c30c3e…` |
| stable link | `/home/junhyeong/gello_software_hil_current` -> 위 schema 3 stage |
| run root | `/home/junhyeong/hil-serl-data/runs/cube_in_cup_manual_schema3_thr05_20260730_1715` |
| 계약 | transport protocol `2`, transition schema `3`, threshold `0.5` |
| model/reward | `hil-serl-hybrid-sac-resnet10-trunk-cache-v1` / `cube-in-cup-all3-ckpt150+sidecar-v1` |
| 당시 카운터 | online replay `400`, intervention `225`, learner `301`, gradient `602`, policy version `6` |
| checkpoint | 아직 없음; checkpoint period 5,000 전이며 `.learner-writer.lock`만 존재 |

이 run은 schema 3 transition을 실제로 수신하고 학습했다. schema 3에서 transition meta는
`auto_success`와 `operator_success`를 함께 운반한다. 다만 observation은 계속 **canonical
observation schema v2**이고, `/hil/actor_status` JSON은 별도의 **status schema v2**다. 세 번호를
하나의 schema로 섞어 말하지 않는다.

옛 `/home/junhyeong/gello_software_hil` checkout은 schema 2/threshold 0.2 코드이며 현재 learner가
아니다. 현재 `run_hil_server.sh`는 stable link를 canonicalize해 schema 3 stage를 검증한다.

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

### 4.1 현재 launcher와 episode state machine

정상 경로는 terminal 세 개에서 실행하는 **3-CLI**다.

```text
Terminal 1  run_hil_server.sh    Kanu learner 검증/재사용 + SSH tunnel
Terminal 2  run_hil_hardware.sh  UR7e + gripper + passive GELLO reader
Terminal 3  run_hil_session.sh   cameras + compact HIL GUI + preposition + actor
```

`run_hil_session.sh`가 GUI의 fresh `ENGAGED` heartbeat를 최대 120초 동안 polling한다. 예전처럼
조작자가 Enter를 눌러 다음 단계로 보내는 프롬프트는 없다. 그래도 실제 gate가 약해진 것은 아니다.
`run_hil_actor.sh --arm`이 read-only preflight에서 heartbeat 3개를 확인하고 controller switch
직전에 다시 3개를 확인한다. stale이면 actor는 fail-stop하며 policy로 자동 복귀하지 않는다.

handoff 뒤 첫 episode와 매 terminal 뒤 순서는 다음과 같다.

```text
startup: HOMING -> HOME -> WAIT_SCENE_READY
         -> GUI START / NEXT ITERATION -> BeginEpisode -> POLICY_RUNNING

terminal: SUCCESS / TRUNCATED / EPISODE_LIMIT
          -> WAIT_HOME_APPROVAL (terminal pose HOLD)
          -> GUI APPROVE HOME — ROBOT WILL MOVE
          -> HOMING -> HOME -> WAIT_SCENE_READY
          -> scene를 정리한 뒤 GUI START / NEXT ITERATION
          -> BeginEpisode -> POLICY_RUNNING
```

개입 중에는 `HUMAN_INTERVENTION`, deadman/GELLO/RPC 이상에는 `HOLD` 또는 `FAULT`로 간다.
즉 commissioning handoff와 policy-first episode가 분리됐고, terminal 뒤 HOME과 scene 재배치도
각각 별도의 조작자 승인 단계다.

---

## 5. Reward classifier와 episode 동작의 현재 사실

### 5.1 classifier는 MANUAL에서도 계속 돈다

sidecar scheduler는 success mode와 독립적이다.

```text
classifier sidecar: every 5 steps when TCP speed < 0.05 m/s
(every step once p >= 0.05)
```

- 기본적으로 5 step마다, TCP 선속도 0.05 m/s 미만에서 무크롭 cam1/cam2 JPEG를 채점한다.
- 직전 `p >= 0.05`이면 매 step으로 올린다. **0.05는 cadence escalation 기준이지 success
  threshold가 아니다.**
- success threshold는 **0.5**, 비교는 엄격한 `p > 0.5`, `success_confirmations=1`이다.
- MANUAL에서도 classifier probability/threshold/verdict를 계속 계산하고 GUI에 표시하며 schema 3
  transition/replay meta에 저장한다. 단 classifier만으로 reward/terminal을 만들지 않는다.
- AUTO에서는 `auto_success=true`이고 classifier verdict가 참일 때 server가 success를 승인한다.

server가 사용하는 effective success는 다음 한 줄이다.

```text
operator_success OR (auto_success AND classifier_success)
```

reward 권위는 계속 server에 있으며 laptop이 임의 reward를 주입하지 않는다. 영구 learner JSONL이
모든 raw per-transition probability를 보존하는 것은 아니다. 현재 보존 범위는 전송된 transition과
RAM replay sidecar이고, 장기 사후감사에는 별도 bounded artifact/log가 여전히 필요하다.

### 5.2 compact GUI의 success 제어와 표시

통합 HIL GUI는 compact layout으로 다음을 한 화면에 표시한다.

- control owner와 actor state, episode/step, terminal reason
- classifier evaluated 여부, 마지막 `p(success)`, threshold, verdict
- `MANUAL` / `AUTO` mode
- `MARK SUCCESS (current episode)`
- 상태에 따라 바뀌는 `APPROVE HOME — ROBOT WILL MOVE` / `START / NEXT ITERATION (policy)`

MANUAL의 `MARK SUCCESS`는 현재 `(run_id, episode_id)`에 묶인 one-shot token을 만들고 다음
transition에서 `operator_success=true`로 소비된다. AUTO에서는 이 버튼/service가 거절된다.
반대로 MANUAL의 classifier verdict는 진단값으로 계속 보이지만 스스로 episode를 끝내지 않는다.

### 5.3 terminal 뒤에는 HOME 승인과 scene 준비를 각각 기다린다

`cube_in_cup`의 `MAX_EPISODE_LENGTH`는 100 step이다. terminal 원인은 server success,
truncated, local episode limit 중 하나다. 현재 actor는 terminal ACK 뒤 다음 순서를 강제한다.

```text
terminal pose HOLD
-> WAIT_HOME_APPROVAL
-> operator APPROVE HOME
-> HOMING / HOME
-> WAIT_SCENE_READY
-> operator가 scene을 재배치
-> START / NEXT ITERATION
-> fresh HOME observation -> BeginEpisode -> policy
```

HOME 이동은 optional local deepcopy/pickle I/O보다 먼저 수행된다. 마지막 `max_steps` terminal은
HOME 승인 후 종료하고 새 episode를 열지 않는다. WAIT 동안 status는 0.5초마다 재발행되므로 GUI를
늦게 띄우거나 다시 띄워도 현재 승인 단계를 복구할 수 있다.

실 gripper channel이 활성(`ACTION_SCALE[2] != 0`)이면 각 reset은 HOME 도착 뒤 Robotiq에
`position_percent=0.0`(OPEN)을 무조건 명령하고 fresh 상태가 open band에 들어왔는지 확인한다.
2초 안에 확인되지 않으면 WARNING과 reset info `succeed=False`를 남기지만 actor/episode를
중단하지 않는다. channel disabled/fake env에서는 하드웨어를 건드리지 않는다.

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
| online classifier verdict 육안/로그 검증 | **코드 PASS / 실기 미완료** | MANUAL/AUTO 모두 GUI 표시 + schema 3 transition/replay 기록; durable JSONL은 미완료 |
| policy-first startup UX | **코드 PASS / 실기 미완료** | handoff ENGAGED 뒤 HOME/WAIT, START가 DISENGAGE 후 policy-first 시작 |
| scene reset WAIT/Resume | **코드 PASS / 실기 미완료** | terminal -> WAIT_HOME_APPROVAL -> 승인 HOME -> WAIT_SCENE_READY -> START; WAIT 중 Step/Begin 없음 |
| 장시간 10 Hz 연속 운용 | **FAIL/PARTIAL** | 07-29 replay 201 부근에서 당시 0.6 s deadline 초과 |
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
| 오프라인 회귀 | **PASS** | 최신 `614 passed / 11 skipped / 1 xfailed` |
| actor(gRPC) 경로의 서브스텝 | **미검증** | 07-30 리그에 learner/gRPC가 없다. `run_remote_rlpd_actor.py`로는 아직 안 돌렸다 |
| mock RViz 개입 루프 | **미실행** | `04` §3 — 오프라인에서 실기로 바로 갔다 |
| 창 **사이**(RPC 구간) 부드러움 | **FAIL / G21 종속** | 주기 0.700 s에서 HOLD 51.5 %, 리더 속도 추종 66 % (오프라인 실측, `04` §9 표 1) |
| `substeps`/`governed`/포화의 운영 가시성 | **미구현** | CSV에만 있다 → §8 P3 |

### 6.2 2026-07-30 schema 3 MANUAL 경로

| 기능 | 상태 | 판정 범위 |
| --- | --- | --- |
| schema 3 learner ingress/update | **PASS** | online replay 400, learner 301, policy version 6 스냅샷 |
| MANUAL/AUTO success 분리 | **코드 PASS** | `operator_success` / `auto_success` wire·server 검증 |
| MANUAL classifier 지속 실행/표시/기록 | **코드 PASS** | mode와 무관한 sidecar scheduler + compact GUI + replay meta |
| terminal 승인 state machine | **코드/DDS PASS, 실기 연속 미완료** | WAIT 재발행, HOME 승인, scene-ready service 회귀 |
| production checkpoint | **미완료** | learner step 5,000 전 |

---

## 7. 남은 RPC timeout의 해석

이 절의 201-transition 분석은 **07-29 당시 0.6초 deadline run의 역사 기록**이다. 현재 코드는
RPC 하나당 **1.5초** deadline과 요청 생성부터 응답 수락까지 **2.0초** maximum response age를
함께 쓴다. retry가 성공해도 총 age가 2.0초를 넘으면 늦은 action으로 거절한다.

07-29 timeout은 이전의 “transition 100에서 처음 JIT compile이 수십 초 걸린 문제”와 다르다.
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

현재 경계를 1.5/2.0초로 늘린 것은 장애를 무한 대기시키지 않기 위한 fail boundary다. delayed
action을 로봇이 뒤늦게 실행하는 것은 실시간 제어 문제를 해결하지 않으므로, acceptance는 여전히
plain/sidecar RPC p99와 실제 `env.step` 간격을 따로 재는 것이다.

---

## 8. 다음 개발 방향과 우선순위

> **2026-07-30 재검토 결과:** P2와 P3는 코드/headless/DDS 수준에서 완료됐고 실물 연속
> acceptance가 남았다. P0/P1/P4/P5는 계속 열려 있다. 개입 손맛 작업(§3A)이 닫은 것은
> 별개 갭(`08_OPEN_GAPS.md` **G24**)이다.
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

### P2. commissioning handoff와 정상 policy-first episode를 분리한다 — 코드 완료

필요한 의미는 두 개뿐이다.

1. `commissioning`: 지금처럼 ENGAGED로 시작해 사람이 첫 action을 소유한다.
2. `policy-first`: fresh heartbeat를 요구하되 DISENGAGED 상태로 시작하고 policy가 첫 action을 소유한다.

현재 구현은 controller handoff까지만 commissioning ENGAGED를 요구하고, actor가 HOME/WAIT에
들어간 뒤 GUI START가 명시적으로 DISENGAGE하여 policy-first episode를 연다. deadman
heartbeat stale 시 fail-stop, GELLO stale 시 HOLD, controller proof는 그대로다. 남은 것은
실물에서 첫 START와 재시작을 확인하는 일이다.

### P3. operator episode state machine과 GUI를 만든다 — 코드 완료

현재 구현 상태는 다음과 같다.

```text
startup HOMING -> WAIT_SCENE_READY
WAIT_SCENE_READY -- operator START --> POLICY_RUNNING

POLICY_RUNNING
   -- ENGAGE ---------> HUMAN_INTERVENTION
   -- terminal --------> WAIT_HOME_APPROVAL

HUMAN_INTERVENTION
   -- DISENGAGE ------> POLICY_RUNNING
   -- terminal --------> WAIT_HOME_APPROVAL

WAIT_HOME_APPROVAL
   -- APPROVE HOME ---> HOMING -> WAIT_SCENE_READY

any state
   -- heartbeat/RPC fault --> FAULT + controller cleanup
```

GUI에 필요한 최소 표시는 다음이다.

- control owner: `POLICY / HUMAN / HOLD / FAULT`
- current episode/step
- classifier evaluated 여부, `p(success)`, threshold
- terminal reason: `SUCCESS / TIME_LIMIT / FAULT`
- MANUAL/AUTO, `MARK SUCCESS`, HOME 승인, `START / NEXT ITERATION` 버튼

**2026-07-30 추가 후보 — 개입 품질 표시.** `4197f5b` 이후 `info`에 다음이 들어 있지만 조작자는
볼 수 없다. 실기에서는 CSV를 사후에 열어야만 확인됐다(§3A.3).

- `intervention_substeps` — 이번 창에서 타깃이 몇 번 갱신됐나. **0이면 서브스텝 경로가 아예
  돌지 않은 것**이므로 "왜 갑자기 빳빳해졌나"에 즉답이 된다.
- `governed` / `governed_scale` — governor rate cap이 요청을 깎았나(대각 이동에서 특히).
- `reject_reason == BUDGET_EXHAUSTED` — 창 예산 소진(= 리더가 예산보다 빠름). **HOLD가 아니다.**
  실기에서 개입 창의 48~60 %가 여기 걸렸는데 조작자에게는 아무 신호도 없었다.

우선순위는 위 5개보다 낮다. 다만 이 셋은 **새로 계산할 것이 없고** `info`에서 그대로 읽어
쓰기만 하면 된다.

구현은 `/hil/actor_status` **status schema v2** JSON, `/hil/manual_success`,
`/hil/set_auto_success`, `/hil/scene_ready` Trigger를 사용한다. sparse classifier step에서는
actor가 마지막 evaluated 값과 env step을 유지하고, WAIT status는 0.5초마다 재발행되어 GUI
late join/restart가 복구된다. 새 `run_id`는 이전 GUI 비동기 요청과 classifier/terminal latch를
초기화한다. terminal 뒤 HOME은 optional local pickle보다 먼저 실행된다. 현재 남은 acceptance는
실제 UR7e에서 terminal→WAIT_HOME_APPROVAL→HOME→WAIT_SCENE_READY→START→policy의 연속 확인이다.

### P4. classifier 실기 정합을 눈으로 검증한다

- success/failed scene에서 GUI probability를 확인한다.
- server outcome과 standalone viewer가 같은 장면에서 대체로 일치하는지 본다.
- cam1을 팔이 가리는 `take_21` 유형은 sidecar로 해결되지 않는다. false negative가 반복되면
  threshold를 무작정 낮추지 말고 cam1 배치를 먼저 조정한다.
- schema 3 transition/replay에 저장된 classifier와 operator/auto success를 bounded artifact나
  durable JSONL에도 남겨 장기 사후 검증 가능하게 한다.

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

아래는 **현재 코드 그대로의 정상 3-CLI**다. actor 시작 전 fresh ENGAGED gate는 유지되지만
Enter 입력은 없고, controller handoff 뒤 HOME/WAIT에서 GUI START가 policy-first episode를 연다.

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

순서는 **1 → 2 → 3**이다. controller proof, topic rate probe, preflight heartbeat 3개와 switch
직전 heartbeat 3개는 유지된다. 제거된 것은 반복적인 Enter/GO 입력뿐이다.

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

### 9.4 Terminal 3의 operator 절차

`run_hil_session.sh`는 다음 순서로 진행한다.

```text
cam1/cam2 READY -> compact HIL GUI -> preposition
-> ENGAGED heartbeat 자동 대기 -> read-only armed preflight -> 실제 actor
-> HOME -> WAIT_SCENE_READY -> START / NEXT ITERATION
```

GUI를 `ENGAGED`로 만들고 GELLO를 고정하면 session wrapper가 heartbeat를 자동 감지한다.
별도 Enter는 누르지 않는다. actor가 HOME 뒤 `WAIT_SCENE_READY`에 오면 scene을 놓고
`START / NEXT ITERATION`을 누른다. 버튼이 deadman을 DISENGAGE한 뒤 policy episode를 연다.
개입하려면 `ENGAGE`, policy에 돌려주려면 다시 `DISENGAGE`한다. **DISENGAGE는 정지 명령이
아니라 policy 제어 복귀다.**

preposition은 현재 자세가 RESET의 0.10 rad 안이면 proof만 새로 쓰고 움직이지 않는다. 밖이면
기본값은 checklist를 표시한 뒤 **즉시 JTC trajectory를 전송**한다. wrapper 자체에는
`RESET_MAX_DIST_RAD=0.9` 최대거리 거부가 없고, joint-space 경로에 collision avoidance도 없다.
이 checklist를 강제 승인 gate로 오해하지 않는다. 과거 GO 입력이 필요하면
`PREPOSITION_CONFIRM=1`, 취소 가능한 지연이 필요하면 `PREPOSITION_DELAY_S=N`을 명시한다.
수락된 trajectory의 즉시 정지는 E-STOP이다.

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
3. 현재 Kanu schema 3 MANUAL run은 스냅샷 기준 transition 400, learner 301, policy version 6까지 갔다.
4. 07-29의 0.6초 timeout은 역사 증거다. 현재 fail boundary는 RPC 1.5초 / response age 2.0초이며,
   다음 blocker는 이 경계 안이라는 사실이 아니라 **실제 10 Hz p99와 연속 운용 검증**이다.
5. compact GUI, MANUAL/AUTO, classifier 지속 표시·기록, policy-first와 두 단계 reset 승인은
   코드/DDS PASS다. 다음 실기는 terminal→WAIT_HOME_APPROVAL→HOME→WAIT_SCENE_READY→START를 확인한다.
6. 개입 손맛은 **2026-07-30에 실기 PASS**다(§3A, 커밋 `4197f5b`) — 단 정책 zero·gRPC 없는
   **격리 리그**였고, 창 **사이**(RPC 구간)의 부드러움은 여전히 **G21 종속**이다. 그래서
   P1(gRPC 지연)과 P4/P5 실기 검증은 여전히 남아 있다.
