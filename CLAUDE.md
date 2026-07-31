# CLAUDE.md

문서 색인이다. 상태 보고서가 아니다 — 숫자와 근거는 링크된 문서에 있다.

> # 🔴 2026-07-31 — **GPU 서버가 `kanu`에서 `junhyeong_ai`로 옮겨졌다**
>
> **이 파일을 포함해 이 리포의 거의 모든 문서가 `kanu`를 전제로 쓰여 있다.**
> 데이터·모델·코드 checkout의 **현재 위치**는 아래 문서가 정본이다:
>
> ### 👉 [`serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md)
>
> | | 예전 | 지금 |
> | --- | --- | --- |
> | ssh 별칭 | `kanu` | **`junhyeong_ai`** (166.104.146.29, hostname `junhyeong`) |
> | GPU | A4000 ×8 (sm_86), GPU 5 | **RTX 5070 Ti ×1 (sm_120 Blackwell), GPU 0** |
> | HIL checkout | `~/gello_software_hil_current` | **`~/gello_software_runtime`** |
> | 데이터·모델 | 네 군데 분산 | **전부 `~/hil-serl-data/` 아래** |
> | 환경변수 | `HIL_KANU_*` | **`HIL_REMOTE_*`** (+ 신설 `HIL_REMOTE_DATA_ROOT`. 옛 이름은 alias) |
>
> **코드 기본값은 `5594d0e`에서 이미 전환됐다** — `./run_hil_server.sh`를 환경변수 없이
> 그냥 쓴다. 환경변수 0개로 실기동해서 증명했다(burn-in #2: demo 2,037 로드,
> `jax_backend=gpu`, GPU 0에 4400 MiB 실점유).
>
> 이전은 **복사**였고 **kanu에서 지운 것은 없다.** kanu learner는 사용자가 직접 멈출
> 때까지 살아 있다. ⚠️ 다만 **`run_hil_server.sh`로는 이제 kanu를 볼 수 없다** —
> classifier가 kanu에서만 다른 뿌리에 있어 단일 `HIL_REMOTE_DATA_ROOT`로 표현되지 않는다.
> kanu는 `ssh kanu 'ps -p <pid>'` 같은 읽기 전용으로 본다(위 문서 §6).
> ⚠️ 새 서버의 **`/home/junhyeong/gello_software`(접미사 없음)는 다른 사람의 작업
> 트리다.** 읽지도 쓰지도 말 것. HIL이 쓰는 것은 **`gello_software_runtime`**이다 —
> 편집하는 트리와 실행하는 트리를 이름으로 갈라 놓았다.
> ⚠️ 학습된 policy 체크포인트는 **어디에도 없다** — kanu의 run root 8개 전부
> `checkpoints/`가 비어 있었다(실측). 이유는 위 문서 §4.

> # 🔴 2026-07-31 merge `5e508d3` — **지금 learner를 재기동하지 마라**
>
> 서버 이전 작업과 **공유 frozen-trunk feature 작업**(다른 분의 7커밋 `40e8305..a93d330`)이
> 한 브랜치로 합쳐졌다. 충돌은 0이었고 양쪽 다 온전하다. 그런데 **그 작업은 진행 중이고
> 순서가 뒤집힌 채로 들어왔다:**
>
> ```
> 안전한 순서:  크롭 기준 재학습 → 체크포인트 교체 → classifier loader 전환
> 실제 상태:                        (아직 없다)       ✅ 이미 전환됨
> ```
>
> `a93d330`이 production learner의 classifier 입력을 **크롭된 policy 관측에서 나온
> frozen-trunk feature**로 바꿨다(`run_rlpd_learner_server.py`가 `FeatureReplayIngress` +
> `FaultGatedReplayIngress`를 쓰고 둘 다 `prime_observation`을 구현하므로 **production
> 경로는 항상 feature를 받는다.** receive-only 서버는 평범한 `ReplayIngress`라 영향 없다).
> 그런데 핀으로 박힌 `checkpoint_150`의 classifier는 그 텐서를 **못 먹는다** — CPU 실측:
> `PIXEL input OK` / `FEATURE input FAILED: shapes=[(128,128,512), (3,)]`.
>
> **그래서 merge된 코드로 learner를 새로 띄우면 첫 채점 스텝에서 죽는다**
> (`RewardClassifierError` → actor 사망). 지금 살아 있는 learner는 merge **이전** 코드라
> 정상이다 — 옛 sidecar 픽셀 경로를 쓴다.
>
> 🛑 **그리고 뻔한 한 줄 수정이 이 시끄러운 실패를 정확히 G15로 바꾼다.**
> `frozen_trunk.create_frozen_trunk_classifier`가 이미 있고 param tree가 **52 leaf 전부
> 동일**해서 `checkpoint_150`이 **에러 없이 복원되고 숫자를 뱉는다.** 그러면 무크롭으로
> 학습된 가중치가 크롭 유래 feature를 채점한다 — 100% → 33.3% recall 실측과 *비슷한* 게
> 아니라 **산술적으로 같은 실험**이다(두 trunk가 같은 SHA의 `resnet10_params.pkl`, 둘 다
> `train=False`). **체크포인트 SHA 가드는 통과한다** — 가중치는 안 바뀌었으니까.
>
> ⚠️ **재사용 검사도 이걸 못 본다.** merge 전후로 observation schema hash ·
> policy/reward model ID · `validate_process_contract`가 **전부 byte-identical**이라
> `--check`의 `healthy`는 프로세스 계약에 대해선 참이지만 **코드 버전에 대해선 침묵**한다.
> run root 어디에도 기동 시점 커밋이 기록되지 않는다.

## 이 리포에서 진행 중인 작업

**HIL-SERL(사람 개입 온라인 RL)을 실기 UR7e에서 돌린다.**

```
laptop3                                  junhyeong_ai (GPU 서버, 2026-07-31~)
  GELLO 리더팔 (USB, EEF teleop)                 정책 추론 (SAC)
  RealSense ×2 (USB)          ──gRPC :50053──►   온라인 학습 (RLPD)
  UR7e (이더넷, ROS2 Humble)  ◄──── 액션 ─────    reward classifier
```

laptop3의 GPU가 약해 **정책·학습·reward classifier를 전부 GPU 서버에서** 돌리고 gRPC로
실시간 통신한다. reward 권위는 서버에 있다. 명목 제어 루프는 10 Hz지만 **production actor
실측은 1.95 Hz**(주기 512 ms, 최대 854 ms)다 — 🗄️ **이 값은 kanu에서 측정했고 새 서버에서
재측정하지 않았다.** 새 서버는 per-RPC가 더 빠르므로(BeginEpisode tail 372.8 → 82.2 ms)
루프도 빨라졌을 가능성이 크지만 **측정 전에는 모른다** — `env.step`은 100 ms로 자체 페이싱하고 나머지 ~412 ms는
블로킹 gRPC Step RPC + 카메라 디코드로 `env.step` **밖**에 있다. 이 격차가 2026-07-30 개입
손맛 문제의 뿌리다(아래). 2026-07-30 startup에서 정상 reply가 832.3 ms에 도착해 옛 0.6/0.8 s
경계를 넘었으므로 현재 RPC timeout/response-age는 bounded `1.5/2.0 s`로 완화했다.
분류기는 정책 관측이 아니라 **자기 전용 무크롭 이미지(sidecar)를 약 2 Hz로** 따로 받는다.
⚠️ **`5e508d3` merge 이후 production learner는 그 픽셀을 더 이상 채점하지 않는다** — 위 🔴
박스. sidecar는 여전히 **채점 시점을 정하는 게이트**라 지우면 reward가 통째로 꺼진다.

**branch `feat/gello-ur7e-humble-22.04`**. 2026-07-29 머지 `3f199d4`가 로봇/하드웨어
작업을 이 브랜치로 가져왔다. **워크트리 분리는 끝났다** — 로봇 코드와 learner 코드가 **다른
checkout에 있다**고 적힌 문서는 전부 낡은 것이다(아직 여러 개 남아 있다). 작업은
`/home/laptop3/gello_software` 한 곳에서만 한다.

## 지금 상태 한 줄

**2026-07-30 현재 HIL-SERL 원형은 실물 UR7e에서 구동됐다.** Kanu policy action으로 팔이
움직이고, GUI `ENGAGE` 중에는 GELLO 개입, 해제 뒤에는 policy 제어로 돌아가며, online
transition과 learner update까지 관측했다. 현재 정상 운용은 아래 세 terminal뿐이다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_hil_server.sh    # Terminal 1: 서버 learner 재사용/기동 + tunnel (환경변수 0개)
./run_hil_hardware.sh  # Terminal 2: UR7e + Robotiq + passive GELLO
./run_hil_session.sh   # Terminal 3: cameras + GUI + actor
```

**현재 episode 운영 계약:** 성공 판정은 기본 `MANUAL`이다. 서버 classifier는 MANUAL에서도
계속 평가·표시·replay 기록되지만 terminal 권한은 GUI `MARK SUCCESS`에 있다. `AUTO`로
전환하면 strict `p(success) > 0.5`가 성공 권한을 가진다. 성공 또는 episode limit 뒤에는
`WAIT_HOME_APPROVAL`에서 로봇을 hold하고, GUI `APPROVE HOME` 뒤 HOME, 장면을 사람이
재배치한 뒤 `START / NEXT ITERATION`을 눌러 다음 policy episode를 연다. classifier headline은
16 pt의 compact GUI로 표시된다. **2026-07-30 저녁부터 세 번째 버튼 `END EPISODE`가 있다** —
망친 episode를 지금 끝내는 truncation이다(아래 별도 항목, **데이터는 지워지지 않는다**).

**시작 조작 간소화:** `run_hil_preposition.sh`의 예전 대문자 `GO` 입력은 기본 경로에서
없어졌다. RESET 0.10 rad 밖이면 체크리스트를 출력한 뒤 기본값은 곧바로 JTC 이동을 시작한다
(`PREPOSITION_DELAY_S=N`으로 취소 가능한 카운트다운, `PREPOSITION_CONFIRM=1`로 옛 GO
프롬프트를 opt-in할 수 있다). `run_hil_session.sh`의 별도 Enter 프롬프트도 없어졌고 deadman
heartbeat가 요구 상태가 될 때까지 폴링한다. 이것은 **타이핑 제거**이지 controller/pose proof나
fresh heartbeat 검사를 없앤 것이 아니다.
> **이전 판(보존):** *"GUI가 **ENGAGED**가 될 때까지 폴링한다 (…) fresh **ENGAGED** heartbeat
> 검사를 없앤 것이 아니다."* 2026-07-30 저녁부터 요구 상태는 **DISENGAGED**다 — 아래
> "세션은 이제 DISENGAGED로 시작한다" 항목.

**서버 현재 계약:** actor transport는 protocol 2 / schema 3, reward threshold는 0.5다.
2026-07-31 `junhyeong_ai`(GPU 0)에서 canonical offline demo 2,037개를 로드해 health-ready가
됐고, **실기 세션이 PASS했다** — replay 316 / intervention 210 / last_env_step 68이
`~/hil-serl-data/runs/cube_in_cup_real_20260731_054929`로 유입됐다. `intervention 210`이
load-bearing이다(합성 run이 증명 못 하던 `intervened=1` ingress와 실제 classifier sidecar를
닫았다). 이 수치는 계속 변한다. **warm start는 없다** — 물려받은 checkpoint가 없어 lineage는
깨끗하다(§4 아래 🔴). PID와 run root는 스냅샷이므로 매번 `./run_hil_server.sh --check`로 읽는다.

> 🗄️ **이전 판(kanu 기록, 보존):** *"2026-07-30 새 learner는 GPU 5에서 canonical offline
> demo 2,037개를 로드해 health-ready가 됐다. 최신 읽기 전용 스냅샷은 online replay 400 /
> intervention 225, learner 301 / gradient 602 / policy version 6 (…). 이전 learner RAM에만
> 있던 테스트 replay 257 / intervention 107 / learner step 158은 checkpoint가 없고 의미 없는
> 시험값이라는 사용자 판단에 따라 폐기했다. offline demo pickle은 그대로 보존했다."*
> 이 숫자들은 전부 **kanu에서 측정한 kanu의 사실**이다. 새 서버로 옮겨 적지 않는다.

**Kanu repo 배치 (2026-07-31 정리) — 스택당 checkout 하나씩, 그게 전부다.**

> 🗄️ **아래 표는 이제 `kanu` 기록이다.** learner는 `junhyeong_ai`로 옮겨졌고 현재 배치는
> [`serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md) §2가 정본이다.
> 아래 🪤와 ✅ 두 문단은 **여전히 유효한 교훈**이라 남긴다 — 새 서버의 checkout도 정확히
> 그 가드를 통과하도록 만들었고, `run_hil_server.sh`가 지금도 그것들을 검사한다.

| 경로 | 용도 |
| --- | --- |
| `gello_software_hil_current` → `gello_software_hil_schema3_stage_20260730` | **HIL 유일 checkout.** 자립 clone(`.git`이 **디렉터리**), GitHub origin, `git pull --ff-only`로 전진 |
| `workspace/youngwoong/gello_software` | FM/diffusion 배포 소스(도커 `gello-remote-policy:fm-070000-…`). **다른 스택 — 섞지 말 것**(§E) |

심링크가 `HIL_KANU_REPO`의 기본값이고 스크립트가 `readlink -f`로 실경로를 쓰므로 그것만 타이핑한다.
🗑️ kanu의 `~/gello_software_hil`(worktree였다)과 `/tmp` worktree 2개는 은퇴했다.

🪤 **왜 정리했나 — 조용히 틀리는 종류의 사고였다.** staging checkout의 `origin`이 GitHub이 아니라
**로컬 경로**를 가리켜, `git fetch origin`이 **rc=0으로 성공하고 아무것도 안 가져왔다.** 그래서
kanu가 하루치 뒤처진 채 모든 점검이 healthy로 보였고, 그 stale `FETCH_HEAD`로 reset했으면
operator-session 기능 전체가 롤백됐을 것이다(diff 2,349줄 삭제로 발각). 구조적 결함은 셋이었다 —
checkout이 여럿, 하나가 GitHub이 아니라 **다른 checkout에 사슬로 물림**, 그리고 **kanu HEAD와
laptop3 HEAD를 비교하는 코드가 어디에도 없었다.**

✅ **그래서 `run_hil_server.sh`에 가드가 생겼다.** GitHub을 조회하는 대신 **ssh 채널로 laptop3↔kanu
HEAD를 직접 비교한다** — 사고가 "kanu가 laptop3에서 갈라진 것"이고 ssh는 이미 그 둘을 잇고 있으므로
**네트워크 없이 성립한다.** 함께: `--git-dir == --git-common-dir`(worktree 사슬 금지) + origin URL이
canonical GitHub인지. 실패 동작은 **의도적으로 갈랐다** — 새 lineage 시작은 `remote_die`,
**기존 learner 재사용과 `--check`는 큰 배너 후 진행**한다(재사용까지 막으면 살아 있는 learner 밑에서
checkout을 올리거나 lineage 중간에 laptop3 개발을 얼려야 한다. 재사용의 교차버전 안전성은
schema-hash/model-id/process-contract가 이미 본다. **고칠 결함은 불일치가 아니라 침묵이었다**).
탈출구 `HIL_ACCEPT_HEAD_MISMATCH=1`. GitHub tip 조회는 부가 진단이고 실패해도 치명적이지 않다.
⚠️ **한계: HEAD만 본다.** laptop3 작업 트리는 상시 dirty라(07-30 저녁 배치 전체가 uncommitted로 돌았다)
설계상 범위 밖이다.

## 2026-07-30 저녁 — 조작자 경로 4건 (**실기 미검증**)

아래 넷은 **아직 커밋 전**이고 **실기에서 한 번도 돌지 않았다.** 오프라인 단위 테스트만
있다(맨 아래 테스트 기준선 참조). 실기 검증 항목표는
[`docs/testing/README.md`](docs/testing/README.md) §1의 17~20행.

### 1) 세션은 이제 **DISENGAGED**로 시작한다 (ENGAGE 불필요)

세 gate 전부 — `run_hil_session.sh`의 폴링 루프, `run_hil_actor.sh` preflight `[11]`,
그리고 `[ARM]` 직전 재검증 — 이 **연속 fresh DISENGAGED heartbeat 3개**를 요구한다.
`_hil_deadman_check.py`에 `--require engaged|disengaged`가 생겼고, `HIL_STARTUP_DEADMAN`이
그 값을 고른다. 오타는 **fail-closed**다(두 스크립트 모두 `case` 검사 후 즉시 종료 — "gate
없음"으로 조용히 해석되지 않는다). `ENGAGE_WAIT_S`는 `DEADMAN_WAIT_S`로 이름이 바뀌었고
**옛 이름은 alias로 남아 있다**(`${DEADMAN_WAIT_S:-${ENGAGE_WAIT_S:-120}}`).

**gate의 의미는 처음부터 의도(intent)가 아니라 데드맨 채널의 생존 증명(LIVENESS PROOF)이었다** —
"GUI가 살아서 `/hil/deadman`에 20 Hz로 퍼블리시하고 있다". fresh heartbeat 3개는 요구 상태가
무엇이든 그것을 똑같이 증명한다. DISENGAGED로 시작하는 것이 조작자의 요구였고, 무의미한
`ENGAGE → START → 자동 DISENGAGE` 왕복을 없앤다 — `RosOperatorSession._on_scene_ready`는
`WAIT_SCENE_READY`에서 **원래부터** engaged면 START를 거부했다
("DISENGAGE the deadman before resuming policy"). 되돌리기: `HIL_STARTUP_DEADMAN=engaged`.

🟡 **알면서 받아들인 trade-off(사용자 명시 거절: "굳이 불필요한 안전장치는 만들지 않아도 돼").**
이제 **policy가 팔을 몰기 전에 ENGAGE 전이가 한 번도 실행되지 않는다.** GUI의
`engage_button_enabled`는 actor status의 state가 `ACTIVE_CONTROL_STATES`
(`POLICY_RUNNING`/`HUMAN_INTERVENTION`/`HOLD`)일 때만 ENGAGE를 허용하므로,
`HOMING`/`WAIT_SCENE_READY`/`WAIT_HOME_APPROVAL` 구간에서는 버튼이 죽어 있다. 즉 **ENGAGE
경로가 깨져 있으면 조작자가 실제로 개입해야 하는 순간에 발견된다.**
정확히 하자면: status가 아직 하나도 없으면(actor 기동 전) `engage_button_enabled`는 legacy로
`True`를 반환하므로 버튼 자체는 눌린다 — 다만 그러면 DISENGAGED gate가 막혀 세션이 진행되지
않는다. 그리고 옛 ENGAGED gate가 증명하던 것도 "GUI 버튼 → 토픽에 `engaged=1`"까지였지
env 쪽 follower arming은 아니었다(그 시점엔 env가 없다).
**탈출구는 `HIL_STARTUP_DEADMAN=engaged`** — 옛 gate가 그대로 돌아오고 세 gate가 다시
ENGAGED를 요구한다.

⚠️ **"START 전에는 팔이 안 움직인다"는 두 모드 모두에서 거짓이다.** actor는 첫
`WAIT_SCENE_READY` **전에** `env.reset()`을 부르고, 그것이 `go_to_reset()`(수 초짜리 20 Hz
스트리밍 이동)과 `open_gripper_for_reset()`을 실행한다(`remote_actor` 기동 시퀀스:
`publish(HOMING)` → `env.reset()` → `wait_for_scene_ready`). 이번 작업이 만든 것도 바꾼 것도
아니고 `RESET_MAX_DIST_RAD`로 bounded지만, **문서가 "아무것도 안 움직인다"로 읽히게 두지 말 것.**

### 2) 새 조작자 버튼 `END EPISODE` (구현 중 이름은 ABORT였다)

`/hil/abort_episode` (`std_srvs/Trigger`). `(run_id, episode_id)`에 묶인 **one-shot 토큰**이고,
`ACTIVE_CONTROL_STATES`에서만 합법이며 `terminal_reason`이 이미 세워졌으면 거부한다
(그 창에서 수락하면 다음 publish의 non-active-state 규칙에 조용히 버려지기 때문 —
`_on_abort_episode` docstring). MANUAL/AUTO **양쪽에서** 받는다(망친 episode를 버리는 것은
성공 주장이 아니다). episode를 즉시 `done=False, truncated=True, masks=1.0, success=False`로
끝내고 평소와 같은 `WAIT_HOME_APPROVAL` → HOME → `WAIT_SCENE_READY` 경로를 탄다.
`terminal_reason`은 `OPERATOR_ABORT`이며 이것이 `_terminal_reason`에서 SUCCESS·TRUNCATED보다
우선한다.

🛑 **정직성 항목 — 이 버튼을 언급하는 곳마다 같이 적어야 한다: 아무것도 버려지지 않는다.**
proto에 cancel/retract RPC가 **없다**(`Health` / `GetServerInfo` / `GetBufferStatus` /
`BeginEpisode` / `Step` 5개뿐). 그리고 서버는 Ack를 만들기 **전에** Step 핸들러 안에서
동기적으로 replay store에 insert한다(`GrpcActorServicer.Step` → `self._service.step()` →
`_insert_route(self.replay_store, …)`). 즉 **클릭 시점까지의 모든 전이는 — 잘못된 것까지 포함해 —
이미 learner 버퍼에 있고 정상적으로 학습된다.** 게다가 그 순간 조작자는 대개 GELLO를 잡고
있으므로 그 행들은 `intervened=1`이라 **두 버퍼 모두에** 들어가고 RLPD 50:50 분할이
**가중치를 올려 준다.** 버튼이 사주는 것은 딱 둘이다: **(a)** step limit이 아니라 지금 끝내는 것,
**(b)** 조작된 terminal 대신 **정직한 bootstrap-safe truncation**.

⏱️ **즉시가 아니다.** actor는 토큰을 **iteration당 두 번** 읽는다 — `env.step` **직전**(그러면
대기 중이던 policy action이 실행되지 않고 폐기된다)과 **직후**(그러면 방금 실행된 전이가
truncated로 기록된다). 최악은 **루프 한 주기**(실측 평균 512 ms, 최대 854 ms)다. GUI는 Trigger
**전에** 데드맨을 먼저 놓아 GELLO 추종을 **~33 ms**에 멈춘다(`_do_abort_episode`, 100 ms 뒤
Trigger 발사). 🛑 **그러나 데드맨은 policy 경로를 전혀 게이팅하지 않는다** — 배경 follower와
`GelloIntervention.action()`만 본다. **policy가 몰고 있을 때 데드맨을 놓는 것은 아무것도 멈추지
않는다.** GUI 문구가 이것을 말하도록 되어 있다(`_ABORT_RELEASE_TEXT`).

### 3) 충돌 복구 — 하드웨어 번들 재기동에서 세션이 살아남는다

카메라와 GUI는 그대로 살아 있고 **3~5단계(preposition → preflight → actor)만** 재시도 루프를
돈다. `run_hil_actor.sh`에 **종료 코드 계약**이 생겼다: **75 = recoverable** — 단 감시자가
`/hil/actor_status`에서 `env_step >= 0`을 실제로 본 경우에만(= actor가 transition 루프까지
갔다. `_OperatorReporter`는 `env_step`을 −1로 시작한다), **1 = arming 자체가 없었음**,
**70 = controller 복귀 실패**, **>=128 = 신호**. 승격을 끄려면 `HIL_ACTOR_EXIT_MAP=0`.

🔧 **2026-07-31 실기 수정.** Terminal 2를 내리면 controller 복귀가 실패하는 게 당연한데
그것이 `70`(재시도 금지)으로 잡혀 **세션이 자동 종료됐다** — 복구 루프가 막으려던 바로 그
상황을 복구 루프가 막았다. `hil_read_controller_states`가 이제 **"스택이 이상하다"(1)와
"스택이 아예 없다"(2)를 구분**한다. controller_manager 무응답이거나 우리 controller 쌍이
목록에 하나도 없으면 **번들 사망**이고, 그때는 ros2_control 컨트롤러가 하나도 없으므로 팔을
몰 수 있는 것도 없다 → `75`. 이 경로만 **진행 증거(`env_step>=0`)를 요구하지 않는다** —
그 게이트는 결정론적 startup 실패를 막는 것인데, 그런 실패는 controller_manager를 사라지게
만들지 않기 때문이다.

재시도는 추가로 이 셋을 **전부** 요구한다: 조작자가 번들을 정말 재기동했다는 증거(세
entrypoint `ur_control.launch.py` / `robotiq_gripper_modbus` / `gello_publisher`가 모두 있고
**PID 집합이 이전 세대와 하나도 겹치지 않음**), 세 토픽이 `run_hil_hardware.sh` 자신의 READY
기준을 만족, 그리고 **dashboard 서비스로 읽은 robot mode `RUNNING` + safety mode `NORMAL`**.
마지막 것이 핵심이다 — 토픽 probe 셋은 전부 RTDE **읽기**라 `PROTECTIVE_STOP` 중에도 계속
흐르므로, 그것만 보고 재arming하면 움직이지 않는 로봇에 명령을 흘리게 된다.
한계는 `HIL_ACTOR_RETRY`(0/1) / `HIL_ACTOR_RETRY_MAX`(기본 3, 상한 10) /
`HIL_HARDWARE_RECYCLE_WAIT_S`(기본 900) / `HIL_RETRY_RESUME_DELAY_S`(기본 5).
**모든 안전 proof가 매 시도마다 처음부터 다시 돈다 — 캐시되는 것은 없다.** resume이 아니라
**새 arming**이다.

🛑 **재시도 경로에서 RESET 복귀는 shell의 `run_hil_preposition.sh`가 한다 — GUI의
`APPROVE HOME`이 아니다.** 그 버튼의 서비스는 actor 프로세스 안에서 만들어지므로
(`RosOperatorSession`이 actor의 backend node에 서비스를 연다) **actor가 죽어 있는 동안에는
존재하지 않는다.**

### 4) Qt 폰트 경고 억제

`launch_cameras.sh`(뷰어)와 `run_hil_actor.sh`(actor의 `DISPLAY_IMAGE` 창)에서 **정확히 두
줄**만 stderr에서 걸러 낸다(`QFontDatabase: Cannot find font directory …/cv2/qt/fonts.` 와 그
다음 `Note that Qt no longer ships fonts. …`). **뻔한 해법 둘이 왜 안 되는지 기록해 둔다 —
다시 시도하지 말 것:** `cv2/config-3.py`가 **import 시점에** `QT_QPA_FONTDIR`을
`<cv2>/qt/fonts`로 **덮어쓰므로** 환경변수는 무효고(2026-07-30 측정), 이 메시지는 **카테고리
없는 qWarning**이라 `qt.qpa.fonts.warning=false` 같은 scoped `QT_LOGGING_RULES`도 무효다.
듣는 규칙은 `default.warning=false` 하나뿐인데 그건 **진짜 Qt 오류까지 숨기므로 쓰지 않는다.**
패턴은 양 끝을 고정했다 — 문구가 바뀌면 필터가 안 걸리고 그대로 보이는 쪽이 옳은 실패 방향이다.
필터는 `trap '' INT TERM` 아래에서 돈다: 그게 없으면 터미널 Ctrl-C가 (setsid로 빠져나간 actor
대신) foreground process group의 `grep`을 **먼저** 죽여 **actor의 종료 경로 stderr가 통째로
사라진다**(KeyboardInterrupt traceback + publisher/gRPC teardown 진단, 그리고 actor는
BrokenPipeError를 본다). `run_hil_gui.sh`에는 필터가 **없고 그게 맞다** — 그 GUI는 시스템
Qt5(fontconfig)를 쓰고 cv2를 import하지 않는다.

**2026-07-30 저녁 — 개입은 이제 `env.step` 창이 아니라 배경 추종 스레드가 몬다 (`edbb3f5`).**
같은 날 오전의 창 안(`in_window`) 수정으로는 부족했다. 30 Hz 재샘플링이 `env.step`의 **명목
100 ms 창 안에서만** 돌았는데 실제 루프는 512 ms라 나머지 ~412 ms는 타깃 갱신이 없었고,
`InterventionBudget`이 창당 1× `ACTION_SCALE`로 제어를 묶어 개입 최고속이
`ACTION_SCALE/T` = **2.4 cm/s**였다. 이제 `UR7eEnv`가 **데몬 추종 스레드**(`_follow_loop` /
`_follow_tick`)를 돌린다 — ENGAGED 동안 팔이 30 Hz로 리더를 따라가고 `env.step`은 관찰자가
되어 transition만 뽑는다. **예산은 개입 제어 경로에서 완전히 제거**했다. 남은 상한은
governor(`v_max` 0.15 m/s) + 워크스페이스 박스 + 250 Hz 업샘플러뿐이고 셋 다
`PolicyDeltaController.step` 안에 있다. 기존 경로는 `follow_mode="in_window"`로 보존했다.

| | 이전(`in_window`) | 이후(`background`) |
| --- | --- | --- |
| 개입 최고속 | 2.4 cm/s | **15 cm/s** (governor `v_max` 0.15 m/s 유래. ⚠️ `edbb3f5` 커밋 요약의 "12.5 cm/s"는 `ACTION_SCALE[0]×HZ`에서 나온 **틀린 값**이다 — `follow_xi`에는 예산도 `_paced_request`도 없다. 코드가 인용하는 검증 리그 EEF teleop 실측은 12.4 cm/s → `04_HIL_INTERVENTION.md` §9 말미 📌) |
| 창 밖 HOLD | 27.7% | 없음 (타깃이 계속 흘러 `target_stale_s` 미도달) |
| 업샘플러 slew 상한 | 46% (soft-start 재무장) | 100% |
| 데드맨 release 지연 | 512 ms | **33 ms** (추종 틱마다 재읽기) |

설계 근거 전문은 [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md)
**§10**(§9는 오전 `in_window` 경로다), 갭은 `08_OPEN_GAPS.md` G24/G32/G33/G34.

마지막 줄은 손맛이 아니라 **안전 이득**이다. 🟡 **검증 수준**: 조작자가 실기에서 "개입 속도는
좀 고쳐졌어"라고 확인했을 뿐이고(2026-07-30), 오전 `4197f5b` 때처럼 **CSV 불변식 전수를 돌린
체계적 실기 검증은 아직 없다 — 미검증으로 취급한다.**

같이 들어간 것:

- **`PolicyDeltaController` 스레드 안전화**(`edbb3f5`). 적대적 검수가 실제 컨트롤러 + `ur_kin`으로
  재현해 확증한 결함 3건: (a) `(T_cmd, q_cmd)`가 **11.1%** 확률로 서로 다른 틱의 것
  (108,949 중 12,143) → IK가 남의 해를 seed로 풀어 브랜치 연속성이 깨지고 **10초짜리 눈먼 한 바퀴
  스윕**으로 실행된다, (b) `_hold()`가 `self.q_cmd`를 반환해 **HOLD 틱이 움직임을 발행**
  (130,028 중 25건, 최대 0.2047 rad = `dq_step_max`의 3.3배), (c) `dq_step_max` 게이트가
  **6.08%** 누출. 수정: 자체 `RLock`, `_commit()` 한 곳에서 원자적 갱신, `_hold()`는
  `_last_issued` 반환.
- **One-Euro 시정수를 창 개수 → 벽시계로**(`6ca35ac`). `dt`가 생성 시 `1/substep_hz`로 고정인데
  `filtered()`는 창당 2번만 불려, T=1.12 s면 실제 간격 560 ms를 33.3 ms로 착각해 **16.8배 과다
  평활**했다. 실효 시정수 2.945 s → **0.498 s**.
- **그리퍼를 세션 시작(`run_hil_preposition.sh` `[5b/6]`)과 매 에피소드 경계
  (`UR7eEnv.open_gripper_for_reset`)에서 연다.** 모든 offline demo가 열린 그리퍼에서 시작하고
  `gripper_position`은 `state[0]`이다.
- **실기에서 한 번 터진 것**(`d6965a9`): 예산을 빼면서 harvest가 기록 액션을 축별
  `np.clip(raw, -1, 1)`로 잘랐는데, 그러면 norm이 √3까지 허용된다. production 체인의
  `RelativeFrame.transform_action_inv`가 `blockdiag(R, R)`을 곱하고 **회전은 norm은 보존하지만
  축별 최댓값은 보존하지 않는다** → `[1.0, 1.0, 0]`(norm 1.41)이 `[1.41, 0, 0]`으로 나와
  `ActorProtocolError: executed_action must be within [-1, 1]`로 actor가 즉시 죽었다.
  **norm 비례 축소**로 고쳤다(아래 "반드시 지킬 것" 참조).
- **프롬프트 2개 제거**(`e86edd5`, 아래 "시작 조작 간소화" 항목이 이것이다). 되돌리기:
  `PREPOSITION_DELAY_S=5` / `PREPOSITION_CONFIRM=1` / `ENGAGE_WAIT_S`.

> **이전 판 문구(보존):** *"연속 운용에서 창이 늘어지면 **예산이 소진되고 남은 시간은
> HOLD**이며, 그건 필터·rate·외삽으로 못 고친다 — G21(RPC 지연) 종속이다."* 이 문장은
> `in_window` + `InterventionBudget` 기준이다. `background`에는 예산이 없고 추종 스레드가 RL
> 루프와 무관하게 틱하므로 **창 길이가 개입 속도를 더 이상 묶지 않는다.** G21은 여전히
> 열려 있지만 이제 **transition 밀도**와 policy 반응성의 문제이지 사람 손맛의 문제가 아니다.

**2026-07-30 오전 — 창 안(`in_window`) 1차 수정이 실기에서 PASS (`4197f5b`).**
아래 표는 **`in_window` 경로의 실측**이며 shipped 기본값(`background`)의 증거가 **아니다** —
다만 원인 규명과 "되돌리면 안 되는 것" 3건은 그대로 유효하다. 개입 중 팔이 "빳빳"했던 원인은
10 Hz 자체가 아니라 **타깃 갱신 방식**이었다 — `env.step`이 100 ms 창에서 관절 타깃을 한 번만
세우고 그 사이 리더를 다시 읽지 않아, 250 Hz 업샘플러가 매 창마다 가속→도달→제동→**정지**를
반복했다(gRPC 왕복이 `env.step` 밖이라 실효 갱신은 6~10 Hz). 해결은 검증된 EEF teleop과 같은
방식이다: 개입 중 **리더를 30 Hz로 재샘플링** + **One-Euro**(`bridge_stages.py`에서 비트 동일
이식) + `InterventionBudget`이 창당 총 변위를 `ACTION_SCALE`로 묶는다
(신규 `ur_env/envs/leader_stream.py`, `ros_backend.py`는 무변경).
실기 3 run 중 뒤 두 run 전체 PASS (2026-07-30, 실제 UR7e, `tests/run_real_hil.py`).
첫 run(DRY `--scale 0.5`, 개입 144)은 고친 판정으로 **SKIP**이다 — 포화 표본을 빼면
축별 여기가 2 cm 게이트에 못 미친다. 포화를 포함하면 세 run **전부** FAIL로 나온다
(잔차 0.776 / 0.154 / 0.358) — 그래서 포화 제외는 첫 run을 구제하는 사후 변명이 아니다:

| run | 결과 |
| --- | --- |
| DRY RUN `--scale 1.0` (300스텝, 개입 272) | frame-map 잔차 **0.016** / alpha **1.005** / 표본 141(포화 131 제외) · action-exec dp_ratio 중앙값 **1.000** · held **0 %** |
| ARMED `--scale 1.0 --max-steps 150` (개입 120) | 잔차 **0.130** / alpha **0.983** / 표본 51(포화 69 제외) · dp_ratio **1.000** · held **0 %** · **조작자 손맛 확인 양호** |

모든 개입 스텝에 `substeps=2`(창당 타깃 3회 갱신 = 첫 타깃 + 서브스텝 2)가 기록됐고
`governed=0`이다 — 예산이 governor보다 타이트해 **먼저 묶는** 설계대로의 동작이다. ⚠️ 단 이 세 run은 **첫 타깃만 집계하던 코드**로 측정됐고, shipped config에서는 `_paced_request`가 요청을 `ACTION_SCALE/3`(0.00417 m)로 깎아 서브스텝 governor 캡(`v_max/substep_hz`=0.0050 m)에 **닿을 수 없다** — 즉 `governed=0`은 "창 전체에서 절삭 없음"이 아니라 애초에 절삭이 불가능했다는 뜻이다.
원인 규명·금지사항·5개 실측표는 [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md) §9,
요약은 [`docs/testing/08_OPEN_GAPS.md`](docs/testing/08_OPEN_GAPS.md) G24.

**2026-07-29 첫 실물 production-model E2E smoke 성공.** 실제 UR7e에서
`ENGAGE=GELLO`, `DISENGAGE=policy`를 확인했고, Kanu에 online transition **201개**
(intervention **153개**)가 들어가 learner **102 step / gradient 204 / policy version 2**까지
진행했다. actor의 RPC deadline 예외 뒤 controller도 FPC에서 STJC로 자동 복귀했다.

**크롭 불일치(G15)는 재학습이 아니라 분리(decoupling)로 해결됐고 실물 actor 경로에도
들어갔다.** 🔴 **단 `5e508d3` merge가 이 전략을 되돌리는 중이다** — `8bd3248`이 recorder
take를 **선택한 크롭 기준으로** classifier item으로 재export하고
(`ur_env/learner/classifier_dataset.py` docstring이 "all three must see the same pixels"라고
명시한다), `b34f49a`가 분류기에 공유 trunk feature를 준다. 즉 방향이 **분리 → 크롭 기준
재학습**으로 바뀌었다. 재학습된 가중치는 **아직 없다.** 아래 서술은 그 전 기준이다. 액터가 분류기에게 **무크롭 원본 JPEG를 sidecar로 따로** 보낸다
(`ur_env/classifier_sidecar.py`). 정책은 측정된 `IMAGE_CROP`을 그대로 유지한다. 같은 변경에서
checkpoint 디렉터리 해시(G19)도 고쳤다. GUI에서 실제 episode의 마지막 classifier
확률/threshold/verdict가 표시되는 것까지 관측했다. MANUAL에서도 이 telemetry는 계속 돈다.
장시간 사후 감사를 위한 별도 영구 verdict 로그 정리는 여전히 남아 있다.

> **이전 판 문구(보존):** *"안 되는 것 — RL 루프의 reward. 뷰어는 믿어도 되고 RL reward는
> 믿으면 안 된다."* 이 경고는 sidecar 이전 기준이다. sidecar 도입 뒤 뷰어와 RL 경로는 **같은 그림**을 봤다
> (같은 무크롭 JPEG, 같은 `decode_classifier_image()` 레시피). 🔴 **`5e508d3` merge로 다시
> 갈라졌다** — 뷰어는 무크롭 픽셀, production RL 경로는 크롭 유래 feature다(위 🔴 박스).

**아직 남은 것** — 첫 E2E에서 0.6/0.8 s 경계가 정상 832.3 ms reply를 stale로 잘못 거부한
문제는 현재 1.5/2.0 s bounded 값으로 완화했다. 하지만 장시간 run에서 learner/GPU contention과
RPC tail latency가 어떻게 변하는지는 계속 계측해야 한다. classifier는 현재 정확도가 충분하지
않아 MANUAL이 기본이고, AUTO를 production 기본으로 되돌리려면 새 데이터로 재학습·재검증해야
한다. ⚠️ **그 재학습은 이제 데이터가 아니라 도구가 막고 있다** — 코퍼스는
`junhyeong_ai:~/hil-serl-data/datasets/`로 옮겨졌지만 `cube_classifier_pipeline.py`와 채점용
`.venv-train`은 kanu의 FM 스택 트리에 있었고 **의도적으로 복사 대상이 아니었다.**
`third_party/hil-serl`에도 없다. sidecar가 고치지 못하는 cam1 **가림(occlusion)**과
`08_OPEN_GAPS.md`의 G27/G28 (실행 액션 기록 정합성)도 남아 있다. schema-3 lineage의
transition/학습은 새 서버에서 실제로 진행 중이며, MANUAL `MARK SUCCESS` 버튼으로 끝낸
episode의 one-shot provenance만 별도로 한 번 확인하면 된다.

🔴 **그리고 시간 초과가 진짜 종료로 학습되고 있다 — `08_OPEN_GAPS.md` G35.**
`UR7eEnv.step`이 `truncated`를 **리터럴 `False`**로 반환하고 `MAX_EPISODE_LENGTH` 도달을
`done=True`에 섞어 넣는다 → `build_data`가 `masks=0.0`을 저장하고 critic이 "100스텝에서 세상이
끝난다"고 배운다. episode마다 정확히 한 줄이다. **기존 결함이고 이번 작업과 무관**하지만,
`END EPISODE`가 이 스택에서 `truncated=True`를 내는 **최초의 코드 경로**라 거기서 드러났다
(그래서 지금은 `truncated`가 우연히 ABORT의 고유 표식이다 — 고치면 그 성질이 사라진다).
⚠️ **고치면 `masks` 의미가 바뀌므로 새 lineage가 필요하다** — learner fingerprint에
`MAX_EPISODE_LENGTH`가 **없어서**(G18, `ACTION_SCALE`과 같은 이유) 그 불일치는 조용히 통과한다.
canonical offline demo 2,037개는 영향 없다(진짜 terminal이다).

**개입 쪽에서 남은 것 3개** (실기 차단은 아니지만 알고 있어야 한다 — 차례로
`08_OPEN_GAPS.md` **G32 / G33 / G34**):

1. **`suspend_follower()` 호출부는 여전히 배선되지 않았다 — 그러나 G32의 위험 자체는
   2026-07-30 저녁 `END EPISODE` 작업에서 다른 수단으로 닫혔다.** `UR7eEnv.suspend_follower`는
   아직 production 호출부 0(`rg suspend_follower` → `tests/test_intervention_follower.py`뿐)이고
   docstring도 "NOT WIRED UP YET"인 채다. 대신 `remote_actor._park_follower`가 새로 생겨
   **모든 blocking operator wait 앞에서** `await_follower_quiescent()`(내부적으로
   `disarm_intervention_follow`)를 부른다 — 공유 terminal 경로 `_close_episode`가
   `wait_for_home_approval` **전에**, 기동 시퀀스는 첫 `wait_for_scene_ready` **전에**
   `env.reset()`으로. 대기 중 재arming도 불가능하다: **승격 지점은 코드 전체에서
   `GelloIntervention._update_follow_arming` 하나뿐이고 그것은 RL 스레드의 step 경계에서만
   도는데, 대기 중에는 바로 그 스레드가 블록돼 있다.**
   ⚠️ 그래도 **실기 미검증**이고 `WAIT_HOME_APPROVAL`이 데드맨을 안 보는 것 자체는 그대로다.
   G35 표기와 마찬가지로 `08_OPEN_GAPS.md`는 다른 세션이 소유한다 — 그 파일의 G32 문구가
   이 문단보다 낡았을 수 있다.
   > **이전 판(보존):** *"`remote_actor.py`의 `WAIT_SCENE_READY`/`WAIT_HOME_APPROVAL`은 RL
   > 스레드를 무한 블록하고 **후자는 데드맨을 아예 보지 않는다** → 그 대기 화면에서 GELLO를
   > 잡으면 팔이 따라온다. 현재 완화책은 조작자 안내뿐이다."*
2. **포화 transition의 서버측 제외가 미구현.** 예산이 빠졌으므로 사람이 빠르게 움직인 창은
   기록 액션이 실제 이동을 **과소** 진술한다(측정은 `info["intervention_saturation"]`로 된다).
   서버가 그런 전이를 버리려면 proto 신규 필드 + pb2 재생성 + `SCHEMA_VERSION` bump가 필요하고,
   **protobuf가 unknown field를 조용히 버리므로 반쪽 업그레이드는 무증상 오염**이다.
3. **`tests/run_real_hil.py`는 여전히 `DefaultUR7eEnvConfig`를 쓴다 → 워크스페이스 박스가 꺼져
   있다(G1).** 예산이 빠진 지금 박스가 **유일한 위치 상한**이므로, 그 러너로 `--arm` 하는 것이
   production 3-CLI 경로보다 **덜 안전하다.** 손맛 확인용으로 쓰되 이 사실을 알고 쓴다.

## 읽는 순서 — 이 셋만 읽고 멈춰라

1. **이 파일** — 무엇을 만들고 있고 어디를 봐야 하는지. (2분)
2. [`serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) —
   **현재 진입점.** 첫 실물 E2E의 증거, 정확한 PASS/미완료 경계, HIL 제어권 의미, 다음 구현
   우선순위와 재현 CLI. (15분)
3. [`docs/testing/08_OPEN_GAPS.md`](docs/testing/08_OPEN_GAPS.md) — 지금 무엇이 깨져 있는지.
   **G15(크롭 불일치)가 어떻게 sidecar로 닫혔고 무엇이 안 닫혔는지(가림), 워크스페이스 박스가
   꺼져 있던 이유(G1)가 여기 있다.**

그다음은 **하려는 일 하나만** 아래에서 골라 읽는다. 나머지는 필요할 때 펼친다 —
순서대로 다 읽지 마라.

## 문서 지도

### A. 상태를 더 깊이 이해하려면

| 문서 | 무엇이 들어 있나 |
| --- | --- |
| [`serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) | **최신 정본.** 실물 E2E, operator episode 상태기계, 3-CLI와 다음 방향 |
| [`serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md`](serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md) | 첫 E2E 이전의 상세 리그 조사 기록. 최신 상태 지침은 위 문서가 대체 |
| [`serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md) | 전체 기록. learner 구현 §1–10 / actor·하드웨어 §11 / classifier 조사 §12 |
| [`serl_ur_infra/REWARD_CLASSIFIER_THRESHOLD_KO.md`](serl_ur_infra/REWARD_CLASSIFIER_THRESHOLD_KO.md) | 0.85 → 0.5 → 0.2의 **역사적** 근거와 07-29 누출 감사. 현재 runtime 기본값은 사용자 결정으로 다시 **0.5**이며 코드 상수가 권위다 |
| [`docs/testing/README.md`](docs/testing/README.md) | 하드웨어·통신 검증 런북 인덱스(00~09) + 항목별 PASS/미검증 상태표 |
| [`serl_ur_infra/README.md`](serl_ur_infra/README.md) | env 설계 규약(좌표계·액션 계약·컨트롤러 격차) + `serl_ur_infra/` 문서 인덱스 |
| [`serl_ur_infra/REMOTE_ACTOR_GRPC.md`](serl_ur_infra/REMOTE_ACTOR_GRPC.md) | gRPC 전송 계약 v2 — 서버 entrypoint 3종의 차이 (영문) |

### B. 무언가를 실행하려면 (런북)

| 하려는 일 | 문서 |
| --- | --- |
| 셋업 · 빌드 · 인터프리터 함정 · 비상 정지 | [`docs/testing/00_SETUP_AND_SAFETY.md`](docs/testing/00_SETUP_AND_SAFETY.md) |
| 실물 HIL 세션 기동 (운영용 3-CLI, preflight, actor) | [`docs/testing/09_HIL_ACTOR_RUNBOOK.md`](docs/testing/09_HIL_ACTOR_RUNBOOK.md) — 정상 운용은 `run_hil_server.sh` / `run_hil_hardware.sh` / `run_hil_session.sh` 세 terminal |
| 서버에서 learner 띄우기 | **`./run_hil_server.sh` 하나가 정상 경로다**(환경변수 0개). 수동 CLI·계약 감사는 🗄️ [`serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md`](serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md) — **kanu 시절 기록이고 실행 절차가 아니다.** 호스트·GPU·경로는 [`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md)가 우선 |
| 녹화 take를 learner용 offline demo로 변환 | [`serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md`](serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md) (`40b99f8`) — **`--outcome success\|truncated`는 사람이 명시한다.** 변환기는 성공을 추측하지 않는다. learner는 offline demo가 0이면 학습을 시작하지 않는다 |
| 라이브 reward classifier 뷰어 보기 | [`serl_ur_infra/REWARD_CLASSIFIER_LIVE_KO.md`](serl_ur_infra/REWARD_CLASSIFIER_LIVE_KO.md) — **2026-07-29 실기 검증 완료.** 터미널 4개 절차·인터프리터 함정·크롭 주의·트러블슈팅 |
| mock RViz로 개입 경로 확인 (실기 위험 0) | [`serl_ur_infra/RVIZ_HIL_TEST_CLI.md`](serl_ur_infra/RVIZ_HIL_TEST_CLI.md) |
| GELLO 개입 손맛·좌표계 **실기** 검증 | [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md) §4.5·§9 — `serl_ur_infra/tests/run_real_hil.py`. 기본 `DRY_RUN`이고 `--arm`을 줄 때만 움직인다. **정책이 zero 고정이라 learner도 gRPC 서버도 필요 없다** — 3-CLI 운영 워크플로우와 혼동하지 말 것(실제로 혼동이 있었다). 컨트롤러 요구도 다르다(FPC) → [`docs/testing/00_SETUP_AND_SAFETY.md`](docs/testing/00_SETUP_AND_SAFETY.md) §3.5 |
| GELLO로 실기 팔 텔레옵 (HIL 개입이 이 경로 위에 있다) | [`docs/ros2/GELLO_UR7E_EEF_MODE.md`](docs/ros2/GELLO_UR7E_EEF_MODE.md) · 조인트 모드는 [`GELLO_UR7E_REAL_ROBOT.md`](docs/ros2/GELLO_UR7E_REAL_ROBOT.md) |
| 처음부터 환경 세팅 / 세션 전 프리플라이트 | [`docs/ros2/GELLO_UR7E_SETUP_CLI.md`](docs/ros2/GELLO_UR7E_SETUP_CLI.md) |
| 그리퍼만 단독으로 | [`docs/ros2/GELLO_UR7E_GRIPPER.md`](docs/ros2/GELLO_UR7E_GRIPPER.md) |
| 카메라가 안 뜰 때 | [`docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md`](docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md) · [`docs/testing/06_SENSORS.md`](docs/testing/06_SENSORS.md) |

### C. 특정 진행 중 작업을 이어받으려면

| 작업 | 시작점 |
| --- | --- |
| **분류기를 개입·학습에 연결** | [`serl_ur_infra/REWARD_TO_RL_INTEGRATION_KO.md`](serl_ur_infra/REWARD_TO_RL_INTEGRATION_KO.md) — reward/termination 계약, 개입 이중 라우팅, RLPD 50:50, 연결 순서. **코드에서 직접 추적해 쓴 문서다.** 아래 두 블로커가 선행 조건 |
| ~~크롭 불일치 해소~~ → **sidecar verdict 가시화** | `08_OPEN_GAPS.md` G15 → `ur_env/classifier_sidecar.py` (모듈 docstring이 설계 근거 전부). production actor 배선은 첫 실물 E2E에서 사용됐다. 남은 것은 per-transition probability/verdict GUI·영구 로그 검증이다. **`IMAGE_CROP`을 지우는 건 여전히 해결이 아니다** |
| ~~checkpoint 로딩~~ → **해결됨(G19)** | `checkpoint_sha256()`이 `classifier_sidecar.directory_sha256()`에 위임해 orbax **디렉터리**를 해시한다. 두 `DEFAULT_*_SHA256` 상수도 은퇴 체크포인트(`e329986b…`)에서 교체됐다 |
| **canonical demo artifact** (learner 시작 조건) | ✅ 2026-07-29 사람 승인·생성 완료. `take_23` 제외 23 take → 2,037 transition, SHA256 `f9718558…032fa`; laptop3와 Kanu strict-load 통과. 경로·품질 주의는 [`serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md`](serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md) + `08_OPEN_GAPS.md` G20 |
| actor entrypoint 실기 재실행·timeout 계측 | [`serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) + [`docs/testing/09_HIL_ACTOR_RUNBOOK.md`](docs/testing/09_HIL_ACTOR_RUNBOOK.md) |
| 개입 루프·좌표계·메타데이터 | [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md) |
| 장애 주입 매트릭스 (거의 미검증) | [`docs/testing/07_FAILURE_INJECTION.md`](docs/testing/07_FAILURE_INJECTION.md) |

### D. 🗄️ 낡음 / 대체됨 — 찾더라도 따르지 말 것

| 문서 | 왜 |
| --- | --- |
| [`serl_ur_infra/HIL_SERL_STATUS_AND_NEXT.md`](serl_ur_infra/HIL_SERL_STATUS_AND_NEXT.md) | 2026-07-24 서버 핸드오프 아카이브. agentlace 5588/5589 · `train_rlpd.py` · jax 0.4.35 핀을 권고한다 — **그대로 준비하면 learner가 즉사한다.** 문서 머리에 대조표가 있다 |
| [`serl_ur_infra/RL_RECEIVE_SERVER.md`](serl_ur_infra/RL_RECEIVE_SERVER.md) · [`HIL_RLPD_RECEIVE_SERVER_KO.md`](serl_ur_infra/HIL_RLPD_RECEIVE_SERVER_KO.md) | receive-only 마일스톤 기록. 브랜치·체크포인트가 은퇴했고 **19-D `state` 순서를 v1(틀린 순서)로 적은 곳이 남아 있다** |
| [`serl_ur_infra/ACTOR_ADAPTER.md`](serl_ur_infra/ACTOR_ADAPTER.md) | agentlace 로컬 어댑터(`scripts/train_rlpd_actor.py`) 기준. **실기 경로가 아니다** — transition 계약 설명만 유효 |
| [`docs/rl/GELLO_UR7E_HIL_SERL_PLAN.md`](docs/rl/GELLO_UR7E_HIL_SERL_PLAN.md) · [`GELLO_UR7E_SERL_ENV_STATUS.md`](docs/rl/GELLO_UR7E_SERL_ENV_STATUS.md) | 2026-07-24 채택 계획과 첫 커밋 스냅샷. upstream 구조 설명만 참고 가치 |
| [`docs/ros2/GELLO_UR_ROS2_PLAN.md`](docs/ros2/GELLO_UR_ROS2_PLAN.md) · [`GELLO_UR_ROS2_BRINGUP.md`](docs/ros2/GELLO_UR_ROS2_BRINGUP.md) | UR5e + Jazzy 시절. 이 브랜치는 UR7e + Humble이고 실기 경로가 이미 있다 |
| [`docs/ros2/GELLO_UR7E_EEF_TELEOP_PLAN.md`](docs/ros2/GELLO_UR7E_EEF_TELEOP_PLAN.md) | EEF 설계·근거 아카이브. 대체된 부분에 `⛔ SUPERSEDED` 표시가 붙어 있고, 조작자 정본은 `GELLO_UR7E_EEF_MODE.md`다 |
| [`README.md`](README.md) (루트) | 2026-07-20. Panda/UR5e sim 소개 중심이라 HIL-SERL이 없다. 데이터셋·HF 절만 유효 |
| [`docs/sim/`](docs/sim) · [`docs/ros2/GELLO_FRANKA_RVIZ.md`](docs/ros2/GELLO_FRANKA_RVIZ.md) · [`GELLO_ROS2_CONTROL_REFERENCE.md`](docs/ros2/GELLO_ROS2_CONTROL_REFERENCE.md) · [`ros2/`](ros2) · [`docs/reference/`](docs/reference) | Panda/Jazzy 계보이거나 upstream 원본. **이 작업 범위 밖** |

### E. 모방학습(ACT/Diffusion/FM) 시대 문서 — 유효하지만 HIL과 다른 스택

같은 로봇·같은 브리지를 쓰지만 **RL 경로와 코드가 다르다.** 섞지 말 것.
[`docs/ros2/GELLO_UR7E_ACT_DEPLOY.md`](docs/ros2/GELLO_UR7E_ACT_DEPLOY.md)(유일하게 실기 검증) ·
[`GELLO_UR7E_DIFFUSION_DEPLOY.md`](docs/ros2/GELLO_UR7E_DIFFUSION_DEPLOY.md) ·
[`GELLO_UR7E_FM_DEPLOY.md`](docs/ros2/GELLO_UR7E_FM_DEPLOY.md)(뒤 둘은 실기 미검증) ·
[`ros2_ur_ws/src/gello_policy/README.md`](ros2_ur_ws/src/gello_policy/README.md) ·
[`ros2_ur_ws/REMOTE_DIFFUSION_RUNBOOK.md`](ros2_ur_ws/REMOTE_DIFFUSION_RUNBOOK.md).
데이터 기록은 [`ros2_ur_ws/src/gello_recorder/README.md`](ros2_ur_ws/src/gello_recorder/README.md) ·
[`docs/ros2/GELLO_UR7E_RECORDING.md`](docs/ros2/GELLO_UR7E_RECORDING.md).
단위 확인은 [`docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md`](docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md).
종결된 조사 기록: [`ros2_ur_ws/gello_logs/experiments/README.md`](ros2_ur_ws/gello_logs/experiments/README.md)(startup-snap,
수정 반영 완료) · [`docs/ros2/GELLO_DIFFUSION_ENSEMBLE_OFFLINE.md`](docs/ros2/GELLO_DIFFUSION_ENSEMBLE_OFFLINE.md)(결과 부정적, 기능 default-OFF).

## 코드 지도

```
serl_ur_infra/
  ur_env/envs/ur7e_env.py          get_im, clip_safety_box, go_to_reset, 관측 조립
                                   + **개입 배경 추종 스레드**: _start_follow_thread /
                                   _follow_loop / _follow_tick(30 Hz), 소유권 mux
                                   _emit_arm_command(유일한 관절 명령 출구),
                                   arm_/disarm_intervention_follow,
                                   await_follower_quiescent, suspend_follower(미배선),
                                   _harvest_follow_window(창 순변위 -> 기록 액션, norm 축소),
                                   open_gripper_for_reset
  ur_env/envs/config.py            속도 3층(ACTION_SCALE/GOVERNOR/UPSAMPLER), 카메라 토픽,
                                   INTERVENTION["follow_mode"]="background"|"in_window"
                                   (**왜 창 밖으로 나가야 했는지 주석에 실측과 함께 있다**)
  ur_env/envs/wrappers.py          GelloIntervention, 데드맨, 그리퍼 페널티,
                                   follow_xi(배경 추종 프로토콜)
  ur_env/envs/policy_delta_controller.py  스레드 안전 컨트롤러 — RLock, _commit()이 유일한
                                   (T_cmd, q_cmd, _last_issued) writer, _hold()는 _last_issued 반환
  ur_env/envs/leader_stream.py     개입용 리더 30 Hz 재샘플링 — One-Euro 이식(bridge_stages.py
                                   비트 동일, dt는 **벽시계**) + InterventionBudget(창당 변위
                                   예산 — **`in_window` 전용, 기본 경로에는 없다**).
                                   **설계 근거가 모듈 docstring에 전부 있다**
  ur_env/envs/frame_wrappers.py    RelativeFrame, Quat2EulerWrapper
  ur_env/envs/ros_backend.py       rclpy 백엔드, 250 Hz 업샘플러
  ur_env/remote_actor.py           actor 루프, 전이 생성·전송, sidecar 부착 계측
                                   + END EPISODE: _consume_operator_abort(iteration당 pre/post
                                   step 2회), _terminal_reason(aborted=True -> OPERATOR_ABORT),
                                   _park_follower(모든 blocking wait 앞 — G32 완화),
                                   _close_episode / _open_episode(공유 terminal 경계),
                                   _OperatorReporter.clear_terminal
  ur_env/operator_session.py       GUI 상태/서비스, MANUAL/AUTO·HOME/scene-ready operator gate
                                   + ABORT_EPISODE_SERVICE(/hil/abort_episode), _on_abort_episode,
                                   consume_operator_abort(one-shot, (run_id, episode_id) scope),
                                   resolve_follow_controls
  ur_env/classifier_sidecar.py     분류기 sidecar 계약 — build/validate/decode, 정지·2 Hz 게이트,
                                   directory_sha256. **설계 근거가 모듈 docstring에 전부 있다**
  ur_env/rlpd_receive_server.py    서버 ingress + RewardClassifierRuntime + checkpoint_sha256
  ur_env/learner/                  RLPD learner, checkpoint, fingerprint
  ur_experiments/cube_in_cup.py    태스크 config (측정값 전부 여기, IMAGE_CROP 포함)
  scripts/run_remote_rlpd_actor.py actor entrypoint
  tests/run_real_hil.py            실기 개입 러너 (파일 상단 주석이 안전 설계를 설명)
ros2_ur_ws/
  run_hil_server.sh                Terminal 1: 서버 learner 검증/재사용·기동 + SSH tunnel
                                   (기본값 junhyeong_ai/GPU 0/HIL_REMOTE_DATA_ROOT — 5594d0e)
  run_hil_hardware.sh              Terminal 2: UR7e + Robotiq + passive GELLO supervisor
  run_hil_session.sh               Terminal 3: cameras + GUI + preposition/preflight + actor
                                   + 하드웨어 재기동 복구 루프: hil_hardware_owner_lines /
                                   hil_wait_for_hardware_recycle / hil_hardware_topics_ready /
                                   hil_robot_state_ready(dashboard RUNNING+NORMAL)
  run_hil_actor.sh                 actor 실행 래퍼 (11단계 preflight + controller cleanup)
                                   + 종료 코드 계약(75/1/70/>=128, HIL_ACTOR_EXIT_MAP),
                                   hil_start_progress_watch(env_step>=0 증거),
                                   hil_filter_qt_font_noise
  run_hil_preposition.sh           RESET pose 이동 + controller handoff proof 생성
                                   (재시도 경로의 HOME 복귀는 GUI가 아니라 **여기**가 한다)
  run_hil_gui.sh                   데드맨/개입 GUI (**Qt 폰트 필터 없음 — 시스템 Qt5라 불필요**)
  _hil_deadman_check.py            deadman 채널 생존 증명 — --require engaged|disengaged
  launch_cameras.sh                RealSense 2대 (시리얼 자동 해석) + Qt 폰트 필터
  src/ur_gello_bringup/.../gello_hil_gui_node.py   END EPISODE 버튼(2-click, 데드맨 먼저 release)
  src/ur_gello_bringup/.../hil_actor_status.py     abort_episode_enabled(MANUAL/AUTO 공통)
  run_classifier_viewer.sh         라이브 분류기 뷰어 (랩톱 CPU)
  run_remote_classifier_viewer.sh  라이브 분류기 뷰어 (서버 GPU + SSH 터널)
```

## 반드시 지킬 것

- **GELLO Dynamixel에 토크를 걸지 말 것.** 수동 read-only 리더다.
- **gRPC 코드는 `/home/laptop3/venvs/gello-hil-actor/bin/python`으로만.** 시스템 `python3`의
  grpcio 1.30.2가 손상돼 오류 없이 100% CPU로 무한 정지한다.
  **단 "무조건 venv"는 아니다** — `tests/run_real_hil.py`는 gRPC를 안 쓰고 rclpy를 쓰므로
  **시스템 `python3`가 맞다.** 그리고 **ROS 러너에서 `PYTHONPATH`를 덮어쓰면 rclpy가 사라진다**
  (`RuntimeError: rclpy not available` — 2026-07-30 실기에서 발생). pytest 정본 명령은 반대로
  덮어쓰는 게 맞다. 두 규칙의 대비는 [`docs/testing/00_SETUP_AND_SAFETY.md`](docs/testing/00_SETUP_AND_SAFETY.md) §3.4에 표로 있다.
- **`IMAGE_CROP`을 "분류기가 안 맞으니" 지우지 말 것.** 데이터셋 측정값이고 정책이 1차 소비자다.
  해결은 **분류기에게 무크롭 sidecar를 따로 주는 것**이었다(`ur_env/classifier_sidecar.py`).
  🔴 **`5e508d3` merge가 이 방향을 크롭 기준 재학습으로 되돌리는 중이다** — 그래도
  **`IMAGE_CROP`을 지우지 말라는 규칙은 그대로다.** 오히려 새 방향에서는 크롭이 세 소비자
  전부의 공통 기준이 되므로 더 중요해졌다.
  *(이전 판은 "해결은 classifier 재학습이다"라고 적었다 — 재학습은 채택되지 않았다. 분리를
  택한 덕분에 `REWARD_CLASSIFIER_THRESHOLD_KO.md`의 측정값이 전부 살아남았다.)*
- **개입이 빳빳하다는 이유로 가속도 제한 · `target_stale_s` · `soft_start_s`를 되돌리지 말 것.**
  실측은 셋 다 **반대로 간다**: `UPSAMPLER.max_accel_rad_s2`(8.0) 제거 → 관절 완전정지
  16 % → **76 %**(리플 2.27 → 4.17), `target_stale_s` 0.30 → 0.50 → 정지 17.5 % → **54.5 %**,
  `soft_start_s` 0.7 → 0 → 리플 1.92 → **4.09**. 셋은 원인이 아니라 낮은 갱신 주기가 만든
  stop-and-go를 **완화하고 있던 것**이다. 근거(창 조건·메커니즘 포함)는
  [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md) §9(특히 §9.3),
  요약은 `08_OPEN_GAPS.md` G24.
- **개입 액션을 축별 `np.clip`으로 자르지 말 것 — norm 비례 축소만.** 근거 2개다:
  (1) **방향 왜곡** — 포화된 대각을 축별로 자르면 기록 경로가 꺾인다(`[2.0, 0.5]` →
  `[1.0, 0.5]`). 이건 `GelloIntervention._expert_delta_xi`에 이미 주석으로 박혀 있던 규칙이다.
  (2) **`RelativeFrame`이 이 벡터를 회전시킨다** — `transform_action_inv`가 `blockdiag(R, R)`을
  곱하는데 회전은 각 3-벡터의 **norm은 보존하지만 축별 최댓값은 보존하지 않는다.** 축별 clip은
  norm을 √3 = 1.73까지 남겨 두므로 `[1.0, 1.0, 0]`이 회전 뒤 `[1.41, 0, 0]`으로 나온다.
  **2026-07-30 실기에서 이것으로 actor가 즉시 죽었다** (`ActorProtocolError: executed_action must
  be within [-1, 1]`, `remote_actor.build_data` → `validate_action`; 수정 `d6965a9`).
  norm ≤ 1이면 **어떤 프레임에서도** 모든 성분이 `[-1, 1]` 안이다. 옛 예산 경로가 이 문제를 안
  겪은 이유도 같다 — 그쪽 clamp가 norm 기준이라 프레임 변환을 공짜로 통과했고, **예산을
  제거하면서 그 보장이 같이 사라졌다.** 회귀 테스트가
  `tests/test_intervention_follower.py`에 있다(축별 clip으로 되돌리면 FAIL).
- **`ACTION_SCALE`을 "개입이 느리다"는 이유로 올리지 말 것.** 그건 안전 한계가 아니라
  **정책의 액션 의미 자체**다 — 0.0125 m/step × 10 Hz = **12.5 cm/s가 정책의 최고속**이다.
  올리면 사람이 **정책이 실행할 수 없는 시범**을 보이게 되고, canonical demo 2,037 transition과
  액션의 의미가 갈리는데 learner fingerprint에 `ACTION_SCALE`이 **없어서**(G18) 조용히 통과한다.
  **개입 경로에서 `ACTION_SCALE`은 이제 "기록 스케일"일 뿐이다** — 창 순변위를 `ACTION_SCALE`로
  나눈 뒤 norm으로 축소해 저장할 때만 쓰인다. 팔의 실제 상한은 **governor(`v_max` 0.15 m/s) +
  워크스페이스 박스 + 250 Hz 업샘플러** 셋뿐이다.
  ⚠️ **그래서 지금은 개입이 `ACTION_SCALE`을 넘을 수 있다** — 넘으면 포화로 기록되어
  **저장이 실제 움직임을 과소기록**한다(`info["intervention_saturation"]`). 이것이 "올리지 말
  것"의 새 맥락이다: 예전엔 예산이 초과를 **막았고 지금은 막지 않는다.** 상세는
  [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md) §10.2와
  `08_OPEN_GAPS.md` G33.
  손맛 확인용으로는 `serl_ur_infra/tests/run_real_hil.py --scale`을 쓴다 — 3층
  (`ACTION_SCALE`/`GOVERNOR`/`UPSAMPLER`)을 **함께** 곱하고 그 배율을 CSV 헤더에 남긴다.
  **이전 판(보존):** *"개입 변위 예산도 여기서 직접 파생된다"* — `edbb3f5`가
  `InterventionBudget`을 개입 **제어** 경로에서 제거했으므로 더 이상 사실이 아니다.
  `follow_mode="in_window"`에서는 여전히 맞다.
- **테스트는 passed 수를 볼 것 — 그리고 어느 인터프리터인지 같이 적을 것.** PYTHONPATH에서
  `serl_launcher`가 빠지면 조용히 떨어지고 skip 사유가 거짓말을 한다. 2026-07-31 merge
  `5e508d3` 기준선은 **804 passed / 14 skipped / 1 xfailed** (14.27 s 실측,
  `/home/laptop3/venvs/gello-hil-actor/bin/python`, numpy 2.2.6).
  `ur_gello_bringup` 패키지 suite는 **별도로 489 passed**(시스템 `python3` + ROS overlay,
  7.70 s 실측 — 두 숫자를 합치지 말 것. 인터프리터도 PYTHONPATH도 다르다).
  🪤 **`ur_gello_bringup`도 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`이 필수다** — 그게 없으면
  테스트가 하나도 안 돌고 **collection에서 죽는다**(시스템 pytest 6.2.5 ↔ `~/.local`의
  anyio 4.13.0이 pytest 7의 `_pytest.scope`를 요구. 그 패키지 `pytest.ini`는
  `launch_testing` 계열만 끄고 anyio는 안 끈다). 4월부터 그랬으므로 07-30의 489도 이미
  이 플래그로 측정된 값이고, 기록에만 빠져 있었다.
  > **이전 판(보존):** *"2026-07-30 저녁(조작자 경로 4건) 기준선은 **768 passed /
  > 11 skipped / 1 xfailed** (13.86 s 실측 …)"*, 그 이전 `d6965a9`는 701,
  > `ur_gello_bringup`은 436.
  **`xfailed 1`을 빼고 인용하지 말 것** — 그건 통계 잡음이 아니라 **알려진 결함의 표식**이다
  (`ee8af5e`가 박은 strict xfail: 저장 액션이 IK line-search 경로에서 실행 액션을 과대 진술할 수
  있다. 고치면 XPASS로 터진다). 재현 명령 정본은
  `serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md`의 「테스트 — 이 명령 그대로」 절이다
  (줄번호로 인용하지 말 것 — 밀린다. ⚠️ 그쪽 판은 다른 세션 작업 2파일을 빼는
  `--ignore=` 두 줄이 더 붙은 07-30 스냅샷이라 인용된 passed 수도 아래와 다르다):

  ```bash
  cd /home/laptop3/gello_software
  set +u; source /opt/ros/humble/setup.bash; source ros2_ur_ws/install/setup.bash; set -u
  OVERLAY=$(python3 -c "import ur_gello_bringup,os;print(os.path.dirname(os.path.dirname(ur_gello_bringup.__file__)))")
  env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher:$OVERLAY" \
    /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
    -p no:cacheprovider serl_ur_infra/tests
  ```

  계보: 333 → 337(`40b99f8`) → 429(classifier sidecar) → 497(07-30 오전, `4197f5b` 이전)
  → 579(`4197f5b`) → 595(`ee8af5e`) → 701(`d6965a9`; 배경 추종 스레드 + 컨트롤러
  스레드 안전화 + norm 축소 회귀 — `test_intervention_follower` 신규)
  → 768(07-30 저녁; END EPISODE + DISENGAGED gate + 재기동 복구 —
  `tests/test_operator_abort.py`(26) · `tests/test_actor_abort_lifecycle.py`(23) 신규)
  → **804**(`5e508d3` merge; 공유 frozen-trunk feature + 전처리 규칙 일원화 —
  `test_observation_preprocess.py`(12) · `test_shared_feature_pipeline.py`(7) ·
  `test_classifier_dataset.py`(8, 2 skip) 신규. skip 11 → 14는 전부 의도된 게이트다:
  실코퍼스 2개 + `RUN_HIL_SERL_ACTUAL_FEATURE_AGENT=1` 1개).
  옛 문서에 남은 333·337·429·497·579·595·701·768은 전부 이전 값이다.
  🪤 **인터프리터를 안 적은 "passed 개수"는 무의미하다.** 같은 명령을
  `/home/laptop3/venvs/hilserl/bin/python`(jax 0.5.3 있음, numpy 1.26.4)으로 돌리면
  jax 테스트가 더 돌아 **741 passed / 4 skipped / 1 xfailed**가 된다(skipped 11 → 4).
  gRPC·actor 경로의 정본 인터프리터는 actor venv이므로 **기준선은 actor venv 값**이다.

  ✅ **해소됨** — 한때 hilserl에서 `test_governor_dt.py::test_env_step_surfaces_governed_in_info`
  1건이 떨어졌다(`0.8485281248` vs `0.8485281374 ± 8.5e-10`). 원인은 numpy 승격 차이가
  아니라 **허용범위가 애초에 잘못됐던 것**이다: 액션 dtype이 계약상 `float32`라
  `xi = action * ACTION_SCALE`에 ~1e-7 상대오차가 실리는데 `rel=1e-9`로 잡혀 있었다.
  actor venv에서 통과한 건 우연이다. `rel=1e-6`으로 고쳐 **두 인터프리터 모두 통과**한다.
  잡으려는 회귀(캡이 안 물림 1.0, 3배 오차)는 1e-6에서 수십만 배 떨어져 있다.
  그리고 `serl_ur_infra/tests/test_env_fake_backend.py`는 pytest에서 **0개 수집**되므로 이
  총계에 **아무 흔적도 남기지 않는다** → `08_OPEN_GAPS.md` G25.
- **메인 브랜치는 여러 사람이 공유한다.** 머지·리베이스 전에 상대 checkout이 깨끗한지 확인할 것.
