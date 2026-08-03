# BC 정책 실기 평가 런북 (조작자용, 한국어)

> **상태: 실기 검증 완료 (2026-07-31 밤 ~ 08-02).** 이 절차 그대로 production HIL actor
> 파이프라인으로 실제 BC 평가가 실기 UR7e에서 돌았다. 아래 명령은 전부 복사해서 그대로
> 붙여넣을 수 있다. 한 단계씩 하고, 각 단계의 **「성공 표식」을 눈으로 확인한 뒤에만**
> 다음으로 넘어간다.
>
> 📖 **여러 정책(BC/FM/…)을 통틀어 "실기 평가를 어떻게 돌리나"의 최상위 가이드는
> [`POLICY_EVAL_KO.md`](POLICY_EVAL_KO.md)다.** 이 문서는 그중 **BC 경로의 상세 런북**이다.

---

## 1. 이게 뭐고, 뭐가 아닌가

다른 담당자가 서버(`junhyeong_ai`)에서 **BC(Behavior Cloning) 정책**을 학습했다
(artifact 형식 `hil-serl-bc-init`, 새로 녹화한 **34 take / 2,824 transition** 기반).
이 런북은 그 정책을 **production HIL actor 파이프라인 그대로** 실기 UR7e에 물려 "얼마나 잘
하는지" 보는 절차다. 로봇·카메라·GELLO·GUI·안전장치는 **평소 HIL 세션과 100% 같은 것**을
쓰고, 정책을 서빙하는 서버만 다른 프로세스(포트 50054)로 갈아 끼운다.
**HIL 온라인 학습이 아니다** — 학습도 reward classifier도 쓰지 않고, 성공 판정은 사람이 GUI에서
누른다. production learner(`:50053`)는 서버에서 계속 돌지만 **이 평가와 무관하며 절대 건드리지
않는다.** 하드웨어 준비·컨트롤러·preflight 상세는 중복하지 않았다 — 정본은
[`docs/testing/09_HIL_ACTOR_RUNBOOK.md`](../docs/testing/09_HIL_ACTOR_RUNBOOK.md)다.

---

## 2. 구조

```
laptop3 (production 스크립트 수정 0 — 환경변수 핀만 다르다)
  T2  ./run_hil_hardware.sh          (production 그대로: UR7e + Robotiq + GELLO)
  T3  EXPECTED_MODEL_ID=… EXPECTED_REWARD_AUTHORITY=local \
      EXPECTED_REWARD_MODEL_ID=… ./run_hil_session.sh --no-classifier-sidecar
        actor ──gRPC──► 127.0.0.1:50153
  T1  ./run_bc_server.sh  ──ssh tunnel 50153 → 서버:50054──►  junhyeong_ai
                                                               run_bc_policy_server.py
                                                               (GPU 0, learner와 공존)
  T4  ./run_bc_rollout_recorder.sh   (선택 — 순수 구독자, 로봇 스트림/카메라 녹화)

production learner(:50053)는 계속 산다 — 무접촉. 우리는 50053도 그 터널도 쓰지 않는다.
```

바뀌는 것은 **터널의 반대쪽 끝**뿐이다. actor는 예나 지금이나 `127.0.0.1:50153`만 본다.

---

## 3. 사전 조건 체크리스트

| # | 확인할 것 | 방법 / 조치 |
|---|---|---|
| 1 | **production `run_hil_server.sh` 터미널이 열려 있으면 먼저 `Ctrl-C`로 닫는다** | 그 터널이 `50153`을 잡고 있어서, 닫지 않으면 T1이 포트 충돌로 실패한다. 🟢 **그 `Ctrl-C`는 터널만 닫는다 — 서버의 learner는 그대로 살아 있다**(→ 09 런북 §5.4) |
| 2 | 50153이 비었는가 | `ss -ltn \| grep 50153` → **아무 줄도 안 나와야 한다** |
| 3 | 로봇 전원 ON, 펜던트 REMOTE 모드, E-STOP 손 닿는 곳 | 09 런북 §4.1 |
| 4 | 작업 공간 정리, 큐브와 컵 준비 | 사람이 손을 넣을 일이 없게 |
| 5 | 다른 텔레옵 브리지·이전 actor가 떠 있지 않은가 | 떠 있으면 preflight `[9]`가 거부한다. 끄고 시작 |
| 6 | GELLO는 손대지 않은 상태 | 세션은 **DISENGAGED**로 시작한다(09 §1.4) |

---

## 4. 단계별 실행

모든 명령은 다음 디렉터리 기준이다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
```

### 4.1 T1 — BC 서버 + 터널 (제일 먼저)

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_bc_server.sh
```

**성공 표식 (셋 다 봐야 한다)**

1. `BC_SERVER_RESULT=started`
2. `[bc-server] ready ...` 로 시작하는 줄 (서버가 실제로 정책을 로드하고 대기 중)
3. 그 뒤로 프롬프트가 **돌아오지 않는다** — 터널이 foreground로 붙잡혀 있는 상태가 정상이다

**이 창은 세션이 끝날 때까지 그대로 둔다.** 여기서 `Ctrl-C`를 누르면 BC 서버와 터널이
둘 다 내려가고, T3의 actor가 다음 RPC에서 죽는다.

**실패하면:** 스크립트가 서버 로그 **마지막 40줄**을 출력한다. 그 줄들을 그대로 복사해
메인 세션(담당자)에게 전달한다. 로봇을 켜기 전 단계이므로 위험은 없다.

### 4.2 (선택, 권장) dry handshake — **로봇을 전혀 쓰지 않는다**

정책 서버와 actor가 서로 말이 통하는지를 로봇 없이 30초 만에 확인한다. T1이 떠 있는
상태에서 **다른 터미널**에 붙여넣는다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
EXPECTED_MODEL_ID=bc-cube-in-cup-raw0731-bcinit-v1 EXPECTED_REWARD_AUTHORITY=local EXPECTED_REWARD_MODEL_ID=operator-manual-success-v1 ./run_hil_actor.sh --fake-env --no-classifier-sidecar
```

**성공 표식:** preflight가 통과하고 핸드셰이크 오류 없이 `BeginEpisode`까지 간 뒤 종료한다.
센서 토픽 경고(WARN)는 `--fake-env`에서 정상이다.

**실패하면:** `ActorProtocolError: server model_id is '...', expected '...'` 같은 메시지는
**핀 3종 중 하나가 서버 광고값과 다르다는 뜻**이다. 값을 임의로 바꾸지 말고 메시지를 그대로
담당자에게 전달한다(잘못된 정책을 실기에서 돌리는 것을 막는 게이트다).

### 4.3 T2 — 하드웨어 (production 그대로)

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_hil_hardware.sh
```

**성공 표식:** READY 배너와 세 토픽(`/joint_states`, `/robotiq_gripper/position_percent`,
`/gello/joint_states`)이 요구 rate를 만족한다는 출력.
**실패하면:** 09 런북 §4.2 / §5를 따른다 — 이 단계는 평소와 완전히 동일하다.
`ERROR: another HIL hardware bundle still owns …/hil-hardware-<uid>.lock`으로 **즉시 거부**되면
이전 번들의 프로세스가 아직 살아 있다는 뜻이다 — `ps aux | grep run_hil_hardware`로 감독
프로세스를 찾아 `kill -INT <pid>`로 정리하고(`[cleanup] complete`를 기다린다) 다시 실행한다.
락은 `flock`이라 **프로세스가 죽으면 자동으로 풀린다** — 남아 있는 락 *파일*은 무해하니
지우려 하지 말 것.

### 4.4 T3 — 카메라 + GUI + actor (핀 3개 + sidecar 끄기)

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
EXPECTED_MODEL_ID=bc-cube-in-cup-raw0731-bcinit-v1 EXPECTED_REWARD_AUTHORITY=local EXPECTED_REWARD_MODEL_ID=operator-manual-success-v1 ./run_hil_session.sh --no-classifier-sidecar
```

핀 3개는 "지금 붙은 서버가 정말 그 BC 정책인가"를 검사한다. `--no-classifier-sidecar`는
reward classifier를 끈다 — 이 평가에서 성공 판정은 **사람만** 한다.

**성공 표식 (순서대로)**

1. 카메라 2대와 HIL GUI가 뜬다. **GELLO는 손대지 않는다**(gate는 fresh DISENGAGED 3개)
2. 팔이 RESET 자세가 아니면 `run_hil_preposition.sh`가 자동으로 옮긴다 — ⚠️ **팔이 움직인다**
3. preflight `[1]`~`[11]` 전부 통과 → controller handoff
4. GUI가 `WAIT_SCENE_READY`를 표시하고 팔이 멈춘다

**실패하면:** 09 런북 §5의 실패 대응표를 그대로 쓴다. 단 `[6] TCP 127.0.0.1:50153 연결 실패`는
이 세션에서는 **T1(BC 서버)이 죽었거나 아직 안 떴다는 뜻**이다 — production `run_hil_server.sh`를
띄우지 말고 T1을 다시 확인한다.

**`preposition proof attempt 1/3 …` 이 보여도 당황하지 말 것 (정상 동작).** 예전에는 팔이 RESET
자세에 **제대로 도착했는데도** controller 상태를 단 한 번 읽어 그 순간 `inactive/inactive`가
보이면 세션이 그대로 죽었다(UR 드라이버의 `controller_stopper`가 로봇 프로그램이 잠깐만
멈춰도 모션 컨트롤러를 내린다). `31d6567`부터는 **최대 3회 재시도**하고, 그중 유일하게
양성인 상태 — **자세 증명을 통과한 정지된 팔을 아무도 잡고 있지 않은 both-inactive** — 이면
STJC를 **1회만 안전 재활성화**(움직임이 아니라 hold다)한 뒤 다시 검증한다. 그 뒤에도 실패하면
그때는 **진짜로** 뭔가 잘못된 것이다 → **펜던트에서 프로그램이 실행 중(▶ RUNNING)인지부터
확인**한다(정지돼 있으면 재활성화가 실패하고 스크립트가 그렇게 경고한다).
놉(기본값을 바꿀 일은 거의 없다): `HIL_PREPOSITION_PROOF_RETRIES`(기본 3) ·
`HIL_PREPOSITION_PROOF_RETRY_DELAY_S`(기본 2초) · `HIL_PREPOSITION_AUTOACTIVATE`(기본 1,
`0`이면 재활성화 없이 재시도만).

### 4.5 (선택) T4 — rollout 상세 녹화

평가 중 **로봇 쪽 원시 데이터**를 native rate로 함께 남기고 싶을 때만 연다. 로봇에 아무 명령도
보내지 않는 **순수 구독자**라, 띄우지 않아도 평가는 그대로 정상 진행된다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws && ./run_bc_rollout_recorder.sh
```

**시작 시점:** §4.4의 T3가 세션을 연 **뒤**라면 아무 때나 (이미 떠 있는 cam1/cam2를 구독하므로
T3보다 먼저 띄우지 않는다). **종료:** 평가가 끝나면 이 창에서 `Ctrl-C`. T1~T3보다 먼저 끊어도
나중에 끊어도 안전하다.

**무엇이 남나** — `ros2_ur_ws/gello_logs/bc_rollouts/rollout_<ts>/`

- `robot/vectors.h5` — 로봇 스트림: UR 관절 ~100 Hz, bridge command, gripper, wrench, TCP pose
- `robot/cam1.mp4` · `robot/cam2.mp4` — 이미 떠 있는 cam1/cam2를 구독해 저장
- `status.jsonl` — `/hil/actor_status` · `/hil/deadman` 타임라인 (에피소드 경계 정렬용)

**성공 표식:** 시작 배너에 rollout 디렉터리 경로가 출력되고, `Ctrl-C` 하면 마지막 줄에
`[bc-rollout] saved -> …` 가 나온다.

> ⚠️ **mp4는 `Ctrl-C`로 T4를 정상 종료해야 재생 가능해진다.** 재생 인덱스(moov atom)는
> `cv2.VideoWriter`가 **release될 때**, 즉 종료 시 기록된다 — **녹화 중에 파일을 열면 재생이
> 안 되는 것이 정상이고 고장이 아니다.** 같은 이유로 `kill -9` 같은 강제 종료로 끝내면
> **그 mp4는 영구히 재생 불가**다(`vectors.h5`/`metadata.json`도 종료 시 finalize된다).
> 반드시 그 창에서 `Ctrl-C`로 끝낼 것. `Ctrl-C` 직후 rclpy shutdown traceback이 보이는 것은
> **알려진 무해한 현상**이며, finalize는 그와 무관하게 수행된다.

**실패하면:** 이 창만 `Ctrl-C`로 닫고 **평가는 그대로 계속한다** — 녹화는 부가 기능이라 평가를
막지 않는다. 출력은 담당자에게 전달한다.

> 서버 쪽 기록(episode pickle · `actions.jsonl`)은 **T4 없이도 항상** 남는다(§7).
> T4는 거기에 **로봇 원시 데이터와 고주기 신호**를 더하는 것이다.
> `inference.jsonl`에는 조건이 하나 붙는다 — §7을 볼 것.

---

## 5. 평가 프로토콜

**한 episode의 흐름** (production과 동일)

1. 큐브를 놓고 장면을 배치한다 → GUI `START / NEXT ITERATION` → 정책이 팔을 몬다
2. **성공하면 GUI `MARK SUCCESS`** (MANUAL 판정만 쓴다)
3. 실패/이상하면 GUI `END EPISODE` (지금 끝내고 re-home. 데이터는 지워지지 않는다)
4. `WAIT_HOME_APPROVAL` → `APPROVE HOME` → HOME → 큐브 재배치 → 다음 episode

**규칙**

- **총 10 episode.**
- 큐브 위치는 매 episode 바꾼다 — **demo를 녹화할 때의 분포처럼 넓게 분산**시킨다.
  같은 자리만 반복하면 이 평가는 아무것도 알려주지 않는다.
- **GELLO 개입(ENGAGE)은 위험 회피용으로만.** 이 세션의 목적은 **정책 단독 거동**을 보는
  것이다. 부딪히겠다 싶으면 주저 없이 ENGAGE 하거나 `END EPISODE`를 누르되, 그 episode는
  기록표에 개입했다고 적는다.
- deadman, 워크스페이스 박스, `END EPISODE`, `APPROVE HOME`은 **전부 production 그대로**
  유효하다.

**기록 양식** (종이나 메모장에 그대로 옮겨 적는다)

| # | 성공? (MARK SUCCESS 눌렀나) | grasp 시도 횟수 | gripper 채터 | 정성 평가 (demo-directed / wandering / stuck) | 개입·비고 |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |
| **합계** | **성공 __ / 10** | | | | |

**grasp 시도** = 큐브 근처에서 그리퍼를 실제로 닫으려 한 횟수(성공 여부 무관) ·
**gripper 채터** = 그리퍼가 의미 없이 여닫히며 떠는 현상(있으면 ✓) ·
**정성 평가** = `demo-directed`(시범과 비슷한 방향) / `wandering`(그럴듯하지만 목적 없음) /
`stuck`(거의 안 움직임) 중 하나.

---

## 6. 실패 시그니처 — 무엇이 잘못됐는지 구분하는 표

| 보이는 현상 | 무슨 문제인가 | 그 자리에서 할 일 |
|---|---|---|
| 팔이 **정지**해 있거나 액션이 계속 0 | **배관 문제** (정책이 아니라 연결) | T1 창의 서버 로그를 확인하고 담당자에게 전달 |
| **난폭하거나 포화된 움직임** (한 방향으로 최대속으로 밀어붙임) | **후처리 / graft 문제** | **즉시 `END EPISODE`**, 위험하면 펜던트 protective stop. 세션 중단하고 담당자에게 보고 |
| 그럴듯하게 움직이는데 **엉뚱한 위치로 간다** | **relative-frame 원점 이슈** — demo는 take 첫 프레임 기준이다. **알고 진행하는 기지 리스크** | 중단하지 말고 **관찰만 기록**한다("어디로 얼마나 빗나갔는지"). 이게 이 실험의 답 중 하나다 |

> 성공률이 낮은 것 자체는 **고장이 아니다** — 그건 이 실험이 측정하려는 값이다.
> 위 표는 "정책이 못한다"와 "배선이 틀렸다"를 가르기 위한 것이다.

---

## 7. 종료 절차와 기록물, 사후 분석

**끄는 순서를 지킨다 (T3 → T2 → T1).**

```
① T3에서 Ctrl-C   — actor 종료 + controller가 STJC로 자동 복귀
② T2에서 Ctrl-C   — 하드웨어 번들 정리
③ T1에서 Ctrl-C   — 트랩이 서버측 BC 서버를 pid 파일로 종료하고 터널을 닫는다
```

③은 **자기가 띄운 BC 서버만** 종료한다. production learner(`:50053`)는 아무 영향도 받지 않는다.
T4(§4.5)를 띄웠다면 **이 순서에 끼지 않는다** — 순수 구독자라 앞이든 뒤든 아무 때나 `Ctrl-C`.

**기록물 위치 (서버):** `junhyeong_ai:~/hil-serl-data/bc_eval/bc_eval_<타임스탬프>/served/`
— episode별 pickle + `actions.jsonl`. 둘 다 T4 없이도 **자동으로** 생성된다.

⚠️ **`inference.jsonl`(모델 출력·추론 지연·입력 state 벡터)에는 조건이 있다** — 이 파일은
**커밋 `084ec1a` 이후에 기동된 BC 서버부터** 생성된다. 그 전에 떠 있던 T1이 서빙한 run에는
**아예 없다**(실측: 2026-07-31 run에 없었다). 파일이 안 보이면 고장이 아니라 서버 프로세스가
낡은 것이므로, **T1을 `Ctrl-C` 후 다시 띄우면 그 다음 run부터 자동으로 남는다.**
`analyze_bc_rollout.py`는 이 파일이 없어도 동작한다 — 추론 지연 관련 항목만 빠진다.
**기록물 위치 (laptop3, T4를 띄웠을 때만):** `ros2_ur_ws/gello_logs/bc_rollouts/rollout_<ts>/`.

**회수와 분석은 메인 세션(담당자)이 한다.** 조작자는 서버에 접속하지 않는다. 조작자가
넘겨야 할 것은 §5의 **기록표**와, 이상이 있었다면 그때의 터미널 출력이다.
담당자가 `served/`를 회수한 뒤 돌리는 분석기는 다음 하나다.

```bash
cd /home/laptop3/gello_software && /home/laptop3/venvs/gello-hil-actor/bin/python serl_ur_infra/scripts/analyze_bc_rollout.py --served <서버에서 회수한 served 디렉터리> [--robot ros2_ur_ws/gello_logs/bc_rollouts/rollout_<ts>]
```

→ `rollout_report.md` / `report.json`이 생성된다. `--robot`은 T4를 띄운 경우에만 붙인다.

**새 artifact로 다시 평가할 때:** T1만 내렸다가 올리면 된다. 코드는 아무것도 바꾸지 않는다.

```bash
BC_ARTIFACT_DIR=<새 artifact 경로> ./run_bc_server.sh
```

---

## 8. 🛑 절대 금지

1. **production learner(`:50053`)를 정지하거나 재시작하지 말 것.** 이 평가와 무관하게
   살아 있어야 한다.
2. **이 세션 중에 `run_hil_server.sh`의 learner 쪽 명령을 실행하지 말 것.**
   (§3-1의 `Ctrl-C`로 그 터미널을 닫는 것은 터널만 닫는 것이라 허용된다.)
3. **GELLO Dynamixel에 토크를 걸지 말 것.** 수동 read-only 리더다.
4. **서버의 `/home/junhyeong/gello_software`(뒤에 `_runtime`이 없는 것)에 접근하지 말 것** —
   다른 사람의 작업 트리다. 애초에 조작자는 서버에 접속하지 않는다.
5. 핸드셰이크 핀 3개를 **오류를 없애려고 임의로 바꾸지 말 것.** 그 오류가 "잘못된 정책을
   실기에서 돌리는 것"을 막는 게이트다.
