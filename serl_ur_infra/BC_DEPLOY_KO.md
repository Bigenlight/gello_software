# BC 정책 실기 평가 런북 (조작자용, 한국어)

> **상태: 실기 미검증.** 이 문서는 BC 평가 세션을 **처음** 돌리기 위한 절차다.
> 아래 명령은 전부 복사해서 그대로 붙여넣을 수 있다. 한 단계씩 하고, 각 단계의
> **「성공 표식」을 눈으로 확인한 뒤에만** 다음으로 넘어간다.

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

## 7. 종료 절차와 기록물

**끄는 순서를 지킨다 (T3 → T2 → T1).**

```
① T3에서 Ctrl-C   — actor 종료 + controller가 STJC로 자동 복귀
② T2에서 Ctrl-C   — 하드웨어 번들 정리
③ T1에서 Ctrl-C   — 트랩이 서버측 BC 서버를 pid 파일로 종료하고 터널을 닫는다
```

③은 **자기가 띄운 BC 서버만** 종료한다. production learner(`:50053`)는 아무 영향도 받지 않는다.

**기록물 위치 (서버):** `junhyeong_ai:~/hil-serl-data/bc_eval/bc_eval_<타임스탬프>/served/`
— episode별 pickle + `actions.jsonl`.
**회수와 분석은 메인 세션(담당자)이 한다.** 조작자는 서버에 접속하지 않는다. 조작자가
넘겨야 할 것은 §5의 **기록표**와, 이상이 있었다면 그때의 터미널 출력이다.

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
