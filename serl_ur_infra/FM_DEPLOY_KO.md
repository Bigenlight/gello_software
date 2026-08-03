# FM 정책 실기 평가 런북 (조작자용, 한국어) — **BC 런북의 델타**

> **상태: 2026-08-01~02 실기 검증 완료** — 실제 UR7e에서 FM 평가 세션이 돌았다(조작자 확인).
> 이 문서는 [`BC_DEPLOY_KO.md`](BC_DEPLOY_KO.md)를 **먼저 읽었다는 전제**로 쓴 차분(델타)이다.
> 절차·안전·기록양식은 BC와 **완전히 같고** 여기 적힌 것은 **FM에서만 다른 것**뿐이다. BC 런북을
> 옆에 띄워 두고 함께 본다. 평가 전반의 최상위 가이드는 [`POLICY_EVAL_KO.md`](POLICY_EVAL_KO.md)다.

## 1. FM이 뭐고, BC와 뭐가 다른가

같은 담당자가 **같은 34-take 데이터**로 학습한 **JAX Flow Matching**(rectified flow) 정책이다.
BC와 로봇·카메라·GUI·안전장치는 100% 같고 **정책을 서빙하는 서버 프로세스만** 다르다.

**동작 방식 (BC와 다른 점)**

- 한 번 추론할 때 가우시안 노이즈에서 시작해 **Euler 8-step 적분**으로 **16 스텝짜리 액션
  청크**(horizon 16)를 만든다.
- 그런데 **첫 액션 하나만 실행**하고 나머지는 버린 뒤 다음 스텝에서 **처음부터 다시 계획**한다
  (receding horizon). 즉 매 스텝 새로 추론한다.
- 🟢 **추론이 확률적이라 같은 장면에서도 액션이 매번 조금씩 다르다 — 정상이다.**
  BC와 달리 "완전히 똑같이 재현"되지 않는 것은 고장이 아니다.
- ⏱️ **추론 지연은 컴파일 뒤 ~72 ms다** — `0c0f094`가 chunk 샘플러를 `jax.jit`으로 감쌌다(eager
  시절엔 호출당 **502 ms**, 100 ms 제어주기를 한참 넘겼다. 둘 다 서버 GPU smoke 실측). 첫 1~2회
  호출은 컴파일로 느린데 **기동 smoke가 `ready` 전에 다 태우므로 조작자는 체감하지 않는다**(§2.1).

**학습 지표 (담당자 보고값)** — first-action continuous MSE **0.04724** ·
first-action gripper 정확도 **96.8 %** · chunk gripper 정확도 **92.0 %**.
**로컬 CPU 게이트 실측**(holdout 30 obs) — first-action MSE **0.0375** · translation cosine **0.637** · gripper **96.7 %**.

> 참고로 BC의 MSE는 0.0325라 **FM이 약간 높다.** 그래도 실기 성공률이 어느 쪽이 높은지는
> **모른다** — 그걸 재려고 이 평가를 하는 것이다. §4의 A/B 팁을 보라.

## 2. FM에서만 다른 명령 두 개

### 2.1 T1 — FM 서버 + 터널

```bash
cd /home/laptop3/gello_software/ros2_ur_ws && ./run_fm_server.sh
```

**성공 표식:** `FM_SERVER_RESULT=started` + `[fm-server] ready ...` 줄 + 프롬프트가 안 돌아옴.

⏳ **`ready`까지 수십 초 걸리는 것은 정상이다.** 기동 smoke가 ODE 추론을 **두 번** 돌려 jit
컴파일을 미리 태운다(§1) — 멈춘 것처럼 보여도 기다린다. 그 두 번은 기록물에 안 남는다(§4).

🛑 **터널은 하나뿐이다.** FM 서버의 원격 포트는 **50055**라 BC 서버(50054)·production
learner(50053)와 서버 쪽에서는 충돌하지 않는다. **그러나 로컬 `50153` 터널은 셋이 공유한다** —
production `run_hil_server.sh`든 `run_bc_server.sh`든 **다른 T1 터미널이 열려 있으면 먼저
`Ctrl-C`로 닫은 뒤** 위 명령을 실행한다 (BC 런북 §3의 1·2번 항목과 같은 규칙).

**artifact 바꿔서 다시 평가할 때** (T1만 내렸다 올린다, 코드 수정 0)

```bash
FM_ARTIFACT_DIR=<새 경로> ./run_fm_server.sh    # 다른 artifact
FM_WHICH=final ./run_fm_server.sh               # best 대신 final 체크포인트
```

### 2.2 T3 — 핀 3개가 FM 값으로 바뀐다

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
EXPECTED_MODEL_ID=fm-cube-in-cup-raw0731-h16-euler8-v1 EXPECTED_REWARD_AUTHORITY=local EXPECTED_REWARD_MODEL_ID=operator-manual-success-v1 ./run_hil_session.sh --no-classifier-sidecar
```

**(선택, 권장) dry handshake** — 로봇을 전혀 쓰지 않는다. 같은 핀으로:

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
EXPECTED_MODEL_ID=fm-cube-in-cup-raw0731-h16-euler8-v1 EXPECTED_REWARD_AUTHORITY=local EXPECTED_REWARD_MODEL_ID=operator-manual-success-v1 ./run_hil_actor.sh --fake-env --no-classifier-sidecar
```

## 3. 나머지는 전부 BC 런북을 그대로 쓴다

| 무엇 | 어디를 보나 | FM에서 달라지는 것 |
|---|---|---|
| 사전 조건 체크리스트 | BC §3 | 없음 (1·2번의 "다른 T1 닫기"에 BC 서버도 포함된다) |
| T2 하드웨어 | BC §4.3 | **없음 — 명령까지 동일** |
| T3 성공 표식 / preflight 실패 대응 | BC §4.4 | `[6] 50153 연결 실패` = **FM 서버(T1)**가 죽었다는 뜻 |
| MANUAL 평가 프로토콜·10 episode·기록표 | BC §5 | 없음 (`MARK SUCCESS` / `END EPISODE` 동일) |
| 실패 시그니처 구분표 | BC §6 | 없음. ⚠️ 단 "같은 장면에서 액션이 조금 다름"은 **이상 아님**(§1) |
| T4 rollout 녹화 (선택) | BC §4.5 | 없음 — 같은 `./run_bc_rollout_recorder.sh`를 그대로 쓴다 (정책과 무관한 순수 구독자) |
| 종료 순서 **T3 → T2 → T1** | BC §7 | 없음. ③이 종료하는 대상만 FM 서버로 바뀐다 |
| 🛑 절대 금지 | BC §8 | **없음 — 그대로 전부 적용된다** (특히 핀 3개를 임의로 고치지 말 것) |

## 4. 기록물과 A/B 비교

**기록물 (서버, T4 없이도 항상 생성):**
`junhyeong_ai:~/hil-serl-data/fm_eval/fm_eval_<타임스탬프>/served/`
— episode별 pickle + `actions.jsonl` + `inference.jsonl`.
🟢 **FM 서버는 처음부터 최신 코드로 떴으므로 `inference.jsonl`이 첫 run부터 있다** (BC의
07-31 첫 run에는 없다). 기동 smoke 2회는 로거를 그 뒤에 붙이므로 이 파일에 섞이지 않는다.
**회수·분석은 메인 세션(담당자)이 한다** — 조작자는 서버에 접속하지 않는다.
분석기는 **BC와 같은 것**을 쓴다(BC 런북 §7의 `analyze_bc_rollout.py --served …`).

**BC vs FM A/B 비교 팁**

- **같은 큐브 배치 세트**를 쓴다. 1~10번 자리를 미리 정해 두고 BC 10 episode, FM 10 episode를
  그 순서 그대로 돈다. 배치가 다르면 두 숫자를 비교할 수 없다.
- **MANUAL 성공 판정 기준을 양쪽에서 똑같이** 적용한다("컵 안에 들어가 손을 떼도 유지"처럼
  한 문장으로 미리 못 박고 세션 내내 바꾸지 않는다).
- 두 세션의 기록표를 **나란히** 담당자에게 넘긴다. 성공 __/10 두 개가 이 실험의 결론이다.
