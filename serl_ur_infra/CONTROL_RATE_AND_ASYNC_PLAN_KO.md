# 제어 주파수 조사와 비동기화 방안 — G21 / §8 P1

> 조사일: **2026-07-31 KST** · 대상 branch `feat/gello-ur7e-humble-22.04`
>
> 이 문서는 **"policy 제어가 2 Hz밖에 안 된다"**는 조작자 관찰에서 출발해,
> upstream hil-serl과 대조하고 시간이 어디로 가는지를 분해한 기록이다.
>
> 🛑 **이 문서의 결론은 아직 "고쳤다"가 아니다.** 아래 Stage 1 계측이 끝나기 전에는
> 어떤 방안도 채택하지 않는다. 서버에는 현재 **per-stage 타이머가 하나도 없다.**
>
> 관련: [`../docs/testing/08_OPEN_GAPS.md`](../docs/testing/08_OPEN_GAPS.md) **G21** ·
> [`HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) **§8 P1** ·
> [`REMOTE_ACTOR_GRPC.md`](REMOTE_ACTOR_GRPC.md)(전송 계약)

> # 🗄️ 이 조사의 시간 수치는 **kanu에서 잰 것이다** (같은 날 서버가 옮겨졌다)
>
> 2026-07-31에 learner가 **`kanu` → `junhyeong_ai`** 로 이전됐다(RTX A4000 8장 중 GPU 5 →
> **RTX 5070 Ti 1장, GPU 0**). 이 문서의 **465 ms · 512/854 ms · 1.123 s ↔ 0.457 s ·
> ICMP 2.02 ms**는 전부 **kanu 상대 관측**이며, 그래서 그대로 둔다 — 새 호스트의 개선
> 폭은 이 값들과 대조해야만 나온다.
>
> **새 호스트에서 오늘 실제로 측정된 것** *(출처:
> [`SERVER_MIGRATION_E2E_JUNHYEONG_AI.md`](./SERVER_MIGRATION_E2E_JUNHYEONG_AI.md), 합성
> 수락 시험 200 transition — 로봇 없음)*:
>
> | | kanu | **junhyeong_ai** |
> | --- | --- | --- |
> | `BeginEpisode` mean / max | 84.9 / **372.8 ms** | **57.7 / 82.2 ms** (tail 약 4.5배 짧음) |
> | `Step` mean / max | — | 156.1 / 211.0 ms |
> | ICMP RTT (min/avg/max, n=10) | 1.272 / 2.518 / 6.314 ms | 1.078 / **2.811** / 6.979 ms |
>
> *(kanu ICMP는 §3.1에도 이 조사 자체의 값 `avg 2.02 / min 1.02 / max 5.57`이 따로 있다 —
> 같은 호스트를 다른 시각에 잰 것이고 둘 다 유효하다. 결론은 같다: **2~3 ms**.)*
>
> 📏 **ICMP가 사실상 같다 = 네트워크는 원인이 아니었고, 지금도 아니다.** 즉 §3.2의
> "390~445 ms가 서버 안"이라는 **분해 구조 자체는 이전으로 바뀌지 않는다.** 호스트가
> 빨라져 그 덩어리가 줄었을 뿐이고, **얼마나 줄었는지는 아직 모른다.**
>
> 🛑 **바뀌지 않은 것 — 이 문서의 결론과 계획은 전부 유효하다.** 락 공유 구조(C1/C2),
> RPC 분할(A′), 파이프라이닝 불가(§5), C3 금지(§8.1), 그리고 무엇보다 **Stage 1 계측이
> 먼저**라는 순서. 서버가 빨라졌다고 per-stage 타이머가 생기지는 않았다 —
> **§10의 "인용 금지 목록"은 한 줄도 해소되지 않았다.**
>
> 🔴 **그리고 실기 루프 주기는 새 호스트에서 아직 재지 않았다.** 위 합성 시험의
> transition당 RPC 합계 213.8 ms를 kanu의 512 ms에서 **그냥 빼지 마라** — 후자는 카메라
> 디코드와 `env.step`의 100 ms 페이싱을 포함한 **로봇 루프 전체 주기**다. Stage 0의
> "`POLICY_RUNNING` 구간만 걸러 다시 잰다"가 **여전히 첫 할 일**이고, 이제는 새 호스트
> 기준선을 세우는 일이기도 하다.

---

## 0. 한 줄 결론

**upstream hil-serl은 이 문제를 극복한 것이 아니라 애초에 만들지 않는다** — 정책 추론이
actor 인프로세스이고 learner와의 통신은 전부 제어 루프 **밖**이다. 우리는 그 구조를 뒤집어
**매 스텝 동기 gRPC 왕복**을 넣었고, 그 왕복 안에 **classifier · frozen-trunk 인코딩 ·
replay insert · 정책 추론**이 전부 직렬로 들어 있다.

측정상 **465 ms 중 390~445 ms(83~96%)가 서버 안**이며, **정책 추론 자체는 무죄**다
(replay·learner 없는 probe에서 inference 12.08 ms / 왕복 53.03 ms).

---

## 1. ⚠️ 먼저 — 출발점이 된 측정값은 제어율의 증거가 아니다

조작자 관찰:

```
$ ros2 topic hz /hil/actor_status
average rate: 2.149    min: 0.115s  max: 0.500s  std dev: 0.121s  window: 9
```

**`max: 0.500s`는 `WAIT_STATUS_REPUBLISH_S = 0.5`와 정확히 일치한다**
(`ur_env/operator_session.py:25`, 사용처 `:618`). `WAIT_SCENE_READY` /
`WAIT_HOME_APPROVAL`에서 status가 **0.5초마다 재발행**되므로, 그 구간의 2 Hz 하트비트가
표본에 섞였다. 반대로 `min: 0.115s` 구간은 오히려 **8.7 Hz**다. 표본도 `window: 3~9`로 작다.

**⇒ 이 측정만으로 제어율을 판정하지 않는다.** 다만 걱정 자체는 유효하다 — production actor
실측 **1.95 Hz / 512 ms(최악 854 ms)**가 독립적으로 기록돼 있다(CLAUDE.md, G21).

**제대로 재는 법:** `POLICY_RUNNING` 상태의 간격만 모은다.

```bash
ros2 topic echo /hil/actor_status --field data 2>/dev/null \
  | python3 -c "
import sys, json, time
prev=None
for line in sys.stdin:
    line=line.strip()
    if not line or line.startswith('---'): continue
    try: s=json.loads(line)
    except Exception: continue
    now=time.monotonic()
    if prev and s.get('state')=='POLICY_RUNNING':
        print(f\"{(now-prev)*1000:7.1f} ms  step={s.get('episode_step')}\")
    prev=now
"
```

---

## 2. upstream hil-serl 대조 — 무엇이 비동기인가

📌 아래는 전부 `third_party/hil-serl` @ `c32939b` 코드에서 직접 확인했다.

### 2.1 정책 추론은 upstream도 **동기**다 — 다만 로컬이다

```
upstream actor 한 스텝 (examples/train_rlpd.py::actor)
  agent.sample_actions(obs)      :156-166   동기. 인프로세스 JAX. 네트워크 없음
  env.step(action)               :169-171   동기. franka_env.step()이 1/hz로 페이싱
  data_store.insert(transition)  :198       로컬 deque append. 네트워크 아님
  ── episode 끝에서만 ──
  client.request("send-stats")   :209       블로킹
  client.update()                :215       블로킹. 여기서 전이가 실제 전송된다
```

- actor 루프 자체에 `time.sleep`이 **없다**. 페이싱은
  `franka_env.py:230-231`의 `time.sleep(max(0, 1/self.hz - dt))`, `hz=10`(`:84,110`).
- **파라미터는 비동기 PUB/SUB**: `client.recv_network_callback(update_params)`
  (`train_rlpd.py:137`) → `BroadcastClient.async_start`가 **전용 스레드**를 띄운다
  (`agentlace/zmq_wrapper/broadcast.py:55-66`). `zmq.CONFLATE=True`로 **오래된 가중치는
  버린다**(`:50-51`) — 주석에 이유가 "actor가 학습을 못 하게 되니까"로 적혀 있다.

**⇒ 비동기인 것은 `learner와의 두 채널`이지 추론이 아니다.**
인과가 반대다: 비동기라서 빠른 게 아니라, **추론이 로컬이라 비동기가 가능**했다.

### 2.2 upstream에 action chunking은 **없다**

| 항목 | 코드에 존재? | shipped config에서 사용? |
|---|---|---|
| obs history stacking (`obs_horizon`) | 예 (`wrappers/chunking.py:46-53,76`) | ❌ 전부 `1` |
| action chunking (`act_exec_horizon`) | 예 (`chunking.py:54-72`) | ❌ 전부 `None` → 1로 강제(`:62-65`) |
| temporal ensembling / `n_action_steps` | **아예 없음** | — |

experiment 4종(`ram_insertion:114`, `usb_pickup_insertion:122`, `object_handover:192`,
`egg_flip:44`) 전부 `obs_horizon=1, act_exec_horizon=None`이다.

부드러움은 학습된 청크가 아니라 **하드웨어 임피던스 컨트롤러**가 만든다 — 10 Hz 정책이
`/cartesian_impedance_controller/equilibrium_pose`에 setpoint 하나를 던지면
(`robot_servers/franka_server.py:51-55`) 로봇의 실시간 루프가 수렴시킨다.
**우리 250 Hz 업샘플러가 정확히 그 자리다.**

### 2.3 upstream 카메라는 **백그라운드 스레드**다

`VideoCapture._reader`(`franka_env/camera/video_capture.py:13-28`)가 전용 스레드에서
계속 `cap.read()`하고 **1칸 큐에 최신 프레임만** 남긴다(`get_nowait()`로 낡은 것 폐기).
`step()`은 큐 pop 비용만 낸다. **우리는 동기 디코드다**(§3.1).

---

## 3. 우리 시간은 어디로 가나

### 3.1 laptop 쪽 (측정)

| 구간 | 값 | 종류 | 근거 |
|---|---:|---|---|
| `env.step`의 `1/HZ` 페이싱 | 100 ms | 설계값 | `ur7e_env.py:1463-1464` |
| 카메라 디코드+크롭+리사이즈 (2대) | 중앙값 **8.96 ms** / p90 10.19 | **측정**(2026-07-31, 합성 JPEG 대용) | 이번 조사 벤치 |
| 관측 96.57 KiB 전선 시간 | **9.53 / 16.55 / 60.85 ms** | 계산 | 83 / 47.8 / 13 Mbit/s — 셋 다 과거 실관측 |
| sidecar 추가 전선(~2 Hz 스텝) | 1.31 / 2.28 / 8.39 ms | 계산 | 13.32 KiB |
| kanu ICMP RTT | avg **2.02 ms** (min 1.02 / max 5.57) | **측정**(유휴, n=10) | 이번 조사 |
| *(참고)* `junhyeong_ai` ICMP RTT | avg **2.811 ms** (min 1.078 / max 6.979) | **측정**(n=10, 2026-07-31) | `SERVER_MIGRATION_E2E_JUNHYEONG_AI.md` §3.1 — **같은 자릿수. 이전으로 링크 조건은 안 바뀌었다** |
| **laptop + 전선 합계** | **18~78 ms = 4~17%** | | |

🪤 **`env.step`의 100 ms 페이싱은 `get_im`을 덮지 않는다.** `ur7e_env.py:1444-1464`에서
sleep이 끝난 **뒤에** `_harvest_follow_window` → `_update_currpos()` → `_get_obs()`(디코드)가
돈다. 즉 명목 창은 100 ms인데 실제는 **100 ms + 디코드**다.

📌 `_update_currpos`(`:1785`)와 `get_im`(`:1939`)은 **대기하지 않는다** — stale이면 즉시
`raise`한다. 숨은 블로킹 대기는 없고 **CPU 디코드가 전부**다.

📌 `--checkpoint-path`는 production에 **걸려 있지 않다**(`run_hil_actor.sh` /
`run_hil_session.sh`에 인자 없음, `cube_in_cup.py:271` `buffer_period=0`).
따라서 `remote_actor.py:1314-1318`의 스텝당 `copy.deepcopy` 2회는 **현재 실행되지 않는다.**

### 3.2 ⇒ 나머지 **390~445 ms (83~96%)가 서버 안**이다

그리고 **정책 forward pass는 무죄**다:

| 값 | 조건 | 근거 |
|---:|---|---|
| 서버 inference **12.08 ms**, 왕복 **53.03 ms** | **kanu**, disposable 서버, **replay insert 0건**, learner 없음 | `HIL_SERL_KANU_RUNBOOK_KO.md` — "no-submit probe"로 검색 (원래 `:91-93`) |
| inference **18.44 ms**, 왕복 **47.75 ms** | 같은 no-submit probe 계열 (**kanu**) | `HANDOFF_NEXT_SESSION_KO.md` — 같은 문구로 검색 (원래 `:76-77`) |

**⇒ replay·learner가 붙는 순간 느려진다.**

🗄️ **위 두 줄은 kanu probe다.** 새 호스트에는 대응하는 no-submit probe가 **아직 없다** —
`junhyeong_ai`에서 잰 것은 replay·learner가 **붙은** 상태의 `Step` 156.1 ms 뿐이므로
(§맨 위 표) 두 값을 같은 줄에 놓고 비교하면 안 된다.

### 3.3 서버 구조 — 가장 유력한 단일 용의자

```
ActorSessionService.step()                       actor_network.py:626-777
  with self._lock:                               :478  프로세스 전역 RLock, 본문 전체
    validate → classifier(sidecar 있을 때만)
    accept_data ─────────────────────────────►   :695
      FaultGatedReplayIngress.__call__           ingress.py:103
        with self._lock:                         ingress.py:70  ◄── learner와 공유
          FeatureReplayIngress.__call__          feature_replay.py:553-599
            _encode(raw_transition)              🔴 GPU forward ×2, 락 안
            replay_store.insert(encoded)
            intervention_store.insert(encoded)
    policy inference                             :738-741
    serialize
```

- 그 락을 learner가 같이 쓴다: `sample_replay`(`ingress.py:123`) /
  `sample_intervention`(`:128`). 배선은 `learner/composition.py:475,489`(같은
  `assembly.ingress`를 `accept_data`로 주입)와 `learner/batches.py:50-79`에서 확인된다.
- `--max-workers`는 이 경로를 병렬화하지 **못한다** — 다른 RPC 종류만 겹칠 수 있다.
- 📌 **`ActorSessionService`의 전역 락은 actor가 하나뿐이라 사실상 무해하다.**
  범인 후보는 **learner와 공유하는 ingress 락**이다. 이 구분이 해법을 가른다.
- 코로케이션은 사고가 아니라 **명시된 설계 제약**이다 — `learner/composition.py:5-7`:
  replay store가 RAM-only이고 프로토콜에 replay 샘플링 RPC가 없어 gRPC ingress와 learner
  worker가 한 프로세스에 있어야 한다.

**정황 증거:**

| | 값 | 근거 |
|---|---:|---|
| learner step 중앙값 (actor 동시) | **1.123 s** | `HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md` — "learner step 중앙값"으로 검색 (원래 `:149`) |
| learner step 중앙값 (actor 종료 후) | **0.457 s** | 같은 문서 (원래 `:150`) |

**서로 비슷한 크기로 밀어내고 있다** — actor가 learner에 +0.67 s, learner가 actor에 약
+350 ms. 이건 "락을 잠깐 기다린다"보다 **GPU 포화**에 가까운 그림이다(§6 참조).

🛑 **그러나 이건 전부 정황이다.** 락 대기 / 인코딩 / 정책 추론이 **분리 계측된 적이 없다.**
"412 ms가 RPC"라는 문구도 **코드 주석**(`remote_actor.py:1007-1008`)이고 다른 run의 값이다.
살아 있는 계측은 `rpc_ms`의 **run-end mean/max뿐**이며, 직렬화+전선+서버+역직렬화를 통째로
재서 분해가 안 된다.

---

## 4. 해법 공간

📌 아래 "추론 위치"의 **GPU 서버**는 2026-07-30까지 kanu였고 **2026-07-31부터
`junhyeong_ai`(RTX 5070 Ti 1장)** 다. 방안의 구조는 호스트와 무관하다.
🟡 다만 **D는 전제가 약해졌다** — 새 서버는 GPU가 **1장뿐이고 다른 사람과 공유**하므로,
"별도 GPU"는 kanu의 8장 시절과 달리 지금은 **여유 GPU를 가정할 수 없다.**

| | 방식 | 추론 위치 | 제어 경로 비용 | 비용 | 건드리는 계약 |
|---|---|---|---:|---|---|
| **A** | 정책 추론을 laptop3 CPU로 (upstream 구조) | laptop | ~20 ms | 대 | 프로토콜 + echo 검증 + venv 통합 |
| **A′** | **RPC 분할** — `GetAction` / `SubmitTransition` | GPU 서버 유지 | **~31 ms** | 중 | proto 신규 + **ACK 의미** |
| **B** | 파이프라이닝 (t 실행 중 t+1 요청) | — | — | — | 🔴 **불가** (§5) |
| **C1** | `_encode`를 공유 락 밖으로 | GPU 서버 | ? | **소** | **없음** |
| **C2** | learner 샘플링에 스냅샷/더블버퍼 | GPU 서버 | ? | 중 | 없음 |
| **C3** | 인코딩·insert를 큐로 빼고 액션 먼저 반환 | GPU 서버 | 큼 | 중 | 🔴 **ACK=삽입완료 파기** |
| **D** | 추론을 서버의 별도 프로세스/GPU로 | GPU 서버(별도) | ? | 중 | 없음 (🟡 새 서버는 GPU 1장 공유 — 위 주의) |
| **E** | **유선 NIC 연결** | — | −0~50 ms | **0** | 없음 |
| **F** | action chunking | — | — | 대 | upstream 선례 **없음** |

### 4.1 E — 지금 당장, 코드 0줄

```
$ ip -br link
enx00e04c3600bd  DOWN  <NO-CARRIER,BROADCAST,MULTICAST,UP>
```

USB 이더넷 NIC이 **존재하는데 안 꽂혀 있다**(이번 조사 측정). 세션 간 **약 6배**
대역폭 변동(13 / 47.8 / 83 Mbit/s — 셋 다 실관측)이 통째로 사라지고, 최악 링크에서
**~50 ms**를 즉시 회수한다. 문서가 며칠째 같은 말을 하고 있다.

### 4.2 C1 — 인코딩을 락 밖으로 (제안 형태)

`_encode`는 `raw_transition`만 읽고 `feature_extractor`를 두 번 부른다
(`feature_replay.py:644-657`). **ingress 공유 상태를 전혀 건드리지 않는 순수 함수**인데
지금은 락 안에 있다.

```python
# 현재: feature_replay.py:553-599
with self._lock:
    route = self._ledger.get(...)
    if route is not None and route.completed: return
    encoded = self._encode(raw_transition)      # 🔴 GPU ×2, 락 안
    self.replay_store.insert(encoded)
    ...

# C1
with self._lock:                                # ① 멱등성만 빠르게 확인
    route = self._ledger.get(...)
    if route is not None and route.completed: return
encoded = self._encode(raw_transition)          # ② 🟢 락 없이 GPU
with self._lock:                                # ③ 원장 + 삽입만
    route = self._ledger.get(...)               #    락을 놨으므로 재확인
    ...
```

**안전한 이유:** actor의 `accept_data`는 어차피 한 스레드에서만 불린다
(`ActorSessionService._lock`이 `step()` 전체를 이미 직렬화). 락의 진짜 역할인 **learner
스레드 방어**는 ③이 그대로 한다. ACK 의미 · exactly-once · fail-closed · 원장 전부 불변.

**대가:** 재시도 시 인코딩 중복 가능(대부분 ①과 상류 reply 캐시가 차단).

### 4.3 A′ — RPC 분할 (구조적 해법)

```
GetAction(observation)     → 정책 추론만. replay insert 없음, classifier 없음, 공유 락 안 탐
                             동기·제어 경로:  추론 12 ms + 전선 10 ms + 디코드 9 ms ≈ 31 ms
SubmitTransition(data)     → 백그라운드 스레드. classifier · 인코딩 · insert 전부 여기로
```

**추론을 GPU 서버에 둔 채로 10 Hz가 나온다.** 지금 465 ms인 이유가 "액션을 받으려면 replay
insert까지 끝나야 해서"이기 때문이다.

이것은 새 발명이 아니라 **upstream이 실제로 하는 구조**다 — 다만 upstream은 추론까지
로컬이라 `GetAction`조차 네트워크가 아닐 뿐이다(§2.1).

---

## 5. B(파이프라이닝)가 불가능한 이유 — 재시도 금지

전송 계층은 문제가 없다(`unary_unary` + `.future()`, proto에 `stream` 없음).
막는 것은 **세션 계약**이다:

1. **`StepRequest`가 "전이 t 커밋"과 "O(t+1) 액션 요청"을 한 메시지에 묶었다**
   (`proto/actor_transport.proto:128-139`). 즉 **O(t+1)이 있어야 요청을 만들 수 있고**,
   그건 로봇이 액션 t를 다 실행해야 나온다 — 겹치려는 바로 그 구간이 전제조건이다.
2. **echo 검증**: 다음 요청의 `meta.policy_action` / `policy_version`이 직전 응답과
   **바이트 동일**해야 한다(`actor_network.py:1320, 1327`). 그 값은 응답 전엔 클라이언트에
   존재하지 않는다.
3. **엄격 단조 카운터** `request_id`/`step_id`/`env_step`, 커밋 시점에만 증가
   (`actor_network.py:642-646, 715-718`).
4. reply 캐시는 **재시도 멱등성**용이지 파이프라인 큐가 아니다(같은 id + 다른 내용 → 거부,
   `:1430`).
5. 뚫어도 **RL 의미가 깨진다**: `RelativeFrame`이 이미 "액션은 step 이전 행렬, 관측은 이후
   행렬"로 **정확히 한 주기**를 전제한다(`ur_env/envs/frame_wrappers.py:104-130`).
   한 주기가 더 얹히면 델타가 **틀린 tool frame으로 회전**하고 **아무 에러도 안 난다.**

---

## 6. 🛑 C1에 대한 정직한 기대치

§3.3의 두 숫자(1.123 s ↔ 0.457 s)는 **상호 GPU 포화**에 가깝다. frozen ResNet-10 forward
2회가 A4000에서 10~20 ms 수준이라면 **C1의 상한은 400 ms 중 20 ms**로 사실상 무의미하다.

🟡 **이 추정의 GPU가 바뀌었다.** 두 숫자는 kanu(A4000, GPU 5) 관측이고 현행 서버는
RTX 5070 Ti다. 방향은 그대로일 공산이 크지만 — 카드가 빨라지면 forward도 락 점유도 같이
줄어 **C1의 상한은 오히려 더 낮아진다** — **재측정 없이 단정하지 마라.** 어차피 결론은
같다: **Stage 1 계측이 먼저다.**

C1이 크게 먹히는 경우는 **learner가 `sample_replay`로 락을 오래 잡을 때**뿐인데, 샘플링은
링에서 memcpy라 원래 빠를 공산이 크다.

**⇒ C1은 싸고 안전한 복권이다. 상한이 낮을 가능성이 높고, 그렇다면 답은 A′다.**
그리고 이 판단은 **Stage 1 계측이 30분 안에 내려준다.**

---

## 7. 단계별 계획

### Stage 0 — 지금, 코드 0줄
- [ ] **유선 NIC `enx00e04c3600bd`를 꽂는다** (§4.1).
- [ ] `POLICY_RUNNING` 구간만 걸러 **실제 스텝 주기를 다시 잰다** (§1).

### Stage 1 — 계측 (반나절) ← **어떤 방안이든 여기가 먼저다**

문서가 §8 P1에서 **이미 요구하고 있는 바로 그 타이머**다
(`HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md:883-893`).

| 타이머 | 위치 |
|---|---|
| `env.step()` 전체 소요 | `ur_env/remote_actor.py:1086` — `perf_counter` 쌍, 기존 `rpc_ms`와 **per-step** 로그 |
| 락 획득 대기 | `ur_env/learner/feature_replay.py:553` `with self._lock` 직전/직후 |
| `_encode` 소요 | `ur_env/learner/feature_replay.py:644` 진입/반환 |
| 정책 추론 소요 | `ur_env/actor_network.py:1399` `_infer` |

전부 `transition_id`와 함께 남긴다. `rpc_ms`도 run-end mean/max가 아니라 **per-step**으로.

### 분기점

| 계측 결과 | 다음 |
|---|---|
| `accept_data` **락 대기**가 지배 | → **C1**, 필요하면 C2 |
| `_encode` + 정책 추론 **자체**가 큼 (= GPU 포화) | → **C1 버리고 A′** (필요하면 D 병행) |
| 전선이 지배 | → **E**로 이미 해결됐어야 함. 재확인 |

### Stage 2 — 채택안 적용 후 **같은 계측으로 재측정**
### Stage 3 — A(로컬 추론)는 A′로도 부족할 때만 (§8.3)

---

## 8. 위험

### 8.1 🔴 C3는 하지 마라 — 조용한 데이터 손실

ACK은 현재 **"모든 RAM 경로 삽입 성공"**을 뜻하고 시스템은 fail-closed다
(`REMOTE_ACTOR_GRPC.md` Failure rules). 액션을 먼저 반환하면 **로봇이 이미 움직인 뒤에**
삽입 실패를 알게 된다 → 그 전이는 replay에 없는데 로봇은 움직였다.

이 리포는 정확히 이 계열(크롭 불일치 G15, 축별 clip `d6965a9`, 시간초과 `masks=0` G35)로
세 번 데였다. **A′로 갈 때도 이 대가를 "받아들이는 결정"으로 명시해야 한다** — upstream은
이미 받아들이고 있다(로컬 큐 + episode 끝 flush, learner가 죽어도 actor는 돈다).
완화: 업로드 실패 시 **loud fail + 세션 중단**, 미전송 큐를 디스크에 남긴다.
**조용히 잃는 것만 막으면 된다.**

### 8.2 🟠 A′ — proto 변경은 양단 동시 업그레이드

**protobuf가 unknown field를 조용히 버리므로 반쪽 업그레이드는 무증상 오염**이다.
`SCHEMA_VERSION` bump와 핸드셰이크 거부를 같이 넣는다. 순서·exactly-once 검사
(`request_id`/`step_id` 단조)가 두 채널로 갈리는 것도 재설계 대상이다.

### 8.3 🟠 A(로컬 추론) — 세 가지

1. **echo 검증의 의미 상실.** 서버는 지금 다음 요청의 `meta.policy_action`이 직전에 발행한
   액션과 바이트 동일한지 검사한다(`actor_network.py:1327`) — 이것이 **"저장 액션 == 실행
   액션"의 서버측 보증**이다. 로컬 추론이면 서버가 그 액션을 모르므로 이 검사가 무의미해진다.
   **대체 보증 설계 없이 진행하면 이게 가장 큰 구멍이다.**
2. **RT 커널 지터.** JAX forward pass를 actor 스레드에서 돌리면 250 Hz 업샘플러와 ROS
   타이머를 흔들 수 있다. **별도 스레드/프로세스 + CPU affinity 분리**가 전제다.
3. **venv 통합.** actor venv(`gello-hil-actor`)에는 **jax가 없고**, jax가 있는
   `hilserl` venv의 grpcio 상태는 미확인이다. rclpy + jax + grpcio 1.74 공존이 선결 조건.

📌 **reward 권위는 어느 안에서도 서버 유지다** — reward는 sidecar로 서버가 계산한다.

---

## 9. laptop3 로컬 추론의 실현 가능성 (A안 참고자료)

📌 전부 2026-07-31 laptop3에서 직접 측정.

### 9.1 GPU는 지금 **못 쓴다** — 원인은 드라이버 버전이 아니라 RT 커널

```
$ nvidia-smi     → NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
$ lspci -k       → GA106M [GeForce RTX 3060 Mobile / Max-Q];  Kernel modules: nvidiafb, nouveau
$ lsmod|grep nvidia → (비어 있음)
$ uname -r       → 6.8.2-rt11
$ dkms status    → nvidia/595.84, 6.8.0-124-generic / 6.8.0-136-generic   ← -rt11 없음
$ ls /lib/modules/6.8.2-rt11/updates/dkms/   → (비어 있음)
$ dpkg -l | grep "linux-headers.*rt11"       → (없음)
```

dkms가 `generic` 커널용으로만 빌드됐고 부팅된 `-rt11`용 헤더가 없어 `nvidia.ko`가 **컴파일된
적이 없다.** 리부트로 안 고쳐진다.

⚠️ **`HIL_SERL_STATUS_AND_NEXT.md:216`의 "리부트하면 복구될 전형적 케이스"는 다른 원인의
07-24 스냅샷이고 현재 상황이 아니다.** 그리고 **RT 커널은 실시간 제어를 위한 의도적 선택일
가능성이 높다 — 함부로 건드릴 문제가 아니다.**

### 9.2 그러나 CPU 추론이면 충분할 공산이 크다

```
$ /home/laptop3/venvs/hilserl/bin/python -c "import jax; print(jax.__version__, jax.default_backend())"
0.5.3 cpu
$ ... validate_learner_dependencies()
{'jax':'0.5.3','jaxlib':'0.5.3','flax':'0.10.5','distrax':'0.1.5','tensorflow_probability':'0.25.0'}
```

**SAC 에이전트를 만들 수 있는 스택이 이미 있고 pin과 정확히 일치한다.**
(⇒ `HIL_SERL_STATUS_AND_NEXT.md:68`의 *"로컬에 jax 미설치 — 의도적"*은 낡았다.)

정책과 분류기는 **같은 frozen ResNet-10 백본**을 쓴다. 그 분류기(7.27M 파라미터)의 laptop3
**CPU** 실측이 forward **6.6~7.1 ms**, 디코드 포함 **11.9 ms(~84 Hz 여력)**다
(`REWARD_CLASSIFIER_LIVE_KO.md:429-433`). 정책은 `hidden_dims=[256,256]`가 더 붙지만
**같은 자릿수**로 추정된다 — 🟡 **파라미터 수 기반 추정이고 실측이 아니다.**

### 9.3 그리고 코드가 이미 있다

`ur_env/rlpd_actor.py`(커밋 `2b50d34`)가 upstream과 같은 구조를 이미 구현해 뒀다:
로컬 `agent.sample_actions`(`:196-206`), 비동기 파라미터 콜백(`:175-179`),
agentlace datastore 전송(`:96-104,162-179`).

죽어 있는 이유는 버그가 아니라 **결정**이다 — `actor_network.py:1467-1471`이
`network.type='agentlace'`를 `NotImplementedError`로 막는다. "laptop3 GPU가 약하니 전부
kanu로"라는 판단의 산물인데, **그 전제(GPU)가 지금은 성립하지 않는다**(§9.1).

⚠️ 다만 그 경로를 그대로 부활시키면 gRPC 스택(schema 3, sidecar, operator session, MANUAL/
AUTO)을 통째로 잃는다. **A는 "agentlace 복귀"가 아니라 "현재 gRPC에 파라미터 push 채널
추가"로 설계해야 한다.**

---

## 10. 아직 계측되지 않은 것 (인용 금지 목록)

- 서버 **per-stage 시간** — 락 대기 / `_encode` / 정책 추론 / 직렬화. **하나도 없다.**
- `env.step()` 자체의 wall-clock. 100 ms 페이싱과 디코드 꼬리가 분리되지 않는다.
- `build_sidecar`(resize+JPEG encode) 소요. 부착 **여부**만 센다(`sidecar_attached_steps`).
- per-step RPC latency 분포. 465 ms가 **상시인지 tail spike인지 알 수 없다.**
- 카메라 디코드 §3.1 수치는 **합성 JPEG 대용** 측정이다(실제 D435 캡처 아님).
- 정책의 CPU 추론 실측(§9.2는 추정).
- 🔴 **새 서버(`junhyeong_ai`) 기준 실기 루프 주기.** 이 문서의 465 / 512 / 854 ms는 전부
  **kanu 상대**다. 2026-07-31 이전 뒤 잰 것은 **로봇 없는 합성 RPC 수치**뿐이므로
  "이제 몇 Hz인가"에 답하지 못한다. Stage 0을 새 호스트에서 다시 돌리는 것이 그 답이다.
- 🔴 **새 서버 기준 no-submit probe**(replay·learner 없이 추론만). §3.2의 12.08 ms에
  대응하는 값이 아직 없어 "정책 추론은 무죄"를 새 호스트에서 재확인하지 못했다.

---

## 11. 근거 파일 색인

🪤 **다른 `.md` 문서를 가리키는 줄번호는 이미 밀렸다.** 2026-07-31 서버 이전 문서 정리에서
여러 문서 앞에 marker 블록이 들어갔다(확인함: `HIL_SERL_KANU_RUNBOOK_KO.md:91-93`,
`HANDOFF_NEXT_SESSION_KO.md:76-77`, `HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md:149-150`은
더 이상 그 내용이 아니다). **문서 줄번호는 인용 문구로 검색해서 찾아라.**
아래 표의 **코드 파일** 줄번호는 코드가 바뀌지 않는 한 유효하다.

| 주장 | 위치 |
|---|---|
| upstream actor 루프 · 로컬 추론 | `third_party/hil-serl/examples/train_rlpd.py:123-137,156-216` |
| upstream 비동기 파라미터 | `agentlace/trainer.py:377-381`, `agentlace/zmq_wrapper/broadcast.py:47-66` |
| upstream chunking 비활성 | `serl_launcher/wrappers/chunking.py:34-84` + experiment config 4종 |
| upstream 10 Hz 페이싱 | `serl_robot_infra/franka_env/envs/franka_env.py:84,110,230-231` |
| upstream 카메라 스레드 | `franka_env/camera/video_capture.py:13-28` |
| 우리 `env.step` 페이싱과 디코드 위치 | `ur_env/envs/ur7e_env.py:1444-1464, 1489-1490, 1931-1974` |
| actor 루프 직렬 구조 | `ur_env/remote_actor.py:1086,1140-1168,1190,1216,1342` |
| 블로킹 gRPC 호출 | `ur_env/grpc_actor_transport.py:670-690, 813` |
| 서버 전역 락 | `ur_env/actor_network.py:478, 626-777` |
| ingress 공유 락 | `ur_env/learner/ingress.py:70,103,123,128` |
| 인코딩이 락 안 | `ur_env/learner/feature_replay.py:553-599, 644-657` |
| 락 공유 배선 | `ur_env/learner/composition.py:475,489`, `ur_env/learner/batches.py:50-79` |
| 코로케이션이 설계 제약 | `ur_env/learner/composition.py:5-7` |
| 파이프라이닝 차단 | `proto/actor_transport.proto:128-139`, `actor_network.py:1320,1327,642-646,715-718,1430` |
| `RelativeFrame` 한 주기 전제 | `ur_env/envs/frame_wrappers.py:104-130` |
| WAIT 재발행 0.5 s | `ur_env/operator_session.py:25, 618` |
| "412 ms는 RPC" (주석, 계측 아님) | `ur_env/remote_actor.py:1007-1008` |
