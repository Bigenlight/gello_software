# POLICY_EVAL_CODE_MAP_KO.md — BC/FM 평가 스택 코드·스키마 정본

**이 문서는 "어떻게 만들어져 있나"다.** 실행 절차·셋업은 `POLICY_EVAL_KO.md`가 담당한다.
줄번호는 밀린다 — **함수/클래스 이름이 정본**이고 줄번호는 보조다.

한 줄 요약: **production HIL-SERL의 gRPC 서버(`ActorSessionService`)를 그대로 쓰고, 정책
callable과 `accept_data` sink만 갈아 끼운 것**이 BC/FM 평가 스택 전부다. actor·하드웨어·GUI
코드는 한 줄도 바뀌지 않았다 — **유일한 예외가 옵트인 스텝 타이밍 계측**(`HIL_STEP_TIMING=1`,
기본 OFF)이고 그것도 액터 파이썬에 기록 경로 하나가 붙었을 뿐 **셸 스크립트는 여전히 무수정**이다
(→ §3.3).

```
actor(laptop3, 무수정) ──Step(픽셀 관측)──► 50153 터널 ──► 50054 BC / 50055 FM
                                                             │
   BC : VersionedPolicyRuntime(grafted params) ──argmax──────┤
   FM : pixels→FrozenResNet10TrunkExtractor→(1,1,4,4,512)→jit sample_action_chunks→chunk[0]
                                                             │
                          InferenceLoggingPolicy(로그) ───────┤
                          EpisodeRecordingSink(accept_data) ──┘  ← learner 없음, replay 없음
```

---

## 1. 코드 지도

### 1.1 Serving 부품 (`serl_ur_infra/ur_env/`)

| 파일 | 핵심 심볼 | 역할 / 반드시 알아야 할 것 |
| --- | --- | --- |
| `learner/bc_init.py` | `load_bc_init_manifest` · `verify_resnet_asset` · `load_bc_init_params` · `deterministic_sample_action` | bc-init artifact 로더. **`flax.serialization.from_bytes`를 쓰지 않는다** — flax 0.10.5 실측으로 여분 키를 조용히 버리고 shape 바뀐 leaf를 통과시키는 fail-open이라서. `msgpack_restore` 후 `_audit_against_template`가 template state-dict와 **양방향** 전수 대조(키·shape·dtype)하고, 통과해야만 `from_state_dict`가 돈다. graft 대상은 `BC_INIT_SUBTREES = ("modules_actor","modules_grasp_critic")` 둘뿐 — critic/temperature/frozen trunk는 template 것이 남아 trunk invariant가 살아 있다. **모듈 스코프는 stdlib 전용**(manifest 절반은 jax 없는 actor venv에서 돈다). |
| ⚠️ 같은 파일 | `deterministic_sample_action` | 두 개의 의도된 핀. **(a) runtime이 첫 인자로 넘기는 `params`를 `agent.replace(state=agent.state.replace(params=params))`로 반드시 주입한다** — 무시하고 `agent`만 클로저로 잡으면 **graft 안 된 template**이 서빙되는데 파라미터 서명·trunk invariant·action smoke가 **전부 초록으로 통과한다**(리뷰 지적사항). **(b) 와이어의 `deterministic`을 일부러 버리고 `argmax=True` 고정** — actor는 항상 `deterministic=False`를 보내므로 존중하면 BC의 노이즈를 평가하게 된다. |
| `fm_serving.py` | `FmServedPolicy` · `first_action_from_chunk` · `FM_MODEL_ID` | 픽셀↔feature, chunk↔단일 액션 **어댑터**. `_fm_observation`이 extractor 결과 `(1,4,4,512)`에 배치 축을 붙여 `(1,1,4,4,512)`로(`_fm_camera_feature`), state는 `(1,19)` 그대로. **receding horizon**: 매 Step 전체 chunk를 새로 계획하고 **첫 액션만** 실행 → 서버에 세션 상태가 없어 actor/ingress/sink 무변경. chunk 실행 모드가 필요하면 고칠 지점은 `first_action_from_chunk` **한 곳**이다. **재클램프 금지** — `sample_action_chunks`가 이미 `[-1,1]` clip + gripper 이산화를 했으므로 여기선 **검증하고 raise**한다. `deterministic`은 무시(가우시안 노이즈에서 적분 → "평균 액션"이 없다). 재현성은 `rng_seed`가 담당. |
| ⚠️ 같은 파일 | `_jitted_sampler` | `sample_action_chunks`를 **jit해 1회 캐시**. eager는 Euler 스텝마다 op 디스패치를 물어 **~500 ms/call 실측**(제어 주기 ~100 ms). 관측 shape가 계약으로 고정이라 트레이스 하나면 세션 전체를 덮고, **컴파일은 서버 startup smoke가 흡수**한 뒤 ready를 찍는다. 락은 PRNG split/`call_count`만 감싸고 **적분은 락 밖**에서 돈다. |
| `learner/flow_matching.py` | `FlowMatchingConfig` · `FlowMatchingPolicy` · `sample_action_chunks` · `load_flow_artifact` | 학습 담당자 인도물(모델 정의 + 로더). `_SpatialFeatureEncoder`가 rank-5 `(B,1,4,4,512)`를 요구하는 것이 `fm_serving`의 배치 축 추가 이유다. `sample_action_chunks`: `jax.lax.fori_loop`로 Euler 적분(기본 8스텝, `FlowMatchingConfig.integration_steps`) → `jnp.clip(-1,1)` → `discretize_gripper=True`면 채널 6을 `±1.0`으로. 학습 스크립트는 리포 안에 있다(`scripts/train_flow_matching_on_demos.py`). |
| `bc_recording_sink.py` | `EpisodeRecordingSink` | `ActorSessionService`의 `accept_data(data, intervened)` 구현. Step RPC 안, ACK **전에** 동기 호출된다. terminal 전이가 와야 episode pickle을 쓴다(중도 사망 시 그 episode의 pickle은 없고 `actions.jsonl`만 남는다 — 의도된 트레이드오프). **`open(path, "xb")` — 절대 `"wb"` 금지**: `(run_id, episode_id)` 중복은 서버가 구분 못 하는 두 episode를 봤다는 뜻이고 덮어쓰면 재현 불가능한 기록이 사라진다. `metadata.json`이 이미 있으면 `artifact_sha256`/`model_id` 일치를 요구한다(record root에 두 정책 혼입 금지). |
| 🛑 같은 파일 | **`prime_observation` 부재가 의도** | `ActorSessionService._prime_replay_observation`이 **sink에서** `getattr(self._accept_data, "prime_observation", None)`을 찾고(있으면 정책 입력을 픽셀 → frozen-trunk feature로 **바꾼다**), 없으면 `None`으로 떨어진다(`actor_network.py:1490`). BC agent와 FM extractor 둘 다 **원본 픽셀**을 받아야 하므로 이 클래스에 그 속성을 **추가하면 안 된다**. `FmServedPolicy`·`InferenceLoggingPolicy`도 같은 이유로 금지(전자는 위임 `__getattr__`조차 두지 않는다). |
| `bc_inference_log.py` | `InferenceLoggingPolicy` | 정책 callable을 감싸 **호출당 1줄** JSONL. **fail-open이 load-bearing**: 이 객체는 실기 UR7e 추론 경로 한복판에 있고, 여기서 던진 예외는 `ActorSessionService._infer`가 `PolicyInferenceError`로 바꿔 **팔이 뻗은 채 episode가 죽는다**. 그래서 로깅 블록 전체를 잡아 `log_error_count`에 세고 **stderr에 딱 한 번** 경고한다. 생성자는 파일시스템을 건드리지 않고(스트림은 첫 추론에 lazy open, append 전용), **결과 튜플은 객체 동일성 그대로 반환**한다(하류 `validate_action`이 정책이 만든 것을 그대로 봐야 한다). 이미지는 절대 안 남기고 state는 정확히 19-D일 때만. `__getattr__`이 감싼 정책에 위임하므로 `model_id`/`policy_version` 등은 그대로 읽힌다. |
| `step_timing.py` | `FailOpenJsonlWriter` · `StepTimingRecorder` · `TimingSink` · `ServiceStepTimingProxy` | **옵트인 스텝 latency 계측**(`HIL_STEP_TIMING=1`, 기본 OFF, **stdlib 전용**). `FailOpenJsonlWriter`는 `bc_inference_log.py`와 **같은 fail-open 패턴** — 기록이 실패해도 서빙을 절대 죽이지 않는다. `ServiceStepTimingProxy`는 `GrpcActorServicer`가 쓰는 **5개 메서드 표면**(`health` / `get_server_info` / `get_buffer_status` / `begin_episode` / `step`)을 그대로 노출한다. 스키마·조인 규칙은 **§3.3이 정본**이다. |
| 🛑 같은 파일 | `TimingSink`의 `prime_observation` 거부 | sink 래퍼는 위 🛑 규칙의 **반대편 함정**을 막는다: `prime_observation`을 가진 sink를 감싸면 `ActorSessionService._prime_replay_observation`의 `getattr`이 실패해 **feature-priming이 조용히 사라진다**. 그래서 그런 sink는 **생성 시 `ValueError`로 거부**하고, `TimingSink` 자신은 `prime_observation`도 위임 `__getattr__`도 두지 않는다. BC/FM sink(`EpisodeRecordingSink`)에는 애초에 그 속성이 없으므로 평가 스택에서는 항상 통과한다. |

### 1.2 서버 entrypoint (`serl_ur_infra/scripts/`)

| | `run_bc_policy_server.py` | `run_fm_policy_server.py` |
| --- | --- | --- |
| 포트(기본) | **50054** | **50055** |
| 로그 접두사 / ready 마커 | `[bc-server] ready ` | `[fm-server] ready ` |
| artifact 로드 | `load_bc_init_manifest` → `verify_resnet_asset` → agent template → `load_bc_init_params`(graft) | `load_flow_artifact(dir, which=best\|final)` → format 재확인 → `verify_resnet_asset` |
| 정책 객체 | `VersionedPolicyRuntime(agent_template, params=grafted, sample_action=deterministic_sample_action(...), parameter_validator=extractor.validate_parameter_invariant)` | `FmServedPolicy(model, params, extractor, policy_version=manifest["best_epoch"], rng_seed, integration_steps)` |
| startup smoke | **runtime 생성자가 수행**(`VersionedPolicyRuntime._smoke`) | **스크립트가 직접 2회** — `validated_smoke_action`이 `_validated_policy_action`을 미러링(dtype float32 / `validate_action` shape (7,) / gripper ∈{-1,0,1} / `validate_counter`). `FmServedPolicy`는 runtime 래퍼가 없어 아무도 안 보기 때문 |

**두 스크립트가 공유하는 조립 순서와 그 이유** (`_serve()`):

1. **jax import 전에** `_configure_env`가 `XLA_PYTHON_CLIENT_PREALLOCATE=false` / `CUDA_VISIBLE_DEVICES`를 `setdefault` — 그래서 파일 상단 `sys.path` 조작 위에서는 **jax/flax/grpc/protobuf를 import하지 않는다**(모든 `ur_env` import가 `_serve` 안에 있다).
2. `LearnerConfig()` 기본값 고정 — **config 노브를 노출하지 않는다.** 불일치는 load 시점 graft 거부로 터져야지 조용히 agent를 리셰이프하면 안 된다.
3. `record_root.mkdir(exist_ok=False)`는 **on-disk 부수효과 중 마지막** — 조립 실패가 빈 run 디렉터리를 남기지 않고 같은 경로 재시도가 된다.
4. `InferenceLoggingPolicy`는 **record_root 생성 이후에** 감싼다 — 로그 경로가 그 아래이고, startup smoke 추론이 rollout 로그에 섞이면 안 된다.
5. `ActorSessionService(..., reward_authority="local", accept_data=sink, observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH)` → `create_grpc_server`.
6. `server.start()` 뒤 **ready 한 줄**을 정확한 형식으로 출력 — 런처가 `grep -F`로 찾는 계약이다.

**loopback 전용**(`_LOOPBACK_HOSTS`) + **50053 거부**(`_PRODUCTION_LEARNER_PORT`): learner가 죽은 뒤 그 포트를 물면 frozen 가중치가 learner인 척 대답하게 된다. FM은 **50054를 경고만** 한다(BC eval이 끝났다면 합법).

### 1.3 런처 (`ros2_ur_ws/run_bc_server.sh` · `run_fm_server.sh`)

FM은 BC의 클론이고 차이는 **`FM_WHICH`(best/final)** 와 **`FM_ALLOW_PORT_50054`** 둘뿐이다.

| 단계 | 하는 일 | 실패 시 |
| --- | --- | --- |
| [1/5] 로컬 포트 | `BC_LOCAL_PORT`(50153) 점유 검사 — production `run_hil_server.sh` 터널과 같은 포트 | 배너 후 종료. **원격은 아무것도 안 건드림** |
| [2/5] HEAD 일치 | laptop3 `git rev-parse HEAD` vs `ssh <host> git -C $REMOTE_REPO rev-parse HEAD` | `die` (탈출구 `*_ACCEPT_HEAD_MISMATCH=1`). **보고이지 수리가 아니다** — fetch/reset/pull 안 함 |
| [3/5] 원격 stale | 원격 `ss -tln`으로 대상 포트 점유 확인. **ssh 실패를 FREE로 읽지 않는다**(grep miss와 종료코드가 같다) | 종료. 자동 kill 없음 |
| [4/5] 기동 | `rsh bash -s -- ...` **힙 문서로 마샬링** — `cd X && … & echo $!` 한 줄짜리는 `&`가 AND-list를 끝내 **셸 PID**를 잡고, 그러면 pidfile 기반 정지 경로가 통째로 죽는다 | `die` |
| [5/5] ready 폴링 | 포트가 아니라 **로그의 ready/FATAL 라인**을 폴링(포트는 gRPC만 증명). `*_START_TIMEOUT_S` 기본 300 | 마지막 40줄 출력 + **프로세스는 살려 둠**(증거 보존) |

**종료 trap의 3중 가드**(`stop_bc_server`): ① 자기 pidfile에서 읽은 숫자 PID일 것 ② 원격 `/proc/<pid>/cmdline`에 `run_bc_policy_server.py`가 있을 것 ③ TERM 10 s 후 KILL **직전에 cmdline 재검증**(PID 재사용 대비). **`pkill`/이름 매칭/패턴 매칭은 어디에도 없다.** ready에 도달한 뒤에야 trap을 설치한다 — 실패한 서버는 로그와 함께 살아 남는다.

환경변수는 production과 **공유 fallback**: `HIL_SSH_HOST`(junhyeong_ai) · `HIL_REMOTE_REPO`(`~/gello_software_runtime`) · `HIL_REMOTE_PYTHON` · `HIL_REMOTE_DATA_ROOT` · `HIL_GPU_INDEX`. run dir은 `<data root>/{bc,fm}_eval/{bc,fm}_eval_<UTC stamp>/`이고 서버 record-root는 그 아래 **`served/`**.

### 1.4 녹화 — 선택적 Terminal 4

| 파일 | 알아야 할 것 |
| --- | --- |
| `ros2_ur_ws/run_bc_rollout_recorder.sh` | 검증된 headless `gello_recorder`를 재사용해 **로봇 쪽만** 기록(READ-ONLY, 카메라를 **띄우지 않는다** — T3가 이미 소유). 두 자식을 `setsid`로 띄워 **각자 프로세스 그룹**을 준다: `ros2 run`은 파이썬 래퍼라 **신호를 전달하지 않고**, 래퍼만 INT하면 h5가 미완결로 남는다. 그래서 `stop_child`가 `kill -INT -<pgid>`를 쏜다. 대가: 이 스크립트가 SIGKILL되면 자식은 고아로 계속 녹화한다. |
| `ros2_ur_ws/_bc_rollout_status_logger.py` | `/hil/actor_status`(String) + `/hil/deadman`(Float32MultiArray)을 JSONL로. **시스템 `python3` + ROS overlay**로 돌린다(actor venv에는 rclpy 없음, `PYTHONPATH` 덮어쓰기 금지). ⚠️ **QoS는 두 퍼블리셔의 고정 계약인 depth 10 reliable = volatile이라 history가 없다 → actor보다 먼저 띄워야 초기 상태가 남는다.** `ts`는 퍼블리셔 스탬프가 아니라 **이 프로세스의 수신 시각**(두 토픽 다 스탬프가 없다). 파서가 없거나 payload가 깨져도 `parsed: null` + `raw` 보존, 절대 예외로 죽지 않는다. |

### 1.5 분석 — `serl_ur_infra/scripts/analyze_bc_rollout.py`

서버 쪽 `served/`와 laptop3 쪽 로봇 기록을 **하나의 리포트**(`<out>/report.json` + `rollout_report.md`, 기본 `<served>/analysis`)로 합친다. stdlib + numpy, **h5py는 `read_robot()` 안에서만** import(서버-only 경로는 h5py 의존이 없다). jax 없음, 소켓 없음, 픽셀은 리포트에 절대 안 들어간다(shape/dtype만, pickle은 하나씩 열고 즉시 버린다).

🕐 **시계 3개는 같은 시계가 아니다 — 정렬 앵커는 서버 ts가 아니다.**

| 소스 | 시계 | 쓰임 |
| --- | --- | --- |
| `actions.jsonl` `ts` | **GPU 서버 프로세스**의 `datetime.now(utc)`. laptop3 대비 skew **무보정** | fallback |
| pickle `meta.timestamp_ns` | **laptop3 actor**의 `time.time_ns()`(`remote_actor.EnvTimestampAdapter`) — 레코더와 **같은 호스트·같은 epoch** | ✅ **정렬 앵커(우선)** |
| `vectors.h5` `t_rel_s` | `time.time() - t0`(`RecordingSession.__init__`의 t0) = **상대 시계** | `synchronized.t_wall`(생 `time.time()`)로 `median(t_wall - t_rel_s)` = epoch 복원. 없으면 레코더 `metadata.json`의 `start_wall`(**naive 로컬시각**)로 강등 |

어느 쪽이 쓰였는지는 리포트 §4가 산문으로 명시한다 — 코드를 다시 열 필요가 없게.

**스텝 타이밍 입력**: 서버 `timing.jsonl`은 `--served` 디렉터리에서 **자동 발견**하고, 액터 쪽은
`--actor-timing <액터 jsonl>`로 직접 준다. 둘 중 하나라도 있으면 리포트 **§2 Inference log 아래**에
`### Latency breakdown (step timing)` 절이 붙어 phase별 count/mean/p50/p95/max와 `wire_ms`를 찍는다.
**아무 파일도 없으면 `NOT RECORDED` 한 줄로 우아하게 빠지고** 나머지 리포트는 그대로 나온다
(증거 절반이 없어도 리포트가 나온다는 §1.6의 규칙 그대로).

### 1.6 테스트 8파일

| 파일 | 인터프리터 / 게이트 | 개수 | 무엇을 못 박나 |
| --- | --- | --- | --- |
| `test_bc_init_loader.py` | actor venv (jax 불필요) | 21 | manifest/completion 바이트 게이트. payload는 **일부러 진짜 msgpack이 아니다** |
| `test_bc_recording_sink.py` | actor venv | 12 | 실 wire 계약 fixture → `load_demo_object` 왕복, `"xb"` 중복 거부 |
| `test_bc_inference_log.py` | actor venv | 19 | 결과 **무변경 반환**, 로깅 실패가 추론을 못 죽임 |
| `test_bc_server_inproc.py` | actor venv (grpc·jax 없이 서비스 객체 직접) | 7 | `GetServerInfo` 핸드셰이크 4종 · MANUAL MARK SUCCESS · **sink에 `prime_observation`이 없으면 픽셀이 그대로 간다** |
| `test_fm_serving.py` | actor venv | 23 | `first_action_from_chunk` 수용/거부 + 세 wire 상수 문자열 |
| `test_analyze_bc_rollout.py` | actor venv | 12 | fixture를 **진짜 sink가 쓴다**(추측 레이아웃 금지), 증거 절반이 없어도 리포트가 나온다 |
| `test_flow_matching.py` | **hilserl venv**(jax CPU), `importorskip("jax")` | 4 | 모델/손실/샘플러 계약 |
| `test_actual_bc_init_agent.py` | hilserl venv, **`RUN_HIL_SERL_ACTUAL_BC=1`** | 7 | 실 agent에 graft → **params 무시 훅이면 FAIL**, from_bytes가 삼키는 3형태(여분 키·리셰이프·누락)가 여기선 거부 |
| `test_actual_fm_serving.py` | hilserl venv, **`RUN_HIL_SERL_ACTUAL_FM=1`** (+`JAX_PLATFORMS=cpu`) | 14 | 서빙 액션 = 새 chunk의 첫 액션, **같은 `rng_seed`면 두 인스턴스가 정확히 일치**, artifact 왕복 + 다이제스트 부패 2종 |

정본 명령(기본 스위트 — 게이트 없는 6파일이 돌고 jax 3파일은 skip):

```bash
cd /home/laptop3/gello_software
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \
  /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q -p no:cacheprovider \
  serl_ur_infra/tests
```

jax 쪽은 같은 명령을 `/home/laptop3/venvs/hilserl/bin/python`으로 + `RUN_HIL_SERL_ACTUAL_BC=1 RUN_HIL_SERL_ACTUAL_FM=1 JAX_PLATFORMS=cpu`.

---

## 2. Artifact 포맷 계약

### `hil-serl-bc-init` v1 (`bc_init.BC_INIT_FORMAT` / `..._VERSION`)

| 파일 | 키 | 검사 |
| --- | --- | --- |
| `manifest.json` | `format` · `format_version` | 정확히 `"hil-serl-bc-init"` / `1` |
| | `parameter_subtrees` | **`["modules_actor","modules_grasp_critic"]`와 리스트 동등** |
| | `parameter_file` | artifact 디렉터리 **내부 상대경로**(절대·`..` 거부) |
| | `parameter_bytes` · `parameter_sha256` | 실제 파일 크기·SHA256과 일치 |
| | `resnet_sha256` | `verify_resnet_asset`가 **로컬 ResNet 자산**과 대조(graft는 template trunk를 남기므로 다른 trunk로 학습한 head는 하류에서 탐지 불가) |
| `completion.json` | `complete` | **`is not True` 거부** — truthy 문자열/1은 완료가 아니라 잘못된 writer |
| | `parameter_sha256` | manifest 값과 동일해야 함 |
| | `manifest_sha256` | **manifest 바이트**의 SHA256 (서명 체인을 닫는 고리) |
| `README.md` | — | 사람용, **파싱하지 않는다** |

orbax 체크포인트가 **아니다**(`production_checkpoint_compatible: false`). 리포에 BC 학습 코드는 **없다** — `bc_init.py`는 로드·검증·graft만 한다(의도).

### `hil-serl-jax-flow-matching` v1 (writer: `scripts/train_flow_matching_on_demos.py`)

| 키 | 내용 |
| --- | --- |
| `format` / `format_version` | `"hil-serl-jax-flow-matching"` / `1` |
| `parameter_files.best` / `.final` | `{path, bytes, sha256}` — 서버 `--which`가 고른다 |
| `model_config` | `FlowMatchingConfig.document()` (horizon 16, integration_steps 8, hidden_dim 256 …) → 로더가 `FlowMatchingConfig(**...)`로 되살려 template을 만든다 |
| `observation_contract` | `"cam1/cam2 frozen ResNet10 maps (1,4,4,512) + state (1,19)"` |
| `action_contract` | `"chunk[horizon,7]: normalized EEF delta[6] + gripper[-1,+1] …"` |
| `resnet_sha256` | **bc-init과 같은 키 이름** — 그래서 `verify_resnet_asset`를 두 패밀리가 그대로 공유한다 |
| `best_epoch` | 서버가 **`policy_version`으로 광고**한다 |
| `demo_files` / `training` / `*_metrics` | 학습 출처·하이퍼파라미터 (검증에는 안 쓰임) |
| `completion.json` | `complete` · `manifest_sha256` · `best_parameter_sha256` · `final_parameter_sha256` |

**새 artifact를 받으면 코드 변경 없이 `BC_ARTIFACT_DIR` / `FM_ARTIFACT_DIR`(또는 `--artifact-dir`)만 바꾸면 되는 이유**: 로더가 구조를 **런타임에 template과 전수 대조**한다 — BC는 `_audit_against_template`가 키·shape·dtype을, FM은 `model_config`로 세운 template에 `from_bytes`가. 구조가 다르면 **load에서 거부**되고 서버는 ready를 찍지 못한다. 다만 **`model_id`는 코드 상수**(`BC_MODEL_ID`/`FM_MODEL_ID`)이므로, 새 artifact가 다른 정책이면 상수와 actor 핀을 같이 바꿔야 한다(§4).

---

## 3. 기록물 스키마

### 3.1 서버측 `<run dir>/served/` (writer: `bc_recording_sink.py`, `bc_inference_log.py`)

| 경로 | 스키마 |
| --- | --- |
| `metadata.json` | `{artifact_sha256, model_id, created_at_utc}` — 존재하면 앞 둘의 **일치를 요구**한다 |
| `actions.jsonl` | 전이 1개당 1줄: `{ts, run_id, episode_id, step_id, env_step, actions[7], intervened, dones, truncated, success, rewards}`. **이미지 없음**(tail 가능해야 한다) |
| `inference.jsonl` | 추론 1회당 1줄: `{ts, latency_ms, deterministic, policy_version, action[7], state[19]\|null}`. `--no-inference-log`면 없음. **`actions.jsonl`과 줄 수가 다른 게 정상** — 캐시된 Step 응답·warm-up·actor가 버린 추론이 여기에만 남는다 |
| `timing.jsonl` | 옵트인 스텝 latency 분해(`HIL_STEP_TIMING=1`, 기본 OFF). 필드는 **§3.3**이 정본 |
| `<run_id>/episode_XXXX.pkl` | **완료된 episode당 1개**, pickle protocol 4. `{"meta":…, "transition":…}` wrapper dict의 **평범한 `list`** → `ur_env/learner/demo.py::load_demo_object`가 `set(item)=={"meta","transition"}`을 특수 처리하므로 **리포의 strict 로더로 그대로 읽힌다** |

`meta`: `schema_version, run_id, actor_id, session_id, transition_id, env_step, timestamp_ns, policy_version, policy_action, intervened, auto_success, operator_success, policy_actions_synthetic` / `transition`: `episode_id, step_id, observation_id, next_observation_id, actions, rewards, masks, dones, truncated, observations, next_observations, success, classifier_*`. **`run_id`는 `meta`가, `episode_id`/`step_id`는 `transition`이 권위**다.

### 3.2 로봇측 rollout 디렉터리 (T4)

```
<ROLLOUT_DIR>/
  robot/vectors.h5      9개 테이블: synchronized, gello_joint_states, ur_joint_states,
                        command, gripper, wrench, tcp_pose, cam1_frames, cam2_frames
                        (컬럼 순서는 각 그룹 attrs['columns'] JSON이 정본)
  robot/cam1.mp4 / cam2.mp4   프레임이 실제로 도착했을 때만 생성(lazy open) +
                        camera_warmup_s 기본 3.0 s 동안 드롭 → 짧은 rollout은 영상 없음
  robot/metadata.json   Ctrl-C 정상 종료 시 finalized
  status.jsonl          {ts, topic, raw, parsed} — parsed는 실패 시 null, raw는 항상 보존
```

### 3.3 스텝 타이밍 — `HIL_STEP_TIMING=1` (기본 OFF, writer: `step_timing.py`)

한 스텝의 시간이 **통신 / 모델 / 기록 / 로봇 관측** 중 어디에 쓰였는지 스텝 단위로 분해한다.
서버(T1)와 액터(T3)를 **각각** 켜며, 한쪽만 켜도 그쪽 분해는 나온다 — **둘 다 켜야 순수 통신
시간(`wire_ms`)이 계산된다.**

| 켜는 곳 | 방법 | 산출물 |
| --- | --- | --- |
| 서버(T1) | `HIL_STEP_TIMING=1 ./run_bc_server.sh` / `./run_fm_server.sh` (런처가 ssh 너머 서버 프로세스까지 전달) · 서버 스크립트 직접 기동이면 `--step-timing` | `<run dir>/served/timing.jsonl` (`inference.jsonl` 옆) |
| 액터(T3) | 평소 세션 명령 앞에 `HIL_STEP_TIMING=1`만 — 환경변수가 `run_hil_session.sh` → `run_hil_actor.sh` → actor 파이썬까지 그대로 흐르므로 **셸 스크립트는 무수정** | `gello_logs/step_timing/actor_step_timing_<ts>.jsonl` (또는 `--step-timing-path`) |

켜졌다는 표식: 서버는 **ready 라인에 `step_timing=1` 토큰**이 추가되고, 액터는 기동 로그에
`[actor] step timing ON -> <경로>` 한 줄.

**서버 `timing.jsonl`** — Step RPC당 1줄 + BeginEpisode당 1줄.

| 필드 | 뜻 |
| --- | --- |
| `ts` | 기록 시각 |
| `kind` | `"step"` \| `"begin_episode"` |
| `run_id` · `episode_id` · `step_id` · `env_step` · `transition_id` | 식별자 (조인 키는 `transition_id`) |
| `handler_ms` | **서비스 핸들러 총시간** (아래 셋의 상위 집합) |
| `infer_ms` | 정책 추론 — `ActionResult.server_inference_ms` 유래. **terminal 스텝은 `null`** |
| `sink_ms` | 기록 sink 쓰기 (`EpisodeRecordingSink`) |
| `overhead_ms` | `handler_ms − infer_ms − sink_ms` = **관측 디코드 / 검증 / 복사** |
| `terminal` · `deduplicated` · `error` | 그 Step의 성격 표식 |

**액터 jsonl** — **Step RPC를 완료한 루프 반복당 1줄**.

| 필드 | 뜻 |
| --- | --- |
| `ts` | 기록 시각 |
| `run_id` · `episode_id` · `step_id` · `env_step` · `transition_id` | 식별자 |
| `env_step_ms` | `env.step` — ⚠️ **~100 ms 자체 페이싱 sleep이 포함된다.** "환경 처리 비용"으로 읽으면 안 된다 |
| `build_ms` | 전이 조립 (`build_data`) |
| `sidecar_ms` | 분류기 sidecar JPEG 인코드. **미부착 스텝은 `null`** (평가 세션은 `--no-classifier-sidecar`라 항상 `null`) |
| `rpc_ms` | **클라이언트가 관측한** Step RPC 왕복 — proto 조립/파싱 포함 |
| `round_trip_ms` | 전송계층이 자체 측정한 왕복 |
| `server_inference_ms` | 서버가 회신한 추론 시간 |
| `loop_ms` | 반복 시작 → RPC 완료 |
| `post_prev_ms` | 직전 반복 emit → 이번 반복 시작. **RPC 이후 북키핑과 주기적 pickle 덤프가 여기 잡힌다** |
| `attached_sidecar` · `intervened` · `terminal` | 플래그 |

**조인 키와 파생값**

```
transition_id = "<run_id>:<env_step>"           # 서버·액터 공통 조인 키
wire_ms       = 액터 rpc_ms − 서버 handler_ms   # 터널 + gRPC 프레이밍 + 큐잉 = 순수 통신
```

**설계 사실 — 알고 있어야 할 것**

- **proto / `SCHEMA_VERSION` 무변경.** 신규 데이터는 전부 **서버·액터 로컬 jsonl**이고 조인은
  오프라인에서 `transition_id`로 한다. 그래서 한쪽만 켜도, 한쪽만 배포돼도 와이어가 깨지지 않는다.
- **`infer_ms` ≠ `inference.jsonl`의 `latency_ms`.** 앞은 **서비스 측정**이라 inference-log 래퍼의
  쓰기 비용까지 들어가고, 뒤는 **순수 모델 호출**만이다. µs 단위 차이지만 **같은 값이 아니다.**
- **기동 smoke 추론은 `timing.jsonl`에 섞이지 않는다** — smoke는 service가 아니라 policy를 **직접**
  호출하기 때문이다.
- **production HIL learner 서버(`run_rlpd_learner_server.py`, `:50053`)는 이 기능과 무관하게 무수정**이다.
- 기본이 OFF라 **평소 실행에는 오버헤드가 없고**, 켰을 때의 비용은 **스텝당 JSONL 한 줄 쓰기**
  수준이다. 그 이상은 아직 실측이 없다 — 숫자를 지어내지 말 것.

---

## 4. 핸드셰이크 계약 (actor가 실제로 검사하는 것)

| 항목 | 값 / 출처 |
| --- | --- |
| `EXPECTED_MODEL_ID` | BC `bc-cube-in-cup-raw0731-bcinit-v1` / FM `fm-cube-in-cup-raw0731-h16-euler8-v1` (production 기본값은 `hil-serl-hybrid-sac-…`이므로 **반드시 덮어써야 한다**) |
| `EXPECTED_REWARD_AUTHORITY` | **`local`** (production은 `server_classifier`) |
| `EXPECTED_REWARD_MODEL_ID` | **`operator-manual-success-v1`** — BC/FM이 **같은 문자열을 고의로 공유**한다(의미가 같다) |
| protocol / schema | `PROTOCOL_VERSION="2"` / `SCHEMA_VERSION=3` (`actor_network.py:43-44`) — 서버가 그대로 광고 |
| 관측 스키마 | `CANONICAL_OBSERVATION_SCHEMA_HASH` (actor 기본 `OBS_SCHEMA_HASH=3459098d…2903`) |
| 액션 | `validate_action(shape=(7,))` — float32, 유한, **모든 성분 ∈[-1,1]** |
| gripper | 채널 6 ∈ `{-1.0, 0.0, +1.0}` |

세 `expected_*`는 `GrpcActorNetwork`가 **하나라도 설정되면** `GetServerInfo`를 검증한다(빈 문자열은 생성자에서 거부). BC/FM 서버는 세 값을 `ActorSessionService(model_id=…, reward_authority="local", reward_model_id=…)`로 광고한다.

🟢 **MANUAL `MARK SUCCESS`는 서버 classifier 없이 동작한다.** 프로세스에 분류기가 없어도 서비스의 **기본 identity finalizer**(`ActorSessionService._finalize_transition_identity`)가 `meta.operator_success`를 `rewards=1.0 / masks=0.0 / dones=True / truncated=False`로 승격하고, `classifier_evaluated=0`으로 기록한다. `auto_success`와 동시 설정은 `ActorProtocolError`. **평가 세션의 유일한 성공 경로다.**

---

## 5. 확장 포인트

**새 정책 패밀리(예: diffusion) 추가 — 6단계.** 아래 순서로 하면 기존 파일 수정은 0에 가깝다.

1. **serving 모듈** `ur_env/<family>_serving.py` — `fm_serving.py`를 본떠 `(observation, deterministic) -> (np.float32(7,), int)` 하나만 만족시킨다. 모듈 스코프는 stdlib+numpy(actor venv 임포트 가능), jax는 메서드 안에서 lazy. **`prime_observation`도, 위임 `__getattr__`도 만들지 않는다.**
2. **서버 스크립트** `scripts/run_<family>_policy_server.py` — **새 포트**(50056…)를 쓰고 50053/50054/50055를 거부·경고한다. §1.2의 6단계 조립 순서와 ready 라인 형식을 그대로 유지(런처가 grep한다).
3. **런처** `ros2_ur_ws/run_<family>_server.sh` — `run_fm_server.sh`를 복사하고 포트·마커·기본 artifact만 교체. 5단계 게이트와 3중 가드는 **손대지 않는다**.
4. **테스트** — 최소 3개: 순수 numpy 계약(actor venv), artifact 로더 게이트, `RUN_HIL_SERL_ACTUAL_<X>=1` 게이트의 실 jax 통합.
5. **런북** — `POLICY_EVAL_KO.md`에 터미널 절차 추가.
6. **핀 결정** — `<X>_MODEL_ID` 상수를 정하고, 그 값이 그대로 `EXPECTED_MODEL_ID` export가 된다. reward는 `local` + `operator-manual-success-v1`을 재사용한다(의미가 같으면 새 id를 만들지 않는다).

**킵된 후속 기획: `policy_detail.jsonl`.** 정책 패밀리별 내부 상태(BC는 분포 파라미터/엔트로피, FM은 **chunk 전체**와 적분 스텝)를 `inference.jsonl` 옆에 따로 남기자는 안. ⚠️ **현재 코드에는 존재하지 않는다**(리포 전체 grep 0건). 구현한다면 자리는 `InferenceLoggingPolicy`가 아니라 **각 serving 모듈**이다 — 로거는 정책 내부를 알아서는 안 되고, `fail-open` 규칙(§1.1)을 그대로 물려받아야 한다.
