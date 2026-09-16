# IFQL orange_bowl_in_purple_bowl 실물 배포 — 인수인계 (2026-09-16 저녁, 작업 중단 상태)

> 🛑 **이 문서는 "어디까지 됐고 무엇이 안 됐는지"의 스냅샷이다.** 2026-09-16 저녁, 이 작업을 하던
> 세션이 API 세션 한도로 끊겼다(에이전트 3명이 동시에 중단). 다른 계정/세션이 이어받는다.
> carrot 배포의 정본은 [`GELLO_UR7E_IFQL_DEPLOY.md`](./GELLO_UR7E_IFQL_DEPLOY.md) — 이 문서는 그
> 위에 orange 태스크를 얹는 델타만 다룬다.

## 0. 요청 (학습 세션 fm-rl-6e의 핸드오프 원문 요약)

- 어제 orange 모델(`ifql_orange_k0.9_lead2_s0`)이 **자연광 유입으로 실패** → 오늘 (a) feature-level 이미지
  증강(K=8, crop/shift/photometric) 재학습, (b) end-to-end 픽셀 런 추가.
- HF **public** `Bigenlight/orange-bowl-in-purple-bowl-ifql`(23 run). 후보 3개, **전부 실물 미검증**:
  1. **`ifql_orange_k0.9_lead2_aug8_p0.5_s0` @100000** — 증강판 1순위(val actor loss 0.151, −30%)
  2. `ifql_orange_k0.9_lead2_s0` @100000 — 어제 모델(대조군)
  3. `px_orange_impala_lead2_s0` @100000 — 픽셀 e2e, `policy_type: ifql_px`, 96²→84² center crop,
     norm_stats `norm_stats_orange_lead2_px96.json`, **서빙 latency 미실측 → refill 먼저 재라**
  (`px_orange_r18ss_lead2_s0`는 kanu에서 학습 중, 끝나면 HF에 추가 예정)
- 1·2의 norm_stats: `norm_stats_r18_ss_orange_lead2.json`. 학습 obs = frozen R18 spatial-softmax 1024×2캠 +
  proprio 7 = 2055-D, 30 Hz, chunk 24프레임(8 knot stride 3), **action_lead 2**. 서버 코드 변경 불필요.
- **평가 요청**: 각 후보 **bc(K=1) 먼저 → 되면 bon32**, 같은 세션 안에서 paired(같은 물체 초기 위치),
  **조명 조건(블라인드/자연광) 기록**, 실패 양상(정지/엉뚱한 곳/그립 실패) 한 줄씩. 결과는 fm-rl-6e 세션으로.

## 1. 된 것 ✅

| 항목 | 상태 | 위치 |
|---|---|---|
| orange 후보 3개 다운로드 (`flags.json`·norm_stats·`serve_meta.json`·`RUN.md`·`train.csv`·`params_100000.pkl`) | ✅ 539 MB | `~/carrot_ifql/hf_orange/<run>/` |
| 학습 세션 `deploy/` (`ifql_deploy_orange.yaml`, `orange_real_limits.json`, norm_stats 4종) | ✅ | `~/carrot_ifql/hf_orange/deploy/` |
| 새 코드 타르볼 `carrot_ifql_code_20260916` 추출 | ✅ (sha·D_c-null 패치 여부 **미확인** — §2-3) | `~/carrot_ifql/code/carrot_ifql_code_20260916/` |
| **`config/ifql_deploy_orange.yaml`** | ✅ 작성 + 검증 완료 | `ros2_ur_ws/src/gello_policy/config/ifql_deploy_orange.yaml` |
| 옛 carrot T2 스택 정리 | ✅ 5595 free, GPU 12 MiB | — |

`ifql_deploy_orange.yaml` 검증(2026-09-16, PyYAML): `start_pose [3.0719, -1.5399, 1.8313, -1.9200, -1.5763, -3.2654]`
(**j0 +3.07 — carrot의 −3.16과 반대 branch**, 학습 세션이 "+3.x 확인"), `joint_limits_lo/hi` =
`[2.2802, -1.9525, 1.2604, -2.7674, -1.8565, -4.0124] / [3.7485, -0.8701, 2.4085, -1.4695, -1.2226, -2.2844]`
— `orange_real_limits.json`의 `safety_lo_1p2x/hi_1p2x`와 **6축 전부 일치**(52 ep / raw_min·max 1.2× 대칭 확장),
학습 세션 `deploy/ifql_deploy_orange.yaml`과도 일치. start_pose 6축 envelope 안. `act_timeout 0.8 <
obs 0.9 < staleness 1.0`, `act_port 5595`. carrot yaml 대비 바뀐 키는 start_pose·envelope뿐이어야 한다(diff로 재확인 권장).

## 2. 안 된 것 ❌ — 이어받는 사람이 할 일 (순서대로)

### 2-1. `run_ur7e_ifql_real.sh` 태스크 일반화 — **반쯤 됐고, 이대로는 orange를 거부한다**
작업 트리에 WIP가 있다(커밋 `WIP` 표시). 들어간 것: `IFQL_TASK=carrot|orange` 프로파일 블록(run dir·norm_stats·
로그 태그 기본값, `hf_orange/` 서브디렉터리, 오타 fail-closed), 헤더 문서. **안 들어간 것**:
- **norm_stats 가드 본문이 아직 `*real_lead6*` 하드코딩** (`case "$(basename "${IFQL_NORM_STATS}")"` 부근과
  기동 후 로그 검사 두 곳). 설계는 헤더에 적혀 있다: **"서버 로그의 `norm_stats: <path>` basename ==
  `$IFQL_NORM_STATS` basename"** 정확 일치로 바꾼다(태스크·lead·px 무관, 지금보다 엄격). 불일치면 kill+fail.
- **`IFQL_PARAMS_FILE` 기본값이 태스크와 무관하게 `ifql_deploy.yaml`** — orange면 `ifql_deploy_orange.yaml`로.
- px 판별(헤더에 적힌 `agent ready: ... px=<bool>` 검사, `flags.json`의 `is_px_run` 규칙: `agent.encoder`가
  `multicam*`이거나 `env_name`이 `_px<digits>.npz`) — 미구현.
- 기동 전 프리플라이트(run dir에 `flags.json`+`params_<step>.pkl`, norm_stats 존재, yaml `act_port == IFQL_PORT`) — 미구현.
- **carrot 회귀 검증 필요**: 환경변수 0개로 어제와 같은 명령이 나와야 한다. 가짜 로그로 가드 3케이스(일치/불일치/줄 없음) 테스트.
- `ifql/setup_ifql_workspace.sh`는 **손대지 않았다**(`IFQL_TASK=orange` → `HF_REPO`·`hf_orange/`·후보 3개 run 선택이 계획).

### 2-2. yaml 마무리
- `diff config/ifql_deploy.yaml config/ifql_deploy_orange.yaml`로 바뀐 키가 start_pose·envelope(+주석)뿐인지 확인.
- 런처는 yaml을 **src 절대경로**로 넘기므로 `colcon build` 불필요.

### 2-3. 새 타르볼 `carrot_ifql_code_20260916` 확인
- `sha256sum`을 `hf_orange/code/*.sha256`와 대조. `diff` 20260915 ↔ 20260916 `vision_carrot/ifql_server.py`:
  **D_c-null 버그**(`int(d.get("D_c", …))`, orange norm_stats도 `"D_c": null`)가 고쳐졌는지 — 안 고쳐졌으면
  `ros2_ur_ws/ifql/ifql_server_Dc_null.patch` 적용(`.orig` 보존). 학습 세션은 "로컬 소스엔 이미 있고 타르볼만 stale"이라 했다.
- 런처의 `find ifql_server.py`가 **스냅샷 2개를 찾으면 에러**한다 → `IFQL_SERVER_PY`로 하나를 명시하거나 옛 스냅샷을 치운다.

### 2-4. 오프라인 GPU smoke (로봇 없이, 후보당 1회, 한 번에 서버 하나)
carrot 때 절차(`GELLO_UR7E_IFQL_DEPLOY.md` §7 smoke) 그대로: `--sampler bc --device cuda`, orange take frame 0
(HF `Bigenlight/orange_bowl_in_purple_bowl_raw`), RESET + ACT 48회, `obs_assembler.py` 클라이언트, **PID로만 종료**.
특히 **px 후보의 refill이 `act_timeout 0.8 s` 안인지** — 이것이 핸드오프의 명시 요청.

### 2-5. 실물
`launch_cameras.sh` → `IFQL_TASK=orange IFQL_SAMPLER=bc HEADLESS=true ./run_ur7e_ifql_real.sh` →
후보 바꿀 땐 `IFQL_RUN_DIR=~/carrot_ifql/hf_orange/<run>`만. 트라이얼마다 **조명(블라인드/자연광)·배치 id·bc/bon·결과·
실패 양상·time-to-success** 기록. 결과는 fm-rl-6e 세션에 회신(`refill_stats.jsonl`도 같이).

## 3. 환경 사실 (laptop3, 2026-09-16 저녁)

- GPU RTX 3060 정상(`nvidia-smi` OK). 🪤 **suspend/resume 뒤 새 프로세스 CUDA가 죽는다** — 뚜껑 닫지 말 것,
  죽으면 모듈 재로드/재부팅(`GELLO_UR7E_IFQL_DEPLOY.md` §7).
- 디스크 **14 GB 남음**(97%) — 더 받지 말 것. `~/carrot_ifql/.venv-svf` 7.5 GB가 제일 크다.
- 카메라(T1)·로봇 스택(T2) **모두 내려가 있다.** 로봇은 마지막에 carrot start pose(HOLD)에서 정지.
- 이 리포 branch `feat/sim-data-collection`, 세 세션(학습 fm-rl-6e · 수집 · 추론)이 같은 트리를 공유.
- 학습 세션 연락: cross-session `bridge:session_011FYPbxQ6iYg81vXi9MNPmf`(이름 "IFQL 세션 kanu GPU 정책 확인").

## 4. 관련 커밋
`1525819`(carrot glue) · `495e1b3`(QFLOW_DIR/D_c) · `ec71878`(GO TO START) · `10cf3b9`(setup 스크립트) · 이 문서 + WIP.
