# research/policy-deploy — 정책 배포·실기 평가 스택

이 브랜치는 slim 베이스 `ur7e-jazzy` 위에 **학습된 정책을 실물 UR7e에서 돌리고 평가하는 코드**를
얹은 연구 브랜치다. GELLO 텔레옵·레코더·브링업 같은 공통 코어는 `ur7e-jazzy`에 있으므로,
코어 업데이트를 받으려면 이 브랜치를 `ur7e-jazzy` 위로 **rebase**한다.

## 들어 있는 것

- `ros2_ur_ws/src/gello_policy/` — ACT / Diffusion / FM 원격 정책 서빙, `policy_leader_node`,
  IFQL / SVF / DSRL 배포 config(`config/*_deploy.yaml`), 앙상블·OOD 연구 코드
- `ros2_ur_ws/ifql/` — IFQL 워크스페이스 셋업·패치
- `ros2_ur_ws/setup_jazzy/` — eval 서버, preflight, smoke, `real_eval_recording_dev/`
- 평가 도구 — `camera_viewer_qt.py`(START/HOLD/GO TO START 버튼 + TagPanel), `eval_log_hub.py`,
  `pending_eval_tags.py`, `dataset_overlay.py`, `placement_metadata.py`
- `artifacts/ifql_eval_video_samples/` — IFQL 평가 영상 샘플

## 주요 런처 (`ros2_ur_ws/`)

| 런처 | 용도 |
| --- | --- |
| `run_real_policy.sh <profile> [--dry-run]` | **통합 진입점.** `--list`로 profile 목록 |
| `run_ur7e_{act,diffusion,fm,ifql,svf,dsrl}_real.sh` | 정책별 실기 실행 |
| `run_ur7e_*carrot*.sh`, `run_real_common_policy.sh` | carrot / common encoder 변형 |
| `launch_cameras.sh` | RealSense 2대 + 평가 뷰어(`--eval-log-root`) |
| `build_ur7e.sh` | `ur_gello_bringup gello_policy gello_recorder` 빌드 |

시작은 [`ros2_ur_ws/REAL_EVAL_QUICKSTART.md`](ros2_ur_ws/REAL_EVAL_QUICKSTART.md).
상세는 `ros2_ur_ws/REAL_EVAL_RECORDING.md`, `REAL_EVAL_DSRL.md`, `TRAINING_DATA_CONTRACT.md`,
`docs/ros2/GELLO_UR7E_{ACT,DIFFUSION,FM,IFQL}_DEPLOY.md`.

## 주의

- **모델 가중치는 git에 없다** — `ros2_ur_ws/setup_jazzy/staged_models/`는 gitignore 대상이다.
- `ros2_ur_ws/run_{bc,fm}_server*.sh`, `fetch_policy_artifacts.sh`, `setup_local_policy_venv.sh`,
  `run_bc_rollout_recorder.sh`, `_bc_rollout_status_logger.py`(serl_ur_infra BC/FM 서버 + HIL actor 경로)는
  **research/hil-serl 브랜치**에 있다. 이 브랜치의 FM 서버는 `src/gello_policy/scripts/run_fm_server.sh`다.
- `run_ur7e_ifql_real.sh`의 `REAL_EVAL_RECORDING=1` 기본 renderer hook
  (`sim_collect/eval/render_eval_video.py`)은 **research/sim-collect**에 있다.
