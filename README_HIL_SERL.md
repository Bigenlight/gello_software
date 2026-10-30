# research/hil-serl — HIL-SERL 온라인 RL 스택

이 브랜치는 **실기 UR7e에서 HIL-SERL(사람 개입 온라인 RL)** 을 돌리는 연구 스택이다.
코어(텔레옵·레코더·EEF GUI·브리지)는 슬림 베이스 **`ur7e-jazzy`** 에 있고, 이 브랜치는 그 위에
HIL 전용 파일만 얹는다.

- 코어 업데이트를 받으려면 **`ur7e-jazzy` 위로 rebase** 한다: `git rebase ur7e-jazzy`
- 코어 파일을 여기서 고치지 말 것 — `ur7e-jazzy`에서 고치고 rebase로 받는다.

## 이 브랜치가 더하는 것

| 영역 | 경로 |
| --- | --- |
| env · actor · learner · 서버 | `serl_ur_infra/` (`ur_env/offline_rl/`은 `research/offline-rl` 소속) |
| upstream hil-serl | `third_party/hil-serl` (submodule: `git submodule update --init third_party/hil-serl`) |
| 운영 스크립트 | `ros2_ur_ws/run_hil_*.sh`, `_hil_*.{sh,py}`, BC/FM 정책 서버 래퍼(`run_bc_*`, `run_fm_*`, `fetch_policy_artifacts.sh`, `setup_local_policy_venv.sh`) |
| GUI / 분류기 | `gello_hil_gui`(ur_gello_bringup), `task_recorder_gui` · `reward_classifier` · `remote_reward_classifier` · `classifier_view_gui`(gello_recorder) |
| 문서 | `CLAUDE.md`(HIL 색인), `AGENTS.md`, `docs/testing/`, `serl_ur_infra/*.md` |

## 실행 (3 terminal)

```bash
cd ros2_ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur_gello_bringup gello_recorder

./run_hil_server.sh    # Terminal 1: GPU 서버 learner 재사용/기동 + SSH tunnel
./run_hil_hardware.sh  # Terminal 2: UR7e + Robotiq + passive GELLO
./run_hil_session.sh   # Terminal 3: cameras + GUI + preposition/preflight + actor
```

⚠️ 세 스크립트 모두 **실제 로봇을 움직이거나 GPU 서버에 접속한다.** 안전 수칙·preflight·
비상 정지는 `docs/testing/00_SETUP_AND_SAFETY.md`, 운영 절차는 `docs/testing/09_HIL_ACTOR_RUNBOOK.md`.

## 더 읽을 것

**`CLAUDE.md`** 가 이 스택의 문서 색인이다(상태·금지사항·테스트 기준선). 그 안의 정책 배포
(ACT/Diffusion/FM, `gello_policy`) 문서 링크는 `research/policy-deploy` 브랜치에, 일부 낡은 문서는
archive에 있으므로 이 브랜치에서는 깨진 링크로 보일 수 있다.
