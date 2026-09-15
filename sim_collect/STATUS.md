# sim_collect — 현재 상황과 만든 기능 (2026-09-15 기준)

이 문서는 **상태 보고서**다. 세부 절차는 링크된 런북에 있고, 여기에는 "무엇을 만들었고, 어디까지 됐고, 무엇이 남았나"만 적는다.
브랜치: `feat/sim-data-collection` (origin에 푸시됨).

## 한 줄 요약

GELLO로 MuJoCo 속 UR7e를 조종해 **실기 레코더와 같은 형식**으로 시연을 녹화하는 환경을 만들었고, 그걸로 모은
**당근→냄비 시연 22개**를 허깅페이스에 raw/LeRobot 두 가지로 올렸으며, 학습된 정책의 **성공률(SR)을 MuJoCo에서
자동 측정하는 평가 하네스**를 만들어 첫 모델(sim ACT)의 held-out SR을 쟀다. 지금은 Diffusion·Flow-Matching 학습이
kanu에서 진행 중이고 체크포인트가 나올 때마다 같은 하네스로 SR을 잰다.

## 만든 것

| 기능 | 위치 | 상태 |
| --- | --- | --- |
| **시뮬 데이터 수집 환경** — 물리·GELLO·EEF/joint 제어(`sim_main`), 카메라 2대 + 레코더(`capture`), 조작 GUI(`gui`), 런처 | `sim_collect/`, 런북 [`README.md`](README.md), 설계 [`DESIGN.md`](DESIGN.md) | ✅ 실기 GELLO로 22 take 수집 완료 |
| 씬: UR7e(menagerie UR5e 기하 + 정확한 URDF 오프셋) + 2F-85, LIBERO 나무 바닥, 절차적 당근·냄비, seed로 배치·방향 흔들기; 후보 물체 8종·용기 3종은 에셋으로 보관 | `assets/`, `configs/carrot_in_pot_sim.yaml`, `scene.py` | ✅ |
| 실기 형식 저장(`vectors.h5` 9테이블 + cam1/cam2 mp4, depth 옵션) + **MuJoCo 씬 재구성 데이터**(`/sim_scene` MJCF, `/sim_mj_state` 전체 상태 125 Hz) | `recorder.py`, `tools/replay_take.py` | ✅ 재구성 오차 0.00 mm |
| **재타이밍** — 라이브 캡처가 24~27 fps일 때 저장된 상태로 정확히 30 Hz 프레임을 다시 렌더 | `tools/retime_take.py` | ✅ 22 take 적용 |
| 데이터셋 변환·검증(depth 없는 sim 모드) | `scripts/dataset/convert_carrot_to_lerobot_sim.py`, `validate_carrot_conversion.py --sim`, `make_carrot_raw_stats.py --no-depth` | ✅ 검증 63항목 PASS |
| **SR 평가 하네스** — 실기 배포와 같은 ZMQ 정책 프로토콜·클램프·250 Hz 업샘플, 고정 seed, 정책 4종(zmq/scripted/replay/zero), 리포트·비교, kanu 서빙 래퍼 | `sim_collect/eval/`, 런북 [`eval/README.md`](eval/README.md), 설계 [`eval/DESIGN.md`](eval/DESIGN.md) | ✅ 기준선: zero 0/5, 오라클 20/20, 시연 재생 18/22 |

테스트: `sim_collect/tests` **257 passed** (`.venv`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`).

## 공개한 데이터셋 (허깅페이스, `Bigenlight`)

| 리포 | 내용 |
| --- | --- |
| `Bigenlight/carrot_in_pot_sim_raw` | 원본 22 take(`takes/`) + 30 Hz 재타이밍 22 take(`retimed_30hz/`), 통계·카드·첫 프레임, 1.3 GB |
| `Bigenlight/carrot_in_pot_sim_lerobot_v3` (태그 `v3.0`) | 22 에피소드 / 10,725 프레임 / 30 fps, 관절 7/7(실기 `carrot_in_pot_lerobot_v3`와 같은 feature 이름), task `"Put carrot in pot"`, depth 없음, 추가 `observation.sim.object_poses`(정책 입력 금지) |

## 학습·평가 현황

학습은 옆 세션 `laptop3 학습`이 kanu(GPU 최대 2장)에서 수행, 평가는 이 리포의 하네스로 laptop3에서 수행.

| 모델 | 학습 | held-out SR (seed 100~119, 20 에피소드) |
| --- | --- | --- |
| sim ACT | ✅ 50k 종료 | 10k 7 · 20k 7 · 30k 6 · 40k 7 · **50k 10** (/20) |
| sim Diffusion | ✅ 100k 종료(8h01m) | 10k 0/10 · 20k 1/10 · 30k 12 · 40k 10 · 50k 7 · 60k 9 · **70k 13/20**(최고) · 80k 8 · 90k 8 · 100k 11 — 30k 이후 ±20 % 구간 안 등락 |
| sim Flow-Matching | 🔄 kanu GPU 1에서 진행 중(100k, ~20:40 종료) | 10k 4 · 20k 7 · 30k 4 · 40k 6 · **50k 9** · 60k 7 (/20; CPU, Euler 10스텝 — 실기 서버 기본 100스텝과 다름) · 이후 나오는 대로 |

- 지배적 실패는 **"당근에 접근을 못 함"**(50k held-out 실패 10건 중 8건). 시연 22개의 배치 범위(당근 x 0.40~0.49, y 0.12~0.24)가
  좁아 그 밖으로 일반화가 안 되는 것으로 보인다.
- ⚠️ **seed 함정**: 시연은 RESET마다 seed 0,1,2,…로 녹화돼 eval seed 0~19가 학습 배치와 **정확히 같다**. 그 seed로 잰 SR
  (ACT 11~12/20)은 학습 배치 재현율이지 일반화 지표가 아니다. 기본 seed는 100~119로 바꿨다.
- kanu 체크포인트 회전(사용자 승인): 각 작업의 마지막 것만 보존, 나머지는 laptop3로 복사·평가 후 삭제. laptop3
  `sim_collect/eval/ckpts/`(git 제외)에는 ACT 10k~50k 전부, Diffusion 30k·70k·80k+, FM 20k·40k+를 보관하고, 평가가 끝난
  하위 체크포인트(Diff 10k/20k/40k/50k/60k, FM 10k/30k)는 디스크 확보를 위해 로컬에서도 지웠다(결과는 위 표에 기록).
- CPU 서빙 시 Diffusion/FM은 추론이 실기 클라이언트 타임아웃(0.6 s)보다 느려 `--timeout-s 30`으로 잰다(록스텝이라 결과는
  같고 시간만 걸린다: Diffusion 에피소드당 ~78 s). kanu GPU 서빙이면 기본값으로 된다.
- 체크포인트 크기: lerobot Diffusion은 ResNet18×2 + U-Net(down_dims 512/1024/2048, ~2.6억 파라미터)이라 fp32 1.1 GB가
  정상이고 kanu의 3.3 GiB는 옵티마이저 상태 포함. 줄이려면 학습 설정 `down_dims`를 낮춘다.
- 평가 영상: `--video`로 뽑는다. ACT 50k seed 100~103 예시가 `sim_collect/eval/runs/act_carrot_sim_050000_video/`에 있다.


## 다른 PC에서 재현하기 (리포만 받아서)

GELLO 하드웨어는 **수집에만** 필요하다. 데이터셋 다운로드·변환·SR 평가는 리포 + 공개 데이터셋만으로 된다.

```bash
git clone <origin> gello_software && cd gello_software && git checkout feat/sim-data-collection
git submodule update --init third_party/mujoco_menagerie third_party/DynamixelSDK   # UR5e 메시 + 2F-85 (필수)
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r sim_collect/requirements-sim.txt \
   && uv pip install --python .venv/bin/python -e . -e third_party/DynamixelSDK/python
# 정책 서버(평가할 때만): python3.12 venv + sim_collect/requirements-policy-server.txt (torch는 CUDA에 맞게 먼저)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider sim_collect/tests   # 257 passed 기대
```

- **렌더 백엔드**: laptop3는 NVIDIA 드라이버가 없어 `MUJOCO_GL=glfw`+`DISPLAY`가 필요하지만, GPU 드라이버가 있는 PC는
  `MUJOCO_GL=egl`로 디스플레이 없이 돌아간다(`sim_collect/eval/README.md` 트러블슈팅).
- **데이터셋**: `Bigenlight/carrot_in_pot_sim_lerobot_v3`(학습용, LeRobot이 자동 다운로드)·`Bigenlight/carrot_in_pot_sim_raw`
  (`hf download Bigenlight/carrot_in_pot_sim_raw --repo-type dataset`; take를 `sim_collect/tools/replay_take.py`로 재구성 가능).
- **평가**: 체크포인트를 `sim_collect/eval/serve_policy.sh`로 서빙하고 `run_eval.py`를 돌린다(`eval/README.md` §2~§4).
  ROS는 필요 없다 — `ros2_ur_ws/src/{ur_gello_bringup,gello_recorder,gello_policy}`의 순수 파이썬 모듈만 import한다.
- `sim_collect/eval/ckpts/`·`eval/runs/`는 git에 없다. 체크포인트는 kanu 경로 또는 학습 세션에서 받는다.

## 알아 둘 함정 (전부 문서에 상세)

- 이 PC는 NVIDIA 드라이버가 안 떠 있어 렌더가 소프트웨어 GL(`MUJOCO_GL=glfw`만 됨). CPU 부하 시 캡처가 24~27 fps로
  떨어지며 레코더가 `sim_meta.problems`에 남긴다 → 재타이밍으로 해결.
- Dynamixel 드라이버는 GELLO 포트를 잡은 프로세스를 죽인다(플레이그라운드/ROS 텔레옵과 동시 실행 금지).
- 처음 22 take의 `layout_seed` 메타데이터는 0으로 잘못 기록됐다(실제 배치는 `sim_object_poses`에 정확). 이후 수정.
- 평가는 mujoco(`.venv`)와 lerobot(`act_venv`/kanu)이 다른 인터프리터라 정책 서버 프로세스를 따로 띄운다.

## 남은 결정 / 다음 단계

1. Diffusion·FM 체크포인트 SR 측정(자동, 알림 오는 대로). 20 seed 해상도(±20 %)에서는 체크포인트 간 차이가 대부분 잡음이라, 최종 비교는 seed를 50~100개로 늘려 상위 후보(Diff 30k, ACT 50k, Diff 60k)만 다시 재는 것이 맞다.
2. 일반화 개선: 더 넓은 배치(`random_xy_radius`·seed 범위 확대)에서 시연 추가 수집 → 재학습. **사용자 결정 필요.**
3. 실제 GELLO로 ENGAGE 텔레옵·파지는 수집 과정에서 사용자가 이미 수행(22 take). depth 수집은 옵션(`--depth`).
4. 실기 전이(sim→real)는 범위 밖: 시각 도메인 격차 큼(`eval/README.md` §8).

## 기록 위치

- 커밋 이력: `git log --oneline feat/gello-ur7e-humble-22.04..feat/sim-data-collection`
- 평가 결과 파일: `sim_collect/eval/runs/<이름>/summary.md` (git 제외), 요약 표는 `eval/README.md` "결과 기록" 절
- 데이터 카드: 각 허깅페이스 리포의 README
