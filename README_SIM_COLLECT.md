# research/sim-collect — MuJoCo GELLO 데이터 수집 시뮬레이터

`ur7e-jazzy`(slim 베이스) 위에 `sim_collect/` 연구 스택을 복원한 브랜치다. 베이스 코드가 바뀌면
`git rebase ur7e-jazzy`로 따라간다 (`ros2_ur_ws`의 `ur_gello_bringup`, `gello_recorder`를 import한다).

## 무엇인가
- `sim_collect/` — UR7e MuJoCo 씬(당근/냄비, 그릇 쌓기), GELLO 리더 입력 → EEF 제어, 하드웨어와 같은
  `RecordingSession` 포맷으로 take 기록, 평가 하네스(`sim_collect/eval/`), 에셋(LIBERO/YCB 유래, 라이선스 동봉).
- `scripts/dataset/convert_carrot_to_lerobot*.py`, `validate_carrot_conversion.py`, `make_carrot_raw_stats.py`
  및 검증 리포트 `*.json` — carrot 코퍼스 전용 변환 도구. 일반 변환기(`convert_takes_to_lerobot.py`)의
  거부 메시지가 이 브랜치를 가리킨다.
- 설계/상태 문서: `sim_collect/README.md`, `DESIGN.md`, `STATUS.md`, `eval/README.md`.

## 실행
```bash
# 수집 (GUI: DISPLAY + GELLO 리더 필요) — 자세한 옵션은 sim_collect/README.md
./sim_collect/run_sim_collect.sh
# 헤드리스 테스트
git submodule update --init third_party/mujoco_menagerie   # ur7e.xml이 menagerie 메시를 참조한다
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
  conda run -n gello-sim python -m pytest -q sim_collect/tests \
  --ignore=sim_collect/tests/test_eval_policies.py --ignore=sim_collect/tests/test_eval_run.py
```

## 알려진 의존/한계
- `sim_collect/eval`의 ZmqPolicy/`serve_policy.sh`는 `ros2_ur_ws/src/gello_policy`가 필요하다 →
  `research/policy-deploy`와 함께 쓸 때만 동작 (이 브랜치 단독으로는 해당 테스트 8건이 실패/수집 불가).
- `test_controller.py`는 `v_max == 0.16`을 가정하는데 현재 설정은 0.2 → 9 errors (9568bdc에서도 동일, 기존 불일치).
- `test_recorder.py`의 일부는 `ur_env`(research/hil-serl)가 있을 때만 돈다 (없으면 skip).

## 베이스 브랜치의 작은 시뮬 예제와의 관계
`docs/sim/QUICKSTART_KO.md`(베이스, 작성 중)는 최소 예제(Panda/UR 시뮬 텔레옵)를 다룬다.
이 브랜치는 그 위의 연구용 수집·평가 스택이며 베이스에는 들어가지 않는다.
