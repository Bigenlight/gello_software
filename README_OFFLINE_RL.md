# research/offline-rl (WIP)

`ur7e-jazzy` 위에 얹은 오프라인 IFQL 학습 스택 연구 브랜치다. 로봇/ROS 코드는 없다.

## 내용
- `serl_ur_infra/ur_env/offline_rl/` — 온라인 RLPD 런타임과 독립인 오프라인 RL 부품:
  IFQL(expectile V + flow-matching actor 후보 샘플링), 청크 배치, 정규화
  (none/mean_std/min_max/quantile), 비전 feature(flatten / spatial_softmax), Q 집계, trainer.
  상세는 `serl_ur_infra/ur_env/offline_rl/README.md`.
- `serl_ur_infra/scripts/smoke_offline_ifql.py` — 합성 feature로 실제 MLP head를 최적화하는 CPU 스모크.
- 원본은 WIP 커밋 `7d5e9e2`(end-to-end 미검증). 같은 커밋의 replay-snapshot 복원
  (`learner/emergency.py`)과 actor/learner GPU 게이트(`learner/scheduling.py`)는 `research/hil-serl` 쪽에 있다.

## 실행
```bash
export JAX_PLATFORMS=cpu PYTHONPATH=serl_ur_infra   # conda env `il` (jax 0.5.3, flax, optax; pytest 없음)
python -m unittest serl_ur_infra/tests/test_offline_rl.py serl_ur_infra/tests/test_offline_normalization.py
python serl_ur_infra/scripts/smoke_offline_ifql.py --steps 3
```
`ur_env/__init__.py`는 이 브랜치에 없다(원본도 빈 파일) — namespace package로 import 된다.
flow sampler 어댑터 테스트 1건은 `ur_env.learner`가 필요해 이 브랜치에서는 skip 된다.

## 관계
- 기반은 `ur7e-jazzy`다. 코어 갱신은 rebase로 받는다.
- 경로를 원본과 동일하게 유지했으므로 `research/hil-serl`과 충돌 없이 합칠 수 있다.
