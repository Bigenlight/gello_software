# IFQL eval diagnostic video samples

`sim_collect.eval.render_eval_video` 출력 예시임. 세 파일 모두 1760×360, 30 fps,
H.264/yuv420p이며 cam1·cam2·diagnostic panel 순서로 배치됨.

- [ep_101_success_diagnostic.mp4](ep_101_success_diagnostic.mp4) — 실제 기록 episode의
  H5 kinematic replay. 성공, 266 frames. 이 과거 run은 Q sidecar가 없어서 panel과 graph에
  `Q: MISSING`, `UNAVAILABLE / Q NOT LOGGED`가 표시됨.
- [ep_100_timeout_diagnostic.mp4](ep_100_timeout_diagnostic.mp4) — 실제 기록 episode의
  H5 kinematic replay. timeout, 600 frames. Q sidecar 없음.
- [ep_101_success_diagnostic_synthetic.mp4](ep_101_success_diagnostic_synthetic.mp4) — panel과
  temporal graph layout 검사용 synthetic Q/latency. **실험 결과가 아니며**, 영상 camera 영역
  하단에 `SYNTHETIC DIAGNOSTICS - NOT AN EXPERIMENT RESULT` watermark가 항상 표시됨.

원본 eval run은 `ifql_sim_k0.9_s0_100000_bc1_s100to149`의 episode 100/101임.
실제 Q가 기록된 새 run은 `--policy-log-dir`을 함께 전달해 렌더하면 됨.
