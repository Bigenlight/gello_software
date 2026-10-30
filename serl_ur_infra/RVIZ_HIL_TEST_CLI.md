# RViz HIL 테스트 CLI 사용법 (mock, 실기 위험 0)

`serl_ur_infra`의 human-in-the-loop 개입 경로(`GelloIntervention`)를 **가짜 하드웨어(use_fake_hardware) ros2_control** 상대로 돌려, **실제 GELLO 리더**로 개입하면 RViz 속 UR7e가 따라오는 걸 확인하는 테스트다. 실제 로봇은 전혀 움직이지 않는다.

터미널 4개를 쓴다. 각 터미널 공통 선행:

```bash
cd ~/gello_software && source /opt/ros/humble/setup.bash && source ros2_ur_ws/install/setup.bash
```

---

## 사전 준비 (한 번만)

```bash
# 1) serl_ur_infra editable 설치 + pynput (데드맨 리스너)
python3 -m pip install --user pynput
python3 -m pip install --user -e ~/gello_software/serl_ur_infra --no-deps

# 2) ur_gello_bringup 빌드 (gello_publisher / gello_hil_gui 엔트리 + rviz 뷰)
cd ~/gello_software/ros2_ur_ws && colcon build --packages-select ur_gello_bringup gello_policy
```

> **빌드 규칙:** ROS 패키지(`ur_gello_bringup`, `gello_policy`)의 파이썬 노드/launch/entry-point/config를 바꾸면 `colcon build` 필요. 반면 **`serl_ur_infra`는 editable(pip -e) 설치라 재빌드 불필요** — 러너/wrapper/컨트롤러(`wrappers.py`, `policy_delta_controller.py`, `run_rviz_hil.py`)를 고치면 **T4만 재실행**하면 반영된다.

---

## 4개 터미널

### T1 — mock UR7e + RViz

```bash
cd ~/gello_software/ros2_ur_ws && ./run_mock_rviz.sh
```

- `run_mock_rviz.sh`가 `use_fake_hardware:=true`를 박아 넣는다. **직접 `ros2 launch`로 띄우지 말 것** — 이 랩톱의 Humble 드라이버는 Jazzy 이름 `use_mock_hardware`를 **조용히 무시**하고 실제 로봇(0.0.0.0:30001/30004)에 접속하려다 실패한다.
- 이 스크립트는 `launch_rviz:=false`로 스택을 띄우고, 오퍼레이터 시점 뷰(`rviz/hil_operator_view.rviz`)로 RViz를 직접 실행한다(공식 launch가 뷰를 하드코딩해 override 불가).

### T2 — 가짜 카메라

```bash
ros2 run gello_policy fake_diffusion_observations
```

> 실행 파일명은 **`fake_diffusion_observations`**(복수형, `_node` 없음)다. `reset()`이 매 에피소드에서 카메라 프레임을 디코드하므로 T2가 없으면 러너가 시작 직후 죽는다.

### T3 — 실제 GELLO 리더 발행 (`/gello/joint_states`)

```bash
GELLO_REPO_ROOT=$HOME/gello_software \
ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \
  ~/gello_software/ros2_ur_ws/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml
```

> **`GELLO_REPO_ROOT`가 반드시 필요.** `gello_publisher`는 리포 루트의 `gello` 파이썬 패키지를 import하는데, `ros2 run`으로 직접 띄우면 cwd가 sys.path에 안 올라가 못 찾는다(`run_ur7e_gello_real.sh`는 이 변수를 export하므로 문제없다). 물리 GELLO가 꽂혀 있어야 하고, 로봇 브리지는 필요 없다(리더 관절만 발행).

### T4 — HIL 러너

```bash
cd ~/gello_software/serl_ur_infra
python3 tests/run_rviz_hil.py --deadman topic      # GUI로 개입 (권장)
# 또는
python3 tests/run_rviz_hil.py                      # 스페이스바로 개입 (기본)
```

시작 시 `joint_states ok → leader ok → cameras ok → OPERATOR INSTRUCTIONS` 순으로 찍히면 정상.

유용한 플래그:
- `--deadman {spacebar,topic}` — `spacebar`(기본, 터미널 포커스 잡고 홀드) 또는 `topic`(아래 GUI)
- `--policy {zero,scripted,random}` — 개입 안 할 때 팔의 기본 동작(기본 `zero`=정지)
- `--episodes N`, `--max-steps N`, `--leader-timeout S`

---

## 개입 방법 두 가지

### (A) GUI (권장) — `--deadman topic`

T4를 `--deadman topic`으로 띄운 뒤, **별도 터미널**에서:

```bash
cd ~/gello_software/ros2_ur_ws && ./run_hil_gui.sh
```

- **ENGAGE**(초록, 두 번 클릭 확인) → GELLO를 움직이면 RViz 팔이 추종. **DISENGAGE**(파랑) → 정책 복귀.
- **감도 슬라이더**(0.10–1.00) = 리더 이동 배율. **다음 ENGAGE 때** 반영(진행 중 재스케일 안 함).
- GUI는 `/hil/deadman`(`Float32MultiArray [engaged, gain]`, 20 Hz)만 발행한다. 첫 메시지를 받은 뒤 하트비트가 0.5 s 끊기면 env가 `DeadmanHeartbeatStaleError`로 러너를 중단하며 정책으로 복귀하지 않는다.

### (B) 스페이스바 — 기본

T4를 플래그 없이 띄우고, **T4 터미널에 포커스를 준 채 스페이스바를 홀드**하면서 GELLO를 움직인다. 놓으면 정책 복귀.

> X11에서 키 오토리핏이 홀드를 "뗐다 눌렀다"로 잘못 보내 개입이 깜빡일 수 있다. 그럴 땐 GUI(A)를 쓰는 게 안정적이다.

---

## 동작 원리 / 기대치

- **개입 = engage 이후의 변화량**을 따라간다. ENGAGE 순간 앵커가 잠기고, 그 뒤 GELLO가 움직인 만큼이 로봇 명령이 된다. 누르기만 하고 GELLO를 안 움직이면 정지가 정상.
- 팔은 **10 Hz rate-limited chase**로 따라온다(천천히 움직이면 잘 붙고, 빠르면 부드럽게 지연). 실기 eef가 더 쫀쫀한 건 250 Hz라서다.
- HUD의 `held=True reject=STEP_LIMIT`가 **드물게** 뜨는 건 특이점 근처 안전 HOLD다. (예전엔 이게 폭풍처럼 떴는데, `policy_delta_controller.py`에 line-search를 넣어 거부 대신 스텝을 축소하도록 고쳤다.)

---

## 방향/좌표 주의 (중요)

- **제어 코드의 방향 매핑은 검증됐다:** offline 실측에서 리더 base +X/+Y/+Z → 로봇 base +X/+Y/+Z (identity), 기록되는 `intervene_action`도 참 base-frame이다. **절대 `wrappers.py`에서 X/Y를 부호 반전으로 "고치지" 말 것** — RViz 그림만 맞아 보이고 SERL 버퍼에 저장되는 base-frame 액션이 오염된다.
- RViz에서 방향이 반대로 **보이는** 원인은 (1) 기본 ur_description 카메라가 반대편에서 보는 것, (2) mock 시작 자세 `RESET_JOINTS`가 실기 작업 자세와 다른 쪽(예: −x 뒤쪽)이라 같은 올바른 모션이 팔 기준으론 반대로 보이는 것 — 관측/자세 이슈이지 매핑 버그가 아니다. (해결: 오퍼레이터 시점 rviz 뷰 + 실기와 같은 forward 시작 자세로 맞추기.)

---

## 자주 겪는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `Failed to connect to robot on IP 0.0.0.0:30001/30004`, spawner FATAL | T1을 `use_mock_hardware`로 띄웠거나 직접 launch함 → **`./run_mock_rviz.sh` 사용** |
| `ur_control_fake_safe.launch.py ... not found` | `gello_policy` 미빌드 → `colcon build --packages-select gello_policy` |
| `No module named 'gello'` (T3) | `GELLO_REPO_ROOT=$HOME/gello_software` 를 T3 명령 앞에 붙였는지 확인 |
| T4가 `reset()`에서 카메라 에러로 죽음 | T2(`fake_diffusion_observations`) 안 띄움 |
| `No module named 'ur_env'` | `pip install --user -e serl_ur_infra --no-deps` 안 함 |
| 스페이스바 개입이 깜빡임 | X11 키 오토리핏 → GUI(`--deadman topic`) 사용 |
