# carrot_in_pot EEF 세션의 "0.9 초 지연" — 원인 규명 노트

작성: A7, 2026-09-14. 대상 데이터 `/home/laptop3/youngwoong_ws/Put_carrot_in_pot/take_*`
(54 take, EEF 텔레옵, 2026-09-14) 와 비교군 `/home/laptop3/youngwoong_ws/Put_cube_in_cup/take_*`
(joint 텔레옵, 2026-07-20). 코드는 워크트리 `/home/laptop3/gello_software_humble`.
**이 노트는 브리지를 수정하지 않는다.** 수정 제안은 §7에 적고 구현하지 않았다.

---

## 0. 한 줄 결론

> **팔은 늦지 않았다. 늦은 것은 기록이다.**
> `command` 가 `ur_joint_states` 를 0.9 초 앞서는 것은 로봇·드라이버·컨트롤러·One-Euro
> 필터와 아무 관계가 없다. **레코더의 구독 큐에 쌓인 오래된 메시지를 도착 시각으로
> 찍었기 때문**이고, 각 토픽이 늦는 양은 정확히 `QoS depth ÷ 발행 주파수` 다.
> 그래서 "제어 지연"으로 고칠 것은 없고, **테이블마다 다른 상수만큼 시간축을 되돌려야**
> 한다 (§6 표).

물리적으로 실재하는 지연은 따로 있고 훨씬 작다: **리더 → command 약 0.25 초**
(One-Euro + 슬루), 그리고 **v_max 0.16 m/s 속도 거버너** (§4). 배포 경로에 남는 것은
이 둘뿐이다 (§5).

---

## 1. 측정 요약

전부 **속도 프로파일 상호상관** 으로 쟀다 — 위치 거리가 아니라 속도를 쓰면 상수 오프셋
(예: 0.174 m 툴 오프셋) 이 결과를 오염시키지 못한다. `+τ` = 앞 스트림이 뒤 스트림보다
τ 초 **앞선다** (= 뒤 스트림이 τ 초 낡았다). 스크립트: `scratchpad/xcorr.py` (§8).

| 쌍 | carrot (9월, EEF, depth ON) | cube (7월, joint, depth 없음) |
| --- | --- | --- |
| `command` → `ur_joint_states` | **+0.910 s** (r 0.73–0.89) | +0.05 s |
| `command` → `tcp_pose` | **+0.407 s** | +0.07 s |
| `tcp_pose` → `ur_joint_states` | **+0.495 s** | **+0.000 s** (8/8 take) |
| `tcp_pose` → `wrench` | **+0.000 s** (7/8 take) | **+0.000 s** (8/8 take) |
| `gello_joint_states` → `command` | +0.260 s | +0.253 s |
| `gello_grip` → `grip_cmd` | +0.015 s | +0.015 s |
| `grip_cmd` → `grip_pos` | +0.367 s | +0.335 s |

TCP 거리 기준(적분자가 쓴 지표)으로도 같다: 54 take 중 11 take 전수에서 최적 τ = **+0.90 s**,
그 지점 잔차 중앙값 **0.42–0.64 mm**, τ=0 에서는 44–67 mm (`scratchpad/lag1.py`).
**잔차가 0.5 mm 로 떨어진다는 것 자체가 진단이다** — 서보 추종 지연은 저역통과라 형상이
뭉개지지만, 여기서는 파형이 완전히 동일하고 시간축만 밀려 있다. 즉 **순수 지연**이다.

---

## 2. "레코더 아티팩트" 를 반증 불가능하게 만드는 네 가지

### 2.1 `tcp_pose` 와 `ur_joint_states` 가 0.495 초 어긋난다 — 물리적으로 불가능하다

둘 다 **측정값**이고, UR 드라이버 안에서 **같은 RTDE 패킷** 으로부터 같은
controller_manager 사이클에 `joint_state_broadcaster` / `tcp_pose_broadcaster` 가 각각
발행한다. 물리적 시차는 0 이어야 한다. 그런데 기록에서는 0.495 초 벌어져 있다
(54 take 중 검사한 11 take 전부 0.465–0.535). 팔이 늦었다는 가설로는 이 값을 만들 수 없다.
**레코더가 두 스트림을 서로 다른 양만큼 늦게 스탬프했다는 뜻 외에는 해석이 없다.**

덤으로: 같은 τ=+0.5 에서 `fk(q) ⊕ 0.174` 와 `tcp_pose` 가 **0.85–1.10 mm** 로 일치한다
(τ=0 에서는 22–35 mm). DESIGN §1 의 "TCP 가 recorder 의 `tcp_pose` 와 ~1 mm 일치" 는
**이 0.50 초를 뺀 뒤에만** 성립한다. `TOOL_Z_M = 0.174` 자체는 이로써 오히려 더 강하게 확인됐다.

### 2.2 `tcp_pose` 와 `wrench` 는 정확히 0.000 초다

두 토픽은 **같은 QoS depth(50)** 이고 **같은 발행 주파수**(같은 controller_manager 사이클)다.
모델이 예측하는 상대 지연은 정확히 0 이고, 측정값도 5 ms 해상도에서 **0.000** 이다
(9월 7/8, 7월 8/8). depth 가 다른 쌍만 어긋나고 depth 가 같은 쌍은 어긋나지 않는다 —
이것이 §3 법칙의 가장 날카로운 검증이다.

### 2.3 씬 카메라가 진짜 팔을 본다

`cam1.mp4` 를 160×90 그레이스케일로 디코딩해 프레임간 모션 에너지를 만들고,
`command` TCP 속도 / `fk(ur_joint_states)` TCP 속도와 상호상관했다
(`scratchpad/vid.py`):

| take | cam1 모션 → `command` | cam1 모션 → `ur_joint_states` |
| --- | --- | --- |
| take_01 | **−0.055 s** (r 0.63) | **+0.865 s** (r 0.59) |
| take_30 | **+0.010 s** (r 0.61) | **+0.935 s** (r 0.53) |
| take_54 | (축퇴: 스캔 경계, r 0.42 — 반복 동작 긴 take. 무시) | (동일) |

**카메라는 command 가 말한 그 순간에 팔이 움직이는 것을 봤다.** 그리고 그 움직임을 담은
`ur_joint_states` 행은 0.9 초 뒤에 기록됐다. 팔이 0.9 초 늦었다면 영상도 0.9 초 늦게
움직였어야 한다. 그렇지 않았다.

### 2.4 카메라 자체는 22 ms 밖에 안 낡았다 — 절대 기준점

9월 세션은 depth 를 기록하면서 **소스 헤더 스탬프** 를 같이 저장한다
(`depth.h5:<cam>/stamp_s`, 커밋 `c694d2c`). take 폴더명이 곧 세션 t0 (초 단위 절사) 이므로
54 take 에 대해 `name − (stamp_s − t_rel_s)` 를 계산하면 값이 `age − f`, `f ~ U[0,1)` 이 된다.
따라서 **54 개 중 최댓값이 절대 나이의 추정치**다 (`scratchpad/age.py`):

```
cam1: min −0.958  max +0.022      cam2: min −0.956  max +0.022
take 내 나이 표류: 중앙값 −0.008 s, 최대 0.010 s (즉 큐가 자라지 않는다)
```

→ **깊이(그리고 같은 30 Hz·depth 10 인 컬러) 프레임은 도착 시점에 약 0.02 초밖에 안 낡았다.**
이미지는 신선하다. 낡은 것은 로봇 상태 테이블뿐이다.

---

## 3. 메커니즘 — 어디서 정확히 생기는가

### 3.1 이 take 들을 쓴 것은 헤드리스 노드가 아니라 GUI 레코더다

`take_NN_<stamp>/` 라는 폴더명은 `gello_recorder/gello_gui_node.py:213`
(`GelloRecorderGuiNode.start_recording`) 만 만든다. 헤드리스
`gello_ur_recorder_node.py` 는 `session_<stamp>` 를 만든다.

> ⚠️ **적분자 전달 내용 정정 1.** "ur/cmd/tcp/wrench 테이블을 타이머 하나에서 캐시된 최신
> 메시지로 샘플링한다" 는 **사실이 아니다.** 여덟 개 native-rate 테이블은 전부
> **구독 콜백 안에서** 직접 쓰인다 (`gello_gui_node.py` `_on_cmd`/`_on_ur`/`_on_tcp`/
> `_on_wrench` → `RecordingSession.write_*`). 타이머가 채우는 것은 아홉 번째
> `synchronized` 테이블뿐이고, **그 테이블은 54 take 전부 비어 있다**(0 행). 행 수가
> 거의 같아지는 것은 공유 타이머 때문이 아니라 §3.3 의 라운드로빈 때문이다.

스탬프는 `RecordingSession.t()` = `time.time() - t0`, 즉 **콜백이 실행되는 순간**이다.
메시지 헤더 스탬프는 쓰지 않는다 (`recording_session.py` 모듈 docstring: "Every method
stamps its own row's `t_rel_s` via `t()`"). 그래서 **큐에서 기다린 시간이 그대로
타임스탬프 오차가 된다.**

### 3.2 모든 콜백이 단일 spin 스레드 하나에서 돈다

`task_gui_node.py` 의 THREADING CONTRACT 절이 명시한다: "Every ROS callback … runs on the
**single** background rclpy spin thread". 그 스레드가 매 프레임 하는 일:

* `_on_cam` ×2 @30 Hz: 1280×720 JPEG `cv2.imdecode` (GUI 프리뷰) **+** `write_cam1_frame(raw)`
  (비디오 라이터가 다시 디코드 후 MPEG-4 인코드)
* **(9월 신규, `c694d2c`)** `_on_cam_depth` ×2 @30 Hz: 88.6 kB PNG 페이로드를 HDF5 로
  (**≈ 5.3 MB/s**), `_on_depth_info` ×2 @30 Hz
* 로봇/그리퍼 토픽 8 개

rclpy `SingleThreadedExecutor` 는 한 라운드에 **구독당 메시지 한 개**를 처리한다. 따라서
구독별 서비스 주파수 = 라운드 주파수이고, 이 세션에서 그 값은 **약 60–69 Hz** 다.

**증거: 굶은 스트림 네 개의 기록 주파수가 소수점 5 자리까지 같다.**
take_01 → `command` 59.79218 / `ur_joint_states` 59.79089 / `tcp_pose` 59.79282 /
`wrench` 59.79153 Hz. take_02 → 69.084 / 69.086 / 69.172 / 69.166 Hz. 서로 다른 발행
주파수를 가진 네 토픽이 같은 값으로 수렴하는 것은 **라운드로빈 서비스 한도에 전부 걸렸다**는
뜻이다. 반대로 굶지 않은 스트림은 자기 native rate 를 그대로 유지한다:
`gello_joint_states` 29.99–30.06 Hz, `cam1_frames` 29.96–30.01 Hz, `gripper` 37.8–40.1 Hz.

### 3.3 낡은 정도 = `QoS depth ÷ 발행 주파수`

발행 주파수 > 서비스 주파수인 토픽은 KEEP_LAST(depth) 히스토리가 항상 가득 차고, 리더가
꺼내는 것은 **가장 오래된** 샘플이므로 그 나이는 `depth / 발행률` 로 고정된다. 또한 나이가
상수이므로 신호 형상은 그대로다 — §1 의 0.5 mm 잔차가 여기서 나온다.

| 토픽 | QoS depth (`gello_gui_node.py`) | 발행 | 서비스 | 예상 나이 | 측정(상대) |
| --- | --- | --- | --- | --- | --- |
| `/joint_states` | **100** | ~100 Hz (CM) | 굶음 | **1.00 s** | 기준 |
| `/tcp_pose_broadcaster/pose` | **50** | ~100 Hz (CM) | 굶음 | **0.50 s** | js 대비 **−0.495** ✅ |
| `/force_torque_sensor_broadcaster/wrench` | **50** | ~100 Hz (CM) | 굶음 | **0.50 s** | tcp 대비 **0.000** ✅ |
| `/forward_position_controller/commands` | 50 | 250 Hz (`publish_rate_hz`) | 굶음 | 0.20 s | js 대비 **−0.910** → 실측 나이 ≈ **0.08 s** ⚠️ |
| `/gello/joint_states` | 50 | 30 Hz | 따라감 | ~0 | 0 |
| `/robotiq_gripper/*`, 리더 트리거 | 20 | 38–40 Hz | 따라감 | ~0 | 0 |
| 컬러 `…/compressed` | 10 | 30 Hz | 따라감 | ~0 | **0.022 s 실측** |
| depth `…/compressedDepth` | 10 | 30 Hz | 따라감 | ~0 | **0.022 s 실측** |

두 개의 독립 관측 `js − tcp = 0.495` 와 `tcp − wrench = 0.000` 을 함께 풀면
`50 / R = 0.495` → **브로드캐스터 발행률 R ≈ 101 Hz**, 따라서 `ur_joint_states` 나이 ≈ **0.99 s**,
`tcp_pose`/`wrench` 나이 ≈ **0.50 s** 다. R ≈ 100 Hz 는 7월 세션이 독립적으로 확인해 준다 —
그때는 아무 스트림도 굶지 않았고 이 세 토픽의 기록 주파수가 **98.9 Hz** 였다.
(참고: ros2_control 의 기본 `update_rate` 는 100 Hz 이고, `ur_robot_driver` 에는
`ur7e_update_rate.yaml` 이 **없다** — `ur3e/ur5e/…` 만 있다. 즉 이 스택의 CM 은 500 Hz 가
아니라 기본 100 Hz 로 돌고 있다. `ur7e_gello.yaml` 의 "500 Hz e-series servo cycle" 주석은
드라이버 내부 사이클 이야기이지 ROS 측 발행률이 아니다.)

⚠️ **표에서 한 칸만 딱 안 맞는다**: `command` 는 명목 250 Hz × depth 50 → 0.20 초를 예상하지만,
위 연립을 만족하려면 **0.08 초**여야 한다 (= 실효 500–600 Hz 상당). `_publish` 는
`_on_timer` 당 정확히 한 번 호출되므로(호출부 3 곳 전부 동일 틱의 배타적 분기)
발행률이 250 을 넘을 구조적 이유는 찾지 못했다. **그래서 `command` 의 절대 나이는
`0.02 s ≤ A_cmd ≤ 0.20 s` 로만 묶어 둔다** (하한은 §2.3 의 카메라 정렬, 상한은 depth/250 Hz).
이 잔여 불확실성이 데이터셋 전체의 "이미지 ↔ 상태" 정렬 정확도의 바닥이다.

### 3.4 왜 9월에만 터졌나

7월 세션에는 **depth 기록이 없었다.** `c694d2c` 가 같은 spin 스레드에 구독 4 개(+ 콜백
약 120 회/초)와 약 5.3 MB/s 의 HDF5 쓰기를 얹었고, 그 결과 라운드 주파수가
**98–100 Hz → 60–69 Hz** 로 내려가면서 **발행률 100 Hz 이상인 토픽만** 큐에 갇혔다.
100 Hz 짜리 `/joint_states`·`tcp_pose`·`wrench` 와 250 Hz 짜리 `commands` 가 정확히 그것들이고,
30–40 Hz 짜리 카메라·GELLO·그리퍼는 그대로 신선하다. **역치가 두 세션 사이에 정확히
문제의 토픽들을 가로질렀다.**

> ⚠️ **적분자 전달 내용 정정 2.** "7월 take 는 `tcp_pose` 가 1.0 초 낡았다" 는 **관측되지 않는다.**
> 7월 8 개 take 전부에서 `tcp_pose → ur_joint_states` 는 **정확히 0.000 초**이고
> `tcp_pose → wrench` 도 0.000 초다. 7월에는 아무것도 굶지 않았다.
> 그렇게 보인 이유는 거의 확실히 **툴 오프셋 착시**다: 7월 세션은 펜던트 TCP 가 설정되어
> 있지 않아 `tcp_pose` 가 플랜지를 보고하므로 `fk(q) ⊕ 0.174` 와 **모든 τ 에서 일정하게
> 174 mm** 떨어져 있다. 위치 거리로 τ 를 찾으면 최솟값이 없어 스캔 경계로 굴러간다
> (내 1 차 스캔도 정확히 그렇게 −0.3 이라는 쓰레기값을 냈다). 속도 상호상관으로 바꾸면
> 0.000 이 나온다. carrot 세션은 반대로 펜던트 TCP 가 0.174 로 잡혀 있다.

---

## 4. (a) 팔은 무엇에 묶여 있었나

**UR7e 자신의 한계에는 근처도 가지 않았다.** 54 take 전수, 100 Hz 균일 그리드
(`scratchpad/limits.py`, n = 61,762):

| 지표 | 값 |
| --- | --- |
| `|q̇|` 관절별 p99 | 0.34 / 0.32 / 0.37 / 0.53 / 0.32 / 0.50 rad/s |
| `|q̇|` 관절별 최대 | 0.58 / 0.62 / 1.00 / 0.93 / 0.82 / 0.94 rad/s |
| 드라이버 관절 속도 한계 | 3.14 rad/s (180°/s) — **한 번도 근처에 못 간다** |
| 브리지 슬루 캡 `max_step_rad × publish_rate_hz` | 0.0025 × 250 = **0.625 rad/s** |
| 캡의 0.9 배를 넘은 샘플 | **1.66 %** (캡 초과 0.40 %, 보간 잡음 포함) |
| 명령 TCP 속도 | p50 **0.073**, p99 **0.185**, 최대 0.222 m/s |
| EEF 거버너 `v_max` | **0.16 m/s** |
| `v_max` 의 0.95 배 초과 샘플 | **15.7 %** (초과 10.2 %) |

명령 TCP 속도 히스토그램은 0.01 m/s 부터 0.13 m/s 까지 단조 감소하다가 **0.15–0.17 m/s 에서
다시 솟는다**(6.6 % + 5.4 %). 이 봉우리가 거버너다.

> **(a) 답: 팔을 묶은 것은 로봇도, 드라이버도, 추종 능력도 아니라 브리지의 EEF 카테시안
> 속도 거버너 `v_max = 0.16 m/s` 이고, 전체 프레임의 약 10–16 % 가 그 천장에 붙어 있다.**
> 관절 슬루 캡 0.625 rad/s 는 2 차 제약으로 1.7 % 에서만 근접한다. 그리고 **어느 쪽도
> 0.9 초 지연을 만들지 않는다** — 거버너는 속도를 깎지 시간을 밀지 않는다.

덧붙여 거버너는 **norm 비례**다: `eef_delta.py` 의 `rate_scale()` 이 스칼라 `sc` 를 계산해
`xi_raw` (6-D twist) **전체**에 곱한다. 축별 clip 이 아니므로 **포화 프레임에서도 방향은
보존된다** (CLAUDE.md 의 "축별 np.clip 금지" 규칙과 같은 방향). 단 병진이 걸리면 회전도
같은 비율로 줄어든다 (병진·회전이 한 스칼라로 묶여 있다).

---

## 5. (b) 배포 경로(`policy_leader_node` → joint 모드)에서도 같은 지연이 나오나

**0.9 초는 나오지 않는다.** 그것은 레코더의 큐에서 생겼고 **배포 루프에는 레코더가 없다.**
`policy_leader_node` 는 `/joint_states` 를 자기 콜백에서 직접 읽고, 그 노드의 부하는
GUI 레코더와 비교가 안 된다(카메라 인코딩·depth 기록이 없다). 다만 **원인이 큐 깊이 ÷
발행률이라는 사실이 곧 재발 조건**이다 — `/joint_states` 를 depth 100 으로 구독하는
어떤 노드든, 콜백을 100 Hz 미만으로 처리하기 시작하면 **같은 1.0 초짜리 무증상 지연**을
얻는다. 배포 노드는 (i) depth 를 1–5 로 줄이고 (ii) 콜백을 가볍게 유지해야 한다.

**대신 배포 경로에는 실재하는 지연이 있고, 그 값은 이미 측정돼 있다.**

* `gello_joint_states → command` = **+0.25 초**, 그것도 **7월(joint 모드) 0.253 s 와
  9월(EEF 모드) 0.260 s 가 동일**하다. 7월에는 아무 스트림도 굶지 않았으므로 이 0.25 초는
  아티팩트가 아니라 **브리지의 진짜 리더→명령 지연**이다.
* 성분: joint 모드에서 `command_pipeline` 은 `filter_stage_joint`(One-Euro,
  `one_euro_min_cutoff = 1.0 Hz`, `beta = 2.0`) → `clamp_stage`(슬루 0.0025 rad/틱) 를 탄다.
  저속 구간에서 One-Euro 의 1 Hz 컷오프는 1 차 지연 상수 **1/(2π·1) ≈ 0.16 초**이고,
  나머지는 슬루와 30 Hz 리더 샘플링이다. 0.25 초와 정확히 맞는다.
* ⚠️ **`deadband_rad = 0.004` 는 배포에서 작동하지 않는다.** `filter_stage_joint` 는
  데드밴드를 **EMA 경로에서만** 쓴다 (`euro is None` 분기). 설정은 `filter_type: "one_euro"`
  이므로 데드밴드는 무시된다. (반대로 `filter_type` 을 `ema` 로 바꾸면 정책의 미세 동작이
  0.004 rad 계단으로 잘린다 — 30 fps 프레임의 **35 %** 는 최대 관절 스텝이 0.004 rad 미만이다.
  바꾸지 말 것.)

> **(b) 답: 0.9 초는 재현되지 않는다. 재현되는 것은 ~0.25 초의 리더→명령 지연과
> 0.625 rad/s 슬루 캡이며, 둘 다 joint 모드에서 이미 측정된 값이다.**

DESIGN §3 의 lead clamp 는 바로 이 0.25 초를 흡수하도록 되어 있고, 여유가 그리 크지 않다:
정상 상태에서 `|p_target − p_live| ≈ v · 0.25 s` 이므로 데이터셋 p50 속도(0.073 m/s)에서
**18 mm**, 거버너 천장(0.16 m/s)에서 **40 mm** 다. 📌 **배포되는 값은 DESIGN §3 의 기본값
0.06 이 아니라 A2 가 실측으로 정한 `eef_max_lead_m = 0.0700`**(30-frame `|dp|` 중앙값
64.10 mm 올림, `config/carrot_eef_{act,fm}.yaml`) 이므로 여유는 **1.75 배**다 — 0.06 기준의
1.5 배가 아니다. 어느 쪽이든 넉넉하지 않다: 이 값을 더 줄이면 빠른 이송 구간에서 lead clamp
가 상시 물려 정책이 스스로 감속하는 것처럼 보이게 된다. (A4/A6 참고용. 수정 권고는 아니다.)

---

## 6. (c) DESIGN §2 (action = 달성 동작, 배포 시 chunk 누적) 에 대한 함의

### 6.1 "command 를 쓰지 않는다" 는 결정은 옳았고, 이유는 문서보다 강하다

DESIGN §1 은 "그래서 `command` 는 next pose 가 아니다" 라고만 적었다. 실제 이유는
**`command` 도 `ur_joint_states` 도 절대적으로 옳지 않다는 것** — 둘 다 낡았고 낡은 양이
다를 뿐이다. 달성 동작(`q_t → q_{t+1}`)으로 정의하면 **두 프레임이 같은 테이블에서 오므로
상수 오프셋이 차분에서 소거된다.** 즉 §2 의 정의는 이 아티팩트에 **구조적으로 면역**이다.
차분이 아니라 절대 시각이 필요한 곳 — **이미지와 상태를 짝짓는 순간** — 에서만 문제가 남고,
그것이 §6.2 다.

🛑 **만약 `command` 로 액션을 만들었다면 얼마나 틀렸을까 — 실측했다.** 10 take 에서
`delta_action(q_t, cmd_t)` (최근접 스탬프 정렬) 의 `‖dp‖` 는 **중앙값 55.5 mm, p99 142.3 mm**
다. 달성 동작의 실제 프레임당 값은 **2.2 mm** (DESIGN §1) 이므로 **25 배 과대**이고,
게다가 그 55 mm 는 동작이 아니라 **0.9 초짜리 고정 오프셋**이라 정지 구간에서도 0 이 되지
않는다. 정책은 "항상 55 mm 앞으로" 를 배웠을 것이다.

⚠️ **그런데 `gello_policy/eef_space.py` 모듈 docstring 은 아직 `command` 기준으로 액션을
정의하고 있다** (26–30 행: `dp = p_tcp(command_t) - p_tcp(q_t)`, 그리고
`delta_action(q, q_cmd, grip_cmd)` 의 인자 이름과 한 줄 요약 "TCP(command) relative to
TCP(q)"). 산술 자체는 두 관절 벡터를 받는 범용 함수라 **코드는 옳다** — 틀린 것은 문서뿐이다.
그러나 그 파일은 "single source of truth" 로 읽히므로 **DESIGN §2 와 정면으로 모순된
설명을 그대로 두면 안 된다.** 소유자(적분자)에게 수정 제안을 보고했다(A7 리포트 1 번 항목).

### 6.2 A1 의 `tau_q ≈ 0.90` 은 옳다. 그러나 테이블마다 값이 다르다

마스터 클럭은 **카메라**이고 카메라는 신선하다(0.022 초). 각 테이블을 카메라 시간축으로
되돌리려면 `t_rel_s` 에서 아래를 **빼야** 한다:

| 테이블 | 빼야 할 값 | 근거 |
| --- | --- | --- |
| `cam1_frames` / `cam2_frames` / `depth.h5` | 0 (기준) | §2.4 실측 0.022 s |
| `gello_joint_states` | ~0 | native rate 유지 → 안 굶음 |
| `gripper` (`grip_cmd`/`grip_pos`/`gello_grip`) | ~0 | native rate 유지 → 안 굶음 |
| `command` | 0.02–0.20 s | §3.3 ⚠️ |
| **`ur_joint_states`** | **0.90–0.95 s** | cam1 모션 직접 상관 +0.865 / +0.935 |
| `tcp_pose`, `wrench` | **0.40–0.50 s** | js 대비 −0.495 |

**A1 이 `ur_joint_states` 만 τ_q ≈ 0.90 만큼 당기는 것은 정확히 옳다.** 다만 두 가지:

1. **τ_q 는 `command` 가 아니라 카메라 기준으로 재는 것이 원칙이다.** `command` 자신이
   0.02–0.20 초 낡았기 때문이다. 다행히 두 방법의 차이는 ≤ 0.06 초(= 2 프레임 이하)이고,
   내가 cam1 모션으로 직접 잰 값(+0.865 / +0.935, 중앙값 ≈ 0.90)이 `command` 기준 0.910 과
   일치하므로 **0.90 은 그대로 쓰면 된다.**
2. 🛑 **그리퍼 채널을 같이 밀면 안 된다.** `grip_cmd` 는 굶지 않은 테이블이라 이미
   카메라와 같은 시간축에 있다. 여기에 0.9 초를 더 적용하면 **없던 0.9 초 오정렬을
   새로 만든다.** (`grip_cmd → grip_pos` 의 +0.37 초는 Robotiq 의 실제 개폐 시간이고
   7월에도 +0.34 초였다 — 건드릴 대상이 아니다.)
3. `tcp_pose` 를 검증·비교에 쓸 때는 0.45 초를 따로 빼야 한다. `ur_joint_states` 와 같은
   값을 쓰면 0.5 초 틀린다.

### 6.3 액션은 실제로 속도 포화돼 있다 — 그리고 그것은 대체로 이득이다

프레임의 **10–16 %** 가 `v_max = 0.16 m/s` 천장에 붙어 있다(§4). 30 fps 로 환산하면
**5.3 mm/frame** 이고, DESIGN §1 의 `|Δp| p99 = 6.6 mm, max 9.6 mm` 와 정확히 같은 이야기다
(p99 6.6 mm ≈ 0.20 m/s 는 보간·One-Euro 오버슈트 몫). 즉 **액션 분포의 위쪽 꼬리는
사람의 의도가 아니라 거버너의 모양**이다.

이게 문제인가 — 세 갈래로 나눠 답한다.

* **방향은 안전하다.** 거버너가 norm 비례라 포화 프레임에서도 단위 방향이 보존된다(§4).
  축별 clip 이었다면 데이터셋의 대각 이동이 축 정렬로 꺾여 있었을 것이고 그건 치명적이었다.
  다행히 아니다.
* **배포 일관성 면에서는 오히려 이득이다.** 배포 때도 **같은 브리지·같은 거버너**가
  정책과 팔 사이에 그대로 있다. 사람의 진짜 의도(클램프 전 리더 속도)로 학습시켰다면
  정책은 **실행 불가능한 속도를 요구**하고 거버너가 매번 깎았을 것이다. 지금은 정책이
  "이 스택이 실제로 낼 수 있는 속도" 를 배운다.
* 🟠 **진짜 위험은 포화가 아니라 필터가 두 번 걸린다는 것이다.** 데이터셋 액션은 이미
  One-Euro + 거버너를 **통과한 뒤**의 달성 동작이다. 배포 때 그 액션을 다시
  `policy_leader_node → 브리지 joint 모드` 로 보내면 **또 한 번** One-Euro(0.16 s) + 슬루를
  통과한다. 즉 **평활이 이중으로 걸린다.** DESIGN §3 의 chunk 누적(타깃에 Δ 를 누적하고
  live 에 누적하지 않음)이 이것을 부분적으로 상쇄한다 — 타깃이 앞서 나가므로 브리지가
  깎아도 위치 오차가 누적되지 않고, lead clamp 가 그 앞섬을 **0.07 m**(배포 yaml 의
  `eef_max_lead_m`; DESIGN §3 기본값 0.06 이 아니다) 로 묶는다. **설계가 이미 옳은
  방향이다.** 다만 §5 에서 봤듯 여유가 1.75 배뿐이다.
  - 후속 선택지(구현 안 함, A6 소관): 정책 배포용 파라미터에서 `one_euro_min_cutoff` 를
    올리는 것. One-Euro 와 데드밴드는 **사람 손떨림과 Dynamixel 엔코더 잡음**을 죽이려고
    있는 것인데, 30 Hz 정책 출력에는 둘 다 없다. 다만 이것은 **실기 검증이 필요한 변경**이고
    지금 브랜치의 범위 밖이다.
* **학습 자체에는 영향이 적다.** 상태와 액션이 **같은(지연된) 궤적**에서 나오므로 정책이
  배우는 것은 자기 일관적인 state→Δstate 사상이다. 깨지는 것은 오직 **이미지 ↔ 상태**
  정렬이고, 그게 §6.2 의 τ_q 다.

---

## 7. 권고 — 레코더 수정안 (지금 구현하지 않는다)

우선순위 순. 1 번만으로도 이번 결함은 사라진다.

1. **구독 depth 를 낮춘다.** `/joint_states` 100 → **1~5**, `tcp_pose`/`wrench`/`commands`
   50 → **1~5**. 이 레코더는 모든 테이블을 "도착한 것을 그대로" 쓰므로 **깊은 큐는 아무
   이득이 없고 오직 나이만 만든다.** depth 1 이면 최악의 나이가 10 ms 로 떨어진다.
   대신 굶은 만큼 **행이 빠진다** — 지금도 빠지고 있는데(67 Hz 기록 vs 100 Hz 발행) 지금은
   그 사실이 낡음으로 위장돼 있을 뿐이다. **빠지는 편이 낡는 편보다 훨씬 낫다.**
2. **헤더 스탬프를 같이 저장한다.** `JointState`/`PoseStamped`/`WrenchStamped` 는 전부
   `header.stamp` 를 갖고 있다. depth 경로가 이미 `stamp_s` 로 정확히 이것을 하고 있고
   (`stamp_to_seconds`, `c694d2c`), 그 덕분에 §2.4 의 절대 측정이 가능했다. 각 테이블에
   `stamp_s` 열을 하나 추가하면 **이런 결함이 앞으로는 데이터 자체에서 즉시 보인다.**
   (`Float64MultiArray` 인 `commands` 만 헤더가 없어 불가 — 그래서 1 번이 먼저다.)
3. **카메라/depth I/O 를 spin 스레드에서 뺀다.** `MultiThreadedExecutor` + 콜백 그룹, 또는
   프레임을 큐에 넣고 쓰기 스레드가 소비. 다만 `RecordingSession` 은 스레드 안전하지
   않으므로(모듈 docstring: "Deliberately ROS-free / Qt-free / **thread-free**") 락 설계가
   필요하다 — 1·2 번보다 훨씬 큰 작업이다.
4. **회귀 감시**: 세션 종료 시 native-rate 테이블들의 기록 주파수를 비교해 **서로 같아지면
   경고**한다. §3.2 에서 봤듯 "굶었다" 의 서명이 정확히 그것이고, 한 줄로 검출된다.

⚠️ 이번 54 take 는 **다시 녹화할 필요가 없다.** §6.2 의 상수 시프트로 완전히 복구된다
(신호 형상이 보존돼 있다는 것을 §1 의 0.5 mm 잔차가 보증한다).

---

## 8. 재현 명령

전부 시스템 `python3` (numpy/h5py/opencv, ROS 오버레이 불필요) + `ur_gello_bringup` 을
PYTHONPATH 에 올려서 돌렸다. 스크립트는 세션 스크래치패드에 있고 리포에 커밋하지 않았다.

```bash
WT=/home/laptop3/gello_software_humble
export PYTHONPATH=$WT/ros2_ur_ws/src/ur_gello_bringup

# 1) TCP 거리 기반 τ 스캔 + 스트림별 기록 주파수 (carrot 11 take + cube 10 take)
python3 scratchpad/lag1.py

# 2) 속도 상호상관 (오프셋 불변) — §1 표 전체
python3 scratchpad/xcorr.py

# 3) depth 헤더 스탬프로 카메라 절대 나이 (54 take)
python3 scratchpad/age.py

# 4) 속도 포화 / 슬루·거버너 분석 (54 take, 100 Hz 그리드)
python3 scratchpad/limits.py

# 5) cam1.mp4 모션 에너지 ↔ command / ur_joint_states 상호상관
python3 scratchpad/vid.py
```

핵심 코드 위치:

| 무엇 | 어디 |
| --- | --- |
| take 폴더명·구독·QoS depth·콜백 스탬프 | `ros2_ur_ws/src/gello_recorder/gello_recorder/gello_gui_node.py` (구독 209–225, `_on_cam` 623, depth 콜백) |
| 행 스탬프 = 콜백 실행 시각 | `.../gello_recorder/recording_session.py` (`t()`, `write_cmd`/`write_ur`/`write_tcp`) |
| 단일 spin 스레드 계약 | `.../gello_recorder/task_gui_node.py` THREADING CONTRACT 절 |
| depth 기록 추가(= 굶김의 원인) | 커밋 `c694d2c` |
| 명령 발행 (틱당 1 회) | `ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/gello_ur_bridge_node.py` `_publish` (2816), 호출부 1153/1372/1403 |
| One-Euro / 데드밴드 / 슬루 3 단 | `.../ur_gello_bringup/bridge_stages.py` `filter_stage_joint`, `clamp_stage`, `command_pipeline` |
| 카테시안 거버너 (norm 비례) | `.../ur_gello_bringup/eef_delta.py` `rate_scale()` (약 501–547) |
| 슬루·필터 상수 | `config/ur7e_gello.yaml` (`max_step_rad` 0.0025, `publish_rate_hz` 250, `one_euro_min_cutoff` 1.0, `soft_start_s` 0.7) |
| `v_max` / `w_max` | `config/ur7e_gello_eef.yaml` (0.16 m/s, 1.0 rad/s) |
