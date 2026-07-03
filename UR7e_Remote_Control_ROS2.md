# UR7e Remote Mode 로 ROS2 제어하기 (펜던트 Play 안 누르기)

> UR e-Series / PolyScope **5.24.1** 기준
> 목적: 매번 티칭 펜던트에서 Program 화면 → Play 를 손으로 누르던 걸 없애고,
> **ROS2 명령(또는 launch 인자)만으로** 로봇 제어를 시작한다.

---

## 0. 핵심 개념 — Local vs Remote

| | **Local 모드** | **Remote 모드** |
|---|---|---|
| 제어 주체 | 티칭 펜던트(사람) | 외부 네트워크 (Dashboard 서버 / RTDE) |
| 프로그램 Play | 펜던트에서 **손으로** 눌러야 함 | 네트워크 명령으로 **원격 실행** |
| 펜던트 조그 / Move / Freedrive | 가능 | **잠김 (불가)** |
| 펜던트 Load Program / Play·Stop 버튼 | 활성 | **회색(비활성)** — 정상 |

- **Remote → Local 전환은 반드시 펜던트에서** 해야 함 (원격 강제 전환 불가, 안전 설계).
- Remote 모드에서는 펜던트로 로봇을 **수동으로 못 움직임**. 손으로 옮기고 싶으면 잠깐 Local 로 토글.

---

## 1. 펜던트에서 Remote 모드 켜기 (최초 1회 세팅)

1. 우측 상단 **햄버거 메뉴(≡)** → **Settings → System → Remote Control** → **Enable**
2. 화면 **우측 상단 헤더에 Local/Remote 토글 아이콘**이 생김 → 눌러서 **Remote** 로 전환
3. (PolyScope 5.10+) **Settings → Security → Services** 에서 아래가 켜져 있는지 확인 (admin 비번 필요)
   - **Dashboard Server**
   - **Primary Client Interface**
   - **RTDE**

### Remote 모드가 제대로 켜졌는지 확인 (펜던트 화면)
- 우측 상단에 **`Remote`** 아이콘 표시
- `Load Program` 버튼 **회색** (정상)
- Control 의 **▶ Play / ■ Stop 버튼 회색** (정상 — 이제 ROS2가 대신 누름)
- 좌측 하단 안전 상태 **`Normal`(초록)**

> ⚠️ 우측 하단이 **`Simulation`** 이면 실제 팔이 안 움직임 → **`Real Robot`** 으로 토글할 것.
> ⚠️ 프로그램 이름에 **별표(`*`)** = 저장 안 됨 → 필요하면 Local 로 돌려 **Save**.

---

## 2. 실행 방법 A — External Control URCap 프로그램 + Dashboard 로 원격 Play

펜던트에 External Control URCap 이 들어간 프로그램(예: `ur_caps.urp`)을 두고, **Play 만 ROS2가 대신** 눌러주는 방식.

```bash
# 터미널 1 — 드라이버 launch
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e robot_ip:=192.168.10.11 launch_rviz:=true

# 터미널 2 — 프로그램 로드
ros2 service call /dashboard_client/load_program \
  ur_dashboard_msgs/srv/Load "{filename: ur_caps.urp}"

# 터미널 2 — Play (펜던트 Play 버튼 대신)
ros2 service call /dashboard_client/play std_srvs/srv/Trigger {}
```

- 성공 시: 펜던트 **Status `Stopped` → `Playing`**, 드라이버 로그에
  **`Robot ready to receive control commands.`**
- `filename` 은 실제 저장된 프로그램 이름에 맞출 것 (펜던트 Open 화면에서 확인).
- ⚠️ 서비스 타입 주의: `load_program` 은 `ur_dashboard_msgs/srv/Load`(filename 필드),
  `play`/`stop` 은 `std_srvs/srv/Trigger`(빈 `{}`).

---

## 3. 실행 방법 B — Headless 모드 (프로그램/Play 자체가 필요 없음) ⭐ 가장 깔끔

External Control URCap 프로그램도, Play 도 필요 없음. 드라이버가 URScript 를 직접 쏴서 **launch 하는 순간 자동 시작**.

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e robot_ip:=192.168.10.11 \
  headless_mode:=true launch_rviz:=true
```

- **전제조건**: 로봇이 **Remote 모드**여야 함 (e-Series/PolyScope5 필수).
- ⚠️ 방법 A(URCap 프로그램)와 **섞어 쓰지 말 것** — 둘 중 하나만.

---

## 4. MoveIt 붙이기 (기존 워크플로우 유지)

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5e launch_rviz:=true
```

---

## 5. 자주 쓰는 Dashboard / 상태 서비스

```bash
ros2 service call /dashboard_client/stop            std_srvs/srv/Trigger {}
ros2 service call /dashboard_client/power_on         std_srvs/srv/Trigger {}
ros2 service call /dashboard_client/brake_release     std_srvs/srv/Trigger {}
ros2 service call /dashboard_client/unlock_protective_stop std_srvs/srv/Trigger {}
ros2 service call /dashboard_client/get_robot_mode    ur_dashboard_msgs/srv/GetRobotMode {}
ros2 service call /dashboard_client/get_loaded_program ur_dashboard_msgs/srv/GetLoadedProgram {}
```

---

## 6. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `Connection to reverse interface dropped.` | 실행 중 스크립트가 끊김(펜던트 조작·새 스크립트 등). 아래로 재시작 |
| ↑ 재시작 (Remote/URCap) | `ros2 service call /dashboard_client/play std_srvs/srv/Trigger {}` |
| ↑ 재시작 (Headless) | `ros2 service call /io_and_status_controller/resend_robot_program std_srvs/srv/Trigger {}` |
| dashboard `play`/`power_on` 이 안 먹음 | 로봇이 Remote 모드가 아님 → 펜던트에서 Remote 로 토글 |
| 팔이 안 움직임 | 펜던트 우측 하단이 `Simulation` → `Real Robot` 로 전환 |
| `load_program` 실패 | 해당 `.urp` 가 펜던트에 없음(업로드 아님, 로드만 됨) / Installation 불일치 |
| 펜던트 Play 회색 | 정상 (Remote 모드). ROS2 로 play 할 것 |

---

## 7. 참고

- **하드웨어 참고**: 실제 로봇은 **UR7e**(신형 e-Series). 위 launch 는 `ur_type:=ur5e` 로 사용 중.
  최신 드라이버가 `ur_type:=ur7e` 를 지원하는지 확인 권장 — 운동학/페이로드 파라미터 정확도 관련.
- 공식 문서
  - Remote Control (PolyScope 5.24 매뉴얼)
  - Universal_Robots_ROS2_Driver — operation modes / startup / dashboard_client
  - Dashboard Server 명령 표 (Remote 모드에서만 되는 명령 목록)
