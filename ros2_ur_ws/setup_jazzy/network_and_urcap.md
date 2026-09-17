# UR7e 전용 이더넷 — Ubuntu 24.04 / ROS 2 Jazzy

이 PC에서 UR7e 제어망은 **USB Ethernet NIC `enx00e04c3600bd` 하나와 NetworkManager
프로파일 `Wired connection 2`만** 쓴다. 현재 확인된 계약은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| 제어 PC NIC / MAC | `enx00e04c3600bd` / `00:E0:4C:36:00:BD` |
| NetworkManager 프로파일 | `Wired connection 2` |
| 제어 PC 주소 | `192.168.10.100/24` |
| UR7e 컨트롤러 | `192.168.10.11` |
| gateway / default route | 없음 (`ipv4.never-default=yes`) |
| IPv6 | disabled |

이 분리는 Wi-Fi나 다른 유선 NIC의 기본 경로를 바꾸지 않는다. 로봇 연결이 끊긴 상황에서
무관한 NIC에 주소를 넣는 것을 막기 위해, 아래 스크립트는 로봇 도달성·현재 주소·활성
프로파일·프로파일 MAC으로 인터페이스를 찾고, 후보가 둘 이상이면 실패한다.

## 상태 확인 (기본, read-only)

```bash
cd /home/junhyeong/gello_software_jazzy/ros2_ur_ws
./setup_jazzy/setup_ur7e_ethernet.sh --check
```

성공하면 프로파일의 주소/route/IPv6 설정, 활성 프로파일, 라이브 주소, 로봇 route와
Dashboard (`29999`), Primary (`30001`), RTDE (`30004`) TCP 포트를 확인한다. 로봇이 꺼져
있으면 포트 검사는 실패할 수 있지만 NetworkManager 설정을 바꾸지는 않는다.

명시적으로 NIC를 지정해 진단할 수도 있다.

```bash
./setup_jazzy/setup_ur7e_ethernet.sh --check --device enx00e04c3600bd
```

## 영구 설정 적용

이 작업은 `Wired connection 2`만 수정한다. `--apply`는 **연결을 up/down 하지 않으므로**
실행 중인 로봇 NIC를 끊지 않는다. 프로파일이 이미 위 계약과 같으면 idempotent다.

```bash
cd /home/junhyeong/gello_software_jazzy/ros2_ur_ws
sudo ./setup_jazzy/setup_ur7e_ethernet.sh --apply --yes \
  --device enx00e04c3600bd
```

프로파일 변경이 현재 활성 연결에 반영되어야 하는 경우에는, **로봇이 정지한 별도 시점에만**
사용자가 직접 `sudo nmcli connection up "Wired connection 2"`를 실행한다. 스크립트는 그
재활성화를 절대 자동 실행하지 않는다.

주소나 로봇 IP를 바꿔야 하는 별도 셀 구성은 명시값으로만 한다.

```bash
sudo ./setup_jazzy/setup_ur7e_ethernet.sh --apply --yes \
  --device enx00e04c3600bd \
  --host-cidr 192.168.10.100/24 --robot-ip 192.168.10.11
```

## URCap / 드라이버 연결

네트워크가 정상이어도 External Control URCap에는 이 PC 주소 `192.168.10.100`과
`script_sender_port` `50002`가 설정돼 있어야 한다. ROS 드라이버는 PC에서 reverse port
`50001`과 script sender port `50002`를 열고, 로봇은 그 PC endpoint로 역접속한다.

`HEADLESS=true`로 실행할 때는 펜던트가 Remote Control 모드여야 한다. 그 외의 로봇
handshake·캘리브레이션 절차는
[`GELLO_UR7E_REAL_ROBOT.md`](../../../docs/ros2/GELLO_UR7E_REAL_ROBOT.md)를 따른다.

## 빠른 장애 구분

| 증상 | 확인 / 조치 |
| --- | --- |
| 스크립트가 NIC 후보가 모호하다고 종료 | `--device enx00e04c3600bd`를 명시한다. |
| `robot route` 또는 TCP 포트 실패 | 케이블/스위치, 로봇 전원, `192.168.10.11`을 확인한다. |
| URCap이 역접속하지 않음 | 펜던트 External Control의 Host IP가 `192.168.10.100`, 포트가 `50002`인지 확인한다. |
| `HEADLESS=true`가 거부됨 | 펜던트를 Remote Control 모드로 전환한다. |
