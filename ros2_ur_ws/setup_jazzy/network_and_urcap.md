# 이 PC(`junhyeong`)에서 UR7e 네트워크 재페어링 — 측정값 + URCap 재확인 절차

이 문서는 [`docs/ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md`](../../../docs/ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md)의
네트워크/URCap 델타를 분리한 것이다. 로봇 쪽 handshake·캘리브레이션 절차는
[`GELLO_UR7E_REAL_ROBOT.md`](../../../docs/ros2/GELLO_UR7E_REAL_ROBOT.md) §2–§4가 정본이고 여기서
반복하지 않는다. 이 문서가 다루는 것은 딱 하나 — **제어 PC가 laptop3에서 이 PC로 바뀌면서
네트워크 계층에서 새로 확인/설정해야 하는 것**이다.

## 0. 이 PC는 이더넷 포트가 하나다 — 그것이 의미하는 것

이 PC는 물리 이더넷 NIC이 `enp12s0` **하나뿐**이다(무선 `wlp13s0`은 별개로 항상 살아 있고,
로봇과는 무관 — 랩 와이파이에 연결돼 있다). `nmcli` **프로파일** 하나에 랩 주소와
`192.168.10.x` 주소를 **동시에**(secondary address로) 넣어 둘 수는 있지만, **물리 링크는
한 번에 한 곳에만 꽂혀 있을 수 있다.** 즉:

- 케이블이 랩 스위치에 꽂혀 있으면 → 랩 주소(166.104.146.x)만 살아 있고 `192.168.10.x`는 설정돼
  있어도 아무 데도 도달하지 못한다(로봇이 그 세그먼트에 없으므로).
- 케이블을 로봇 스위치/직결로 옮기면 → **랩 네트워크가 이더넷에서 통째로 사라진다.** SSH로
  이 PC에 붙어 있었다면 **그 세션이 끊긴다** — 와이파이(`wlp13s0`)가 살아 있으므로 그쪽으로
  재접속해야 한다. 랩 이더넷과 로봇 이더넷을 동시에 쓸 방법은 이 PC에 **없다**(포트가 하나뿐).

## 1. 측정된 현재 상태 (이 세션에서 read-only로 확인)

```
$ ip -br addr
enp12s0          DOWN
wlp13s0          UP             192.168.0.44/24 ...
```

```
$ cat /sys/class/net/enp12s0/carrier
0
```

```
$ nmcli con show "Wired connection 1" | grep -E "ipv4.addresses|ipv4.gateway|ipv4.method"
ipv4.method:      manual
ipv4.addresses:   166.104.146.29/24
ipv4.gateway:     166.104.146.1
```

즉 `enp12s0`는 현재 **carrier=0(케이블 미연결 또는 반대쪽 다운) → DOWN**이고, `nmcli` 프로파일
`Wired connection 1`은 **랩 고정 주소 166.104.146.29/24**(게이트웨이 166.104.146.1)만 갖고
있다. `192.168.10.x`는 **아직 이 프로파일에 없다.**

이 값은 이 세션의 한 시점 스냅샷이다 — **케이블이 랩 스위치와 로봇 사이를 오가는 중**이므로
다음에 확인할 때 carrier=1(랩에 꽂힘)일 수도 있다. 작업 시작 전 항상 `ip -br addr`로
다시 확인할 것.

## 2. `192.168.10.x`를 secondary로 추가하기 (sudo, 사용자가 직접 실행)

아래 명령은 **랩 주소를 지우지 않고** `192.168.10.x/24`를 같은 프로파일에 추가한다. `nmcli`는
`ipv4.addresses`에 공백으로 구분한 여러 CIDR을 받으므로, 기존 값을 유지한 채 추가하면 된다.
**정확한 로컬 IP는 로봇 펜던트가 아니라 이 PC 쪽에서 임의로 골라도 되지만, laptop3가 실제로
쓰던 값을 재사용하는 편이 §3의 URCap 문제를 피하는 유일한 길이다 — 그 값이 어디에도 문서화돼
있지 않으므로 §3을 먼저 읽을 것.**

```bash
# 🛑 sudo 필요 — 사용자가 직접 실행
sudo nmcli con modify "Wired connection 1" \
  +ipv4.addresses "192.168.10.<N>/24"
sudo nmcli con up "Wired connection 1"
```

`<N>`은 로봇(`192.168.10.11`)과 겹치지 않는 값 — laptop3가 실제 썼던 주소를 §3에서 읽어
그대로 쓰는 것을 권장한다(모르면 관례적으로 `.10` 또는 `.12`, 단 §3을 보기 전엔 확정하지 말 것).

케이블을 로봇 쪽으로 연결한 뒤 검증:

```bash
ip addr show enp12s0              # 166.104.146.29/24 와 192.168.10.<N>/24 둘 다 보여야 함
cat /sys/class/net/enp12s0/carrier   # 1이어야 함(링크 업)
ping -c 3 192.168.10.11           # UR7e 컨트롤러
nc -zv 192.168.10.11 29999        # Dashboard 서버
nc -zv 192.168.10.11 30004        # RTDE
nc -zv 192.168.10.11 30001        # 로봇이 듣는 Primary interface
```

`50001`과 `50002`는 로봇 주소로 probe하지 않는다. 둘 다 드라이버 호스트가 여는 PC 쪽
endpoint이므로, 드라이버를 실행한 뒤 PC에서 다음처럼 확인한다:

```bash
ss -ltn '( sport = :50001 or sport = :50002 )'
```

**되돌리기(랩으로 복귀):**

```bash
sudo nmcli con modify "Wired connection 1" \
  -ipv4.addresses "192.168.10.<N>/24"
sudo nmcli con up "Wired connection 1"
```

## 3. URCap 재확인 — 진짜 블로커는 이것이다

`ur_robot_driver`는 **reverse connection** 모델을 쓴다. 드라이버 호스트가 `reverse_port`
**50001**과 `script_sender_port` **50002**를 열고, 로봇 컨트롤러/External Control URCap이
각 endpoint로 접속한다. 반대로 로봇의 **Primary interface는 로봇 쪽 `30001`**이며, PC가
로봇 주소의 `30001`로 접속한다. [`ur_control.launch.py`](https://github.com/UniversalRobots/UniversalRobots_ROS2_Driver/blob/main/ur_robot_driver/launch/ur_control.launch.py)
와 [driver parameter docs](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/main/ur_robot_driver/doc/hardware_interface_parameters.rst)가
이 기본 포트와 방향을 정의한다. External Control 프로그램에는 **접속할 제어 PC의 IP와
`script_sender_port`(기본 50002)** 설정이 들어 있다.

**그리고 이 리포 어디에도 laptop3가 실제로 썼던 그 리터럴 `192.168.10.x` 주소가 기록돼
있지 않다** — grep으로 `GELLO_UR7E_REAL_ROBOT.md`·`GELLO_UR7E_SETUP_CLI.md`를 포함해 훑어봐도
`robot_ip`(로봇 컨트롤러 자신의 주소, `192.168.10.11`)만 나올 뿐, **제어 PC 쪽** 주소는 한 번도
문서화되지 않았다. 그 값은 **펜던트의 External Control 프로그램 설정 화면 안에만** 존재한다.

즉 선택지는 둘뿐이다:

- **(a) 이 PC에 그 주소를 그대로 부여한다** — §2에서 `<N>`을 laptop3가 쓰던 값으로 맞추면
  펜던트 프로그램을 건드릴 필요가 없다. 단 그 값을 **펜던트에서 직접 읽어야** 한다(아래).
- **(b) 펜던트의 External Control 프로그램을 이 PC의 새 주소로 고친다** — 프로그램 편집기에서
  IP 필드를 바꾸고, **PolyScope X라면 반드시 "Update program"을 눌러야** 변경이 실행 바이너리에
  반영된다(PolyScope 5는 이 단계가 없음). 안 누르면 Play해도 예전 IP로 접속을 시도한다.

**펜던트에서 현재 설정된 IP를 읽는 법:** Program 탭 → External Control 노드 선택 → Installation
파라미터(또는 노드 자체의 설정 패널)에 있는 "PC 주소"/"Host IP" 필드를 확인한다. PolyScope 5와
PolyScope X는 메뉴 위치가 다르므로 실제 펜던트 화면에서 확인할 것 — 이 문서로는 정확한 탭 이름을
확정할 수 없다(**unverified**, 펜던트 소프트웨어 버전에 의존).

### PolyScope와 URCap 형식

이 문서에서는 이 로봇의 PolyScope 세대, 정확한 버전, 설치된 URCap 형식을 확정하지 않는다.
PolyScope 5와 PolyScope X는 서로 다른 설치 형식을 사용할 수 있으므로, 실제 연결 전에 펜던트의
`Settings > About`와 URCap 관리 화면에서 직접 확인하고 해당 형식의 External Control 패키지를
사용한다. [`GELLO_UR7E_REAL_ROBOT.md`](../../../docs/ros2/GELLO_UR7E_REAL_ROBOT.md) §3의
handshake 절차는 그 확인 뒤에 따른다.

### ur_robot_driver 3.8.0(Jazzy)의 URCap/PolyScope 하한

현재 설치된 Jazzy 패키지는 `ur_robot_driver` **3.8.0**이다. 공식
[External Control URCap README](https://github.com/UniversalRobots/Universal_Robots_ExternalControl_URCap#prerequisites)는
이 URCap의 요구사항을 **URCap library 1.3.0 이상**, **PolyScope 3.7 이상(CB3) / 5.1
이상(e-Series)**으로 명시한다. 이 문서에서 3.8.0이 별도의 더 높은 하한을 요구한다고
추정하지 않는다. 연결 전에 펜던트에서 실제 PolyScope와 External Control 설치 형식/버전을
읽어 이 하한을 확인할 것. PolyScope 5와 PolyScope X의 정확한 형식·호환성은 해당 펜던트와
URCap 배포판에 따라 달라지므로 여기서 특정하지 않는다.

## 4. 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| `ping: connect: Network is unreachable` / no route to host | 케이블이 로봇 쪽에 없거나(§0), `192.168.10.x`가 아직 `enp12s0`에 없음(§2 미실행) | `ip -br addr`로 carrier·주소 확인 → §2 재실행 |
| `nc -zv 192.168.10.11 30004` → `Connection refused` | 로봇이 POWER_OFF거나 드라이버가 기대하는 RTDE 클라이언트 슬롯이 이미 다른 프로세스(예: laptop3 쪽 남은 드라이버)에 점유됨 | `echo -e 'robotmode\n' \| nc 192.168.10.11 29999`로 로봇 상태 확인, 다른 PC에서 드라이버 프로세스가 살아 있지 않은지 확인 |
| Play를 눌러도 "Program not running" / External Control이 연결을 못 함 | 펜던트 프로그램에 박힌 제어 PC IP가 **이 PC의 주소가 아님**(§3, 가장 흔한 원인) | 펜던트에서 IP 재확인 → §3의 (a) 또는 (b) 중 하나 실행 |
| 로봇이 **LOCAL** 모드라 Play 버튼이 안 눌리거나 `HEADLESS=true`가 거부됨 | headless(Method B)는 펜던트가 **REMOTE** 모드여야 함 | 펜던트 hamburger → Settings → System → Remote Control 활성화, 우측 상단 토글을 REMOTE로. `HEADLESS=true` 쓸 때만 필요 — Method A(수동 Play)는 LOCAL도 무방 |
| Protective stop 반복 | 네트워크 문제가 아님 — 워크스페이스/속도 제한 관련. `GELLO_UR7E_REAL_ROBOT.md` §2·트러블슈팅 참고 | 이 문서 범위 밖 |
| `nmcli con up` 이후에도 `enp12s0`가 `DOWN`으로 남음 | 물리 케이블 미연결(carrier=0) — nmcli는 IP를 프로파일에 넣을 뿐 링크를 만들지 않는다 | 케이블 연결 확인 후 `cat /sys/class/net/enp12s0/carrier`로 링크 재확인 |

## 요약 — unverified 항목 전체

- 펜던트에 현재 박혀 있는 제어 PC IP.
- 실제 펜던트의 PolyScope 버전과 External Control 설치 형식/버전.
