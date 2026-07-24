# Kanu 원격 Cube-in-Cup Classifier UI 실행 가이드

이 문서는 다음 구성을 실험실에서 그대로 재현하기 위한 실행 가이드다.

```text
laptop3의 cam1/cam2
  → SSH local forwarding
  → kanu GPU의 HIL-SERL reward classifier
  → laptop3의 ROS classifier status
  → laptop3의 실시간 UI
```

Classifier UI는 모니터 전용이다. 로봇 명령을 발행하거나 episode를 종료하지
않으며, policy server도 필요하지 않다. 로봇 상태 토픽이 이미 실행 중이면 UI의
상태 패널에 함께 표시되지만 classifier 확인에 필수는 아니다.

## 검증된 기준 상태

- 브랜치: `feat/remote-cube-classifier-viewer`
- 구현 검증 기준 커밋:
  `a2733ee6baa7399ae7149f43771b9a131f276133`
- checkpoint: `classifier_ckpt/cube_in_cup/checkpoint_150`
- checkpoint 크기: `87,217,768 bytes`
- checkpoint SHA-256:
  `e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997`
- 판정 threshold: `0.5`
- 원격 포트: kanu의 `127.0.0.1:5594`
- kanu 검증 당시 선택한 GPU: `3`

## 1. 저장소 배치

기존 `~/youngwoong_ws/gello_software`는 다른 작업 브랜치이므로 변경하거나
삭제하지 않는다. classifier 전용 clone은 다음 별도 경로를 사용한다.

### laptop3

현재 검증된 경로:

```bash
/home/laptop3/youngwoong_ws/gello_software_remote_classifier
```

새로 준비해야 하는 경우 laptop3에서:

```bash
cd ~/youngwoong_ws
git clone \
  --branch feat/remote-cube-classifier-viewer \
  --single-branch \
  https://github.com/Bigenlight/gello_software.git \
  gello_software_remote_classifier
cd gello_software_remote_classifier
git submodule update --init third_party/hil-serl
```

### kanu

현재 검증된 경로:

```bash
/home/junhyeong/workspace/youngwoong/gello_software_remote_classifier
```

새로 준비해야 하는 경우 kanu에서:

```bash
cd ~/workspace/youngwoong
git clone \
  --branch feat/remote-cube-classifier-viewer \
  --single-branch \
  https://github.com/Bigenlight/gello_software.git \
  gello_software_remote_classifier
cd gello_software_remote_classifier
git submodule update --init third_party/hil-serl
```

양쪽에서 다음을 확인한다.

```bash
git branch --show-current
sha256sum classifier_ckpt/cube_in_cup/checkpoint_150
```

브랜치가 `feat/remote-cube-classifier-viewer`이고 checkpoint SHA가
`e329986...a997`인지 확인한다. `a2733ee...`는 구현과 E2E 검증에 사용한 기준
커밋이다. 이후 문서 수정 등이 같은 브랜치에 추가되면 fresh clone의 HEAD는 이
값보다 새 커밋일 수 있으므로 HEAD가 정확히 일치할 필요는 없다.

## 2. 의존성과 빌드 확인

### laptop3

laptop3에서는 ROS 2 Humble과 다음 Python 모듈이 확인됐다.

- `rclpy`
- `cv2`
- `numpy`
- `PyQt5`
- `zmq`

다음 명령이 오류 없이 끝나는지 확인한다.

```bash
source /opt/ros/humble/setup.bash
python3 -c 'import rclpy, cv2, numpy, PyQt5, zmq'
```

`zmq`만 없다면 이 프로젝트의 실행 스크립트가 안내하는 패키지는
`python3-zmq`다.

```bash
sudo apt install python3-zmq
```

ROS package를 빌드한다.

```bash
cd ~/youngwoong_ws/gello_software_remote_classifier/ros2_ur_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select gello_recorder --symlink-install
```

이 명령은 laptop3에서 실제로 성공했다.

### kanu

검증된 Python은 다음 conda 환경이다.

```bash
/home/junhyeong/miniconda3/envs/il/bin/python
```

이 환경에 `pyzmq 27.1`, `einops 0.8.2`가 설치되어 있고 JAX `0.5.3`의
GPU backend가 확인됐다. 확인 명령:

```bash
/home/junhyeong/miniconda3/envs/il/bin/python -m pip install pyzmq einops
nvidia-smi
CUDA_VISIBLE_DEVICES=<GPU_ID> \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/junhyeong/miniconda3/envs/il/bin/python -c \
  'import cv2, flax, jax, numpy, zmq, einops; print(jax.__version__, jax.default_backend())'
```

`<GPU_ID>`는 `nvidia-smi`에서 확인한 여유 GPU 번호로 바꾼다. 마지막 출력의
backend가 `gpu`여야 한다. 또한 아래 두 파일이 있어야 한다.

```bash
test -f \
  ~/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150
test -f ~/.serl/resnet10_params.pkl
```

## 3. laptop3에서 kanu SSH 확인

laptop3의 기존 `~/.ssh/config` 내용을 덮어쓰지 말고 다음 host block이 있는지
확인한다. 없다면 기존 파일 끝에 추가한다.

```sshconfig
Host kanu
  HostName 166.104.35.33
  User junhyeong
  ServerAliveInterval 60
  ServerAliveCountMax 3
  TCPKeepAlive yes
```

공개키 로그인이 아직 설정되지 않았다면 laptop3에서 다음을 실행한다.

```bash
ssh-copy-id kanu
```

이는 kanu의 기존 `~/.ssh/authorized_keys`를 교체하는 명령이 아니라 새 공개키를
추가하는 용도로 사용한다. `~/.ssh/config`나 `authorized_keys`를 새 파일로
덮어쓰지 않는다.

laptop3에서 다음 명령이 암호 입력 없이 성공해야 한다.

```bash
ssh kanu hostname
```

이 연결은 classifier 이미지와 결과를 전달하는 SSH tunnel에도 그대로 사용된다.

## 4. kanu GPU classifier server 실행

먼저 kanu에서 `nvidia-smi`를 확인하고 여유 GPU 하나를 고른다.

```bash
nvidia-smi
```

아래 `<GPU_ID>`를 선택한 번호로 바꿔 kanu 터미널에서 실행한다. E2E 검증
당시에는 여유 GPU `3`을 사용했다.

```bash
cd ~/workspace/youngwoong/gello_software_remote_classifier
CUDA_VISIBLE_DEVICES=<GPU_ID> \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
REWARD_CLASSIFIER_PYTHON=/home/junhyeong/miniconda3/envs/il/bin/python \
./serl_ur_infra/run_remote_reward_classifier_server.sh
```

정상 준비 완료 로그:

```text
JAX backend: gpu
remote reward classifier ready on tcp://127.0.0.1:5594
```

이 터미널은 계속 열어 둔다.

> `CUDA_VISIBLE_DEVICES=<GPU_ID>`와
> `XLA_PYTHON_CLIENT_PREALLOCATE=false`를 생략하지 않는다. 제한 없이
> 실행했을 때 JAX가 모든 GPU context와 GPU 0의 약 12 GiB를 점유하는 현상이
> 실제 확인됐다. 검증 당시 선택한 GPU 3의 점유량은 약 2.26 GiB였다.

Server는 외부 인터페이스가 아닌 kanu loopback에만 bind된다. laptop3는 다음
단계의 SSH forwarding을 통해서만 접근한다.

## 5. laptop3에서 두 카메라 실행

laptop3의 별도 물리 데스크톱 터미널에서:

```bash
cd ~/youngwoong_ws/gello_software_remote_classifier/ros2_ur_ws
VIEW=false ./launch_cameras.sh
```

기본 camera mapping과 토픽은 다음과 같다.

- cam1: scene 카메라,
  `/cam1/cam1/color/image_raw/compressed`
- cam2: close-up 카메라,
  `/cam2/cam2/color/image_raw/compressed`

다른 터미널에서 실제 publish를 확인한다.

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /cam1/cam1/color/image_raw/compressed
ros2 topic hz /cam2/cam2/color/image_raw/compressed
```

`launch_cameras.sh`도 두 stream이 약 25 Hz 이상인지 검사한 뒤 준비 완료를
표시한다. 카메라 터미널은 계속 열어 둔다.

주의: 검증 당시 laptop3에 이미 존재하던 `/camera/color/image_raw` 하나만으로는
classifier가 동작하지 않았다. 위의 정확한 cam1/cam2 compressed 토픽 두 개가
모두 필요하다.

## 6. laptop3에서 SSH tunnel과 classifier UI 실행

카메라와 kanu server가 준비된 뒤, laptop3의 물리 데스크톱 터미널에서:

```bash
cd ~/youngwoong_ws/gello_software_remote_classifier/ros2_ur_ws
./run_remote_classifier_viewer.sh
```

이 스크립트가 SSH forwarding을 자동으로 구성하며 다음 프로세스를 함께
관리한다.

1. 두 ROS compressed image를 정렬해 kanu로 전송하는 client
2. 두 카메라와 classifier 상태를 표시하는 UI

SSH로 laptop3에 들어간 터미널은 일반적으로 `DISPLAY`가 없으므로 UI를 띄울 수
없다. 반드시 laptop3의 실제 데스크톱에서 터미널을 열어 실행한다.

## 7. 정상 UI 판별

정상 상태에서는 다음을 확인할 수 있다.

- 두 카메라 실시간 preview
- 파란색 `REMOTE CONNECTED`
- `p(success)` 실수값
- threshold `0.50`
- 성공이면 초록색 `SUCCESS`
- 실패면 빨간색 `FAILURE`
- camera skew
- kanu inference latency
- 전체 round-trip latency
- frame age

이 표시는 진단 전용이다. UI의 판정이 로봇을 움직이거나 episode를 종료하지
않는다.

실제 synthetic JPEG E2E 검증 결과:

- `p(success)=0.0813422571`
- `success=false`
- `threshold=0.5`
- camera skew `15 ms`
- warm inference `2.6–4.6 ms`
- warm ZMQ round trip `4.8–8.3 ms`
- 최초 요청은 JAX 초기화 영향으로 약 `151 ms`

ROS client까지 포함한 검증에서는 다음 status가 확인됐다.

- `ready=true`
- `remote=true`
- inference `4.36 ms`
- round trip `9.98 ms`
- capture age `144 ms`

실제 카메라 영상의 확률과 latency는 장면 및 네트워크 상태에 따라 달라진다.

## 8. 종료

1. laptop3의 classifier UI 터미널에서 `Ctrl-C`
2. laptop3의 카메라 터미널에서 `Ctrl-C`
3. kanu server 터미널에서 `Ctrl-C`

`run_remote_classifier_viewer.sh`는 종료할 때 UI, ROS client, SSH tunnel을 함께
정리한다. kanu server는 별도 프로세스이므로 kanu 터미널에서 직접 종료한다.

정상 종료 후 kanu GPU 3 사용량은 검증 환경에서 약 2 MiB로 돌아왔고, 양쪽
5594 forwarding/server도 정리됐다.

## 9. 문제 해결

### `SSH tunnel to kanu failed`

```bash
ssh kanu hostname
```

먼저 공개키 로그인이 되는지 확인한다. kanu server가 실행 중인지와 ready 로그도
확인한다.

### kanu에서 `GPU required but JAX backend is cpu`

반드시 검증된 conda Python과 GPU 제한을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
REWARD_CLASSIFIER_PYTHON=/home/junhyeong/miniconda3/envs/il/bin/python \
./serl_ur_infra/run_remote_reward_classifier_server.sh
```

### UI가 `REMOTE OFFLINE` 또는 `REMOTE STALE`

다음 순서로 확인한다.

1. kanu server에 ready 로그가 있는가
2. `ssh kanu hostname`이 성공하는가
3. cam1/cam2 compressed 토픽이 모두 publish 중인가
4. 두 카메라의 timestamp 차이가 지나치게 크지 않은가

Client는 기본적으로 camera skew가 `100 ms`를 넘는 pair를 사용하지 않고,
camera/result age가 `500 ms`를 넘으면 stale로 표시한다.

### UI는 열리지만 영상이 없다

정확한 두 토픽을 확인한다.

```bash
ros2 topic list | grep color/image_raw/compressed
```

필요한 이름은 기본적으로 다음 두 개다.

```text
/cam1/cam1/color/image_raw/compressed
/cam2/cam2/color/image_raw/compressed
```

### `qt.qpa... could not connect to display`

SSH 접속 터미널에서 UI를 실행한 경우다. laptop3 물리 데스크톱의 터미널에서
`./run_remote_classifier_viewer.sh`를 실행한다.

### GPU 메모리를 과도하게 점유한다

kanu server를 종료한 뒤, 4절의 명령을 그대로 사용해 다시 실행한다.
`CUDA_VISIBLE_DEVICES=<GPU_ID>`와
`XLA_PYTHON_CLIENT_PREALLOCATE=false`를 확인한다.
