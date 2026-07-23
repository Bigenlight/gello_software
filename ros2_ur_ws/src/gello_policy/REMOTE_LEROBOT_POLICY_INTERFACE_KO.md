# 범용 LeRobot Policy 원격 추론 인터페이스 사용 가이드

이 문서는 로봇 PC의 observation을 **Kanu GPU 서버**로 보내고, Kanu에서
LeRobot policy checkpoint를 이용해 추론한 뒤, 생성된 7차원 action을 다시
로봇 PC로 받는 방법을 설명한다.

현재 cube-in-cup Flow Matching 70,000-step checkpoint를 실제 UR7e에 적용하는
구체적인 절차는
[`FLOW_MATCHING_KANU_REAL_ROBOT_GUIDE_KO.md`](FLOW_MATCHING_KANU_REAL_ROBOT_GUIDE_KO.md)를
함께 참고한다.

대상 브랜치:

```text
feat/remote-gpu-server
```

## 1. 지원 범위

정책 종류를 ACT, Diffusion, Flow Matching으로 서버 코드에 고정하지 않는다.
checkpoint의 `config.type`을 읽어 LeRobot의 `get_policy_class()`로 policy class를
결정한다.

다음 조건을 모두 만족하는 checkpoint를 지원한다.

1. LeRobot 표준 `Policy Class`와 `from_pretrained()`로 불러올 수 있다.
2. 저장된 preprocessor와 postprocessor가 존재한다.
3. `policy.reset()`과 `policy.select_action()` 인터페이스를 사용한다.
4. 입출력이 현재 UR7e 데이터 계약과 일치한다.

```text
입력
  observation.state          shape (7,)
  observation.images.cam1    RGB camera
  observation.images.cam2    RGB camera
  task                       필요한 정책만 사용

출력
  action                     shape (7,)
```

7차원 state와 action의 현재 의미는 다음과 같다.

```text
observation.state = [ur_q1, ur_q2, ur_q3, ur_q4, ur_q5, ur_q6, grip_pos]
action            = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, grip_cmd]
```

state/action 차원, 관절 순서, 카메라 key가 다른 checkpoint는 자동 변환하지
않는다. 이런 checkpoint는 Kanu 서버 시작 단계에서 거부되며 별도의 adapter가
필요하다.

## 2. 전체 구조

```text
로봇 PC                                           Kanu GPU 서버
--------                                          -------------
/joint_states                                     Docker container
gripper position                                  LeRobotPolicyWrapper
cam1 JPEG                 SSH tunnel + gRPC       config.type 확인
cam2 JPEG              ----------------------->   policy class/weight 로드
                                                   preprocessor
policy_leader_node     <-----------------------   select_action (GPU)
안전 clamp                 action[7]               postprocessor
UR7e/Robotiq 제어
```

로봇 PC가 담당하는 것:

- ROS2 topic 구독
- 카메라·관절·gripper observation 생성
- observation freshness와 통신 timeout 검사
- joint limit 및 현재 관절 대비 최대 변화 clamp
- 시작 자세, HOLD/EXECUTE/FAULT 상태 관리
- UR7e와 Robotiq 실제 제어

Kanu가 담당하는 것:

- checkpoint와 LeRobot 환경 보관
- policy class 및 saved processor 로드
- 이미지 decode와 정책별 resize
- GPU inference
- action과 latency metadata 반환

## 3. 구현 파일 위치

### Kanu 측

| 파일 | 역할 |
|---|---|
| `policy_server/lerobot_policy_wrapper.py` | checkpoint config, policy class, weight, pre/postprocessor 로드와 공통 추론 |
| `policy_server/remote_lerobot_server.py` | Kanu gRPC 서버, checkpoint 계약 검사, CUDA warm-up, action 응답 |
| `deploy/remote_diffusion/Dockerfile` | Python 3.12, CUDA PyTorch, LeRobot 0.6.1 실행 환경 |
| `deploy/remote_diffusion/compose.yaml` | `policy-server` Docker Compose profile |
| `deploy/remote_diffusion/.env.example` | Kanu 런타임 환경변수 예시 |

### 로봇 PC 측

| 파일 | 역할 |
|---|---|
| `gello_policy/remote_policy_client.py` | 공통 원격 policy client 이름과 타입 제공 |
| `gello_policy/remote_diffusion_client.py` | 검증된 gRPC session, latest-only observation, timeout 처리 구현 |
| `gello_policy/policy_leader_node.py` | observation 생성, action 수신, 안전 clamp, UR용 synthetic leader 발행 |
| `run_ur7e_diffusion_remote.sh` | SSH tunnel과 ROS2 실행. 파일명은 기존 호환을 위해 유지 |

protobuf service 이름도 기존 `RemoteDiffusion` v1을 유지한다. 이름만 역사적으로
Diffusion일 뿐, observation/session/action 전송 부분은 특정 policy 계산을 포함하지
않으므로 공통 서버에서도 그대로 재사용한다.

## 4. 두 머신에서 저장소 준비

브랜치가 GitHub에 push된 뒤 로봇 PC와 Kanu 양쪽에서 동일한 브랜치를 받는다.

```bash
git clone https://github.com/Bigenlight/gello_software.git
cd gello_software
git fetch origin
git switch feat/remote-gpu-server
```

이미 저장소가 있다면 다시 clone할 필요가 없다.

```bash
cd /path/to/gello_software
git fetch origin
git switch feat/remote-gpu-server
git pull --ff-only
```

## 5. Kanu에 checkpoint 준비

공통 서버가 checkpoint를 자동으로 다운로드하지는 않는다. 사용할
`pretrained_model` 디렉터리를 Kanu에 먼저 준비한다.

예시 구조:

```text
/path/to/server/project-workspace/
├── gello_software/
└── models/
    └── cube_flow_matching_070000/
        ├── config.json
        ├── model.safetensors
        ├── policy_preprocessor.json
        ├── policy_postprocessor.json
        └── processor에 필요한 safetensors/json 파일
```

checkpoint는 Hugging Face에서 다운로드하거나 학습 output에서 복사할 수 있다.
항상 `pretrained_model` 디렉터리 자체를 `CHECKPOINT_DIR`로 지정한다.

현재 cube-in-cup Flow Matching 70,000-step checkpoint는 Kanu의 다음 위치에
미리 저장되어 있다.

```text
/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/checkpoints/flow_matching_cube_in_cup_checkpoints/070000/pretrained_model
```

따라서 이 checkpoint를 사용할 때는 위 경로를 `CHECKPOINT_DIR`로 지정한다.

## 6. Kanu `.env` 작성

```bash
cd gello_software/ros2_ur_ws/src/gello_policy/deploy/remote_diffusion
cp .env.example .env
```

`.env`는 commit하지 않는다. 사용할 모델에 맞게 다음 값을 수정한다.

### Flow Matching 예시

```dotenv
CHECKPOINT_DIR=/path/to/server/project-workspace/models/cube_flow_matching_070000
HF_CACHE_DIR=/path/to/server/project-workspace/huggingface-cache
GPU_DEVICE=0

INFERENCE_BIND_IP=127.0.0.1
POLICY_INFERENCE_PORT=50052

MODEL_ID=Bigenlight/cube_flow_matching
CHECKPOINT_REVISION=sha256:실제_체크포인트_해시

POLICY_TASK=put the cube in the cup
POLICY_WARMUP_STATE=[3.106,-1.817,1.653,-1.618,-1.628,-3.195,0.0]
POLICY_CONFIG_OVERRIDES={"num_integration_steps":10,"n_action_steps":24}
EXTERNAL_IMAGE_SIZE=native
```

### Diffusion 예시

```dotenv
POLICY_TASK=
POLICY_CONFIG_OVERRIDES={"noise_scheduler_type":"DDIM","num_inference_steps":10,"n_action_steps":32}
EXTERNAL_IMAGE_SIZE=360x640
```

### ACT 예시

```dotenv
POLICY_TASK=
POLICY_CONFIG_OVERRIDES={"n_action_steps":30}
EXTERNAL_IMAGE_SIZE=360x640
```

주의 사항:

- `POLICY_TASK`는 학습 데이터에 사용한 문구와 정확히 같아야 한다.
- saved preprocessor에 tokenizer가 있는데 task가 비어 있으면 서버가 시작되지 않는다.
- MultiTaskDiT는 policy weight를 읽기 전에 CLIP encoder를 `from_pretrained()`로
  생성한다. `HF_CACHE_DIR`에는 checkpoint config의 encoder 이름에 해당하는
  모델을 서버 실행 전에 미리 받아 두어야 한다. 컨테이너는 재현성과 실기 중
  네트워크 접근 방지를 위해 이 캐시를 read-only/offline으로 사용한다.
- `EXTERNAL_IMAGE_SIZE=native`는 MultiTaskDiT처럼 모델 내부에서 resize하는 정책에 사용한다.
- 학습 당시 dataset transform이 360×640이었다면 `360x640`을 명시한다.
- 추론 step을 줄이기 전에는 모델별 offline action 평가가 필요하다.

## 7. Kanu GPU 선택 및 서버 실행

공유 GPU를 사용하기 직전에 빈 GPU를 확인한다.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv
```

선택한 GPU 번호를 `.env`의 `GPU_DEVICE`에 기록한 뒤 이미지 설정을 확인한다.

```bash
docker compose --env-file .env --profile generic config
```

최초 빌드 및 실행:

```bash
docker compose --env-file .env --profile generic \
  up --build -d policy-server
```

로그 확인:

```bash
docker compose --env-file .env --profile generic logs -f policy-server
```

정상 시작 시 다음 과정이 로그에 나타난다.

1. checkpoint `config.json` 로드
2. `config.type`에 해당하는 Policy Class 선택
3. weight와 saved pre/postprocessor 로드
4. 입력 `state[7] + cam1 + cam2`, 출력 `action[7]` 계약 검사
5. 합성 observation으로 CUDA warm-up 2회
6. policy queue reset
7. `0.0.0.0:50051`에서 container 내부 gRPC 준비

호스트에서는 기본적으로 `127.0.0.1:50052`로만 공개된다. 공용 IP에 직접
노출하지 않는다.

상태 확인:

```bash
docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' \
  gello-remote-policy
```

## 8. 로봇 PC client 환경 준비

```bash
cd /path/to/gello_software/ros2_ur_ws
./setup_remote_client_venv.sh
```

ROS 패키지 코드가 변경됐으므로 checkout 후 한 번 빌드한다.

```bash
./build_ur7e.sh
source install/setup.bash
```

## 9. 로봇 없이 SSH/gRPC 왕복 검사

먼저 Kanu의 `policy-server`가 실행 중인지 확인한다. 로봇 PC에서 다음을 실행한다.

```bash
cd /path/to/gello_software/ros2_ur_ws

SSH_HOST=kanu \
REMOTE_GRPC_PORT=50052 \
LOCAL_GRPC_PORT=50052 \
ROUNDTRIP_ONLY=1 \
./run_ur7e_diffusion_remote.sh
```

기본 합성 이미지 크기는 서버가 외부 resize 크기를 보고하면 그 값을 사용하고,
서버가 `0/0`(모델 내부 resize)을 보고하면 360×640을 사용한다. 다른 native 입력
크기가 필요하면 `ROUNDTRIP_IMAGE_HEIGHT`, `ROUNDTRIP_IMAGE_WIDTH`를 지정한다.
합성 state도 바꾸려면 `ROUNDTRIP_STATE='[q1,q2,q3,q4,q5,q6,grip]'`를 지정한다.

`SSH_HOST`는 실제 hostname 또는 `~/.ssh/config` alias다. 이 명령은:

1. `127.0.0.1:50052` SSH tunnel을 연다.
2. gRPC HTTP/2 연결을 확인한다.
3. held state와 두 장의 검은 JPEG를 Kanu로 보낸다.
4. Kanu에서 실제 checkpoint inference를 수행한다.
5. 유한한 7차원 action을 로컬에서 받았는지 검사한다.
6. ROS2와 실제 로봇은 실행하지 않는다.

`ROUNDTRIP_ONLY PASS`가 나오기 전에는 실제 로봇 단계로 넘어가지 않는다.

## 10. 로컬 contract 설정

로컬 client는 잘못된 모델에 로봇을 연결하지 않도록 다음 값을 엄격하게 비교한다.

- `MODEL_ID`
- `CHECKPOINT_REVISION`
- policy type/objective에 해당하는 sampling method
- sampling/integration step 수
- `n_action_steps`
- 외부 resize 크기
- state/action 차원
- CUDA 사용 여부

protobuf v1 호환 때문에 공통 서버는 기존 필드를 다음 의미로 사용한다.

| 기존 ServerInfo 필드 | 공통 서버에서의 의미 |
|---|---|
| `scheduler` | `<policy.type>` 또는 `<policy.type>:<objective>` |
| `num_inference_steps` | inference/integration step 수, 없으면 1 |
| `n_action_steps` | action chunk에서 실행하는 action 수 |
| `resize_height`, `resize_width` | 외부 resize 크기. 모델 내부 resize면 `0, 0` |

예를 들어 Flow Matching은 다음과 같이 보고된다.

```text
scheduler             = multi_task_dit:flow_matching
num_inference_steps   = 10
n_action_steps        = 24
resize_height/width   = 0/0
```

실제 로봇 launch에 사용할 parameter YAML은 해당 task의 기존 deploy YAML을
복사해 안전 limit과 start pose를 유지한 뒤, `policy_leader_node` 아래의
`expected_*` 값을 Kanu `.env`와 동일하게 설정한다.

```yaml
policy_leader_node:
  ros__parameters:
    expected_model_id: "Bigenlight/cube_flow_matching"
    expected_checkpoint_revision: "sha256:실제_체크포인트_해시"
    expected_scheduler: "multi_task_dit:flow_matching"
    expected_inference_steps: 10
    expected_action_steps: 24
    expected_resize_height: 0
    expected_resize_width: 0
```

## 11. 실제 ROS2 연결

실제 로봇 실행은 기존 원격 Diffusion 런처의 SSH/gRPC 및 안전 스택을 재사용한다.
아래의 `/absolute/path/remote_policy_deploy.yaml`은 앞 절에서 준비한 전체 parameter
파일이어야 한다.

```bash
cd /path/to/gello_software/ros2_ur_ws

SSH_HOST=kanu \
REMOTE_GRPC_PORT=50052 \
LOCAL_GRPC_PORT=50052 \
ROBOT_IP=192.168.10.11 \
HEADLESS=true \
./run_ur7e_diffusion_remote.sh \
  params_file:=/absolute/path/remote_policy_deploy.yaml \
  launch_rviz:=false
```

파일명에 `diffusion`이 남아 있지만 이 스크립트의 역할은 SSH tunnel과 공통 gRPC
client 및 ROS launch다. 실제 policy 계산은 Kanu의 `remote_lerobot_server.py`가
담당한다.

로봇은 handshake 후 HOLD 상태에 있어야 한다. observation, server contract,
시작 자세와 주변 안전을 확인한 뒤에만 실행한다.

```bash
ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
```

중지:

```bash
ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger
```

위험 상황에서는 명령보다 pendant 정지와 비상정지를 우선한다.

## 12. 종료

로봇 PC에서는 ROS launch를 `Ctrl-C`로 종료한다. 스크립트가 자신이 생성한 SSH
tunnel을 함께 정리한다.

Kanu:

```bash
docker compose --env-file .env --profile generic stop policy-server
```

## 13. 새 policy 적용 체크리스트

- [ ] LeRobot `get_policy_class(config.type)`가 policy를 지원한다.
- [ ] Kanu Docker 이미지에 새 policy의 Python dependency가 있다.
- [ ] 외부 encoder/tokenizer가 필요한 policy는 `HF_CACHE_DIR`에 파일이 준비돼 있다.
- [ ] checkpoint에 config, model, preprocessor, postprocessor 파일이 있다.
- [ ] observation key와 shape가 현재 7D·2-camera 계약과 일치한다.
- [ ] action이 절대 UR joint target 6개와 gripper command 1개다.
- [ ] task 문자열과 카메라 순서가 학습 시점과 동일하다.
- [ ] 이미지 resize가 학습 시점과 동일하다.
- [ ] Kanu GPU가 비어 있고 CUDA warm-up이 성공했다.
- [ ] `ROUNDTRIP_ONLY=1` 검사가 통과했다.
- [ ] recorded observation으로 기존/reference inference와 action을 비교했다.
- [ ] p99 추론 왕복 시간이 로컬 timeout보다 짧다.
- [ ] start pose와 joint safety envelope를 해당 데이터셋에 맞게 검토했다.
- [ ] 실제 로봇 최초 실행은 저속·비상정지 준비 상태에서 수행한다.

## 14. 현재 검증 상태

현재 코드 수준에서 확인된 항목:

- 기존 remote client/session/timeout 및 공통 server 테스트 20개
- protocol 오류는 복구하고 policy/후처리 오류는 restart-required로 전환하는 동작
- 가짜 LeRobot policy를 이용한 JPEG decode, 입력 shape, task, action queue lifecycle
- LeRobot factory와 config override import
- MultiTaskDiT extra 의존성 선언과 영속·offline HF cache mount
- Docker Compose `policy-server` 설정 해석
- Python 문법 및 Git diff 검사

아직 환경별로 수행해야 하는 항목:

- 실제 Kanu checkpoint mount
- Kanu Docker image build와 CUDA 실행
- ACT/Diffusion/Flow Matching 각각의 reference action parity
- SSH tunnel 기준 latency 측정
- 실제 UR7e closed-loop 검증

## 15. 2026-07-22 추가 검토 및 수정 내역

추가 검증 과정에서 다음 문제를 발견해 현재 작업 트리에 반영했다.

1. 원격 Docker 이미지에 MultiTaskDiT/Flow Matching이 요구하는
   `transformers` 계열 의존성이 빠져 있었다. Dockerfile이 고정 LeRobot commit의
   `multi-task-dit` extra를 설치하도록 수정했다.
2. MultiTaskDiT가 초기화 중 CLIP encoder/tokenizer를 `from_pretrained()`로
   생성하지만 컨테이너 캐시가 `/tmp`라 재시작마다 사라지는 문제가 있었다.
   `HF_CACHE_DIR`를 read-only로 마운트하고 `HF_HUB_OFFLINE=1`,
   `TRANSFORMERS_OFFLINE=1`로 실행하도록 수정했다.
3. 프로토콜 입력 오류와 policy/후처리 `ValueError`가 구분되지 않았다. action
   queue가 변경된 뒤 오류가 발생했을 가능성이 있는 추론 오류는 서버를
   restart-required 상태로 전환하고, policy 호출 전 검출한 프로토콜·세션 오류만
   해당 요청에서 복구하도록 수정했다.
4. `ROUNDTRIP_ONLY=1` 검사가 기존 Diffusion model ID, DDIM 설정, 360×640 크기를
   하드코딩하고 있었다. 서버의 실제 공통 policy contract를 조회하고 ACT,
   Diffusion, Flow Matching 모두에 사용할 수 있도록 수정했다.
5. 관련 client/server 테스트 20개, 가짜 LeRobot policy lifecycle, Python/Bash
   문법, Docker Compose 해석, PEP 508 extra 선언, `git diff --check`를 통과했다.

공통 원격 GPU inference 개발·배포 브랜치는 `feat/remote-gpu-server`다. 로컬과
Kanu에서는 이 브랜치의 동일한 commit을 사용해야 한다. Kanu 실환경에서는 아직
실제 checkpoint Docker load, CUDA warm-up, SSH/gRPC 왕복, latency 및 실물 UR7e
검증이 남아 있다.
