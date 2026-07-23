# Cube Flow Matching Kanu 원격 추론 및 UR7e 실물 실행 가이드

이 문서는 70,000-step cube-in-cup Flow Matching checkpoint를 Kanu GPU 서버에서
추론하고, 로컬 노트북이 UR7e observation을 전송해 받은 7차원 action으로 로봇을
제어하는 전체 절차를 설명한다.

실물 로봇을 움직이기 전 반드시 로봇 없는 왕복 검사까지 통과해야 한다. 최초
실행은 저속 설정, pendant 및 비상정지에 즉시 접근 가능한 상태에서 진행한다.

## 1. 현재 구현 상태

대상 브랜치:

```text
feat/remote-gpu-server
```

구현이 완료된 범위:

- LeRobot `config.type` 기반 공통 policy class 선택 및 `from_pretrained()` 로드
- 저장된 preprocessor/postprocessor와 `reset()`/`select_action()` lifecycle
- Kanu의 Docker/CUDA 기반 gRPC inference server
- SSH tunnel을 통한 observation 전송과 7차원 action 반환
- 로컬의 latest-only 요청, timeout, session/request 순서 검사
- 로컬의 observation freshness, joint limit, 최대 관절 변화 및 HOLD/FAULT 처리
- MultiTaskDiT용 의존성과 영속·offline Hugging Face cache
- policy 호출 전 프로토콜 오류와 상태가 변경될 수 있는 추론 오류의 분리
- 관련 자동 테스트 20개와 가짜 policy lifecycle 검사

아직 완료되지 않은 실환경 검증:

- 아래 실제 checkpoint의 Kanu Docker load 및 CUDA warm-up
- 로컬 노트북에서 Kanu까지 실제 SSH/gRPC 왕복 추론
- 기존 Flow Matching inference와의 reference action 비교
- 실측 refill/p99 latency가 로컬 timeout 안에 들어오는지 확인
- 실제 UR7e/Robotiq closed-loop 실행

공통 원격 GPU inference 코드는 `feat/remote-gpu-server` 브랜치에서 관리한다.
Kanu와 로컬 노트북은 반드시 이 브랜치의 동일한 commit을 사용한다.

## 2. 머신별 역할과 데이터 흐름

```text
로컬 노트북                                      Kanu GPU 서버
-----------                                      -------------
UR7e joint/gripper 상태                          checkpoint/config 로드
cam1/cam2 JPEG             SSH tunnel + gRPC     MultiTaskDiT + Flow Matching
policy_leader_node  -------------------------->  CUDA select_action
안전 검사 및 clamp      <----------------------  action[7]
UR7e/Robotiq 제어
```

Kanu는 action을 계산할 뿐 로봇 controller에 직접 연결하지 않는다. 로봇 명령과
모든 실물 안전 처리는 로컬 노트북이 소유한다.

## 3. 주요 폴더와 파일

저장소 기준 주요 구현 위치는 다음과 같다.

| 위치 | 역할 |
|---|---|
| `ros2_ur_ws/src/gello_policy/policy_server/lerobot_policy_wrapper.py` | 공통 LeRobot checkpoint load/inference |
| `ros2_ur_ws/src/gello_policy/policy_server/remote_lerobot_server.py` | Kanu 공통 gRPC server와 7D·2-camera 계약 검사 |
| `ros2_ur_ws/src/gello_policy/deploy/remote_diffusion/Dockerfile` | CUDA PyTorch, LeRobot 및 MultiTaskDiT 환경 |
| `ros2_ur_ws/src/gello_policy/deploy/remote_diffusion/compose.yaml` | Kanu `policy-server` container 정의 |
| `ros2_ur_ws/src/gello_policy/gello_policy/remote_policy_client.py` | policy-neutral 원격 client 이름 |
| `ros2_ur_ws/src/gello_policy/gello_policy/remote_diffusion_client.py` | gRPC session, timeout, latest-only transport |
| `ros2_ur_ws/src/gello_policy/gello_policy/policy_leader_node.py` | observation 생성, action 안전 검사와 synthetic leader 발행 |
| `ros2_ur_ws/run_ur7e_diffusion_remote.sh` | SSH tunnel, 왕복 검사 및 ROS launch |
| `ros2_ur_ws/remote_diffusion_roundtrip_smoke.py` | 로봇 없는 실제 checkpoint 왕복 추론 검사 |
| `ros2_ur_ws/src/gello_policy/config/fm_deploy.yaml` | Flow Matching용 기존 시작 자세·안전 설정의 기준 |

파일명과 protobuf service에 `diffusion`이 남은 부분은 기존 원격 전송 계약과
배포 호환성을 유지하기 위한 것이다. 공통 서버의 policy 계산은 Diffusion에
한정되지 않는다.

## 4. 사용할 Kanu checkpoint

Hugging Face에 저장했던 70,000-step checkpoint는 Kanu에 다음 경로로 미리
다운로드되어 있다.

```text
/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/checkpoints/flow_matching_cube_in_cup_checkpoints/070000/pretrained_model
```

이 문서에서는 다음과 같이 줄여 쓴다.

```bash
export FM_CHECKPOINT=/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/checkpoints/flow_matching_cube_in_cup_checkpoints/070000/pretrained_model
```

Kanu에서 필수 파일과 policy contract를 확인한다.

```bash
test -f "$FM_CHECKPOINT/config.json"
test -f "$FM_CHECKPOINT/model.safetensors"
test -f "$FM_CHECKPOINT/policy_preprocessor.json"
test -f "$FM_CHECKPOINT/policy_postprocessor.json"

python3 - "$FM_CHECKPOINT/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
for key in (
    "type", "objective", "num_integration_steps", "n_action_steps",
    "vision_encoder_name", "text_encoder_name", "input_features",
    "output_features",
):
    print(f"{key}: {config.get(key)}")
PY

sha256sum "$FM_CHECKPOINT/model.safetensors"
```

확인해야 할 핵심 값은 다음과 같다.

- `type: multi_task_dit`
- `objective: flow_matching`
- 입력 key가 `observation.state`, `observation.images.cam1`,
  `observation.images.cam2`
- state와 action shape가 각각 `(7,)`
- 학습에 사용한 정확한 task 문구
- 실제 `n_action_steps`와 encoder 이름

서버가 출력하는 checkpoint manifest SHA-256을 뒤의
`EXPECTED_CHECKPOINT_REVISION`과 로컬 `expected_checkpoint_revision`에
동일하게 사용한다.

## 5. 양쪽 머신에 코드 준비

먼저 작업 PC에서 현재 변경을 커밋하고 GitHub에 branch를 push한다. 그 후 Kanu와
로컬 로봇 노트북 양쪽에서 같은 commit을 사용한다.

```bash
cd /path/to/gello_software
git fetch origin
git switch feat/remote-gpu-server
git pull --ff-only
git rev-parse HEAD
```

두 머신의 `git rev-parse HEAD` 결과가 같아야 한다. 저장소가 이미 있다면 다시
clone할 필요가 없다.

## 6. Kanu의 CLIP/Hugging Face cache 확인

MultiTaskDiT는 policy weight를 적용하기 전에 config에 지정된 CLIP vision/text
encoder와 tokenizer를 `from_pretrained()`로 생성한다. 공통 container는 실기 중
외부 다운로드를 막기 위해 cache를 read-only/offline으로 사용한다.

학습에 사용한 Kanu cache가 기본 위치에 있다면 다음 경로를 우선 사용한다.

```text
/home/junhyeong/.cache/huggingface
```

config에서 확인한 encoder가 cache에 있는지 점검한다.

```bash
find /home/junhyeong/.cache/huggingface/hub -maxdepth 1 \
  -type d -name 'models--*clip*' -print
```

없다면 Docker server를 시작하기 전에, 네트워크가 가능한 Kanu 학습 환경에서
config의 `vision_encoder_name`과 `text_encoder_name`을 한 번
`from_pretrained()`로 불러 동일 cache에 저장한다. 빈 cache로 container를
실행하면 offline load 단계에서 정상적으로 실패한다.

## 7. Kanu `.env` 작성

Kanu 저장소의 배포 디렉터리로 이동한다.

```bash
cd /path/to/gello_software/ros2_ur_ws/src/gello_policy/deploy/remote_diffusion
cp .env.example .env
```

`.env`를 다음과 같이 설정한다. `POLICY_TASK`는 반드시 학습 데이터에 사용된
문구로 확인하고, 해시도 앞 절의 실제 결과로 교체한다.

```dotenv
CHECKPOINT_DIR=/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/checkpoints/flow_matching_cube_in_cup_checkpoints/070000/pretrained_model
HF_CACHE_DIR=/home/junhyeong/.cache/huggingface

GPU_DEVICE=0
INFERENCE_BIND_IP=127.0.0.1
POLICY_INFERENCE_PORT=50052

IMAGE_TAG=flow-matching-070000
LEROBOT_EXTRAS=multi-task-dit
EXPECTED_POLICY_TYPE=multi_task_dit
EXPECTED_CHECKPOINT_REVISION=sha256:아래_명령으로_계산한_manifest

POLICY_TASK=put the cube in the cup
POLICY_TASK_MODE=required
POLICY_WARMUP_STATE=[3.106,-1.817,1.653,-1.618,-1.628,-3.195,0.0]
POLICY_CONFIG_OVERRIDES={"num_integration_steps":10,"n_action_steps":24}
EXTERNAL_IMAGE_SIZE=native
```

주의할 점:

- `HF_CACHE_DIR`는 `hub/models--...`를 포함하는 Hugging Face cache 루트다.
  Compose는 공개 모델의 offline 추론에 필요한 `hub` 하위 폴더만 read-only로
  마운트하며 호스트의 `token` 파일은 컨테이너에 전달하지 않는다.

- `GPU_DEVICE`는 실행 직전 빈 GPU를 확인한 후 선택한다.
- `POLICY_TASK=put the cube in the cup`은 예시다. 실제 학습 문구가 다르면 반드시
  실제 문구로 바꾼다.
- `EXPECTED_CHECKPOINT_REVISION`은 단일 weight 파일 해시가 아니다. 배포
  디렉터리의 상위 `gello_policy`에서 다음 명령으로 먼저 계산한 값을 사용한다.
  `python3 -m policy_server.checkpoint_identity "$CHECKPOINT_DIR"`
  서버는 이 값이 실제 config, 모든 weight shard/index, preprocessor 및
  postprocessor 파일의 manifest와 다르면 시작하지 않는다.
- `n_action_steps`도 checkpoint 및 기존 offline 평가 설정과 일치시킨다.
- `num_integration_steps=10`은 기존 프로젝트의 offline 평가/latency 설정을
  따르는 값이다. 다른 step 수를 사용하려면 먼저 reference action과 latency를
  다시 검증한다.
- `EXTERNAL_IMAGE_SIZE=native`는 MultiTaskDiT 내부 resize/crop을 사용한다는
  의미다.

## 8. Kanu Docker server 실행

먼저 GPU 사용량과 최종 Compose 설정을 확인한다.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv

docker compose --env-file .env --profile generic config
```

선택한 GPU가 비어 있고 checkpoint/cache 경로가 올바르면 빌드하고 시작한다.

```bash
docker compose --env-file .env --profile generic \
  up --build -d policy-server

docker compose --env-file .env --profile generic logs -f policy-server
```

정상 로그 순서는 config/policy/processors 로드, 7D·2-camera 계약 검사, 합성
observation CUDA warm-up 2회, queue reset, gRPC ready다. 오류 없이 ready가 나온
뒤 별도 터미널에서 상태를 확인한다.

```bash
docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' \
  gello-remote-policy
```

서버는 Kanu의 `127.0.0.1:50052`에만 공개되므로 외부에서는 SSH tunnel을 통해서만
접속한다.

## 9. 로컬 노트북 준비

로컬 저장소의 현재 실제 위치는 다음과 같다.

```text
/home/laptop3/youngwoong_ws/gello_software
```

다른 노트북에서는 자신의 checkout 경로로 바꾼다. 이후 명령에서는 이를
`GELLO_REPO`로 지정한다. branch와 commit을 확인하고 원격 client 환경 및 ROS
package를 빌드한다.

```bash
export GELLO_REPO=/home/laptop3/youngwoong_ws/gello_software
cd "$GELLO_REPO"
git branch --show-current
git rev-parse HEAD

cd ros2_ur_ws
./setup_remote_client_venv.sh
./build_ur7e.sh
source install/setup.bash
```

`~/.ssh/config`의 `kanu` alias가 실제 서버에 로그인되는지도 확인한다.

```bash
ssh kanu 'hostname'
```

현재 개발 환경에서는 Kanu SSH가 `Permission denied (publickey,password)`로
실패했으므로, key 또는 계정 인증을 먼저 해결해야 이후 검증을 진행할 수 있다.

## 10. 로컬 실물 배포 YAML 만들기

저장소에 포함된 `config/fm_remote_deploy.yaml`을 사용한다. 다음 contract 값 중
placeholder revision만 Kanu 서버가 출력한 실제 manifest 값으로 교체한다.

```yaml
    inference_transport: "grpc"
    grpc_port: 50052

    expected_model_id: "multi_task_dit"
    expected_checkpoint_revision: "sha256:사전에_계산한_manifest_해시"
    expected_scheduler: "multi_task_dit:flow_matching"
    expected_inference_steps: 10
    expected_action_steps: 24
    expected_resize_height: 0
    expected_resize_width: 0
```

이 값들은 Kanu `.env`와 정확히 같아야 한다. 또한 다음 기존 값이 이번
cube-in-cup 데이터셋에도 맞는지 반드시 다시 확인한다.

- `start_pose`, `start_gripper`
- `joint_limits_lo`, `joint_limits_hi`
- `max_dev_rad`
- cam1/cam2 topic과 카메라 순서
- `act_timeout_s < obs_timeout_s < staleness_timeout_s`

원격 Flow Matching refill의 p99 시간이 기본 `act_timeout_s=0.6`초보다 길다면 실제
로봇을 실행하지 않는다. 먼저 integration step과 네트워크 latency를 줄이고 다시
측정한다. 단순히 timeout을 크게 늘리면 로봇 정지 반응도 늦어질 수 있다.

YAML을 추가한 뒤 ROS package를 다시 빌드한다.

```bash
./build_ur7e.sh
source install/setup.bash
```

## 11. 실물 로봇 없이 실제 checkpoint 왕복 검사

Kanu server가 실행 중인 상태에서 로컬 노트북에서 수행한다.

```bash
cd "$GELLO_REPO/ros2_ur_ws"

SSH_HOST=kanu \
REMOTE_GRPC_PORT=50052 \
LOCAL_GRPC_PORT=50052 \
ROUNDTRIP_TIMEOUT_S=30 \
ROUNDTRIP_STATE='[3.106,-1.817,1.653,-1.618,-1.628,-3.195,0.0]' \
ROUNDTRIP_ONLY=1 \
./run_ur7e_diffusion_remote.sh
```

이 검사는 ROS2와 로봇 controller를 시작하지 않고 검은 JPEG 두 장과 7차원 state를
전송한다. Kanu가 실제 checkpoint로 추론한 유한한 action 7개와 policy contract,
서버/왕복 latency가 출력되어야 한다.

반드시 다음을 확인한다.

- `ROUNDTRIP_ONLY PASS`
- `model_id`, checkpoint revision, scheduler가 의도한 값
- action 길이가 7이고 모든 값이 finite
- refill 시 `inference_ms`와 전체 왕복 시간이 timeout보다 충분히 작음
- 여러 번 실행해 action과 latency가 비정상적으로 튀지 않음

이 검사가 실패하면 실물 로봇 단계로 넘어가지 않는다.

## 12. 기록 observation과 reference action 비교

가능하면 학습 또는 기존 로컬 FM inference에서 사용했던 실제 기록 observation을
동일 전처리·task로 Kanu server에 보내 기존 결과와 비교한다. 검은 이미지 smoke는
통신과 실행 가능성만 확인할 뿐, 카메라 순서·task·normalization의 의미적 오류를
찾지 못한다.

최소한 다음을 비교한다.

- 동일 checkpoint와 integration/action step
- 동일 task 문자열
- cam1/cam2 순서와 RGB/JPEG decode 결과
- state joint 순서와 gripper 범위
- 첫 action 및 action chunk의 수치 차이
- queue reset 직후와 연속 요청 중 결과

reference parity가 설명되지 않으면 실제 로봇에 연결하지 않는다.

## 13. 실물 UR7e 연결 및 HOLD 상태 시작

로봇 IP, 카메라, Robotiq tool communication, 주변 안전을 확인한 뒤 실행한다.

```bash
cd "$GELLO_REPO/ros2_ur_ws"

SSH_HOST=kanu \
REMOTE_GRPC_PORT=50052 \
LOCAL_GRPC_PORT=50052 \
ROBOT_IP=192.168.10.11 \
HEADLESS=true \
./run_ur7e_diffusion_remote.sh \
  params_file:="$GELLO_REPO/ros2_ur_ws/src/gello_policy/config/fm_remote_deploy.yaml" \
  launch_rviz:=false
```

이 명령은 SSH tunnel을 열고 기존 UR7e launch와 공통 gRPC client를 실행한다.
프로세스가 시작되어도 policy leader는 즉시 EXECUTE하지 않고 HOLD 자세를
발행해야 한다.

실행 전 확인 순서:

1. Kanu 로그에 server ready가 유지되는지 확인한다.
2. 로컬 로그에서 model/checkpoint/scheduler/step/resize contract handshake가
   통과했는지 확인한다.
3. `/joint_states`, gripper position, cam1/cam2 timestamp와 freshness를 확인한다.
4. move-to-start 완료와 forward controller handover를 확인한다.
5. 실제 관절이 YAML의 `start_pose` 근처인지 확인한다.
6. pendant를 잡고 저속 모드 및 비상정지 준비 상태인지 확인한다.

모든 조건을 확인한 뒤에만 실행을 허가한다.

```bash
ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
```

즉시 HOLD로 되돌리기:

```bash
ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger
```

통신 timeout, stale observation, contract mismatch, 비정상 action 또는 카메라 오류가
발생하면 원인을 해결하기 전 재시작·재실행하지 않는다. 위험 상황에서는 ROS
명령보다 pendant 정지와 비상정지를 우선한다.

## 14. 종료와 재시작

로컬 launch는 `Ctrl-C`로 종료한다. runner가 자신이 연 SSH tunnel도 정리한다.

Kanu server 종료:

```bash
cd /path/to/gello_software/ros2_ur_ws/src/gello_policy/deploy/remote_diffusion
docker compose --env-file .env --profile generic stop policy-server
```

추론 또는 후처리 오류로 server가 restart-required가 되었다면, checkpoint/cache와
원인을 확인한 뒤 container를 명시적으로 다시 시작하고 warm-up 및 왕복 검사를
처음부터 반복한다.

## 15. 최종 실행 체크리스트

- [ ] 공통 interface 변경사항이 커밋·push되어 Kanu와 로컬 commit이 같다.
- [ ] 지정된 070000 `pretrained_model` 필수 파일과 SHA256을 확인했다.
- [ ] config가 `multi_task_dit`/`flow_matching`, 7D·2-camera 계약을 만족한다.
- [ ] 학습 당시 task, 카메라 순서, state/action 의미를 확인했다.
- [ ] Kanu HF cache에 CLIP encoder/tokenizer가 준비되어 있다.
- [ ] 빈 GPU를 선택했고 Docker CUDA warm-up과 healthcheck가 통과했다.
- [ ] `.env`와 `fm_remote_deploy.yaml` contract 값이 정확히 같다.
- [ ] 로봇 없는 `ROUNDTRIP_ONLY=1` 검사가 반복 통과한다.
- [ ] 기록 observation에 대한 reference action parity를 확인했다.
- [ ] refill/p99 왕복 latency가 안전 timeout 안에 들어온다.
- [ ] start pose, joint limits, gripper 범위와 카메라 topic을 확인했다.
- [ ] 실제 로봇은 HOLD에서 시작하고 저속·비상정지 준비 후 실행한다.
