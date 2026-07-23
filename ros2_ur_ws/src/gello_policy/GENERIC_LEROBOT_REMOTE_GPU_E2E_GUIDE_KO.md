# 범용 LeRobot Policy Kanu 원격 추론·UR7e 실행 가이드

이 문서는 이 저장소를 처음 보는 사용자가 자신의 LeRobot checkpoint를 Kanu GPU
서버에서 불러와 추론하고, 로컬 로봇 PC에서 observation을 보내 `action[7]`을
받아 UR7e를 움직이기까지 따라 할 수 있도록 작성한 end-to-end 절차다.

> **실물 안전:** 이 코드는 policy가 반환한 목표 관절값으로 실제 UR7e를 움직인다.
> 반드시 로봇 없는 왕복 검사를 먼저 통과하고, 최초 실물 실행은 작업 공간을
> 비운 뒤 teach pendant와 비상정지에 즉시 접근할 수 있는 상태에서 저속으로 한다.
> `start_execution` 전까지 정책 자율 동작은 시작되지 않는다.

## 1. 이번에 구현한 것

고정된 네트워크 인터페이스를 사이에 두어 정책 계산과 로봇 제어를 분리했다.

```text
로컬 로봇 PC                                      Kanu GPU 서버
-------------                                     -------------
UR7e joint/gripper state                          LeRobot checkpoint
cam1/cam2 JPEG             SSH tunnel + gRPC      saved preprocessor
policy_leader_node  -------------------------->   policy.select_action() on CUDA
안전 검사·clamp    <---------------------------   saved postprocessor
UR7e/Robotiq 제어                 action[7]
```

Kanu 서버는 checkpoint의 `config.json`에서 policy type을 읽고 LeRobot
`get_policy_class()`와 `from_pretrained()`를 이용해 실제 class, weight,
preprocessor, postprocessor를 불러온다. 따라서 서버 전송 코드를 ACT, Diffusion,
Flow Matching 같은 특정 학습 objective에 맞춰 다시 만들지 않는다.

로컬은 기존 Diffusion 원격 실행에서 검증한 gRPC session, latest-only observation,
timeout 및 ROS2 안전 파이프라인을 재사용한다. 파일명이나 protobuf service 이름에
`diffusion`이 남아 있는 것은 기존 전송 계약과의 호환성을 위한 것이며, 공통
`policy-server`의 policy 계산이 Diffusion으로 제한된다는 뜻이 아니다.

## 2. 사용할 수 있는 policy checkpoint의 필수 조건

아래 조건을 **모두** 만족해야 adapter 없이 사용할 수 있다.

1. 이 배포 image에 설치된 LeRobot 버전에서 표준 Policy Class로 등록되어 있고
   `from_pretrained()`로 불러올 수 있어야 한다.
2. checkpoint 디렉터리에 `config.json`, model weight와 saved
   preprocessor/postprocessor 파일이 있어야 한다.
3. policy가 `reset()` 및 `select_action()` lifecycle을 지원해야 한다.
4. checkpoint의 feature 계약이 정확히 다음과 같아야 한다.

```text
입력 key                         kind       shape
observation.state                STATE      [7]
observation.images.cam1          VISUAL     [3, H, W]
observation.images.cam2          VISUAL     [3, H, W]

출력 key                         kind       shape
action                           ACTION     [7]
```

두 카메라는 RGB·CHW·3채널이어야 하며 shape도 서로 같아야 한다. 네트워크에서는
로컬 카메라 영상을 JPEG로 보내고 서버에서 RGB tensor로 복원한다.

현재 7차원 의미와 순서는 다음과 같다.

```text
observation.state = [ur_q1, ur_q2, ur_q3, ur_q4, ur_q5, ur_q6, grip_pos]
action            = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, grip_cmd]
```

관절 순서, 단위, gripper 범위, 카메라 key 또는 feature shape가 다르면 이
wrapper가 자동으로 변환하지 않는다. checkpoint와 데이터 계약에 맞춘 별도
adapter를 먼저 구현하고 검증해야 한다.

추가 확인 사항:

- 앞의 6개 state/action은 UR 관절각/목표 관절각(`rad`), 마지막 값은 이
  ROS 파이프라인 기준 gripper 위치/명령(`0.0=open`, `1.0=closed`)이다.
  학습 dataset도 같은 순서·단위·gripper 방향이어야 한다.
- `cam1`과 `cam2`의 물리 카메라 배치, 좌우 순서, 색상, 렌즈/해상도 및
  전처리가 학습 dataset과 같아야 한다. 단순히 tensor shape만 같다고 policy
  의미가 보존되는 것은 아니다.
- task-conditioned policy는 학습 당시와 **정확히 같은 task 문구**를 알아야 한다.
- policy가 CLIP 등 외부 encoder/tokenizer를 사용한다면 Kanu Hugging Face cache에
  해당 모델이 미리 저장되어 있어야 한다.
- LeRobot 및 주요 dependency 버전은 checkpoint를 만든 학습 환경과 맞춰야 한다.
- inference step, scheduler 및 `n_action_steps` override는 기존 offline 평가와
  같은 값부터 사용한다. 학습 step 수와 inference solver step 수는 같은 개념이
  아니다.
- 현재 고정 계약 밖의 policy는 서버 시작 단계에서 거부되는 것이 정상이다.

## 3. 코드와 데이터 위치

### 3.1 로컬 PC

개발 및 Git 기준 저장소:

```text
/home/laptop3/youngwoong_ws/gello_software
```

브랜치:

```text
feat/remote-gpu-server
```

주요 파일:

| 절대 경로 기준 하위 위치 | 역할 |
|---|---|
| `ros2_ur_ws/src/gello_policy/policy_server/lerobot_policy_wrapper.py` | 공통 Policy Class·processor load와 inference |
| `ros2_ur_ws/src/gello_policy/policy_server/remote_lerobot_server.py` | Kanu gRPC 서버, 계약 검사, CUDA warm-up |
| `ros2_ur_ws/src/gello_policy/deploy/remote_diffusion/Dockerfile` | 범용 서버 image |
| `ros2_ur_ws/src/gello_policy/deploy/remote_diffusion/compose.yaml` | `policy-server` Compose service |
| `ros2_ur_ws/src/gello_policy/config/fm_remote_deploy.yaml` | 검증된 FM용 로컬 안전 설정 예시 |
| `ros2_ur_ws/setup_remote_client_venv.sh` | 로컬 gRPC 1.74 환경 생성 |
| `ros2_ur_ws/run_ur7e_diffusion_remote.sh` | SSH tunnel, 왕복 검사, 실물 ROS launch |

### 3.2 Kanu

코드 checkout:

```text
/home/junhyeong/workspace/youngwoong/gello_software
```

checkpoint, dataset, Hugging Face cache는 Git 저장소 밖에 둔다. 현재 검증한 FM
checkpoint 예시는 다음 위치다.

```text
/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/checkpoints/flow_matching_cube_in_cup_checkpoints/070000/pretrained_model
```

Kanu 런타임 `.env` 권장 위치:

```text
/home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/<policy-name>.env
```

이 문서의 `my_policy`, `0123456789abcdef...` 같은 예시값은 사용자 환경의 실제
값으로 바꾼다. 로컬과 Kanu의 Linux 사용자명 또는 workspace 경로가 이 문서와
다르면 두 머신에서 각각 절대 경로를 다시 확인한다. Bash 명령에서는 꺾쇠
자리표시자를 사용하지 않는다. `<`가 입력 redirection으로 해석될 수 있기 때문이다.

## 4. 앞으로의 개발·배포 운영 원칙

1. 개발과 코드 수정은 로컬
   `/home/laptop3/youngwoong_ws/gello_software`에서만 한다.
2. 로컬에서 검토·테스트가 끝난 수정만 `feat/remote-gpu-server`에 commit하고
   GitHub `origin`으로 push한다.
3. Kanu의 `/home/junhyeong/workspace/youngwoong/gello_software`는 같은 branch의
   검증된 특정 commit SHA를 checkout하여 실행한다.
4. Kanu에서 평상시 코드를 직접 수정하지 않는다. 긴급 수정이 필요했다면 patch나
   commit으로 로컬에 회수하여 로컬에서 검증한 뒤 다시 push한다.
5. checkpoint, dataset, Hugging Face cache만 기존 sync/download 방식으로
   Git 저장소 밖에서 관리한다.
6. 전체 workspace를 `rsync`로 덮어쓰지 않는다. 코드 버전은 Git SHA로, 대용량
   artifact는 별도 경로와 manifest로 식별한다.
7. 이 feature branch를 default branch에 merge하는 작업은 별도 선배들의 검토와
   명시적인 승인을 거친다.

## 5. 로컬에서 검증된 코드를 GitHub에 올리기

다음 명령의 입력 위치는 로컬 PC의
`/home/laptop3/youngwoong_ws/gello_software`다.

```bash
cd /home/laptop3/youngwoong_ws/gello_software
```

```bash
git status --short --branch
```

```bash
git branch --show-current
```

현재 branch가 `feat/remote-gpu-server`인지 확인하고, 검증된 변경만 commit한 뒤
해당 branch로 push한다.

```bash
git push origin feat/remote-gpu-server
```

배포할 SHA를 기록한다.

```bash
export DEPLOY_SHA="$(git rev-parse HEAD)" && printf '%s\n' "$DEPLOY_SHA"
```

출력된 40자리 SHA를 Kanu 운영자에게 전달한다.

## 6. Kanu 코드 checkout을 정확한 SHA로 맞추기

다음 명령의 입력 위치는 Kanu다. 먼저 기존 수정 여부를 확인한다.

```bash
cd /home/junhyeong/workspace/youngwoong/gello_software
```

```bash
git status --short --branch
```

출력이 깨끗할 때만 다음을 진행한다. Kanu의 미회수 수정이 있으면 reset하거나
덮어쓰지 말고 먼저 patch로 로컬에 회수한다.

```bash
test -z "$(git status --porcelain)" || { echo 'KANU WORKTREE DIRTY - STOP'; false; }
```

```bash
git fetch origin feat/remote-gpu-server
```

```bash
export DEPLOY_SHA=0123456789abcdef0123456789abcdef01234567
```

위 예시 SHA를 로컬에서 전달받은 실제 40자리 SHA로 바꾼 뒤 checkout한다.

```bash
git checkout --detach "$DEPLOY_SHA"
```

```bash
git rev-parse HEAD
```

로컬에서 기록한 `DEPLOY_SHA`와 정확히 같아야 한다. 새 clone이 필요한 경우에는
Kanu의 `/home/junhyeong/workspace/youngwoong`에서 다음처럼 만든다.

```bash
cd /home/junhyeong/workspace/youngwoong
```

```bash
git clone --branch feat/remote-gpu-server --single-branch https://github.com/Bigenlight/gello_software.git gello_software
```

## 7. Kanu에 checkpoint와 HF cache 준비

`CHECKPOINT_DIR`는 `pretrained_model` 디렉터리 자체를 가리켜야 한다.

```bash
export CHECKPOINT_DIR=/absolute/path/to/pretrained_model
```

필수 파일을 확인한다.

```bash
find "$CHECKPOINT_DIR" -maxdepth 2 -type f -printf '%P\n' | sort
```

최소한 `config.json`, model weight, `policy_preprocessor.json`,
`policy_postprocessor.json`과 processor가 참조하는 state 파일이 보여야 한다.

manifest 계산 명령의 입력 위치는 Kanu의 `gello_policy` package 루트다.

```bash
cd /home/junhyeong/workspace/youngwoong/gello_software/ros2_ur_ws/src/gello_policy
```

```bash
python3 -m policy_server.checkpoint_identity "$CHECKPOINT_DIR"
```

출력의 `policy_type`과 `revision`을 기록한다. `revision`은 단일 weight 파일
해시가 아니라 config, weight, preprocessor, postprocessor를 함께 식별하는
manifest SHA-256이다.

학습 환경의 실제 dependency를 기록한다. 아래 경로는 학습에 사용한 가상환경의
Python 절대 경로로 바꾼다.

```bash
/absolute/path/to/training/venv/bin/python -c "import importlib.metadata as m; names=['lerobot','torch','transformers','huggingface-hub','tokenizers','safetensors','diffusers','accelerate']; print({n:(m.version(n) if any((d.metadata.get('Name') or '').lower()==n.lower() for d in m.distributions()) else 'NOT INSTALLED') for n in names})"
```

학습 환경이 editable Git checkout의 LeRobot을 사용했다면 package 버전뿐 아니라
그 checkout의 commit SHA와 clean 여부도 기록한다. 배포 image의
`LEROBOT_COMMIT`은 이 값과 호환되는 검증된 commit이어야 한다. 버전 불일치를
경고만 무시하고 실물 로봇을 실행하지 않는다.

외부 encoder를 사용하는 policy는 학습 환경에서 사용한 cache 루트를 찾는다.
Kanu 기본 예시는 다음과 같다.

```text
/home/junhyeong/.cache/huggingface
```

```bash
find /home/junhyeong/.cache/huggingface/hub -maxdepth 1 -type d -name 'models--*' -printf '%f\n' | sort
```

필요한 encoder/tokenizer가 없다면 container 시작 전에 네트워크가 가능한 학습
환경에서 한 번 내려받는다. container는 이 cache의 `hub`만 read-only/offline으로
사용하며 호스트 token 파일은 전달받지 않는다.

## 8. Kanu 서버용 `.env` 작성

runtime 디렉터리를 만든다.

```bash
mkdir -p /home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server
```

이번 배포를 식별할 영문·숫자·`_`·`-` 기반 이름을 정한다.

```bash
export POLICY_NAME=my_policy
```

예시를 복사한다.

```bash
cp /home/junhyeong/workspace/youngwoong/gello_software/ros2_ur_ws/src/gello_policy/deploy/remote_diffusion/.env.example "/home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/${POLICY_NAME}.env"
```

```bash
chmod 600 "/home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/${POLICY_NAME}.env"
```

파일을 열어 최소한 다음 항목을 자신의 checkpoint에 맞춘다.

```bash
nano "/home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/${POLICY_NAME}.env"
```

```dotenv
CHECKPOINT_DIR=/absolute/path/to/pretrained_model
HF_CACHE_DIR=/home/junhyeong/.cache/huggingface
GPU_DEVICE=1
INFERENCE_BIND_IP=127.0.0.1
POLICY_INFERENCE_PORT=50052
IMAGE_TAG=my-policy-abcdef0
POLICY_CONTAINER_NAME=gello-remote-policy-my-policy
LEROBOT_COMMIT=8a74e0ac6d01706d67fddfed682a09d694d9c8c0
LEROBOT_EXTRAS=multi-task-dit
TRANSFORMERS_VERSION=5.13.0
HUGGINGFACE_HUB_VERSION=1.22.0
EXPECTED_POLICY_TYPE=multi_task_dit
EXPECTED_CHECKPOINT_REVISION=sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
POLICY_TASK=put the cube in the cup
POLICY_TASK_MODE=required
POLICY_WARMUP_STATE=[3.106,-1.817,1.653,-1.618,-1.628,-3.195,0.0]
POLICY_CONFIG_OVERRIDES={"num_integration_steps":10,"n_action_steps":24}
EXTERNAL_IMAGE_SIZE=native
```

위 값은 검증된 cube Flow Matching 예시이며 다른 checkpoint에 그대로 사용하는
기본값이 아니다. 앞 단계에서 기록한 자신의 policy type, manifest, dependency,
task, warm-up state와 offline inference 설정으로 각각 바꾼다.

설정 원칙:

- task를 쓰는 policy는 `POLICY_TASK_MODE=required`, task가 없는 policy는
  `POLICY_TASK_MODE=disabled`로 명시한다. 확실하지 않다는 이유로 실물 배포에서
  `auto`에 의존하지 말고, checkpoint의 saved preprocessor와 학습 코드를
  확인한다.
- `POLICY_WARMUP_STATE`는 학습 분포 안의 안전한 7D state여야 한다.
- `LEROBOT_EXTRAS`는 LeRobot 설치 extra 이름을 쉼표로 연결한 값이다. policy가
  extra를 요구하지 않으면 빈 값으로 둘 수 있다. extra 또는 dependency pin이
  달라지면 image를 다시 build한다.
- 모델 내부가 resize를 담당하면 `native`, 서버가 먼저 고정 resize해야 하면
  학습 해상도에 맞춘 `HxW`를 쓴다.
- override는 기존 offline inference와 같은 값부터 시작한다. 예를 들어 검증된
  cube FM은 `{"num_integration_steps":10,"n_action_steps":24}`다.
- `POLICY_CONFIG_OVERRIDES`는 checkpoint config에 실제로 존재하는 최상위
  field만 허용한다. offline reference action과 비교하지 않은 임의 override를
  실물 실행에 사용하지 않는다.
- `.env`에는 token이나 비밀번호를 넣지 않으며 Git에 commit하지 않는다.

## 9. Kanu Docker image build와 서버 시작

아래 명령의 입력 위치는 Kanu의 `gello_policy` package 루트다.

```bash
cd /home/junhyeong/workspace/youngwoong/gello_software/ros2_ur_ws/src/gello_policy
```

환경변수 두 개를 지정한다.

```bash
export POLICY_NAME=my_policy
```

```bash
export POLICY_ENV="/home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/${POLICY_NAME}.env"
```

```bash
export POLICY_COMPOSE=deploy/remote_diffusion/compose.yaml
```

공유 GPU 상태를 확인하고 비어 있는 GPU 번호를 `.env`의 `GPU_DEVICE`로 선택한다.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

Compose 구문과 최종 image 이름을 확인한다.

```bash
docker compose -f "$POLICY_COMPOSE" --env-file "$POLICY_ENV" --profile generic config --quiet
```

```bash
docker compose -f "$POLICY_COMPOSE" --env-file "$POLICY_ENV" --profile generic config --images
```

최초 실행 또는 dependency/code가 바뀐 경우 build한다. Docker layer cache가
유효하면 이미 받은 base image와 package layer를 재사용한다.

```bash
docker compose -f "$POLICY_COMPOSE" --env-file "$POLICY_ENV" --profile generic build policy-server
```

반드시 service 이름을 같은 줄 끝에 포함해 공통 `policy-server`만 시작한다.
bare `docker compose up`은 기존 전용 `diffusion-server`까지 대상으로 삼을 수
있으므로 사용하지 않는다.

```bash
docker compose -f "$POLICY_COMPOSE" --env-file "$POLICY_ENV" --profile generic up -d --no-build --force-recreate policy-server
```

로그를 확인한다.

```bash
docker compose -f "$POLICY_COMPOSE" --env-file "$POLICY_ENV" --profile generic logs -f policy-server
```

정상 로그에는 checkpoint/processor load, CUDA warm-up, queue reset과
`ready at 0.0.0.0:50051`이 나타난다. 로그 follow는 `Ctrl+C`로 빠져나와도
background container는 계속 실행된다.

로그의 `policy_type`, input/output feature, `task_required`,
`checkpoint_revision`, sampling method/step을 기록한 값과 대조한다. CLIP의 전체
checkpoint를 vision/text class가 각각 읽으면서 반대쪽 weight를
`UNEXPECTED`로 보고하는 것과, **자신의 policy checkpoint 자체에서**
`Missing key(s)`/`Unexpected key(s)`가 발생하는 것은 다르다. 후자는 dependency
불일치 가능성이 있으므로 중단한다.

현재 Compose service의 container ID를 통해 health를 확인한다.

```bash
docker inspect "$(docker compose -f "$POLICY_COMPOSE" --env-file "$POLICY_ENV" --profile generic ps -q policy-server)" --format 'image={{.Config.Image}} status={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} exit={{.State.ExitCode}}'
```

`status=running health=healthy`여야 한다. Kanu host에서는 policy port를
loopback에만 열어 두고 외부에 직접 공개하지 않는다.

container 내부 Health RPC를 직접 확인한다.

```bash
docker compose -f "$POLICY_COMPOSE" --env-file "$POLICY_ENV" --profile generic exec policy-server python /app/healthcheck.py
```

종료 코드가 0이어야 한다. 이어서 합성 black image와 warm-up state로 Kanu 내부
gRPC inference를 한 번 실행한다.

```bash
docker compose -f "$POLICY_COMPOSE" --env-file "$POLICY_ENV" --profile generic exec -e SMOKE_TARGET=127.0.0.1:50051 policy-server python /app/smoke_client.py
```

출력의 action이 유한한 7개 값이고, `model_id`, `checkpoint_revision`,
`scheduler`, inference/action step이 앞서 기록한 계약과 일치해야 한다. smoke
후에도 Health RPC가 정상인지 다시 확인한다.

```bash
docker compose -f "$POLICY_COMPOSE" --env-file "$POLICY_ENV" --profile generic exec policy-server python /app/healthcheck.py
```

## 10. 로컬 PC 준비

다음 명령의 입력 위치는 로컬 PC의 ROS workspace다.

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
```

Kanu와 동일한 코드 SHA인지 확인한다.

```bash
git rev-parse HEAD
```

최초 한 번 또는 lock 변경 후 gRPC client 환경을 만든다.

```bash
./setup_remote_client_venv.sh
```

ROS package를 빌드한다.

```bash
./build_ur7e.sh
```

SSH alias가 동작하는지 확인한다. 이 문서에서는 `~/.ssh/config`의 Kanu alias가
`kanu`라고 가정한다.

```bash
ssh -o ConnectTimeout=5 kanu 'echo KANU_SSH_OK'
```

## 11. 로컬 runtime YAML 만들기

학습 시 사용한 start pose, joint 안전 범위, task contract를 반영한 YAML이
필요하다. 새 policy가 검증된 cube FM과 동일한 UR7e 데이터 계약 및 안전 범위를
사용한다면 저장소 예시를 runtime 영역으로 복사한다.

```bash
mkdir -p /home/laptop3/youngwoong_ws/runtime/remote-gpu-server
```

Kanu `.env`에 사용한 이름과 같은 배포 이름을 정한다.

```bash
export POLICY_NAME=my_policy
```

```bash
cp /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws/src/gello_policy/config/fm_remote_deploy.yaml "/home/laptop3/youngwoong_ws/runtime/remote-gpu-server/${POLICY_NAME}-deploy.yaml"
```

runtime YAML에서 다음 값을 서버 `.env` 및 실제 checkpoint와 일치시킨다.

- `expected_model_id`
- `expected_checkpoint_revision`
- `expected_scheduler`
- `expected_inference_steps`
- `expected_action_steps`
- `expected_resize_height`, `expected_resize_width`
- `start_pose`, `start_gripper`
- 실제 데이터 envelope에서 검증한 `joint_limits_lo`, `joint_limits_hi`
- 측정한 refill p99에 맞는 timeout
- 실제 compressed image 토픽인 `cam1_topic`, `cam2_topic`
- 실제 JPEG의 `camera_height`, `camera_width`, 허용 가능한
  `max_camera_skew_s`

manifest placeholder만 바꾸는 예시는 다음과 같다.

```bash
export CHECKPOINT_REVISION=sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

예시값을 Kanu에서 계산한 실제 manifest로 바꾼 뒤 runtime YAML에 반영한다.

```bash
sed -i "s#REPLACE_WITH_SHA256_MANIFEST#${CHECKPOINT_REVISION}#" "/home/laptop3/youngwoong_ws/runtime/remote-gpu-server/${POLICY_NAME}-deploy.yaml"
```

저장소의 원본 YAML을 직접 수정하지 않는다. 다른 robot joint envelope에서 학습한
policy에 기존 joint limit를 무비판적으로 재사용하지 않는다.

이 로컬 노드는 기본적으로
`/cam1/cam1/color/image_raw/compressed`,
`/cam2/cam2/color/image_raw/compressed`, `/joint_states`,
`/robotiq_gripper/position_percent`를 구독한다. 카메라 드라이버가 raw
`sensor_msgs/Image`만 발행하면 그대로 사용할 수 없으므로 compressed transport를
준비하거나 adapter가 필요하다. `camera_height`/`camera_width`는 메시지에 실려
서버 계약 검사에 사용되는 메타데이터이므로 실제 전송 JPEG와 일치시킨다.

## 12. 실물 로봇 없이 실제 원격 왕복 확인

Kanu container가 healthy인 상태에서 로컬 ROS workspace에서 실행한다.

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
```

black JPEG 두 장과 7D state를 보내 실제 checkpoint에서 7D action이 돌아오는지
검사한다.

```bash
SSH_HOST=kanu REMOTE_GRPC_PORT=50052 LOCAL_GRPC_PORT=50052 ROUNDTRIP_TIMEOUT_S=30 ROUNDTRIP_IMAGE_HEIGHT=720 ROUNDTRIP_IMAGE_WIDTH=1280 ROUNDTRIP_STATE='[3.106,-1.817,1.653,-1.618,-1.628,-3.195,0.0]' ROUNDTRIP_ONLY=1 ./run_ur7e_diffusion_remote.sh
```

`ROUNDTRIP_ONLY PASS`와 유한한 7개 action, 예상 model/checkpoint/scheduler/step
계약이 모두 일치해야 한다. 이 검사는 로봇이나 ROS controller를 시작하지 않는다.
위 해상도와 state도 cube FM 예시이므로 자신의 카메라 해상도와 학습 분포 안의
7D state로 바꾼다.
black image 검사는 연결·계약 검사일 뿐 policy 품질 검증이 아니다. 가능하면 학습
환경에서 기록한 동일 observation을 offline과 원격 양쪽에 넣어 action/reference
parity를 확인한 뒤 실물로 진행한다.

## 13. 실물 UR7e 실행

사전 조건:

- Kanu container가 healthy이고 실제 왕복 검사가 통과했다.
- cam1/cam2 topic, `/joint_states`, gripper feedback이 실제로 들어온다.
- UR7e External Control 설정, robot IP, Robotiq 연결과 calibration을 확인했다.
- runtime YAML의 start pose와 현재 로봇 자세가 안전하다.
- runtime YAML의 `auto_start_on_stream`이 안전 기본값 `false`다.
- 작업 공간을 비우고 pendant·비상정지를 잡은 운영자가 있다.

로컬에서는 세 terminal을 사용한다.

### 13.1 Terminal A: 두 카메라 시작·배치 확인

로컬 ROS workspace로 이동한다.

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
```

두 RealSense의 serial, 이름, 해상도는 학습 dataset을 기록할 때와 같아야 한다.
저장소 script의 기본 serial이 실제 장치와 일치하는지 먼저
`launch_cameras.sh` 상단에서 확인한다. 필요하면 환경변수로 실제 serial을
지정하고 두 영상을 눈으로 확인한다.

```bash
CAM1_SERIAL=147122072740 CAM2_SERIAL=243222072700 CAM1_NAME=cam1 CAM2_NAME=cam2 COLOR_PROFILE=1280x720x30 VIEW=true ./launch_cameras.sh
```

예시 serial과 profile을 자신의 학습 장비 값으로 바꾼다. viewer에서 cam1/cam2가
뒤바뀌지 않았고 색상·시야가 학습 때와 같은지 확인한 후 이 terminal을 계속 열어
둔다. 카메라 종료는 이 terminal에서 `Ctrl+C`로 수행한다.

### 13.2 Terminal B: UR7e launch와 SSH tunnel 시작

새 terminal에서 로컬 ROS workspace로 이동한다.

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
```

실제 로봇 IP, calibration 파일과 배포 이름을 지정한다.

```bash
export POLICY_NAME=my_policy ROBOT_IP=192.168.10.11 CALIB=/absolute/path/to/ur7e_calibration.yaml
```

```bash
SSH_HOST=kanu REMOTE_GRPC_PORT=50052 LOCAL_GRPC_PORT=50052 ROBOT_IP="$ROBOT_IP" HEADLESS=false CALIB="$CALIB" ./run_ur7e_diffusion_remote.sh params_file:="/home/laptop3/youngwoong_ws/runtime/remote-gpu-server/${POLICY_NAME}-deploy.yaml" launch_rviz:=true
```

이 script가 SSH tunnel을 만들고 gRPC preflight를 통과한 뒤 ROS launch를 시작한다.
handshake 동안 policy leader는 YAML의 start pose를 HOLD하며 자율 추론을 시작하지
않는다. pendant에서 External Control 프로그램이 필요한 구성이라면 launch 로그의
안내에 맞춰 프로그램을 시작하고 move-to-start handshake가 성공할 때까지
관찰한다.

### 13.3 Terminal C: observation과 handshake 확인 후 실행 허가

새 로컬 terminal에서 ROS 환경을 source한다.

```bash
set +u; source /opt/ros/humble/setup.bash; source /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws/install/setup.bash; set -u
```

필수 observation topic이 실제로 발행되는지 각각 확인한다. 각 명령은 메시지를
한 번 출력한 뒤 종료한다.

```bash
ros2 topic echo /joint_states --once
```

```bash
ros2 topic echo /cam1/cam1/color/image_raw/compressed --once
```

```bash
ros2 topic echo /cam2/cam2/color/image_raw/compressed --once
```

```bash
ros2 topic echo /robotiq_gripper/position_percent --once
```

그 다음 Terminal B의 로그에서 move-to-start handshake와 controller 전환이
성공했고 leader가 HOLD 상태인지 확인한다. 실제 관절이 runtime YAML의
`start_pose`에서 joint별 약 0.1rad 이내이고 주변이 안전할 때만 자율 실행을
허용한다.

```bash
ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
```

첫 실행은 즉시 정지 가능한 짧은 구간과 낮은 robot speed slider로 수행한다.
timeout, stale observation, 계약 불일치 또는 server 오류가 발생하면 leader가
FAULT로 전환되고 stale target을 계속 보내지 않아야 한다.

## 14. 종료

정상적인 일시정지는 먼저 별도 로컬 terminal에서 HOLD를 요청한다. HOLD는 현재
live 관절 자세와 마지막 gripper 명령을 유지하고 policy inference를 중단한다.

```bash
ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger
```

HOLD 전환과 로봇 정지를 확인한 뒤 로컬 실행 terminal에서 `Ctrl+C`를 눌러 ROS
launch와 runner가 만든 SSH tunnel을 종료한다. 위험 동작에서는 service나
`Ctrl+C`보다 teach pendant 정지·비상정지를 우선한다. 물체를 잡은 상태라면
gripper node를 종료했을 때의 장비 동작을 가정하지 말고, 먼저 물체를 안전하게
내려놓는 별도 절차를 수행한다. Kanu container lifecycle은 서버 운영자가 별도로
관리한다.

Kanu에서 해당 policy service만 멈춘다.

```bash
cd /home/junhyeong/workspace/youngwoong/gello_software/ros2_ur_ws/src/gello_policy
```

```bash
export POLICY_NAME=my_policy
```

```bash
docker compose -f deploy/remote_diffusion/compose.yaml --env-file "/home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/${POLICY_NAME}.env" --profile generic stop policy-server
```

## 15. 자주 발생하는 문제

### `checkpoint identity mismatch`

`CHECKPOINT_DIR`가 다른 checkpoint를 가리키거나 processor 파일이 바뀐 것이다.
현재 디렉터리에서 manifest를 다시 계산하고 서버 `.env`와 로컬 runtime YAML의
revision을 모두 같은 값으로 갱신한다. 수동으로 임의 SHA를 넣지 않는다.

### `policy type` 또는 feature contract 오류

checkpoint의 `config.json`에서 type, feature kind/key/shape를 확인한다.
`state[7] + cam1 + cam2 -> action[7]`이 아니면 현재 wrapper의 직접 지원 대상이
아니므로 adapter가 필요하다.

### Hugging Face offline load 오류

`HF_CACHE_DIR`가 cache root인지, 그 아래 `hub/models--...`에 필요한 공개
encoder/tokenizer가 완전히 저장되어 있는지 확인한다. host token 권한을
container에 열어 해결하지 않는다.

### checkpoint weight의 대량 `Missing key(s)` 또는 `Unexpected key(s)`

학습 환경과 배포 image의 LeRobot, Transformers, Hugging Face Hub 버전을
비교한다. CLIP 전체 checkpoint에서 vision class가 text weight를 무시하거나 text
class가 vision weight를 무시한다는 load report와, **로컬 policy checkpoint
자체의 key mismatch**는 구분해야 한다.

### SSH tunnel 또는 gRPC preflight 실패

Kanu container가 healthy인지, host의 `127.0.0.1:50052`가 열렸는지, SSH alias가
정확한지 확인한다. 로컬에서는 `./setup_remote_client_venv.sh`를 다시 실행해
grpcio 1.74.0을 사용한다.

### action은 오지만 로봇이 시작되지 않음

이것은 안전상 정상일 수 있다. handshake, start pose 정렬, observation freshness를
확인하고 운영자가 명시적으로 `/policy_leader_node/start_execution`을 호출해야
한다.

### 새로운 policy family가 image에서 load되지 않음

Policy Class가 LeRobot에 등록되어 있어도 추가 dependency가 image에 없을 수 있다.
필요한 LeRobot extra/dependency를 Dockerfile의 허용된 설치 경로에 추가하고
로컬 테스트 후 새 SHA를 push한 다음 Kanu에서 image를 다시 build한다.

## 16. 배포 기록에 반드시 남길 값

실험마다 다음을 함께 기록하면 재현과 문제 회수가 가능하다.

- 로컬 및 Kanu Git commit SHA
- checkpoint의 절대 경로와 manifest revision
- policy type, task, scheduler/objective, inference step, action step
- LeRobot 및 주요 dependency 버전
- Docker image tag와 image ID
- GPU UUID
- runtime `.env`와 YAML의 비밀정보를 제외한 설정
- 왕복/refill latency와 실행 일시

이 구조를 따르면 코드는 Git SHA로, 학습 artifact는 manifest로, 실행 설정은
머신별 runtime 파일로 분리되어 Kanu의 코드를 직접 수정하지 않고도 같은
LeRobot 정책을 반복 배포할 수 있다.
