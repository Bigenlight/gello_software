# Cube-in-Cup Flow Matching 070000 체크포인트 실물 UR7e 실행 Runbook

이 문서는 이 저장소를 처음 보는 운영자가 이미 Kanu에 저장된 cube-in-cup
Flow Matching 070000 체크포인트를 GPU에서 추론하고, `laptop3`에서 실물
UR7e와 Robotiq gripper를 제어하기 위한 전용 절차다. 다른 checkpoint를 위한
범용 설명은 `GENERIC_LEROBOT_REMOTE_GPU_E2E_GUIDE_KO.md`를 참고한다.

> **실물 안전 경고:** `start_execution`은 policy 자율 동작을 막는 gate이지만,
> 로컬 launch의 move-to-start 과정 자체는 로봇을 움직일 수 있다. 로봇 launch
> 전부터 작업 공간을 비우고, teach pendant와 비상정지에 즉시 접근하며,
> UR 속도 slider를 낮춘 상태로 운영자 두 명이 확인한다. 이상 동작 시 ROS
> service보다 pendant 정지 또는 비상정지를 우선한다.

## 1. 이번 실행에서 사용하는 고정 값

| 항목 | 값 |
|---|---|
| Policy | `multi_task_dit` |
| Objective/scheduler | `flow_matching` / `multi_task_dit:flow_matching` |
| Checkpoint | 070000 `pretrained_model` |
| Task | `put the cube in the cup` |
| 입력 | `observation.state[7] + cam1 RGB + cam2 RGB` |
| 출력 | `action[7]` |
| Action chunk | 24 |
| Euler inference step | 10 |
| Kanu host port | `127.0.0.1:50052` |
| 로컬 forwarded port | `127.0.0.1:50052` |
| Manifest | `sha256:93993806a4d72c993fb5759fc2cdc44ab833bf314acfa9700b83a9a75bc518d6` |
| LeRobot commit | `8a74e0ac6d01706d67fddfed682a09d694d9c8c0` |
| Transformers | `5.13.0` |
| Hugging Face Hub | `1.22.0` |

Kanu checkpoint 절대 경로:

```text
/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/checkpoints/flow_matching_cube_in_cup_checkpoints/070000/pretrained_model
```

코드 위치:

```text
로컬: /home/laptop3/youngwoong_ws/gello_software
Kanu: /home/junhyeong/workspace/youngwoong/gello_software
브랜치: feat/remote-gpu-server
```

앞의 6개 state/action은 UR 관절 순서의 radian 값이고 마지막 값은 gripper
명령이다. 이 배포에서는 `0.0=open`, `1.0=closed`다. cam1은 전체 scene,
cam2는 작업 공간 close-up이어야 한다.

검증 당시 480회 원격 요청이 모두 성공했고, refill inference p99는 약
95.33 ms, refill 전체 서버 처리 p99는 약 114.36 ms, refill 왕복 p99는 약
273.02 ms였다. 이는 현재 YAML의 `act_timeout_s=0.6`,
`obs_timeout_s=0.7`, `staleness_timeout_s=0.8` 안에 들어온다.

## 2. 시작 전 사람이 반드시 확인할 값

다음 값은 현장 장비를 보고 확인해야 하며 추측해서 입력하면 안 된다.

- UR7e 실제 IP: 예시 기본값은 `192.168.10.11`이지만 pendant와 로컬
  네트워크 설정에서 재확인한다.
- UR calibration YAML 절대 경로: 해당 로봇에서 추출한 파일인지 확인한다.
  파일이 없다면 임의 파일을 지정하지 말고 담당자에게 확인한다.
- 두 RealSense serial: 저장소 기본 예시는 cam1 `147122072740`, cam2
  `243222072700`이다. 실제 학습 촬영에 사용한 장치와 일치해야 한다.
- cam1/cam2 물리 배치와 순서: cam1=scene, cam2=close-up인지 화면으로 본다.
- UR External Control 프로그램, 로컬 PC IP, Robotiq tool communication 설정.
- cup, cube, 카메라, 로봇 base의 실제 배치가 학습 데이터 수집 때와 같은지.
- 시작 자세 `[3.106, -1.817, 1.653, -1.618, -1.628, -3.195]`가 현재
  공간에서 충돌 없이 도달 가능한지.

하나라도 확인할 수 없으면 실물 실행 절차를 중단한다.

## 3. 로컬과 Kanu 코드를 같은 Git SHA로 맞추기

개발과 Git 기준은 로컬 저장소다. Kanu에서는 코드를 수정하지 않고 검증된
`feat/remote-gpu-server` commit을 받아 실행한다.

### 3.1 로컬 PC에서 배포 SHA 확인

입력 위치: 로컬 PC `/home/laptop3/youngwoong_ws/gello_software`

```bash
cd /home/laptop3/youngwoong_ws/gello_software
```

```bash
git status --short --branch
```

```bash
git branch --show-current
```

```bash
git rev-parse HEAD
```

branch가 `feat/remote-gpu-server`이고 worktree가 깨끗해야 한다. 출력된 40자리
SHA를 `검토·검증된 배포 SHA`로 기록해 Kanu에서도 똑같이 사용한다. 단순히
branch의 최신 commit이라는 이유만으로 실물 로봇에 배포하지 않는다.

### 3.2 Kanu에서 같은 SHA checkout

입력 위치: Kanu `/home/junhyeong/workspace/youngwoong/gello_software`

```bash
cd /home/junhyeong/workspace/youngwoong/gello_software
```

```bash
test -z "$(git status --porcelain)" || { echo 'KANU WORKTREE DIRTY - STOP'; false; }
```

```bash
git fetch origin feat/remote-gpu-server
```

fetch한 원격 branch tip을 배포할 특정 SHA로 고정하고 출력값을 기록한다.

```bash
export DEPLOY_SHA="$(git rev-parse origin/feat/remote-gpu-server)" && printf '%s\n' "$DEPLOY_SHA"
```

```bash
git checkout --detach "$DEPLOY_SHA"
```

```bash
git rev-parse HEAD
```

두 머신의 출력이 정확히 같아야 한다. Kanu의 checkpoint, dataset 및 HF cache는
Git 밖에서 관리하므로 checkout 대상에 포함하지 않는다.

## 4. Kanu checkpoint와 manifest 확인

입력 위치: Kanu

```bash
export CHECKPOINT_DIR=/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/checkpoints/flow_matching_cube_in_cup_checkpoints/070000/pretrained_model
```

```bash
find "$CHECKPOINT_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
```

다음 파일들이 보여야 한다.

```text
config.json
model.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
policy_preprocessor.json
policy_preprocessor_step_4_normalizer_processor.safetensors
train_config.json
```

입력 위치를 Kanu의 `gello_policy` package root로 바꾼다.

```bash
cd /home/junhyeong/workspace/youngwoong/gello_software/ros2_ur_ws/src/gello_policy
```

```bash
python3 -m policy_server.checkpoint_identity "$CHECKPOINT_DIR"
```

출력이 `policy_type: multi_task_dit`이고 revision이 아래 값과 정확히 같아야 한다.

```text
sha256:93993806a4d72c993fb5759fc2cdc44ab833bf314acfa9700b83a9a75bc518d6
```

다르면 파일이 바뀐 것이므로 이 runbook의 설정으로 실행하지 않는다.

checkpoint의 고정 contract도 자동 확인한다.

```bash
python3 -c 'import json,os; c=json.load(open(os.path.join(os.environ["CHECKPOINT_DIR"],"config.json"))); assert c["type"]=="multi_task_dit"; assert c["objective"]=="flow_matching"; assert c["n_action_steps"]==24; assert c["n_obs_steps"]==2; assert c["horizon"]==32; assert c["input_features"]["observation.state"]["shape"]==[7]; assert c["input_features"]["observation.images.cam1"]["shape"]==[3,720,1280]; assert c["input_features"]["observation.images.cam2"]["shape"]==[3,720,1280]; assert c["output_features"]["action"]["shape"]==[7]; print("CHECKPOINT CONTRACT OK")'
```

`CHECKPOINT CONTRACT OK`가 출력되지 않으면 중단한다.

CLIP cache를 확인한다.

```bash
test -d /home/junhyeong/.cache/huggingface/hub/models--openai--clip-vit-base-patch16 && echo 'CLIP CACHE OK' || echo 'CLIP CACHE MISSING - STOP'
```

## 5. Kanu 서버 runtime `.env` 준비

`.env`는 Git에 넣지 않는다. 입력 위치: Kanu

```bash
mkdir -p /home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server
```

```bash
cp /home/junhyeong/workspace/youngwoong/gello_software/ros2_ur_ws/src/gello_policy/deploy/remote_diffusion/.env.example /home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/fm-070000.env
```

```bash
chmod 600 /home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/fm-070000.env
```

GPU 사용량을 확인하고 비어 있는 GPU 번호를 고른다. 이전 검증에서 GPU 1을
사용했더라도 현재 점유 상태가 달라질 수 있으므로 고정해서 재사용하지 않는다.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

선택한 GPU에 다른 사용자의 process가 있거나 점유 여부가 불명확하면 임의로
container를 종료하지 말고 서버 담당자에게 확인한다.

파일을 연다.

```bash
nano /home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/fm-070000.env
```

파일의 다음 항목을 아래 실제 값으로 맞춘다. `GPU_DEVICE`만 방금 선택한 번호를
사용한다. 기존 Diffusion용 변수는 이 `policy-server` 실행에서 사용되지 않는다.

```dotenv
CHECKPOINT_DIR=/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/checkpoints/flow_matching_cube_in_cup_checkpoints/070000/pretrained_model
HF_CACHE_DIR=/home/junhyeong/.cache/huggingface
GPU_DEVICE=선택한_GPU_번호
INFERENCE_BIND_IP=127.0.0.1
POLICY_INFERENCE_PORT=50052
IMAGE_TAG=fm-070000
POLICY_CONTAINER_NAME=gello-remote-policy-fm-070000
LEROBOT_COMMIT=8a74e0ac6d01706d67fddfed682a09d694d9c8c0
LEROBOT_EXTRAS=multi-task-dit
TRANSFORMERS_VERSION=5.13.0
HUGGINGFACE_HUB_VERSION=1.22.0
EXPECTED_POLICY_TYPE=multi_task_dit
EXPECTED_CHECKPOINT_REVISION=sha256:93993806a4d72c993fb5759fc2cdc44ab833bf314acfa9700b83a9a75bc518d6
POLICY_TASK=put the cube in the cup
POLICY_TASK_MODE=required
POLICY_WARMUP_STATE=[3.106,-1.817,1.653,-1.618,-1.628,-3.195,0.0]
POLICY_CONFIG_OVERRIDES={"num_integration_steps":10,"n_action_steps":24}
EXTERNAL_IMAGE_SIZE=native
```

image를 Git SHA별로 보존하려면 `IMAGE_TAG`를 `fm-070000-` 뒤에 실제
`DEPLOY_SHA` 앞 7자를 붙인 값으로 정해도 된다. 학습 config의 100 integration
step은 저장된 기본값이고, 이번 원격 실행은 기존 offline `euler10` 평가와 원격
latency 검증에 맞춘 10 step을 명시적으로 사용한다.

Compose 구문을 검사한다.

```bash
cd /home/junhyeong/workspace/youngwoong/gello_software/ros2_ur_ws/src/gello_policy
```

```bash
export FM_ENV=/home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/fm-070000.env
```

```bash
export FM_COMPOSE=deploy/remote_diffusion/compose.yaml
```

```bash
docker compose -f "$FM_COMPOSE" --env-file "$FM_ENV" --profile generic config --quiet
```

## 6. Kanu Docker image와 서버 실행

코드 또는 dependency pin이 바뀌었거나 해당 tag image가 없을 때만 build한다.
Docker layer cache가 있으면 이미 받은 layer를 재사용한다.

```bash
docker compose -f "$FM_COMPOSE" --env-file "$FM_ENV" --profile generic build policy-server
```

공통 `policy-server`만 시작한다. service 이름을 생략한 `docker compose up`은
기존 Diffusion service까지 대상으로 삼을 수 있으므로 사용하지 않는다.

```bash
docker compose -f "$FM_COMPOSE" --env-file "$FM_ENV" --profile generic up -d --no-build --force-recreate policy-server
```

로그를 확인한다.

```bash
docker compose -f "$FM_COMPOSE" --env-file "$FM_ENV" --profile generic logs -f policy-server
```

다음 항목을 확인한 뒤 `Ctrl+C`로 log follow만 종료한다.

- local checkpoint와 preprocessor/postprocessor가 로드됨
- CUDA warm-up 완료와 queue reset
- state `[7]`, cam1/cam2 `[3,720,1280]`, action `[7]`
- `checkpoint_revision`이 이 문서의 manifest와 같음
- `task_required: true`
- `ready at 0.0.0.0:50051; sampling=multi_task_dit:flow_matching/10`

CLIPVisionModel이 full CLIP checkpoint의 text weight를, CLIPTextModel이 vision
weight를 `UNEXPECTED`로 표시하는 것은 알려진 메시지다. 그러나 **local policy
checkpoint 자체**의 `Missing key(s)` 또는 `Unexpected key(s)`가 나오면 중단한다.

health와 Kanu 내부 inference를 확인한다.

```bash
docker inspect "$(docker compose -f "$FM_COMPOSE" --env-file "$FM_ENV" --profile generic ps -q policy-server)" --format 'image={{.Config.Image}} status={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} exit={{.State.ExitCode}}'
```

```bash
docker compose -f "$FM_COMPOSE" --env-file "$FM_ENV" --profile generic exec policy-server python /app/healthcheck.py
```

```bash
docker compose -f "$FM_COMPOSE" --env-file "$FM_ENV" --profile generic exec -e SMOKE_TARGET=127.0.0.1:50051 policy-server python /app/smoke_client.py
```

유한한 action 7개, 올바른 manifest, scheduler, inference step 10,
action step 24가 출력되어야 한다.

## 7. 로컬 PC 준비와 runtime YAML 생성

입력 위치: 로컬 PC `/home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws`

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
```

```bash
./setup_remote_client_venv.sh
```

```bash
./build_ur7e.sh
```

```bash
mkdir -p /home/laptop3/youngwoong_ws/runtime/remote-gpu-server
```

저장소 원본은 수정하지 않고 runtime 영역에 복사한다.

```bash
cp /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws/src/gello_policy/config/fm_remote_deploy.yaml /home/laptop3/youngwoong_ws/runtime/remote-gpu-server/fm-070000-deploy.yaml
```

manifest placeholder를 실제 값으로 교체한다.

```bash
sed -i 's#REPLACE_WITH_SHA256_MANIFEST#sha256:93993806a4d72c993fb5759fc2cdc44ab833bf314acfa9700b83a9a75bc518d6#' /home/laptop3/youngwoong_ws/runtime/remote-gpu-server/fm-070000-deploy.yaml
```

계약과 안전값을 확인한다.

```bash
grep -nE 'expected_model_id|expected_checkpoint_revision|expected_scheduler|expected_inference_steps|expected_action_steps|start_pose|start_gripper|act_timeout_s|obs_timeout_s|auto_start_on_stream|joint_limits_lo|joint_limits_hi|max_dev_rad|staleness_timeout_s|max_step_rad' /home/laptop3/youngwoong_ws/runtime/remote-gpu-server/fm-070000-deploy.yaml
```

반드시 `auto_start_on_stream: false`이고 timeout 순서가
`0.6 < 0.7 < 0.8`이어야 한다. start pose와 joint limit가 현장 로봇에서
안전한지 사람이 다시 확인한다.

SSH alias를 확인한다.

```bash
ssh -o ConnectTimeout=5 kanu 'echo KANU_SSH_OK'
```

연결이 성공해도 출력에 token, `.env`, SSH key 또는 credential을 복사해 공유하지
않는다.

## 8. 로봇 연결 전 실제 SSH/gRPC 왕복 검사

입력 위치: 로컬 PC `/home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws`

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
```

```bash
SSH_HOST=kanu REMOTE_GRPC_PORT=50052 LOCAL_GRPC_PORT=50052 ROUNDTRIP_TIMEOUT_S=30 ROUNDTRIP_IMAGE_HEIGHT=720 ROUNDTRIP_IMAGE_WIDTH=1280 ROUNDTRIP_STATE='[3.106,-1.817,1.653,-1.618,-1.628,-3.195,0.0]' ROUNDTRIP_ONLY=1 ./run_ur7e_diffusion_remote.sh
```

`ROUNDTRIP_ONLY PASS`, finite 7D action, 올바른 manifest,
`multi_task_dit:flow_matching`, 10/24 계약을 확인한다. 실패하면 카메라나
로봇을 시작하지 않는다.

## 9. 실물 실행 전 안전 준비

다음 체크리스트를 운영자 두 명이 함께 확인한다.

- [ ] Kanu container가 `running/healthy`이고 smoke와 로컬 왕복이 통과했다.
- [ ] 로컬과 Kanu의 Git SHA가 같다.
- [ ] UR7e IP와 해당 로봇의 calibration 파일을 확인했다.
- [ ] UR External Control 프로그램과 Robotiq 연결을 확인했다.
- [ ] 시작 자세로 이동할 전체 경로에 충돌물이 없다.
- [ ] cup/cube와 두 카메라의 위치가 학습 배치와 같다.
- [ ] pendant 속도 slider를 낮췄다.
- [ ] 한 명은 명령을 실행하고 한 명은 pendant·비상정지를 담당한다.
- [ ] `start_execution` 전에도 move-to-start가 실제 로봇을 움직일 수 있음을 안다.

## 10. 실물 UR7e 실행

로컬에서 Terminal A, B, C 세 개를 연다.

### 10.1 Terminal A: 두 카메라 실행

입력 위치: 로컬 PC `/home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws`

아래 serial은 저장소의 기존 장비 예시다. **실제 학습 장비 serial을 확인한
뒤 필요하면 교체한다.**

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
```

연결된 RealSense의 serial을 먼저 열거한다.

```bash
rs-enumerate-devices -s
```

두 장비가 보이지 않거나 serial과 물리 배치를 확정할 수 없으면 중단한다.

```bash
CAM1_SERIAL=147122072740 CAM2_SERIAL=243222072700 CAM1_NAME=cam1 CAM2_NAME=cam2 COLOR_PROFILE=1280x720x30 VIEW=true ./launch_cameras.sh
```

viewer의 왼쪽이 전체 scene(cam1), 오른쪽이 close-up(cam2)인지 확인한다.
두 영상이 약 30 Hz이고 색상, 방향, 시야와 물체 배치가 학습 당시와 같아야 한다.
이 terminal을 계속 열어 둔다.

### 10.2 Terminal B: 원격 policy와 실물 UR7e launch

먼저 실제 값을 지정한다. 아래 robot IP는 반드시 현장에서 재확인한다.
calibration 파일을 확인했다면 그 절대 경로를 `CALIB`에 입력한다.

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
```

```bash
export ROBOT_IP=192.168.10.11
```

```bash
export CALIB=/실제/UR7e/calibration.yaml
```

로봇 IP의 네트워크 연결과 calibration 파일을 확인한다. ping 성공만으로 올바른
로봇이라는 뜻은 아니므로 pendant에 표시된 IP 및 로봇 식별정보와 함께 대조한다.

```bash
ping -c 3 "$ROBOT_IP"
```

`test`가 실패하거나 calibration이 이 UR7e에서 추출된 것임을 확인할 수 없으면
launch하지 않는다.

```bash
test -f "$CALIB" && echo 'CALIBRATION FILE OK' || { echo 'CALIBRATION FILE MISSING - STOP'; false; }
```

다음 한 줄을 실행한다. 이 명령이 SSH tunnel도 열며, 비밀번호 인증이면 prompt에
Kanu 비밀번호를 입력한다.

```bash
SSH_HOST=kanu REMOTE_GRPC_PORT=50052 LOCAL_GRPC_PORT=50052 ROBOT_IP="$ROBOT_IP" CALIB="$CALIB" HEADLESS=false ./run_ur7e_diffusion_remote.sh params_file:=/home/laptop3/youngwoong_ws/runtime/remote-gpu-server/fm-070000-deploy.yaml launch_rviz:=false
```

launch 직후 자율 policy는 HOLD지만 move-to-start는 시작될 수 있다. 다음을
확인할 때까지 `start_execution`을 호출하지 않는다.

1. UR driver와 Robotiq가 오류 없이 연결된다.
2. move-to-start가 안전하게 끝나고 controller handover가 완료된다.
3. 실제 관절이 start pose 부근에 있고 로봇이 HOLD로 정지한다.
4. 로그의 model, manifest, scheduler, step 계약이 모두 일치한다.
5. `/joint_states`, gripper, cam1, cam2가 fresh하다.

### 10.3 Terminal C: 상태 확인, 실행, HOLD

ROS 환경을 source한다.

```bash
set +u; source /opt/ros/humble/setup.bash; source /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws/install/setup.bash; set -u
```

필수 topic을 확인한다.

```bash
ros2 topic hz /joint_states
```

```bash
ros2 topic hz /cam1/cam1/color/image_raw/compressed
```

```bash
ros2 topic hz /cam2/cam2/color/image_raw/compressed
```

운영자 두 명이 HOLD와 주변을 최종 확인한 뒤에만 policy 실행을 허가한다.

```bash
ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
```

응답 `success: true`와 Terminal B의 `RESET ok -> EXECUTE`를 확인한다. 로봇,
케이블, gripper, cube/cup을 계속 관찰한다.

정상 시험을 끝내거나 이상 징후가 보이면 즉시 HOLD를 요청한다.

```bash
ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger
```

HOLD service가 늦거나 로봇이 위험하면 pendant 정지 또는 비상정지를 먼저 누른다.
timeout, stale camera, contract mismatch, non-finite/OOD action으로 FAULT가 나면
원인을 해결하기 전 `start_execution`을 다시 호출하지 않는다.

## 11. 안전 종료 순서

1. Terminal C에서 `/policy_leader_node/hold`를 호출한다.
2. 로봇이 완전히 정지했는지 눈으로 확인한다.
3. Terminal B에서 `Ctrl+C`를 한 번 눌러 ROS launch와 SSH tunnel을 종료한다.
4. controller와 driver 종료를 확인한다.
5. Terminal A에서 `Ctrl+C`를 눌러 viewer와 두 카메라를 종료한다.
6. Kanu에서 policy container를 정지한다.

Kanu 입력 위치:

```bash
cd /home/junhyeong/workspace/youngwoong/gello_software/ros2_ur_ws/src/gello_policy
```

```bash
export FM_ENV=/home/junhyeong/workspace/youngwoong/runtime/remote-gpu-server/fm-070000.env
```

```bash
export FM_COMPOSE=deploy/remote_diffusion/compose.yaml
```

```bash
docker compose -f "$FM_COMPOSE" --env-file "$FM_ENV" --profile generic stop policy-server
```

```bash
docker compose -f "$FM_COMPOSE" --env-file "$FM_ENV" --profile generic ps -a policy-server
```

## 12. 실패 시 중단 기준

다음 중 하나라도 발생하면 실물 실행을 중단한다.

- checkpoint manifest 또는 Git SHA 불일치
- local policy checkpoint의 missing/unexpected key
- Kanu container가 unhealthy이거나 CUDA warm-up 실패
- 왕복 smoke 실패 또는 action이 7D finite 값이 아님
- scheduler/step/action chunk 계약 불일치
- refill 지연이 0.6초에 근접하거나 초과
- cam1/cam2가 뒤바뀌거나 frame이 stale함
- joint/gripper 단위, 순서 또는 방향이 학습 데이터와 다름
- start pose, calibration, robot IP 또는 camera serial을 확인하지 못함
- move-to-start 중 충돌 가능성 또는 예상 밖 움직임
- joint limit clamp, timeout, OOD/non-finite action 또는 FAULT 로그

문제를 해결한 뒤에는 Kanu health → Kanu smoke → 로컬 왕복 → HOLD 확인 순서를
처음부터 다시 수행한다.
