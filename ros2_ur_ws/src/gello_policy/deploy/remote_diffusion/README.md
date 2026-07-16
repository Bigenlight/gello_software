# GPU server container

This container runs only the JOINT Diffusion inference service. ROS2, cameras,
robot drivers, safety clamps, and controller watchdogs remain on the robot laptop.

## Server assumptions

The initial target is `kanu` (`x86_64`, Ubuntu 24.04, eight RTX A4000 GPUs,
NVIDIA driver 550.144.03, Docker 27, Compose 2.32, NVIDIA Container Toolkit
1.17.3). The host driver must not be upgraded for this project.

The image deliberately tries the project's tested PyTorch cu128 build first. Driver
550 advertises CUDA 12.4, so compatibility is **not assumed**: the CUDA smoke test
below is a mandatory gate. A failure must be reported; the service never silently
falls back to CPU.

LeRobot 0.6.1 is not published on PyPI. The image installs the experiment's exact
upstream commit `8a74e0ac6d01706d67fddfed682a09d694d9c8c0` instead of substituting the
incompatible PyPI 0.6.0 release.

## GPU selection

Choose a physical GPU for every invocation. Check current occupancy first:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
```

For example, to use GPU 3:

```bash
GPU_DEVICE=3 docker compose --env-file .env up --build
```

Inside the container that selected physical device is exposed as its only GPU and
normally appears to PyTorch as `cuda:0`. Do not pass the host index to PyTorch.

## Configuration

Keep every project-owned file below the dedicated workspace; the shared account's
home directory must not be used as an unstructured project store:

```text
/home/junhyeong/workspace/youngwoong/
├── gello_software/   # repository checkout
├── models/           # downloaded checkpoints
├── logs/             # inference logs
└── runtime/          # local runtime configuration or future certificates
```

Copy `.env.example` to `.env` on the server and set `CHECKPOINT_DIR` to an
absolute path below that workspace. Do not commit `.env`, credentials, or the
checkpoint. Docker images and layers remain in the shared Docker daemon's system
storage; inspect their disk impact separately with `docker system df` and never
prune other users' images or containers.

The default published endpoint is `127.0.0.1:50051`, intended for an SSH tunnel.
Do not change `INFERENCE_BIND_IP` to `0.0.0.0` on the public server address without
a firewall/VPN and transport authentication.

## Mandatory CUDA gate

After the image exists, run this without starting the robot:

```bash
GPU_DEVICE=0 docker compose --env-file .env run --rm --entrypoint python \
  diffusion-server -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

The gate passes only if `torch.cuda.is_available()` is `True` and the selected RTX
A4000 is printed. If driver/runtime initialization fails, stop here and choose a
driver-compatible PyTorch/CUDA image deliberately; do not widen robot timeouts or
fall back to CPU.
