# syntax=docker/dockerfile:1.7

# Keep Python independent of the ROS2 Humble interpreter. PyTorch's cu128 wheel
# carries the CUDA user-space runtime; NVIDIA Container Toolkit injects the host
# driver into the container at run time.
FROM python:3.12.10-slim-bookworm

ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    XDG_CACHE_HOME=/tmp/cache \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the tested CUDA 12.8 PyTorch build explicitly. The server deliberately
# refuses to fall back to CPU if the host driver cannot initialize this runtime.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "torch==${TORCH_VERSION}+cu128" \
        "torchvision==${TORCHVISION_VERSION}+cu128" \
        --index-url https://download.pytorch.org/whl/cu128

COPY policy_server/requirements-remote-diffusion.lock /tmp/requirements.lock
RUN python -m pip install -r /tmp/requirements.lock

COPY proto/ /app/proto/
COPY policy_server/ /app/policy_server/
COPY deploy/remote_diffusion/healthcheck.py /app/healthcheck.py

# Generate Python protobuf/gRPC bindings inside the pinned build environment.
# Both modules are emitted into policy_server so its relative imports are stable.
RUN python -m grpc_tools.protoc \
      -I/app/proto \
      --python_out=/app/policy_server \
      --grpc_python_out=/app/policy_server \
      /app/proto/remote_diffusion.proto \
    && sed -i \
      's/^import remote_diffusion_pb2 as remote__diffusion__pb2$/from . import remote_diffusion_pb2 as remote__diffusion__pb2/' \
      /app/policy_server/remote_diffusion_pb2_grpc.py

RUN useradd --create-home --uid 10001 inference \
    && chown -R inference:inference /app
USER inference

EXPOSE 50051

HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=3 \
  CMD ["python", "/app/healthcheck.py"]

ENTRYPOINT ["python", "-m", "policy_server.remote_diffusion_server"]
