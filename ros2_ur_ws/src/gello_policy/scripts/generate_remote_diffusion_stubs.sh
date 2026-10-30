#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

protoc \
  -I"${ROOT}/proto" \
  --python_out="${ROOT}/gello_policy" \
  "${ROOT}/proto/remote_diffusion.proto"
