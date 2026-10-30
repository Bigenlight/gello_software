"""Process-start compatibility settings for the local HIL-SERL runtime.

Python imports ``sitecustomize`` automatically when ``serl_ur_infra`` is on
``PYTHONPATH``.  The checked-in protobuf bindings predate the upb runtime, so
protobuf must select its pure-Python implementation before either W&B or gRPC
imports ``google.protobuf``.
"""

from importlib.metadata import PackageNotFoundError, version
import os


try:
    _protobuf_major = int(version("protobuf").split(".", 1)[0])
except (PackageNotFoundError, ValueError):
    _protobuf_major = 4

# Protobuf 3.x can run the legacy generated module with its fast C++ runtime.
# Protobuf 4+ (including the pinned 7.34.1 learner environment) requires the
# pure-Python descriptor implementation.
if _protobuf_major >= 4:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
