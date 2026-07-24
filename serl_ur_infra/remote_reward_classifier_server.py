#!/usr/bin/env python3
"""GPU-side HIL-SERL reward classifier ZMQ server (no ROS dependency)."""

import argparse
import os
import sys
import time

import jax
import numpy as np
import zmq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDER_PACKAGE = os.path.join(REPO_ROOT, "ros2_ur_ws", "src", "gello_recorder")
sys.path.insert(0, RECORDER_PACKAGE)

from gello_recorder.remote_classifier_runtime import (  # noqa: E402
    decode_request, encode_reply)
from gello_recorder.reward_classifier_runtime import (  # noqa: E402
    IMAGE_KEYS, make_observation, sigmoid_probability, validate_threshold)


def load_classifier(hil_serl_root, checkpoint):
    sys.path.insert(0, os.path.join(hil_serl_root, "serl_launcher"))
    from serl_launcher.networks.reward_classifier import load_classifier_func

    sample = {
        "state": np.zeros((1, 1), np.float32),
        "cam1": np.zeros((1, 128, 128, 3), np.uint8),
        "cam2": np.zeros((1, 128, 128, 3), np.uint8),
    }
    classifier = load_classifier_func(
        key=jax.random.PRNGKey(0), sample=sample, image_keys=list(IMAGE_KEYS),
        checkpoint_path=checkpoint)
    np.asarray(classifier(sample)).item()  # compile/warm up before ready
    return classifier


def serve(bind, classifier, max_jpeg_bytes):
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(bind)
    print("remote reward classifier ready on %s" % bind, flush=True)
    while True:
        request_id = ""
        try:
            header, cam1, cam2 = decode_request(
                sock.recv_multipart(), max_jpeg_bytes=max_jpeg_bytes)
            request_id = header["request_id"]
            started = time.monotonic()
            logit = float(np.asarray(classifier(make_observation(cam1, cam2))).item())
            inference_ms = (time.monotonic() - started) * 1000.0
            probability = sigmoid_probability(logit)
            threshold = validate_threshold(header.get("threshold", 0.5))
            sock.send(encode_reply(
                ok=True, request_id=request_id,
                probability=probability,
                success=probability > threshold,
                threshold=threshold,
                inference_ms=inference_ms,
                cam_skew_ms=abs(
                    header["cam1_stamp_ns"] - header["cam2_stamp_ns"]) / 1e6,
                message="ok"))
        except KeyboardInterrupt:
            break
        except Exception as exc:
            sock.send(encode_reply(
                ok=False, request_id=request_id, message=str(exc)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="tcp://127.0.0.1:5594")
    parser.add_argument("--hil-serl-root", default=os.path.join(REPO_ROOT, "third_party", "hil-serl"))
    parser.add_argument("--checkpoint", default=os.path.join(REPO_ROOT, "classifier_ckpt", "cube_in_cup"))
    parser.add_argument("--max-jpeg-bytes", type=int, default=20_000_000)
    args = parser.parse_args()
    serve(args.bind, load_classifier(args.hil_serl_root, args.checkpoint),
          args.max_jpeg_bytes)


if __name__ == "__main__":
    main()
