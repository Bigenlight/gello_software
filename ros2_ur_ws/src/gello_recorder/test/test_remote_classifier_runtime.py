import json
import pytest
from gello_recorder.remote_classifier_runtime import (
    PROTOCOL_VERSION, decode_reply, decode_request, encode_reply, encode_request,
    oldest_pair_receipt_monotonic)


def test_request_round_trip_preserves_id_stamps_and_jpegs():
    frames = encode_request("req-7", 100, 104, b"cam-one", b"cam-two")
    header, cam1, cam2 = decode_request(frames)
    assert header["protocol"] == PROTOCOL_VERSION
    assert (header["request_id"], header["cam1_stamp_ns"],
            header["cam2_stamp_ns"]) == ("req-7", 100, 104)
    assert header["threshold"] == 0.5
    assert (cam1, cam2) == (b"cam-one", b"cam-two")


def test_request_rejects_oversized_or_malformed_payload():
    frames = encode_request("req", 1, 2, b"1234", b"2")
    with pytest.raises(ValueError, match="oversized"):
        decode_request(frames, max_jpeg_bytes=3)
    with pytest.raises(ValueError, match="contain"):
        decode_request(frames[:2])


def test_reply_requires_matching_request_id_and_protocol():
    frame = encode_reply(ok=True, request_id="abc", probability=0.9)
    assert decode_reply(frame, "abc")["probability"] == 0.9
    with pytest.raises(ValueError, match="request_id"):
        decode_reply(frame, "different")
    bad = json.dumps({"protocol": "old", "request_id": "abc"}).encode()
    with pytest.raises(ValueError, match="protocol"):
        decode_reply(bad, "abc")


def test_pair_age_uses_older_camera_receipt():
    assert oldest_pair_receipt_monotonic(12.5, 11.0) == 11.0
    # At now=13, this represents worst-case age 2s, not newest-frame age 0.5s.
    assert 13.0 - oldest_pair_receipt_monotonic(12.5, 11.0) == 2.0
