"""Original JPEG/reference recordings and index-aligned MP4 archives.

Frames are BGR uint8. Timestamp lookup ALWAYS uses the H5 stream, never MP4 PTS.
No fallback to lower-resolution render videos and no JPEG re-encoding.
"""
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

STREAMS = ('cam1', 'cam2', 'native_cam1', 'native_cam2')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def recording_path(episode):
    episode = Path(episode)
    original = episode / 'recording.h5'
    if original.is_file():
        return original
    metadata = episode / 'recording.metadata.h5'
    if metadata.is_file() and (episode / 'archive.json').is_file():
        return metadata
    raise FileNotFoundError(f'No recording H5 in {episode}')


class RecordingFrames:
    def __init__(self, recording, episode=None):
        self.f = recording
        self.path = Path(episode) if episode is not None else Path(recording.filename).parent
        self.manifest = None
        self._caps = {}
        self._last = {}
        self._positions = {}
        self.is_archive = any(k in recording and 'payload' not in recording[k] for k in STREAMS)
        if not self.is_archive:
            return
        try:
            self.manifest = json.loads((self.path / 'archive.json').read_text())
            if self.manifest.get('verification') != 'PASS':
                raise ValueError('Archive verification did not pass')
            if self.manifest.get('schema') != 'real-eval-mp4-archive/v2':
                raise ValueError('Archive needs v2 integrity/index metadata')
            if sha256_file(recording.filename) != self.manifest['metadata_sha256']:
                raise ValueError('Archive metadata H5 hash mismatch')
            for stream in STREAMS:
                if stream not in recording:
                    continue
                info = self.manifest['streams'][stream]
                timestamps = self.timestamps(stream)
                if len(timestamps) != info['frames']:
                    raise ValueError(f'Archive timestamp count mismatch: {stream}')
                if hashlib.sha256(timestamps.tobytes()).hexdigest() != info['timestamp_sha256']:
                    raise ValueError(f'Archive timestamp hash mismatch: {stream}')
                if len(recording[stream + '/original_jpeg_sha256']) != len(timestamps):
                    raise ValueError(f'Original JPEG hash count mismatch: {stream}')
                if not len(timestamps):
                    continue
                name = info['video']
                if Path(name).name != name:
                    raise ValueError('Archive video must be a local basename')
                path = self.path / name
                if sha256_file(path) != info['sha256']:
                    raise ValueError(f'Archive video hash mismatch: {stream}')
                cap = cv2.VideoCapture(str(path))
                self._caps[stream] = cap
                if (not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(timestamps)
                        or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != info['width']
                        or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != info['height']):
                    raise ValueError(f'Archive video geometry/count mismatch: {stream}')
                self._positions[stream] = 0
        except BaseException:
            self.close()
            raise

    def timestamps(self, stream):
        return np.asarray(self.f[stream + '/timestamp_ns'], dtype=np.int64)

    def _check_index(self, stream, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.f[stream + '/timestamp_ns']):
            raise IndexError(f'Invalid original frame index: {stream}[{index}]')
        return int(index)

    def jpeg(self, stream, index):
        """Exact original JPEG bytes; unavailable in MP4 archives."""
        index = self._check_index(stream, index)
        if self.is_archive:
            raise ValueError('MP4 archive has no original JPEG bytes; use bgr() or verify_jpeg()')
        group = self.f[stream]
        if 'native_index' in group:
            native_index = int(group['native_index'][index])
            if native_index >= 0:
                return bytes(self.f[group.attrs['payload_ref']]['payload'][native_index])
        return bytes(group['payload'][index])

    def verify_jpeg(self, stream, index, jpeg):
        """Verify NPZ still contains the original input; never compare recompressed pixels."""
        index = self._check_index(stream, index)
        if self.is_archive:
            recorded = bytes(self.f[stream + '/original_jpeg_sha256'][index]).decode('ascii')
            equal = hashlib.sha256(jpeg).hexdigest() == recorded
        else:
            equal = self.jpeg(stream, index) == jpeg
        if not equal:
            raise ValueError(f'{stream} original JPEG mismatch at ACT {index}')

    def bgr(self, stream, index):
        """Get original source index, including skipped/backwards/repeated requests."""
        index = self._check_index(stream, index)
        if not self.is_archive:
            pixels = cv2.imdecode(np.frombuffer(self.jpeg(stream, index), np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError(f'Invalid JPEG: {stream}[{index}]')
            return pixels
        if stream in self._last and self._last[stream][0] == index:
            return self._last[stream][1]
        # Decode ordinally rather than trusting approximate codec seeks/PTS.
        if index < self._positions[stream]:
            self._caps[stream].release()
            self._caps[stream] = cv2.VideoCapture(str(self.path / self.manifest['streams'][stream]['video']))
            self._positions[stream] = 0
        while self._positions[stream] <= index:
            ok, pixels = self._caps[stream].read()
            if not ok:
                raise ValueError(f'Missing archive frame: {stream}[{self._positions[stream]}]')
            current = self._positions[stream]
            self._positions[stream] += 1
            self._last[stream] = current, pixels
        return self._last[stream][1]

    def provenance(self):
        if not self.is_archive:
            return dict(storage='h5-jpeg', original_jpeg_available=True)
        return dict(storage='mp4-indexed', codec=self.manifest['codec'],
                    archive_manifest_sha256=sha256_file(self.path / 'archive.json'),
                    original_recording_sha256=self.manifest['source_sha256'],
                    metadata_sha256=self.manifest['metadata_sha256'],
                    lossless_pixels=all(i.get('lossless_pixels', False) for i in self.manifest['streams'].values() if i['frames']),
                    original_jpeg_available=False, timestamp_source='H5 timestamp_ns; MP4 frame i = original frame i')

    def close(self):
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._last.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
