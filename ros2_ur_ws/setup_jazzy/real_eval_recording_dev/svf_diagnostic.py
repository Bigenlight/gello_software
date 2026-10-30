"""CPU-only, post-HOLD Q estimates for recorded SVF full action chunks.

No policy sampling, socket, ROS, or controller calls. The two Q heads are
checkpoint critic estimates, not calibrated confidence or success probabilities.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Must precede torch/JAX imports, including indirect ones in svf_server.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
if hasattr(os, 'nice'):
    os.nice(10)

import cv2
import h5py
import numpy as np
from recording import atomic_json
from frame_io import RecordingFrames, recording_path


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified(path, files):
    path = Path(path)
    if str(path) not in files or sha(path) != files[str(path)]:
        raise ValueError(f'Missing or mismatched recorded SHA: {path}')
    return path


def validate_alignment(f, z, meta, frames=None):
    requests = [json.loads(bytes(x)) for x in f['request/payload']]
    times = np.asarray(f['cam1/timestamp_ns'])
    if not np.array_equal(times, f['cam2/timestamp_ns'][:]):
        raise ValueError('Camera timestamps do not match')
    n = len(requests)
    if n != len(times) or n != len(z['state']):
        raise ValueError('ROS and server ACT counts differ; refusing guessed alignment')
    if int(z['reset_counter']) != int(meta['reset_reply']['reset_counter']):
        raise ValueError('Server reset counter mismatch')
    for i, request in enumerate(requests):
        if request['control'].get('cmd', '').lower() != 'act':
            raise ValueError('Unexpected non-ACT request')
        if not np.allclose(request['control']['state'], z['state'][i], atol=1e-6, rtol=0):
            raise ValueError(f'State mismatch at ACT {i}')
        for cam in ('cam1', 'cam2'):
            if frames is not None:
                frames.verify_jpeg(cam, i, bytes(z[cam + '_jpeg'][i]))
            elif bytes(f[cam + '/payload'][i]) != bytes(z[cam + '_jpeg'][i]):
                raise ValueError(f'{cam} JPEG mismatch at ACT {i}')
    chunk_t = np.asarray(z['chunk_t'])
    if (not len(chunk_t) or len(chunk_t) != len(z['chunks']) or
            np.any(chunk_t < 0) or np.any(chunk_t >= n) or
            np.any(np.diff(chunk_t) <= 0)):
        raise ValueError('Invalid chunk/refill indices')
    return requests, times, chunk_t


def encoder_digest(identity):
    value = identity.get('sha256')
    digest = value.get('encoder.pt') if isinstance(value, dict) else value
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Missing or malformed pinned encoder SHA256')
    return digest


def posthoc_encoder(server, stats):
    identity = stats.raw['encoder_identity']
    digest = encoder_digest(identity)
    package = Path(identity['root'])
    if sha(package / 'encoder.pt') != digest:
        raise ValueError('Task encoder SHA does not match recorded normalization')
    if isinstance(identity['sha256'], dict):
        return server.CommonPackageEncoderBridge(str(package), stats, device='cpu')
    # Task serving uses common_encoder directly, not the carrot-only package
    # bridge (which also pins carrot's unrelated Flow head and normalization).
    # Mirror TaskEncoder in task_svf_server.py without executing its server main.
    import torch
    from common_encoder import SharedDualCameraEncoder, imagenet_tensor
    task = identity.get('task')
    # The 15 Hz first50 campaigns reuse the SAME task encoder package but tag their
    # normalization variant with a '_15hz' suffix (r18_common_cube_15hz). The encoder
    # identity itself is already verified by SHA above, so accept both spellings; the
    # 30 Hz variant string keeps working byte-identically.
    if task not in ('bowl', 'bowl_triple', 'cube') or stats.variant not in (
            'r18_common_' + task, 'r18_common_' + task + '_15hz'):
        raise ValueError('Task encoder/normalization variant mismatch')
    payload = torch.load(package / 'encoder.pt', map_location='cpu', weights_only=False)
    if payload.get('schema') != 'common-gello-encoder/v1' or payload.get('contract', {}).get('spatial_softmax_order') != 'all_x_then_all_y':
        raise ValueError('Unsupported task encoder contract')
    model = SharedDualCameraEncoder(pretrained=False)
    model.load_state_dict(payload['encoder'], strict=True)
    model.eval().requires_grad_(False)

    class TaskEncoder:
        @torch.inference_mode()
        def encode_pair(self, jpeg1, jpeg2):
            x1 = imagenet_tensor(server.jpeg_bytes_to_rgb(jpeg1))[None]
            x2 = imagenet_tensor(server.jpeg_bytes_to_rgb(jpeg2))[None]
            f1 = model.cam1(x1).float().cpu().numpy()[0]
            f2 = model.cam2(x2).float().cpu().numpy()[0]
            if f1.shape != (stats.D_f1,) or f2.shape != (stats.D_f2,):
                raise ValueError('Task encoder feature dimension mismatch')
            return f1, f2
    return TaskEncoder()


def compute(episode, output, source, runtime, wait_seconds=0):
    doc = json.loads((episode / 'episode.json').read_text())
    meta = doc['metadata']
    if meta.get('method') != 'svf' or doc['status'] != 'ready' or doc.get('dropped'):
        raise ValueError('Requires complete, valid SVF episode')
    if doc.get('stop_ns') is None:
        raise ValueError('Episode has not stopped')
    run = Path(meta['run_dir'])
    ckpt = verified(run / f"params_{int(meta['step'])}.pkl", meta['files'])
    flags = json.loads(verified(run / 'flags.json', meta['files']).read_text())
    norm_paths = [p for p in meta['files'] if Path(p).name.startswith('norm_stats') and p.endswith('.json')]
    if len(norm_paths) != 1:
        raise ValueError('Expected one SHA-pinned normalization file')
    norm = verified(norm_paths[0], meta['files'])
    if meta['reset_reply']['checkpoint'] != str(ckpt):
        raise ValueError('RESET checkpoint does not match episode checkpoint')
    if flags['agent'].get('use_critic_obs') or flags['agent']['correction_mode'] != 'soft_value':
        raise ValueError('Requires symmetric SVF critic')
    source_files = meta.get('diagnostic_source_files', {})
    for path in source_files:
        verified(path, source_files)
    server_npz = Path(meta['server_npz'])
    deadline = time.monotonic() + wait_seconds
    while not server_npz.exists() and time.monotonic() < deadline:
        time.sleep(.25)
    if not server_npz.exists():
        raise FileNotFoundError(f'Server NPZ not finalized: {server_npz}')
    with np.load(server_npz, allow_pickle=True) as npz:
        if not all(k in npz.files for k in ('cam1_jpeg', 'cam2_jpeg')):
            raise ValueError('Original JPEGs were stripped from server NPZ; retain the original H5 '
                             'or a verified original-JPEG backup for exact critic recomputation. '
                             'A lossy MP4 is not silently substituted; existing critic CSV values remain valid.')
    os.environ['QFLOW_DIR'] = str(runtime)
    sys.path.insert(0, str(source))
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    import svf_server as server
    import jax
    import jax.numpy as jnp
    stats = server.NormStats.load(norm)
    action_dim = int(stats.raw.get('action_dim', 56))  # 56 for H8; 35/70 for the H5/H10 horizon ablation
    agent, _ = server.make_agent(flags['agent'], stats.D_a, stats.D_c, action_dim, flags['seed'])
    _, _, restore = server._import_qflow()
    agent = restore(agent, str(run), int(meta['step']))
    identity = stats.raw.get('encoder_identity')
    if not identity:
        raise ValueError('This renderer requires a pinned common encoder package')
    encoder = posthoc_encoder(server, stats)
    q_fn = jax.jit(lambda obs, action: agent.network.select('critic')(obs, action))
    rows = []
    with np.load(server_npz, allow_pickle=True) as archive, h5py.File(recording_path(episode), 'r') as f, RecordingFrames(f, episode) as frames:
        z = {k: archive[k] for k in ('meta', 'reset_counter', 'state', 'cam1_jpeg', 'cam2_jpeg', 'chunk_t', 'chunks')}
        sm = json.loads(str(z['meta']))
        if sm.get('ckpt') != str(ckpt):
            raise ValueError('NPZ checkpoint mismatch')
        requests, times, chunk_t = validate_alignment(f, z, meta, frames)
        frame_storage = frames.provenance()
        for ci, act_index in enumerate(chunk_t):
            i = int(act_index)
            feats = encoder.encode_pair(bytes(z['cam1_jpeg'][i]), bytes(z['cam2_jpeg'][i]))
            obs = stats.actor_obs(z['state'][i], feats[0], feats[1])
            action = np.asarray(z['chunks'][ci], np.float32).reshape(1, action_dim)
            q = np.asarray(q_fn(jnp.asarray(obs), jnp.asarray(action))).reshape(-1)
            if len(q) != 2 or not np.isfinite(q).all():
                raise ValueError('Expected two finite SVF Q heads')
            rows.append(dict(chunk_index=ci, act_index=i, request_id=requests[i]['request_id'],
                             elapsed_s=float((int(times[i])-doc['start_ns']) / 1e9),
                             q1=float(q[0]), q2=float(q[1]), q_mean=float(q.mean())))
            if ci % 20 == 0:
                print(f'SVF Q postprocess: {ci + 1}/{len(chunk_t)} chunks', flush=True)
    with (output / 'svf_critic.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    provenance = dict(schema='svf-posthoc-critic/v1', episode_id=doc['episode_id'],
        checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt), norm_sha256=sha(norm),
        encoder_sha256=encoder_digest(identity), server_npz=str(server_npz),
        encoder_bridge='legacy_common_package' if isinstance(identity['sha256'], dict) else 'task_common_encoder_cpu',
        server_npz_sha256=sha(server_npz), recording_sha256=sha(recording_path(episode)), frame_storage=frame_storage,
        encoder_input='original JPEG retained in server NPZ; original JPEG bytes/hashes validated',
        critic='critic (online heads stored in checkpoint, not target_critic)',
        action=f'recorded normalized {action_dim}D planned chunk before ROS clamps; NOT executed 7D command',
        clock='ROS recorder host monotonic; exact state/original-JPEG sequence checked against server NPZ',
        interpretation='post-hoc expected-return estimate; not success probability or online action selection',
        runtime_pinned_at_recording=bool(source_files), source=str(source), runtime=str(runtime),
        current_source_sha256={str(p): sha(p) for p in (source / 'svf_server.py', runtime / 'agents/qflow_rc.py')},
        renderer_sha256=sha(__file__), chunk_count=len(rows), q_agg=flags['agent']['q_agg'],
        recorded_execution_chunk=meta['reset_reply'].get('chunk'))
    atomic_json(output / 'svf_critic.json', provenance)
    return doc, rows


def graph(rows, elapsed, duration, width=1280, height=300, window_seconds=10):
    canvas = np.full((height, width, 3), (27, 24, 22), np.uint8)
    def text(s, x, y, color=(225, 225, 225), scale=.6):
        cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    text('SVF critic | recorded planned chunk | post-hoc estimate', 20, 27)
    colors = [(220, 170, 80), (170, 100, 240), (90, 220, 100)]
    values = np.array([[r[k] for k in ('q1', 'q2', 'q_mean')] for r in rows])
    low, high = values.min(), values.max()
    pad = max((high-low)*.12, .5); low -= pad; high += pad
    left, right, top, bottom = 90, width-25, 62, height-57
    window_start = max(0., elapsed-window_seconds)
    window_end = window_start + window_seconds
    def point(t, q):
        return (int(left + np.clip((t-window_start)/window_seconds, 0, 1)*(right-left)),
                int(bottom - (q-low)/(high-low)*(bottom-top)))
    for q in np.linspace(low, high, 4):
        y = point(0, q)[1]
        cv2.line(canvas, (left, y), (right, y), (58, 55, 52), 1)
        text(f'{q:.1f}', 8, y+5, scale=.47)
    for t in np.linspace(window_start, window_end, 6):
        x = point(t, low)[0]
        text(f'{t:.1f}s', x-18, bottom+22, scale=.45)
    visible = [r for r in rows if r['elapsed_s'] <= elapsed]
    # Retain one value preceding the left edge to continue the step curve.
    before = [r for r in visible if r['elapsed_s'] < window_start]
    visible = before[-1:] + [r for r in visible if r['elapsed_s'] >= window_start]
    for j, name in enumerate(('q1', 'q2', 'q_mean')):
        pts = []
        for k, row in enumerate(visible):
            t = row['elapsed_s']
            if k:
                pts.append(point(t, visible[k-1][name]))
            pts.append(point(t, row[name]))
        if visible:
            pts.append(point(elapsed, visible[-1][name]))
            cv2.polylines(canvas, [np.asarray(pts, np.int32)], False, colors[j], 2 if j==2 else 1, cv2.LINE_AA)
        label = f'{name}: {visible[-1][name]:.2f}' if visible else f'{name}: waiting'
        text(label, 710+j*180, 27, colors[j], .52)
    cv2.line(canvas, point(elapsed, low), point(elapsed, high), (220, 220, 220), 1)
    text('Q = expected return, not success probability. Head spread is not calibrated uncertainty.', 20, height-10, scale=.49)
    return canvas


def video(episode, output, doc, rows, window_seconds=10):
    # Existing diagnostic video has the two timestamp-aligned camera views at top.
    cap = cv2.VideoCapture(str(episode / 'diagnostic.mp4'))
    if not cap.isOpened():
        raise ValueError('Render base diagnostic.mp4 first')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width != 1280 or height != 520:
        raise ValueError('Unexpected base video geometry or FPS')
    mapping = json.loads((episode / 'video_frames.json').read_text())
    duration = (doc['stop_ns']-doc['start_ns'])/1e9
    writer = cv2.VideoWriter(str(output / 'svf_diagnostic.partial.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, 660))
    if not writer.isOpened():
        cap.release(); raise RuntimeError('Could not open video encoder')
    try:
        for idx, frame_map in enumerate(mapping):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Video ended before timestamp mapping')
            t = (frame_map['monotonic_ns']-doc['start_ns'])/1e9
            panel = graph(rows, t, duration, window_seconds=window_seconds)
            combined = np.vstack((frame[:360], panel))
            cv2.putText(combined, f"t={t:.2f}s  SVF step {doc['metadata']['step']}", (15, 345),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2, cv2.LINE_AA)
            writer.write(combined)
            if idx == len(mapping)//2:
                cv2.imwrite(str(output / 'svf_diagnostic_preview.png'), combined)
        if cap.read()[0]:
            raise ValueError('Extra frames beyond timestamp mapping')
    finally:
        cap.release(); writer.release()
    os.replace(output / 'svf_diagnostic.partial.mp4', output / 'svf_diagnostic.mp4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--wait-seconds', type=float, default=120)
    parser.add_argument('--lock-owned', action='store_true')
    parser.add_argument('--window-seconds', type=float, default=10,
                        help='Scrolling graph time window (default: 10 seconds)')
    args = parser.parse_args()
    if not np.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error('--window-seconds must be finite and positive')
    episode = args.episode.resolve()
    output = args.output.resolve() if args.output else episode
    output.mkdir(parents=True, exist_ok=True)
    lock = episode / '.tag.lock'
    if not args.lock_owned:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    try:
        meta = json.loads((episode / 'episode.json').read_text())['metadata']
        # Legacy campaign layout fallback is recorded explicitly in provenance.
        base = Path(meta['run_dir']).parents[3]
        source = args.source or Path(meta.get('diagnostic_source', base / 'source/vision_carrot'))
        runtime = args.runtime or Path(meta.get('diagnostic_runtime', base / 'runtime/qflow_svf_merged'))
        atomic_json(output / 'svf_diagnostic_status.json', {'status':'computing', 'device':'cpu'})
        doc, rows = compute(episode, output, source, runtime, args.wait_seconds)
        video(episode, output, doc, rows, args.window_seconds)
        atomic_json(output / 'svf_diagnostic_status.json', {'status':'complete', 'chunks':len(rows), 'video':'svf_diagnostic.mp4'})
    except Exception as exc:
        atomic_json(output / 'svf_diagnostic_status.json', {'status':'failed', 'error':str(exc)})
        raise
    finally:
        if not args.lock_owned:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
