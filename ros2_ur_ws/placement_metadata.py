"""Durable GUI START snapshots, bound to one run and service-call time interval."""
import json
import os
import re
import threading
import uuid
from pathlib import Path
from recording import atomic_json
from eval_log_hub import episode_files

# 2026-09-22 operator request: show the scene in the folder name instead of renaming by hand.
# The episode folder becomes `<stem>__<scene>__<outcome>`; `recording.tag_episode` keeps the
# scene segment when the label changes. `placement_reference.json` stays authoritative.
SCENE_MAX = 48


def scene_tag(snapshot):
    """'cube-ep59-take59' / 'bowl_triple-ep59-take27' / 'carrot-trainmean56' / 'cube-off'.

    None when the snapshot names no task (nothing to say). Only characters safe in a path.
    """
    if not isinstance(snapshot, dict):
        return None
    task = snapshot.get('task')
    if not task:
        return None
    parts = [str(task)]
    if not snapshot.get('enabled'):
        parts.append('off')
    elif snapshot.get('mode') == 'single_episode':
        indices = snapshot.get('episode_indices') or []
        if len(indices) == 1:
            parts.append(f'ep{indices[0]}')
        take = str(snapshot.get('selected_take') or '')
        match = re.match(r'take_(\d+)_', take)
        if match:
            parts.append('take' + match.group(1))
    else:
        split = str(snapshot.get('split') or '')
        split = re.sub(r'[^A-Za-z0-9]', '', split) or 'split'
        parts.append(f'{split}mean{len(snapshot.get("episode_indices") or [])}')
    tag = re.sub(r'[^A-Za-z0-9_.-]', '', '-'.join(parts))
    return tag[:SCENE_MAX] or None


def scene_name(name, tag):
    """Insert/replace the scene segment of `<stem>[__<scene>]__<outcome>`; unchanged when tag is None."""
    parts = name.split('__')
    if tag is None or len(parts) < 2:
        return name
    return '__'.join([parts[0], tag, parts[-1]])

class PlacementMetadata:
    def __init__(self, hub):
        self.directory = Path(hub)/'.pending_placement_metadata'
        self.lock = threading.Lock()
        self.message = ''

    def enqueue(self, run, request_ns, ack_ns, snapshot):
        self.directory.mkdir(parents=True, exist_ok=True)
        data = dict(run_dir=str(run), request_ns=request_ns, ack_ns=ack_ns, snapshot=snapshot)
        atomic_json(self.directory/(str(uuid.uuid4())+'.json'), data)
        self.message = '배치 기준 저장 예약됨 (START 시점)'

    def process(self):
        if not self.lock.acquire(blocking=False):
            return
        try:
            for request in self.directory.glob('*.json'):
                try:
                    data = json.loads(request.read_text())
                    matches = []
                    for f, meta in episode_files(data['run_dir']):
                        if (meta.get('accepted_start') and
                                data['request_ns'] <= meta['start_ns'] <= data['ack_ns']):
                            matches.append((f.parent, meta))
                    if len(matches) != 1:
                        self.message = '배치 기준 연결 대기: START와 일치하는 episode 확인 중'
                        continue
                    path, meta = matches[0]
                    if meta['status'] not in ('ready', 'invalid', 'partial'):
                        continue
                    if meta.get('metadata', {}).get('render') and meta.get('counts', {}).get('cam1'):
                        status = path/'render_status.json'
                        if not status.exists() or json.loads(status.read_text()).get('status') == 'rendering':
                            continue
                    fd = os.open(path/'.tag.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
                    current = path
                    try:
                        result = dict(data['snapshot'], episode_id=meta['episode_id'],
                                      start_request_ns=data['request_ns'], start_ack_ns=data['ack_ns'],
                                      episode_start_ns=meta['start_ns'], binding='GUI_START_service_interval')
                        destination = path/'placement_reference.json'
                        if destination.exists():
                            if json.loads(destination.read_text()) != result:
                                raise ValueError('existing placement reference differs; not overwritten')
                        else:
                            atomic_json(destination, result)
                        # Scene in the folder name, so the log tree reads at a glance. Cosmetic only:
                        # placement_reference.json / outcome.json stay authoritative, and a name
                        # collision leaves the directory untouched.
                        target = path.with_name(scene_name(path.name, scene_tag(data['snapshot'])))
                        if target != path and not target.exists():
                            os.rename(path, target)
                            current = target
                        request.unlink()
                        self.message = '배치 기준 저장 완료: '+meta['episode_id']
                    finally:
                        os.close(fd)
                        (current/'.tag.lock').unlink()
                except (FileNotFoundError, FileExistsError):
                    continue  # directory rename / renderer lock; retry, never steal
                except Exception as exc:
                    self.message = '배치 기준 저장 확인 필요: '+str(exc)
        finally:
            self.lock.release()

    def tick(self):
        if not self.lock.locked():
            threading.Thread(target=self.process, daemon=True).start()
