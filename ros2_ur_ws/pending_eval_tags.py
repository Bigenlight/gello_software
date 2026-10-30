"""Durable UUID-bound operator labels, applied only after writer/renderer release."""
import json
import threading
import uuid
from pathlib import Path
from recording import atomic_json, tag_episode


class PendingTags:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.directory = self.root / '.pending_eval_tags'
        self.lock = threading.Lock()
        self.message = ''

    def enqueue(self, path, identity, outcome):
        identity = str(uuid.UUID(identity))
        path = Path(path).absolute()
        path.relative_to(self.root)
        outcome = 'partial_success' if outcome == 'success_0.5' else outcome
        if outcome not in ('success', 'partial_success', 'failure', 'unknown'):
            raise ValueError('invalid outcome')
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            atomic_json(self.directory / (identity+'.json'), {
                'episode_id': identity, 'parent': str(path.parent), 'outcome': outcome})
            self.message = f'선택 저장됨: {outcome.upper()} · {identity} · 저장/영상 완료 후 자동 적용'

    def process(self):
        if not self.lock.acquire(blocking=False):
            return
        try:
            for request in self.directory.glob('*.json'):
                try:
                    data = json.loads(request.read_text())
                    if data['outcome'] == 'success_0.5':
                        data['outcome'] = 'partial_success'
                    identity = str(uuid.UUID(data['episode_id']))
                    parent = Path(data['parent'])
                    parent.relative_to(self.root)
                    matches = list(parent.glob('ep_*'+identity+'__*'))
                    if len(matches) != 1:
                        continue
                    path = matches[0]
                    meta = json.loads((path/'episode.json').read_text())
                    if meta['episode_id'] != identity or not meta.get('accepted_start'):
                        raise ValueError('episode identity mismatch')
                    if meta['status'] not in ('ready', 'invalid', 'partial') or (path/'.tag.lock').exists():
                        continue
                    if meta.get('metadata', {}).get('render') and meta.get('counts', {}).get('cam1'):
                        status = path/'render_status.json'
                        if not status.exists() or json.loads(status.read_text())['status'] == 'rendering':
                            continue
                    outcome = json.loads((path/'outcome.json').read_text())
                    if outcome['outcome'] != data['outcome'] or not path.name.endswith('__'+data['outcome']):
                        tag_episode(path, data['outcome'], labeler='camera_viewer_pending')
                    request.unlink()
                    self.message = f'태그 적용 완료: {data["outcome"].upper()} · {identity}'
                except FileExistsError:
                    continue  # renderer/tag lock race; never steal a lock
                except Exception as exc:
                    self.message = f'태그 예약 유지 / 확인 필요: {request.name}: {exc}'
        finally:
            self.lock.release()

    def tick(self):
        threading.Thread(target=self.process, daemon=True).start()
