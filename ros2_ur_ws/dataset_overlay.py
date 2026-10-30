"""Display-only first-frame placement references; no ROS or recorder dependency."""
import json
import hashlib
import time
from pathlib import Path
import queue
import threading

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QCheckBox, QSlider

PACKAGE_ROOT = Path('/home/junhyeong/common_encoder_lighted_20260919')
SUBSET = Path('/home/junhyeong/svf_real_carrot20_20260919/selection_manifest.json')


def catalog(root=PACKAGE_ROOT, subset=SUBSET):
    """Only explicit manifest episode IDs define splits; never guess by filename order."""
    result, errors = [], []
    for folder in ('carrot_seed0', 'bowl_seed0', 'bowl_triple_seed0', 'cube_seed0'):
        path = root / folder / 'manifest.json'
        if not path.exists():
            continue
        try:
            m = json.loads(path.read_text())
            records = m.get('decoded_image_cache', {}).get('files', [])
            for task, split in m['split']['tasks'].items():
                takes, cached = {}, {}
                for row in records:
                    if row['task'] == task:
                        idx = int(row['episode'])
                        if idx in takes and takes[idx] != row['take']:
                            raise ValueError('episode/take mapping conflict')
                        takes[idx] = row['take']
                        cached.setdefault(idx,{})[int(row['camera'])] = row['path']
                train, val = list(split['train_episodes']), list(split['val_episodes'])
                if set(train) & set(val) or not takes or set(train+val) != set(takes):
                    raise ValueError('incomplete or overlapping split')
                if len(set(takes.values())) != len(takes):
                    raise ValueError('duplicate take mapping')
                if any(set(v) != {1,2} for v in cached.values()):
                    raise ValueError('missing paired image cache')
                entry = dict(task=task, label=f'{folder} · full {len(takes)}',
                             root=m['raw_roots'][task], takes=takes, cached=cached, train=train, val=val,
                             manifest=str(path), note=('공통 encoder/base 학습 split · '+
                                 ('full56 SVF / IFQL / DSRL' if task=='carrot' else 'task별 Flow BC package')),
                             fingerprint=m.get('source_fingerprint', ''))
                result.append(entry)
                if task == 'carrot' and subset.exists():
                    sm = json.loads(subset.read_text())
                    selected = sm['selected_episode_indices']
                    if not set(selected) <= set(train) or len(set(selected)) != len(selected):
                        raise ValueError('carrot20 selection is outside train')
                    if [takes[i] for i in selected] != sm['selected_takes']:
                        raise ValueError('carrot20 episode/take mapping differs')
                    result.append(dict(entry, label='Carrot20 · downstream (recover 제외)',
                        train=selected, takes={i:takes[i] for i in selected+val},
                        manifest=str(subset), note='SVF / IFQL / DSRL downstream train20 + C3 val9; encoder/base는 train56 재사용'))
        except Exception as exc:
            errors.append(f'{folder}: {exc}')
    return result, errors


def first_frames(entry, indices, cancelled=lambda: False):
    """Mean first aligned RGB cache row per episode; undo only known letterboxing."""
    if not indices:
        raise ValueError('선택된 episode가 없습니다')
    totals, shape, count = [None, None], [None, None], 0
    for idx in indices:
        if cancelled():
            return None
        take = Path(entry['root']) / entry['takes'][idx]
        for cam in (0, 1):
            path = take / f'cam{cam+1}.mp4'
            cap = cv2.VideoCapture(str(path))
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok or frame is None:
                raise ValueError(f'첫 프레임을 읽을 수 없음: {path}')
            # Native geometry preserved; smaller display reference for bounded RAM.
            target = (640, round(frame.shape[0]*640/frame.shape[1]))
            cache = np.load(entry['cached'][idx][cam+1],mmap_mode='r',allow_pickle=False)
            if cache.dtype != np.uint8 or cache.ndim != 4 or cache.shape[1:] != (224,224,3):
                raise ValueError('학습 이미지 cache 형식 불일치')
            scale = min(224/frame.shape[1],224/frame.shape[0])
            rw,rh = round(frame.shape[1]*scale),round(frame.shape[0]*scale)
            x,y = (224-rw)//2,(224-rh)//2
            rgb = np.array(cache[0,y:y+rh,x:x+rw],copy=True)
            del cache
            image = cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),target,interpolation=cv2.INTER_LINEAR)
            if totals[cam] is None:
                shape[cam] = frame.shape
                totals[cam] = image.astype(np.float32)
            else:
                if shape[cam] != frame.shape:
                    raise ValueError(f'카메라 해상도가 혼합되어 있음: {path}')
                totals[cam] += image
        count += 1
    return [np.rint(x/count).astype(np.uint8) for x in totals]


class OverlayPanel(QWidget):
    loaded = pyqtSignal(int, object, str)

    def __init__(self, parent=None, entries=None):
        super().__init__(parent)
        self.setFixedWidth(310)
        self.references = None
        self.resized = {}
        self.generation = 0
        self.closed = False
        self.jobs = queue.Queue(maxsize=1)
        self.entries, errors = catalog() if entries is None else (entries, [])
        box = QVBoxLayout(self)
        title = QLabel('데이터셋 배치 가이드')
        title.setStyleSheet('font-size: 18px; font-weight: bold;')
        box.addWidget(title)
        self.enabled = QCheckBox('오버랩 켜기 (화면 전용)')
        box.addWidget(self.enabled)
        self.task = QComboBox(); self.dataset = QComboBox()
        self.split = QComboBox(); self.mode = QComboBox(); self.episode = QComboBox()
        for label, widget in [('Task',self.task),('데이터셋 / 학습 범위',self.dataset),
                              ('Split',self.split),('표시 방식',self.mode),('Episode',self.episode)]:
            box.addWidget(QLabel(label)); box.addWidget(widget)
            widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            widget.setMinimumContentsLength(12)
        self.mode.addItems(['선택 split 전체 첫 프레임 평균','선택 episode 첫 프레임'])
        self.opacity_text = QLabel('참조 이미지 투명도: 25%')
        box.addWidget(self.opacity_text)
        self.opacity = QSlider(Qt.Horizontal); self.opacity.setRange(0,100); self.opacity.setValue(25)
        box.addWidget(self.opacity)
        self.status = QLabel('오버랩 꺼짐'); self.status.setWordWrap(True)
        self.detail = QLabel(); self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(self.status); box.addWidget(self.detail)
        notice = QLabel('학습 cache의 정렬된 첫 프레임\ncam1 → SCENE / cam2 → WRIST\n배치 비교용 · ID/OOD 자동 판정 아님\nval/전체 선택은 평가 배치에 영향을 줄 수 있음')
        notice.setWordWrap(True); box.addWidget(notice)
        if errors:
            error = QLabel('일부 데이터셋 사용 불가:\n'+'\n'.join(errors))
            error.setWordWrap(True); box.addWidget(error)
        box.addStretch()
        self.task.addItems(sorted({e['task'] for e in self.entries}))
        self.task.currentIndexChanged.connect(self.tasks_changed)
        self.dataset.currentIndexChanged.connect(self.dataset_changed)
        self.split.currentIndexChanged.connect(self.split_changed)
        self.mode.currentIndexChanged.connect(self.request)
        self.episode.currentIndexChanged.connect(self.request)
        self.enabled.toggled.connect(self.request)
        self.opacity.valueChanged.connect(lambda v:self.opacity_text.setText(f'참조 이미지 투명도: {v}%'))
        self.loaded.connect(self.accept)
        threading.Thread(target=self.worker, daemon=True, name='dataset-overlay').start()
        self.tasks_changed()

    def current(self):
        return self.dataset.currentData()

    def snapshot(self):
        """Freeze placement reference settings at GUI START, not at HOLD."""
        e = self.current()
        single = self.mode.currentIndex() == 1
        ids = ([self.episode.currentData()] if single else list(self.split.currentData() or []))
        ids = [i for i in ids if i is not None]
        data = dict(schema='eval-placement-reference/v1', captured_wall_ns=time.time_ns(),
                    enabled=self.enabled.isChecked(), reference_ready=self.references is not None,
                    opacity=self.opacity.value()/100., mode='single_episode' if single else 'split_mean',
                    split=['train','val','all'][max(0,self.split.currentIndex())],
                    status=self.status.text(), frame_source='aligned_training_cache', frame_index=0,
                    display_only=True, affects_policy_input=False,
                    camera_mapping={'1':'SCENE','2':'WRIST'})
        if e:
            data.update(task=e['task'], dataset=e['label'], raw_root=e['root'],
                        manifest=e['manifest'], source_fingerprint=e.get('fingerprint'),
                        episode_indices=ids, takes=[e['takes'][i] for i in ids],
                        selected_take=e['takes'][ids[0]] if single and ids else None,
                        cache_files=[{'episode':i, 'take':e['takes'][i],
                                      'cameras':e['cached'][i]} for i in ids])
            try:
                data['manifest_sha256'] = hashlib.sha256(Path(e['manifest']).read_bytes()).hexdigest()
            except OSError as exc:
                data['manifest_read_error'] = str(exc)
        if self.references is not None:
            data['display_reference_sha256'] = [hashlib.sha256(ref.tobytes()).hexdigest() for ref in self.references]
        return data

    def tasks_changed(self, *_):
        self.dataset.blockSignals(True); self.dataset.clear()
        for entry in self.entries:
            if entry['task'] == self.task.currentText():
                self.dataset.addItem(entry['label'], entry)
        self.dataset.blockSignals(False); self.dataset_changed()

    def dataset_changed(self, *_):
        e = self.current()
        self.split.blockSignals(True); self.split.clear()
        if e:
            for name, ids in [('train',e['train']),('val',e['val']),('전체 (train + val)',e['train']+e['val'])]:
                self.split.addItem(f'{name} · {len(ids)} episodes', ids)
            # 2026-09-23 user request: default split = 전체 (train + val). The package train/val
            # here is the older split, not the first50 (0-49) basis, so "all" avoids a mismatch.
            self.split.setCurrentIndex(2)
            self.detail.setText(e['note']+'\n'+e['manifest'])
        self.split.blockSignals(False); self.split_changed()

    def split_changed(self, *_):
        e = self.current()
        self.episode.blockSignals(True); self.episode.clear()
        if e:
            for idx in self.split.currentData() or []:
                self.episode.addItem(f"{idx}: {e['takes'][idx]}", idx)
        self.episode.blockSignals(False); self.request()

    def request(self, *_):
        self.generation += 1
        self.references = None; self.resized.clear()
        self.episode.setEnabled(self.mode.currentIndex()==1)
        e = self.current()
        if not self.enabled.isChecked():
            self.status.setText('오버랩 꺼짐'); return
        if not e:
            self.status.setText('사용 가능한 manifest가 없습니다'); return
        indices = ([self.episode.currentData()] if self.mode.currentIndex()==1 else self.split.currentData())
        if not indices or any(i is None for i in indices):
            self.status.setText('선택된 episode가 없습니다'); return
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        self.status.setText(f'첫 프레임 {len(indices)}개 불러오는 중…')
        self.jobs.put_nowait((self.generation,e,list(indices)))

    def worker(self):
        while not self.closed:
            try:
                generation, entry, indices = self.jobs.get(timeout=.5)
            except queue.Empty:
                continue
            try:
                refs = first_frames(entry,indices,lambda:self.closed or generation!=self.generation)
                if refs is not None and not self.closed:
                    self.loaded.emit(generation,refs,f'표시 중: {len(indices)} episodes · 첫 프레임')
            except Exception as exc:
                if not self.closed:
                    self.loaded.emit(generation,None,str(exc))

    def accept(self, generation, references, message):
        if generation != self.generation or not self.enabled.isChecked():
            return
        self.references = references; self.resized.clear(); self.status.setText(message)

    def compose(self, frame, camera):
        if not self.enabled.isChecked() or self.references is None or not self.opacity.value():
            return frame
        ref = self.references[camera]
        h,w=frame.shape[:2]
        if abs((w/h)/(ref.shape[1]/ref.shape[0])-1) > .01:
            self.status.setText('화면과 참조의 종횡비 불일치 — 오버랩 중지')
            return frame
        key=(camera,h,w)
        if key not in self.resized:
            self.resized[key]=cv2.resize(ref,(w,h),interpolation=cv2.INTER_LINEAR)
        alpha=self.opacity.value()/100.
        # Allocates a new display frame. Never changes feed.frame or ROS payload.
        return cv2.addWeighted(frame,1-alpha,self.resized[key],alpha,0)

    def shutdown(self):
        self.closed=True
        self.generation+=1
