"""Native Qt camera/operator display; never changes recording or control policy."""
import os
import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import signal
import sys
import time
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QImage, QPixmap, QDesktopServices
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                            QVBoxLayout, QHBoxLayout, QSizePolicy, QScrollArea)

sys.path.insert(0, str(Path(__file__).parent / 'setup_jazzy/real_eval_recording_dev'))
from viewer_tags import TagPanel, latest_episode
from pending_eval_tags import PendingTags
from dataset_overlay import OverlayPanel
from eval_log_hub import hub_root, read_active, run_episode, publish as publish_active
from placement_metadata import PlacementMetadata

# Keep VLA prompt UI and protocol in the independent FM_RL module.
sys.path.insert(0, '/home/junhyeong/FM_RL/vla_eval/models/n15')
from prompt_panel import PromptPanel

# 15 Hz model hub (2026-09-23): model selection panel, hidden unless a hub is running.
sys.path.insert(0, '/home/junhyeong/FM_RL/model_hub')
from hub_panel import HubPanel


def elapsed_text(meta, now_ns=None):
    if not meta or not meta.get('accepted_start'):
        return '경과 시간  --:--.-'
    end = meta.get('stop_ns')
    if end is None:
        end = time.monotonic_ns() if now_ns is None else now_ns
    tenths = max(0, (end - meta['start_ns']) // 100_000_000)
    minutes, rem = divmod(tenths, 600)
    seconds, tenth = divmod(rem, 10)
    return f'경과 시간  {minutes:02d}:{seconds:02d}.{tenth}'


class Window(QWidget):
    def __init__(self, node, bar, root):
        super().__init__()
        self.node, self.bar = node, bar
        self.hub = hub_root()
        self.placement = PlacementMetadata(self.hub)
        self.start_snapshot = None
        self.placement_retry = []
        self.tags = TagPanel(self.hub)
        self.pending = PendingTags(self.hub)
        self.pending_pools = {str(self.hub): self.pending}
        self.active_run = None
        self.active_profile = None
        self.hold_target = None
        self.held_ids = set()
        self.command_pending = {}
        self.live_timer = None
        self.start_request_ns = None
        self.selected = None
        self.last_scan = 0.0
        self.setWindowTitle('Real evaluation — SCENE / WRIST')
        self.resize(1400, 690)
        self.setMinimumSize(820, 510)
        self.setStyleSheet('''
            QWidget { background: #20242b; color: #edf1f5; font-size: 14px; }
            QPushButton { padding: 12px; border: 1px solid #687482; border-radius: 5px; font-weight: bold; }
            QPushButton:disabled { background: #353b43; color: #9098a1; }
            QLabel { padding: 3px; }
        ''')
        shell = QHBoxLayout(self)
        layout = QVBoxLayout()
        shell.addLayout(layout, 1)
        cameras = QHBoxLayout()
        self.images, self.headings = [], []
        for feed in (node.cam1, node.cam2):
            col = QVBoxLayout()
            heading = QLabel(feed.label)
            image = QLabel('카메라 수신 대기 중')
            image.setAlignment(Qt.AlignCenter)
            image.setMinimumSize(1, 120)
            image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
            col.addWidget(heading)
            col.addWidget(image, 1)
            cameras.addLayout(col, 1)
            self.images.append(image)
            self.headings.append(heading)
        self.overlay = OverlayPanel(self)
        overlay_scroll = QScrollArea()
        overlay_scroll.setWidgetResizable(True)
        overlay_scroll.setWidget(self.overlay)
        overlay_scroll.setFixedWidth(335)
        self.hub_panel = HubPanel(self.hub, publish_active, on_new_hub=self.follow_hub_task)
        self.hub_panel.setFixedWidth(335)
        right = QVBoxLayout()
        right.addWidget(self.hub_panel)
        right.addWidget(overlay_scroll, 1)
        shell.addLayout(right)
        layout.addLayout(cameras, 1)
        controls = QHBoxLayout()
        self.controls = []
        for controller, color in zip(bar._buttons, ('#17668b', '#a8630b', '#376f6b')):
            button = QPushButton(controller.label)
            button.setStyleSheet(f'QPushButton:enabled {{ background: {color}; }}')
            button.clicked.connect(lambda checked=False, c=controller: self.control_clicked(c))
            controls.addWidget(button)
            self.controls.append(button)
        layout.addLayout(controls)
        self.prompt_panel = PromptPanel(self)
        layout.addWidget(self.prompt_panel)
        # Prompt follows the overlay Task combo by default (VLA runs only; the
        # panel stays hidden for non-VLA launchers). Applied on next START.
        self.overlay.task.currentIndexChanged.connect(
            lambda *_: self.prompt_panel.sync_overlay_task(self.overlay.task.currentText()))
        self.prompt_panel.sync_overlay_task(self.overlay.task.currentText())
        self.response = QLabel('정책 서비스 연결 대기 중')
        self.response.setWordWrap(True)
        layout.addWidget(self.response)
        self.elapsed = QLabel(elapsed_text(None))
        self.elapsed.setStyleSheet('font-size: 26px; font-weight: bold;')
        layout.addWidget(self.elapsed)
        self.identity = QLabel(os.environ.get('REAL_EVAL_DISPLAY_PROFILE', '모델 정보: episode 대기 중'))
        self.identity.setWordWrap(True)
        layout.addWidget(self.identity)
        self.identity_run = None
        self.episode = QLabel('기록된 episode 대기 중')
        self.episode.setWordWrap(True)
        self.episode.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.episode)
        labels = QHBoxLayout()
        self.label_buttons = []
        for title, outcome, color in [('SUCCESS', 'success', '#28743e'),
                                      ('PARTIAL (0.5)', 'partial_success', '#807025'),
                                      ('FAILURE', 'failure', '#9c3838'),
                                      ('UNKNOWN', 'unknown', '#59616c')]:
            button = QPushButton(title)
            button.setStyleSheet(f'QPushButton:enabled {{ background: {color}; }}')
            if outcome == 'partial_success':
                button.setMaximumWidth(140)
                button.setToolTip('Partial success · success_score = 0.5')
            button.clicked.connect(lambda checked=False, value=outcome: self.label_episode(value))
            button.setEnabled(False)
            labels.addWidget(button)
            self.label_buttons.append(button)
        layout.addLayout(labels)
        self.message = QLabel(str(root or '녹화 경로 미설정 — 타이머 사용 불가'))
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.placement_status = QLabel('배치 기준: 다음 GUI START 시 저장')
        self.placement_status.setWordWrap(True)
        layout.addWidget(self.placement_status)
        self.log_path = None
        self.path_label = QLabel()
        self.path_label.setTextFormat(Qt.PlainText)
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        path_buttons = QHBoxLayout()
        self.copy_path = QPushButton('경로 복사')
        self.open_path = QPushButton('전체 로그 폴더 열기')
        self.copy_path.clicked.connect(self.copy_log_path)
        self.open_path.clicked.connect(self.open_log_path)
        path_buttons.addWidget(self.copy_path)
        path_buttons.addWidget(self.open_path)
        layout.addLayout(path_buttons)
        self.update_log_path()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)

    def update_log_path(self):
        if self.selected:
            run = Path(self.selected[2].get('metadata', {}).get('recording_run_dir') or self.selected[1].parent.parent)
            identity_key = (run, self.selected[2]['episode_id'])
            if identity_key != self.identity_run:
                try:
                    manifest = json.loads((run / 'launch_manifest.json').read_text())
                    policy = manifest['policy']
                    checkpoint = manifest['artifacts']['inputs']['checkpoint']['path']
                    stamp = re.search(r'run_(\d{8}T\d{6})', run.name)
                    local = (datetime.strptime(stamp[1], '%Y%m%dT%H%M%S')
                             .replace(tzinfo=timezone.utc).astimezone(ZoneInfo('Asia/Seoul'))
                             .strftime('%Y-%m-%d %H:%M:%S KST')) if stamp else '시간 미확인'
                    wall = self.selected[2].get('start_wall_ns')
                    episode_time = datetime.fromtimestamp(wall/1e9, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST') if wall else '기존 기록: START 날짜 없음'
                    self.identity.setText(f"{self.active_profile or policy.get('family', policy.get('type', '?'))} · checkpoint step {policy.get('step', '?')}\n{checkpoint}\nEpisode START: {episode_time} · CLI: {local}")
                    self.identity_run = identity_key
                except (OSError, ValueError, KeyError, TypeError):
                    self.identity.setText('모델 정보 확인 불가 — launch manifest를 확인하세요')
        self.log_path = (self.selected[1].absolute() if self.selected else
                         self.active_run if self.active_run else self.hub)
        prefix = '표시 중인 episode 저장 폴더' if self.selected else '로그 기본 경로 (episode 대기 중)'
        self.path_label.setText(f'{prefix}:\n{self.log_path}' if self.log_path else '로그 경로 미설정')
        self.copy_path.setEnabled(self.log_path is not None)
        self.open_path.setEnabled(self.hub.is_dir())
        self.open_path.setToolTip(str(self.hub))

    def copy_log_path(self):
        if self.log_path:
            QApplication.clipboard().setText(str(self.log_path))

    def open_log_path(self):
        if self.hub.is_dir():
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.hub))):
                self.message.setText('폴더를 열지 못했습니다. 경로 복사를 사용해 주세요.')

    def discover_run(self):
        active = read_active(self.hub)
        run = Path(active['run_dir']) if active else None
        if run == self.active_run:
            return
        self.active_run = run
        self.prompt_panel.sync_overlay_task(self.overlay.task.currentText())
        self.prompt_panel.set_run(run)
        self.active_profile = active['profile'] if active else None
        self.selected = None
        self.identity_run = None
        self.hold_target = None
        self.held_ids.clear()
        self.command_pending.clear()
        self.start_request_ns = None
        switched = self.hub_panel.consume_switch()
        if switched is not None:
            # Model hub switch: hide this arm's older episodes until the next START.
            self.start_request_ns = switched
        self.live_timer = None
        self.identity.setText(f"{self.active_profile} · 새 실행의 episode 대기 중" if active else '정책 CLI 실행 대기 중')
        # Normal runs share one durable queue; custom external roots retain their
        # own queue. Keep old queues ticking after switching policy launches.
        pending_root = self.hub if run is None or run.is_relative_to(self.hub) else run.parent
        key = str(pending_root)
        if key not in self.pending_pools:
            self.pending_pools[key] = PendingTags(pending_root)
        self.pending = self.pending_pools[key]
        self.pending.message = ''

    def control_clicked(self, controller):
        if controller.ready and not controller.in_flight():
            index = self.bar._buttons.index(controller)
            if index == 0:
                self.discover_run()
                if not self.prompt_panel.prepare_start():
                    return
                snapshot = self.overlay.snapshot()
                snapshot['camera_display'] = []
                for i, feed in enumerate((self.node.cam1, self.node.cam2)):
                    frame = feed.frame
                    ref = self.overlay.references[i] if self.overlay.references is not None else None
                    applied = bool(snapshot['enabled'] and snapshot['opacity'] > 0 and frame is not None and ref is not None
                                   and abs((frame.shape[1]/frame.shape[0])/(ref.shape[1]/ref.shape[0])-1) <= .01)
                    snapshot['camera_display'].append({'camera':i+1, 'overlay_applied':applied,
                                                      'feed_stalled':feed.is_stalled()})
            request_ns = time.monotonic_ns()
            self.command_pending[index] = (getattr(controller, 'result_expiry', 0), request_ns)
            if index == 0:
                self.start_snapshot = (self.active_run, request_ns, snapshot)
        if controller is self.bar._buttons[1] and self.selected and controller.ready and not controller.in_flight():
            self.hold_target = (self.selected[2]['episode_id'], time.monotonic())
        controller.clicked(time.monotonic())

    def label_episode(self, outcome):
        if not self.pending or not self.selected:
            return
        _, path, meta = self.selected
        if self.start_request_ns is not None and meta['start_ns'] < self.start_request_ns:
            return
        if 0 in self.command_pending:
            return
        if meta['status'] == 'recording' and meta.get('stop_ns') is None and meta['episode_id'] not in self.held_ids:
            return
        try:
            self.pending.enqueue(path, meta['episode_id'], outcome)
        except Exception as exc:
            self.pending.message = f'선택 저장 실패 — 다시 선택해 주세요: {exc}'
        self.message.setText(self.pending.message)

    def refresh(self):
        now = time.monotonic()
        self.bar.update(now, 1400)
        for index, (previous_result, request_ns) in list(self.command_pending.items()):
            controller = self.bar._buttons[index]
            if controller.in_flight() or getattr(controller, 'result_expiry', 0) == previous_result:
                continue
            del self.command_pending[index]
            if not controller.result_ok:
                if index == 0:
                    self.start_snapshot = None
                continue
            if index == 0:
                if self.start_snapshot and self.start_snapshot[1] == request_ns:
                    run, captured_ns, snapshot = self.start_snapshot
                    if run:
                        self.placement_retry.append((run, captured_ns, time.monotonic_ns(), snapshot))
                    else:
                        self.placement.message = '배치 기준 저장 불가: START 시 등록된 정책 run 없음'
                    self.start_snapshot = None
                self.start_request_ns = request_ns
                self.live_timer = {'accepted_start': True, 'start_ns': time.monotonic_ns(), 'stop_ns': None}
                self.selected = None
                self.hold_target = None
                if self.pending:
                    self.pending.message = ''
            elif self.live_timer and self.live_timer['stop_ns'] is None:
                self.live_timer['stop_ns'] = time.monotonic_ns()
        hold = self.bar._buttons[1]
        if self.hold_target and not hold.in_flight() and getattr(hold, 'result_expiry', 0) > self.hold_target[1]:
            if hold.result_ok:
                self.held_ids.add(self.hold_target[0])
            self.hold_target = None
        for button, controller in zip(self.controls, self.bar._buttons):
            button.setEnabled(controller.ready and not controller.in_flight())
            button.setText(controller.label + (' …' if controller.in_flight() else ''))
        responses = [c for c in self.bar._buttons if c.result_text]
        if responses:
            last = max(responses, key=lambda c: c.result_expiry)
            self.response.setText(f'{last.label}: {last.result_text}')
        for feed, label, heading in zip((self.node.cam1, self.node.cam2), self.images, self.headings):
            state = 'STALLED' if feed.is_stalled() else f'{feed.fps():.1f} fps'
            heading.setText(f'{feed.label}  |  {state if feed.frame is not None else "WAITING"}')
            heading.setStyleSheet('color: #ff7676;' if feed.is_stalled() else '')
            if feed.frame is not None:
                display = self.overlay.compose(feed.frame, self.images.index(label))
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
                label.setPixmap(QPixmap.fromImage(qimage).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.tags and not self.tags.busy and now-self.last_scan >= .2:
            self.discover_run()
            self.selected = run_episode(self.active_run) if self.active_run else None
            if (self.selected and self.start_request_ns is not None
                    and self.selected[2]['start_ns'] < self.start_request_ns):
                self.selected = None
            self.last_scan = now
            for pending in self.pending_pools.values():
                pending.tick()
            for item in list(self.placement_retry):
                try:
                    self.placement.enqueue(*item)
                    self.placement_retry.remove(item)
                except Exception as exc:
                    self.placement.message = '배치 기준 저장 재시도 중: '+str(exc)
            self.placement.tick()
        if self.placement.message:
            self.placement_status.setText(self.placement.message)
        meta = self.selected[2] if self.selected else None
        # START/HOLD responses drive the UI clock independently of recording
        # discovery. An old finalized episode must never overwrite a new timer.
        if (self.live_timer and self.live_timer['stop_ns'] is None and meta
                and meta['start_ns'] >= self.start_request_ns and meta.get('stop_ns') is not None):
            # Matching external HOLD/FAULT metadata may also stop the timer.
            self.live_timer['stop_ns'] = max(self.live_timer['start_ns'], meta['stop_ns'])
        self.elapsed.setText(elapsed_text(self.live_timer if self.live_timer is not None else meta))
        self.elapsed.setToolTip('GUI timer: START/HOLD approval responses. Precise recording times are in episode.json.')
        ready = False
        if not self.selected:
            self.episode.setText('현재 START의 녹화 episode 대기 중 — 계속 표시되면 GUI/정책 로그 경로를 확인하세요')
        if self.selected:
            _, path, meta = self.selected
            locked = (path/'.tag.lock').exists()
            self.episode.setText(f'{path.name}\n{meta["status"]}' + (' · 영상 생성/저장 중' if locked else '') +
                                 f' · {meta.get("termination_reason") or "실행 중"}')
            ready = (meta['status'] in ('ready', 'invalid', 'partial') or
                     meta.get('stop_ns') is not None or meta['episode_id'] in self.held_ids)
        for button in self.label_buttons:
            button.setEnabled(ready and 0 not in self.command_pending)
        if self.tags:
            self.message.setText(self.pending.message or self.tags.message)
        self.hub_panel.tick(self.hub_block_reason())
        self.update_log_path()

    def follow_hub_task(self, task):
        # A newly started model hub sets the overlay Task to the hub's task (user can still change it).
        index = self.overlay.task.findText(task)
        if index >= 0:
            self.overlay.task.setCurrentIndex(index)

    def hub_block_reason(self):
        if 0 in self.command_pending:
            return 'START 처리 중'
        if self.live_timer and self.live_timer['stop_ns'] is None:
            return 'episode 실행 중'
        if self.selected:
            _, path, meta = self.selected
            if (meta.get('accepted_start') and path.name.endswith('__unlabeled') and self.pending is not None
                    and not (self.pending.directory / (meta['episode_id'] + '.json')).exists()):
                return '직전 episode 라벨을 먼저 달아 주세요'
        return None


def run(args, node_class, bar_class):
    import rclpy
    os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv[:1])
    rclpy.init()
    node = node_class(args)
    window = Window(node, bar_class(node), args.eval_log_root)
    pump = QTimer()
    def spin():
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)
        else:
            app.quit()
    pump.timeout.connect(spin)
    pump.start(5)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    window.show()
    try:
        return app.exec_()
    finally:
        pump.stop()
        window.timer.stop()
        window.overlay.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
