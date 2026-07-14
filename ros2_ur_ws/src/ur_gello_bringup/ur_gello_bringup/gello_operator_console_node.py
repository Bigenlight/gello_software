#!/usr/bin/env python3
"""Interactive numbered-menu operator console for the init_align handshake.

Run this in a SECOND terminal while the real teleop launch is running with
``start_mode:=init_align``. It shows a simple numbered menu; the operator just
types a number to authorize each safety gate. Under the hood it calls the
``gello_move_to_start`` Trigger services (which work reliably inside ros2
launch, unlike stdin) and prints the response — including the per-joint
alignment report when a handover is refused.

Menu
----
    1) 진행 (proceed)        -> /gello_move_to_start/proceed
    2) 정지 (abort)          -> /gello_move_to_start/abort
    3) 강제 진행 (override)  -> /gello_move_to_start/override_follow
    q) quit the console (does NOT stop the robot; use Ctrl-C / E-STOP for that)

This console is READ/authorize-only: it never commands the robot or GELLO
directly; it only relays the operator's explicit authorization to the
move-to-start node, which owns all safety checks.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

# Node names whose services we drive: the handshake node (gates) and the
# streaming bridge (pause/resume of following).
TARGET = "gello_move_to_start"
BRIDGE = "gello_ur_bridge"
GRIPPER = "gello_gripper_bridge"

MENU = """
========== GELLO ↔ UR7e 조작 콘솔 ==========
  1) 진행/재개   — 핸드셰이크 게이트 통과 / 팔로잉 중이면 재개
  2) 정지/일시정지 — 핸드셰이크 중단 / 팔로잉 중이면 멈춤(로봇 제자리 유지)
  3) 강제 진행   — 오차 감수하고 팔로잉 (핸드셰이크, 안전 한계 이내만)
  4) UR 홈으로   — 로봇을 init pose 로 다시 이동 (핸드셰이크)
  5) 차이 계산   — GELLO↔홈 관절별 오차만 보고 (핸드셰이크)
  6) 재개(글라이드) — 팔로잉 재개, 팔이 GELLO 로 미끄러지듯 이동 (간격 제한/정지 게이트)
  7) 그리퍼 일시정지 — 그리퍼 출력 정지 (Robotiq 현재 위치 유지)
  8) 그리퍼 재개   — 실제 위치에서 시드→램프 (신선/실제위치 게이트, 실패시 정지 유지)
  q) 콘솔 종료 (로봇은 안 멈춤 — 급정지는 Ctrl-C / E-STOP)
============================================
선택 > """


class OperatorConsole(Node):
    """Numbered-menu front-end that calls the move-to-start gate services."""

    def __init__(self) -> None:
        super().__init__("gello_operator_console")
        # NOTE: do NOT name this `self._clients` — rclpy.Node uses that internally
        # for its client list; shadowing it breaks the executor (str has no
        # _executor_event) and destroy_node (KeyError 0).
        self._svc = {
            "proceed": self.create_client(Trigger, f"/{TARGET}/proceed"),
            "abort": self.create_client(Trigger, f"/{TARGET}/abort"),
            "override_follow": self.create_client(
                Trigger, f"/{TARGET}/override_follow"
            ),
            "go_home": self.create_client(Trigger, f"/{TARGET}/go_home"),
            "check_alignment": self.create_client(
                Trigger, f"/{TARGET}/check_alignment"
            ),
            # Streaming-phase (bridge) pause/resume of following.
            "resume": self.create_client(Trigger, f"/{BRIDGE}/resume"),
            "pause": self.create_client(Trigger, f"/{BRIDGE}/pause"),
            # Bridge resume-CHASE: gated glide across a bounded gap (fallback for
            # when the strict ~/resume refuses a larger — but safe — offset).
            "resume_chase": self.create_client(
                Trigger, f"/{BRIDGE}/resume_chase"
            ),
            # Gripper bridge pause/resume (drop-hazard mitigation). Resume is
            # fail-closed: refuses (staying paused & silent) on a stale leader or
            # unknown actual position.
            "gripper_pause": self.create_client(
                Trigger, f"/{GRIPPER}/pause"
            ),
            "gripper_resume": self.create_client(
                Trigger, f"/{GRIPPER}/resume"
            ),
        }

    def call_first(self, names: list[str]) -> None:
        """Call the FIRST available service among ``names`` (phase-aware routing).

        Lets 1 and 2 mean go/stop in BOTH phases: during the handshake they hit
        gello_move_to_start (proceed/abort); once following they hit the bridge
        (resume/pause). Whichever node is alive answers.
        """
        for name in names:
            if self._svc[name].service_is_ready() or self._svc[name].wait_for_service(
                timeout_sec=1.0
            ):
                self.call(name)
                return
        print(
            "[!] 해당 서비스가 지금 단계에 없습니다. 텔레오퍼(터미널1)가 실행 중이고 "
            "핸드셰이크 게이트 또는 팔로잉 상태인지 확인하세요."
        )

    def call(self, name: str) -> None:
        """Call one Trigger service and print its response."""
        cli = self._svc[name]
        if not cli.wait_for_service(timeout_sec=3.0):
            print(
                f"[!] 서비스 {name} 를 찾을 수 없습니다. 텔레오퍼가 실행 중이고 "
                "노드가 해당 단계에 도달했는지 확인하세요."
            )
            return
        future = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        resp = future.result()
        if resp is None:
            print("[!] 응답이 없습니다 (타임아웃).")
            return
        # ✅ only for a genuinely good outcome. Treat a not-aligned / refused
        # report as ❌ even if the service itself returned success=True (the
        # check_alignment query "succeeds" but the state is not-OK).
        low = resp.message.lower()
        bad = (
            not resp.success
            or low.startswith("not yet aligned")
            or low.startswith("not aligned")
            or "refused" in low
            or "not within tolerance" in low
        )
        mark = "❌" if bad else "✅"
        print(f"{mark} {resp.message}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OperatorConsole()
    print(
        "GELLO ↔ UR7e 조작 콘솔을 시작합니다. 텔레오퍼(터미널 1)의 게이트 "
        "안내에 맞춰 숫자를 입력하세요."
    )
    try:
        while rclpy.ok():
            try:
                choice = input(MENU).strip().lower()
            except EOFError:
                break
            if choice in ("q", "quit", "exit"):
                print("콘솔을 종료합니다. (로봇은 계속 동작 중 — 필요하면 터미널 1에서 Ctrl-C)")
                break
            if choice == "1":
                node.call_first(["proceed", "resume"])   # 진행(핸드셰이크) / 재개(팔로잉)
            elif choice == "2":
                node.call_first(["abort", "pause"])       # 중단(핸드셰이크) / 일시정지(팔로잉)
            elif choice == "3":
                node.call("override_follow")
            elif choice == "4":
                node.call("go_home")
            elif choice == "5":
                node.call("check_alignment")
            elif choice == "6":
                node.call("resume_chase")   # 재개(글라이드) — 간격 제한 게이트
            elif choice == "7":
                node.call("gripper_pause")  # 그리퍼 일시정지 (출력 정지)
            elif choice == "8":
                node.call("gripper_resume")  # 그리퍼 재개 (시드→램프, 실패시 정지)
            else:
                print("1, 2, 3, 4, 5, 6, 7, 8, q 중에서 입력하세요.")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
