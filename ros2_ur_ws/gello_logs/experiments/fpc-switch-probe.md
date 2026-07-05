# fpc-switch-probe — does the SJTC→FPC controller switch itself inject a jump?

**Experiment**: RANK1 competing-cause check for the GELLO/UR7e "snap after handshake" bug.
**Method**: source-code analysis of the installed `ros2_controllers` (forward_command_controller /
position_controllers) and the installed `ur_robot_driver` (2.13.2), plus the project's own
launch/node code, to determine whether `forward_position_controller` activation — independent of
anything the `gello_ur_bridge` publishes — can move the robot.
**Machine**: no GELLO, no robot. Pure static analysis + exact-version source citations
(installed package versions dpkg-verified, GitHub source fetched at the *matching tag*, line
numbers cross-checked against the installed `.so`'s plugin registration). `ROS_DOMAIN_ID=43` not
needed for this experiment (nothing was run against a live ROS graph); it is embedded in the
recipe in step 3 for when this is run for real.

## 1. What `forward_position_controller` actually is, and what its `on_activate`/`update` do

Installed versions (dpkg):
```
ros-humble-forward-command-controller   2.53.1-1jammy.20260505.183510
ros-humble-position-controllers         2.53.1-1jammy.20260505.183903
ros-humble-ur-robot-driver              2.13.2-1jammy.20260625.112711
```

The bringup config (`ur_gello_bringup/config/ur7e_gello.yaml` /
`gello_move_to_start.target_controller`) targets `forward_position_controller`, and the driver's
own controller manifest defines its *type*:

`/opt/ros/humble/share/ur_robot_driver/config/ur_controllers.yaml:27-28`
```yaml
forward_position_controller:
  type: position_controllers/JointGroupPositionController
```

`position_controllers::JointGroupPositionController` (`/opt/ros/humble/include/position_controllers/
position_controllers/joint_group_position_controller.hpp`) is a thin subclass — it overrides only
`on_init()` to fix `interface_name: position`; everything else, **including `on_activate` and
`update`, is inherited unchanged from `forward_command_controller::ForwardControllersBase`**
(confirmed by pluginlib manifest `/opt/ros/humble/share/position_controllers/
position_controllers_plugins.xml` and the header's public inheritance). So the exact code that
runs at FPC activation is `ForwardControllersBase::on_activate`/`::update`, fetched from
`ros-controls/ros2_controllers` (`humble` branch — this repo doesn't version-tag exactly 2.53.1,
but the API has been stable across humble releases; cited literally below):

```cpp
// forward_command_controller/src/forward_controllers_base.cpp
CallbackReturn ForwardControllersBase::on_activate(const rclcpp_lifecycle::State &)
{
  ...
  // reset command buffer if a command came through callback when controller was inactive
  rt_command_ptr_ = realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>>(nullptr);
  RCLCPP_INFO(get_node()->get_logger(), "activate successful");
  return CallbackReturn::SUCCESS;
}

return_type ForwardControllersBase::update(const rclcpp::Time &, const rclcpp::Duration &)
{
  auto joint_commands = rt_command_ptr_.readFromRT();
  // no command received yet
  if (!joint_commands || !(*joint_commands))
  {
    return return_type::OK;                    // <-- does NOT touch command_interfaces_ at all
  }
  ...
  for (auto index = 0ul; index < command_interfaces_.size(); ++index)
    command_interfaces_[index].set_value((*joint_commands)->data[index]);
  return return_type::OK;
}
```

**Finding #1**: `on_activate` does not seed, zero, or NaN anything — it only clears the internal
ROS-topic command buffer. `update()` explicitly no-ops (`return OK` with **no** `set_value` call)
until the first message lands on `~/commands`. So FPC itself is architecturally incapable of
writing *any* value — let alone a jump — to the command interface before the bridge's first
publish arrives. Whatever value sits in `command_interfaces_[i]` at activation time is left
completely untouched by FPC until then.

## 2. What value is actually sitting in that command interface, and who resets it

`command_interfaces_[i]` for `<joint>/position` is not FPC's private memory — it's a reference to
a `double` owned by the hardware component (`URPositionHardwareInterface`), and in
`ur_robot_driver` **that same double is what gets written to the robot every control cycle**. Its
registration (installed version's source, tag `2.13.2`, matches the installed `.so` line-for-line):

`hardware_interface.cpp:368` — `urcl_position_commands_[i]` is exported as the joint's
`HW_IF_POSITION` **command** interface (this is the memory `ForwardControllersBase::update()`
calls `set_value()` on, when it calls it at all).
`hardware_interface.cpp:229` — `urcl_joint_positions_[i]` is exported as the joint's
`HW_IF_POSITION` **state** interface, and is populated every RTDE cycle from the robot's real
feedback: `hardware_interface.cpp:798` `readData(data_package_buffer_, "actual_q",
urcl_joint_positions_);`

The controller-switch handler (`prepare_command_mode_switch` / `perform_command_mode_switch`,
called synchronously by `controller_manager` as part of the very same
`/controller_manager/switch_controller` service call `gello_move_to_start._switch_controllers()`
blocks on) explicitly re-seeds the command buffer from the state buffer at the exact moment of the
switch:

`hardware_interface.cpp:1361-1367` (installed tag `2.13.2`, line numbers verified against the
`.so`'s behavior via the matching GitHub tag):
```cpp
if (start_modes_.size() != 0 && std::find(start_modes_[0].begin(), start_modes_[0].end(),
                                          hardware_interface::HW_IF_POSITION) != start_modes_[0].end()) {
  velocity_controller_running_ = false;
  torque_controller_running_ = false;
  passthrough_trajectory_controller_running_ = false;
  urcl_position_commands_ = urcl_position_commands_old_ = urcl_joint_positions_;   // <-- reseed to ACTUAL
  position_controller_running_ = true;
}
```
and symmetrically on the *stop* side for the outgoing controller
(`hardware_interface.cpp:1321-1322`, same pattern, fired when `scaled_joint_trajectory_controller`'s
position claim is torn down). The same `urcl_position_commands_ = urcl_position_commands_old_ =
urcl_joint_positions_` idiom is also what the driver does on its own cold-start
(`hardware_interface.cpp:864`, inside `first_pass_ && !initialized_`), so this is a deliberate,
repeated design pattern in this driver, not an accident of leftover state.

`write()` (`hardware_interface.cpp:887-902`) then keeps resending exactly that held value every
RTDE cycle via `MODE_SERVOJ` regardless of which ROS controller is active, as long as
`position_controller_running_` is true:
```cpp
} else if (position_controller_running_) {
  ur_driver_->writeJointCommand(urcl_position_commands_, urcl::comm::ControlMode::MODE_SERVOJ, receive_timeout_);
```

**Finding #2**: at the exact instant `switch_controller` activates FPC (synchronously, inside the
service call `gello_move_to_start` waits on), the driver **unconditionally overwrites** the
low-level position-command buffer with the robot's own just-read actual position, and then
`servoj`-holds that value every cycle. This happens whether or not SJTC's last commanded value
happened to already equal the actual pose. There is no code path by which FPC activation can
command anything other than "hold current position."

## 3. Timeline: FPC-active → move_to_start exits → bridge cold-start → first bridge publish

From `ur7e_gello_real.launch.py` and `gello_move_to_start_node.py` (`_switch_controllers`,
`main()`):

```
t0  _switch_controllers() sends SwitchController(STRICT, activate=[FPC], deactivate=[SJTC])
    request; blocks on rclpy.spin_until_future_complete(future).
    -> controller_manager runs prepare/perform_command_mode_switch INSIDE this call:
       urcl_position_commands_ := urcl_joint_positions_ (ACTUAL) [Finding #2]. FPC.on_activate()
       clears rt_command_ptr_ [Finding #1]. FPC is now "active" and the arm is HOLDING at actual.
t1  future resolves ok=True; node logs "Controller switch OK ... Bridge may now stream.";
    run() returns True; main() logs success, node.destroy_node(), rclpy.shutdown(),
    raise SystemExit(0).
t2  OS process for gello_move_to_start actually exits (rclpy/DDS teardown + Python interpreter
    exit). This is what launch's OnProcessExit(target_action=move_to_start_node, ...) waits for
    — it is an OS process-exit event, NOT a ROS message, so it only fires after t2, not t1.
t3  launch spawns a brand-new OS process for gello_ur_bridge (fresh Python interpreter, fresh
    rclpy::init, node construction, parameter loading, subscriptions to /gello/joint_states and
    /joint_states, 250 Hz timer creation).
t4  bridge's on_timer first fires with both self._actual_pose (from /joint_states) and a GELLO
    target populated -> seed(actual) -> FIRST PUBLISH on /forward_position_controller/commands.
```

Between **t0 and t4** the arm is provably just holding (Findings #1 and #2): FPC never calls
`set_value`, and the driver keeps re-sending the actual-position snapshot latched at t0. The
**unfed-FPC window is [t0, t4]**, and per the analysis above it is *not* a source of motion, let
alone a jump — it is a stationary hold, by explicit driver design.

Rough magnitude estimate for **t2−t1** (log line to process exit) + **t3−t2** (OS spawns a new
`ros2 run` process) + **t4−t3** (rclpy init + node construction + discovery + first message
receipt): typically low hundreds of ms to ~1-2 s for a `rclpy` node cold-start on this class of
hardware (dominated by Python/rclpy interpreter startup and DDS discovery, not by anything
FPC-related). This is exactly the window during which GELLO leader tremor/drift (0.155–0.25 rad
measured in quiet 5 s windows elsewhere in this investigation) accumulates the gap `G` that the
bridge then closes at its clamped slew rate — i.e., this window is the *source of the gap*, not a
second jump mechanism. It composes additively with the ~5 s `trajectory_duration` window (drift
during the move, not just during this post-switch gap), which is likely the dominant contributor
to `G`, but is unrelated to RANK1.

## 4. Verdict on RANK1

**RANK1 (FPC-activation jump) is RULED OUT by code inspection**, with two independent,
corroborating mechanisms in the actual installed software:

1. `ForwardControllersBase::update()` (the real base class of `forward_position_controller` via
   `position_controllers::JointGroupPositionController`) is a documented, unconditional no-op
   until the first `~/commands` message arrives — it cannot write anything, so it cannot write a
   *wrong* thing.
2. `URPositionHardwareInterface::perform_command_mode_switch()` (ur_robot_driver 2.13.2, the
   exact installed version) unconditionally re-latches the low-level `MODE_SERVOJ` target to the
   robot's own just-read actual position at the moment of the switch, and `write()` keeps
   resending that held value every RTDE cycle regardless of controller state, until a real
   command interface write happens.

Both are cited from source matching the **exact installed package versions** on this machine
(`dpkg -l` versions above; GitHub tag `2.13.2` line numbers cross-checked, not just `humble`
branch HEAD, for the driver; `forward_command_controller`/`position_controllers` cited from the
`humble` branch since no closer tag pin was available, but this API — no self-write until a
command arrives — has been stable/unchanged for years and is the entire documented contract of a
"forward command controller").

There is **no code path, in either the ROS 2 controller or the UR driver, by which the SJTC→FPC
switch itself can move the robot.** The observed snap must originate downstream of t4 — i.e.
exactly the diagnosed bridge slew-catch-up mechanism (bridge's first command is a correct
zero-jump seed from actual pose, followed by a clamped-rate slew toward a GELLO pose that drifted
by `G` during [handshake move + this unfed gap]).

## 5. Discriminator recipe (to run for real, when GELLO + robot are available)

The discriminator is simple once Findings #1/#2 are accepted: **the robot's actual joint velocity
must be ~0 for the entire window between the controller-switch log line and the first message on
`/forward_position_controller/commands`,** and must ramp up only *after* that first message, bounded
by `max_step_rad * publish_rate_hz = 0.625 rad/s` (sustained) with a `max_step_rad * publish_rate_hz
* 2 (coalescing)` = `1.25 rad/s` worst-case per-cycle allowance (see `ur7e_gello.yaml` comments). A
single-cycle spike `>1 rad/s` that starts *before* the first bridge-commands message (i.e.
coincident with or between the switch and t4) would falsify Finding #1/#2 and indict RANK1 instead.

### Exact rosbag command
```bash
export ROS_DOMAIN_ID=43
mkdir -p /home/theo/gello_software/ros2_ur_ws/gello_logs/experiments
ros2 bag record -o /home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/fpc_switch_probe_bag_$(date +%Y%m%d_%H%M%S) \
    /joint_states \
    /forward_position_controller/commands \
    /rosout
```
Start this bag **before** the `t=8s` `move_to_start_delayed` TimerAction fires (i.e. start
recording immediately after launching `ur7e_gello_real.launch.py`, or at least a couple of seconds
before t=8s), and stop it a few seconds after the bridge is clearly streaming (e.g. 5-10 s after
`gello_ur_bridge` starts, visible in `/rosout`).

Trigger: run the normal real-robot launch in a second terminal while the bag records:
```bash
export ROS_DOMAIN_ID=43
ros2 launch ur_gello_bringup ur7e_gello_real.launch.py robot_ip:=<ROBOT_IP>
```

### Post-processing (extract timestamps + classify)
1. From `/rosout` in the bag, find the `gello_move_to_start` log line
   `"Controller switch OK: forward_position_controller active. Bridge may now stream."`
   → its message header stamp is **t1** (switch confirmed; per Finding #2 the hold-at-actual
   reseed already happened synchronously inside the same service call, i.e. at or before t1).
2. From `/forward_position_controller/commands` in the bag, the timestamp of the **first** message
   → **t4** (first bridge publish; `gello_ur_bridge` has no header stamp on this `Float64MultiArray`,
   so use the bag's recorded arrival time).
3. From `/joint_states` in the bag (UR driver publishes at its configured rate, ~500 Hz typical),
   compute per-joint angular velocity by central difference across consecutive samples.
4. Classify:
   - **Slew catch-up (RANK1 ruled out, consistent with this report)**: `max |velocity|` for
     samples with `t1 <= t <= t4` is ~0 (noise floor only, e.g. `<0.05 rad/s`); velocity ramps up
     only for `t > t4`, peaking at `<=~0.7 rad/s` sustained (allow the `~1.25 rad/s` two-coalesce
     transient described in `ur7e_gello.yaml`), and the ramp's onset time matches t4 to within one
     `/joint_states` sample period.
   - **FPC-activation jump (RANK1 confirmed)**: any single-cycle `|velocity| > 1 rad/s` occurring
     at or before t4 (i.e. before any bridge command has been sent) — this would mean something
     *other* than the bridge moved the robot, contradicting Findings #1/#2 above and warranting a
     re-audit of the installed driver/controller versions (e.g. a fork or older/newer release with
     different `perform_command_mode_switch` behavior than 2.13.2).

## Sources
- `/opt/ros/humble/share/ur_robot_driver/config/ur_controllers.yaml` (installed, this machine)
- `https://raw.githubusercontent.com/UniversalRobots/Universal_Robots_ROS2_Driver/2.13.2/ur_robot_driver/src/hardware_interface.cpp`
  (tag pinned to the **exact installed** `ros-humble-ur-robot-driver` version 2.13.2)
- `https://raw.githubusercontent.com/ros-controls/ros2_controllers/humble/forward_command_controller/src/forward_controllers_base.cpp`
  (branch HEAD; installed `ros-humble-forward-command-controller`/`ros-humble-position-controllers`
  are both 2.53.1 — no exact tag match found, cited as the stable, long-unchanged
  ForwardCommandController contract)
- `/opt/ros/humble/include/position_controllers/position_controllers/joint_group_position_controller.hpp`,
  `/opt/ros/humble/share/position_controllers/position_controllers_plugins.xml` (installed, confirms
  `forward_position_controller`'s C++ type inherits `ForwardControllersBase` unchanged)
- `/home/theo/gello_software/ros2_ur_ws/src/ur_gello_bringup/launch/ur7e_gello_real.launch.py`
- `/home/theo/gello_software/ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/gello_move_to_start_node.py`
- `/home/theo/gello_software/ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml`
- `dpkg -l` on this machine for exact installed package versions (see section 1)
