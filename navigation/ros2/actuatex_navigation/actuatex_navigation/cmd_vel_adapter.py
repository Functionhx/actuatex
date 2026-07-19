"""ROS 2 adapter from Nav2 velocity output to the TinyMal policy command."""

from __future__ import annotations

import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node

from .command_filter import CommandFilter, VelocityCommand, VelocityLimits


class CmdVelAdapter(Node):
    """Rate-limit Nav2 commands and stop when the command stream becomes stale."""

    def __init__(self) -> None:
        super().__init__("cmd_vel_adapter")

        self.declare_parameter("input_topic", "/cmd_vel")
        self.declare_parameter("output_topic", "/actuatex/policy_cmd")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("command_timeout_s", 0.30)
        for name, value in VelocityLimits().__dict__.items():
            self.declare_parameter(name, value)

        publish_rate = float(self.get_parameter("publish_rate_hz").value)
        self._timeout = float(self.get_parameter("command_timeout_s").value)
        if publish_rate <= 0.0 or self._timeout <= 0.0:
            raise ValueError("publish_rate_hz and command_timeout_s must be positive")

        limits = VelocityLimits(
            **{
                name: float(self.get_parameter(name).value)
                for name in VelocityLimits().__dict__
            }
        )
        self._filter = CommandFilter(limits)
        self._target = VelocityCommand.zero()
        self._last_command_time: float | None = None
        self._last_tick_time = time.monotonic()
        self._was_stale = True

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._publisher = self.create_publisher(Twist, output_topic, 10)
        self._subscription = self.create_subscription(
            Twist, input_topic, self._on_command, 10
        )
        self._timer = self.create_timer(1.0 / publish_rate, self._on_timer)
        self.get_logger().info(
            f"safe command path: {input_topic} -> {output_topic} at {publish_rate:g} Hz"
        )

    def _on_command(self, message: Twist) -> None:
        self._target = VelocityCommand(
            x=float(message.linear.x),
            y=float(message.linear.y),
            yaw=float(message.angular.z),
        )
        self._last_command_time = time.monotonic()

    def _publish(self, command: VelocityCommand) -> None:
        message = Twist()
        message.linear.x = command.x
        message.linear.y = command.y
        message.angular.z = command.yaw
        self._publisher.publish(message)

    def _on_timer(self) -> None:
        now = time.monotonic()
        dt = max(1.0e-4, min(now - self._last_tick_time, 0.10))
        self._last_tick_time = now
        stale = (
            self._last_command_time is None
            or now - self._last_command_time > self._timeout
        )
        if stale:
            command = self._filter.stop()
            if not self._was_stale:
                self.get_logger().warning("command watchdog expired; publishing zero")
        else:
            command = self._filter.update(self._target, dt)
        self._was_stale = stale
        self._publish(command)

    def publish_stop(self) -> None:
        self._publish(self._filter.stop())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Humble's default signal handler may invalidate the context before
        # ``spin`` returns.  The downstream simulator watchdog still stops the
        # robot, and publishing on an invalid context would mask a clean exit.
        if rclpy.ok():
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
