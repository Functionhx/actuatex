"""Launch Nav2 and the TinyMal velocity safety adapter."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("actuatex_navigation"))
    nav2_share = Path(get_package_share_directory("nav2_bringup"))

    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("rviz")

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(nav2_share / "launch" / "bringup_launch.py")),
        launch_arguments={
            "map": map_file,
            "params_file": params_file,
            "use_sim_time": use_sim_time,
            "autostart": "True",
            "slam": "False",
            # Humble's bringup launch inserts this value into a PythonExpression.
            # Keep the Python spelling rather than the lowercase ROS boolean.
            "use_composition": "False",
        }.items(),
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(nav2_share / "launch" / "rviz_launch.py")),
        condition=IfCondition(use_rviz),
        launch_arguments={"use_namespace": "false"}.items(),
    )
    command_adapter = Node(
        package="actuatex_navigation",
        executable="cmd_vel_adapter",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value=str(package_share / "maps" / "tinymal_nav_arena.yaml"),
                description="Absolute path to a Nav2 map YAML file.",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=str(package_share / "config" / "nav2_params.yaml"),
                description="ActuateX Nav2 parameter file.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            nav2,
            command_adapter,
            rviz,
        ]
    )
