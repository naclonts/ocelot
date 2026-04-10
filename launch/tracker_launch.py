from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('ocelot'),
        'config', 'tracker_params.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'record',
            default_value='false',
            description='Record /camera/image_raw and /cmd_vel to /ws/bags/',
        ),
        DeclareLaunchArgument(
            'visualize',
            default_value='false',
            description='Run visualizer_node → /camera/image_annotated',
        ),
        DeclareLaunchArgument(
            'use_haar',
            default_value='false',
            description='Use classical Haar cascade tracker instead of VLA model.',
        ),
        DeclareLaunchArgument(
            'use_remote_vla',
            default_value='false',
            description='Send frames to a remote HTTP VLA server instead of local ONNX.',
        ),
        DeclareLaunchArgument(
            'vla_checkpoint',
            default_value='/ws/src/ocelot/models/active.onnx',
            description='Path to the ONNX model (symlink → models/active.onnx by default).',
        ),
        DeclareLaunchArgument(
            'vla_command',
            default_value='track the face',
            description='Language command passed to the VLA node.',
        ),
        DeclareLaunchArgument(
            'vla_deadband',
            default_value='0.03',
            description='Deadband threshold (rad/s) — model outputs below this are zeroed.',
        ),
        DeclareLaunchArgument(
            'remote_vla_url',
            default_value='http://127.0.0.1:8765/infer',
            description='HTTP endpoint for offboard VLA inference.',
        ),

        Node(
            package='ocelot',
            executable='camera_node',
            name='camera_node',
            parameters=[params],
        ),
        Node(
            package='ocelot',
            executable='servo_node',
            name='servo_node',
            parameters=[params],
        ),
        Node(
            package='ocelot',
            executable='tracker_node',
            name='tracker_node',
            parameters=[params],
            condition=IfCondition(LaunchConfiguration('use_haar')),
        ),
        Node(
            package='ocelot',
            executable='vla_node',
            name='vla_node',
            parameters=[{
                'checkpoint':  LaunchConfiguration('vla_checkpoint'),
                'token_cache': '/ws/src/ocelot/models/active_tokens.json',
                'command':     LaunchConfiguration('vla_command'),
                'deadband':    LaunchConfiguration('vla_deadband'),
                'enabled':     True,
            }],
            condition=IfCondition(PythonExpression([
                "'",
                LaunchConfiguration('use_haar'),
                "' == 'false' and '",
                LaunchConfiguration('use_remote_vla'),
                "' == 'false'",
            ])),
        ),
        Node(
            package='ocelot',
            executable='remote_vla_client_node',
            name='remote_vla_client_node',
            parameters=[{
                'server_url': LaunchConfiguration('remote_vla_url'),
                'command': LaunchConfiguration('vla_command'),
                'deadband': LaunchConfiguration('vla_deadband'),
                'enabled': True,
            }],
            condition=IfCondition(LaunchConfiguration('use_remote_vla')),
        ),
        Node(
            package='web_video_server',
            executable='web_video_server',
            name='web_video_server',
            parameters=[{'port': 8080}],
        ),

        Node(
            package='ocelot',
            executable='visualizer_node',
            name='visualizer_node',
            parameters=[params],
            condition=IfCondition(LaunchConfiguration('visualize')),
        ),

        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '--storage', 'mcap',
                '--compression-mode', 'file',
                '--compression-format', 'zstd',
                '/camera/image_raw',
                '/cmd_vel',
                '/tracking/face_roi',
            ],
            cwd='/ws/bags',
            condition=IfCondition(LaunchConfiguration('record')),
        ),
    ])
