from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('ocelot'),
        'config', 'tracker_params.yaml'
    )

    return LaunchDescription([
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
        ),
    ])
