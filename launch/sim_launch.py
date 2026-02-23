import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('ocelot')
    headless = LaunchConfiguration('headless').perform(context)
    use_oracle = LaunchConfiguration('use_oracle').perform(context) == 'true'

    urdf_path = os.path.join(pkg, 'urdf', 'pan_tilt.urdf')
    controllers_yaml = os.path.join(pkg, 'config', 'controllers.yaml')
    tracker_params = os.path.join(pkg, 'config', 'tracker_params.yaml')
    world_file = os.path.join(pkg, 'sim', 'worlds', 'tracker_world.sdf')

    # GZ_SIM_RESOURCE_PATH: Gazebo searches these directories for model://
    # URIs.  The source path is listed first so generated textures (written
    # by sim/generate_face_texture.py to the bind-mounted source tree) take
    # priority over any installed copies that may be missing the PNG.
    source_models = '/ws/src/ocelot/sim/models'           # bind-mount path inside container
    installed_models = os.path.join(pkg, 'sim', 'models') # colcon install path
    gz_resource = source_models + ':' + installed_models
    existing_gz = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if existing_gz:
        gz_resource = gz_resource + ':' + existing_gz
    set_gz_resource = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource)

    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    # Gazebo rejects base_link because it has no <inertial> (it's the fixed
    # root mount point — zero mass breaks URDF→SDF conversion).  Inject a
    # minimal inertial block so Gazebo keeps the link and its child joints.
    # The real hardware stack ignores this because it never parses the URDF
    # for physics; it reads joint configs from the servo driver directly.
    base_link_fix = (
        '<link name="base_link">\n'
        '    <inertial>\n'
        '      <mass value="0.001"/>\n'
        '      <inertia ixx="1e-9" ixy="0" ixz="0"'
        ' iyy="1e-9" iyz="0" izz="1e-9"/>\n'
        '    </inertial>\n'
        '  </link>'
    )
    robot_description = urdf_content.replace(
        '<link name="base_link"/>', base_link_fix
    )

    # Inject the gz_ros2_control Gazebo system plugin at launch time so the
    # URDF file itself remains unmodified (the real hardware stack ignores it).
    plugin_block = (
        '  <gazebo>\n'
        '    <plugin filename="gz_ros2_control-system"'
        ' name="gz_ros2_control::GazeboSimROS2ControlPlugin">\n'
        f'      <parameters>{controllers_yaml}</parameters>\n'
        '    </plugin>\n'
        '  </gazebo>\n'
    )
    robot_description = robot_description.replace('</robot>', plugin_block + '</robot>')

    # -r: run simulation immediately; -s: server only (no GUI) when headless
    gz_args = f'-r -s {world_file}' if headless == 'true' else f'-r {world_file}'

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            )
        ),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen',
    )

    # Spawn the robot model from the /robot_description topic published by RSP.
    # The injected gz_ros2_control plugin inside the description activates
    # GazeboSimSystem and starts the controller_manager when the model loads.
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'ocelot', '-topic', 'robot_description'],
        output='screen',
    )

    # Bridge Gazebo → ROS for simulation-time clock, camera image, and face pose.
    # joint_state_broadcaster publishes /joint_states directly to ROS via
    # ros2_control — no additional bridge is needed for that topic.
    # /model/face_0/pose: published by the PosePublisher plugin in model.sdf;
    # gives oracle_node the ground-truth world position of the face billboard.
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/model/face_0/pose@geometry_msgs/msg/Pose[gz.msgs.Pose',
        ],
        output='screen',
    )

    # Delay controller spawners to give Gazebo time to finish renderer
    # initialization (EGL/Mesa can take several seconds in containers).
    # The spawner executables will still wait for controller_manager on
    # their own, so this just avoids a race against the render-loop startup.
    # gz_ros2_control auto-loads and activates joint_group_velocity_controller
    # from the parameters yaml as soon as the GazeboSimSystem hardware is ready
    # (≈2 s after startup).  Spawning it again would fail with "cannot configure
    # from 'active' state".  Only joint_state_broadcaster needs an explicit
    # spawner because gz_ros2_control does NOT auto-activate it.
    spawn_jsb = TimerAction(
        period=12.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', 'joint_group_velocity_controller'],
            output='screen',
        )],
    )

    # cmd_vel publisher — mutually exclusive:
    #   use_oracle=false (default): tracker_node (Haar cascade, Phase 1 behaviour)
    #   use_oracle=true:            oracle_node  (privileged ground-truth FK, Step 5)
    # Only one node runs at a time to avoid conflicting /cmd_vel writes.
    # tracker_node publishes Twist() zeros when disabled, which would zero out
    # oracle commands — so we simply don't launch the other node.
    if use_oracle:
        cmd_vel_node = Node(
            package='ocelot',
            executable='oracle_node',
            parameters=[{'enabled': True}],
            output='screen',
        )
    else:
        cmd_vel_node = Node(
            package='ocelot',
            executable='tracker_node',
            parameters=[tracker_params],
            output='screen',
        )

    # Oscillate the face billboard automatically once the world is stable.
    # 15 s gives Gazebo time to finish renderer init and spawn the model.
    # move_face.py runs until the launch is torn down (Ctrl-C kills it).
    move_face = TimerAction(
        period=15.0,
        actions=[ExecuteProcess(
            cmd=['python3', '/ws/src/ocelot/sim/move_face.py'],
            output='screen',
        )],
    )

    # Translates /cmd_vel (Twist) → /joint_group_velocity_controller/commands
    # (Float64MultiArray) so tracker_node drives the simulated joints.
    cmd_vel_adapter = Node(
        package='ocelot',
        executable='cmd_vel_adapter',
        output='screen',
    )

    # Publishes /camera/image_annotated (bounding box, error vector, cmd_vel overlay).
    # View with: ros2 run rqt_image_view rqt_image_view → select /camera/image_annotated
    visualizer = Node(
        package='ocelot',
        executable='visualizer_node',
        output='screen',
    )

    actions = [set_gz_resource, gz_sim, rsp, spawn_robot, bridge, spawn_jsb,
               cmd_vel_node, cmd_vel_adapter, visualizer, move_face]

    # Open rqt_image_view to monitor the annotated camera feed.
    # Runs in both headless and GUI modes — in headless mode this is the only
    # visual; in GUI mode it appears alongside Gazebo.  Skipped only when
    # DISPLAY is unset (fully headless server with no X11 forwarding).
    if os.environ.get('DISPLAY'):
        actions.append(TimerAction(
            period=8.0,
            actions=[Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                arguments=['/camera/image_annotated'],
                output='screen',
            )],
        ))

    return actions


def generate_launch_description():
    # Rendering env vars (LIBGL_ALWAYS_SOFTWARE, GALLIUM_DRIVER, etc.) are
    # set by the docker-compose files, not here.  docker-compose.sim.yml sets
    # software-renderer defaults; docker-compose.sim.gpu.yml overrides them to
    # empty/0 for NVIDIA GPU mode.  Setting them here would re-override the
    # GPU compose values and re-trigger the Mesa EGL / gallium segfault.
    return LaunchDescription([
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run Gazebo in server-only mode (no GUI). Passes -s to gz_args.',
        ),
        DeclareLaunchArgument(
            'use_oracle',
            default_value='false',
            description=(
                'Use oracle_node (privileged FK tracker) instead of tracker_node '
                '(Haar cascade). Only one /cmd_vel publisher runs at a time.'
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
