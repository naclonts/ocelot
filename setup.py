from setuptools import setup

package_name = 'ocelot'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/tracker_launch.py',
            'launch/sim_launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/tracker_params.yaml',
            'config/controllers.yaml',
        ]),
        ('share/' + package_name + '/urdf', ['urdf/pan_tilt.urdf']),
        ('share/' + package_name + '/sim/worlds', [
            'sim/worlds/tracker_world.sdf',
            'sim/worlds/scenario_world.sdf',
        ]),
        ('share/' + package_name + '/sim/models/face_billboard', [
            'sim/models/face_billboard/model.config',
            'sim/models/face_billboard/model.sdf',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nathan',
    maintainer_email='todo@example.com',
    description='Ocelot pan-tilt face tracking robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = ocelot.camera_node:main',
            'servo_node = ocelot.servo_node:main',
            'tracker_node = ocelot.tracker_node:main',
            'oracle_node = ocelot.oracle_node:main',
            'cmd_vel_adapter = ocelot.cmd_vel_adapter:main',
            'visualizer_node = ocelot.visualizer_node:main',
            'oracle_validator = ocelot.oracle_validator:main',
            'vla_node = ocelot.vla_node:main',
        ],
    },
)
