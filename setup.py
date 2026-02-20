from setuptools import setup

package_name = 'ocelot'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/tracker_launch.py']),
        ('share/' + package_name + '/config', ['config/tracker_params.yaml']),
        ('share/' + package_name + '/urdf', ['urdf/pan_tilt.urdf']),
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
        ],
    },
)
