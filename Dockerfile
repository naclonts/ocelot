FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ROS 2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-cv-bridge \
    ros-jazzy-launch-ros \
    ros-jazzy-rqt-image-view \
    ros-jazzy-web-video-server \
    python3-colcon-common-extensions \
    python3-pip \
    python3-opencv \
    i2c-tools \
    && rm -rf /var/lib/apt/lists/*

# Pin setuptools to a version compatible with colcon's --symlink-install
# (newer setuptools dropped setup.py develop support)
RUN pip3 install --break-system-packages 'setuptools<74'

# Pi-specific Python packages.
# picamera2 depends on host libcamera — we bind-mount from the host OS
# at runtime rather than installing mismatched Ubuntu versions.
RUN pip3 install --break-system-packages \
    adafruit-circuitpython-servokit \
    smbus2 \
    lgpio

# Python 3.11 for capture_worker.py.
# picamera2/libcamera bindings on the host (Pi OS Bookworm) are compiled for
# Python 3.11; the host .venv/lib/python3.11/site-packages is bind-mounted
# at runtime to provide numpy with the correct .cpython-311 .so files.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Workspace setup
WORKDIR /ws
RUN mkdir -p /ws/src

# Source ROS 2 and the workspace overlay in every interactive shell
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc && \
    echo '[[ -f /ws/install/setup.bash ]] && source /ws/install/setup.bash' >> /etc/bash.bashrc

CMD ["bash"]
