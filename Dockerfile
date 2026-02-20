FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ROS 2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-cv-bridge \
    ros-jazzy-launch-ros \
    python3-colcon-common-extensions \
    python3-pip \
    python3-opencv \
    i2c-tools \
    && rm -rf /var/lib/apt/lists/*

# Pi-specific Python packages.
# picamera2 depends on host libcamera — we bind-mount from the host OS
# at runtime rather than installing mismatched Ubuntu versions.
RUN pip3 install --break-system-packages \
    adafruit-circuitpython-servokit \
    smbus2 \
    lgpio

# Python 3.11 is needed to run capture_worker.py — picamera2's libcamera
# bindings are compiled for Python 3.11 (Pi OS Bookworm) and cannot load
# under Python 3.12 (ROS Jazzy / Ubuntu Noble).
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --no-cache-dir numpy

# Workspace setup
WORKDIR /ws
RUN mkdir -p /ws/src

# Source ROS 2 in every shell
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc

CMD ["bash"]
