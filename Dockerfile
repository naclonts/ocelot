FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ROS 2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-cv-bridge \
    python3-pip \
    python3-opencv \
    i2c-tools \
    && rm -rf /var/lib/apt/lists/*

# Pi-specific Python packages.
# picamera2 depends on host libcamera — we bind-mount from the host OS
# at runtime rather than installing mismatched Ubuntu versions.
RUN pip3 install --break-system-packages \
    adafruit-circuitpython-servokit \
    smbus2

# Workspace setup
WORKDIR /ws
RUN mkdir -p /ws/src

# Source ROS 2 in every shell
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc

CMD ["bash"]
