from setuptools import setup, find_packages

setup(
    name="ocelot",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "adafruit-circuitpython-servokit",
        "smbus2",
        "picamera2",
        "opencv-python",
    ],
)
