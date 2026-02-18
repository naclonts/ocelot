from setuptools import setup, find_packages

setup(
    name="ocelot",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pantilthat",
        "picamera2",
        "opencv-python",
    ],
)
