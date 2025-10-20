from setuptools import setup, find_packages

setup(
    name="streamdiffusion-spout-server",
    version="0.1.0",
    description="StreamDiffusion Spout server for real-time AI image generation via OSC",
    author="Alex Olwal",
    author_email="alex@tactam.com",
    url="https://github.com/olwal/streamdiffusion-spout-server",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pillow",
        "PyOpenGL",
        "python-osc",
        "torch",
        "torchvision",
        "diffusers",
        "accelerate",
        # Note: SpoutGL needs to be installed separately on Windows
    ],
    python_requires=">=3.9",
    entry_points={
        'console_scripts': [
            'streamdiffusion-spout-server=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
