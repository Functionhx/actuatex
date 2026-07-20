from glob import glob
from setuptools import find_packages, setup


PACKAGE_NAME = "actuatex_navigation"


setup(
    name=PACKAGE_NAME,
    version="0.2.0",
    packages=find_packages(exclude=("test",)),
    package_data={PACKAGE_NAME: ["data/*.xz", "data/*.md"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE_NAME}/maps", glob("maps/*")),
    ],
    install_requires=["numpy", "PyYAML", "setuptools"],
    zip_safe=True,
    maintainer="Yuchen Fan",
    maintainer_email="functionhx@gmail.com",
    description="Nav2 and point-timed Livox Mid-360 simulation for ActuateX.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "cmd_vel_adapter = actuatex_navigation.cmd_vel_adapter:main",
        ],
    },
)
