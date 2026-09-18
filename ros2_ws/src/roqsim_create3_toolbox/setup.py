from glob import glob

from setuptools import find_packages, setup

package_name = "roqsim_create3_toolbox"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/worlds", glob("worlds/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Frederik Pasch",
    maintainer_email="frederik.pasch@h-ka.de",
    description="The Create 3 / TurtleBot 4 stack over roqsim: the simulator adapter and its launch.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bumper_hazard = roqsim_create3_toolbox.bumper_hazard:main",
        ],
    },
)
