from setuptools import find_packages, setup

package_name = "roqsim_ros_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/turtlebot.launch.py"]),
        ("share/" + package_name + "/worlds", ["worlds/turtlebot_ros2.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Frederik Pasch",
    maintainer_email="frederik.pasch@h-ka.de",
    description="ROS 2 bridge for roqsim, provided as roqsim plugins.",
    license="Apache-2.0",
    entry_points={
        # Discoverable by roqsim's plugin registry via short names.
        "roqsim.plugins": [
            "ros2_bridge = roqsim_ros_bridge.ros2_bridge:Ros2Bridge",
            "sim_interfaces = roqsim_ros_bridge.sim_interfaces:SimInterfacesPlugin",
        ],
        # Convenience launcher that runs a roqsim world with the ROS bridge.
        "console_scripts": [
            "roqsim_bridge = roqsim_ros_bridge.run_bridge:main",
        ],
    },
)
