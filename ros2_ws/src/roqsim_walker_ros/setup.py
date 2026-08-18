from setuptools import find_packages, setup

package_name = "roqsim_walker_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/walker_nav.launch.py"]),
        ("share/" + package_name + "/worlds", ["worlds/walker_nav2.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Frederik Pasch",
    maintainer_email="frederik.pasch@h-ka.de",
    description="ROS 2 goal interface for roqsim walkers: nav2 NavigateThroughPoses.",
    license="Apache-2.0",
    entry_points={
        # Imported once by roqsim_ros_bridge at start-up; the import registers this package's
        # action handler(s) into the bridge's registry. This is the generic seam by which any
        # package teaches the bridge a new action or message type -- no bridge edits, and no
        # nav2_msgs dependency in the core bridge.
        "roqsim_ros_bridge.extensions": [
            "walker_nav = roqsim_walker_ros.actions",
        ],
    },
)
