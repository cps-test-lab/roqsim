from setuptools import find_packages, setup

package_name = "roqsim_nav_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Frederik Pasch",
    maintainer_email="frederik.pasch@h-ka.de",
    description="ROS 2 goal interface for roqsim navigators: nav2 NavigateToPose / ThroughPoses.",
    license="Apache-2.0",
    entry_points={
        # Imported once by roqsim_ros_bridge at start-up; the import registers this package's action
        # handlers into the bridge's registry. The generic seam by which any package teaches the
        # bridge a new action type -- no bridge edits, and no nav2_msgs dependency in the core bridge.
        #
        # ONE package registers these two types. The registry overwrites silently and extension load
        # order is unspecified, so a second package registering either one would decide by install
        # order which handler serves a goal.
        "roqsim_ros_bridge.extensions": [
            "roqsim_nav = roqsim_nav_ros.actions",
        ],
    },
)
