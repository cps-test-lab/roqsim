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
    description="A walker driven over ROS 2: the demo launch and world. Handlers live in roqsim_nav_ros.",
    license="Apache-2.0",
    # No `roqsim_ros_bridge.extensions` entry point any more. The NavigateThroughPoses handler this
    # package used to register moved to `roqsim_nav_ros`, which serves it for EVERY mover -- a
    # walker, an opponent robot, a driven prop -- from one place. Two packages registering one action
    # type would let install order decide which handler runs, silently: the bridge's registry
    # overwrites without complaint and extension load order is unspecified.
    #
    # Nothing a user sees changed: the endpoint, the action type and the action name are the same.
)
