from setuptools import find_packages, setup

package_name = "roqsim_nav2_example"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/nav2_turtlebot.launch.py",
                "launch/nav2_turtlebot_depot.launch.py",
                "launch/nav2_g1.launch.py",
                "launch/nav2_spot.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/params",
            [
                "params/nav2_params.yaml",
                "params/nav2_params_depot.yaml",
                "params/nav2_params_g1.yaml",
                "params/nav2_params_spot.yaml",
            ],
        ),
        (
            "share/" + package_name + "/maps",
            ["maps/empty_room.yaml", "maps/empty_room.pgm", "maps/depot.yaml", "maps/depot.pgm"],
        ),
        (
            "share/" + package_name + "/worlds",
            [
                "worlds/turtlebot_nav2.yaml",
                "worlds/depot_nav2.yaml",
                "worlds/g1_nav2.yaml",
                "worlds/spot_nav2.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Frederik Pasch",
    maintainer_email="frederik.pasch@h-ka.de",
    description="Minimal nav2 example on roqsim: TurtleBot 4, Unitree G1 humanoid, or Spot quadruped.",
    license="Apache-2.0",
)
