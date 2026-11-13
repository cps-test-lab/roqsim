# The libfranka program test_franka_libfranka.py runs against the simulated FCI. libfranka itself is
# built from its source at the pinned release by the test (tests/franka_build.py), and found here
# through Franka_DIR, so the client is the real library, not a copy of its protocol.
cmake_minimum_required(VERSION 3.16)
project(franka_clients CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Franka REQUIRED)
find_package(Threads REQUIRED)
add_executable(fci_client fci_client.cpp)
target_link_libraries(fci_client Franka::Franka Threads::Threads)
