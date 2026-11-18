// One program, one subcommand per behaviour test_franka_libfranka.py checks, each printing one JSON
// line. It is a libfranka program like any other: what it sees, a program on the robot sees.
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

template <size_t N>
std::string arr(const std::array<double, N>& a) {
  std::ostringstream s;
  s.precision(10);
  s << "[";
  for (size_t i = 0; i < N; i++) s << (i ? "," : "") << a[i];
  s << "]";
  return s.str();
}

std::string quote(const std::string& text) {
  std::string out = "\"";
  for (char c : text) {
    if (c == '"' || c == '\\') out += '\\';
    out += (c == '\n') ? ' ' : c;
  }
  return out + "\"";
}

void defaults(franka::Robot& robot, double force) {
  robot.setCollisionBehavior({{40, 40, 40, 40, 40, 40, 40}}, {{40, 40, 40, 40, 40, 40, 40}},
                             {{force, force, force, force, force, force}},
                             {{force, force, force, force, force, force}});
}

int info(franka::Robot& robot) {
  franka::RobotState s = robot.readOnce();
  franka::Model model = robot.loadModel();
  std::cout << "{\"server_version\":" << robot.serverVersion() << ",\"q\":" << arr(s.q)
            << ",\"robot_mode\":" << static_cast<int>(s.robot_mode)
            << ",\"O_T_EE\":" << arr(s.O_T_EE) << ",\"time_ms\":" << s.time.toMSec()
            << ",\"gravity\":" << arr(model.gravity(s)) << ",\"mass\":" << arr(model.mass(s))
            << ",\"zero_jacobian\":" << arr(model.zeroJacobian(franka::Frame::kFlange, s))
            << ",\"flange\":" << arr(model.pose(franka::Frame::kFlange, s))
            << ",\"upper_velocity\":" << arr(robot.getUpperJointVelocityLimits(s.q)) << "}"
            << std::endl;
  return 0;
}

// Zero torques for `ms` callbacks: the arm holds on the robot's gravity compensation alone.
int torque_hold(franka::Robot& robot, int ms, double force, bool limit_rate, double step) {
  defaults(robot, force);
  int count = 0;
  uint64_t first = 0, last = 0;
  std::vector<double> periods;
  try {
    robot.control(
        [&](const franka::RobotState& s, franka::Duration period) -> franka::Torques {
          if (count == 0) first = s.time.toMSec();
          last = s.time.toMSec();
          if (count > 0) periods.push_back(period.toSec());
          count++;
          std::array<double, 7> tau{};
          if (step != 0.0 && count > 10) tau[0] = step;
          franka::Torques out(tau);
          if (count >= ms) return franka::MotionFinished(out);
          return out;
        },
        limit_rate);
  } catch (const franka::ControlException& e) {
    std::cout << "{\"ok\":false,\"error\":" << quote(e.what()) << ",\"callbacks\":" << count << "}"
              << std::endl;
    return 0;
  }
  double max_period = 0, min_period = 1;
  for (double p : periods) {
    max_period = std::max(max_period, p);
    min_period = std::min(min_period, p);
  }
  franka::RobotState s = robot.readOnce();
  std::cout << "{\"ok\":true,\"callbacks\":" << count << ",\"first_ms\":" << first
            << ",\"last_ms\":" << last << ",\"min_period\":" << min_period
            << ",\"max_period\":" << max_period << ",\"q\":" << arr(s.q)
            << ",\"robot_mode\":" << static_cast<int>(s.robot_mode) << "}" << std::endl;
  return 0;
}

// libfranka's generate_joint_position_motion example, shortened: joints 4, 5 and 7 swing.
int joint_motion(franka::Robot& robot, double seconds, double jump) {
  defaults(robot, 60.0);
  double time = 0.0;
  std::array<double, 7> q0{};
  try {
    robot.control([&](const franka::RobotState& s, franka::Duration period) -> franka::JointPositions {
      time += period.toSec();
      if (time == 0.0) q0 = s.q_d;
      double delta = M_PI / 8.0 * (1 - std::cos(M_PI / 2.5 * time));
      franka::JointPositions out = {{q0[0], q0[1], q0[2], q0[3] + delta, q0[4] + delta, q0[5],
                                     q0[6] + delta}};
      if (jump != 0.0 && time > 0.05) out.q[0] = q0[0] + jump;
      if (time >= seconds) return franka::MotionFinished(out);
      return out;
    });
  } catch (const franka::ControlException& e) {
    std::cout << "{\"ok\":false,\"error\":" << quote(e.what()) << "}" << std::endl;
    return 0;
  }
  franka::RobotState s = robot.readOnce();
  std::cout << "{\"ok\":true,\"q\":" << arr(s.q) << ",\"q0\":" << arr(q0) << "}" << std::endl;
  return 0;
}

int recover(franka::Robot& robot) {
  robot.automaticErrorRecovery();
  franka::RobotState s = robot.readOnce();
  std::cout << "{\"robot_mode\":" << static_cast<int>(s.robot_mode)
            << ",\"cartesian_reflex\":" << s.current_errors.cartesian_reflex << "}" << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: fci_client <command> <address> [args]" << std::endl;
    return 2;
  }
  std::string cmd = argv[1];
  auto arg = [&](int i, double d) { return argc > i ? std::stod(argv[i]) : d; };
  try {
    if (cmd == "dual") {
      franka::Robot left(argv[2], franka::RealtimeConfig::kIgnore);
      franka::Robot right(argv[3], franka::RealtimeConfig::kIgnore);
      std::array<std::vector<uint64_t>, 2> times;
      auto run = [&](franka::Robot& r, std::vector<uint64_t>& t) {
        defaults(r, 60.0);
        int n = 0;
        r.control([&](const franka::RobotState& s, franka::Duration) -> franka::Torques {
          t.push_back(s.time.toMSec());
          franka::Torques out{{0, 0, 0, 0, 0, 0, 0}};
          return ++n >= 500 ? franka::MotionFinished(out) : out;
        });
      };
      std::thread a([&] { run(left, times[0]); });
      std::thread b([&] { run(right, times[1]); });
      a.join();
      b.join();
      std::cout << "{\"left\":[" << times[0].front() << "," << times[0].back() << "," << times[0].size()
                << "],\"right\":[" << times[1].front() << "," << times[1].back() << ","
                << times[1].size() << "]}" << std::endl;
      return 0;
    }
    franka::Robot robot(argv[2], franka::RealtimeConfig::kIgnore);
    if (cmd == "info") return info(robot);
    if (cmd == "torque_hold")
      return torque_hold(robot, static_cast<int>(arg(3, 1000)), arg(4, 60.0), arg(5, 0) != 0, arg(6, 0));
    if (cmd == "joint_motion") return joint_motion(robot, arg(3, 2.0), arg(4, 0.0));
    if (cmd == "recover") return recover(robot);
  } catch (const franka::Exception& e) {
    std::cout << "{\"exception\":" << quote(e.what()) << "}" << std::endl;
    return 1;
  }
  std::cerr << "unknown command " << cmd << std::endl;
  return 2;
}
