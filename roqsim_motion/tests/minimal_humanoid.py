"""A minimal humanoid robot written for the tests: a free base, a 3-DoF waist, 7-DoF arms with a
grip site each, 6-DoF legs, and position servos on the arms (none on the legs), so the robot output
has servos to use and to add. No vendor's look and no meshes."""

from __future__ import annotations

ARM = """
      <body name="{s}_shoulder_pitch_link" pos="0 {y} 0.30">
        <joint name="{s}_shoulder_pitch_joint" axis="0 1 0" range="-3 2.5"/>
        <geom type="sphere" size="0.04" mass="0.5"/>
        <body name="{s}_shoulder_roll_link"><joint name="{s}_shoulder_roll_joint" axis="1 0 0" range="{rlo} {rhi}"/>
          <body name="{s}_shoulder_yaw_link"><joint name="{s}_shoulder_yaw_joint" axis="0 0 1" range="-2.6 2.6"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.24" size="0.035" mass="1"/>
            <body name="{s}_elbow_link" pos="0 0 -0.24"><joint name="{s}_elbow_joint" axis="0 1 0" range="-2.5 0.3"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.03" mass="0.6"/>
              <body name="{s}_wrist_roll_link" pos="0 0 -0.22"><joint name="{s}_wrist_roll_joint" axis="0 0 1" range="-2 2"/>
                <body name="{s}_wrist_pitch_link"><joint name="{s}_wrist_pitch_joint" axis="0 1 0" range="-1.6 1.6"/>
                  <body name="{s}_wrist_yaw_link"><joint name="{s}_wrist_yaw_joint" axis="1 0 0" range="-1.6 1.6"/>
                    <geom name="{s}_palm" type="box" pos="0 0 -0.05" size="0.04 0.015 0.05" mass="0.3"/>
                    <site name="{s}_grasp" pos="0 {gy} -0.09" quat="{gq}"/>
                  </body></body></body></body></body></body></body>"""
LEG = """
    <body name="{s}_hip_pitch_link" pos="0 {y} -0.05">
      <joint name="{s}_hip_pitch_joint" axis="0 1 0" range="-2.5 2.5"/>
      <body name="{s}_hip_roll_link"><joint name="{s}_hip_roll_joint" axis="1 0 0" range="-0.5 0.5"/>
        <body name="{s}_hip_yaw_link"><joint name="{s}_hip_yaw_joint" axis="0 0 1" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.36" size="0.05" mass="3"/>
          <body name="{s}_knee_link" pos="0 0 -0.36"><joint name="{s}_knee_joint" axis="0 1 0" range="-0.1 2.6"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.36" size="0.04" mass="2"/>
            <body name="{s}_ankle_pitch_link" pos="0 0 -0.36"><joint name="{s}_ankle_pitch_joint" axis="0 1 0" range="-0.9 0.5"/>
              <body name="{s}_ankle_roll_link"><joint name="{s}_ankle_roll_joint" axis="1 0 0" range="-0.3 0.3"/>
                <geom type="box" pos="0.04 0 -0.03" size="0.1 0.04 0.02" mass="0.5"/>
              </body></body></body></body></body></body>"""


def mjcf() -> str:
    # The grip frames follow roqsim_motion.hands: +x out of the palm (toward the body), +z along the
    # fingers (the forearm's -z): for the left hand x=-y_w, z=-z_w; for the right x=+y_w, z=-z_w.
    arms = ARM.format(s="left", y=0.18, rlo=-0.5, rhi=2.5, gy=-0.02, gq="0 0.7071068 0.7071068 0") + ARM.format(
        s="right", y=-0.18, rlo=-2.5, rhi=0.5, gy=0.02, gq="0 0.7071068 -0.7071068 0"
    )
    legs = LEG.format(s="left", y=0.1) + LEG.format(s="right", y=-0.1)
    servos = "".join(
        f'<position name="{s}_{j}" joint="{s}_{j}_joint" kp="80" kv="4"/>'
        for s in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    )
    return f"""<mujoco model="minimal_humanoid">
  <compiler angle="radian" boundmass="0.05" boundinertia="0.0001"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="base_free"/>
      <geom type="box" size="0.08 0.12 0.06" mass="5"/>
      <body name="waist_yaw_link" pos="0 0 0.1"><joint name="waist_yaw_joint" axis="0 0 1" range="-2.6 2.6"/>
        <body name="waist_roll_link"><joint name="waist_roll_joint" axis="1 0 0" range="-0.5 0.5"/>
          <body name="torso_link"><joint name="waist_pitch_joint" axis="0 1 0" range="-0.5 1.5"/>
            <geom type="box" pos="0 0 0.18" size="0.08 0.14 0.15" mass="8"/>
{arms}
          </body></body></body>
{legs}
    </body>
  </worldbody>
  <actuator>{servos}</actuator>
</mujoco>"""


TABLE = {
    "humanoid": "minimal_humanoid",
    "base": {"body": "pelvis", "height": 0.8},
    "joints": {
        "spine": [{"z": "waist_yaw_joint"}, {"x": "waist_roll_joint"}, {"y": "waist_pitch_joint"}],
        **{
            f"shoulder_{c}": [{"y": f"{s}_shoulder_pitch_joint"}, {"x": f"{s}_shoulder_roll_joint"},
                              {"z": f"{s}_shoulder_yaw_joint"}]
            for c, s in (("l", "left"), ("r", "right"))
        },
        **{f"elbow_{c}": [{"y": f"{s}_elbow_joint"}] for c, s in (("l", "left"), ("r", "right"))},
        **{
            f"wrist_{c}": [{"z": f"{s}_wrist_roll_joint"}, {"y": f"{s}_wrist_pitch_joint"},
                           {"x": f"{s}_wrist_yaw_joint"}]
            for c, s in (("l", "left"), ("r", "right"))
        },
        **{
            f"hip_{c}": [{"y": f"{s}_hip_pitch_joint"}, {"x": f"{s}_hip_roll_joint"}, {"z": f"{s}_hip_yaw_joint"}]
            for c, s in (("l", "left"), ("r", "right"))
        },
        **{f"knee_{c}": [{"y": f"{s}_knee_joint"}] for c, s in (("l", "left"), ("r", "right"))},
        **{f"ankle_{c}": [{"y": f"{s}_ankle_pitch_joint"}, {"x": f"{s}_ankle_roll_joint"}]
           for c, s in (("l", "left"), ("r", "right"))},
        "head": "none",
    },
    "hands": {"l": {"frame": "left_grasp"}, "r": {"frame": "right_grasp"}},
    "gaze": "spine",
    "physical": {"servos": "own", "kp": 400.0, "kv": 20.0},
}
