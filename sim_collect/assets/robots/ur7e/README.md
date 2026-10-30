# ur7e.xml — MuJoCo UR7e for `sim_collect`

`ur_description` 2.7.0 ships `config/ur7e/*.yaml` as **byte-identical copies of the UR5e
files** (DH d=[0.1625,0,0,0.1333,0.0997,0.0996], a=[0,-0.425,-0.3922,0,0,0], same limits,
`visual_parameters.yaml` points at `meshes/ur5e/`). So this model is the menagerie
`universal_robots_ur5e/ur5e.xml` structure with:

- `model="ur7e"`, class `ur7e`;
- the five rounded link offsets replaced by the exact URDF values
  (shoulder z 0.163→0.1625, wrist_1 z 0.392→0.3922, wrist_2 y 0.127→0.1263,
  wrist_3 z 0.1→0.0997, attachment_site y 0.1→0.0996) and elbow range ±3.14159;
- meshes referenced from the menagerie UR5e assets via `meshdir` (nothing vendored).

Verified 2026-09-14: `attachment_site` world pose == `ur_kin.fk(q)` to 0.000 mm over 2000
random q (menagerie original: 0.98 mm mean / 1.49 mm max). Visual geometry is the UR5e
enclosure; the real UR7e differs cosmetically only. Factory calibration of the real arm
(~0.1–1 mm) is not modelled.
