# Mesh & URDF licenses, attribution, and provenance

robot-renderer's own code is MIT-licensed (see the repo-root `LICENSE`).
Every robot in `src/robot_renderer/robots/`, however, uses third-party CAD
geometry. This folder holds the applicable upstream license texts; this file
records where each asset came from, what we changed, and why two robots'
source-derived assets are **downloaded locally** (`make assets`) instead of
being shipped with the repo. Consult each upstream repository for its
authoritative copyright notice.

| Robot | Upstream source | Copyright holder | License | Source-derived files shipped? |
|---|---|---|---|---|
| `panda` | [frankaemika/franka_ros](https://github.com/frankaemika/franka_ros) `franka_description` [1] | Franka Emika GmbH | **Apache-2.0** (`Apache-2.0.txt`) | yes |
| `baxter` | [RethinkRobotics/baxter_common](https://github.com/RethinkRobotics/baxter_common) [1] | Rethink Robotics, Inc. | **BSD-3-Clause** (`BSD-3-Clause.txt`) | yes |
| `lbr_med7` | [lbr-stack/lbr_fri_ros2_stack](https://github.com/lbr-stack/lbr_fri_ros2_stack) `lbr_description` | lbr-stack contributors | **Apache-2.0** (`Apache-2.0.txt`) | yes |
| `xarm7` | URDF parameters: [xArm-Developer/xarm_ros2](https://github.com/xArm-Developer/xarm_ros2); meshes: [ootts/EasyHeC](https://github.com/ootts/EasyHeC) textured GLBs (geometry matches the official `xarm_ros2` STLs) | UFACTORY (URDF) / SU Lab, UCSD (meshes) | **BSD-3-Clause** (URDF) / **MIT** (meshes; `MIT.txt`) | yes |
| `meca500` | [Vanderbilt-Applied-Robotics-Lab/meca500_ros2](https://github.com/Vanderbilt-Applied-Robotics-Lab/meca500_ros2) (meshes trace to Mecademic CAD) | unknown | **none declared** (`package.xml`: "TODO: License declaration") | no; `make assets` |
| `owi535` | [RoboPose](https://github.com/ylabbe/robopose) dependency archive hosted by INRIA (`archive_ylabbeprojectsdata/robopose/deps/owi-description/`); the model originates from the [CRAVES](https://github.com/zuoym15/craves.ai) project | unknown | **none declared** on the archive | no; `make assets` |

[1] The `panda` and `baxter` OBJ files reached this repository via the
MIT-licensed [CtRNet repo](https://github.com/ucsdarclab/CtRNet-robot-pose-estimation)
(UCSD ARC Lab), which performed the original DAE-to-OBJ conversions; the
upstream licenses above govern the geometry itself.

## Assets fetched by `make assets`

The repository and its package builds exclude the upstream files whose sources
declare no license. `tools/fetch_assets.py` downloads them from their public
locations and re-applies the required modifications locally, pinned by
`tools/asset_manifest.json`. The tool is safe to re-run: valid existing assets
are skipped (`--force` regenerates them):

- **owi535**: fetch the URDF; bake each URDF `<visual>` origin into the
  vertices and convert mm to m; add
  `mtllib`/`usemtl` headers; write our custom non-textured material colors
  (our own values, stored in the manifest); decimate the six
  over-tessellated meshes with meshoptimizer
  (`tools/decimate_owi535_meshes.py` documents why).
- **meca500**: fetch the robot description and seven `.dae` visual meshes
  from a pinned upstream commit; remove collision elements and rewrite the
  visual paths into a plain local URDF; convert the meshes to OBJ with
  duplicate vertices merged. The OBJ files reference our in-repo
  `material.lib` files.

Downloads are pinned by SHA-256, so the pipeline always starts from the same
upstream files. Generated meshes are not hash-checked (mesh libraries are not
bit-deterministic across platforms); they are checked structurally instead.

The renderer treats robot meshes as double-sided, since converted CAD is not
guaranteed to be closed and consistently wound.

## Statement of asset changes

What we changed relative to each upstream. (Apache-2.0 section 4(b) requires stating
changes; BSD-3 requires retaining attribution. This section is that statement.)

How to read the verification claims below:

- **byte-identical**: the file is exactly the same as the upstream one, bit
  for bit.
- **bbox delta**: we compare the bounding box (the smallest box that contains
  the mesh) of our converted mesh against the upstream original. This catches
  the errors that would actually matter (wrong units, flipped axes, offsets,
  missing geometry); deltas well under a millimetre are just float rounding
  from writing text files, far below anything the renderer can see.
- **extent delta**: same idea, but comparing the box's width/height/depth
  instead of its corner coordinates.

**panda**
- `panda.urdf`: byte-identical to CtRNet `urdfs/Panda/panda.urdf`. No changes.
- Meshes: byte-identical to CtRNet's (48/48 files verified, incl.
  `material.lib`s and collision OBJs). Conversion chain: Franka CAD
  (franka_ros, Apache-2.0), converted to OBJ in CtRNet with Aspose.3D, then
  copied here unmodified. The flat `meshes/visual/<link>.obj` files are duplicates of the
  per-directory copies (layout convenience only).

**baxter**
- `baxter.urdf`: from CtRNet
  `urdfs/Baxter/baxter_description/urdf/baxter.urdf` with exactly one change:
  the fixed joint `left_gripper_base` origin `xyz="0 0 0"` to `"0 0 0.025"`
  (+2.5 cm along Z, closing the visual gap between `left_hand` and the
  gripper base).
- Meshes (all ultimately baxter_common, BSD-3-Clause; every link verified
  geometrically against the upstream DAEs):
  - `S0.obj` + `material.lib`s: CtRNet's Aspose conversions, unmodified.
  - `E0, E1, S1, W0, W2`: our own conversions (assimp v6.0.0) from
    `baxter_description/meshes/*/**.DAE`, with bbox deltas at most 0.12 mm.
  - `W1`: same conversion, shape identical, plus a -3.2 mm Y positional
    shift we baked (link-connection tweak).
  - `gripper_base`: added by us, an assimp conversion of baxter_common
    `rethink_ee_description/meshes/electric_gripper/electric_gripper_w_fingers.DAE`
    (the full gripper *with fingers*) with a +11.3 mm X positional bake.
    Not present in CtRNet.
  - Material colors converted from Rethink's own DAE materials (Baxter red,
    greys), so they are faithful, not hand-picked.

**lbr_med7**
- `lbr_med7.urdf`: written by us as a xacro-expansion of lbr-stack
  `lbr_description` (med7), reduced to visual elements, ROS `package://`
  paths replaced with local mesh paths, `world` anchor link added. All 7
  actuated joints verified identical (xyz, rpy, axis) to
  `med7_description.xacro`.
- Meshes: our OBJ conversions of
  `lbr_description/meshes/med7/visual/link_*.dae` (Apache-2.0). All 8 links
  geometry-verified: extent deltas exactly 0; links 1-7 are re-expressed in
  link-local frames (the upstream DAEs are exported in base-frame-at-zero
  coordinates; the linear offsets are that convention change, not
  modifications). `material.lib` colors are ours (white body + grey rings,
  matching the real Med 7 two-tone).

**xarm7**
- `xarm7.urdf`: written by us, with joint parameters per xArm-Developer
  `xarm_ros2` `xarm_description` (BSD-3-Clause), visual-only, local paths.
  All 7 actuated joints verified: xyz + axis identical to upstream; rpy
  identical up to a 4 microradian rounding of pi/2 (`1.5708` vs `1.57079633`;
  that is about 0.0002 degrees, with no visible effect).
- Meshes: our conversions of EasyHeC's textured GLBs
  ([ootts/EasyHeC](https://github.com/ootts/EasyHeC), MIT; downloads
  SHA-pinned). Geometry verified per link against the official `xarm_ros2`
  STLs (0.000 mm bbox deviation); the GLBs' `white`/`silver` primitives map
  onto our `white`/`dark` materials (`material.lib` definitions are ours).
  `tools/convert_xarm_easyhec.py` is the reproducible conversion path
  (normally not needed, since the converted meshes ship with the repo).

**meca500**
- `meca500.urdf`: not shipped. `make assets-meca500` generates it from the
  pinned upstream description, keeping its links and joints, removing
  collision elements, and rewriting the visual mesh paths.
- Meshes: not shipped. The same command downloads the pinned DAE inputs and
  converts them locally. The tracked `material.lib` definitions are ours.

