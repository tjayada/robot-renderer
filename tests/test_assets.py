"""Dependency-light tests for asset validation and safe replacement."""
import io
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import fetch_assets as fa  # noqa: E402


def _write_triangle(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n"
    )


def _meca_source(parts: tuple[str, ...]) -> str:
    links = []
    for part in parts:
        links.append(
            f"""
  <link name="{part}">
    <visual>
      <geometry>
        <mesh filename="package://meca500_description/meshes/visual/{part}.dae" />
      </geometry>
    </visual>
    <collision>
      <geometry><mesh filename="collision/{part}.stl" /></geometry>
    </collision>
  </link>"""
        )
    return (
        '<robot name="meca500_arm" xmlns:xacro="http://ros.org/wiki/xacro">'
        + "".join(links)
        + """
  <joint name="joint1" type="revolute">
    <limit lower="-1.22173048" upper="1.22173048" effort="10.0" velocity="2.0" />
  </joint>
</robot>
"""
    )


def test_validate_mesh_arrays_rejects_invalid_data():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    fa._validate_mesh_arrays("valid", verts, faces, max_faces=1)

    with pytest.raises(ValueError, match="nonempty"):
        fa._validate_mesh_arrays("empty", np.empty((0, 3)), faces)
    with pytest.raises(ValueError, match="non-finite"):
        fa._validate_mesh_arrays("nan", np.full((3, 3), np.nan), faces)
    with pytest.raises(ValueError, match="outside"):
        fa._validate_mesh_arrays("index", verts, np.array([[0, 1, 3]]))
    with pytest.raises(ValueError, match="above target"):
        fa._validate_mesh_arrays("dense", verts, np.array([[0, 1, 2], [0, 2, 1]]), max_faces=1)


def test_owi_outputs_are_validated_before_skip(tmp_path):
    manifest = {
        "urdf": {"name": "owi535.urdf"},
        "files": {"part.obj": {"decimate_target": 1}},
    }
    dest = tmp_path / "owi535"
    dest.mkdir()
    (dest / "owi535.urdf").write_text("<robot name=\"owi535\" />\n")
    _write_triangle(dest / "part.obj")
    (dest / "part.obj.mtl").write_text(
        "newmtl material_0\nKd 0.5 0.5 0.5\n"
    )

    assert fa.owi535_outputs_present(manifest, tmp_path)

    (dest / "part.obj").write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 4\n"
    )
    assert not fa.owi535_outputs_present(manifest, tmp_path)


def test_owi_joint_limits_injected():
    urdf = (
        '<robot name="owi535">\n'
        '  <joint name="j1" type="revolute">\n'
        '    <limit lower="-1.0" upper="1.0"/>\n'
        '  </joint>\n'
        '  <joint name="j2" type="revolute">\n'
        '    <limit lower="-2.0" upper="2.0" effort="50" velocity="1.0"/>\n'
        '  </joint>\n'
        '</robot>\n'
    )
    limits = ET.fromstring(fa._inject_owi_joint_limits(urdf)).findall(".//limit")
    # A limit missing both attributes gains the nominal values, bounds untouched.
    assert limits[0].get("effort") == fa._OWI_JOINT_EFFORT
    assert limits[0].get("velocity") == fa._OWI_JOINT_VELOCITY
    assert limits[0].get("lower") == "-1.0" and limits[0].get("upper") == "1.0"
    # A limit that already has values is left as-is.
    assert limits[1].get("effort") == "50" and limits[1].get("velocity") == "1.0"


def test_owi_validation_rejects_missing_limit_attrs(tmp_path):
    manifest = {"urdf": {"name": "owi535.urdf"}, "files": {}}
    dest = tmp_path / "owi535"
    dest.mkdir()
    (dest / "owi535.urdf").write_text(
        '<robot name="owi535">'
        '<joint name="j1" type="revolute"><limit lower="-1" upper="1"/></joint>'
        '</robot>\n'
    )
    with pytest.raises(ValueError, match="effort/velocity"):
        fa._validate_owi535_outputs(manifest, dest)


def test_owi_injected_urdf_loads_in_roboticstoolbox(tmp_path):
    pytest.importorskip("roboticstoolbox")
    from roboticstoolbox.robot.ERobot import ERobot

    urdf = (
        '<?xml version="1.0"?>\n'
        '<robot name="owi535">\n'
        '  <link name="base"/>\n'
        '  <link name="arm"/>\n'
        '  <joint name="j1" type="revolute">\n'
        '    <parent link="base"/>\n'
        '    <child link="arm"/>\n'
        '    <axis xyz="0 0 1"/>\n'
        '    <limit lower="-1.0" upper="1.0"/>\n'
        '  </joint>\n'
        '</robot>\n'
    )
    path = tmp_path / "owi535.urdf"
    # Upstream form (no effort/velocity) is what the rtb parser rejects.
    path.write_text(urdf)
    with pytest.raises(ValueError):
        ERobot.URDF_read(str(path))
    # After injection it loads.
    path.write_text(fa._inject_owi_joint_limits(urdf))
    links, name, _, _ = ERobot.URDF_read(str(path))
    assert name == "owi535"
    assert len(links) == 2


def test_meca_outputs_are_validated_before_skip(tmp_path):
    manifest = {
        "urdf": {"name": "meca500.urdf"},
        "files": {"base": {"sha256": "unused"}},
    }
    robot_dir = tmp_path / "meca500"
    (robot_dir / "meca500.urdf").parent.mkdir(parents=True)
    (robot_dir / "meca500.urdf").write_text(
        fa._transform_meca500_urdf(_meca_source(("base",)), manifest)
    )
    output = tmp_path / "meca500" / "meshes" / "visual" / "base" / "base.obj"
    _write_triangle(output)
    assert fa.meca500_outputs_present(manifest, tmp_path)

    output.write_text("")
    assert not fa.meca500_outputs_present(manifest, tmp_path)
    _write_triangle(output)

    urdf_path = robot_dir / "meca500.urdf"
    urdf_path.write_text(
        urdf_path.read_text().replace(fa.MECA500_GENERATED_MARKER, "")
    )
    assert not fa.meca500_outputs_present(manifest, tmp_path)

    urdf_path.write_text(
        fa._transform_meca500_urdf(_meca_source(("base",)), manifest)
    )
    (robot_dir / "meca500.urdf").unlink()
    assert not fa.meca500_outputs_present(manifest, tmp_path)


def test_meca_urdf_is_reduced_from_pinned_source():
    manifest = {
        "urdf": {"name": "meca500.urdf"},
        "files": {
            "base": {"sha256": "unused"},
            "joint": {"sha256": "unused"},
        },
    }
    output = fa._transform_meca500_urdf(
        _meca_source(("base", "joint")),
        manifest,
    )
    root = fa.ET.fromstring(output)

    assert fa.MECA500_GENERATED_MARKER in output
    assert root.get("name") == "meca500"
    assert root.findall(".//collision") == []
    assert {
        mesh.get("filename") for mesh in root.findall(".//visual/geometry/mesh")
    } == {
        "meshes/visual/base/base.obj",
        "meshes/visual/joint/joint.obj",
    }
    assert root.find("joint/limit").get("lower") == "-1.22173048"


def test_meca_urdf_rejects_unexpected_visual_mesh():
    manifest = {
        "urdf": {"name": "meca500.urdf"},
        "files": {"base": {"sha256": "unused"}},
    }
    with pytest.raises(ValueError, match="unknown visual mesh"):
        fa._transform_meca500_urdf(_meca_source(("other",)), manifest)


def test_atomic_text_replacement(tmp_path):
    output = tmp_path / "asset.txt"
    output.write_text("old")
    fa._atomic_write_text(output, "new")
    assert output.read_text() == "new"
    assert list(tmp_path.iterdir()) == [output]


def test_download_replaces_from_temporary_file(tmp_path, monkeypatch):
    payload = b"asset bytes"
    monkeypatch.setattr(
        fa.urllib.request,
        "urlopen",
        lambda _url, timeout: io.BytesIO(payload),
    )
    output = fa._download(
        "https://example.invalid/asset",
        tmp_path / "asset.bin",
        expected_sha=None,
        skip_hash=False,
    )
    assert output.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [output]
