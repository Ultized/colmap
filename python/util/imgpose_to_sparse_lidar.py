"""Convert SHARE SLAM S20 ``ImgPose.txt`` + per-folder intrinsics into a COLMAP
sparse model directory that can be fed to ``import_lidar_priors_to_db.py``.

ImgPose.txt row format::

    <relative_image_path> x y z roll pitch yaw qx qy qz qw timestamp

Important conventions carried over from re_dev's pipeline:

* Quaternion order in the file is **(qx, qy, qz, qw)**; COLMAP ``images.txt`` wants
  **(qw, qx, qy, qz)**.
* ``(x, y, z)`` + ``(qx, qy, qz, qw)`` is the **world-from-camera** pose (camera center
  in world, camera orientation in world). COLMAP stores **camera-from-world**, so
  both the rotation and translation must be inverted.
* Each rig sensor (left / right / ...) becomes its own COLMAP ``CAMERA_ID``. The
  caller provides intrinsics text files named like ``<prefix>_undistort_intrinsic.txt``
  whose content is the 3x3 K matrix::

        fx  0  cx
         0 fy  cy
         0  0   1

  Image dimensions are inferred as ``width=2*cx, height=2*cy`` since these are
  undistorted perspective images centered on the principal point.

Output (COLMAP text sparse model):

    cameras.txt    # PINHOLE rows per sensor
    images.txt     # two lines per image (pose line + empty points2D line)
    points3D.txt   # empty

Typical invocation::

    python imgpose_to_sparse_lidar.py \
        --args.imgpose "undistort/ImgPose.txt" \
        --args.intrinsic-dir "undistort" \
        --args.output-dir "sparse_lidar"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass(frozen=True)
class Args:
    imgpose: Path
    intrinsic_dir: Path
    output_dir: Path
    # Detect unique folder prefixes automatically (e.g. "left", "right") from
    # ImgPose paths and load "<prefix>_undistort_intrinsic.txt" for each.
    intrinsic_suffix: str = "_undistort_intrinsic.txt"


def _parse_intrinsic(path: Path) -> tuple[float, float, float, float, int, int]:
    """Return (fx, fy, cx, cy, width, height)."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"Intrinsic file {path} must have 3 rows")
    rows = [list(map(float, ln.split())) for ln in lines[:3]]
    fx, _, cx = rows[0][:3]
    _, fy, cy = rows[1][:3]
    width = int(round(2.0 * cx))
    height = int(round(2.0 * cy))
    return fx, fy, cx, cy, width, height


def _quat_inv_xyzw_to_wxyz(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Invert a unit quaternion (xyzw input) and return (qw, qx, qy, qz).

    Inverse of a unit quaternion is its conjugate. Output is in COLMAP's
    wxyz order."""
    n2 = qx * qx + qy * qy + qz * qz + qw * qw
    if n2 < 1e-12:
        raise ValueError("zero-norm quaternion")
    inv = np.array([qw, -qx, -qy, -qz]) / n2  # conjugate / |q|^2
    return inv


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main(args: Args) -> None:
    imgpose_rows: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    prefixes: set[str] = set()
    with args.imgpose.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 12:
                continue  # skip malformed
            name = parts[0].replace("\\", "/")
            # Skip header line (first col has no '/' separator, e.g. "index")
            if "/" not in name:
                continue
            prefix = name.split("/", 1)[0]
            prefixes.add(prefix)
            x, y, z = map(float, parts[1:4])
            # parts[4:7] are roll/pitch/yaw (unused — we use quaternion)
            qx, qy, qz, qw = map(float, parts[7:11])
            # Column 12 (index 11) is the device timestamp in seconds.
            ts = float(parts[11])
            # ImgPose: world_from_cam. COLMAP: cam_from_world.
            # For a pose (R_wc, t_wc) where t_wc is camera center in world,
            #   R_cw = R_wc^T          (quaternion conjugate for unit q)
            #   t_cw = -R_cw @ t_wc
            q_wxyz = _quat_inv_xyzw_to_wxyz(qx, qy, qz, qw)
            R_cw = _quat_to_rotmat_wxyz(q_wxyz)
            t_cw = -R_cw @ np.array([x, y, z])
            imgpose_rows.append((name, q_wxyz, t_cw, ts))

    # Assign a camera_id per unique prefix (deterministic: sorted)
    prefix_to_cam = {p: i + 1 for i, p in enumerate(sorted(prefixes))}

    # Load intrinsics per prefix
    cameras_meta: dict[str, tuple[float, float, float, float, int, int]] = {}
    for prefix in sorted(prefixes):
        intrinsic_file = args.intrinsic_dir / f"{prefix.capitalize()}{args.intrinsic_suffix}"
        if not intrinsic_file.exists():
            # fall back: prefix as-is (e.g. "left_undistort_intrinsic.txt")
            intrinsic_file = args.intrinsic_dir / f"{prefix}{args.intrinsic_suffix}"
        if not intrinsic_file.exists():
            raise FileNotFoundError(f"Could not find intrinsic file for prefix={prefix!r}")
        cameras_meta[prefix] = _parse_intrinsic(intrinsic_file)
        print(f"Loaded intrinsic for {prefix}: {intrinsic_file.name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write cameras.txt (PINHOLE: fx fy cx cy)
    cam_path = args.output_dir / "cameras.txt"
    with cam_path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for prefix in sorted(prefixes):
            cam_id = prefix_to_cam[prefix]
            fx, fy, cx, cy, w, h = cameras_meta[prefix]
            f.write(f"{cam_id} PINHOLE {w} {h} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")

    # Write images.txt
    img_path = args.output_dir / "images.txt"
    with img_path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, (name, q_wxyz, t_cw, _ts) in enumerate(imgpose_rows, start=1):
            prefix = name.split("/", 1)[0]
            cam_id = prefix_to_cam[prefix]
            qw, qx, qy, qz = q_wxyz
            tx, ty, tz = t_cw
            f.write(
                f"{idx} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                f"{tx:.12f} {ty:.12f} {tz:.12f} {cam_id} {name}\n\n"
            )

    # Empty points3D.txt
    (args.output_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n",
        encoding="utf-8",
    )

    # Sidecar: image_name -> device timestamp (seconds). Read by
    # import_lidar_priors_to_db.py to populate the 6dof_pose_priors.timestamp
    # column, which is consumed by the constant-velocity temporal prior in
    # six_dof_prior_global_mapper.
    import json
    timestamps_path = args.output_dir / "timestamps.json"
    timestamps_path.write_text(
        json.dumps(
            {name: ts for name, _q, _t, ts in imgpose_rows},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        f"Wrote COLMAP sparse model to {args.output_dir}:\n"
        f"  cameras={len(prefix_to_cam)}, images={len(imgpose_rows)}, "
        f"points3D=0, timestamps=timestamps.json"
    )


if __name__ == "__main__":
    tyro.cli(main)
