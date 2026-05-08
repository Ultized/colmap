"""Voxel-downsample a colored LAS point cloud and write a PLY with per-point
normals compatible with the format expected by the 6DoF + LiDAR pipeline
(see colmap/scene/lidar_point_cloud.h).

Why this script exists: re_dev's ``downsample_lidar_ply_with_plane_filter.py``
starts from PLY; SHARE SLAM S20 dumps LAS. This fills the gap without taking an
Open3D dependency (no Python 3.13 wheel yet). Uses laspy for LAS I/O,
scikit-learn KDTree + numpy PCA for normals.

Output PLY fields match the Open3D-style header of the reference
``downsample_with_normals.ply``::

    x y z (double)
    nx ny nz (double)
    red green blue (uchar)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import tyro
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree


@dataclass(frozen=True)
class Args:
    input_las: Path
    output_ply: Path
    voxel_size: float = 0.05  # meters; 5 cm is a good indoor default
    knn_for_normals: int = 20
    orient_normals_toward: tuple[float, float, float] = (0.0, 0.0, 10.0)


def _voxel_downsample(
    xyz: np.ndarray, rgb: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average points per voxel. Returns (xyz_down, rgb_down)."""
    # voxel indices per point
    origin = xyz.min(axis=0)
    keys = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    # pack (i,j,k) into unique ids via linearized hash
    # using 21-bit shifts avoids collisions for voxel counts up to ~2M per axis
    lin = (keys[:, 0] & 0x1FFFFF) \
        | ((keys[:, 1] & 0x1FFFFF) << 21) \
        | ((keys[:, 2] & 0x1FFFFF) << 42)
    # Vectorized segmented mean via np.add.reduceat (O(N) not O(N_voxels))
    order = np.argsort(lin, kind="stable")
    lin_sorted = lin[order]
    # Indices where a new voxel group starts
    starts = np.concatenate(([0], np.where(np.diff(lin_sorted) != 0)[0] + 1))
    counts = np.diff(np.concatenate((starts, [len(lin_sorted)])))

    def _segmented_mean(data: np.ndarray) -> np.ndarray:
        summed = np.add.reduceat(data[order], starts, axis=0)
        return summed / counts[:, None]

    return _segmented_mean(xyz), _segmented_mean(rgb)


def _estimate_normals(xyz: np.ndarray, knn: int) -> np.ndarray:
    tree = KDTree(xyz, leaf_size=32, metric="euclidean")
    _, idx = tree.query(xyz, k=knn)
    neigh = xyz[idx]  # (N, K, 3)
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    # Covariance per point via einsum
    cov = np.einsum("nki,nkj->nij", centered, centered) / knn  # (N, 3, 3)
    # Smallest eigenvector = normal. np.linalg.eigh returns ascending eigenvalues.
    _, eigvec = np.linalg.eigh(cov)
    normals = eigvec[:, :, 0]
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return normals / norms


def _orient_normals_toward(normals: np.ndarray, xyz: np.ndarray, viewpoint: np.ndarray) -> np.ndarray:
    """Flip normals to point toward a viewpoint (typical scene-facing orientation)."""
    view_vec = viewpoint[None, :] - xyz
    dot = np.einsum("ni,ni->n", normals, view_vec)
    sign = np.where(dot < 0, -1.0, 1.0)
    return normals * sign[:, None]


def main(args: Args) -> None:
    t0 = time.time()
    print(f"Reading {args.input_las} ...")
    las = laspy.read(str(args.input_las))
    xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
    # laspy returns color as uint16 in [0..65535] scaled, regardless of file storage
    if hasattr(las, "red") and las.red.max() > 0:
        rgb16 = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float64)
        # Normalize depending on whether it's 8-bit shifted or 16-bit
        if rgb16.max() > 255:
            rgb = (rgb16 / 256.0).clip(0, 255)
        else:
            rgb = rgb16.clip(0, 255)
    else:
        rgb = np.full_like(xyz, 200.0)
    print(f"  loaded {xyz.shape[0]:,} points in {time.time() - t0:.1f}s")

    t1 = time.time()
    xyz_ds, rgb_ds = _voxel_downsample(xyz, rgb, args.voxel_size)
    print(f"  downsampled to {xyz_ds.shape[0]:,} points "
          f"(voxel={args.voxel_size} m) in {time.time() - t1:.1f}s")

    t2 = time.time()
    normals = _estimate_normals(xyz_ds, args.knn_for_normals)
    normals = _orient_normals_toward(normals, xyz_ds, np.array(args.orient_normals_toward))
    print(f"  normals (knn={args.knn_for_normals}) in {time.time() - t2:.1f}s")

    # Write PLY
    rgb_u8 = rgb_ds.astype(np.uint8)
    verts = np.empty(xyz_ds.shape[0], dtype=[
        ("x", "f8"), ("y", "f8"), ("z", "f8"),
        ("nx", "f8"), ("ny", "f8"), ("nz", "f8"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    verts["x"], verts["y"], verts["z"] = xyz_ds[:, 0], xyz_ds[:, 1], xyz_ds[:, 2]
    verts["nx"], verts["ny"], verts["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]

    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False, byte_order="<").write(str(args.output_ply))
    print(f"Wrote {args.output_ply} ({xyz_ds.shape[0]:,} points, "
          f"total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    tyro.cli(main)
