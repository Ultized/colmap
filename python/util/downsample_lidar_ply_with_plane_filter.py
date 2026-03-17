from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from plyfile import PlyData, PlyElement


DEFAULT_INPUT_PLY = Path(
    r"D:\Data\HP300\2025-12-20_qiantai\mapper\2025-12-20_qiantai_num_1\splat_data\downsample_with_normals.ply"
)


@dataclass(frozen=True)
class Args:
    input_ply: Path = DEFAULT_INPUT_PLY
    output_ply: Path | None = None
    voxel_size: float = 0.25
    min_points_per_voxel: int = 3
    max_plane_rms_error_m: float = 0.02
    min_planarity_ratio: float = 5.0


def format_float_for_name(value: float) -> str:
    return str(value).replace(".", "p")


def resolve_output_path(input_ply: Path, output_ply: Path | None, voxel_size: float) -> Path:
    if output_ply is not None:
        return output_ply.resolve()
    suffix = format_float_for_name(voxel_size)
    return input_ply.with_name(f"{input_ply.stem}_voxel_{suffix}_planar.ply")


def require_vertex_fields(vertex_data: np.ndarray, required_fields: tuple[str, ...]) -> None:
    if vertex_data.dtype.names is None:
        raise ValueError("PLY vertex element has no named fields")
    missing_fields = [field for field in required_fields if field not in vertex_data.dtype.names]
    if missing_fields:
        raise ValueError(f"PLY vertex element is missing required fields: {missing_fields}")


def load_vertex_attributes(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply_data = PlyData.read(str(path))
    if "vertex" not in ply_data:
        raise ValueError(f"PLY file has no vertex element: {path}")

    vertex_data = ply_data["vertex"].data
    require_vertex_fields(
        vertex_data,
        ("x", "y", "z", "red", "green", "blue"),
    )

    xyz = np.column_stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
    ).astype(np.float64)

    if all(field in vertex_data.dtype.names for field in ("nx", "ny", "nz")):
        normals = np.column_stack(
            [vertex_data["nx"], vertex_data["ny"], vertex_data["nz"]]
        ).astype(np.float64)
    else:
        normals = np.full_like(xyz, np.nan, dtype=np.float64)

    colors = np.column_stack(
        [vertex_data["red"], vertex_data["green"], vertex_data["blue"]]
    ).astype(np.uint8)

    finite_mask = np.isfinite(xyz).all(axis=1)
    return xyz[finite_mask], normals[finite_mask], colors[finite_mask]


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector
    return vector / norm


def align_normals_to_reference(normals: np.ndarray, reference_normal: np.ndarray) -> np.ndarray:
    if normals.size == 0:
        return normals
    aligned = normals.copy()
    dot_products = aligned @ reference_normal
    aligned[dot_products < 0] *= -1.0
    return aligned


def build_output_vertex_array(
    xyz: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    vertex_dtype = np.dtype(
        [
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("nx", np.float64),
            ("ny", np.float64),
            ("nz", np.float64),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
        ]
    )
    vertex_array = np.empty(xyz.shape[0], dtype=vertex_dtype)
    vertex_array["x"] = xyz[:, 0]
    vertex_array["y"] = xyz[:, 1]
    vertex_array["z"] = xyz[:, 2]
    vertex_array["nx"] = normals[:, 0]
    vertex_array["ny"] = normals[:, 1]
    vertex_array["nz"] = normals[:, 2]
    vertex_array["red"] = colors[:, 0]
    vertex_array["green"] = colors[:, 1]
    vertex_array["blue"] = colors[:, 2]
    return vertex_array


def compute_group_boundaries(sorted_voxel_keys: np.ndarray) -> np.ndarray:
    if sorted_voxel_keys.shape[0] == 0:
        return np.asarray([0], dtype=np.int64)
    changes = np.any(np.diff(sorted_voxel_keys, axis=0) != 0, axis=1)
    return np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(changes).astype(np.int64) + 1,
            np.asarray([sorted_voxel_keys.shape[0]], dtype=np.int64),
        )
    )


def downsample_with_planar_filter(
    xyz: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    min_points_per_voxel: int,
    max_plane_rms_error_m: float,
    min_planarity_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if min_points_per_voxel < 3:
        raise ValueError("min_points_per_voxel must be at least 3")

    origin = xyz.min(axis=0)
    voxel_keys = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    order = np.lexsort((voxel_keys[:, 2], voxel_keys[:, 1], voxel_keys[:, 0]))
    sorted_keys = voxel_keys[order]
    boundaries = compute_group_boundaries(sorted_keys)

    kept_xyz = []
    kept_normals = []
    kept_colors = []
    planar_voxel_count = 0
    skipped_small_voxel_count = 0
    skipped_nonplanar_voxel_count = 0

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        point_indices = order[start:end]
        if point_indices.size < min_points_per_voxel:
            skipped_small_voxel_count += 1
            continue

        local_xyz = xyz[point_indices]
        centroid = local_xyz.mean(axis=0)
        centered_xyz = local_xyz - centroid
        covariance = centered_xyz.T @ centered_xyz / float(point_indices.size)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        tangent_energy = max(float(eigenvalues[1] + eigenvalues[2]), 1e-12)
        planarity_ratio = tangent_energy / max(float(eigenvalues[0]), 1e-12)
        plane_normal = normalize(eigenvectors[:, 0])
        signed_distances = centered_xyz @ plane_normal
        plane_rms_error = float(np.sqrt(np.mean(np.square(signed_distances))))

        if plane_rms_error > max_plane_rms_error_m or planarity_ratio < min_planarity_ratio:
            skipped_nonplanar_voxel_count += 1
            continue

        local_normals = normals[point_indices]
        finite_normal_mask = np.isfinite(local_normals).all(axis=1)
        if np.any(finite_normal_mask):
            aligned_normals = align_normals_to_reference(
                local_normals[finite_normal_mask], plane_normal
            )
            representative_normal = normalize(aligned_normals.mean(axis=0))
            if float(np.linalg.norm(representative_normal)) <= 0:
                representative_normal = plane_normal
        else:
            representative_normal = plane_normal

        representative_color = np.rint(colors[point_indices].mean(axis=0)).astype(np.uint8)

        kept_xyz.append(centroid)
        kept_normals.append(representative_normal)
        kept_colors.append(representative_color)
        planar_voxel_count += 1

    if kept_xyz:
        output_xyz = np.asarray(kept_xyz, dtype=np.float64)
        output_normals = np.asarray(kept_normals, dtype=np.float64)
        output_colors = np.asarray(kept_colors, dtype=np.uint8)
    else:
        output_xyz = np.empty((0, 3), dtype=np.float64)
        output_normals = np.empty((0, 3), dtype=np.float64)
        output_colors = np.empty((0, 3), dtype=np.uint8)

    stats = {
        "input_points": int(xyz.shape[0]),
        "occupied_voxels": int(boundaries.shape[0] - 1),
        "planar_voxels": int(planar_voxel_count),
        "skipped_small_voxels": int(skipped_small_voxel_count),
        "skipped_nonplanar_voxels": int(skipped_nonplanar_voxel_count),
        "output_points": int(output_xyz.shape[0]),
        "retained_point_ratio": float(output_xyz.shape[0] / max(xyz.shape[0], 1)),
    }
    return output_xyz, output_normals, output_colors, stats


def write_ply(path: Path, xyz: np.ndarray, normals: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_array = build_output_vertex_array(xyz, normals, colors)
    PlyData([PlyElement.describe(vertex_array, "vertex")], text=False).write(str(path))


def main() -> None:
    args = tyro.cli(Args)
    input_ply = args.input_ply.resolve()
    output_ply = resolve_output_path(input_ply, args.output_ply, args.voxel_size)

    xyz, normals, colors = load_vertex_attributes(input_ply)
    output_xyz, output_normals, output_colors, stats = downsample_with_planar_filter(
        xyz=xyz,
        normals=normals,
        colors=colors,
        voxel_size=args.voxel_size,
        min_points_per_voxel=args.min_points_per_voxel,
        max_plane_rms_error_m=args.max_plane_rms_error_m,
        min_planarity_ratio=args.min_planarity_ratio,
    )
    write_ply(output_ply, output_xyz, output_normals, output_colors)

    print(f"input_ply: {input_ply}")
    print(f"output_ply: {output_ply}")
    print(f"voxel_size: {args.voxel_size:.6f}")
    print(f"min_points_per_voxel: {args.min_points_per_voxel}")
    print(f"max_plane_rms_error_m: {args.max_plane_rms_error_m:.6f}")
    print(f"min_planarity_ratio: {args.min_planarity_ratio:.6f}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()