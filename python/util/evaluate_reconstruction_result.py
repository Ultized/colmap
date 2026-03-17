from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from plyfile import PlyData
from scipy.spatial import cKDTree

import pycolmap


@dataclass(frozen=True)
class Args:
    model: Path
    lidar_ply: Path | None = None
    reference_model: Path | None = None
    min_track_length: int = 4
    max_point_reprojection_error: float = 3.0
    alignable_distance_m: float = 0.10
    distance_thresholds_m: list[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.10]
    )


def resolve_model_dir(path: Path) -> Path:
    path = path.resolve()
    if any((path / f"images{suffix}").is_file() for suffix in (".bin", ".txt")):
        return path
    only_child = path / "0"
    if any(
        (only_child / f"images{suffix}").is_file()
        for suffix in (".bin", ".txt")
    ):
        return only_child
    raise FileNotFoundError(f"Could not find COLMAP sparse model at: {path}")


def projection_center(cam_from_world: pycolmap.Rigid3d) -> np.ndarray:
    rotation = np.asarray(cam_from_world.matrix(), dtype=np.float64)[:, :3]
    translation = np.asarray(cam_from_world.translation, dtype=np.float64)
    return (-rotation.T @ translation).astype(np.float64)


def rotation_error_deg(
    ref_cam_from_world: pycolmap.Rigid3d,
    query_cam_from_world: pycolmap.Rigid3d,
) -> float:
    ref_rotation = np.asarray(
        ref_cam_from_world.matrix(), dtype=np.float64
    )[:, :3]
    query_rotation = np.asarray(
        query_cam_from_world.matrix(), dtype=np.float64
    )[:, :3]
    relative_rotation = query_rotation @ ref_rotation.T
    trace_value = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_value)))


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def safe_median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.max())


def load_ply_xyz(path: Path) -> np.ndarray:
    ply_data = PlyData.read(str(path))
    if "vertex" not in ply_data:
        raise ValueError(f"PLY file has no vertex element: {path}")
    vertex_data = ply_data["vertex"].data
    for axis in ("x", "y", "z"):
        if axis not in vertex_data.dtype.names:
            raise ValueError(f"PLY vertex element is missing '{axis}': {path}")
    xyz = np.column_stack(
        [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
    ).astype(np.float64)
    finite_mask = np.isfinite(xyz).all(axis=1)
    return xyz[finite_mask]


def collect_sparse_points(
    reconstruction: pycolmap.Reconstruction,
    min_track_length: int,
    max_point_reprojection_error: float,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = []
    point_errors = []
    for point in reconstruction.points3D.values():
        if point.track.length() < min_track_length:
            continue
        if (
            max_point_reprojection_error > 0
            and point.error > max_point_reprojection_error
        ):
            continue
        point_xyz = np.asarray(point.xyz, dtype=np.float64)
        if not np.isfinite(point_xyz).all():
            continue
        xyz.append(point_xyz)
        point_errors.append(float(point.error))

    if not xyz:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    return (
        np.asarray(xyz, dtype=np.float64),
        np.asarray(point_errors, dtype=np.float64),
    )


def print_metric(name: str, value: object) -> None:
    print(f"{name}: {value}")


def evaluate_visual_consistency(
    reconstruction: pycolmap.Reconstruction,
) -> None:
    registered_images = [
        image for image in reconstruction.images.values() if image.has_pose
    ]
    num_points_per_registered_image = np.asarray(
        [image.num_points3D for image in registered_images], dtype=np.float64
    )
    zero_observation_images = int(np.sum(num_points_per_registered_image <= 0))

    point_errors = np.asarray(
        [
            point.error
            for point in reconstruction.points3D.values()
            if point.error >= 0
        ],
        dtype=np.float64,
    )

    print("[1] reconstruction_status")
    print_metric("registered_images", len(registered_images))
    print_metric("points3D", reconstruction.num_points3D())
    print_metric(
        "reconstruction_ok",
        int(len(registered_images) > 0 and reconstruction.num_points3D() > 0),
    )

    print("[2] visual_internal_consistency")
    print_metric("zero_observation_images", zero_observation_images)
    print_metric(
        "mean_observations_per_registered_image",
        f"{safe_mean(num_points_per_registered_image):.6f}",
    )
    print_metric(
        "mean_reprojection_error_px",
        f"{reconstruction.compute_mean_reprojection_error():.6f}",
    )
    print_metric(
        "median_point_reprojection_error_px",
        f"{safe_median(point_errors):.6f}",
    )
    print_metric(
        "p95_point_reprojection_error_px",
        f"{safe_percentile(point_errors, 95):.6f}",
    )


def evaluate_sparse_lidar_alignment(
    reconstruction: pycolmap.Reconstruction,
    lidar_xyz: np.ndarray,
    min_track_length: int,
    max_point_reprojection_error: float,
    alignable_distance_m: float,
    distance_thresholds_m: list[float],
) -> None:
    sparse_xyz, sparse_point_errors = collect_sparse_points(
        reconstruction=reconstruction,
        min_track_length=min_track_length,
        max_point_reprojection_error=max_point_reprojection_error,
    )
    print("[3] sparse_lidar_alignment")
    print_metric("lidar_points", lidar_xyz.shape[0])
    print_metric("stable_sparse_points_for_lidar_eval", sparse_xyz.shape[0])
    print_metric("stable_sparse_min_track_length", min_track_length)
    print_metric(
        "stable_sparse_max_point_reprojection_error_px",
        f"{max_point_reprojection_error:.6f}",
    )

    if sparse_xyz.shape[0] == 0:
        print_metric("sparse_lidar_eval_skipped", 1)
        return

    lidar_tree = cKDTree(lidar_xyz)
    nearest_distances, _ = lidar_tree.query(sparse_xyz, k=1, workers=-1)
    nearest_distances = np.asarray(nearest_distances, dtype=np.float64)
    alignable_mask = nearest_distances <= alignable_distance_m
    aligned_distances = nearest_distances[alignable_mask]

    print_metric(
        "sparse_to_lidar_mean_nn_distance_m",
        f"{safe_mean(nearest_distances):.6f}",
    )
    print_metric(
        "sparse_to_lidar_median_nn_distance_m",
        f"{safe_median(nearest_distances):.6f}",
    )
    print_metric(
        "sparse_to_lidar_p95_nn_distance_m",
        f"{safe_percentile(nearest_distances, 95):.6f}",
    )
    print_metric(
        "sparse_to_lidar_max_nn_distance_m",
        f"{safe_max(nearest_distances):.6f}",
    )
    print_metric(
        "sparse_to_lidar_alignable_distance_m",
        f"{alignable_distance_m:.6f}",
    )
    print_metric("sparse_to_lidar_alignable_points", int(alignable_mask.sum()))
    print_metric(
        "sparse_to_lidar_alignable_ratio",
        f"{float(alignable_mask.mean()):.6f}",
    )
    print_metric(
        "sparse_to_lidar_alignable_mean_distance_m",
        f"{safe_mean(aligned_distances):.6f}",
    )
    print_metric(
        "sparse_to_lidar_alignable_p95_distance_m",
        f"{safe_percentile(aligned_distances, 95):.6f}",
    )
    print_metric(
        "stable_sparse_mean_point_reprojection_error_px",
        f"{safe_mean(sparse_point_errors):.6f}",
    )
    for threshold_m in sorted(set(distance_thresholds_m)):
        threshold_mask = nearest_distances <= threshold_m
        threshold_label_mm = int(round(threshold_m * 1000.0))
        print_metric(
            f"sparse_to_lidar_ratio_within_{threshold_label_mm}mm",
            f"{float(threshold_mask.mean()):.6f}",
        )


def evaluate_secondary_camera_pose(
    reference_model: Path, query_reconstruction: pycolmap.Reconstruction
) -> None:
    reference_reconstruction = pycolmap.Reconstruction(str(reference_model))
    reference_by_name = {
        str(image.name): image
        for image in reference_reconstruction.images.values()
    }
    query_by_name = {
        str(image.name): image for image in query_reconstruction.images.values()
    }
    common_names = sorted(set(reference_by_name) & set(query_by_name))

    center_errors = []
    rotation_errors = []
    for image_name in common_names:
        ref_image = reference_by_name[image_name]
        query_image = query_by_name[image_name]
        ref_center = projection_center(ref_image.cam_from_world())
        query_center = projection_center(query_image.cam_from_world())
        if not (
            np.isfinite(ref_center).all()
            and np.isfinite(query_center).all()
        ):
            continue
        center_errors.append(float(np.linalg.norm(ref_center - query_center)))
        rotation_errors.append(
            rotation_error_deg(
                ref_image.cam_from_world(),
                query_image.cam_from_world(),
            )
        )

    center_errors_np = np.asarray(center_errors, dtype=np.float64)
    rotation_errors_np = np.asarray(rotation_errors, dtype=np.float64)

    print("[4] camera_pose_delta_secondary")
    print_metric(
        "camera_pose_delta_note",
        "secondary_only_pose_priors_can_be_less_reliable_than_lidar_geometry",
    )
    print_metric("common_images", len(common_names))
    print_metric("mean_center_error_m", f"{safe_mean(center_errors_np):.6f}")
    print_metric(
        "median_center_error_m",
        f"{safe_median(center_errors_np):.6f}",
    )
    print_metric(
        "p95_center_error_m",
        f"{safe_percentile(center_errors_np, 95):.6f}",
    )
    print_metric("max_center_error_m", f"{safe_max(center_errors_np):.6f}")
    print_metric(
        "mean_rotation_error_deg",
        f"{safe_mean(rotation_errors_np):.6f}",
    )
    print_metric(
        "max_rotation_error_deg",
        f"{safe_max(rotation_errors_np):.6f}",
    )


def main(args: Args) -> int:
    model_dir = resolve_model_dir(args.model)
    reconstruction = pycolmap.Reconstruction(str(model_dir))

    print_metric("model", model_dir)
    if args.lidar_ply is not None:
        print_metric("lidar_ply", args.lidar_ply.resolve())
    if args.reference_model is not None:
        print_metric("reference_model", resolve_model_dir(args.reference_model))

    evaluate_visual_consistency(reconstruction)

    if args.lidar_ply is not None:
        lidar_xyz = load_ply_xyz(args.lidar_ply.resolve())
        evaluate_sparse_lidar_alignment(
            reconstruction=reconstruction,
            lidar_xyz=lidar_xyz,
            min_track_length=args.min_track_length,
            max_point_reprojection_error=args.max_point_reprojection_error,
            alignable_distance_m=args.alignable_distance_m,
            distance_thresholds_m=args.distance_thresholds_m,
        )

    if args.reference_model is not None:
        reference_model = resolve_model_dir(args.reference_model)
        evaluate_secondary_camera_pose(reference_model, reconstruction)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))