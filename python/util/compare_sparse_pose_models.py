from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import tyro


@dataclass(frozen=True)
class Args:
    reference_model: Path
    query_model: Path


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


def main(args: Args) -> int:
    reference_model = resolve_model_dir(args.reference_model)
    query_model = resolve_model_dir(args.query_model)

    reference_reconstruction = pycolmap.Reconstruction(str(reference_model))
    query_reconstruction = pycolmap.Reconstruction(str(query_model))

    reference_by_name = {
        str(image.name): image
        for image in reference_reconstruction.images.values()
    }
    query_by_name = {
        str(image.name): image for image in query_reconstruction.images.values()
    }

    common_names = sorted(set(reference_by_name) & set(query_by_name))
    if not common_names:
        raise ValueError("No common image names between the two sparse models")

    center_errors = []
    rotation_errors = []
    skipped_nonfinite = 0

    for image_name in common_names:
        ref_image = reference_by_name[image_name]
        query_image = query_by_name[image_name]
        ref_cam_from_world = ref_image.cam_from_world()
        query_cam_from_world = query_image.cam_from_world()
        ref_center = projection_center(ref_cam_from_world)
        query_center = projection_center(query_cam_from_world)
        if not (
            np.isfinite(ref_center).all()
            and np.isfinite(query_center).all()
        ):
            skipped_nonfinite += 1
            continue

        center_error = np.linalg.norm(ref_center - query_center)
        center_errors.append(float(center_error))
        rotation_errors.append(
            rotation_error_deg(ref_cam_from_world, query_cam_from_world)
        )

    if not center_errors:
        raise ValueError("All common image pairs had non-finite camera centers")

    center_errors_np = np.asarray(center_errors, dtype=np.float64)
    rotation_errors_np = np.asarray(rotation_errors, dtype=np.float64)

    print(f"reference_model:        {reference_model}")
    print(f"query_model:            {query_model}")
    print(f"common_images:          {len(common_names)}")
    print(f"skipped_nonfinite:      {skipped_nonfinite}")
    print(f"mean_center_error_m:    {center_errors_np.mean():.6f}")
    print(f"median_center_error_m:  {np.median(center_errors_np):.6f}")
    print(f"max_center_error_m:     {center_errors_np.max():.6f}")
    print(f"p95_center_error_m:     {np.percentile(center_errors_np, 95):.6f}")
    print(f"mean_rotation_error_deg:{rotation_errors_np.mean():.6f}")
    print(f"max_rotation_error_deg: {rotation_errors_np.max():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))