import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass(frozen=True)
class Args:
    database_path: Path
    image_path: Path
    input_model: Path
    triangulated_model: Path
    ba_model: Path
    colmap_path: Path = Path("build/src/colmap/exe/RelWithDebInfo/colmap.exe")
    clear_points: bool = True
    tri_ba_global_max_refinements: int = 0
    tri_ignore_two_view_tracks: bool = False
    tri_filter_max_reproj_error: float = 8.0
    tri_filter_min_tri_angle: float = 0.5
    tri_create_max_angle_error: float = 3.0
    tri_continue_max_angle_error: float = 3.0
    tri_merge_max_reproj_error: float = 4.0
    tri_complete_max_reproj_error: float = 4.0
    ba_refine_points3D: bool = True
    ba_refine_rig_from_world: bool = True
    ba_refine_sensor_from_rig: bool = False


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(command: list[str], cwd: Path) -> None:
    printable = subprocess.list2cmdline(command)
    print()
    print(f"Running: {printable}")
    subprocess.run(command, cwd=cwd, check=True)


def build_point_triangulator_command(
    args: Args,
    colmap_path: Path,
) -> list[str]:
    return [
        str(colmap_path),
        "point_triangulator",
        "--database_path",
        str(Path(args.database_path)),
        "--image_path",
        str(Path(args.image_path)),
        "--input_path",
        str(Path(args.input_model)),
        "--output_path",
        str(Path(args.triangulated_model)),
        "--clear_points",
        str(int(args.clear_points)),
        "--refine_intrinsics",
        "0",
        "--Mapper.ba_global_max_refinements",
        str(args.tri_ba_global_max_refinements),
        "--Mapper.tri_ignore_two_view_tracks",
        str(int(args.tri_ignore_two_view_tracks)),
        "--Mapper.filter_max_reproj_error",
        str(args.tri_filter_max_reproj_error),
        "--Mapper.filter_min_tri_angle",
        str(args.tri_filter_min_tri_angle),
        "--Mapper.tri_create_max_angle_error",
        str(args.tri_create_max_angle_error),
        "--Mapper.tri_continue_max_angle_error",
        str(args.tri_continue_max_angle_error),
        "--Mapper.tri_merge_max_reproj_error",
        str(args.tri_merge_max_reproj_error),
        "--Mapper.tri_complete_max_reproj_error",
        str(args.tri_complete_max_reproj_error),
    ]


def build_bundle_adjuster_command(
    args: Args,
    colmap_path: Path,
) -> list[str]:
    return [
        str(colmap_path),
        "bundle_adjuster",
        "--input_path",
        str(Path(args.triangulated_model)),
        "--output_path",
        str(Path(args.ba_model)),
        "--BundleAdjustment.refine_focal_length",
        "0",
        "--BundleAdjustment.refine_principal_point",
        "0",
        "--BundleAdjustment.refine_extra_params",
        "0",
        "--BundleAdjustment.refine_rig_from_world",
        str(int(args.ba_refine_rig_from_world)),
        "--BundleAdjustment.refine_sensor_from_rig",
        str(int(args.ba_refine_sensor_from_rig)),
        "--BundleAdjustment.refine_points3D",
        str(int(args.ba_refine_points3D)),
    ]


def main(args: Args) -> int:
    workspace_root = Path(__file__).resolve().parents[2]
    colmap_path = Path(args.colmap_path)
    if not colmap_path.is_absolute():
        colmap_path = workspace_root / colmap_path
    colmap_path = colmap_path.resolve()

    if not colmap_path.is_file():
        raise FileNotFoundError(
            f"COLMAP executable does not exist: {colmap_path}"
        )

    required_paths = {
        "database_path": Path(args.database_path),
        "image_path": Path(args.image_path),
        "input_model": Path(args.input_model),
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    triangulated_model = Path(args.triangulated_model)
    ba_model = Path(args.ba_model)
    ensure_directory(triangulated_model)
    ensure_directory(ba_model)

    point_triangulator_command = build_point_triangulator_command(
        args,
        colmap_path,
    )
    bundle_adjuster_command = build_bundle_adjuster_command(
        args,
        colmap_path,
    )

    print("Stage 1/2: point_triangulator")
    run_command(point_triangulator_command, workspace_root)

    print()
    print("Stage 2/2: bundle_adjuster")
    run_command(bundle_adjuster_command, workspace_root)

    print()
    print("Finished sparse reconstruction refinement pipeline")
    print(f"  triangulated_model: {triangulated_model}")
    print(f"  ba_model:           {ba_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))