import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

import pycolmap

CAMERA_SENSOR_TYPE = 0
MODEL_BASENAMES = (
    "cameras",
    "frames",
    "images",
    "points3D",
    "rigs",
)


@dataclass(frozen=True)
class DatabaseImageRecord:
    image_id: int
    name: str
    camera_id: int
    frame_id: int | None


@dataclass(frozen=True)
class DatabaseFrameRecord:
    frame_id: int
    rig_id: int
    data_ids: tuple[pycolmap.data_t, ...]


@dataclass(frozen=True)
class RewriteStats:
    renamed_images: int
    updated_image_camera_ids: int
    updated_frame_rig_ids: int
    rewritten_frame_data_ids: int
    dropped_images: int
    dropped_frames: int
    filtered_points3D: int
    dropped_points3D: int


@dataclass(frozen=True)
class Args:
    database_path: Path
    input_model: Path
    output_model: Path | None = None
    output_format: Literal["binary", "text"] = "text"
    allow_partial: bool = False


def sensor_type_to_text(sensor_type: int) -> str:
    if sensor_type == -1:
        return "INVALID"
    if sensor_type == 0:
        return "CAMERA"
    if sensor_type == 1:
        return "IMU"
    raise ValueError(f"Unsupported sensor type: {sensor_type}")


def load_database_images(database_path: Path) -> dict[int, DatabaseImageRecord]:
    query = """
        SELECT
            images.image_id,
            images.name,
            images.camera_id,
            frame_data.frame_id
        FROM images
        LEFT JOIN frame_data
            ON images.image_id = frame_data.data_id
           AND frame_data.sensor_type = ?
    """
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(query, (CAMERA_SENSOR_TYPE,)).fetchall()

    image_records: dict[int, DatabaseImageRecord] = {}
    for image_id, name, camera_id, frame_id in rows:
        image_records[int(image_id)] = DatabaseImageRecord(
            image_id=int(image_id),
            name=str(name),
            camera_id=int(camera_id),
            frame_id=None if frame_id is None else int(frame_id),
        )
    return image_records


def load_database_frames(database_path: Path) -> dict[int, DatabaseFrameRecord]:
    query = """
        SELECT
            frames.frame_id,
            frames.rig_id,
            frame_data.data_id,
            frame_data.sensor_id,
            frame_data.sensor_type
        FROM frames
        LEFT JOIN frame_data ON frames.frame_id = frame_data.frame_id
        ORDER BY
            frames.frame_id,
            frame_data.sensor_type,
            frame_data.sensor_id,
            frame_data.data_id
    """
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(query).fetchall()

    frame_records: dict[int, DatabaseFrameRecord] = {}
    pending_data_ids: list[pycolmap.data_t] = []
    current_frame_id: int | None = None
    current_rig_id: int | None = None

    def flush_current_frame() -> None:
        nonlocal pending_data_ids, current_frame_id, current_rig_id
        if current_frame_id is None or current_rig_id is None:
            return
        frame_records[current_frame_id] = DatabaseFrameRecord(
            frame_id=current_frame_id,
            rig_id=current_rig_id,
            data_ids=tuple(pending_data_ids),
        )
        pending_data_ids = []
        current_frame_id = None
        current_rig_id = None

    for frame_id, rig_id, data_id, sensor_id, sensor_type in rows:
        frame_id = int(frame_id)
        rig_id = int(rig_id)
        if current_frame_id != frame_id:
            flush_current_frame()
            current_frame_id = frame_id
            current_rig_id = rig_id

        if data_id is None:
            continue

        pending_data_ids.append(
            pycolmap.data_t(
                sensor_id=pycolmap.sensor_t(
                    type=pycolmap.SensorType(int(sensor_type)),
                    id=int(sensor_id),
                ),
                id=int(data_id),
            )
        )

    flush_current_frame()
    return frame_records


def remove_existing_model_files(model_path: Path) -> None:
    for basename in MODEL_BASENAMES:
        for suffix in (".bin", ".txt"):
            candidate = model_path / f"{basename}{suffix}"
            if candidate.exists():
                candidate.unlink()


def patch_images_text(
    images_path: Path,
    database_images: dict[int, DatabaseImageRecord],
) -> tuple[int, int, list[int]]:
    lines = images_path.read_text(encoding="utf-8").splitlines()
    rewritten_lines: list[str] = []
    missing_image_ids: list[int] = []
    renamed_images = 0
    updated_image_camera_ids = 0
    is_metadata_line = True
    skip_points2D_line = False

    for line in lines:
        if line.startswith("#"):
            rewritten_lines.append(line)
            continue

        if is_metadata_line:
            stripped = line.strip()
            if not stripped:
                raise ValueError(
                    "Encountered an empty metadata line while rewriting "
                    f"{images_path}"
                )
            parts = line.split(" ", 9)
            image_id = int(parts[0])
            database_image = database_images.get(image_id)
            if database_image is None:
                missing_image_ids.append(image_id)
                print(f"drop image_id={image_id}: missing from database")
                skip_points2D_line = True
            else:
                if int(parts[8]) != database_image.camera_id:
                    print(
                        "update image camera_id="
                        f"{image_id}: {parts[8]} -> {database_image.camera_id}"
                    )
                    updated_image_camera_ids += 1
                if parts[9] != database_image.name:
                    print(
                        "rename image_id="
                        f"{image_id}: '{parts[9]}' -> '{database_image.name}'"
                    )
                    renamed_images += 1
                rewritten_lines.append(
                    " ".join(
                        [
                            *parts[:8],
                            str(database_image.camera_id),
                            database_image.name,
                        ]
                    )
                )
        else:
            if skip_points2D_line:
                skip_points2D_line = False
            else:
                rewritten_lines.append(line)

        is_metadata_line = not is_metadata_line

    images_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")
    return updated_image_camera_ids, renamed_images, missing_image_ids


def patch_frames_text(
    frames_path: Path,
    database_frames: dict[int, DatabaseFrameRecord],
) -> tuple[int, int, list[int]]:
    lines = frames_path.read_text(encoding="utf-8").splitlines()
    rewritten_lines: list[str] = []
    missing_frame_ids: list[int] = []
    updated_frame_rig_ids = 0
    rewritten_frame_data_ids = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            rewritten_lines.append(line)
            continue

        parts = stripped.split()
        frame_id = int(parts[0])
        database_frame = database_frames.get(frame_id)
        if database_frame is None:
            missing_frame_ids.append(frame_id)
            print(f"drop frame_id={frame_id}: missing from database")
            continue

        old_rig_id = int(parts[1])
        old_num_data_ids = int(parts[9])
        old_data_tokens = parts[10:]
        new_data_tokens: list[str] = []
        for data_id in database_frame.data_ids:
            new_data_tokens.extend(
                [
                    sensor_type_to_text(int(data_id.sensor_id.type)),
                    str(data_id.sensor_id.id),
                    str(data_id.id),
                ]
            )

        if old_rig_id != database_frame.rig_id:
            print(
                "update frame rig_id="
                f"{frame_id}: {old_rig_id} -> {database_frame.rig_id}"
            )
            updated_frame_rig_ids += 1

        if (
            old_num_data_ids != len(database_frame.data_ids)
            or old_data_tokens != new_data_tokens
        ):
            print(f"rewrite frame data_ids for frame_id={frame_id}")
            rewritten_frame_data_ids += 1

        rewritten_lines.append(
            " ".join(
                [
                    parts[0],
                    str(database_frame.rig_id),
                    *parts[2:9],
                    str(len(database_frame.data_ids)),
                    *new_data_tokens,
                ]
            )
        )

    frames_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")
    return updated_frame_rig_ids, rewritten_frame_data_ids, missing_frame_ids


def patch_points3D_text(
    points3D_path: Path,
    dropped_image_ids: set[int],
) -> tuple[int, int]:
    lines = points3D_path.read_text(encoding="utf-8").splitlines()
    rewritten_lines: list[str] = []
    filtered_points3D = 0
    dropped_points3D = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            rewritten_lines.append(line)
            continue

        parts = stripped.split()
        track_tokens = parts[8:]
        filtered_track_tokens: list[str] = []
        removed_track = False
        for index in range(0, len(track_tokens), 2):
            image_id = int(track_tokens[index])
            point2D_idx = track_tokens[index + 1]
            if image_id in dropped_image_ids:
                removed_track = True
                continue
            filtered_track_tokens.extend([str(image_id), point2D_idx])

        if len(filtered_track_tokens) < 4:
            dropped_points3D += 1
            continue

        if removed_track:
            filtered_points3D += 1

        rewritten_lines.append(
            " ".join([*parts[:8], *filtered_track_tokens])
        )

    points3D_path.write_text(
        "\n".join(rewritten_lines) + "\n",
        encoding="utf-8",
    )
    return filtered_points3D, dropped_points3D


def export_and_patch_text_model(
    reconstruction: pycolmap.Reconstruction,
    output_model: Path,
    database_images: dict[int, DatabaseImageRecord],
    database_frames: dict[int, DatabaseFrameRecord],
) -> tuple[RewriteStats, list[int], list[int]]:
    reconstruction.write_text(str(output_model))

    (
        updated_image_camera_ids,
        renamed_images,
        missing_image_ids,
    ) = patch_images_text(
        output_model / "images.txt",
        database_images,
    )
    updated_frame_rig_ids, rewritten_frame_data_ids, missing_frame_ids = (
        patch_frames_text(output_model / "frames.txt", database_frames)
    )
    filtered_points3D, dropped_points3D = patch_points3D_text(
        output_model / "points3D.txt",
        set(missing_image_ids),
    )

    stats = RewriteStats(
        renamed_images=renamed_images,
        updated_image_camera_ids=updated_image_camera_ids,
        updated_frame_rig_ids=updated_frame_rig_ids,
        rewritten_frame_data_ids=rewritten_frame_data_ids,
        dropped_images=len(set(missing_image_ids)),
        dropped_frames=len(set(missing_frame_ids)),
        filtered_points3D=filtered_points3D,
        dropped_points3D=dropped_points3D,
    )
    return stats, missing_image_ids, missing_frame_ids


def main(args: Args) -> int:
    database_path = args.database_path
    input_model = args.input_model
    output_model = (
        args.output_model
        if args.output_model is not None
        else input_model.parent / "rename"
    )

    if not database_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database_path}")
    if not input_model.is_dir():
        raise FileNotFoundError(f"Input model does not exist: {input_model}")

    output_model.mkdir(parents=True, exist_ok=True)
    remove_existing_model_files(output_model)

    database_images = load_database_images(database_path)
    database_frames = load_database_frames(database_path)

    reconstruction = pycolmap.Reconstruction(str(input_model))

    stats, missing_image_ids, missing_frame_ids = export_and_patch_text_model(
        reconstruction,
        output_model,
        database_images,
        database_frames,
    )

    has_blocking_issue = bool(missing_image_ids or missing_frame_ids)
    if has_blocking_issue and not args.allow_partial:
        if missing_image_ids:
            print(
                "Missing image IDs in database:",
                sorted(set(missing_image_ids)),
            )
        if missing_frame_ids:
            print(
                "Missing frame IDs in database:",
                sorted(set(missing_frame_ids)),
            )
        print(
            "Aborting because the sparse model cannot be made fully "
            "consistent by ID."
        )
        print(
            "Use --allow_partial if you still want to write the adjusted "
            "model."
        )
        return 1

    if args.output_format == "binary" and not has_blocking_issue:
        patched_reconstruction = pycolmap.Reconstruction(str(output_model))
        patched_reconstruction.write(str(output_model))
    elif args.output_format == "binary":
        print(
            "Skip binary export because the patched text model still has "
            "IDs missing from the database."
        )

    print()
    print("Finished rewriting sparse model")
    print(f"  input_model:  {input_model}")
    print(f"  output_model: {output_model}")
    print(f"  output_format: {args.output_format}")
    print(f"  renamed_images: {stats.renamed_images}")
    print(f"  updated_image_camera_ids: {stats.updated_image_camera_ids}")
    print(f"  updated_frame_rig_ids: {stats.updated_frame_rig_ids}")
    print(f"  rewritten_frame_data_ids: {stats.rewritten_frame_data_ids}")
    print(f"  dropped_images: {stats.dropped_images}")
    print(f"  dropped_frames: {stats.dropped_frames}")
    print(f"  filtered_points3D: {stats.filtered_points3D}")
    print(f"  dropped_points3D: {stats.dropped_points3D}")
    print(f"  missing_image_ids: {len(set(missing_image_ids))}")
    print(f"  missing_frame_ids: {len(set(missing_frame_ids))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))