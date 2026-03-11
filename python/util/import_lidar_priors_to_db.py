import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from colmap_data import CAMERA_MODEL_NAMES, qvec2rotmat, read_model

CAMERA_SENSOR_TYPE = 0
SIX_DOF_POSE_PRIOR_TABLE = "6dof_pose_priors"
LEGACY_SIX_DOF_POSE_PRIOR_TABLE = "lidar_pose_priors"
COORDINATE_SYSTEM_TO_CODE = {
    "undefined": -1,
    "wgs84": 0,
    "cartesian": 1,
}


@dataclass(frozen=True)
class CameraRow:
    camera_id: int
    model: int
    width: int
    height: int
    params: np.ndarray
    prior_focal_length: bool


@dataclass(frozen=True)
class DatabaseImageRow:
    image_id: int
    name: str
    camera_id: int


@dataclass(frozen=True)
class MatchRecord:
    image_id: int
    image_name: str
    database_camera_id: int
    source_model_image_id: int
    source_model_camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    position: np.ndarray
    gravity: np.ndarray


@dataclass(frozen=True)
class ImportStats:
    matched_images: int
    skipped_model_images: int
    skipped_missing_files: int
    updated_cameras: int
    reused_cameras: int
    updated_image_camera_links: int
    inserted_pose_priors: int
    updated_pose_priors: int
    upserted_6dof_pose_rows: int


@dataclass(frozen=True)
class Args:
    database_path: Path
    sparse_model_path: Path
    image_path: Path
    output_database_path: Path | None = None
    position_std: float = 0.1
    rotation_std_deg: float = 1.0
    coordinate_system: Literal["cartesian", "wgs84", "undefined"] = "cartesian"
    prior_focal_length: bool = True
    clear_existing_pose_priors: bool = True
    create_6dof_pose_table: bool = True
    drop_legacy_lidar_pose_table: bool = True
    validate_image_files: bool = True
    fail_on_missing_images: bool = False
    overwrite_output: bool = False


def is_model_directory(path: Path) -> bool:
    return any(
        (path / f"cameras{suffix}").is_file() for suffix in (".bin", ".txt")
    ) and any(
        (path / f"images{suffix}").is_file() for suffix in (".bin", ".txt")
    )


def resolve_sparse_model_path(path: Path) -> Path:
    path = path.resolve()
    if is_model_directory(path):
        return path

    child_candidates = [
        child
        for child in path.iterdir()
        if child.is_dir() and is_model_directory(child)
    ]
    if len(child_candidates) == 1:
        return child_candidates[0]

    if not child_candidates:
        raise FileNotFoundError(
            f"Could not find a COLMAP sparse model under: {path}"
        )

    candidate_text = ", ".join(str(candidate) for candidate in child_candidates)
    raise ValueError(
        "Found multiple sparse model directories. Please pass the exact one: "
        f"{candidate_text}"
    )


def prepare_output_database_path(args: Args) -> Path:
    source_path = Path(args.database_path).resolve()
    output_path = (
        source_path
        if args.output_database_path is None
        else Path(args.output_database_path).resolve()
    )

    if output_path == source_path:
        if not source_path.is_file():
            raise FileNotFoundError(
                f"database_path does not exist: {source_path}"
            )
        return output_path

    if not source_path.is_file():
        raise FileNotFoundError(f"database_path does not exist: {source_path}")
    if output_path.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"output_database_path already exists: {output_path}. "
            "Pass --overwrite-output to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    return output_path


def load_database_images(
    connection: sqlite3.Connection,
) -> dict[str, DatabaseImageRow]:
    rows = connection.execute(
        "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
    ).fetchall()
    return {
        str(name): DatabaseImageRow(
            image_id=int(image_id),
            name=str(name),
            camera_id=int(camera_id),
        )
        for image_id, name, camera_id in rows
    }


def load_database_cameras(
    connection: sqlite3.Connection,
) -> dict[int, CameraRow]:
    rows = connection.execute(
        "SELECT camera_id, model, width, height, params, prior_focal_length "
        "FROM cameras ORDER BY camera_id"
    ).fetchall()
    return {
        int(camera_id): CameraRow(
            camera_id=int(camera_id),
            model=int(model),
            width=int(width),
            height=int(height),
            params=np.frombuffer(params, dtype=np.float64).copy(),
            prior_focal_length=bool(prior_focal_length),
        )
        for camera_id, model, width, height, params, prior_focal_length in rows
    }


def load_camera_image_references(
    connection: sqlite3.Connection,
) -> dict[int, set[int]]:
    rows = connection.execute(
        "SELECT camera_id, image_id FROM images ORDER BY camera_id, image_id"
    ).fetchall()
    references: dict[int, set[int]] = {}
    for camera_id, image_id in rows:
        references.setdefault(int(camera_id), set()).add(int(image_id))
    return references


def float64_blob(array: np.ndarray, *, column_major: bool) -> bytes:
    order = "F" if column_major else "C"
    return np.asarray(array, dtype=np.float64, order=order).tobytes(order=order)


def quote_sql_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def camera_matches_row(
    camera_row: CameraRow,
    *,
    model_id: int,
    width: int,
    height: int,
    params: np.ndarray,
    prior_focal_length: bool,
) -> bool:
    return (
        camera_row.model == model_id
        and camera_row.width == width
        and camera_row.height == height
        and camera_row.prior_focal_length == prior_focal_length
        and camera_row.params.shape == params.shape
        and np.allclose(camera_row.params, params)
    )


def make_projection_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation = qvec2rotmat(qvec)
    return (-rotation.T @ tvec).astype(np.float64)


def make_gravity_vector(qvec: np.ndarray) -> np.ndarray:
    rotation = qvec2rotmat(qvec)
    gravity = rotation[1, :].astype(np.float64)
    gravity_norm = np.linalg.norm(gravity)
    if gravity_norm == 0:
        raise ValueError("Computed zero gravity vector from qvec")
    return gravity / gravity_norm


def build_matches(
    *,
    model_images: dict[int, object],
    database_images_by_name: dict[str, DatabaseImageRow],
    image_path: Path,
    validate_image_files: bool,
    fail_on_missing_images: bool,
) -> tuple[list[MatchRecord], list[str], list[str]]:
    matches: list[MatchRecord] = []
    unmatched_model_images: list[str] = []
    missing_image_files: list[str] = []

    for model_image in model_images.values():
        image_name = str(model_image.name)
        database_image = database_images_by_name.get(image_name)
        if database_image is None:
            unmatched_model_images.append(image_name)
            continue

        image_file = image_path / image_name
        if validate_image_files and not image_file.is_file():
            message = f"Missing image file: {image_file}"
            if fail_on_missing_images:
                raise FileNotFoundError(message)
            print(f"skip {image_name}: image file not found")
            missing_image_files.append(image_name)
            continue

        qvec = np.asarray(model_image.qvec, dtype=np.float64)
        tvec = np.asarray(model_image.tvec, dtype=np.float64)
        matches.append(
            MatchRecord(
                image_id=database_image.image_id,
                image_name=image_name,
                database_camera_id=database_image.camera_id,
                source_model_image_id=int(model_image.id),
                source_model_camera_id=int(model_image.camera_id),
                qvec=qvec,
                tvec=tvec,
                position=make_projection_center(qvec, tvec),
                gravity=make_gravity_vector(qvec),
            )
        )

    return matches, unmatched_model_images, missing_image_files


def plan_camera_assignments(
    *,
    connection: sqlite3.Connection,
    model_cameras: dict[int, object],
    matches: list[MatchRecord],
    prior_focal_length: bool,
) -> tuple[dict[int, int], int, int]:
    existing_cameras = load_database_cameras(connection)
    matched_image_ids_by_model_camera: dict[int, set[int]] = {}
    candidate_db_camera_ids_by_model_camera: dict[int, set[int]] = {}
    for match in matches:
        matched_image_ids_by_model_camera.setdefault(
            match.source_model_camera_id, set()
        ).add(match.image_id)
        candidate_db_camera_ids_by_model_camera.setdefault(
            match.source_model_camera_id, set()
        ).add(match.database_camera_id)

    target_ids: dict[int, int] = {}
    updated = 0
    reused = 0

    for model_camera_id in sorted(matched_image_ids_by_model_camera):
        model_camera = model_cameras[model_camera_id]
        camera_model = CAMERA_MODEL_NAMES[str(model_camera.model)]
        params = np.asarray(model_camera.params, dtype=np.float64)
        candidate_db_camera_ids = candidate_db_camera_ids_by_model_camera[
            model_camera_id
        ]
        if len(candidate_db_camera_ids) != 1:
            raise ValueError(
                "Each sparse-model camera must map to exactly one "
                "existing database camera. "
                f"model camera {model_camera_id} maps to database "
                f"cameras {sorted(candidate_db_camera_ids)}"
            )

        target_id = next(iter(candidate_db_camera_ids))
        existing_row = existing_cameras.get(target_id)
        if existing_row is None:
            raise ValueError(
                "Matched images refer to missing database "
                f"camera_id={target_id}"
            )

        if camera_matches_row(
            existing_row,
            model_id=camera_model.model_id,
            width=int(model_camera.width),
            height=int(model_camera.height),
            params=params,
            prior_focal_length=prior_focal_length,
        ):
            target_ids[model_camera_id] = target_id
            reused += 1
            continue
        connection.execute(
            (
                "UPDATE cameras SET model = ?, width = ?, height = ?, "
                "params = ?, prior_focal_length = ? WHERE camera_id = ?"
            ),
            (
                camera_model.model_id,
                int(model_camera.width),
                int(model_camera.height),
                float64_blob(params, column_major=False),
                int(prior_focal_length),
                target_id,
            ),
        )
        existing_cameras[target_id] = CameraRow(
            camera_id=target_id,
            model=camera_model.model_id,
            width=int(model_camera.width),
            height=int(model_camera.height),
            params=params.copy(),
            prior_focal_length=prior_focal_length,
        )
        target_ids[model_camera_id] = target_id
        updated += 1

    return target_ids, updated, reused


def create_6dof_pose_table(connection: sqlite3.Connection) -> None:
    table_name = quote_sql_identifier(SIX_DOF_POSE_PRIOR_TABLE)
    connection.executescript(
        f"CREATE TABLE IF NOT EXISTS {table_name} ("
        "  image_id INTEGER PRIMARY KEY NOT NULL,"
        "  image_name TEXT NOT NULL,"
        "  camera_id INTEGER NOT NULL,"
        "  source_model_image_id INTEGER NOT NULL,"
        "  source_model_camera_id INTEGER NOT NULL,"
        "  camera_model INTEGER NOT NULL,"
        "  camera_model_name TEXT NOT NULL,"
        "  width INTEGER NOT NULL,"
        "  height INTEGER NOT NULL,"
        "  camera_params BLOB NOT NULL,"
        "  qvec BLOB NOT NULL,"
        "  tvec BLOB NOT NULL,"
        "  position BLOB NOT NULL,"
        "  rotation_covariance BLOB NOT NULL,"
        "  position_covariance BLOB NOT NULL,"
        "  gravity BLOB,"
        "  coordinate_system INTEGER NOT NULL,"
        "  source_model_path TEXT NOT NULL,"
        "  imported_at_utc TEXT NOT NULL"
        ");"
    )


def drop_legacy_6dof_pose_table(connection: sqlite3.Connection) -> None:
    table_name = quote_sql_identifier(LEGACY_SIX_DOF_POSE_PRIOR_TABLE)
    connection.execute(
        f"DROP TABLE IF EXISTS {table_name}"
    )


def upsert_pose_prior(
    connection: sqlite3.Connection,
    *,
    image_id: int,
    camera_id: int,
    position: np.ndarray,
    position_covariance: np.ndarray,
    coordinate_system: int,
    gravity: np.ndarray,
) -> bool:
    existing_row = connection.execute(
        (
            "SELECT pose_prior_id FROM pose_priors WHERE corr_data_id = ? "
            "AND corr_sensor_id = ? AND corr_sensor_type = ?"
        ),
        (image_id, camera_id, CAMERA_SENSOR_TYPE),
    ).fetchone()

    if existing_row is None:
        connection.execute(
            (
                "INSERT INTO pose_priors (corr_data_id, corr_sensor_id, "
                "corr_sensor_type, position, position_covariance, "
                "coordinate_system, gravity) VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                image_id,
                camera_id,
                CAMERA_SENSOR_TYPE,
                float64_blob(position, column_major=False),
                float64_blob(position_covariance, column_major=True),
                coordinate_system,
                float64_blob(gravity, column_major=False),
            ),
        )
        return True

    connection.execute(
        (
            "UPDATE pose_priors SET corr_data_id = ?, corr_sensor_id = ?, "
            "corr_sensor_type = ?, position = ?, position_covariance = ?, "
            "coordinate_system = ?, gravity = ? WHERE pose_prior_id = ?"
        ),
        (
            image_id,
            camera_id,
            CAMERA_SENSOR_TYPE,
            float64_blob(position, column_major=False),
            float64_blob(position_covariance, column_major=True),
            coordinate_system,
            float64_blob(gravity, column_major=False),
            int(existing_row[0]),
        ),
    )
    return False


def upsert_6dof_pose_row(
    connection: sqlite3.Connection,
    *,
    match: MatchRecord,
    model_camera: object,
    target_camera_id: int,
    rotation_covariance: np.ndarray,
    position_covariance: np.ndarray,
    gravity: np.ndarray,
    coordinate_system: int,
    source_model_path: Path,
) -> None:
    camera_model = CAMERA_MODEL_NAMES[str(model_camera.model)]
    table_name = quote_sql_identifier(SIX_DOF_POSE_PRIOR_TABLE)
    connection.execute(
        f"INSERT INTO {table_name} ("
        "image_id, image_name, camera_id, source_model_image_id, "
        "source_model_camera_id, "
        "camera_model, camera_model_name, width, height, camera_params, "
        "qvec, tvec, position, rotation_covariance, position_covariance, "
        "gravity, coordinate_system, "
        "source_model_path, imported_at_utc"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(image_id) DO UPDATE SET "
        "image_name = excluded.image_name, "
        "camera_id = excluded.camera_id, "
        "source_model_image_id = excluded.source_model_image_id, "
        "source_model_camera_id = excluded.source_model_camera_id, "
        "camera_model = excluded.camera_model, "
        "camera_model_name = excluded.camera_model_name, "
        "width = excluded.width, "
        "height = excluded.height, "
        "camera_params = excluded.camera_params, "
        "qvec = excluded.qvec, "
        "tvec = excluded.tvec, "
        "position = excluded.position, "
        "rotation_covariance = excluded.rotation_covariance, "
        "position_covariance = excluded.position_covariance, "
        "gravity = excluded.gravity, "
        "coordinate_system = excluded.coordinate_system, "
        "source_model_path = excluded.source_model_path, "
        "imported_at_utc = excluded.imported_at_utc",
        (
            match.image_id,
            match.image_name,
            target_camera_id,
            match.source_model_image_id,
            match.source_model_camera_id,
            camera_model.model_id,
            camera_model.model_name,
            int(model_camera.width),
            int(model_camera.height),
            float64_blob(
                np.asarray(model_camera.params, dtype=np.float64),
                column_major=False,
            ),
            float64_blob(match.qvec, column_major=False),
            float64_blob(match.tvec, column_major=False),
            float64_blob(match.position, column_major=False),
            float64_blob(rotation_covariance, column_major=True),
            float64_blob(position_covariance, column_major=True),
            float64_blob(gravity, column_major=False),
            coordinate_system,
            str(source_model_path),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def import_lidar_priors(
    args: Args,
) -> tuple[Path, Path, ImportStats, list[str], list[str]]:
    database_path = prepare_output_database_path(args)
    sparse_model_path = resolve_sparse_model_path(Path(args.sparse_model_path))
    image_path = Path(args.image_path).resolve()

    if not image_path.is_dir():
        raise NotADirectoryError(f"image_path does not exist: {image_path}")
    if args.position_std <= 0:
        raise ValueError("position_std must be > 0")
    if args.rotation_std_deg <= 0:
        raise ValueError("rotation_std_deg must be > 0")

    model_cameras, model_images, _ = read_model(str(sparse_model_path))
    coordinate_system = COORDINATE_SYSTEM_TO_CODE[args.coordinate_system]
    position_covariance = np.eye(3, dtype=np.float64) * (args.position_std**2)
    rotation_std_rad = np.deg2rad(args.rotation_std_deg)
    rotation_covariance = np.eye(3, dtype=np.float64) * (rotation_std_rad**2)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        database_images_by_name = load_database_images(connection)
        matches, unmatched_model_images, missing_image_files = build_matches(
            model_images=model_images,
            database_images_by_name=database_images_by_name,
            image_path=image_path,
            validate_image_files=args.validate_image_files,
            fail_on_missing_images=args.fail_on_missing_images,
        )

        if not matches:
            raise ValueError(
                "No sparse-model images could be matched to database "
                "images by name"
            )

        target_camera_ids, updated_cameras, reused_cameras = (
            plan_camera_assignments(
                connection=connection,
                model_cameras=model_cameras,
                matches=matches,
                prior_focal_length=args.prior_focal_length,
            )
        )

        updated_image_camera_links = 0
        inserted_pose_priors = 0
        updated_pose_priors = 0
        upserted_6dof_pose_rows = 0

        if args.clear_existing_pose_priors:
            connection.execute("DELETE FROM pose_priors")

        if args.create_6dof_pose_table:
            create_6dof_pose_table(connection)
            if args.drop_legacy_lidar_pose_table:
                drop_legacy_6dof_pose_table(connection)

        for match in matches:
            target_camera_id = target_camera_ids[match.source_model_camera_id]
            db_camera_id = database_images_by_name[match.image_name].camera_id
            if db_camera_id != target_camera_id:
                connection.execute(
                    "UPDATE images SET camera_id = ? WHERE image_id = ?",
                    (target_camera_id, match.image_id),
                )
                updated_image_camera_links += 1

            inserted = upsert_pose_prior(
                connection,
                image_id=match.image_id,
                camera_id=target_camera_id,
                position=match.position,
                position_covariance=position_covariance,
                coordinate_system=coordinate_system,
                gravity=match.gravity,
            )
            if inserted:
                inserted_pose_priors += 1
            else:
                updated_pose_priors += 1

            if args.create_6dof_pose_table:
                upsert_6dof_pose_row(
                    connection,
                    match=match,
                    model_camera=model_cameras[match.source_model_camera_id],
                    target_camera_id=target_camera_id,
                    rotation_covariance=rotation_covariance,
                    position_covariance=position_covariance,
                    gravity=match.gravity,
                    coordinate_system=coordinate_system,
                    source_model_path=sparse_model_path,
                )
                upserted_6dof_pose_rows += 1

        connection.commit()

    stats = ImportStats(
        matched_images=len(matches),
        skipped_model_images=len(unmatched_model_images),
        skipped_missing_files=len(missing_image_files),
        updated_cameras=updated_cameras,
        reused_cameras=reused_cameras,
        updated_image_camera_links=updated_image_camera_links,
        inserted_pose_priors=inserted_pose_priors,
        updated_pose_priors=updated_pose_priors,
        upserted_6dof_pose_rows=upserted_6dof_pose_rows,
    )
    return (
        database_path,
        sparse_model_path,
        stats,
        unmatched_model_images,
        missing_image_files,
    )


def main(args: Args) -> int:
    (
        database_path,
        sparse_model_path,
        stats,
        unmatched_model_images,
        missing_image_files,
    ) = import_lidar_priors(args)

    print("Imported sparse_lidar 6DoF priors into COLMAP database")
    print(f"  database:                {database_path}")
    print(f"  sparse_model:            {sparse_model_path}")
    print(f"  matched_images:          {stats.matched_images}")
    print(f"  skipped_model_images:    {stats.skipped_model_images}")
    print(f"  skipped_missing_files:   {stats.skipped_missing_files}")
    print(f"  updated_cameras:         {stats.updated_cameras}")
    print(f"  reused_cameras:          {stats.reused_cameras}")
    print(f"  updated_image_links:     {stats.updated_image_camera_links}")
    print(f"  inserted_pose_priors:    {stats.inserted_pose_priors}")
    print(f"  updated_pose_priors:     {stats.updated_pose_priors}")
    print(f"  6dof_pose_table:        {SIX_DOF_POSE_PRIOR_TABLE}")
    print(f"  6dof_pose_rows:         {stats.upserted_6dof_pose_rows}")

    if unmatched_model_images:
        print()
        print("First unmatched sparse-model image names:")
        for image_name in unmatched_model_images[:10]:
            print(f"  {image_name}")

    if missing_image_files:
        print()
        print("First sparse-model names missing on disk:")
        for image_name in missing_image_files[:10]:
            print(f"  {image_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
