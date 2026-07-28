from pathlib import Path

import pytest

from dungeon_apprentice.u2_storage import (
    COHORT_SCIENTIFIC_CAP_BYTES,
    COMBINED_PLANNED_CAP_BYTES,
    GIB_BYTES,
    LINEAGE_CAP_BYTES,
    MEDIA_CAP_BYTES,
    DirectoryUsage,
    StorageCapExceededError,
    UnsafeStoragePathError,
    audit_u2_storage,
    enforce_directory_cap,
    ensure_directory_growth,
    format_bytes,
    measure_directory_usage,
    validate_directory_path,
)


def test_u2_storage_constants_are_frozen() -> None:
    assert LINEAGE_CAP_BYTES == 2 * 1024**3
    assert COHORT_SCIENTIFIC_CAP_BYTES == 6 * 1024**3
    assert MEDIA_CAP_BYTES == 10 * 1024**3
    assert COMBINED_PLANNED_CAP_BYTES == 16 * 1024**3
    assert COMBINED_PLANNED_CAP_BYTES == COHORT_SCIENTIFIC_CAP_BYTES + MEDIA_CAP_BYTES
    assert GIB_BYTES == 1024**3


def test_directory_cap_accepts_boundary_and_rejects_cap_plus_one(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abcd")

    usage = enforce_directory_cap(tmp_path, cap_bytes=4, label="test artifacts")
    assert usage.used_bytes == 4
    assert usage.within_cap
    assert usage.remaining_bytes == 0

    payload.write_bytes(b"abcde")
    with pytest.raises(StorageCapExceededError, match="1 B over cap"):
        enforce_directory_cap(tmp_path, cap_bytes=4, label="test artifacts")


def test_known_growth_is_admitted_before_publication(tmp_path: Path) -> None:
    usage = DirectoryUsage(
        path=tmp_path,
        label="lineage",
        used_bytes=3,
        cap_bytes=5,
        exists=True,
    )

    ensure_directory_growth(usage, 2)
    with pytest.raises(
        StorageCapExceededError,
        match=r"publishing 3 B.*1 B over cap",
    ):
        ensure_directory_growth(usage, 3)
    with pytest.raises(ValueError, match="cannot be negative"):
        ensure_directory_growth(usage, -1)


def test_directory_usage_rejects_nested_symlink(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_directory / "alias").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeStoragePathError, match="contains a symlink"):
        measure_directory_usage(run_directory, cap_bytes=100, label="run")


def test_path_validation_rejects_escape_and_symlink_component(tmp_path: Path) -> None:
    root = tmp_path / "storage"
    root.mkdir()

    with pytest.raises(UnsafeStoragePathError, match="escapes managed root"):
        validate_directory_path(root / ".." / "escape", root=root, allow_missing=True)

    real_directory = root / "real"
    real_directory.mkdir()
    alias = root / "alias"
    alias.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(UnsafeStoragePathError, match="contains a symlink"):
        validate_directory_path(alias / "child", root=root, allow_missing=True)


@pytest.mark.parametrize(
    ("target_suffix", "protected_suffix"),
    [
        ("parents", "parents/archive.zip"),
        ("parents/new-run", "parents"),
    ],
)
def test_path_validation_rejects_protected_path_overlap(
    tmp_path: Path, target_suffix: str, protected_suffix: str
) -> None:
    root = tmp_path / "storage"
    root.mkdir()
    protected = root / protected_suffix

    with pytest.raises(UnsafeStoragePathError, match="overlaps protected path"):
        validate_directory_path(
            root / target_suffix,
            root=root,
            protected_paths=[protected],
            allow_missing=True,
        )


def test_missing_directory_requires_explicit_opt_in(tmp_path: Path) -> None:
    missing = tmp_path / "future-run"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        measure_directory_usage(missing, cap_bytes=1024, label="future")

    usage = measure_directory_usage(
        missing,
        cap_bytes=1024,
        label="future",
        allow_missing=True,
    )
    assert not usage.exists
    assert usage.used_bytes == 0
    assert usage.remaining_bytes == 1024
    assert "[not created]" in usage.readable()


def test_usage_has_readable_and_dashboard_safe_reporting(tmp_path: Path) -> None:
    (tmp_path / "payload.bin").write_bytes(b"x" * 512)
    usage = measure_directory_usage(tmp_path, cap_bytes=1024, label="lineage")

    assert usage.readable() == (
        f"lineage: 512 B used of 1.00 KiB cap (512 B remaining) at {tmp_path}"
    )
    assert usage.as_dict() == {
        "path": str(tmp_path),
        "label": "lineage",
        "exists": True,
        "used_bytes": 512,
        "cap_bytes": 1024,
        "remaining_bytes": 512,
        "overage_bytes": 0,
        "within_cap": True,
        "percent_used": 50.0,
        "readable": usage.readable(),
    }
    assert format_bytes(2 * GIB_BYTES) == "2.00 GiB"


def test_u2_audit_reports_separate_scientific_and_media_usage(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    cohort = storage_root / "u2-cohort"
    lineage = cohort / "lineage-1"
    media = storage_root / "u2-media"
    lineage.mkdir(parents=True)
    media.mkdir()
    (lineage / "checkpoint.zip").write_bytes(b"model")
    (media / "capture.mp4").write_bytes(b"video")

    report = audit_u2_storage(
        storage_root=storage_root,
        cohort_directory=cohort,
        lineage_directory=lineage,
        media_directory=media,
    )

    assert report.lineage.used_bytes == 5
    assert report.cohort.used_bytes == 5
    assert report.media is not None
    assert report.media.used_bytes == 5
    assert report.combined_used_bytes == 10
    assert report.projected_combined_used_bytes == 10
    assert report.projected_within_caps
    assert report.as_dict()["combined_cap_bytes"] == 16 * GIB_BYTES
    assert "Combined planned footprint: 10 B used" in report.readable()

    admitted = audit_u2_storage(
        storage_root=storage_root,
        cohort_directory=cohort,
        lineage_directory=lineage,
        media_directory=media,
        anticipated_lineage_bytes=7,
        anticipated_media_bytes=11,
    )
    assert admitted.projected_lineage_used_bytes == 12
    assert admitted.projected_cohort_used_bytes == 12
    assert admitted.projected_media_used_bytes == 16
    assert admitted.projected_combined_used_bytes == 28
    assert admitted.as_dict()["anticipated_lineage_bytes"] == 7


def test_u2_audit_rejects_unmanaged_or_negative_anticipated_growth(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage"
    cohort = storage_root / "cohort"
    lineage = cohort / "lineage"
    lineage.mkdir(parents=True)

    with pytest.raises(ValueError, match="managed media"):
        audit_u2_storage(
            storage_root=storage_root,
            cohort_directory=cohort,
            lineage_directory=lineage,
            anticipated_media_bytes=1,
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        audit_u2_storage(
            storage_root=storage_root,
            cohort_directory=cohort,
            lineage_directory=lineage,
            anticipated_lineage_bytes=-1,
        )


def test_u2_audit_allows_prospective_missing_directories(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    report = audit_u2_storage(
        storage_root=storage_root,
        cohort_directory="cohort",
        lineage_directory="cohort/lineage-1",
        media_directory="media",
        allow_missing=True,
    )

    assert not report.lineage.exists
    assert not report.cohort.exists
    assert report.media is not None
    assert not report.media.exists
    assert report.combined_used_bytes == 0


def test_u2_audit_rejects_lineage_or_media_misplacement(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    cohort = storage_root / "cohort"
    lineage = storage_root / "other-lineage"
    cohort.mkdir(parents=True)
    lineage.mkdir()

    with pytest.raises(UnsafeStoragePathError, match="child of cohort"):
        audit_u2_storage(
            storage_root=storage_root,
            cohort_directory=cohort,
            lineage_directory=lineage,
        )

    nested_media = cohort / "media"
    nested_media.mkdir()
    with pytest.raises(UnsafeStoragePathError, match="must be separate"):
        audit_u2_storage(
            storage_root=storage_root,
            cohort_directory=cohort,
            lineage_directory=nested_media,
            media_directory=nested_media,
        )
