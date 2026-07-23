"""Filesystem safety and frozen storage ceilings for the U2 cohort."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIB_BYTES = 1024**3
LINEAGE_CAP_BYTES = 2 * GIB_BYTES
COHORT_SCIENTIFIC_CAP_BYTES = 6 * GIB_BYTES
MEDIA_CAP_BYTES = 10 * GIB_BYTES
COMBINED_PLANNED_CAP_BYTES = 16 * GIB_BYTES


class UnsafeStoragePathError(ValueError):
    """Raised when a managed path could escape or alias protected storage."""


class StorageCapExceededError(OSError):
    """Raised when a scientific or media directory exceeds its frozen cap."""


def format_bytes(byte_count: int) -> str:
    """Return a stable human-readable binary byte count."""

    if byte_count < 0:
        raise ValueError("byte count cannot be negative")
    if byte_count < 1024:
        return f"{byte_count} B"

    value = float(byte_count)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class DirectoryUsage:
    """Measured logical usage and the cap governing one managed directory."""

    path: Path
    label: str
    used_bytes: int
    cap_bytes: int
    exists: bool

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("usage label cannot be empty")
        if self.used_bytes < 0:
            raise ValueError("used bytes cannot be negative")
        if self.cap_bytes <= 0:
            raise ValueError("cap bytes must be positive")

    @property
    def within_cap(self) -> bool:
        return self.used_bytes <= self.cap_bytes

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.cap_bytes - self.used_bytes)

    @property
    def overage_bytes(self) -> int:
        return max(0, self.used_bytes - self.cap_bytes)

    @property
    def percent_used(self) -> float:
        return 100.0 * self.used_bytes / self.cap_bytes

    def readable(self) -> str:
        state = (
            f"{format_bytes(self.remaining_bytes)} remaining"
            if self.within_cap
            else f"{format_bytes(self.overage_bytes)} over cap"
        )
        existence = "" if self.exists else " [not created]"
        return (
            f"{self.label}: {format_bytes(self.used_bytes)} used of "
            f"{format_bytes(self.cap_bytes)} cap ({state}) at {self.path}{existence}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return dashboard-safe primitive values without losing exact bytes."""

        return {
            "path": str(self.path),
            "label": self.label,
            "exists": self.exists,
            "used_bytes": self.used_bytes,
            "cap_bytes": self.cap_bytes,
            "remaining_bytes": self.remaining_bytes,
            "overage_bytes": self.overage_bytes,
            "within_cap": self.within_cap,
            "percent_used": self.percent_used,
            "readable": self.readable(),
        }


@dataclass(frozen=True)
class U2StorageReport:
    """Complete scientific, lineage, media, and planned-footprint report."""

    lineage: DirectoryUsage
    cohort: DirectoryUsage
    media: DirectoryUsage | None
    anticipated_lineage_bytes: int = 0
    anticipated_media_bytes: int = 0

    def __post_init__(self) -> None:
        if self.anticipated_lineage_bytes < 0:
            raise ValueError("anticipated lineage growth cannot be negative")
        if self.anticipated_media_bytes < 0:
            raise ValueError("anticipated media growth cannot be negative")

    @property
    def combined_used_bytes(self) -> int:
        media_bytes = self.media.used_bytes if self.media is not None else 0
        return self.cohort.used_bytes + media_bytes

    @property
    def projected_lineage_used_bytes(self) -> int:
        return self.lineage.used_bytes + self.anticipated_lineage_bytes

    @property
    def projected_cohort_used_bytes(self) -> int:
        return self.cohort.used_bytes + self.anticipated_lineage_bytes

    @property
    def projected_media_used_bytes(self) -> int:
        current = self.media.used_bytes if self.media is not None else 0
        return current + self.anticipated_media_bytes

    @property
    def projected_combined_used_bytes(self) -> int:
        return self.projected_cohort_used_bytes + self.projected_media_used_bytes

    @property
    def combined_remaining_bytes(self) -> int:
        return max(0, COMBINED_PLANNED_CAP_BYTES - self.combined_used_bytes)

    @property
    def within_combined_cap(self) -> bool:
        return self.combined_used_bytes <= COMBINED_PLANNED_CAP_BYTES

    @property
    def projected_within_caps(self) -> bool:
        media_within = (
            self.projected_media_used_bytes <= MEDIA_CAP_BYTES
            if self.media is not None or self.anticipated_media_bytes
            else True
        )
        return (
            self.projected_lineage_used_bytes <= LINEAGE_CAP_BYTES
            and self.projected_cohort_used_bytes <= COHORT_SCIENTIFIC_CAP_BYTES
            and media_within
            and self.projected_combined_used_bytes <= COMBINED_PLANNED_CAP_BYTES
        )

    def readable(self) -> str:
        lines = [self.lineage.readable(), self.cohort.readable()]
        if self.media is not None:
            lines.append(self.media.readable())
        combined_state = (
            f"{format_bytes(self.combined_remaining_bytes)} remaining"
            if self.within_combined_cap
            else f"{format_bytes(self.combined_used_bytes - COMBINED_PLANNED_CAP_BYTES)} over cap"
        )
        lines.append(
            "Combined planned footprint: "
            f"{format_bytes(self.combined_used_bytes)} used of "
            f"{format_bytes(COMBINED_PLANNED_CAP_BYTES)} cap ({combined_state})"
        )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lineage": self.lineage.as_dict(),
            "cohort": self.cohort.as_dict(),
            "media": self.media.as_dict() if self.media is not None else None,
            "combined_used_bytes": self.combined_used_bytes,
            "combined_cap_bytes": COMBINED_PLANNED_CAP_BYTES,
            "combined_remaining_bytes": self.combined_remaining_bytes,
            "within_combined_cap": self.within_combined_cap,
            "anticipated_lineage_bytes": self.anticipated_lineage_bytes,
            "anticipated_media_bytes": self.anticipated_media_bytes,
            "projected_lineage_used_bytes": self.projected_lineage_used_bytes,
            "projected_cohort_used_bytes": self.projected_cohort_used_bytes,
            "projected_media_used_bytes": self.projected_media_used_bytes,
            "projected_combined_used_bytes": self.projected_combined_used_bytes,
            "projected_within_caps": self.projected_within_caps,
            "readable": self.readable(),
        }


def ensure_directory_growth(
    usage: DirectoryUsage,
    anticipated_bytes: int,
) -> None:
    """Reject a known-size publication before it can cross a directory cap."""

    if anticipated_bytes < 0:
        raise ValueError("anticipated growth cannot be negative")
    projected = usage.used_bytes + anticipated_bytes
    if projected > usage.cap_bytes:
        overage = projected - usage.cap_bytes
        raise StorageCapExceededError(
            f"{usage.label}: publishing {format_bytes(anticipated_bytes)} would use "
            f"{format_bytes(projected)} of {format_bytes(usage.cap_bytes)} "
            f"({format_bytes(overage)} over cap)"
        )


def _absolute_path(path: Path, *, relative_to: Path | None = None) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = (relative_to if relative_to is not None else Path.cwd()) / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _overlaps(first: Path, second: Path) -> bool:
    return _contains(first, second) or _contains(second, first)


def _reject_symlink_components(root: Path, relative_path: Path) -> None:
    current = root
    components = relative_path.parts
    for index, component in enumerate(components):
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise UnsafeStoragePathError(f"storage path contains a symlink: {current}")
        if index < len(components) - 1 and not stat.S_ISDIR(mode):
            raise NotADirectoryError(f"storage path component is not a directory: {current}")


def _reject_all_symlink_components(path: Path) -> None:
    anchor = Path(path.anchor)
    relative = Path(*path.parts[1:])
    _reject_symlink_components(anchor, relative)


def validate_directory_path(
    path: Path,
    *,
    root: Path,
    protected_paths: Iterable[Path] = (),
    allow_missing: bool = False,
    allow_root: bool = False,
) -> Path:
    """Validate a managed directory without following aliases or path escapes.

    Relative targets and protected paths are interpreted beneath ``root``. A target
    may not overlap a protected path in either direction, which keeps both a
    protected artifact and its containing directory out of cleanup scope.
    """

    root_path = _absolute_path(root)
    try:
        root_mode = root_path.lstat().st_mode
    except FileNotFoundError as error:
        raise FileNotFoundError(f"storage root does not exist: {root_path}") from error
    if stat.S_ISLNK(root_mode):
        raise UnsafeStoragePathError(f"storage root cannot be a symlink: {root_path}")
    if not stat.S_ISDIR(root_mode):
        raise NotADirectoryError(f"storage root is not a directory: {root_path}")

    target = _absolute_path(path, relative_to=root_path)
    try:
        relative_target = target.relative_to(root_path)
    except ValueError as error:
        raise UnsafeStoragePathError(
            f"storage directory escapes managed root {root_path}: {target}"
        ) from error
    if relative_target == Path(".") and not allow_root:
        raise UnsafeStoragePathError(f"storage directory cannot be the managed root: {root_path}")

    _reject_symlink_components(root_path, relative_target)
    resolved_root = root_path.resolve(strict=True)
    resolved_target = target.resolve(strict=False)
    if not _contains(resolved_root, resolved_target):
        raise UnsafeStoragePathError(
            f"resolved storage directory escapes managed root {resolved_root}: {resolved_target}"
        )

    try:
        target_mode = target.lstat().st_mode
    except FileNotFoundError:
        if not allow_missing:
            raise FileNotFoundError(f"storage directory does not exist: {target}") from None
    else:
        if stat.S_ISLNK(target_mode):
            raise UnsafeStoragePathError(f"storage directory cannot be a symlink: {target}")
        if not stat.S_ISDIR(target_mode):
            raise NotADirectoryError(f"storage path is not a directory: {target}")

    target_forms = (target, resolved_target)
    for protected_path in protected_paths:
        protected = _absolute_path(protected_path, relative_to=root_path)
        resolved_protected = protected.resolve(strict=False)
        if any(
            _overlaps(target_form, protected_form)
            for target_form in target_forms
            for protected_form in (protected, resolved_protected)
        ):
            raise UnsafeStoragePathError(
                f"storage directory overlaps protected path {protected}: {target}"
            )
    return resolved_target


def _directory_size_bytes(path: Path) -> int:
    total = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise UnsafeStoragePathError(
                        f"managed directory contains a symlink: {entry_path}"
                    )
                if stat.S_ISDIR(mode):
                    pending.append(entry_path)
                elif stat.S_ISREG(mode):
                    total += entry.stat(follow_symlinks=False).st_size
                else:
                    raise UnsafeStoragePathError(
                        f"managed directory contains a special filesystem entry: {entry_path}"
                    )
    return total


def measure_directory_usage(
    path: Path,
    *,
    cap_bytes: int,
    label: str,
    allow_missing: bool = False,
) -> DirectoryUsage:
    """Measure logical file bytes while refusing every symlink encountered."""

    if cap_bytes <= 0:
        raise ValueError("cap bytes must be positive")
    target = _absolute_path(path)
    _reject_all_symlink_components(target)
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        if not allow_missing:
            raise FileNotFoundError(f"storage directory does not exist: {target}") from None
        return DirectoryUsage(
            path=target,
            label=label,
            used_bytes=0,
            cap_bytes=cap_bytes,
            exists=False,
        )
    if stat.S_ISLNK(mode):
        raise UnsafeStoragePathError(f"storage directory cannot be a symlink: {target}")
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError(f"storage path is not a directory: {target}")
    return DirectoryUsage(
        path=target,
        label=label,
        used_bytes=_directory_size_bytes(target),
        cap_bytes=cap_bytes,
        exists=True,
    )


def enforce_directory_cap(
    path: Path,
    *,
    cap_bytes: int,
    label: str,
    allow_missing: bool = False,
) -> DirectoryUsage:
    """Return current usage or raise before a directory is allowed to grow."""

    usage = measure_directory_usage(
        path,
        cap_bytes=cap_bytes,
        label=label,
        allow_missing=allow_missing,
    )
    if not usage.within_cap:
        raise StorageCapExceededError(usage.readable())
    return usage


def audit_u2_storage(
    *,
    storage_root: Path,
    cohort_directory: Path,
    lineage_directory: Path,
    media_directory: Path | None = None,
    protected_paths: Iterable[Path] = (),
    allow_missing: bool = False,
    anticipated_lineage_bytes: int = 0,
    anticipated_media_bytes: int = 0,
) -> U2StorageReport:
    """Validate placement and admit known growth against every frozen cap."""

    if anticipated_lineage_bytes < 0 or anticipated_media_bytes < 0:
        raise ValueError("anticipated U2 storage growth cannot be negative")
    if anticipated_media_bytes and media_directory is None:
        raise ValueError("anticipated media growth requires a managed media directory")

    protected = tuple(protected_paths)
    cohort_path = validate_directory_path(
        cohort_directory,
        root=storage_root,
        protected_paths=protected,
        allow_missing=allow_missing,
    )
    lineage_path = validate_directory_path(
        lineage_directory,
        root=storage_root,
        protected_paths=protected,
        allow_missing=allow_missing,
    )
    if lineage_path == cohort_path or not _contains(cohort_path, lineage_path):
        raise UnsafeStoragePathError(
            f"lineage directory must be a child of cohort directory {cohort_path}: {lineage_path}"
        )

    media_path: Path | None = None
    if media_directory is not None:
        media_path = validate_directory_path(
            media_directory,
            root=storage_root,
            protected_paths=protected,
            allow_missing=allow_missing,
        )
        if _overlaps(cohort_path, media_path):
            raise UnsafeStoragePathError(
                f"media directory must be separate from scientific cohort directory: {media_path}"
            )

    lineage = enforce_directory_cap(
        lineage_path,
        cap_bytes=LINEAGE_CAP_BYTES,
        label="U2 lineage scientific artifacts",
        allow_missing=allow_missing,
    )
    cohort = enforce_directory_cap(
        cohort_path,
        cap_bytes=COHORT_SCIENTIFIC_CAP_BYTES,
        label="U2 cohort scientific artifacts",
        allow_missing=allow_missing,
    )
    media = (
        enforce_directory_cap(
            media_path,
            cap_bytes=MEDIA_CAP_BYTES,
            label="U2 narrative media",
            allow_missing=allow_missing,
        )
        if media_path is not None
        else None
    )
    ensure_directory_growth(lineage, anticipated_lineage_bytes)
    ensure_directory_growth(cohort, anticipated_lineage_bytes)
    if media is not None:
        ensure_directory_growth(media, anticipated_media_bytes)
    report = U2StorageReport(
        lineage=lineage,
        cohort=cohort,
        media=media,
        anticipated_lineage_bytes=anticipated_lineage_bytes,
        anticipated_media_bytes=anticipated_media_bytes,
    )
    if not report.within_combined_cap:
        raise StorageCapExceededError(report.readable())
    if report.projected_combined_used_bytes > COMBINED_PLANNED_CAP_BYTES:
        overage = report.projected_combined_used_bytes - COMBINED_PLANNED_CAP_BYTES
        raise StorageCapExceededError(
            "Combined planned footprint: publishing "
            f"{format_bytes(anticipated_lineage_bytes + anticipated_media_bytes)} "
            f"would use {format_bytes(report.projected_combined_used_bytes)} of "
            f"{format_bytes(COMBINED_PLANNED_CAP_BYTES)} "
            f"({format_bytes(overage)} over cap)"
        )
    return report
