"""External Git-tag anchor for the one-shot U2 qualification report."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ANCHOR_SCHEMA_VERSION = 1
ANCHOR_PROTOCOL = "dungeon-apprentice-v0.2-u2-qualification-anchor"
QUALIFICATION_PROTOCOL = "dungeon-apprentice-v0.2-u2"
ANCHOR_TAG = "u2-preflight-v0.2-u2-20260723"
ANCHOR_REMOTE = "origin"
EXPECTED_ORIGIN_URL = "https://github.com/PeteAndrews1289/dungeon-apprentice.git"
CANONICAL_REPORT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.2-u2-20260723/report.json"
)

ANCHOR_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "qualification_protocol",
        "tag",
        "remote",
        "remote_url",
        "source_commit",
        "report",
        "report_sha256",
        "report_byte_length",
        "attempt_id",
        "attempt_sha256",
        "claim_id",
        "claim_sha256",
        "generator_profile",
        "generator_profile_version",
    }
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40,64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9._:-]{7,127}$")


class U2QualificationAnchorError(RuntimeError):
    """Raised when the external qualification anchor is absent or inconsistent."""


def _read_bootstrap_json(path: Path, label: str) -> Mapping[str, Any]:
    """Read pre-anchor identity material; the qualifier verifier rereads and binds it."""

    if path.is_symlink() or not path.is_file():
        raise U2QualificationAnchorError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise U2QualificationAnchorError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        raise U2QualificationAnchorError(f"{label} is not a JSON object")
    return value


@dataclass(frozen=True)
class U2QualificationAnchor:
    tag: str
    tag_object: str
    remote: str
    remote_url: str
    source_commit: str
    report: str
    report_sha256: str
    report_byte_length: int
    attempt_id: str
    attempt_sha256: str
    claim_id: str
    claim_sha256: str
    generator_profile: str
    generator_profile_version: int

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise U2QualificationAnchorError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise U2QualificationAnchorError("anchor source commit is invalid")
    return value


def _require_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
        raise U2QualificationAnchorError(f"{label} is invalid")
    return value


def _public_evidence(evidence: Any) -> Mapping[str, Any]:
    if hasattr(evidence, "public_dict"):
        value = evidence.public_dict()
    elif isinstance(evidence, Mapping):
        value = evidence
    else:
        raise U2QualificationAnchorError("qualification evidence has no public mapping")
    if not isinstance(value, Mapping):
        raise U2QualificationAnchorError("qualification evidence is not a mapping")
    return value


def build_anchor_payload(
    evidence: Any,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Build the deterministic annotated-tag message from verified evidence."""

    public = _public_evidence(evidence)
    commit = _require_commit(source_commit)
    report = Path(str(public.get("report", ""))).expanduser().absolute()
    if report != CANONICAL_REPORT:
        raise U2QualificationAnchorError(
            f"qualification anchor requires canonical report {CANONICAL_REPORT}"
        )
    if public.get("source_commit") != commit:
        raise U2QualificationAnchorError(
            "qualification evidence and anchor source commits differ"
        )
    report_length = public.get("report_byte_length")
    profile_version = public.get("generator_profile_version")
    if (
        type(report_length) is not int
        or report_length <= 0
        or type(profile_version) is not int
        or profile_version <= 0
    ):
        raise U2QualificationAnchorError(
            "qualification evidence has invalid byte/profile counts"
        )
    profile = public.get("generator_profile")
    if not isinstance(profile, str) or not profile:
        raise U2QualificationAnchorError(
            "qualification evidence has no generator profile"
        )
    return {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "protocol": ANCHOR_PROTOCOL,
        "qualification_protocol": QUALIFICATION_PROTOCOL,
        "tag": ANCHOR_TAG,
        "remote": ANCHOR_REMOTE,
        "remote_url": EXPECTED_ORIGIN_URL,
        "source_commit": commit,
        "report": str(CANONICAL_REPORT),
        "report_sha256": _require_digest(
            public.get("report_sha256"), "qualification report"
        ),
        "report_byte_length": report_length,
        "attempt_id": _require_identity(
            public.get("attempt_id"), "qualification attempt ID"
        ),
        "attempt_sha256": _require_digest(
            public.get("attempt_sha256"), "qualification attempt"
        ),
        "claim_id": _require_digest(
            public.get("claim_id"), "qualification claim ID"
        ),
        "claim_sha256": _require_digest(
            public.get("claim_sha256"), "qualification claim"
        ),
        "generator_profile": profile,
        "generator_profile_version": profile_version,
    }


def anchor_message(evidence: Any, *, source_commit: str) -> str:
    """Return the exact one-line annotated-tag message."""

    return json.dumps(
        build_anchor_payload(evidence, source_commit=source_commit),
        sort_keys=True,
        separators=(",", ":"),
    )


def qualification_evidence_for_anchor(
    repository: Path,
    *,
    expected_source_commit: str,
) -> Any:
    """Verify the completed canonical attempt immediately before anchoring it.

    The preliminary digest and claim identity are bootstrap inputs only.  The
    strict qualifier verifier independently rereads every canonical artifact and
    returns the exact immutable report snapshot used to construct the tag.
    """

    from dungeon_apprentice.v02_u2_qualify import (
        QUALIFICATION_ATTEMPT_ID,
        canonical_qualification_paths,
        verify_u2_qualification_report,
    )

    repo = repository.expanduser().resolve()
    commit = _require_commit(expected_source_commit)
    if _git(repo, ["rev-parse", "--verify", "HEAD^{commit}"], runner=subprocess.run) != commit:
        raise U2QualificationAnchorError(
            "qualification source commit is not the active repository commit"
        )
    if _git(repo, ["status", "--porcelain=v1", "--untracked-files=all"], runner=subprocess.run):
        raise U2QualificationAnchorError(
            "qualification source must remain clean while it is externally anchored"
        )
    paths = canonical_qualification_paths()
    report_path = paths["report"]
    if report_path.is_symlink() or not report_path.is_file():
        raise U2QualificationAnchorError("canonical qualification report is missing or unsafe")
    try:
        report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    except OSError as error:
        raise U2QualificationAnchorError(
            "cannot read the canonical qualification report"
        ) from error
    attempt = _read_bootstrap_json(paths["attempt"], "qualification attempt ledger")
    attempt_id = attempt.get("attempt_id")
    claim_id = attempt.get("claim_id")
    if attempt_id != QUALIFICATION_ATTEMPT_ID:
        raise U2QualificationAnchorError(
            "qualification attempt identity differs from the frozen attempt"
        )
    return verify_u2_qualification_report(
        report_path,
        expected_source_commit=commit,
        expected_sha256=report_digest,
        expected_attempt_id=str(attempt_id),
        expected_claim_id=str(claim_id),
    )


def publish_external_anchor(
    repository: Path,
    evidence: Any,
    *,
    source_commit: str,
    runner: Runner = subprocess.run,
) -> U2QualificationAnchor:
    """Create or recover the fixed annotated tag, push it, then verify origin."""

    repo = repository.expanduser().resolve()
    commit = _require_commit(source_commit)
    message = anchor_message(evidence, source_commit=commit)
    if _git(repo, ["rev-parse", "--verify", "HEAD^{commit}"], runner=runner) != commit:
        raise U2QualificationAnchorError(
            "cannot anchor qualification to a non-active source commit"
        )
    if _git(repo, ["status", "--porcelain=v1", "--untracked-files=all"], runner=runner):
        raise U2QualificationAnchorError(
            "cannot publish qualification anchor from dirty source"
        )
    if _git(repo, ["remote", "get-url", ANCHOR_REMOTE], runner=runner) != EXPECTED_ORIGIN_URL:
        raise U2QualificationAnchorError(
            "qualification anchor remote is not the frozen GitHub repository"
        )

    local = runner(
        ["git", "show-ref", "--verify", f"refs/tags/{ANCHOR_TAG}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if local.returncode == 0:
        if (
            _git(repo, ["cat-file", "-t", f"refs/tags/{ANCHOR_TAG}"], runner=runner)
            != "tag"
            or _git(
                repo,
                ["rev-list", "-n", "1", f"refs/tags/{ANCHOR_TAG}"],
                runner=runner,
            )
            != commit
            or _git(
                repo,
                [
                    "for-each-ref",
                    "--format=%(contents)",
                    f"refs/tags/{ANCHOR_TAG}",
                ],
                runner=runner,
            )
            != message
        ):
            raise U2QualificationAnchorError(
                "existing local qualification tag differs from verified evidence"
            )
    elif local.returncode == 1:
        remote_before = _git(
            repo,
            ["ls-remote", "--tags", ANCHOR_REMOTE, f"refs/tags/{ANCHOR_TAG}"],
            runner=runner,
        )
        if remote_before:
            raise U2QualificationAnchorError(
                "qualification tag already exists remotely but not locally"
            )
        _git(
            repo,
            ["tag", "--annotate", ANCHOR_TAG, "--message", message, commit],
            runner=runner,
        )
    else:
        raise U2QualificationAnchorError(
            "cannot determine whether the qualification tag already exists"
        )

    remote = _git(
        repo,
        ["ls-remote", "--tags", ANCHOR_REMOTE, f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    if not remote:
        _git(
            repo,
            ["push", ANCHOR_REMOTE, f"refs/tags/{ANCHOR_TAG}"],
            runner=runner,
        )
    return verify_external_anchor(repo, runner=runner)


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    runner: Runner,
) -> str:
    try:
        completed = runner(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise U2QualificationAnchorError(
            f"cannot verify qualification anchor with git {' '.join(arguments)}"
        ) from error
    return completed.stdout.rstrip("\n")


def _parse_anchor_payload(raw_message: str) -> Mapping[str, Any]:
    if "\n" in raw_message:
        raise U2QualificationAnchorError(
            "qualification anchor tag message must be one JSON line"
        )
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as error:
        raise U2QualificationAnchorError(
            "qualification anchor tag message is not JSON"
        ) from error
    if not isinstance(payload, Mapping) or set(payload) != ANCHOR_FIELDS:
        raise U2QualificationAnchorError(
            "qualification anchor tag fields are incomplete or unknown"
        )
    return payload


def verify_external_anchor(
    repository: Path,
    *,
    expected_source_commit: str | None = None,
    runner: Runner = subprocess.run,
) -> U2QualificationAnchor:
    """Verify the fixed annotated tag against its remote object and strict payload."""

    repo = repository.expanduser().resolve()
    current_commit = _git(
        repo,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        runner=runner,
    )
    anchor_commit = (
        current_commit
        if expected_source_commit is None
        else str(expected_source_commit)
    )
    if _GIT_OBJECT.fullmatch(anchor_commit) is None:
        raise U2QualificationAnchorError(
            "expected qualification source commit is invalid"
        )
    if _git(repo, ["cat-file", "-t", f"refs/tags/{ANCHOR_TAG}"], runner=runner) != "tag":
        raise U2QualificationAnchorError(
            "qualification anchor must be an annotated Git tag"
        )
    tag_object = _git(
        repo,
        ["rev-parse", "--verify", f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    if _GIT_OBJECT.fullmatch(tag_object) is None:
        raise U2QualificationAnchorError("qualification tag object ID is invalid")
    peeled_commit = _git(
        repo,
        ["rev-list", "-n", "1", f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    if peeled_commit != anchor_commit:
        description = (
            "active source commit"
            if expected_source_commit is None
            else "expected source commit"
        )
        raise U2QualificationAnchorError(
            f"qualification anchor does not point to the {description}"
        )
    remote_url = _git(
        repo,
        ["remote", "get-url", ANCHOR_REMOTE],
        runner=runner,
    )
    if remote_url != EXPECTED_ORIGIN_URL:
        raise U2QualificationAnchorError(
            "qualification anchor remote is not the frozen GitHub repository"
        )
    remote_line = _git(
        repo,
        [
            "ls-remote",
            "--tags",
            ANCHOR_REMOTE,
            f"refs/tags/{ANCHOR_TAG}",
        ],
        runner=runner,
    )
    remote_parts = remote_line.split()
    if (
        len(remote_parts) != 2
        or remote_parts[0] != tag_object
        or remote_parts[1] != f"refs/tags/{ANCHOR_TAG}"
    ):
        raise U2QualificationAnchorError(
            "local qualification tag object is not frozen on origin"
        )
    raw_message = _git(
        repo,
        [
            "for-each-ref",
            "--format=%(contents)",
            f"refs/tags/{ANCHOR_TAG}",
        ],
        runner=runner,
    )
    payload = _parse_anchor_payload(raw_message)
    if (
        payload["schema_version"] != ANCHOR_SCHEMA_VERSION
        or payload["protocol"] != ANCHOR_PROTOCOL
        or payload["qualification_protocol"] != QUALIFICATION_PROTOCOL
        or payload["tag"] != ANCHOR_TAG
        or payload["remote"] != ANCHOR_REMOTE
        or payload["remote_url"] != EXPECTED_ORIGIN_URL
        or payload["source_commit"] != anchor_commit
        or payload["report"] != str(CANONICAL_REPORT)
    ):
        raise U2QualificationAnchorError(
            "qualification anchor identity differs from the frozen protocol"
        )
    report_length = payload["report_byte_length"]
    profile_version = payload["generator_profile_version"]
    profile = payload["generator_profile"]
    if (
        type(report_length) is not int
        or report_length <= 0
        or type(profile_version) is not int
        or profile_version <= 0
        or not isinstance(profile, str)
        or not profile
    ):
        raise U2QualificationAnchorError(
            "qualification anchor contains invalid counts/profile"
        )
    return U2QualificationAnchor(
        tag=ANCHOR_TAG,
        tag_object=tag_object,
        remote=ANCHOR_REMOTE,
        remote_url=EXPECTED_ORIGIN_URL,
        source_commit=current_commit,
        report=str(CANONICAL_REPORT),
        report_sha256=_require_digest(
            payload["report_sha256"], "qualification report"
        ),
        report_byte_length=report_length,
        attempt_id=_require_identity(
            payload["attempt_id"], "qualification attempt ID"
        ),
        attempt_sha256=_require_digest(
            payload["attempt_sha256"], "qualification attempt"
        ),
        claim_id=_require_digest(
            payload["claim_id"], "qualification claim ID"
        ),
        claim_sha256=_require_digest(
            payload["claim_sha256"], "qualification claim"
        ),
        generator_profile=profile,
        generator_profile_version=profile_version,
    )
