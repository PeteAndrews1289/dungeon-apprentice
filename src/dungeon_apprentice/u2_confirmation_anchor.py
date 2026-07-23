"""Remote annotated preregistration anchor for U2 post-training confirmation."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ANCHOR_SCHEMA_VERSION = 1
ANCHOR_PROTOCOL = "dungeon-apprentice-v0.2-u2-confirmation-anchor"
CONFIRMATION_PROTOCOL = "dungeon-apprentice-v0.2-u2-confirmation"
ANCHOR_TAG = "u2-confirmation-v0.2-u2-20260723"
ANCHOR_REMOTE = "origin"
EXPECTED_ORIGIN_URL = "https://github.com/PeteAndrews1289/dungeon-apprentice.git"
PLAN_REPOSITORY_PATH = "docs/v0.2-u2-confirmation-plan.md"
COHORT_SHA256 = "982b1fe6c58f5fa652dff1d2be0894f1918d2927e492b896b3fd102e52da7843"

ANCHOR_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "confirmation_protocol",
        "tag",
        "remote",
        "remote_url",
        "source_commit",
        "plan",
        "plan_sha256",
        "checkpoint_set_sha256",
        "cohort_sha256",
        "candidate_start",
        "candidate_end",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40,64}$")

Runner = Callable[..., subprocess.CompletedProcess[str]]


class U2ConfirmationAnchorError(RuntimeError):
    """Raised when the remote preregistration anchor is absent or changed."""


@dataclass(frozen=True)
class U2ConfirmationAnchor:
    tag: str
    tag_object: str
    remote: str
    remote_url: str
    source_commit: str
    plan: str
    plan_sha256: str
    checkpoint_set_sha256: str
    cohort_sha256: str
    candidate_start: int
    candidate_end: int

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise U2ConfirmationAnchorError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise U2ConfirmationAnchorError("confirmation source commit is invalid")
    return value


def build_anchor_payload(
    *,
    source_commit: str,
    plan_sha256: str,
    checkpoint_set_sha256: str,
) -> dict[str, Any]:
    """Build the exact one-line annotated-tag message."""

    return {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "protocol": ANCHOR_PROTOCOL,
        "confirmation_protocol": CONFIRMATION_PROTOCOL,
        "tag": ANCHOR_TAG,
        "remote": ANCHOR_REMOTE,
        "remote_url": EXPECTED_ORIGIN_URL,
        "source_commit": _require_commit(source_commit),
        "plan": PLAN_REPOSITORY_PATH,
        "plan_sha256": _require_sha256(plan_sha256, "confirmation plan"),
        "checkpoint_set_sha256": _require_sha256(
            checkpoint_set_sha256,
            "checkpoint set",
        ),
        "cohort_sha256": COHORT_SHA256,
        "candidate_start": 15_200_000,
        "candidate_end": 15_239_999,
    }


def anchor_message(
    *,
    source_commit: str,
    plan_sha256: str,
    checkpoint_set_sha256: str,
) -> str:
    return json.dumps(
        build_anchor_payload(
            source_commit=source_commit,
            plan_sha256=plan_sha256,
            checkpoint_set_sha256=checkpoint_set_sha256,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


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
        raise U2ConfirmationAnchorError(
            f"cannot verify U2 confirmation anchor with git {' '.join(arguments)}"
        ) from error
    return completed.stdout.rstrip("\n")


def _parse_payload(raw: str) -> Mapping[str, Any]:
    if "\n" in raw:
        raise U2ConfirmationAnchorError("confirmation anchor message must be one JSON line")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise U2ConfirmationAnchorError("confirmation anchor message is not JSON") from error
    if not isinstance(value, Mapping) or set(value) != ANCHOR_FIELDS:
        raise U2ConfirmationAnchorError(
            "confirmation anchor fields are incomplete or unknown"
        )
    return value


def verify_external_anchor(
    repository: Path,
    *,
    expected_source_commit: str,
    expected_plan_sha256: str,
    expected_checkpoint_set_sha256: str,
    runner: Runner = subprocess.run,
) -> U2ConfirmationAnchor:
    """Require the exact annotated tag object to exist locally and on origin."""

    repo = repository.expanduser().resolve()
    expected_payload = build_anchor_payload(
        source_commit=expected_source_commit,
        plan_sha256=expected_plan_sha256,
        checkpoint_set_sha256=expected_checkpoint_set_sha256,
    )
    head = _git(
        repo,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        runner=runner,
    )
    if head != expected_payload["source_commit"]:
        raise U2ConfirmationAnchorError(
            "active evaluator commit differs from the preregistered source"
        )
    if _git(repo, ["status", "--porcelain=v1", "--untracked-files=all"], runner=runner):
        raise U2ConfirmationAnchorError(
            "evaluator source became dirty after preregistration"
        )
    if _git(repo, ["remote", "get-url", ANCHOR_REMOTE], runner=runner) != EXPECTED_ORIGIN_URL:
        raise U2ConfirmationAnchorError(
            "confirmation anchor remote is not the frozen GitHub origin"
        )
    if _git(repo, ["cat-file", "-t", f"refs/tags/{ANCHOR_TAG}"], runner=runner) != "tag":
        raise U2ConfirmationAnchorError(
            "confirmation preregistration must be an annotated Git tag"
        )
    tag_object = _git(
        repo,
        ["rev-parse", "--verify", f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    if _GIT_OBJECT.fullmatch(tag_object) is None:
        raise U2ConfirmationAnchorError("confirmation tag object ID is invalid")
    peeled = _git(
        repo,
        ["rev-list", "-n", "1", f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    if peeled != head:
        raise U2ConfirmationAnchorError(
            "confirmation tag does not peel to the active evaluator commit"
        )
    remote_line = _git(
        repo,
        ["ls-remote", "--tags", ANCHOR_REMOTE, f"refs/tags/{ANCHOR_TAG}"],
        runner=runner,
    )
    remote_parts = remote_line.split()
    if (
        len(remote_parts) != 2
        or remote_parts[0] != tag_object
        or remote_parts[1] != f"refs/tags/{ANCHOR_TAG}"
    ):
        raise U2ConfirmationAnchorError(
            "local confirmation tag object is not frozen on origin"
        )
    payload = _parse_payload(
        _git(
            repo,
            [
                "for-each-ref",
                "--format=%(contents)",
                f"refs/tags/{ANCHOR_TAG}",
            ],
            runner=runner,
        )
    )
    if dict(payload) != expected_payload:
        raise U2ConfirmationAnchorError(
            "confirmation tag payload differs from source, plan, or checkpoints"
        )
    return U2ConfirmationAnchor(
        tag=ANCHOR_TAG,
        tag_object=tag_object,
        remote=ANCHOR_REMOTE,
        remote_url=EXPECTED_ORIGIN_URL,
        source_commit=head,
        plan=PLAN_REPOSITORY_PATH,
        plan_sha256=str(payload["plan_sha256"]),
        checkpoint_set_sha256=str(payload["checkpoint_set_sha256"]),
        cohort_sha256=str(payload["cohort_sha256"]),
        candidate_start=int(payload["candidate_start"]),
        candidate_end=int(payload["candidate_end"]),
    )


def publish_external_anchor(
    repository: Path,
    *,
    source_commit: str,
    plan_sha256: str,
    checkpoint_set_sha256: str,
    runner: Runner = subprocess.run,
) -> U2ConfirmationAnchor:
    """Create or recover the fixed tag, push it, then verify the remote object."""

    repo = repository.expanduser().resolve()
    message = anchor_message(
        source_commit=source_commit,
        plan_sha256=plan_sha256,
        checkpoint_set_sha256=checkpoint_set_sha256,
    )
    if _git(repo, ["rev-parse", "--verify", "HEAD^{commit}"], runner=runner) != source_commit:
        raise U2ConfirmationAnchorError(
            "cannot preregister a non-active evaluator commit"
        )
    if _git(repo, ["status", "--porcelain=v1", "--untracked-files=all"], runner=runner):
        raise U2ConfirmationAnchorError(
            "cannot preregister confirmation from dirty source"
        )
    if _git(repo, ["remote", "get-url", ANCHOR_REMOTE], runner=runner) != EXPECTED_ORIGIN_URL:
        raise U2ConfirmationAnchorError(
            "confirmation preregistration remote is not the frozen GitHub origin"
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
            != source_commit
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
            raise U2ConfirmationAnchorError(
                "existing local confirmation tag differs from this preregistration"
            )
    elif local.returncode == 1:
        remote_before = _git(
            repo,
            ["ls-remote", "--tags", ANCHOR_REMOTE, f"refs/tags/{ANCHOR_TAG}"],
            runner=runner,
        )
        if remote_before:
            raise U2ConfirmationAnchorError(
                "confirmation tag exists remotely but not as the verified local object"
            )
        _git(
            repo,
            ["tag", "--annotate", ANCHOR_TAG, "--message", message, source_commit],
            runner=runner,
        )
    else:
        raise U2ConfirmationAnchorError(
            "cannot determine whether the confirmation tag already exists"
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
    return verify_external_anchor(
        repo,
        expected_source_commit=source_commit,
        expected_plan_sha256=plan_sha256,
        expected_checkpoint_set_sha256=checkpoint_set_sha256,
        runner=runner,
    )
