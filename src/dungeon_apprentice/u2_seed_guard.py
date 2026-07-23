"""Fail-closed seed-role authorization for the U2 experiment.

The ranges in this module are evidence boundaries, not convenient defaults.
Importing the module grants no access to a sealed, validation, confirmation, or
final-test layout.  Sealed access is issued only from a claimed one-shot
qualifier launch, while validation access is issued only from the immutable
bytes returned by the qualification verifier.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class U2SeedAccessError(RuntimeError):
    """Raised before a seed outside the caller's declared evidence role is opened."""


class U2SeedRole(StrEnum):
    TRAINING = "training"
    ENGINEERING = "engineering"
    SEALED_QUALIFICATION = "sealed_qualification"
    NAVIGATE_VALIDATION = "navigate_validation"
    U0_VALIDATION = "u0_validation"
    U1_VALIDATION = "u1_validation"
    U2_VALIDATION = "u2_validation"
    FUTURE_U2_CONFIRMATION = "future_u2_confirmation"
    FUTURE_NAVIGATE_CONFIRMATION = "future_navigate_confirmation"
    FUTURE_U0_CONFIRMATION = "future_u0_confirmation"
    FUTURE_U1_CONFIRMATION = "future_u1_confirmation"
    U2R_U2_CONFIRMATION = "u2r_u2_confirmation"
    U2R_NAVIGATE_CONFIRMATION = "u2r_navigate_confirmation"
    U2R_U0_CONFIRMATION = "u2r_u0_confirmation"
    U2R_U1_CONFIRMATION = "u2r_u1_confirmation"
    FINAL_TEST = "final_test"


class U2AccessPhase(StrEnum):
    ENGINEERING = "engineering"
    SEALED_PREFLIGHT = "sealed_preflight"
    QUALIFIED_TRAINING = "qualified_training"
    POST_TRAINING_CONFIRMATION = "post_training_confirmation"


@dataclass(frozen=True)
class SeedPartition:
    role: U2SeedRole
    start: int
    stop: int
    opened_by_u2: bool

    def contains(self, seed: int) -> bool:
        return self.start <= seed < self.stop

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["end"] = self.stop - 1
        del value["stop"]
        return value


SEED_PARTITIONS: tuple[SeedPartition, ...] = (
    SeedPartition(U2SeedRole.TRAINING, 0, 1_000_000, True),
    SeedPartition(U2SeedRole.ENGINEERING, 5_200_000, 5_201_000, True),
    SeedPartition(U2SeedRole.SEALED_QUALIFICATION, 5_210_000, 5_212_000, True),
    SeedPartition(U2SeedRole.NAVIGATE_VALIDATION, 10_000_000, 10_000_080, False),
    SeedPartition(U2SeedRole.U0_VALIDATION, 11_000_000, 11_000_080, False),
    SeedPartition(U2SeedRole.U1_VALIDATION, 11_100_000, 11_100_080, False),
    SeedPartition(U2SeedRole.U2_VALIDATION, 11_200_000, 11_200_080, True),
    # The first four confirmation streams were consumed by the terminal U2
    # attempt.  Their legacy enum names remain stable for report compatibility,
    # while ``opened_by_u2`` now records their historical exposure.
    SeedPartition(U2SeedRole.FUTURE_U2_CONFIRMATION, 15_200_000, 15_210_000, True),
    SeedPartition(
        U2SeedRole.FUTURE_NAVIGATE_CONFIRMATION, 15_210_000, 15_220_000, True
    ),
    SeedPartition(U2SeedRole.FUTURE_U0_CONFIRMATION, 15_220_000, 15_230_000, True),
    SeedPartition(U2SeedRole.FUTURE_U1_CONFIRMATION, 15_230_000, 15_240_000, True),
    SeedPartition(U2SeedRole.U2R_U2_CONFIRMATION, 15_240_000, 15_250_000, False),
    SeedPartition(
        U2SeedRole.U2R_NAVIGATE_CONFIRMATION, 15_250_000, 15_260_000, False
    ),
    SeedPartition(U2SeedRole.U2R_U0_CONFIRMATION, 15_260_000, 15_270_000, False),
    SeedPartition(U2SeedRole.U2R_U1_CONFIRMATION, 15_270_000, 15_280_000, False),
    SeedPartition(U2SeedRole.FINAL_TEST, 20_000_000, 20_300_000, False),
)

PARTITION_BY_ROLE = MappingProxyType({partition.role: partition for partition in SEED_PARTITIONS})

_CAPABILITY_MARKER = object()
_SEALED_CLAIM_MARKER = object()
_QUALIFICATION_BINDING_MARKER = object()
_CONFIRMATION_CLAIM_MARKER = object()
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
_IDENTITY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{7,127}$")
_SEALED_ACKNOWLEDGEMENT = "OPEN U2 SEALED PREFLIGHT 5210000-5211999 ONCE"


@dataclass(frozen=True)
class _SealedLaunchClaim:
    """Opaque proof that the canonical launcher durably claimed the one attempt."""

    source_commit: str
    attempt_id: str
    claim_id: str
    claim_sha256: str
    launcher_token_sha256: str
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _SEALED_CLAIM_MARKER:
            raise U2SeedAccessError("sealed launch claims are qualifier-issued only")


@dataclass(frozen=True)
class _VerifiedQualificationBinding:
    """Opaque binding to the exact report, attempt, and claim bytes just verified."""

    source_commit: str
    report_sha256: str
    report_byte_length: int
    attempt_id: str
    attempt_sha256: str
    claim_id: str
    claim_sha256: str
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _QUALIFICATION_BINDING_MARKER:
            raise U2SeedAccessError("qualification bindings are verifier-issued only")


@dataclass(frozen=True)
class _ConfirmationLaunchClaim:
    """Opaque proof that one preregistered confirmation attempt was persisted."""

    source_commit: str
    protocol: str
    plan_sha256: str
    checkpoint_set_sha256: str
    anchor_tag: str
    anchor_tag_object: str
    anchor_remote_url: str
    attempt_id: str
    claim_id: str
    claim_sha256: str
    launcher_token_sha256: str
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _CONFIRMATION_CLAIM_MARKER:
            raise U2SeedAccessError("confirmation claims are evaluator-issued only")


@dataclass(frozen=True)
class U2SeedAccess:
    """A process-local capability; it is deliberately not reconstructible from JSON."""

    phase: U2AccessPhase
    source_commit: str | None
    qualification_sha256: str | None
    qualification_byte_length: int | None = None
    qualification_attempt_id: str | None = None
    qualification_attempt_sha256: str | None = None
    qualification_claim_id: str | None = None
    qualification_claim_sha256: str | None = None
    confirmation_protocol: str | None = None
    confirmation_plan_sha256: str | None = None
    confirmation_checkpoint_set_sha256: str | None = None
    confirmation_anchor_tag: str | None = None
    confirmation_anchor_tag_object: str | None = None
    confirmation_anchor_remote_url: str | None = None
    confirmation_attempt_id: str | None = None
    confirmation_claim_id: str | None = None
    confirmation_claim_sha256: str | None = None
    _marker: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._marker is not _CAPABILITY_MARKER:
            raise U2SeedAccessError("seed access capabilities must be issued by this module")

    def public_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "source_commit": self.source_commit,
            "qualification_sha256": self.qualification_sha256,
            "qualification_byte_length": self.qualification_byte_length,
            "qualification_attempt_id": self.qualification_attempt_id,
            "qualification_attempt_sha256": self.qualification_attempt_sha256,
            "qualification_claim_id": self.qualification_claim_id,
            "qualification_claim_sha256": self.qualification_claim_sha256,
            "confirmation_protocol": self.confirmation_protocol,
            "confirmation_plan_sha256": self.confirmation_plan_sha256,
            "confirmation_checkpoint_set_sha256": (
                self.confirmation_checkpoint_set_sha256
            ),
            "confirmation_anchor_tag": self.confirmation_anchor_tag,
            "confirmation_anchor_tag_object": self.confirmation_anchor_tag_object,
            "confirmation_anchor_remote_url": self.confirmation_anchor_remote_url,
            "confirmation_attempt_id": self.confirmation_attempt_id,
            "confirmation_claim_id": self.confirmation_claim_id,
            "confirmation_claim_sha256": self.confirmation_claim_sha256,
        }


def _valid_commit(value: str) -> str:
    normalized = value.strip().lower()
    if not _COMMIT_PATTERN.fullmatch(normalized):
        raise U2SeedAccessError("a hexadecimal clean source commit is required")
    return normalized


def _valid_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise U2SeedAccessError("a lowercase SHA-256 digest is required")
    return normalized


def _valid_identity(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if not _IDENTITY_PATTERN.fullmatch(normalized):
        raise U2SeedAccessError(f"{label} is not a canonical qualification identity")
    return normalized


def engineering_seed_access() -> U2SeedAccess:
    """Authorize only the permanently non-claim engineering sandbox."""

    return U2SeedAccess(
        phase=U2AccessPhase.ENGINEERING,
        source_commit=None,
        qualification_sha256=None,
        _marker=_CAPABILITY_MARKER,
    )


def sealed_preflight_seed_access(*_args: Any, **_kwargs: Any) -> U2SeedAccess:
    """Refuse ad-hoc sealed access.

    This compatibility refusal is deliberately public: older code cannot mint
    the capability merely by knowing an acknowledgement phrase.
    """

    raise U2SeedAccessError(
        "sealed access is available only inside the claimed canonical qualifier"
    )


def _new_sealed_launch_claim(
    *,
    source_commit: str,
    clean_source: bool,
    attempt_id: str,
    claim_id: str,
    claim_sha256: str,
    launcher_token_sha256: str,
) -> _SealedLaunchClaim:
    """Create the opaque half of an already-persisted canonical launch claim."""

    if not clean_source:
        raise U2SeedAccessError("sealed qualification requires a clean committed source")
    return _SealedLaunchClaim(
        source_commit=_valid_commit(source_commit),
        attempt_id=_valid_identity(attempt_id, "attempt ID"),
        claim_id=_valid_sha256(claim_id),
        claim_sha256=_valid_sha256(claim_sha256),
        launcher_token_sha256=_valid_sha256(launcher_token_sha256),
        _marker=_SEALED_CLAIM_MARKER,
    )


def _sealed_preflight_seed_access(
    *,
    claim: _SealedLaunchClaim,
    launcher_token: str,
) -> U2SeedAccess:
    """Issue sealed access only to the process holding the claimed launch token."""

    if not isinstance(claim, _SealedLaunchClaim) or claim._marker is not _SEALED_CLAIM_MARKER:
        raise U2SeedAccessError("a canonical sealed launch claim is required")
    try:
        token_bytes = bytes.fromhex(launcher_token)
    except ValueError as error:
        raise U2SeedAccessError("the launcher token is not hexadecimal") from error
    if len(token_bytes) != 32:
        raise U2SeedAccessError("the launcher token must contain 256 bits")
    measured = hashlib.sha256(launcher_token.encode("ascii")).hexdigest()
    if measured != claim.launcher_token_sha256:
        raise U2SeedAccessError("the launcher token does not match the canonical claim")
    return U2SeedAccess(
        phase=U2AccessPhase.SEALED_PREFLIGHT,
        source_commit=claim.source_commit,
        qualification_sha256=None,
        qualification_attempt_id=claim.attempt_id,
        qualification_claim_id=claim.claim_id,
        qualification_claim_sha256=claim.claim_sha256,
        _marker=_CAPABILITY_MARKER,
    )


def _new_verified_qualification_binding(
    *,
    source_commit: str,
    report_sha256: str,
    report_byte_length: int,
    attempt_id: str,
    attempt_sha256: str,
    claim_id: str,
    claim_sha256: str,
) -> _VerifiedQualificationBinding:
    """Bind a verifier result to the exact bytes it authenticated."""

    byte_length = int(report_byte_length)
    if isinstance(report_byte_length, bool) or byte_length <= 0:
        raise U2SeedAccessError("qualification report byte length must be positive")
    return _VerifiedQualificationBinding(
        source_commit=_valid_commit(source_commit),
        report_sha256=_valid_sha256(report_sha256),
        report_byte_length=byte_length,
        attempt_id=_valid_identity(attempt_id, "attempt ID"),
        attempt_sha256=_valid_sha256(attempt_sha256),
        claim_id=_valid_sha256(claim_id),
        claim_sha256=_valid_sha256(claim_sha256),
        _marker=_QUALIFICATION_BINDING_MARKER,
    )


def qualified_training_seed_access(
    binding: _VerifiedQualificationBinding,
) -> U2SeedAccess:
    """Issue validation access from a verifier-bound immutable report snapshot."""

    if (
        not isinstance(binding, _VerifiedQualificationBinding)
        or binding._marker is not _QUALIFICATION_BINDING_MARKER
    ):
        raise U2SeedAccessError(
            "qualified training access requires the immutable verifier binding"
        )
    return U2SeedAccess(
        phase=U2AccessPhase.QUALIFIED_TRAINING,
        source_commit=binding.source_commit,
        qualification_sha256=binding.report_sha256,
        qualification_byte_length=binding.report_byte_length,
        qualification_attempt_id=binding.attempt_id,
        qualification_attempt_sha256=binding.attempt_sha256,
        qualification_claim_id=binding.claim_id,
        qualification_claim_sha256=binding.claim_sha256,
        _marker=_CAPABILITY_MARKER,
    )


def _new_confirmation_launch_claim(
    *,
    source_commit: str,
    clean_source: bool,
    protocol: str,
    plan_sha256: str,
    checkpoint_set_sha256: str,
    anchor_tag: str,
    anchor_tag_object: str,
    anchor_remote_url: str,
    attempt_id: str,
    claim_id: str,
    claim_sha256: str,
    launcher_token_sha256: str,
) -> _ConfirmationLaunchClaim:
    """Bind confirmation access to a clean source, plan, policies, and attempt."""

    if not clean_source:
        raise U2SeedAccessError("post-training confirmation requires clean source")
    normalized_protocol = protocol.strip().lower()
    if normalized_protocol != "dungeon-apprentice-v0.2-u2-confirmation":
        raise U2SeedAccessError("the confirmation protocol identity is not frozen U2")
    if anchor_tag != "u2-confirmation-v0.2-u2-20260723":
        raise U2SeedAccessError("the confirmation anchor tag identity is not frozen")
    if not re.fullmatch(r"[0-9a-f]{40,64}", anchor_tag_object):
        raise U2SeedAccessError("the confirmation anchor tag object is invalid")
    if (
        anchor_remote_url
        != "https://github.com/PeteAndrews1289/dungeon-apprentice.git"
    ):
        raise U2SeedAccessError("the confirmation anchor remote is not frozen origin")
    return _ConfirmationLaunchClaim(
        source_commit=_valid_commit(source_commit),
        protocol=normalized_protocol,
        plan_sha256=_valid_sha256(plan_sha256),
        checkpoint_set_sha256=_valid_sha256(checkpoint_set_sha256),
        anchor_tag=anchor_tag,
        anchor_tag_object=anchor_tag_object,
        anchor_remote_url=anchor_remote_url,
        attempt_id=_valid_identity(attempt_id, "confirmation attempt ID"),
        claim_id=_valid_sha256(claim_id),
        claim_sha256=_valid_sha256(claim_sha256),
        launcher_token_sha256=_valid_sha256(launcher_token_sha256),
        _marker=_CONFIRMATION_CLAIM_MARKER,
    )


def _post_training_confirmation_seed_access(
    *,
    claim: _ConfirmationLaunchClaim,
    launcher_token: str,
) -> U2SeedAccess:
    """Open only the reserved U2 confirmation streams for one persisted claim."""

    if (
        not isinstance(claim, _ConfirmationLaunchClaim)
        or claim._marker is not _CONFIRMATION_CLAIM_MARKER
    ):
        raise U2SeedAccessError("a persisted U2 confirmation claim is required")
    try:
        token_bytes = bytes.fromhex(launcher_token)
    except ValueError as error:
        raise U2SeedAccessError("the confirmation launcher token is not hexadecimal") from error
    if len(token_bytes) != 32:
        raise U2SeedAccessError("the confirmation launcher token must contain 256 bits")
    measured = hashlib.sha256(launcher_token.encode("ascii")).hexdigest()
    if measured != claim.launcher_token_sha256:
        raise U2SeedAccessError(
            "the confirmation launcher token does not match the persisted claim"
        )
    return U2SeedAccess(
        phase=U2AccessPhase.POST_TRAINING_CONFIRMATION,
        source_commit=claim.source_commit,
        qualification_sha256=None,
        confirmation_protocol=claim.protocol,
        confirmation_plan_sha256=claim.plan_sha256,
        confirmation_checkpoint_set_sha256=claim.checkpoint_set_sha256,
        confirmation_anchor_tag=claim.anchor_tag,
        confirmation_anchor_tag_object=claim.anchor_tag_object,
        confirmation_anchor_remote_url=claim.anchor_remote_url,
        confirmation_attempt_id=claim.attempt_id,
        confirmation_claim_id=claim.claim_id,
        confirmation_claim_sha256=claim.claim_sha256,
        _marker=_CAPABILITY_MARKER,
    )


def classify_u2_seed(seed: int) -> U2SeedRole | None:
    candidate = int(seed)
    for partition in SEED_PARTITIONS:
        if partition.contains(candidate):
            return partition.role
    return None


def authorize_u2_seed(
    seed: int,
    *,
    expected_role: U2SeedRole,
    access: U2SeedAccess | None = None,
) -> SeedPartition:
    """Verify a seed's exact role and the capability required to open that role."""

    candidate = int(seed)
    expected = U2SeedRole(expected_role)
    actual = classify_u2_seed(candidate)
    if actual is None:
        raise U2SeedAccessError(f"seed {candidate} is outside every frozen U2 partition")
    if actual is not expected:
        raise U2SeedAccessError(
            f"seed {candidate} is reserved for {actual.value}, not {expected.value}"
        )
    if expected is U2SeedRole.TRAINING:
        return PARTITION_BY_ROLE[expected]
    if expected is U2SeedRole.ENGINEERING:
        if access is None or access.phase is not U2AccessPhase.ENGINEERING:
            raise U2SeedAccessError("engineering layouts require engineering seed access")
        return PARTITION_BY_ROLE[expected]
    if expected is U2SeedRole.SEALED_QUALIFICATION:
        if (
            access is None
            or access.phase is not U2AccessPhase.SEALED_PREFLIGHT
            or access.source_commit is None
            or access.qualification_attempt_id is None
            or access.qualification_claim_id is None
            or access.qualification_claim_sha256 is None
        ):
            raise U2SeedAccessError("the sealed qualification range is closed")
        return PARTITION_BY_ROLE[expected]
    if expected in {
        U2SeedRole.NAVIGATE_VALIDATION,
        U2SeedRole.U0_VALIDATION,
        U2SeedRole.U1_VALIDATION,
        U2SeedRole.U2_VALIDATION,
    }:
        if access is None or access.phase not in {
            U2AccessPhase.SEALED_PREFLIGHT,
            U2AccessPhase.QUALIFIED_TRAINING,
        }:
            raise U2SeedAccessError("validation layouts require qualified seed access")
        if (
            access.phase is U2AccessPhase.SEALED_PREFLIGHT
            and expected is not U2SeedRole.U2_VALIDATION
        ):
            raise U2SeedAccessError(
                "the sealed qualifier may open only its U2 validation reference"
            )
        if access.phase is U2AccessPhase.QUALIFIED_TRAINING and (
            access.qualification_sha256 is None
            or access.qualification_byte_length is None
            or access.qualification_attempt_id is None
            or access.qualification_attempt_sha256 is None
            or access.qualification_claim_id is None
            or access.qualification_claim_sha256 is None
        ):
            raise U2SeedAccessError("qualified seed access is not bound to exact evidence bytes")
        return PARTITION_BY_ROLE[expected]
    if expected in {
        U2SeedRole.FUTURE_U2_CONFIRMATION,
        U2SeedRole.FUTURE_NAVIGATE_CONFIRMATION,
        U2SeedRole.FUTURE_U0_CONFIRMATION,
        U2SeedRole.FUTURE_U1_CONFIRMATION,
    }:
        if (
            access is None
            or access.phase is not U2AccessPhase.POST_TRAINING_CONFIRMATION
            or access.source_commit is None
            or access.confirmation_protocol
            != "dungeon-apprentice-v0.2-u2-confirmation"
            or access.confirmation_plan_sha256 is None
            or access.confirmation_checkpoint_set_sha256 is None
            or access.confirmation_anchor_tag
            != "u2-confirmation-v0.2-u2-20260723"
            or access.confirmation_anchor_tag_object is None
            or access.confirmation_anchor_remote_url
            != "https://github.com/PeteAndrews1289/dungeon-apprentice.git"
            or access.confirmation_attempt_id is None
            or access.confirmation_claim_id is None
            or access.confirmation_claim_sha256 is None
        ):
            raise U2SeedAccessError(
                "consumed confirmation layouts require the terminal U2 confirmation claim"
            )
        return PARTITION_BY_ROLE[expected]
    if expected in {
        U2SeedRole.U2R_U2_CONFIRMATION,
        U2SeedRole.U2R_NAVIGATE_CONFIRMATION,
        U2SeedRole.U2R_U0_CONFIRMATION,
        U2SeedRole.U2R_U1_CONFIRMATION,
    }:
        raise U2SeedAccessError(
            "U2r confirmation layouts remain closed until a separately "
            "preregistered post-training confirmation protocol issues access"
        )
    raise U2SeedAccessError(f"{expected.value} is reserved and cannot be opened during U2")


def seed_partition_ledger() -> tuple[dict[str, Any], ...]:
    """Return the complete immutable range ledger without generating a layout."""

    return tuple(partition.public_dict() for partition in SEED_PARTITIONS)


def sealed_acknowledgement() -> str:
    """Expose the exact phrase for CLI help without granting any seed access."""

    return _SEALED_ACKNOWLEDGEMENT
