from __future__ import annotations

from itertools import pairwise

import pytest

from dungeon_apprentice.u2_seed_guard import (
    PARTITION_BY_ROLE,
    SEED_PARTITIONS,
    U2AccessPhase,
    U2SeedAccess,
    U2SeedAccessError,
    U2SeedRole,
    authorize_u2_seed,
    classify_u2_seed,
    engineering_seed_access,
    qualified_training_seed_access,
    sealed_acknowledgement,
    sealed_preflight_seed_access,
    seed_partition_ledger,
)


def test_frozen_partitions_are_non_overlapping_and_have_exact_boundaries() -> None:
    for left, right in pairwise(SEED_PARTITIONS):
        assert left.stop <= right.start
    for partition in SEED_PARTITIONS:
        assert classify_u2_seed(partition.start) is partition.role
        assert classify_u2_seed(partition.stop - 1) is partition.role
        if partition.start and classify_u2_seed(partition.start - 1) is partition.role:
            raise AssertionError(f"{partition.role} starts too early")
        assert PARTITION_BY_ROLE[partition.role] is partition

    assert classify_u2_seed(-1) is None
    assert classify_u2_seed(5_201_000) is None
    assert classify_u2_seed(20_300_000) is None


def test_training_and_engineering_access_are_narrow() -> None:
    assert authorize_u2_seed(0, expected_role=U2SeedRole.TRAINING).role is U2SeedRole.TRAINING
    assert (
        authorize_u2_seed(999_999, expected_role=U2SeedRole.TRAINING).role
        is U2SeedRole.TRAINING
    )
    access = engineering_seed_access()
    assert access.phase is U2AccessPhase.ENGINEERING
    assert (
        authorize_u2_seed(
            5_200_000,
            expected_role=U2SeedRole.ENGINEERING,
            access=access,
        ).role
        is U2SeedRole.ENGINEERING
    )
    with pytest.raises(U2SeedAccessError, match="not engineering"):
        authorize_u2_seed(
            0,
            expected_role=U2SeedRole.ENGINEERING,
            access=access,
        )
    with pytest.raises(U2SeedAccessError, match="closed"):
        authorize_u2_seed(
            5_210_000,
            expected_role=U2SeedRole.SEALED_QUALIFICATION,
            access=access,
        )


def test_sealed_and_validation_ranges_fail_closed_without_opening_them() -> None:
    protected = {
        5_210_000: U2SeedRole.SEALED_QUALIFICATION,
        10_000_000: U2SeedRole.NAVIGATE_VALIDATION,
        11_000_000: U2SeedRole.U0_VALIDATION,
        11_100_000: U2SeedRole.U1_VALIDATION,
        11_200_000: U2SeedRole.U2_VALIDATION,
        15_200_000: U2SeedRole.FUTURE_U2_CONFIRMATION,
        15_210_000: U2SeedRole.FUTURE_NAVIGATE_CONFIRMATION,
        15_220_000: U2SeedRole.FUTURE_U0_CONFIRMATION,
        15_230_000: U2SeedRole.FUTURE_U1_CONFIRMATION,
        20_000_000: U2SeedRole.FINAL_TEST,
    }
    for seed, role in protected.items():
        with pytest.raises(U2SeedAccessError):
            authorize_u2_seed(seed, expected_role=role)


def test_capabilities_cannot_be_directly_constructed_or_minted_from_strings() -> None:
    with pytest.raises(U2SeedAccessError, match="issued"):
        U2SeedAccess(
            phase=U2AccessPhase.SEALED_PREFLIGHT,
            source_commit="abc1234",
            qualification_sha256=None,
            _marker=object(),
        )
    with pytest.raises(U2SeedAccessError, match="canonical qualifier"):
        sealed_preflight_seed_access(
            source_commit="abcdef1",
            clean_source=False,
            acknowledgement=f"{sealed_acknowledgement()} ",
        )
    with pytest.raises(U2SeedAccessError, match="verifier binding"):
        qualified_training_seed_access(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        qualified_training_seed_access(  # type: ignore[call-arg]
            source_commit="abcdef1",
            qualification_sha256="0" * 64,
        )


def test_ledger_is_readable_without_opening_any_protected_layout() -> None:
    ledger = seed_partition_ledger()
    assert len(ledger) == len(U2SeedRole)
    assert ledger[0] == {
        "role": "training",
        "start": 0,
        "opened_by_u2": True,
        "end": 999_999,
    }
    assert ledger[-1]["role"] == "final_test"
    assert ledger[-1]["end"] == 20_299_999
