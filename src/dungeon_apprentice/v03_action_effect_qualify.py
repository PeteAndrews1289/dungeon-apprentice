"""Fail-closed Stage-A qualification for the v0.3 action-effect study.

This module verifies a prospective release; it never creates the annotated
tag and it never launches either scientific arm.  The qualifier deliberately
restarts from the confirmed U1 parent and treats the completed U2-S ablation
as terminal evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dungeon_apprentice import v02_u2 as frozen_u2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as frozen_u2r
from dungeon_apprentice import v02_u2s as frozen_u2s
from dungeon_apprentice import v02_u2s_qualify as u2s_qualify
from dungeon_apprentice import v03_action_effect as v03
from dungeon_apprentice import v03_action_effect_smoke as smoke
from dungeon_apprentice.action_effect import (
    ACTION_EFFECT_DIM,
    ACTION_EFFECT_KEY,
    IMAGE_KEY,
    ActionEffectMode,
)
from dungeon_apprentice.artifacts import file_sha256, runtime_snapshot, utc_now
from dungeon_apprentice.u2_seed_guard import (
    SEED_PARTITIONS,
    U2SeedRole,
)

PROTOCOL = v03.PROTOCOL
SCHEMA_VERSION = 1
KIND = "sealed_stage_a_preflight_qualification"
VERDICT = "qualified"

QUALIFIED_TAG = "action-effect-architecture-v0.3-stage-a-r3-20260724"
QUALIFIED_REMOTE = "origin"
EXPECTED_ORIGIN_URL = "https://github.com/PeteAndrews1289/dungeon-apprentice.git"
PROTOCOL_DOCUMENT = Path(v03.PROTOCOL_DOCUMENT)
DASHBOARD_PORT = 8791

CANONICAL_QUALIFICATION_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r3-20260724"
)
CANONICAL_REPORT = CANONICAL_QUALIFICATION_DIRECTORY / "report.json"
CANONICAL_CHECKSUM = CANONICAL_QUALIFICATION_DIRECTORY / "report.json.sha256"
CANONICAL_CLAIM = CANONICAL_QUALIFICATION_DIRECTORY / "claim.json"
CANONICAL_COHORT_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r3-20260724"
)
CANONICAL_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r3-media-20260724"
)

FAILED_STAGE_A_ATTEMPT_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-20260724"
)
FAILED_STAGE_A_ATTEMPT_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r1-media-20260724"
)
FAILED_STAGE_A_ATTEMPT_QUALIFICATION_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.3-action-effect-stage-a-r1-20260724"
)
FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG = Path(
    "/Volumes/T7 Developer/DungeonApprentice/launch-recovery/"
    "v03-action-effect-stage-a-r1-20260724-launcher.log"
)
FAILED_STAGE_A_ATTEMPT_TAG = "action-effect-architecture-v0.3-stage-a-r1-20260724"
FAILED_STAGE_A_ATTEMPT_TAG_OBJECT = "1f11b98fae6691dc9282682c44cdaa33ea55e5ab"
FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT = "b9dc80b4c5f95b4b0a8bf7a681b9431764c6f0ec"
FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256 = (
    "c59c3033a892bb05fea437bc525090c4f958110c960494fc2366444526d2dcb7"
)
FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CLAIM_SHA256 = (
    "86f6473d5610bbc80e5487db039cc99dc92a7a9558c0b7a7237315b306519e9e"
)
FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256 = (
    "3b12be77dee66a7cafdf8bfb86cbfc380c2688eeb9b33d057d559634004c35a2"
)
FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256 = (
    "647f8237c2dfcb8451a6ccb644e2b6b10f1087bd9a5b5d7a3dcc0b23426fe22d"
)
FAILED_STAGE_A_ATTEMPT_COHORT_SHA256 = (
    "d989b884d1ff63ed87194b196f87e4396673200762167fa1f9de0c735727b83b"
)
FAILED_STAGE_A_ATTEMPT_DASHBOARD_LOG_SHA256 = (
    "3686976b942556d2ba83947da71e33f62021fa85322bf46ad759f7ebf9c81d68"
)
FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG_SHA256 = (
    "598df636ad3f08538636f25311b20933849dc51386335e8dbc65ae76f5a95d7e"
)
FAILED_STAGE_A_ATTEMPT_PREDECESSOR_SHA256 = (
    "0caffd0f2084a1dd2c50744fe9a60876e77fa8ca7302c9f802ca0089e84b9cd9"
)
FAILED_STAGE_A_ATTEMPT_TAG_PAYLOAD_SHA256 = (
    "02531149bcac991fd4255897a6ea2729f751dc499ca8e7576b5221983f385a64"
)
FAILED_STAGE_A_ATTEMPT_FIRST_ROLLOUT_SHA256 = (
    "fa4c7bda99a261f8fa49741a49360cd1bfc6ab3081db51aeffc64266a109ce72"
)
FAILED_STAGE_A_ATTEMPT_INITIAL_RNG_SHA256 = (
    "91e1c8cd5473452a8e6d91614009582b0cd43f6bfa058a8477714dc2e4508d8d"
)
FAILED_STAGE_A_ATTEMPT_LATEST_OBSERVED_SHA256 = (
    "0124df33809e0b81ccef181a98060617130ae12ba7eff116c8675fb3a27ac943"
)
FAILED_STAGE_A_ATTEMPT_FIRST_EXAM_SHA256 = (
    "bcc9342942c672f60e00d547899f6a4cfe1a87a58df01527a6e70f9f38948eb8"
)
FAILED_STAGE_A_ATTEMPT_ARTIFACT_MAP_SHA256 = (
    "eb529df93a43be803d47ba73069ac428bdf953065aa7258d34c2d1dd333bbdc2"
)
FAILED_STAGE_A_ATTEMPT_ROOT_FILE_SHA256 = {
    "cohort-contract.json": FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256,
    "cohort.json": FAILED_STAGE_A_ATTEMPT_COHORT_SHA256,
    "dashboard.log": FAILED_STAGE_A_ATTEMPT_DASHBOARD_LOG_SHA256,
    "dashboard.pid": "5ab8c06c972454b388a27186f5ab1283f67a955a4ca90a01c8819280560277d4",
    "launcher-sham-attempt-0.json": (
        "7a670bc04611027d4e56f9c85817723ce6ca6f17e72edff24084f04c29c77f31"
    ),
    "sham/checkpoints/initial.cases.json": (
        "8d3134f0a6a440520112c3337d057bade276a9e83cc7078e112cb398f0c15694"
    ),
    "sham/checkpoints/initial.integrity.json": (
        "eddedb71fbdae8b70a567ea4ab979f4602d3a166cf4fec2aef52686f32d94635"
    ),
    "sham/checkpoints/initial.json": (
        "d4ac77ccca104a10babaac1c8a4c780398dc973953177509f85e0392ed410342"
    ),
    "sham/checkpoints/initial.zip": (
        "17672c9b233fedc7b5f7d2563cc42dee1f4c367ebea25b338bc3caf62136852c"
    ),
    "sham/checkpoints/latest-observed.integrity.json": (
        "59aa23fe84f7821375a6d2ad7da83cb4a846a8a8b80c6dffdfd8e8c2287dba99"
    ),
    "sham/checkpoints/latest-observed.json": (
        "720b4f2b9318ec62941ece4272f7134f05f51023f17059495bffc8b4ee7ae237"
    ),
    "sham/checkpoints/latest-observed.zip": (FAILED_STAGE_A_ATTEMPT_LATEST_OBSERVED_SHA256),
    "sham/checkpoints/rolling/exam-0032768.cases.json": (
        "56c46f1ecb91082dc4d3ccd404ccbaea82a0b8065fb6fe393e830656d1a82c58"
    ),
    "sham/checkpoints/rolling/exam-0032768.integrity.json": (
        "759f94f6ff8a10e7b12b24e90c39dc475c9006f99699cdc3accbc28fdd3e57d8"
    ),
    "sham/checkpoints/rolling/exam-0032768.json": (
        "e42cef3787805895adb63e0a9b220b4367dc3678f66729d22d1961c82a24b6d9"
    ),
    "sham/checkpoints/rolling/exam-0032768.zip": (FAILED_STAGE_A_ATTEMPT_FIRST_EXAM_SHA256),
    "sham/episode-starts.jsonl": (
        "80b638f861701357d57ba98c18d9db3437271e7748ab99875f755bb7f035d56f"
    ),
    "sham/episodes.jsonl": ("6a7150325de144b7f66eb82b0edd1f35cebc186564ca72645150083284a80011"),
    "sham/evaluations.jsonl": ("0d29f92084d52934c9c4f25e40a32b8c948aa9ee6356fcbca6514802c59ad5aa"),
    "sham/events.jsonl": ("b0473dfaae5100964fad85106346752ca9459c668334f43ebf77b1a4a610fd82"),
    "sham/first-rollout.json": ("8cf663b50973e608a049be40f0217597af9d20a5f5e8fdd76b61df92cf037f2f"),
    "sham/frames/exam-navigate-full.png": (
        "347ef72bfcdb37005396e23e118d5a2551d04e2c05364b57c58c2516cfec4e02"
    ),
    "sham/frames/exam-unlock-u0-visible.png": (
        "535d02397c25922983bc38fb0ad88ba31b8e9a395937fdc5aa23c28a0bde5af8"
    ),
    "sham/frames/exam-unlock-u1-local.png": (
        "abd5bc409ce68d381a5273d28ee21c435ba9ff40b7eabc7643e3601aa51afb96"
    ),
    "sham/frames/exam-unlock-u2-separated.png": (
        "0721499dfd0a5c4b33c2f1253515e00452f0b2155a4d2d52b6be69fad5aaf74c"
    ),
    "sham/frames/latest.png": ("55144ddfb7a7c5817d03406f99621edd9ed0d32c339fb880c99c652495e4288c"),
    "sham/manifest.json": ("8909ef7d12ca8a69be08d88fae060e995f919f33f32958c2eaed7feaa65502e9"),
    "sham/optimizer.jsonl": ("fb4a7a08316582a6be26485a2090a3cbab726d867bc7d07c6ca67ea94b3b1a25"),
    "sham/status.json": ("8cdd9133ce72d14e79d105bf63e88dc9cb9951f788f22eb517f6e72fe8e33be9"),
}
FAILED_STAGE_A_ATTEMPT_ROOT_DIRECTORIES = {
    ".v03-staging",
    "sham",
    "sham/checkpoints",
    "sham/checkpoints/rolling",
    "sham/frames",
}

FAILED_STAGE_A_R2_ATTEMPT_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r2-20260724"
)
FAILED_STAGE_A_R2_ATTEMPT_MEDIA_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/v03-action-effect-stage-a-r2-media-20260724"
)
FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_DIRECTORY = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/"
    "v0.3-action-effect-stage-a-r2-20260724"
)
FAILED_STAGE_A_R2_ATTEMPT_LAUNCHER_LOG = Path(
    "/Volumes/T7 Developer/DungeonApprentice/launch-recovery/"
    "v03-action-effect-stage-a-r2-20260724-launcher.log"
)
FAILED_STAGE_A_R2_ATTEMPT_SCREEN_LOG = Path(
    "/Volumes/T7 Developer/DungeonApprentice/launch-recovery/"
    "v03-action-effect-stage-a-r2-20260724-screen.log"
)
FAILED_STAGE_A_R2_ATTEMPT_TAG = "action-effect-architecture-v0.3-stage-a-r2-20260724"
FAILED_STAGE_A_R2_ATTEMPT_TAG_OBJECT = "87d4ce7d24bd13d3a5e3e182c29889c36db35d87"
FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT = "01b1b910edbb676f5de7375fa55d1b6e3bc6a54c"
FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_REPORT_SHA256 = (
    "148bb469cf66753b3ab998299e5f595c94d4fb868c5c38d93c8ec0e9cd097a9e"
)
FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_CLAIM_SHA256 = (
    "f0a78af0e4445e1a453bfbb4e4aa971de130e88656f36718a7fc6ec5ec5409f6"
)
FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256 = (
    "7dbf37ba9c91cd6af35dea34f2f111bad3149b11d7538d7e3aca17a045967c9f"
)
FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_TAG_PAYLOAD_SHA256 = (
    "e5f07493866176e7270ea47daac9572e67e00960a26453cd0db932103868e512"
)
FAILED_STAGE_A_R2_ATTEMPT_R1_PREDECESSOR_SHA256 = (
    "5d8038681355264b51c51fd4b5c2b93d3e49804286bc78c8d89a1efd21265110"
)
FAILED_STAGE_A_R2_ATTEMPT_CONTRACT_SHA256 = (
    "4114f1d03e5d7e84eab9106676b6dcb9a8b2b0f2b8376a765f599dc61b1b49fa"
)
FAILED_STAGE_A_R2_ATTEMPT_COHORT_SHA256 = (
    "7cd16a4f3cd886d9036be33dca8dce7e23799b0620a7a7272bcc482b2169bacc"
)
FAILED_STAGE_A_R2_ATTEMPT_STATUS_SHA256 = (
    "f0e7e68e6e8057b503db179ae92f7acf59c908c8e7a8ac3027d10d948030a294"
)
FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_FILE_SHA256 = (
    "8cf663b50973e608a049be40f0217597af9d20a5f5e8fdd76b61df92cf037f2f"
)
FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_SHA256 = (
    "fa4c7bda99a261f8fa49741a49360cd1bfc6ab3081db51aeffc64266a109ce72"
)
FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_NO_LF_SHA256 = (
    "3b9ecf3ac69c834ddc879d1a542e9f109d833f30aa2324e80c099f7a2195b81c"
)
FAILED_STAGE_A_R2_ATTEMPT_INITIAL_RNG_SHA256 = (
    "91e1c8cd5473452a8e6d91614009582b0cd43f6bfa058a8477714dc2e4508d8d"
)
FAILED_STAGE_A_R2_ATTEMPT_REPORT_SHA256 = (
    "c90a9b4728fcddbec7b602ac67fde9aad784bb26c3979c8d2608f7dfdcd4251c"
)
FAILED_STAGE_A_R2_ATTEMPT_REPORT_INTEGRITY_SHA256 = (
    "ba38e5f2a79f7ae349a66fd28d1efb8eb91c603d3e41913229d0d804c3bca3b6"
)
FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_CHECKPOINT_SHA256 = (
    "a9f06093d07110fc2eb069f911984cf5b6e2746ea10463e783e0c58b4425ca06"
)
FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_SIDECAR_SHA256 = (
    "5a4e0db926b82498c24749b1097eff0263d0e3100ec491259abac732f51491f2"
)
FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_INTEGRITY_SHA256 = (
    "267c2927ac1be4577871e7a45f868d278ec87d466690023b44769fb93584e0e7"
)
FAILED_STAGE_A_R2_ATTEMPT_FINAL_EXAM_SIDECAR_SHA256 = (
    "1ca57d4bba8adbdb216e26318f89c14a4682daa440c5353f09203f6fcfa87063"
)
FAILED_STAGE_A_R2_ATTEMPT_FINAL_EXAM_INTEGRITY_SHA256 = (
    "b5a31a98eccac66d7f2fb9ace114a2b9994eca71f2c5d8e8a90e9202f8c6ad69"
)
FAILED_STAGE_A_R2_ATTEMPT_FINAL_EXAM_CASES_SHA256 = (
    "b0fa53d79e15bdec7524062f33fcab6756c9bf2c64564fdd3a8e92de1fd0f3eb"
)
FAILED_STAGE_A_R2_ATTEMPT_LAUNCHER_LOG_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
FAILED_STAGE_A_R2_ATTEMPT_SCREEN_LOG_SHA256 = (
    "18efff19216bc33b86284b54bfb6be097842866a275a5d772a63049c039534d7"
)
FAILED_STAGE_A_R2_ATTEMPT_EXAM_INDEX_SHA256 = (
    "e854bc749b56686b9a02afa312a87354aeebaa0297607353383d7b85b73c6f27"
)
FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_MAP_SHA256 = (
    "c988e14a5c9e618c8ee99215f86f9dd9146e02ce7f0ef07f52f653d5cc2a0e35"
)
FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_COUNT = 163
FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_BYTES = 942_965_280
FAILED_STAGE_A_R2_ATTEMPT_TREE_SEALS = {
    "cohort": {
        "regular_file_count": 158,
        "regular_file_bytes": 942_881_820,
        "directory_count_including_root": 6,
        "directory_list_sha256": (
            "178b7c5d2fabf775ce41e33d1b0bfc7fb24dab31831000ca5937d065d48d0f22"
        ),
        "regular_file_map_sha256": (
            "7899040829dda16290ea1ce718b239f0ce0ba78bf1a00d6ab04e7c3861cb2245"
        ),
    },
    "qualification": {
        "regular_file_count": 3,
        "regular_file_bytes": 80_358,
        "directory_count_including_root": 1,
        "directory_list_sha256": (
            "19a3f6872c35183258369865a8dbf1c1d9153746792b755d177241c485a2544f"
        ),
        "regular_file_map_sha256": (
            "c861ebbfc5d88225e8a242daf6594d8fc10cca519bbdf3246bc58eec3af50502"
        ),
    },
    "media": {
        "regular_file_count": 0,
        "regular_file_bytes": 0,
        "directory_count_including_root": 2,
        "directory_list_sha256": (
            "14b40bc8da28427123ed1baeac7faa53d23a0166acf0b55bea64117b698dddb8"
        ),
        "regular_file_map_sha256": (
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        ),
    },
    "launch-recovery": {
        "regular_file_count": 2,
        "regular_file_bytes": 3_102,
        "regular_file_map_sha256": (
            "a84fee359c9cf611bf37e85de5c5b0c1175610463ccf582197fdec843339619a"
        ),
    },
}
FAILED_STAGE_A_R2_ATTEMPT_CRITICAL_SHA256 = {
    "cohort/cohort-contract.json": FAILED_STAGE_A_R2_ATTEMPT_CONTRACT_SHA256,
    "cohort/cohort.json": FAILED_STAGE_A_R2_ATTEMPT_COHORT_SHA256,
    "cohort/launcher-sham-attempt-0.json": (
        "8b19989391378ea62da557b092fddb26d6087cb08a06ca4239e8eaa3caea21a3"
    ),
    "cohort/sham/manifest.json": (
        "1a341ab7a642f41fe916f33ddbc4a86bdcaec1c00cad13189fc7d67c9cb82b84"
    ),
    "cohort/sham/status.json": FAILED_STAGE_A_R2_ATTEMPT_STATUS_SHA256,
    "cohort/sham/first-rollout.json": (
        FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_FILE_SHA256
    ),
    "cohort/sham/report.json": FAILED_STAGE_A_R2_ATTEMPT_REPORT_SHA256,
    "cohort/sham/report.integrity.json": (
        FAILED_STAGE_A_R2_ATTEMPT_REPORT_INTEGRITY_SHA256
    ),
    "cohort/sham/checkpoints/terminal.zip": (
        FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_CHECKPOINT_SHA256
    ),
    "cohort/sham/checkpoints/terminal.json": (
        FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_SIDECAR_SHA256
    ),
    "cohort/sham/checkpoints/terminal.integrity.json": (
        FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_INTEGRITY_SHA256
    ),
    "cohort/sham/checkpoints/rolling/exam-1048576.zip": (
        FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_CHECKPOINT_SHA256
    ),
    "cohort/sham/checkpoints/rolling/exam-1048576.json": (
        FAILED_STAGE_A_R2_ATTEMPT_FINAL_EXAM_SIDECAR_SHA256
    ),
    "cohort/sham/checkpoints/rolling/exam-1048576.integrity.json": (
        FAILED_STAGE_A_R2_ATTEMPT_FINAL_EXAM_INTEGRITY_SHA256
    ),
    "cohort/sham/checkpoints/rolling/exam-1048576.cases.json": (
        FAILED_STAGE_A_R2_ATTEMPT_FINAL_EXAM_CASES_SHA256
    ),
    "qualification/claim.json": FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_CLAIM_SHA256,
    "qualification/report.json": FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_REPORT_SHA256,
    "qualification/report.json.sha256": (
        FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256
    ),
    "launch-recovery/v03-action-effect-stage-a-r2-20260724-launcher.log": (
        FAILED_STAGE_A_R2_ATTEMPT_LAUNCHER_LOG_SHA256
    ),
    "launch-recovery/v03-action-effect-stage-a-r2-20260724-screen.log": (
        FAILED_STAGE_A_R2_ATTEMPT_SCREEN_LOG_SHA256
    ),
}

U2S_TERMINAL_ROOT = Path("/Volumes/T7 Developer/DungeonApprentice/u2s-ablation-r1-20260723")
U2S_TERMINAL_REPORT_SHA256 = "dfd288955bd2f8367ba3818e7242ff29c4a6e8e5a03d248df45e85f43b562e44"
U2S_TERMINAL_INTEGRITY_SHA256 = "a9468a504f48911943c43a5a6165917c3dd05697c21abfd784efe9f882b7faa2"
U2S_TERMINAL_SOURCE_COMMIT = "2e2a91c9864720326a5fa8ba82212f116d9ead04"
U2S_QUALIFICATION_REPORT_SHA256 = "7fd8fa009191c35ef608767a93e41dfc2c91aca20ae43493a5d1c32624cb467a"

PARENT_SIDECAR_SHA256 = u2s_qualify.PARENT_SIDECAR_SHA256
PARENT_MANIFEST_SHA256 = u2s_qualify.PARENT_MANIFEST_SHA256

TAG_SCHEMA_VERSION = 1
TAG_KIND = "v03_action_effect_stage_a_source_preregistration"
TAG_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol",
        "tag",
        "remote",
        "remote_url",
        "source_commit",
        "protocol_document_sha256",
        "parent",
        "u2s_terminal",
        "failed_stage_a_attempt",
        "failed_stage_a_r2_attempt",
        "guard_mapping_sha256",
        "sampler_preflight_sha256",
        "architecture_contract_sha256",
        "protected_partitions_sha256",
        "runtime_contract",
        "roots",
        "dashboard_port",
        "storage_caps",
        "resume_rule",
    }
)

_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "kind",
        "verdict",
        "created_at",
        "claim",
        "source",
        "protocol_document",
        "parent",
        "predecessors",
        "failed_stage_a_attempt",
        "failed_stage_a_r2_attempt",
        "guards",
        "sampler_preflight",
        "architecture_contract",
        "protected_partitions",
        "smoke_evidence",
        "storage_caps",
        "storage_preflight",
        "restrictions",
    }
)
_SHA256 = frozenset("0123456789abcdef")
_GIT_OBJECT_LENGTHS = frozenset({40, 64})
_ARMS = tuple(mode.value for mode in v03.ARM_ORDER)
MAX_REPORT_BYTES = 8 * 1024**2
MAX_SMOKE_BYTES = 8 * 1024**2


class ActionEffectQualificationError(RuntimeError):
    """Raised when Stage A cannot prove its complete pre-action contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_set_sha256(values: Sequence[str] | set[str] | frozenset[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in _SHA256 for character in result):
        raise ActionEffectQualificationError(f"{label} is not a lowercase SHA-256 digest")
    return result


def _require_git_object(value: Any, label: str) -> str:
    result = str(value)
    if len(result) not in _GIT_OBJECT_LENGTHS or any(
        character not in _SHA256 for character in result
    ):
        raise ActionEffectQualificationError(f"{label} is not a Git object ID")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ActionEffectQualificationError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ActionEffectQualificationError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ActionEffectQualificationError(f"{label} is not a JSON object")
    return value


def _regular_file_bytes(path: Path, label: str, *, maximum: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ActionEffectQualificationError(f"{label} is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 1 or metadata.st_size > maximum:
            raise ActionEffectQualificationError(f"{label} has an invalid file bound")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise ActionEffectQualificationError(f"{label} changed while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ActionEffectQualificationError(f"{label} grew while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _reject_symlink_chain(path: Path) -> None:
    current = path.expanduser().absolute()
    while True:
        if current.is_symlink():
            raise ActionEffectQualificationError(f"qualification path contains a symlink: {path}")
        parent = current.parent
        if parent == current:
            return
        current = parent


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    runner: Runner = subprocess.run,
) -> str:
    try:
        result = runner(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ActionEffectQualificationError(
            f"cannot verify v0.3 source with git {' '.join(arguments)}"
        ) from error
    return result.stdout.rstrip("\n")


def _storage_caps() -> dict[str, int]:
    return {
        "per_arm_bytes": v03.LINEAGE_CAP_BYTES,
        "scientific_cohort_bytes": v03.COHORT_SCIENTIFIC_CAP_BYTES,
        "media_bytes": v03.MEDIA_CAP_BYTES,
        "combined_bytes": v03.COMBINED_PLANNED_CAP_BYTES,
    }


def _resume_rule() -> dict[str, Any]:
    return {
        "resume_supported": False,
        "interruption_disposition": "whole_stage_a_operationally_incomplete",
        "single_arm_continuation": False,
        "failed_attempts": [
            "v0.3-action-effect-stage-a-r1-20260724-attempt-0",
            "v0.3-action-effect-stage-a-r2-20260724-attempt-0",
        ],
        "replacement_attempt": ("v0.3-action-effect-stage-a-r3-20260724"),
        "replacement_requires_new_commit_tag_qualification_and_roots": True,
        "replacement_restarts_both_arms_from_confirmed_u1": True,
    }


def _protected_seed_partitions() -> list[dict[str, Any]]:
    protected = [
        partition.public_dict()
        for partition in SEED_PARTITIONS
        if (
            partition.role is U2SeedRole.SEALED_QUALIFICATION
            or "confirmation" in partition.role.value
            or partition.role is U2SeedRole.FINAL_TEST
        )
    ]
    if len(protected) != 10:
        raise ActionEffectQualificationError("v0.3 protected seed inventory changed")
    return protected


def _architecture_contract() -> dict[str, Any]:
    sham = v03.effective_config(ActionEffectMode.SHAM)
    candidate = v03.effective_config(ActionEffectMode.ACTION_EFFECT)
    for config in (sham, candidate):
        if (
            config["parent"]["checkpoint_sha256"] != v03.PARENT_CHECKPOINT_SHA256
            or config["parent"]["policy_tensor_sha256"] != v03.PARENT_POLICY_TENSOR_SHA256
            or config["parent"]["optimizer_state_sha256"] != v03.PARENT_OPTIMIZER_STATE_SHA256
            or config["optimization"]["changed_from_original_u2_control"] is not False
            or config["reward"]["changed_from_original_u2_control"] is not False
            or config["selection"]["development_checkpoint_reuse_authorized"] is not False
        ):
            raise ActionEffectQualificationError("v0.3 effective configuration changed")
    return {
        "arms": list(_ARMS),
        "same_dict_observation_space": True,
        "same_parameter_set_and_initialization": True,
        "image": {
            "key": IMAGE_KEY,
            "shape_hwc": [56, 56, 3],
            "dtype": "uint8",
        },
        "action_effect": {
            "key": ACTION_EFFECT_KEY,
            "shape": [ACTION_EFFECT_DIM],
            "dtype": "float32",
            "previous_action_one_hot": 7,
            "visible_outcome_one_hot": ["changed", "unchanged"],
            "reset": "all_zero",
            "sham": "always_zero",
        },
        "policy": v03.public_policy_contract(),
        "transplant": {
            "legacy_parameters_by_exact_name": True,
            "legacy_adam_moments_by_exact_parameter_name": True,
            "positional_optimizer_loading": False,
            "orthogonal_reinitialization": False,
            "sole_new_parameter": ("features_extractor.action_effect_encoder.weight"),
            "sole_new_parameter_shape": [512, ACTION_EFFECT_DIM],
            "sole_new_parameter_initialization": "exact_zero",
        },
        "equivalence": {
            "zero_context_features_bit_exact": True,
            "zero_context_actions_values_log_probabilities_bit_exact": True,
            "zero_context_recurrent_state_bit_exact": True,
            "first_rollout_digest_profile": (
                v03.FIRST_ROLLOUT_DIGEST_PROFILE
            ),
            "first_rollout_transitions": v03.ROLLOUT_TRANSITIONS,
            "behavioral_divergence_before_first_optimizer_phase": False,
        },
        "randomness": {
            "architecture_initialization_seed": (v03.ARCHITECTURE_INITIALIZATION_SEED),
            "algorithm_seed": v03.ALGORITHM_SEED,
            "worker_streams": list(v03.WORKER_STREAMS),
        },
        "budget": {
            "child_actions_per_arm": v03.CHILD_ACTION_BUDGET,
            "evaluation_interval": v03.EVALUATION_INTERVAL,
            "exam_count_per_arm": v03.EXAM_COUNT,
            "terminal_lifetime_actions": (v03.PARENT_LIFETIME_ACTIONS + v03.CHILD_ACTION_BUDGET),
            "terminal_optimizer_updates": (
                v03.PARENT_OPTIMIZER_UPDATES
                + (v03.CHILD_ACTION_BUDGET // v03.ROLLOUT_TRANSITIONS) * v03.PPO_EPOCHS
            ),
        },
        "selection": {
            "sham_selectable": False,
            "action_effect_only_selectable_definition": True,
            "complete_u2s_terminal_three_gate": True,
            "development_checkpoint_reuse_authorized": False,
            "u3_authorized": False,
        },
        "sham_config_sha256": _canonical_sha256(sham),
        "action_effect_config_sha256": _canonical_sha256(candidate),
    }


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ActionEffectQualificationError(f"{label} escapes the U2-S root")
    path = (root / candidate).absolute()
    _reject_symlink_chain(path)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise ActionEffectQualificationError(f"{label} is missing or unsafe") from error
    if not resolved.is_file():
        raise ActionEffectQualificationError(f"{label} is missing or unsafe")
    return resolved


def _regular_tree_inventory(root: Path, label: str) -> tuple[set[str], set[str]]:
    """Return regular relative files/directories without following links."""

    if root.is_symlink() or not root.is_dir():
        raise ActionEffectQualificationError(f"{label} is missing or unsafe")
    files: set[str] = set()
    directories: set[str] = set()
    for current, child_directories, child_files in os.walk(
        root,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in child_directories:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise ActionEffectQualificationError(f"{label} contains an unsafe directory")
            directories.add(path.relative_to(root).as_posix())
        for name in child_files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ActionEffectQualificationError(f"{label} contains an unsafe file")
            files.add(path.relative_to(root).as_posix())
    return files, directories


def _stable_regular_sha256(path: Path, label: str) -> tuple[str, int]:
    """Hash one immutable regular file and reject a concurrent replacement."""

    _reject_symlink_chain(path)
    try:
        before = path.stat()
    except OSError as error:
        raise ActionEffectQualificationError(f"{label} is missing or unsafe") from error
    if not stat.S_ISREG(before.st_mode):
        raise ActionEffectQualificationError(f"{label} is missing or unsafe")
    digest = file_sha256(path)
    try:
        after = path.stat()
    except OSError as error:
        raise ActionEffectQualificationError(f"{label} changed while hashed") from error
    identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise ActionEffectQualificationError(f"{label} changed while hashed")
    return digest, int(after.st_size)


def _sealed_tree_inventory(
    root: Path,
    label: str,
) -> tuple[dict[str, str], dict[str, int | str]]:
    """Hash a complete immutable tree without retaining its path list in source."""

    files, directories = _regular_tree_inventory(root, label)
    measured: dict[str, str] = {}
    byte_count = 0
    for relative in sorted(files):
        digest, size = _stable_regular_sha256(
            root / relative,
            f"{label} file {relative}",
        )
        measured[relative] = digest
        byte_count += size
    directory_list = [".", *sorted(directories)]
    return measured, {
        "regular_file_count": len(measured),
        "regular_file_bytes": byte_count,
        "directory_count_including_root": len(directory_list),
        "directory_list_sha256": _canonical_sha256(directory_list),
        "regular_file_map_sha256": _canonical_sha256(measured),
    }


def _verify_failed_stage_a_tag(
    repository: Path,
    *,
    runner: Runner = subprocess.run,
) -> None:
    """Require the superseded r1 tag to remain exact on origin."""

    root = repository.expanduser().resolve()
    reference = f"refs/tags/{FAILED_STAGE_A_ATTEMPT_TAG}"
    if (
        _git(root, ["cat-file", "-t", reference], runner=runner) != "tag"
        or _git(root, ["rev-parse", "--verify", reference], runner=runner)
        != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or _git(root, ["rev-list", "-n", "1", reference], runner=runner)
        != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or _git(
            root,
            ["remote", "get-url", QUALIFIED_REMOTE],
            runner=runner,
        )
        != EXPECTED_ORIGIN_URL
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A attempt annotated tag identity changed"
        )
    remote = _git(
        root,
        [
            "ls-remote",
            "--tags",
            QUALIFIED_REMOTE,
            reference,
        ],
        runner=runner,
    ).split()
    if remote != [FAILED_STAGE_A_ATTEMPT_TAG_OBJECT, reference]:
        raise ActionEffectQualificationError(
            "failed Stage-A attempt annotated tag is not exact on origin"
        )


def _verify_failed_stage_a_r2_tag(
    repository: Path,
    *,
    runner: Runner = subprocess.run,
) -> None:
    """Require the superseded r2 tag to remain exact on origin."""

    root = repository.expanduser().resolve()
    reference = f"refs/tags/{FAILED_STAGE_A_R2_ATTEMPT_TAG}"
    if (
        _git(root, ["cat-file", "-t", reference], runner=runner) != "tag"
        or _git(root, ["rev-parse", "--verify", reference], runner=runner)
        != FAILED_STAGE_A_R2_ATTEMPT_TAG_OBJECT
        or _git(root, ["rev-list", "-n", "1", reference], runner=runner)
        != FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT
        or _git(
            root,
            ["remote", "get-url", QUALIFIED_REMOTE],
            runner=runner,
        )
        != EXPECTED_ORIGIN_URL
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A r2 annotated tag identity changed"
        )
    remote = _git(
        root,
        [
            "ls-remote",
            "--tags",
            QUALIFIED_REMOTE,
            reference,
        ],
        runner=runner,
    ).split()
    if remote != [FAILED_STAGE_A_R2_ATTEMPT_TAG_OBJECT, reference]:
        raise ActionEffectQualificationError(
            "failed Stage-A r2 annotated tag is not exact on origin"
        )


def authenticate_failed_stage_a_attempt(
    *,
    repository: Path | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Authenticate the immutable, partial-sham Stage-A r1 attempt."""

    if repository is not None:
        _verify_failed_stage_a_tag(repository, runner=runner)

    root = FAILED_STAGE_A_ATTEMPT_ROOT
    media_root = FAILED_STAGE_A_ATTEMPT_MEDIA_ROOT
    qualification_root = FAILED_STAGE_A_ATTEMPT_QUALIFICATION_DIRECTORY
    contract_path = root / "cohort-contract.json"
    cohort_path = root / "cohort.json"
    report_path = qualification_root / "report.json"
    claim_path = qualification_root / "claim.json"
    checksum_path = qualification_root / "report.json.sha256"

    root_files, root_directories = _regular_tree_inventory(
        root,
        "failed Stage-A attempt root",
    )
    media_files, media_directories = _regular_tree_inventory(
        media_root,
        "failed Stage-A attempt media root",
    )
    qualification_files, qualification_directories = _regular_tree_inventory(
        qualification_root,
        "failed Stage-A attempt qualification root",
    )
    _reject_symlink_chain(FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG)
    if (
        root_files != set(FAILED_STAGE_A_ATTEMPT_ROOT_FILE_SHA256)
        or root_directories != FAILED_STAGE_A_ATTEMPT_ROOT_DIRECTORIES
        or media_files
        or media_directories != {"sham"}
        or qualification_files != {"claim.json", "report.json", "report.json.sha256"}
        or qualification_directories
        or not FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG.is_file()
        or FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG.stat().st_size != 250
    ):
        raise ActionEffectQualificationError("failed Stage-A attempt artifact inventory changed")
    measured_root = {relative: file_sha256(root / relative) for relative in sorted(root_files)}
    measured_qualification = {
        "report.json": file_sha256(report_path),
        "claim.json": file_sha256(claim_path),
        "report.json.sha256": file_sha256(checksum_path),
    }
    expected_qualification = {
        "report.json": (FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256),
        "claim.json": (FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CLAIM_SHA256),
        "report.json.sha256": (FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256),
    }
    if (
        measured_root != FAILED_STAGE_A_ATTEMPT_ROOT_FILE_SHA256
        or measured_qualification != expected_qualification
        or file_sha256(FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG)
        != FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG_SHA256
    ):
        raise ActionEffectQualificationError("failed Stage-A attempt checksum changed")

    contract = _read_json(contract_path, "failed Stage-A cohort contract")
    cohort = _read_json(cohort_path, "failed Stage-A cohort state")
    report = _read_json(report_path, "failed Stage-A qualification report")
    claim = _read_json(claim_path, "failed Stage-A qualification claim")
    status = _read_json(root / "sham/status.json", "failed Stage-A status")
    supervisor = _read_json(
        root / "launcher-sham-attempt-0.json",
        "failed Stage-A supervisor state",
    )
    first_rollout = _read_json(
        root / "sham/first-rollout.json",
        "failed Stage-A first rollout",
    )
    first_exam_integrity = _read_json(
        root / "sham/checkpoints/rolling/exam-0032768.integrity.json",
        "failed Stage-A first exam integrity",
    )
    try:
        checksum_fields = checksum_path.read_text(encoding="ascii").split()
        launcher_lines = FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG.read_text(
            encoding="utf-8"
        ).splitlines()
        episode_lines = (root / "sham/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        evaluation_lines = (
            (root / "sham/evaluations.jsonl").read_text(encoding="utf-8").splitlines()
        )
        optimizer_lines = (root / "sham/optimizer.jsonl").read_text(encoding="utf-8").splitlines()
        last_episode = json.loads(episode_lines[-1])
    except (
        IndexError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise ActionEffectQualificationError(
            "failed Stage-A text evidence is unreadable"
        ) from error

    arms = cohort.get("arms")
    sham = arms[0] if isinstance(arms, list) and len(arms) == 2 else None
    candidate = arms[1] if isinstance(arms, list) and len(arms) == 2 else None
    sham_attempts = sham.get("attempts") if isinstance(sham, Mapping) else None
    sham_attempt = (
        sham_attempts[0] if isinstance(sham_attempts, list) and len(sham_attempts) == 1 else None
    )
    roots = contract.get("roots")
    preregistration = contract.get("preregistration")
    qualification = contract.get("qualification")
    replacement = contract.get("replacement")
    report_source = report.get("source")
    report_predecessor = report.get("failed_stage_a_attempt")
    latest_evaluated = status.get("latest_evaluated_checkpoint")
    if (
        checksum_fields
        != [
            FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256,
            "report.json",
        ]
        or report.get("protocol") != PROTOCOL
        or report.get("verdict") != VERDICT
        or report.get("claim") != claim
        or not isinstance(report_source, Mapping)
        or report_source.get("commit") != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or report_source.get("tag") != FAILED_STAGE_A_ATTEMPT_TAG
        or report_source.get("tag_object") != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or report_source.get("tag_payload_sha256") != FAILED_STAGE_A_ATTEMPT_TAG_PAYLOAD_SHA256
        or not isinstance(report_predecessor, Mapping)
        or _canonical_sha256(report_predecessor) != FAILED_STAGE_A_ATTEMPT_PREDECESSOR_SHA256
        or report_predecessor.get("disposition") != "operationally_incomplete"
        or report_predecessor.get("resume_authorized") is not False
        or report_predecessor.get("reuse_authorized") is not False
        or claim.get("source_commit") != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or claim.get("tag") != FAILED_STAGE_A_ATTEMPT_TAG
        or claim.get("tag_object") != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or contract.get("protocol") != PROTOCOL
        or contract.get("cohort_id") != "v0.3-action-effect-stage-a-r1-20260724"
        or contract.get("source")
        != {"commit": FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT, "dirty": False}
        or not isinstance(preregistration, Mapping)
        or preregistration.get("tag") != FAILED_STAGE_A_ATTEMPT_TAG
        or preregistration.get("tag_object") != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or preregistration.get("peeled_commit") != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or not isinstance(qualification, Mapping)
        or qualification.get("report_sha256") != FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256
        or qualification.get("tag_payload_sha256") != FAILED_STAGE_A_ATTEMPT_TAG_PAYLOAD_SHA256
        or qualification.get("failed_stage_a_attempt_sha256")
        != FAILED_STAGE_A_ATTEMPT_PREDECESSOR_SHA256
        or not isinstance(roots, Mapping)
        or roots.get("cohort") != str(root)
        or roots.get("media") != str(media_root)
        or not isinstance(replacement, Mapping)
        or replacement.get("failed_attempt_evidence_sha256")
        != FAILED_STAGE_A_ATTEMPT_PREDECESSOR_SHA256
        or replacement.get("failed_attempt_resume_authorized") is not False
        or replacement.get("failed_attempt_root_reuse_authorized") is not False
        or replacement.get("restarts_both_arms_from_confirmed_u1") is not True
        or cohort.get("protocol") != PROTOCOL
        or cohort.get("cohort_id") != "v0.3-action-effect-stage-a-r1-20260724"
        or cohort.get("contract_sha256") != FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256
        or cohort.get("source_commit") != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or cohort.get("tag") != FAILED_STAGE_A_ATTEMPT_TAG
        or cohort.get("tag_object") != FAILED_STAGE_A_ATTEMPT_TAG_OBJECT
        or cohort.get("qualification_sha256") != FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256
        or cohort.get("phase") != "operationally_incomplete"
        or cohort.get("active_arm") is not None
        or cohort.get("terminal_report") is not None
        or cohort.get("process_closeout") is not None
        or not isinstance(sham, Mapping)
        or sham.get("id") != "sham"
        or sham.get("state") != "crashed"
        or sham.get("terminal") is not None
        or not isinstance(sham_attempt, Mapping)
        or sham_attempt.get("index") != 0
        or sham_attempt.get("state") != "crashed"
        or sham_attempt.get("trainer_exit_code") != 1
        or not isinstance(candidate, Mapping)
        or candidate.get("id") != "action-effect"
        or candidate.get("state") != "pending"
        or candidate.get("attempts") != []
        or candidate.get("terminal") is not None
        or status.get("protocol") != PROTOCOL
        or status.get("cohort_id") != "v0.3-action-effect-stage-a-r1-20260724"
        or status.get("cohort_contract_sha256") != FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256
        or status.get("qualification_sha256") != FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256
        or status.get("arm") != "sham"
        or status.get("source", {}).get("commit") != FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT
        or status.get("source", {}).get("dirty") is not False
        or status.get("phase") != "training"
        or status.get("child_collected_actions") != 40_004
        or status.get("child_trained_actions") != 38_912
        or status.get("lifetime_collected_actions") != 826_436
        or status.get("lifetime_trained_actions") != 825_344
        or status.get("optimizer_updates") != 1_612
        or status.get("exam_count") != 1
        or status.get("exams_completed") != 1
        or status.get("initial_rng_identity_sha256") != FAILED_STAGE_A_ATTEMPT_INITIAL_RNG_SHA256
        or status.get("first_rollout_identity_sha256")
        != FAILED_STAGE_A_ATTEMPT_FIRST_ROLLOUT_SHA256
        or status.get("first_rollout_verified") is not True
        or status.get("latest_observed_checkpoint_sha256")
        != FAILED_STAGE_A_ATTEMPT_LATEST_OBSERVED_SHA256
        or status.get("latest_safe_checkpoint") is not None
        or status.get("latest_safe_checkpoint_sha256") is not None
        or status.get("report_sha256") is not None
        or status.get("resume_authorized") is not False
        or status.get("replacement_requires_both_fresh_arms") is not True
        or len(episode_lines) != 1_920
        or len(evaluation_lines) != 8
        or len(optimizer_lines) != 19
        or not isinstance(last_episode, Mapping)
        or last_episode.get("child_collected_actions") != 40_960
        or last_episode.get("child_trained_actions") != 38_912
        or not isinstance(latest_evaluated, Mapping)
        or latest_evaluated.get("child_trained_actions") != 32_768
        or latest_evaluated.get("checkpoint_sha256") != FAILED_STAGE_A_ATTEMPT_FIRST_EXAM_SHA256
        or latest_evaluated.get("counts_toward_architecture_gate") is not True
        or supervisor.get("state") != "exited"
        or supervisor.get("forwarded_signal") != "SIGTERM"
        or supervisor.get("exit_status") != -15
        or not isinstance(supervisor.get("supervisor_pid"), int)
        or not isinstance(supervisor.get("trainer_pid"), int)
        or first_rollout.get("arm") != "sham"
        or first_rollout.get("captured_before_first_optimizer") is not True
        or first_rollout.get("checkpoint_reuse_authorized") is not False
        or first_rollout.get("identity", {}).get("aggregate_sha256")
        != FAILED_STAGE_A_ATTEMPT_FIRST_ROLLOUT_SHA256
        or sum(
            int(item.get("transitions", -1))
            for item in first_rollout.get("identity", {}).get("trajectory_identity", [])
            if isinstance(item, Mapping)
        )
        != v03.ROLLOUT_TRANSITIONS
        or first_exam_integrity.get("protocol") != PROTOCOL
        or first_exam_integrity.get("checkpoint_sha256") != FAILED_STAGE_A_ATTEMPT_FIRST_EXAM_SHA256
        or first_exam_integrity.get("sidecar_sha256")
        != FAILED_STAGE_A_ATTEMPT_ROOT_FILE_SHA256["sham/checkpoints/rolling/exam-0032768.json"]
        or launcher_lines
        != [
            "v0.3 Stage-A dashboard: http://127.0.0.1:8789/",
            "Starting v0.3 Stage-A arm sham (fresh matched twin).",
            (
                "v0.3 sham artifacts: /Volumes/T7 Developer/"
                "DungeonApprentice/v03-action-effect-stage-a-r1-20260724/sham"
            ),
            "v0.3 does not have exactly one neural trainer",
        ]
    ):
        raise ActionEffectQualificationError("failed Stage-A attempt disposition changed")

    artifact_map_sha256 = _canonical_sha256(
        {
            "cohort": FAILED_STAGE_A_ATTEMPT_ROOT_FILE_SHA256,
            "qualification": expected_qualification,
            "launcher_log": {
                str(FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG): (
                    FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG_SHA256
                )
            },
        }
    )
    if artifact_map_sha256 != FAILED_STAGE_A_ATTEMPT_ARTIFACT_MAP_SHA256:
        raise ActionEffectQualificationError(
            "failed Stage-A attempt artifact-map identity changed"
        )
    return {
        "attempt_id": "v0.3-action-effect-stage-a-r1-20260724-attempt-0",
        "protocol": PROTOCOL,
        "disposition": "operationally_incomplete",
        "artifact_map": {
            "files": (
                len(FAILED_STAGE_A_ATTEMPT_ROOT_FILE_SHA256) + len(expected_qualification) + 1
            ),
            "sha256": artifact_map_sha256,
        },
        "source_commit": FAILED_STAGE_A_ATTEMPT_SOURCE_COMMIT,
        "tag": FAILED_STAGE_A_ATTEMPT_TAG,
        "tag_object": FAILED_STAGE_A_ATTEMPT_TAG_OBJECT,
        "qualification": {
            "root": str(qualification_root),
            "report_sha256": (FAILED_STAGE_A_ATTEMPT_QUALIFICATION_REPORT_SHA256),
            "claim_sha256": (FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CLAIM_SHA256),
            "checksum_sha256": (FAILED_STAGE_A_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256),
            "verdict": "qualified",
            "tag_payload_sha256": (FAILED_STAGE_A_ATTEMPT_TAG_PAYLOAD_SHA256),
            "predecessor_attempt_sha256": (FAILED_STAGE_A_ATTEMPT_PREDECESSOR_SHA256),
        },
        "cohort": {
            "root": str(root),
            "media_root": str(media_root),
            "contract_sha256": FAILED_STAGE_A_ATTEMPT_CONTRACT_SHA256,
            "state_sha256": FAILED_STAGE_A_ATTEMPT_COHORT_SHA256,
            "phase": "operationally_incomplete",
            "classification": "launcher_process_cardinality_false_positive",
            "manifest_arm_outcome": {
                "arm": "sham",
                "attempt": 0,
                "state": "crashed",
                "exit_code": 1,
            },
            "dashboard_log_sha256": (FAILED_STAGE_A_ATTEMPT_DASHBOARD_LOG_SHA256),
        },
        "recorded_training_evidence": {
            "arm_directories": 1,
            "media_entries": 0,
            "trainer_started": True,
            "trainer_status_files": 1,
            "trainer_supervisor_files": 1,
            "checkpoint_files": 11,
            "status_collected_actions": 40_004,
            "episode_ledger_collected_actions": 40_960,
            "recorded_child_actions": 38_912,
            "optimizer_updates_total": 1_612,
            "optimizer_updates_inherited": 1_536,
            "optimizer_updates_new": 76,
            "rollout_boundaries": 19,
            "frozen_exams": 1,
            "evaluation_rows": 8,
            "evaluation_cases": 640,
            "first_rollout_sha256": (FAILED_STAGE_A_ATTEMPT_FIRST_ROLLOUT_SHA256),
            "initial_rng_sha256": (FAILED_STAGE_A_ATTEMPT_INITIAL_RNG_SHA256),
            "first_exam_checkpoint_sha256": (FAILED_STAGE_A_ATTEMPT_FIRST_EXAM_SHA256),
            "latest_observed_checkpoint_sha256": (FAILED_STAGE_A_ATTEMPT_LATEST_OBSERVED_SHA256),
            "safe_checkpoint_exists": False,
            "action_effect_arm_started": False,
            "scientific_comparison_reached": False,
            "checkpoint_reuse_authorized": False,
        },
        "operational_failure": {
            "launcher_error": ("v0.3 does not have exactly one neural trainer"),
            "trainer_count_false_positive": ("supervisor argv embedded the child trainer command"),
            "launcher_exit_cleanup_ran": False,
            "worker_stop": {
                "supervisor_forwarded_signal": "SIGTERM",
                "worker_exit_status": -15,
                "status_remained_phase": "training",
            },
        },
        "launcher_log": {
            "path": str(FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG),
            "sha256": FAILED_STAGE_A_ATTEMPT_LAUNCHER_LOG_SHA256,
            "bytes": 250,
        },
        "resume_authorized": False,
        "reuse_authorized": False,
        "replacement": {
            "tag": QUALIFIED_TAG,
            "qualification_root": str(CANONICAL_QUALIFICATION_DIRECTORY),
            "cohort_root": str(CANONICAL_COHORT_ROOT),
            "media_root": str(CANONICAL_MEDIA_ROOT),
            "dashboard_port": DASHBOARD_PORT,
            "restarts_both_arms_from_confirmed_u1": True,
        },
    }


def authenticate_failed_stage_a_r2_attempt(
    *,
    repository: Path | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Authenticate the immutable r2 sham terminal and fail-closed disposition."""

    if repository is not None:
        _verify_failed_stage_a_r2_tag(repository, runner=runner)

    root = FAILED_STAGE_A_R2_ATTEMPT_ROOT
    media_root = FAILED_STAGE_A_R2_ATTEMPT_MEDIA_ROOT
    qualification_root = FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_DIRECTORY
    roots = {
        "cohort": root,
        "qualification": qualification_root,
        "media": media_root,
    }
    measured_maps: dict[str, dict[str, str]] = {}
    measured_seals: dict[str, dict[str, int | str]] = {}
    for name, path in roots.items():
        measured_maps[name], measured_seals[name] = _sealed_tree_inventory(
            path,
            f"failed Stage-A r2 {name} root",
        )

    launcher_digest, launcher_bytes = _stable_regular_sha256(
        FAILED_STAGE_A_R2_ATTEMPT_LAUNCHER_LOG,
        "failed Stage-A r2 launcher log",
    )
    screen_digest, screen_bytes = _stable_regular_sha256(
        FAILED_STAGE_A_R2_ATTEMPT_SCREEN_LOG,
        "failed Stage-A r2 screen log",
    )
    recovery_map = {
        FAILED_STAGE_A_R2_ATTEMPT_LAUNCHER_LOG.name: launcher_digest,
        FAILED_STAGE_A_R2_ATTEMPT_SCREEN_LOG.name: screen_digest,
    }
    measured_maps["launch-recovery"] = recovery_map
    measured_seals["launch-recovery"] = {
        "regular_file_count": len(recovery_map),
        "regular_file_bytes": launcher_bytes + screen_bytes,
        "regular_file_map_sha256": _canonical_sha256(recovery_map),
    }
    if measured_seals != FAILED_STAGE_A_R2_ATTEMPT_TREE_SEALS:
        raise ActionEffectQualificationError(
            "failed Stage-A r2 compact artifact inventory changed"
        )

    combined_files = {
        f"{namespace}/{relative}": digest
        for namespace, file_map in measured_maps.items()
        for relative, digest in file_map.items()
    }
    combined_bytes = sum(
        int(seal["regular_file_bytes"]) for seal in measured_seals.values()
    )
    if (
        len(combined_files) != FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_COUNT
        or combined_bytes != FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_BYTES
        or _canonical_sha256(combined_files)
        != FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_MAP_SHA256
        or {
            relative: combined_files.get(relative)
            for relative in FAILED_STAGE_A_R2_ATTEMPT_CRITICAL_SHA256
        }
        != FAILED_STAGE_A_R2_ATTEMPT_CRITICAL_SHA256
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A r2 compact artifact seal changed"
        )

    contract_path = root / "cohort-contract.json"
    cohort_path = root / "cohort.json"
    launcher_path = root / "launcher-sham-attempt-0.json"
    arm_root = root / "sham"
    status_path = arm_root / "status.json"
    first_rollout_path = arm_root / "first-rollout.json"
    report_path = arm_root / "report.json"
    report_integrity_path = arm_root / "report.integrity.json"
    terminal_path = arm_root / "checkpoints/terminal.zip"
    terminal_sidecar_path = terminal_path.with_suffix(".json")
    terminal_integrity_path = terminal_path.with_suffix(".integrity.json")
    qualification_report_path = qualification_root / "report.json"
    qualification_claim_path = qualification_root / "claim.json"
    qualification_checksum_path = qualification_root / "report.json.sha256"

    contract = _read_json(contract_path, "failed Stage-A r2 cohort contract")
    cohort = _read_json(cohort_path, "failed Stage-A r2 cohort state")
    launcher = _read_json(launcher_path, "failed Stage-A r2 launcher state")
    status = _read_json(status_path, "failed Stage-A r2 sham status")
    first_rollout = _read_json(
        first_rollout_path,
        "failed Stage-A r2 first rollout",
    )
    report = _read_json(report_path, "failed Stage-A r2 sham report")
    report_integrity = _read_json(
        report_integrity_path,
        "failed Stage-A r2 sham report integrity",
    )
    terminal_sidecar = _read_json(
        terminal_sidecar_path,
        "failed Stage-A r2 terminal sidecar",
    )
    terminal_integrity = _read_json(
        terminal_integrity_path,
        "failed Stage-A r2 terminal integrity",
    )
    qualification_report = _read_json(
        qualification_report_path,
        "failed Stage-A r2 qualification report",
    )
    qualification_claim = _read_json(
        qualification_claim_path,
        "failed Stage-A r2 qualification claim",
    )
    try:
        qualification_checksum = qualification_checksum_path.read_text(
            encoding="ascii"
        ).split()
        launcher_text = FAILED_STAGE_A_R2_ATTEMPT_LAUNCHER_LOG.read_text(
            encoding="utf-8"
        )
        screen_text = FAILED_STAGE_A_R2_ATTEMPT_SCREEN_LOG.read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError) as error:
        raise ActionEffectQualificationError(
            "failed Stage-A r2 text evidence is unreadable"
        ) from error

    arms = cohort.get("arms")
    sham = arms[0] if isinstance(arms, list) and len(arms) == 2 else None
    action_effect = arms[1] if isinstance(arms, list) and len(arms) == 2 else None
    sham_attempts = sham.get("attempts") if isinstance(sham, Mapping) else None
    sham_attempt = (
        sham_attempts[0]
        if isinstance(sham_attempts, list) and len(sham_attempts) == 1
        else None
    )
    qualification = contract.get("qualification")
    preregistration = contract.get("preregistration")
    replacement = contract.get("replacement")
    matched_design = contract.get("matched_design")
    report_source = qualification_report.get("source")
    r1_predecessor = qualification_report.get("failed_stage_a_attempt")
    history = cohort.get("history")
    last_history = history[-1] if isinstance(history, list) and history else None
    grade = report.get("grade")
    progress = report.get("progress")
    segment = status.get("segment")
    terminal_record = report.get("terminal_checkpoint")
    if (
        qualification_checksum
        != [FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_REPORT_SHA256, "report.json"]
        or qualification_report.get("protocol") != PROTOCOL
        or qualification_report.get("verdict") != VERDICT
        or qualification_report.get("claim") != qualification_claim
        or not isinstance(report_source, Mapping)
        or report_source.get("commit") != FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT
        or report_source.get("tag") != FAILED_STAGE_A_R2_ATTEMPT_TAG
        or report_source.get("tag_object") != FAILED_STAGE_A_R2_ATTEMPT_TAG_OBJECT
        or report_source.get("tag_payload_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_TAG_PAYLOAD_SHA256
        or not isinstance(r1_predecessor, Mapping)
        or _canonical_sha256(r1_predecessor)
        != FAILED_STAGE_A_R2_ATTEMPT_R1_PREDECESSOR_SHA256
        or r1_predecessor.get("resume_authorized") is not False
        or r1_predecessor.get("reuse_authorized") is not False
        or qualification_claim.get("source_commit")
        != FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT
        or qualification_claim.get("tag") != FAILED_STAGE_A_R2_ATTEMPT_TAG
        or qualification_claim.get("tag_object")
        != FAILED_STAGE_A_R2_ATTEMPT_TAG_OBJECT
        or contract.get("protocol") != PROTOCOL
        or contract.get("cohort_id") != "v0.3-action-effect-stage-a-r2-20260724"
        or contract.get("source")
        != {"commit": FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT, "dirty": False}
        or not isinstance(preregistration, Mapping)
        or preregistration.get("tag") != FAILED_STAGE_A_R2_ATTEMPT_TAG
        or preregistration.get("tag_object") != FAILED_STAGE_A_R2_ATTEMPT_TAG_OBJECT
        or preregistration.get("peeled_commit")
        != FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT
        or not isinstance(qualification, Mapping)
        or qualification.get("report_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_REPORT_SHA256
        or qualification.get("tag_payload_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_TAG_PAYLOAD_SHA256
        or qualification.get("failed_stage_a_attempt_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_R1_PREDECESSOR_SHA256
        or contract.get("roots")
        != {"cohort": str(root), "media": str(media_root)}
        or not isinstance(replacement, Mapping)
        or replacement.get("failed_attempt_evidence_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_R1_PREDECESSOR_SHA256
        or replacement.get("failed_attempt_resume_authorized") is not False
        or replacement.get("failed_attempt_root_reuse_authorized") is not False
        or replacement.get("restarts_both_arms_from_confirmed_u1") is not True
        or not isinstance(matched_design, Mapping)
        or matched_design.get("checkpoint_promotable") is not False
        or matched_design.get("u3_authorized") is not False
        or matched_design.get("resumable") is not False
        or cohort.get("protocol") != PROTOCOL
        or cohort.get("cohort_id") != "v0.3-action-effect-stage-a-r2-20260724"
        or cohort.get("contract_sha256") != FAILED_STAGE_A_R2_ATTEMPT_CONTRACT_SHA256
        or cohort.get("source_commit") != FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT
        or cohort.get("tag") != FAILED_STAGE_A_R2_ATTEMPT_TAG
        or cohort.get("tag_object") != FAILED_STAGE_A_R2_ATTEMPT_TAG_OBJECT
        or cohort.get("qualification_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_REPORT_SHA256
        or cohort.get("phase") != "integrity_failed"
        or cohort.get("active_arm") is not None
        or cohort.get("terminal_report") is not None
        or cohort.get("process_closeout") is not None
        or last_history
        != {
            "arm": "sham",
            "at": "2026-07-24T07:50:00+00:00",
            "event": "integrity_failed",
            "stage": "arm_terminal_closeout",
        }
        or not isinstance(sham, Mapping)
        or sham.get("id") != "sham"
        or sham.get("state") != "integrity_failed"
        or sham.get("terminal") is not None
        or not isinstance(sham_attempt, Mapping)
        or sham_attempt.get("index") != 0
        or sham_attempt.get("state") != "integrity_failed"
        or not isinstance(action_effect, Mapping)
        or action_effect.get("id") != "action-effect"
        or action_effect.get("state") != "pending"
        or action_effect.get("attempts") != []
        or action_effect.get("terminal") is not None
        or (root / "action-effect").exists()
        or (media_root / "action-effect").exists()
        or launcher.get("state") != "exited"
        or launcher.get("forwarded_signal") is not None
        or launcher.get("exit_status") != 0
        or not isinstance(launcher.get("supervisor_pid"), int)
        or not isinstance(launcher.get("trainer_pid"), int)
        or status.get("protocol") != PROTOCOL
        or status.get("cohort_id") != "v0.3-action-effect-stage-a-r2-20260724"
        or status.get("cohort_contract_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_CONTRACT_SHA256
        or status.get("qualification_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_REPORT_SHA256
        or status.get("source", {}).get("commit")
        != FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT
        or status.get("source", {}).get("dirty") is not False
        or status.get("arm") != "sham"
        or status.get("phase") != "completed"
        or status.get("child_collected_actions") != 1_048_576
        or status.get("child_trained_actions") != 1_048_576
        or status.get("remaining_action_budget") != 0
        or status.get("lifetime_trained_actions") != 1_835_008
        or status.get("optimizer_updates") != 3_584
        or status.get("exam_count") != 32
        or status.get("exams_completed") != 32
        or status.get("initial_rng_identity_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_INITIAL_RNG_SHA256
        or status.get("first_rollout_identity_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_SHA256
        or status.get("first_rollout_verified") is not True
        or status.get("report_sha256") != FAILED_STAGE_A_R2_ATTEMPT_REPORT_SHA256
        or status.get("resume_authorized") is not False
        or status.get("promotable") is not False
        or status.get("latest_safe_checkpoint") is not None
        or not isinstance(segment, Mapping)
        or segment.get("resume_authorized") is not False
        or segment.get("resume_checkpoint") is not None
        or not isinstance(progress, Mapping)
        or progress.get("child_trained_actions") != 1_048_576
        or progress.get("lifetime_trained_actions") != 1_835_008
        or progress.get("optimizer_updates") != 3_584
        or progress.get("exam_count") != 32
        or report.get("case_count") != 10_240
        or report.get("development_checkpoint_reuse_authorized") is not False
        or report.get("resume_authorized") is not False
        or report.get("promotable") is not False
        or report.get("successor_checkpoint_authorized") is not False
        or report.get("u3_authorized") is not False
        or not isinstance(terminal_record, Mapping)
        or terminal_record.get("path") != "checkpoints/terminal.zip"
        or terminal_record.get("sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_CHECKPOINT_SHA256
        or terminal_record.get("sidecar_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_SIDECAR_SHA256
        or terminal_record.get("integrity_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_INTEGRITY_SHA256
        or terminal_record.get("promotable") is not False
        or not isinstance(grade, Mapping)
        or grade.get("eligible") is not False
        or grade.get("final_exam_actions") != [983_040, 1_015_808, 1_048_576]
        or grade.get("reasons")
        != [
            "terminal_2:cases:max_ineffective_below_10",
            "terminal_2:cases:no_ineffective_tail",
            "terminal_2:cases:repeated_identical_interaction_run_below_10",
        ]
        or report_integrity.get("report_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_REPORT_SHA256
        or report_integrity.get("terminal_checkpoint_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_CHECKPOINT_SHA256
        or report_integrity.get("terminal_sidecar_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_SIDECAR_SHA256
        or report_integrity.get("terminal_integrity_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_INTEGRITY_SHA256
        or terminal_integrity.get("checkpoint_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_CHECKPOINT_SHA256
        or terminal_integrity.get("sidecar_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_SIDECAR_SHA256
        or terminal_sidecar.get("checkpoint_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_CHECKPOINT_SHA256
        or terminal_sidecar.get("resume_authorized") is not False
        or terminal_sidecar.get("resume_eligible") is not False
        or terminal_sidecar.get("development_checkpoint_reuse_authorized") is not False
        or terminal_sidecar.get("promotable") is not False
        or launcher_text != ""
        or "V03ManifestError: sham first-rollout boundary is unauthenticated"
        not in screen_text
        or "V03ManifestError: sham terminal evidence failed closed" not in screen_text
        or "Starting v0.3 Stage-A arm action-effect" in screen_text
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A r2 disposition changed"
        )

    first_identity = report.get("first_rollout_identity")
    if not isinstance(first_identity, Mapping) or first_rollout.get("identity") != first_identity:
        raise ActionEffectQualificationError(
            "failed Stage-A r2 first-rollout evidence changed"
        )
    first_payload = {
        key: value for key, value in first_identity.items() if key != "aggregate_sha256"
    }
    with_lf_sha256 = v03.first_rollout_identity_sha256(first_payload)
    without_lf_sha256 = _canonical_sha256(first_payload)
    if (
        first_identity.get("aggregate_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_SHA256
        or first_rollout.get("qualification_identity_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_SHA256
        or with_lf_sha256 != FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_SHA256
        or without_lf_sha256
        != FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_NO_LF_SHA256
        or with_lf_sha256 == without_lf_sha256
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A r2 LF canonicalization proof changed"
        )

    try:
        from dungeon_apprentice.v03_action_effect_train import (
            verify_arm_terminal_report,
        )

        verified = verify_arm_terminal_report(
            arm_root,
            expected_arm="sham",
            expected_source_commit=FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT,
            expected_cohort_id="v0.3-action-effect-stage-a-r2-20260724",
            expected_contract_sha256=FAILED_STAGE_A_R2_ATTEMPT_CONTRACT_SHA256,
            expected_qualification_sha256=(
                FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_REPORT_SHA256
            ),
            authenticate_frozen_r2_legacy_envelope=True,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ActionEffectQualificationError(
            "failed Stage-A r2 trainer verifier rejected terminal evidence"
        ) from error
    exam_records = verified.get("exam_records")
    if (
        verified.get("status_sha256") != FAILED_STAGE_A_R2_ATTEMPT_STATUS_SHA256
        or verified.get("report_sha256") != FAILED_STAGE_A_R2_ATTEMPT_REPORT_SHA256
        or verified.get("report_integrity_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_REPORT_INTEGRITY_SHA256
        or verified.get("terminal_checkpoint_sha256")
        != FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_CHECKPOINT_SHA256
        or verified.get("eligible") is not False
        or verified.get("case_count") != 10_240
        or not isinstance(exam_records, list)
        or len(exam_records) != 32
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A r2 trainer verification summary changed"
        )

    evaluations = status.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 32:
        raise ActionEffectQualificationError(
            "failed Stage-A r2 scheduled exam inventory changed"
        )
    exam_index: list[dict[str, Any]] = []
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ActionEffectQualificationError(
                "failed Stage-A r2 scheduled exam inventory changed"
            )
        checkpoint = str(evaluation.get("checkpoint", ""))
        checkpoint_path = Path(checkpoint)
        sidecar = checkpoint_path.with_suffix(".json").as_posix()
        integrity = checkpoint_path.with_suffix(".integrity.json").as_posix()
        cases_value = evaluation.get("case_diagnostics")
        if not isinstance(cases_value, Mapping):
            raise ActionEffectQualificationError(
                "failed Stage-A r2 scheduled exam cases changed"
            )
        cases = str(cases_value.get("path", ""))
        exam_index.append(
            {
                "boundary": evaluation.get("child_trained_actions"),
                "zip": checkpoint,
                "zip_sha256": measured_maps["cohort"].get(f"sham/{checkpoint}"),
                "sidecar": sidecar,
                "sidecar_sha256": measured_maps["cohort"].get(f"sham/{sidecar}"),
                "integrity": integrity,
                "integrity_sha256": measured_maps["cohort"].get(
                    f"sham/{integrity}"
                ),
                "cases": cases,
                "cases_sha256": measured_maps["cohort"].get(f"sham/{cases}"),
                "case_count": cases_value.get("case_count"),
            }
        )
    if (
        [record["boundary"] for record in exam_index]
        != list(range(32_768, 1_048_576 + 1, 32_768))
        or sum(int(record["case_count"]) for record in exam_index) != 10_240
        or _canonical_sha256(exam_index)
        != FAILED_STAGE_A_R2_ATTEMPT_EXAM_INDEX_SHA256
    ):
        raise ActionEffectQualificationError(
            "failed Stage-A r2 scheduled exam seal changed"
        )

    no_reuse = {
        "contract_checkpoint_promotable": matched_design["checkpoint_promotable"],
        "contract_u3_authorized": matched_design["u3_authorized"],
        "contract_resume_authorized": replacement["failed_attempt_resume_authorized"],
        "contract_root_reuse_authorized": (
            replacement["failed_attempt_root_reuse_authorized"]
        ),
        "status_resume_authorized": status["resume_authorized"],
        "status_promotable": status["promotable"],
        "segment_resume_authorized": segment["resume_authorized"],
        "segment_resume_checkpoint": segment["resume_checkpoint"],
        "first_rollout_checkpoint_reuse_authorized": (
            first_rollout["checkpoint_reuse_authorized"]
        ),
        "report_development_checkpoint_reuse_authorized": (
            report["development_checkpoint_reuse_authorized"]
        ),
        "report_resume_authorized": report["resume_authorized"],
        "report_promotable": report["promotable"],
        "report_successor_checkpoint_authorized": (
            report["successor_checkpoint_authorized"]
        ),
        "report_u3_authorized": report["u3_authorized"],
        "terminal_development_checkpoint_reuse_authorized": (
            terminal_sidecar["development_checkpoint_reuse_authorized"]
        ),
        "terminal_resume_authorized": terminal_sidecar["resume_authorized"],
        "terminal_resume_eligible": terminal_sidecar["resume_eligible"],
        "terminal_promotable": terminal_sidecar["promotable"],
    }
    expected_no_reuse = {
        key: False for key in no_reuse if key != "segment_resume_checkpoint"
    }
    expected_no_reuse["segment_resume_checkpoint"] = None
    if no_reuse != expected_no_reuse:
        raise ActionEffectQualificationError(
            "failed Stage-A r2 no-reuse disposition changed"
        )

    return {
        "attempt_id": "v0.3-action-effect-stage-a-r2-20260724-attempt-0",
        "protocol": PROTOCOL,
        "disposition": "integrity_failed",
        "source_commit": FAILED_STAGE_A_R2_ATTEMPT_SOURCE_COMMIT,
        "tag": FAILED_STAGE_A_R2_ATTEMPT_TAG,
        "tag_object": FAILED_STAGE_A_R2_ATTEMPT_TAG_OBJECT,
        "artifact_seal": {
            "regular_files": FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_COUNT,
            "regular_file_bytes": FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_BYTES,
            "relative_path_sha256_map_digest": (
                FAILED_STAGE_A_R2_ATTEMPT_COMBINED_FILE_MAP_SHA256
            ),
            "trees": FAILED_STAGE_A_R2_ATTEMPT_TREE_SEALS,
            "critical_artifacts": FAILED_STAGE_A_R2_ATTEMPT_CRITICAL_SHA256,
        },
        "qualification": {
            "root": str(qualification_root),
            "report_sha256": FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_REPORT_SHA256,
            "claim_sha256": FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_CLAIM_SHA256,
            "checksum_sha256": (
                FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_CHECKSUM_SHA256
            ),
            "verdict": "qualified",
            "tag_payload_sha256": (
                FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_TAG_PAYLOAD_SHA256
            ),
            "r1_predecessor_sha256": (
                FAILED_STAGE_A_R2_ATTEMPT_R1_PREDECESSOR_SHA256
            ),
        },
        "cohort": {
            "root": str(root),
            "media_root": str(media_root),
            "contract_sha256": FAILED_STAGE_A_R2_ATTEMPT_CONTRACT_SHA256,
            "state_sha256": FAILED_STAGE_A_R2_ATTEMPT_COHORT_SHA256,
            "phase": "integrity_failed",
            "active_arm": None,
            "sham_state": "integrity_failed",
            "action_effect_state": "pending",
            "action_effect_attempts": 0,
            "action_effect_started": False,
        },
        "sham_terminal": {
            "child_trained_actions": 1_048_576,
            "lifetime_trained_actions": 1_835_008,
            "optimizer_updates": 3_584,
            "exam_count": 32,
            "case_count": 10_240,
            "exam_index_sha256": FAILED_STAGE_A_R2_ATTEMPT_EXAM_INDEX_SHA256,
            "status_sha256": FAILED_STAGE_A_R2_ATTEMPT_STATUS_SHA256,
            "report_sha256": FAILED_STAGE_A_R2_ATTEMPT_REPORT_SHA256,
            "report_integrity_sha256": (
                FAILED_STAGE_A_R2_ATTEMPT_REPORT_INTEGRITY_SHA256
            ),
            "terminal_checkpoint_sha256": (
                FAILED_STAGE_A_R2_ATTEMPT_TERMINAL_CHECKPOINT_SHA256
            ),
            "trainer_verifier_passed": True,
            "scientifically_eligible": False,
            "final_exam_actions": [983_040, 1_015_808, 1_048_576],
            "failure_reasons": list(grade["reasons"]),
        },
        "operational_failure": {
            "stage": "arm_terminal_closeout",
            "manifest_error": "sham first-rollout boundary is unauthenticated",
            "cohort_error": "sham terminal evidence failed closed",
            "first_rollout": {
                "stored_sha256": FAILED_STAGE_A_R2_ATTEMPT_FIRST_ROLLOUT_SHA256,
                "canonical_json_with_lf_sha256": with_lf_sha256,
                "canonical_json_without_lf_sha256": without_lf_sha256,
                "stored_matches_qualification_primitive": True,
                "stored_matches_manifest_without_lf": False,
            },
            "launcher_log": {
                "path": str(FAILED_STAGE_A_R2_ATTEMPT_LAUNCHER_LOG),
                "sha256": launcher_digest,
                "bytes": launcher_bytes,
            },
            "screen_log": {
                "path": str(FAILED_STAGE_A_R2_ATTEMPT_SCREEN_LOG),
                "sha256": screen_digest,
                "bytes": screen_bytes,
            },
        },
        "resume_authorized": False,
        "reuse_authorized": False,
        "no_reuse_evidence": no_reuse,
        "replacement": {
            "tag": QUALIFIED_TAG,
            "qualification_root": str(CANONICAL_QUALIFICATION_DIRECTORY),
            "cohort_root": str(CANONICAL_COHORT_ROOT),
            "media_root": str(CANONICAL_MEDIA_ROOT),
            "dashboard_port": DASHBOARD_PORT,
            "restarts_both_arms_from_confirmed_u1": True,
        },
    }


def authenticate_u2s_terminal(
    *,
    root: Path = U2S_TERMINAL_ROOT,
    expected_report_sha256: str = U2S_TERMINAL_REPORT_SHA256,
    expected_integrity_sha256: str = U2S_TERMINAL_INTEGRITY_SHA256,
) -> dict[str, Any]:
    """Authenticate terminal U2-S outcome and its four arm report bindings."""

    terminal_root = root.expanduser().resolve()
    report_path = terminal_root / "report.json"
    integrity_path = terminal_root / "report.integrity.json"
    if (
        file_sha256(report_path) != expected_report_sha256
        or file_sha256(integrity_path) != expected_integrity_sha256
    ):
        raise ActionEffectQualificationError("terminal U2-S report bytes changed")
    report = _read_json(report_path, "terminal U2-S report")
    integrity = _read_json(integrity_path, "terminal U2-S report integrity")
    selection = report.get("selection")
    checkpoint_rule = report.get("checkpoint_rule")
    if (
        report.get("protocol") != frozen_u2s.PROTOCOL
        or report.get("verdict") != "ablation_failed"
        or report.get("selected_configuration") is not None
        or report.get("source", {}).get("commit") != U2S_TERMINAL_SOURCE_COMMIT
        or not isinstance(selection, Mapping)
        or selection.get("verdict") != "ablation_failed"
        or selection.get("selected_arm") is not None
        or selection.get("selected_mechanism") is not None
        or selection.get("successor_cohort_authorized") is not False
        or selection.get("ablation_checkpoint_reuse_authorized") is not False
        or not isinstance(checkpoint_rule, Mapping)
        or checkpoint_rule.get("ablation_checkpoint_reuse_authorized") is not False
        or checkpoint_rule.get("successor_checkpoint") is not None
        or checkpoint_rule.get("successor_cohort_authorized") is not False
        or integrity.get("protocol") != frozen_u2s.PROTOCOL
        or integrity.get("report") != report_path.name
        or integrity.get("report_sha256") != expected_report_sha256
    ):
        raise ActionEffectQualificationError("terminal U2-S no-selection verdict changed")
    qualification = report.get("qualification")
    arm_evidence = report.get("arm_evidence")
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("report_sha256") != U2S_QUALIFICATION_REPORT_SHA256
        or not isinstance(arm_evidence, Mapping)
        or set(arm_evidence) != set(frozen_u2s.ARM_PRIORITY)
    ):
        raise ActionEffectQualificationError("terminal U2-S qualification or arm inventory changed")
    arms: dict[str, Any] = {}
    for arm in frozen_u2s.ARM_PRIORITY:
        name = arm.value
        binding = arm_evidence.get(name)
        if not isinstance(binding, Mapping):
            raise ActionEffectQualificationError(f"U2-S arm is missing: {name}")
        arm_report_path = _safe_child(
            terminal_root,
            binding.get("report"),
            f"U2-S {name} report",
        )
        arm_integrity_path = _safe_child(
            terminal_root,
            binding.get("report_integrity"),
            f"U2-S {name} report integrity",
        )
        if (
            file_sha256(arm_report_path) != binding.get("report_sha256")
            or file_sha256(arm_integrity_path) != binding.get("report_integrity_sha256")
            or int(binding.get("child_trained_actions", -1)) != frozen_u2s.CHILD_ACTION_BUDGET
            or int(binding.get("exam_count", -1)) != frozen_u2s.EXAM_COUNT
            or binding.get("mechanism_selection_eligible") is not False
            or binding.get("verdict") != "arm_failed"
        ):
            raise ActionEffectQualificationError(f"terminal U2-S arm binding changed: {name}")
        arm_report = _read_json(arm_report_path, f"U2-S {name} report")
        arm_integrity = _read_json(
            arm_integrity_path,
            f"U2-S {name} report integrity",
        )
        if (
            arm_report.get("arm") != name
            or arm_report.get("successor_checkpoint_authorized") is not False
            or arm_report.get("resume_authorized") is not False
            or arm_report.get("progress", {}).get("child_trained_actions")
            != frozen_u2s.CHILD_ACTION_BUDGET
            or arm_report.get("progress", {}).get("exam_count") != frozen_u2s.EXAM_COUNT
            or arm_integrity.get("report_sha256") != binding.get("report_sha256")
        ):
            raise ActionEffectQualificationError(f"terminal U2-S arm report changed: {name}")
        arms[name] = {
            "report": str(arm_report_path),
            "report_sha256": str(binding["report_sha256"]),
            "report_integrity": str(arm_integrity_path),
            "report_integrity_sha256": str(binding["report_integrity_sha256"]),
            "child_trained_actions": frozen_u2s.CHILD_ACTION_BUDGET,
            "exam_count": frozen_u2s.EXAM_COUNT,
        }
    return {
        "root": str(terminal_root),
        "report": str(report_path),
        "report_sha256": expected_report_sha256,
        "report_integrity": str(integrity_path),
        "report_integrity_sha256": expected_integrity_sha256,
        "source_commit": U2S_TERMINAL_SOURCE_COMMIT,
        "verdict": "ablation_failed",
        "selected_configuration": None,
        "successor_cohort_authorized": False,
        "checkpoint_reuse_authorized": False,
        "qualification_report_sha256": U2S_QUALIFICATION_REPORT_SHA256,
        "arms": arms,
        "_report": report,
    }


def extend_action_effect_forbidden_layout_hashes(
    base_mapping: Mapping[lessons.LessonId, frozenset[str]],
    terminal_report: Mapping[str, Any],
    *,
    root: Path = U2S_TERMINAL_ROOT,
) -> tuple[Mapping[lessons.LessonId, frozenset[str]], dict[str, Any]]:
    """Add authenticated U2-S training identities to one static Stage-A guard."""

    if set(base_mapping) != set(lessons.LessonId):
        raise ActionEffectQualificationError("base guard inventory is incomplete")
    arm_evidence = terminal_report.get("arm_evidence")
    if not isinstance(arm_evidence, Mapping) or set(arm_evidence) != set(frozen_u2s.ARM_PRIORITY):
        raise ActionEffectQualificationError("U2-S arm inventory is incomplete")
    by_lesson: dict[lessons.LessonId, set[str]] = {lesson: set() for lesson in lessons.LessonId}
    arm_records: dict[str, dict[str, Any]] = {}
    for arm in frozen_u2s.ARM_PRIORITY:
        name = arm.value
        binding = arm_evidence[name]
        arm_report_path = _safe_child(root, binding["report"], f"{name} report")
        if file_sha256(arm_report_path) != binding["report_sha256"]:
            raise ActionEffectQualificationError(f"{name} report digest changed")
        arm_report = _read_json(arm_report_path, f"{name} report")
        training = arm_report.get("training_episode_evidence")
        ledger_binding = training.get("episode_starts") if isinstance(training, Mapping) else None
        if not isinstance(ledger_binding, Mapping):
            raise ActionEffectQualificationError(f"{name} ledger binding missing")
        ledger_path = _safe_child(
            arm_report_path.parent,
            ledger_binding.get("path"),
            f"{name} episode-start ledger",
        )
        if file_sha256(ledger_path) != ledger_binding.get("sha256"):
            raise ActionEffectQualificationError(f"{name} ledger digest changed")
        records = 0
        try:
            with ledger_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    value = json.loads(line)
                    if not isinstance(value, Mapping) or value.get("type") != "episode_start":
                        raise ActionEffectQualificationError(
                            f"{name} ledger contains an invalid record"
                        )
                    try:
                        lesson = lessons.LessonId(str(value.get("lesson_id")))
                    except ValueError as error:
                        raise ActionEffectQualificationError(
                            f"{name} ledger contains an unknown lesson"
                        ) from error
                    by_lesson[lesson].add(
                        _require_sha256(
                            value.get("layout_sha256"),
                            f"{name} layout",
                        )
                    )
                    records += 1
        except (OSError, json.JSONDecodeError) as error:
            raise ActionEffectQualificationError(
                f"cannot parse {name} episode-start ledger"
            ) from error
        if records != int(ledger_binding.get("record_count", -1)):
            raise ActionEffectQualificationError(f"{name} ledger count changed")
        active = training.get("terminal_active_workers")
        if (
            not isinstance(active, list)
            or len(active) != v03.WORKERS
            or _canonical_sha256(active) != training.get("terminal_active_workers_sha256")
        ):
            raise ActionEffectQualificationError(f"{name} terminal-active inventory changed")
        for value in active:
            try:
                lesson = lessons.LessonId(str(value.get("lesson_id")))
            except (AttributeError, ValueError) as error:
                raise ActionEffectQualificationError(
                    f"{name} terminal-active lesson changed"
                ) from error
            by_lesson[lesson].add(
                _require_sha256(value.get("layout_sha256"), f"{name} active layout")
            )
        arm_records[name] = {
            "report_sha256": str(binding["report_sha256"]),
            "episode_start_ledger": str(ledger_path),
            "episode_start_ledger_sha256": str(ledger_binding["sha256"]),
            "episode_start_records": records,
            "terminal_active_workers": len(active),
            "terminal_active_workers_sha256": str(training["terminal_active_workers_sha256"]),
        }

    applied: dict[lessons.LessonId, frozenset[str]] = {}
    guard_sets: dict[str, dict[str, Any]] = {}
    for lesson in lessons.LessonId:
        values = set(base_mapping[lesson])
        if lesson is not lessons.LessonId.VISIBLE_UNLOCK:
            values.update(by_lesson[lesson])
        applied[lesson] = frozenset(values)
        guard_sets[lesson.value] = {
            "count": len(values),
            "sha256": _hash_set_sha256(values),
            "rule": (
                "development_only_history_overlap_diagnostic"
                if lesson is lessons.LessonId.VISIBLE_UNLOCK
                else "same_lesson_history_through_terminal_u2s"
            ),
            "u2s_unique_layouts_seen": len(by_lesson[lesson]),
        }
    if applied[lessons.LessonId.VISIBLE_UNLOCK] != base_mapping[lessons.LessonId.VISIBLE_UNLOCK]:
        raise ActionEffectQualificationError("Stage A changed the finite U0 guard")
    frozen = MappingProxyType(applied)
    mapping_sha256 = _canonical_sha256(
        {lesson.value: sorted(frozen[lesson]) for lesson in lessons.LessonId}
    )
    return frozen, {
        "base_mapping_sha256": _canonical_sha256(
            {lesson.value: sorted(base_mapping[lesson]) for lesson in lessons.LessonId}
        ),
        "u2s_terminal_report_sha256": U2S_TERMINAL_REPORT_SHA256,
        "u2s_arms": arm_records,
        "guard_sets": guard_sets,
        "applied_mapping_sha256": mapping_sha256,
        "frozen_before_sham": True,
        "sham_history_added_before_action_effect": False,
    }


def _authenticate_parent_and_guards(
    repository: Path,
) -> tuple[
    dict[str, Any],
    Any,
    Mapping[lessons.LessonId, frozenset[str]],
    dict[str, Any],
    dict[str, Any],
]:
    (
        parent,
        base_qualification,
        failed_u2_parent,
        u2r_terminal,
        failed_u2r_r0,
    ) = u2s_qualify._authenticate_predecessors(repository)
    parent_public = u2s_qualify._verify_parent_model(parent)
    if (
        parent_public["checkpoint_sha256"] != v03.PARENT_CHECKPOINT_SHA256
        or parent_public["sidecar_sha256"] != PARENT_SIDECAR_SHA256
        or parent_public["manifest_sha256"] != PARENT_MANIFEST_SHA256
        or parent_public["policy_tensor_sha256"] != v03.PARENT_POLICY_TENSOR_SHA256
        or parent_public["optimizer_state_sha256"] != v03.PARENT_OPTIMIZER_STATE_SHA256
        or parent_public["u1_child_seed"] != v03.PARENT_U1_SEED
        or parent_public["trained_timesteps"] != v03.PARENT_LIFETIME_ACTIONS
        or parent_public["n_updates"] != v03.PARENT_OPTIMIZER_UPDATES
        or parent_public["confirmation_verdict"] != "confirmed"
    ):
        raise ActionEffectQualificationError(
            "confirmed U1 parent provenance or model state changed"
        )
    seed_access = base_qualification.seed_access()
    base_mapping, base_guard_evidence = u2s_qualify.build_u2s_forbidden_layout_hashes(
        base_qualification.verified_report(),
        failed_u2_parent.confirmation_snapshot(),
        access=seed_access,
        terminal_report=u2r_terminal,
    )
    u2s_terminal = authenticate_u2s_terminal()
    mapping, guards = extend_action_effect_forbidden_layout_hashes(
        base_mapping,
        u2s_terminal["_report"],
    )
    terminal_public = {key: value for key, value in u2s_terminal.items() if key != "_report"}
    predecessors = {
        "base_qualification": base_qualification.public_dict(),
        "u2_confirmation": failed_u2_parent.public_dict(),
        "u2r_terminal": {
            "report_sha256": u2s_qualify.U2R_REPORT_SHA256,
            "verdict": u2r_terminal["verdict"],
            "eligible_for_fresh_confirmation": u2r_terminal["eligible_for_fresh_confirmation"],
            "failed_r0_launch": failed_u2r_r0,
        },
        "u2s_terminal": terminal_public,
        "policy_updates_during_authentication": False,
    }
    guards["pre_u2s"] = base_guard_evidence
    return parent_public, base_qualification, mapping, guards, predecessors


def _sampler_preflight(
    mapping: Mapping[lessons.LessonId, frozenset[str]],
    *,
    seed_access: Any,
) -> list[dict[str, Any]]:
    records = frozen_u2r.preflight_u2r_training_layout_sampler(
        mapping,
        seed_access=seed_access,
        worker_streams=v03.WORKER_STREAMS,
        max_attempts=frozen_u2s.LAYOUT_RESAMPLE_ATTEMPTS,
    )
    public = [dict(record) for record in records]
    if len(public) != len(lessons.LessonId) * v03.WORKERS:
        raise ActionEffectQualificationError("v0.3 sampler preflight is incomplete")
    streams = {int(record.get("worker_stream", -1)) for record in public}
    if streams != set(v03.WORKER_STREAMS):
        raise ActionEffectQualificationError("v0.3 sampler preflight worker streams changed")
    return public


def _validate_rng_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "phase",
        "components",
        "aggregate_sha256",
    }:
        raise ActionEffectQualificationError(f"{label} is missing")
    components = value.get("components")
    if (
        value.get("phase") not in {"post_reset_pre_action_one", "post_rollout_pre_optimizer"}
        or not isinstance(components, Mapping)
        or value.get("aggregate_sha256") != _canonical_sha256(components)
    ):
        raise ActionEffectQualificationError(f"{label} changed")
    expected_component_keys = {
        "python_random_sha256",
        "numpy_global_sha256",
        "torch_cpu_sha256",
        "model_action_space",
        "scheduler_sha256",
        "workers",
    }
    if set(components) != expected_component_keys:
        raise ActionEffectQualificationError(f"{label} components changed")
    for key in (
        "python_random_sha256",
        "numpy_global_sha256",
        "torch_cpu_sha256",
        "scheduler_sha256",
    ):
        _require_sha256(components[key], f"{label} {key}")
    _validate_space_rng_identity(
        components["model_action_space"],
        f"{label} model action space",
        expected_type="Discrete",
    )
    workers = components["workers"]
    if not isinstance(workers, list) or len(workers) != v03.WORKERS:
        raise ActionEffectQualificationError(f"{label} workers changed")
    for index, worker in enumerate(workers):
        if not isinstance(worker, Mapping) or set(worker) != {
            "worker_index",
            "worker_stream",
            "curriculum_sha256",
            "base_environment_sha256",
            "action_space",
            "observation_space",
        }:
            raise ActionEffectQualificationError(f"{label} worker schema changed")
        if (
            worker.get("worker_index") != index
            or worker.get("worker_stream") != v03.WORKER_STREAMS[index]
        ):
            raise ActionEffectQualificationError(f"{label} worker identity changed")
        _require_sha256(
            worker["curriculum_sha256"],
            f"{label} worker curriculum",
        )
        _require_sha256(
            worker["base_environment_sha256"],
            f"{label} worker environment",
        )
        _validate_space_rng_identity(
            worker["action_space"],
            f"{label} worker action space",
            expected_type="Discrete",
        )
        _validate_space_rng_identity(
            worker["observation_space"],
            f"{label} worker observation space",
            expected_type="Dict",
        )
    return json.loads(json.dumps(dict(value)))


def _validate_space_rng_identity(
    value: Any,
    label: str,
    *,
    expected_type: str | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise ActionEffectQualificationError(f"{label} is missing")
    kind = value.get("type")
    expected_keys = {"type", "self_sha256"}
    if kind == "Dict":
        expected_keys.add("children")
    if (
        set(value) != expected_keys
        or not isinstance(kind, str)
        or (expected_type is not None and kind != expected_type)
    ):
        raise ActionEffectQualificationError(f"{label} schema changed")
    _require_sha256(value["self_sha256"], f"{label} state")
    if kind == "Dict":
        children = value["children"]
        if not isinstance(children, Mapping) or set(children) != {
            IMAGE_KEY,
            ACTION_EFFECT_KEY,
        }:
            raise ActionEffectQualificationError(f"{label} Dict children changed")
        _validate_space_rng_identity(
            children[IMAGE_KEY],
            f"{label} image child",
            expected_type="Box",
        )
        _validate_space_rng_identity(
            children[ACTION_EFFECT_KEY],
            f"{label} action-effect child",
            expected_type="Box",
        )


def _validate_sampler_preflight(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != (len(lessons.LessonId) * v03.WORKERS):
        raise ActionEffectQualificationError("v0.3 smoke sampler preflight is incomplete")
    expected_pairs = [
        (lesson.value, stream) for lesson in lessons.LessonId for stream in v03.WORKER_STREAMS
    ]
    normalized: list[dict[str, Any]] = []
    for record, expected_pair in zip(value, expected_pairs, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "lesson_id",
            "worker_stream",
            "accepted_seed",
            "accepted_attempt",
            "layout_sha256",
            "forbidden_layouts",
            "max_attempts",
        }:
            raise ActionEffectQualificationError("v0.3 smoke sampler record schema changed")
        pair = (record.get("lesson_id"), record.get("worker_stream"))
        if (
            pair != expected_pair
            or not isinstance(record.get("accepted_seed"), int)
            or not 0 <= int(record["accepted_seed"]) < 1_000_000
            or not isinstance(record.get("accepted_attempt"), int)
            or not 1 <= int(record["accepted_attempt"]) <= 8_192
            or not isinstance(record.get("forbidden_layouts"), int)
            or int(record["forbidden_layouts"]) < 0
            or record.get("max_attempts") != 8_192
        ):
            raise ActionEffectQualificationError("v0.3 smoke sampler record changed")
        _require_sha256(
            record["layout_sha256"],
            "v0.3 smoke sampler layout",
        )
        normalized.append(json.loads(json.dumps(dict(record))))
    return normalized


def _validate_smoke_workers(value: Any, arm_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != v03.WORKERS:
        raise ActionEffectQualificationError(f"v0.3 smoke workers changed: {arm_name}")
    normalized: list[dict[str, Any]] = []
    expected_keys = {
        "worker_index",
        "transitions",
        "episodes_started",
        "episode_starts_sha256",
        "action_counts",
        "lesson_transition_counts",
        "reward_total_hex",
        "extrinsic_total_hex",
        "curiosity_total_hex",
        "penalty_total_hex",
        "penalty_events",
        "penalty_eligible_events",
        "maximum_episode_penalty_count",
        "trajectory_sha256",
        "reward_evidence_sha256",
    }
    for index, worker in enumerate(value):
        if not isinstance(worker, Mapping) or set(worker) != expected_keys:
            raise ActionEffectQualificationError(f"v0.3 smoke worker schema changed: {arm_name}")
        actions = worker["action_counts"]
        lesson_counts = worker["lesson_transition_counts"]
        if (
            worker.get("worker_index") != index
            or worker.get("transitions") != v03.ROLLOUT_STEPS
            or not isinstance(worker.get("episodes_started"), int)
            or int(worker["episodes_started"]) < 1
            or not isinstance(actions, Mapping)
            or set(actions) != {str(action) for action in range(7)}
            or any(not isinstance(count, int) or count < 0 for count in actions.values())
            or sum(actions.values()) != v03.ROLLOUT_STEPS
            or not isinstance(lesson_counts, Mapping)
            or set(lesson_counts) != {lesson.value for lesson in lessons.LessonId}
            or any(not isinstance(count, int) or count < 0 for count in lesson_counts.values())
            or sum(lesson_counts.values()) != v03.ROLLOUT_STEPS
            or worker.get("penalty_events") != 0
            or worker.get("penalty_eligible_events") != 0
            or worker.get("maximum_episode_penalty_count") != 0
        ):
            raise ActionEffectQualificationError(f"v0.3 smoke worker totals changed: {arm_name}")
        for key in (
            "episode_starts_sha256",
            "trajectory_sha256",
            "reward_evidence_sha256",
        ):
            _require_sha256(worker[key], f"{arm_name} worker {key}")
        for key in (
            "reward_total_hex",
            "extrinsic_total_hex",
            "curiosity_total_hex",
            "penalty_total_hex",
        ):
            try:
                measured = float.fromhex(str(worker[key]))
            except ValueError as error:
                raise ActionEffectQualificationError(
                    f"{arm_name} worker {key} is invalid"
                ) from error
            if not math.isfinite(measured):
                raise ActionEffectQualificationError(f"{arm_name} worker {key} is not finite")
        if float.fromhex(str(worker["penalty_total_hex"])) != 0.0:
            raise ActionEffectQualificationError(f"{arm_name} smoke used reward shaping")
        normalized.append(json.loads(json.dumps(dict(worker))))
    return normalized


def _validate_seed_evidence(
    value: Any,
    *,
    episode_starts: int,
    arm_name: str,
) -> dict[str, Any]:
    expected_keys = {
        "episode_starts",
        "minimum_seed",
        "maximum_seed",
        "training_range_only",
        "separated_unlock_roles",
        "protected_roles",
        "protected_seed_hits",
        "confirmation_or_final_seed_generated",
    }
    expected_protected = sorted(role.value for role in smoke.frozen_smoke._PROTECTED_ROLES)
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("episode_starts") != episode_starts
        or not isinstance(value.get("minimum_seed"), int)
        or not isinstance(value.get("maximum_seed"), int)
        or not 0 <= int(value["minimum_seed"]) <= int(value["maximum_seed"]) < 1_000_000
        or value.get("training_range_only") is not True
        or value.get("separated_unlock_roles") not in ([], ["training"])
        or value.get("protected_roles") != expected_protected
        or value.get("protected_seed_hits") != []
        or value.get("confirmation_or_final_seed_generated") is not False
    ):
        raise ActionEffectQualificationError(f"v0.3 smoke seed evidence changed: {arm_name}")
    return json.loads(json.dumps(dict(value)))


def validate_disposable_smoke(
    value: Any,
    *,
    source_commit: str,
    guard_mapping_sha256: str,
    sampler_preflight_sha256: str,
) -> dict[str, Any]:
    """Validate the destroyed matched rollout and return its bound summary."""

    if not isinstance(value, Mapping):
        raise ActionEffectQualificationError("v0.3 disposable smoke is missing")
    expected_top_level = {
        "schema_version",
        "protocol",
        "completed_at",
        "verdict",
        "source",
        "parent_checkpoint",
        "parent_checkpoint_sha256",
        "worker_streams",
        "guard_mapping_sha256",
        "sampler_preflight",
        "sampler_preflight_sha256",
        "actions_per_arm",
        "pre_action_rng_identical",
        "pre_update_behavior_identical",
        "first_rollout_trajectory_identical",
        "first_rollout_policy_outputs_identical",
        "post_rollout_pre_optimizer_rng_identical",
        "first_rollout_episode_ledger_identical",
        "learning_divergence_begins_after_first_update",
        "arms",
        "canonical_predecessor_roots_unchanged",
        "temporary_root_removed",
        "scientific_evidence",
        "checkpoint_reuse_authorized",
    }
    arms = value.get("arms")
    source = value.get("source")
    sampler = _validate_sampler_preflight(value.get("sampler_preflight"))
    if (
        set(value) != expected_top_level
        or value.get("schema_version") != smoke.SCHEMA_VERSION
        or value.get("protocol") != smoke.SMOKE_PROTOCOL
        or value.get("verdict") != "passed"
        or not isinstance(source, Mapping)
        or set(source) != {"commit", "branch", "dirty"}
        or source.get("commit") != source_commit
        or not isinstance(source.get("branch"), str)
        or source.get("dirty") is not False
        or value.get("parent_checkpoint") != str(v03.PARENT_CHECKPOINT)
        or value.get("parent_checkpoint_sha256") != v03.PARENT_CHECKPOINT_SHA256
        or value.get("worker_streams") != list(v03.WORKER_STREAMS)
        or value.get("guard_mapping_sha256") != guard_mapping_sha256
        or value.get("sampler_preflight_sha256") != sampler_preflight_sha256
        or value.get("sampler_preflight_sha256") != _canonical_sha256(sampler)
        or int(value.get("actions_per_arm", -1)) != v03.ROLLOUT_TRANSITIONS
        or not isinstance(arms, Mapping)
        or set(arms) != set(_ARMS)
        or any(
            value.get(label) is not True
            for label in (
                "pre_action_rng_identical",
                "pre_update_behavior_identical",
                "first_rollout_trajectory_identical",
                "first_rollout_policy_outputs_identical",
                "post_rollout_pre_optimizer_rng_identical",
                "first_rollout_episode_ledger_identical",
                "learning_divergence_begins_after_first_update",
                "canonical_predecessor_roots_unchanged",
                "temporary_root_removed",
            )
        )
        or value.get("scientific_evidence") is not False
        or value.get("checkpoint_reuse_authorized") is not False
    ):
        raise ActionEffectQualificationError("v0.3 disposable smoke contract changed")
    sham = arms[ActionEffectMode.SHAM.value]
    candidate = arms[ActionEffectMode.ACTION_EFFECT.value]
    if not isinstance(sham, Mapping) or not isinstance(candidate, Mapping):
        raise ActionEffectQualificationError("v0.3 smoke arms are invalid")
    matched_fields = (
        "before",
        "workers",
        "trajectory_identity",
        "pre_action_rng_identity",
        "policy_output_sha256",
        "post_rollout_rng_identity",
        "episode_ledger",
        "seed_evidence",
    )
    if any(sham.get(field) != candidate.get(field) for field in matched_fields):
        raise ActionEffectQualificationError(
            "matched twins diverged before the first optimizer phase"
        )
    before = sham.get("before")
    if (
        not isinstance(before, Mapping)
        or before.get("trained_timesteps") != v03.PARENT_LIFETIME_ACTIONS
        or before.get("optimizer_updates") != v03.PARENT_OPTIMIZER_UPDATES
        or before.get("policy_tensor_sha256") == sham.get("after", {}).get("policy_tensor_sha256")
        or before.get("optimizer_state_sha256")
        == sham.get("after", {}).get("optimizer_state_sha256")
        or sham.get("effect_projection_nonzero_parameters") != 0
        or int(candidate.get("effect_projection_nonzero_parameters", 0)) <= 0
    ):
        raise ActionEffectQualificationError("v0.3 smoke transplant/update boundary changed")
    for name, arm in (
        (ActionEffectMode.SHAM.value, sham),
        (ActionEffectMode.ACTION_EFFECT.value, candidate),
    ):
        after = arm.get("after")
        transplant = arm.get("transplant")
        equivalence = (
            transplant.get("zero_context_equivalence") if isinstance(transplant, Mapping) else None
        )
        transplant_record = (
            transplant.get("transplant") if isinstance(transplant, Mapping) else None
        )
        workers = _validate_smoke_workers(arm.get("workers"), name)
        trajectory = arm.get("trajectory_identity")
        expected_trajectory = smoke._trajectory_identity(workers)
        ledger = arm.get("episode_ledger")
        episode_starts = sum(int(worker["episodes_started"]) for worker in workers)
        if (
            not isinstance(arm, Mapping)
            or set(arm)
            != {
                "arm",
                "before",
                "after",
                "effect_projection_nonzero_parameters",
                "transplant",
                "workers",
                "trajectory_identity",
                "pre_action_rng_identity",
                "policy_output_sha256",
                "post_rollout_rng_identity",
                "episode_ledger",
                "seed_evidence",
                "updated_archive_reloaded_exactly",
                "development_checkpoint_reuse_authorized",
            }
            or arm.get("arm") != name
            or not isinstance(after, Mapping)
            or after.get("trained_timesteps")
            != v03.PARENT_LIFETIME_ACTIONS + v03.ROLLOUT_TRANSITIONS
            or after.get("optimizer_updates") != v03.PARENT_OPTIMIZER_UPDATES + v03.PPO_EPOCHS
            or not isinstance(transplant, Mapping)
            or not isinstance(equivalence, Mapping)
            or any(
                equivalence.get(key) is not True
                for key in (
                    "features_exact",
                    "actions_exact",
                    "values_exact",
                    "log_probabilities_exact",
                    "recurrent_states_exact",
                )
            )
            or not isinstance(transplant_record, Mapping)
            or transplant_record.get("missing_optimizer_state_names") not in ([], ())
            or transplant_record.get("effect_encoder_zero") is not True
            or arm.get("updated_archive_reloaded_exactly") is not True
            or arm.get("development_checkpoint_reuse_authorized") is not False
            or trajectory != expected_trajectory
            or not isinstance(ledger, Mapping)
            or set(ledger) != {"records", "normalized_sha256"}
            or ledger.get("records") != episode_starts
        ):
            raise ActionEffectQualificationError(f"v0.3 smoke arm contract changed: {name}")
        _require_sha256(
            ledger["normalized_sha256"],
            f"{name} first-rollout episode ledger",
        )
        _require_sha256(
            arm["policy_output_sha256"],
            f"{name} first-rollout policy output",
        )
        pre_action = _validate_rng_identity(
            arm["pre_action_rng_identity"],
            f"{name} pre-action RNG identity",
        )
        if pre_action["phase"] != "post_reset_pre_action_one":
            raise ActionEffectQualificationError(f"{name} pre-action RNG phase changed")
        post_rollout = _validate_rng_identity(
            arm["post_rollout_rng_identity"],
            f"{name} post-rollout RNG identity",
        )
        if post_rollout["phase"] != "post_rollout_pre_optimizer":
            raise ActionEffectQualificationError(f"{name} post-rollout RNG phase changed")
        _validate_seed_evidence(
            arm["seed_evidence"],
            episode_starts=episode_starts,
            arm_name=name,
        )
    pre_action = _validate_rng_identity(
        sham["pre_action_rng_identity"],
        "v0.3 pre-action RNG identity",
    )
    _require_sha256(sham["policy_output_sha256"], "first-rollout policy output")
    _require_sha256(
        sham["episode_ledger"]["normalized_sha256"],
        "first-rollout episode ledger",
    )
    return {
        "protocol": smoke.SMOKE_PROTOCOL,
        "verdict": "passed",
        "source_commit": source_commit,
        "guard_mapping_sha256": guard_mapping_sha256,
        "sampler_preflight_sha256": sampler_preflight_sha256,
        "actions_per_arm": v03.ROLLOUT_TRANSITIONS,
        "pre_action_rng_identity": pre_action,
        "pre_action_rng_identity_sha256": _canonical_sha256(pre_action),
        "first_rollout_policy_output_sha256": str(sham["policy_output_sha256"]),
        "first_rollout_trajectory_sha256": _canonical_sha256(sham["trajectory_identity"]),
        "first_rollout_episode_ledger_sha256": str(sham["episode_ledger"]["normalized_sha256"]),
        "post_rollout_rng_identity_sha256": _canonical_sha256(sham["post_rollout_rng_identity"]),
        "first_rollout_digest_profile": v03.FIRST_ROLLOUT_DIGEST_PROFILE,
        "transplant_sha256": _canonical_sha256(sham["transplant"]),
        "matched_through_first_rollout": True,
        "optimizer_phase_completed_for_both": True,
        "candidate_context_projection_updated": True,
        "sham_context_projection_remained_zero": True,
        "temporary_root_removed": True,
        "checkpoint_reuse_authorized": False,
        "full_report_sha256": _canonical_sha256(value),
    }


def _storage_preflight(
    *,
    require_qualification_absent: bool,
    permit_bound_managed_roots: bool = False,
) -> dict[str, Any]:
    volume = Path("/Volumes/T7 Developer")
    dungeon_root = volume / "DungeonApprentice"
    _reject_symlink_chain(volume)
    _reject_symlink_chain(dungeon_root)
    try:
        volume_stat = volume.stat()
        parent_stat = volume.parent.stat()
        dungeon_stat = dungeon_root.stat()
    except OSError as error:
        raise ActionEffectQualificationError("cannot inspect Stage-A storage") from error
    if (
        not stat.S_ISDIR(volume_stat.st_mode)
        or not stat.S_ISDIR(dungeon_stat.st_mode)
        or not os.path.ismount(volume)
        or volume_stat.st_dev == parent_stat.st_dev
        or dungeon_stat.st_dev != volume_stat.st_dev
    ):
        raise ActionEffectQualificationError("T7 Developer is not a separate mounted volume")
    fresh = (CANONICAL_COHORT_ROOT, CANONICAL_MEDIA_ROOT)
    managed_roots_absent = True
    for path in fresh:
        if path.is_symlink():
            raise ActionEffectQualificationError("Stage-A cohort or media root is unsafe")
        if path.exists():
            managed_roots_absent = False
            if not permit_bound_managed_roots:
                raise ActionEffectQualificationError("Stage-A cohort or media root is not fresh")
            try:
                metadata = path.stat()
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ActionEffectQualificationError(
                    "cannot authenticate a bound Stage-A managed root"
                ) from error
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or resolved != path
                or metadata.st_dev != volume_stat.st_dev
            ):
                raise ActionEffectQualificationError("Stage-A managed root changed device or type")
    if require_qualification_absent and (
        CANONICAL_QUALIFICATION_DIRECTORY.exists() or CANONICAL_QUALIFICATION_DIRECTORY.is_symlink()
    ):
        raise ActionEffectQualificationError("Stage-A qualification root is not fresh")
    protected = (
        v03.PARENT_CHECKPOINT,
        frozen_u2.CANONICAL_U1_CONFIRMATION,
        U2S_TERMINAL_ROOT,
        u2s_qualify.U2R_RUN_DIRECTORY,
        FAILED_STAGE_A_ATTEMPT_QUALIFICATION_DIRECTORY,
        FAILED_STAGE_A_ATTEMPT_ROOT,
        FAILED_STAGE_A_ATTEMPT_MEDIA_ROOT,
        FAILED_STAGE_A_R2_ATTEMPT_QUALIFICATION_DIRECTORY,
        FAILED_STAGE_A_R2_ATTEMPT_ROOT,
        FAILED_STAGE_A_R2_ATTEMPT_MEDIA_ROOT,
    )
    managed = (
        CANONICAL_QUALIFICATION_DIRECTORY,
        CANONICAL_COHORT_ROOT,
        CANONICAL_MEDIA_ROOT,
    )
    for artifact in protected:
        resolved = artifact.expanduser().resolve()
        if any(resolved == root or root in resolved.parents for root in managed):
            raise ActionEffectQualificationError("Stage-A roots overlap predecessor evidence")
    usage = shutil.disk_usage(volume)
    minimum = int(v03.MINIMUM_FREE_GIB * 1024**3)
    if usage.free < minimum:
        raise ActionEffectQualificationError("T7 Developer lacks the Stage-A free-space reserve")
    return {
        "volume": str(volume),
        "dungeon_root": str(dungeon_root),
        "separate_mounted_device": True,
        "volume_device": int(volume_stat.st_dev),
        "parent_device": int(parent_stat.st_dev),
        "managed_roots": [str(path) for path in managed],
        "cohort_and_media_roots_absent": managed_roots_absent,
        "bound_managed_roots_permitted": permit_bound_managed_roots,
        "qualification_root_absent_before_claim": require_qualification_absent,
        "protected_artifacts_outside_managed_roots": True,
        "minimum_free_bytes": minimum,
        "measured_free_bytes": int(usage.free),
        "measured_total_bytes": int(usage.total),
        "passed": True,
        "measured_at": utc_now(),
    }


def _verify_storage_preflight(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ActionEffectQualificationError("storage preflight is missing")
    live = _storage_preflight(
        require_qualification_absent=False,
        permit_bound_managed_roots=True,
    )
    if (
        value.get("volume") != live["volume"]
        or value.get("dungeon_root") != live["dungeon_root"]
        or value.get("separate_mounted_device") is not True
        or value.get("volume_device") != live["volume_device"]
        or value.get("parent_device") != live["parent_device"]
        or value.get("managed_roots") != live["managed_roots"]
        or value.get("cohort_and_media_roots_absent") is not True
        or value.get("bound_managed_roots_permitted") is not False
        or value.get("qualification_root_absent_before_claim") is not True
        or value.get("protected_artifacts_outside_managed_roots") is not True
        or value.get("minimum_free_bytes") != live["minimum_free_bytes"]
        or not isinstance(value.get("measured_free_bytes"), int)
        or int(value["measured_free_bytes"]) < int(value["minimum_free_bytes"])
        or value.get("passed") is not True
    ):
        raise ActionEffectQualificationError("storage preflight changed")
    try:
        timestamp = datetime.fromisoformat(str(value["measured_at"]))
    except (KeyError, ValueError) as error:
        raise ActionEffectQualificationError("storage preflight timestamp is invalid") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(None):
        raise ActionEffectQualificationError("storage preflight timestamp must be UTC")


def _collect_static_inputs(
    repository: Path,
) -> tuple[
    dict[str, Any],
    Any,
    Mapping[lessons.LessonId, frozenset[str]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    parent, base, mapping, guards, predecessors = _authenticate_parent_and_guards(repository)
    sampler = _sampler_preflight(mapping, seed_access=base.seed_access())
    return parent, base, mapping, guards, predecessors, sampler


def build_action_effect_tag_payload(
    repository: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Build the canonical one-line preregistration tag payload."""

    commit = _require_git_object(source_commit, "v0.3 source commit")
    protocol_path = repository.expanduser().resolve() / PROTOCOL_DOCUMENT
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise ActionEffectQualificationError("v0.3 protocol document is missing or unsafe")
    parent, _base, _mapping, guards, predecessors, sampler = _collect_static_inputs(repository)
    protected = _protected_seed_partitions()
    contract = _architecture_contract()
    failed_attempt = authenticate_failed_stage_a_attempt(
        repository=repository,
    )
    failed_r2_attempt = authenticate_failed_stage_a_r2_attempt(
        repository=repository,
    )
    payload = {
        "schema_version": TAG_SCHEMA_VERSION,
        "kind": TAG_KIND,
        "protocol": PROTOCOL,
        "tag": QUALIFIED_TAG,
        "remote": QUALIFIED_REMOTE,
        "remote_url": EXPECTED_ORIGIN_URL,
        "source_commit": commit,
        "protocol_document_sha256": file_sha256(protocol_path),
        "parent": {
            key: parent[key]
            for key in (
                "checkpoint_sha256",
                "sidecar_sha256",
                "manifest_sha256",
                "policy_tensor_sha256",
                "optimizer_state_sha256",
                "confirmation_sha256",
                "trained_timesteps",
                "n_updates",
            )
        },
        "u2s_terminal": {
            key: predecessors["u2s_terminal"][key]
            for key in (
                "report_sha256",
                "report_integrity_sha256",
                "verdict",
                "selected_configuration",
                "successor_cohort_authorized",
                "checkpoint_reuse_authorized",
            )
        },
        "failed_stage_a_attempt": failed_attempt,
        "failed_stage_a_r2_attempt": failed_r2_attempt,
        "guard_mapping_sha256": guards["applied_mapping_sha256"],
        "sampler_preflight_sha256": _canonical_sha256(sampler),
        "architecture_contract_sha256": _canonical_sha256(contract),
        "protected_partitions_sha256": _canonical_sha256(protected),
        "runtime_contract": {
            "snapshot": runtime_snapshot(),
            "training_device": "cpu",
            "qualification_smoke_device": "cpu",
        },
        "roots": {
            "qualification": str(CANONICAL_QUALIFICATION_DIRECTORY),
            "cohort": str(CANONICAL_COHORT_ROOT),
            "media": str(CANONICAL_MEDIA_ROOT),
        },
        "dashboard_port": DASHBOARD_PORT,
        "storage_caps": _storage_caps(),
        "resume_rule": _resume_rule(),
    }
    if set(payload) != TAG_FIELDS:
        raise ActionEffectQualificationError("internal v0.3 tag payload is incomplete")
    return payload


def verify_source_tag(
    repository: Path,
    *,
    expected_source_commit: str | None = None,
    expected_tag_object: str | None = None,
    expected_tag_payload: Mapping[str, Any] | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Verify one clean, pushed commit and its published annotated tag."""

    root = repository.expanduser().resolve()
    head = _require_git_object(
        _git(root, ["rev-parse", "--verify", "HEAD^{commit}"], runner=runner),
        "v0.3 source commit",
    )
    if expected_source_commit is not None and head != _require_git_object(
        expected_source_commit, "expected source commit"
    ):
        raise ActionEffectQualificationError("v0.3 source differs from qualification")
    if _git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        runner=runner,
    ):
        raise ActionEffectQualificationError("v0.3 qualification requires a clean repository")
    ref = f"refs/tags/{QUALIFIED_TAG}"
    if _git(root, ["cat-file", "-t", ref], runner=runner) != "tag":
        raise ActionEffectQualificationError("v0.3 source tag must be annotated")
    tag_object = _require_git_object(
        _git(root, ["rev-parse", "--verify", ref], runner=runner),
        "v0.3 tag object",
    )
    if expected_tag_object is not None and tag_object != _require_git_object(
        expected_tag_object, "expected tag object"
    ):
        raise ActionEffectQualificationError("v0.3 tag object changed")
    if _git(root, ["rev-list", "-n", "1", ref], runner=runner) != head:
        raise ActionEffectQualificationError("v0.3 tag does not point to HEAD")
    remote_url = _git(
        root,
        ["remote", "get-url", QUALIFIED_REMOTE],
        runner=runner,
    )
    if remote_url != EXPECTED_ORIGIN_URL:
        raise ActionEffectQualificationError("v0.3 origin changed")
    remote = _git(
        root,
        ["ls-remote", "--tags", QUALIFIED_REMOTE, ref],
        runner=runner,
    ).split()
    if remote != [tag_object, ref]:
        raise ActionEffectQualificationError("v0.3 annotated tag is not published")
    raw_message = _git(
        root,
        ["for-each-ref", "--format=%(contents)", ref],
        runner=runner,
    )
    if "\n" in raw_message:
        raise ActionEffectQualificationError("v0.3 tag message must be one JSON line")
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as error:
        raise ActionEffectQualificationError("v0.3 tag message is not JSON") from error
    expected = (
        build_action_effect_tag_payload(root, source_commit=head)
        if expected_tag_payload is None
        else json.loads(json.dumps(expected_tag_payload))
    )
    canonical = json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if (
        not isinstance(payload, Mapping)
        or set(payload) != TAG_FIELDS
        or payload != expected
        or raw_message != canonical
    ):
        raise ActionEffectQualificationError("v0.3 tag payload differs from live protocol inputs")
    return {
        "commit": head,
        "dirty": False,
        "tag": QUALIFIED_TAG,
        "tag_object": tag_object,
        "remote": QUALIFIED_REMOTE,
        "remote_url": remote_url,
        "tag_payload": dict(payload),
        "tag_payload_sha256": _canonical_sha256(payload),
    }


def _expected_report(
    repository: Path,
    *,
    source: Mapping[str, Any],
    claim: Mapping[str, Any],
    smoke_report: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    Any,
    Mapping[lessons.LessonId, frozenset[str]],
]:
    parent, base, mapping, guards, predecessors, sampler = _collect_static_inputs(repository)
    sampler_digest = _canonical_sha256(sampler)
    smoke_evidence = validate_disposable_smoke(
        smoke_report,
        source_commit=str(source["commit"]),
        guard_mapping_sha256=str(guards["applied_mapping_sha256"]),
        sampler_preflight_sha256=sampler_digest,
    )
    protocol_path = repository / PROTOCOL_DOCUMENT
    failed_attempt = authenticate_failed_stage_a_attempt(
        repository=repository,
    )
    failed_r2_attempt = authenticate_failed_stage_a_r2_attempt(
        repository=repository,
    )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "kind": KIND,
            "verdict": VERDICT,
            "claim": dict(claim),
            "source": dict(source),
            "protocol_document": {
                "path": str(PROTOCOL_DOCUMENT),
                "sha256": file_sha256(protocol_path),
            },
            "parent": parent,
            "predecessors": predecessors,
            "failed_stage_a_attempt": failed_attempt,
            "failed_stage_a_r2_attempt": failed_r2_attempt,
            "guards": guards,
            "sampler_preflight": sampler,
            "architecture_contract": _architecture_contract(),
            "protected_partitions": _protected_seed_partitions(),
            "smoke_evidence": smoke_evidence,
            "storage_caps": _storage_caps(),
            "restrictions": {
                "arms_run_sequentially": True,
                "each_arm_starts_from_exact_confirmed_u1": True,
                "u2_u2r_u2s_checkpoint_loading_authorized": False,
                "stage_a_resume_supported": False,
                "failed_attempt_resume_authorized": False,
                "failed_attempt_root_reuse_authorized": False,
                "failed_r2_attempt_resume_authorized": False,
                "failed_r2_attempt_root_reuse_authorized": False,
                "replacement_restarts_both_arms_from_confirmed_u1": True,
                "confirmation_or_final_seed_issuer_opened": False,
                "canonical_or_claim_policy_updates_during_qualification": False,
                "disposable_smoke_policies_destroyed": True,
                "development_checkpoint_reuse_authorized": False,
                "stage_b_authorized": False,
                "u3_authorized": False,
            },
        },
        base,
        mapping,
    )


@dataclass(frozen=True)
class ActionEffectQualificationEvidence:
    """Authenticated inputs exposed to the Stage-A trainer."""

    report: str
    report_sha256: str
    checksum: str
    source_commit: str
    tag: str
    tag_object: str
    tag_payload_sha256: str
    verdict: str
    protocol_document_sha256: str
    guard_mapping_sha256: str
    sampler_preflight_sha256: str
    architecture_contract_sha256: str
    smoke_evidence_sha256: str
    protected_partitions_sha256: str
    failed_stage_a_attempt_sha256: str
    failed_stage_a_r2_attempt_sha256: str
    storage_caps: Mapping[str, int]
    _report_bytes: bytes = field(repr=False, compare=False)
    _base_qualification: Any = field(repr=False, compare=False)
    _forbidden_layouts: Mapping[lessons.LessonId, frozenset[str]] = field(
        repr=False,
        compare=False,
    )

    def public_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "report_sha256": self.report_sha256,
            "checksum": self.checksum,
            "source_commit": self.source_commit,
            "tag": self.tag,
            "tag_object": self.tag_object,
            "tag_payload_sha256": self.tag_payload_sha256,
            "verdict": self.verdict,
            "protocol_document_sha256": self.protocol_document_sha256,
            "guard_mapping_sha256": self.guard_mapping_sha256,
            "sampler_preflight_sha256": self.sampler_preflight_sha256,
            "architecture_contract_sha256": self.architecture_contract_sha256,
            "smoke_evidence_sha256": self.smoke_evidence_sha256,
            "protected_partitions_sha256": (self.protected_partitions_sha256),
            "failed_stage_a_attempt_sha256": (self.failed_stage_a_attempt_sha256),
            "failed_stage_a_r2_attempt_sha256": (
                self.failed_stage_a_r2_attempt_sha256
            ),
            "storage_caps": dict(self.storage_caps),
        }

    def verified_report(self) -> dict[str, Any]:
        value = json.loads(self._report_bytes)
        if not isinstance(value, dict):
            raise ActionEffectQualificationError("verified v0.3 report bytes changed")
        return value

    def seed_access(self) -> Any:
        return self._base_qualification.seed_access()

    def forbidden_layout_hashes(
        self,
    ) -> Mapping[lessons.LessonId, frozenset[str]]:
        return MappingProxyType(
            {lesson: frozenset(values) for lesson, values in self._forbidden_layouts.items()}
        )


def _assert_canonical_report(path: Path) -> Path:
    requested = path.expanduser().absolute()
    if requested != CANONICAL_REPORT:
        raise ActionEffectQualificationError(
            f"v0.3 qualification report must be {CANONICAL_REPORT}"
        )
    _reject_symlink_chain(requested)
    return requested


def _verify_checksum(path: Path, digest: str) -> None:
    try:
        fields = (
            _regular_file_bytes(
                path,
                "v0.3 qualification checksum",
                maximum=256,
            )
            .decode("ascii")
            .strip()
            .split()
        )
    except UnicodeDecodeError as error:
        raise ActionEffectQualificationError("v0.3 checksum is not ASCII") from error
    if fields != [digest, CANONICAL_REPORT.name]:
        raise ActionEffectQualificationError("v0.3 qualification checksum changed")


def verify_action_effect_qualification(
    path: Path,
    *,
    expected_source_commit: str,
    expected_tag_object: str | None = None,
    repository: Path | None = None,
) -> ActionEffectQualificationEvidence:
    """Reauthenticate the canonical Stage-A qualification and live inputs."""

    report_path = _assert_canonical_report(path)
    repository = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else repository.expanduser().resolve()
    )
    source = verify_source_tag(
        repository,
        expected_source_commit=expected_source_commit,
        expected_tag_object=expected_tag_object,
    )
    report_bytes = _regular_file_bytes(
        report_path,
        "v0.3 qualification report",
        maximum=MAX_REPORT_BYTES,
    )
    try:
        report = json.loads(report_bytes)
    except json.JSONDecodeError as error:
        raise ActionEffectQualificationError("v0.3 qualification report is invalid JSON") from error
    if not isinstance(report, dict) or set(report) != _REPORT_FIELDS:
        raise ActionEffectQualificationError("v0.3 qualification report fields changed")
    digest = hashlib.sha256(report_bytes).hexdigest()
    _verify_checksum(CANONICAL_CHECKSUM, digest)
    claim = _read_json(CANONICAL_CLAIM, "v0.3 qualification claim")
    if report.get("claim") != claim:
        raise ActionEffectQualificationError("v0.3 qualification claim binding changed")
    smoke_report = report.get("smoke_evidence", {}).get("_full_report")
    if not isinstance(smoke_report, Mapping):
        raise ActionEffectQualificationError("v0.3 report lacks its disposable smoke transcript")
    expected, base, mapping = _expected_report(
        repository,
        source=source,
        claim=claim,
        smoke_report=smoke_report,
    )
    expected["storage_preflight"] = report.get("storage_preflight")
    expected["created_at"] = report.get("created_at")
    expected["smoke_evidence"]["_full_report"] = smoke_report
    if report != expected:
        raise ActionEffectQualificationError("v0.3 qualification evidence differs from live inputs")
    _verify_storage_preflight(report["storage_preflight"])
    try:
        created_at = datetime.fromisoformat(str(report["created_at"]))
    except ValueError as error:
        raise ActionEffectQualificationError("v0.3 qualification timestamp is invalid") from error
    if created_at.tzinfo is None or created_at.utcoffset() != UTC.utcoffset(None):
        raise ActionEffectQualificationError("v0.3 qualification timestamp must be UTC")
    return ActionEffectQualificationEvidence(
        report=str(report_path),
        report_sha256=digest,
        checksum=str(CANONICAL_CHECKSUM),
        source_commit=str(source["commit"]),
        tag=QUALIFIED_TAG,
        tag_object=str(source["tag_object"]),
        tag_payload_sha256=str(source["tag_payload_sha256"]),
        verdict=VERDICT,
        protocol_document_sha256=str(report["protocol_document"]["sha256"]),
        guard_mapping_sha256=str(report["guards"]["applied_mapping_sha256"]),
        sampler_preflight_sha256=_canonical_sha256(report["sampler_preflight"]),
        architecture_contract_sha256=_canonical_sha256(report["architecture_contract"]),
        smoke_evidence_sha256=_canonical_sha256(
            {key: value for key, value in report["smoke_evidence"].items() if key != "_full_report"}
        ),
        protected_partitions_sha256=_canonical_sha256(report["protected_partitions"]),
        failed_stage_a_attempt_sha256=_canonical_sha256(report["failed_stage_a_attempt"]),
        failed_stage_a_r2_attempt_sha256=_canonical_sha256(
            report["failed_stage_a_r2_attempt"]
        ),
        storage_caps=MappingProxyType(dict(report["storage_caps"])),
        _report_bytes=report_bytes,
        _base_qualification=base,
        _forbidden_layouts=mapping,
    )


def collect_action_effect_qualification(
    *,
    repository: Path,
    output: Path = CANONICAL_REPORT,
    smoke_runner: Callable[..., Mapping[str, Any]] = smoke.run_smoke,
) -> ActionEffectQualificationEvidence:
    """Claim and collect one qualification; never create a tag or launch."""

    report_path = _assert_canonical_report(output)
    source = verify_source_tag(repository)
    storage_preflight = _storage_preflight(require_qualification_absent=True)
    (
        _parent,
        smoke_base,
        smoke_mapping,
        smoke_guards,
        _predecessors,
        _sampler,
    ) = _collect_static_inputs(repository)
    report_path.parent.mkdir(mode=0o700)
    claim = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "kind": "qualification_attempt_claim",
        "claim_id": uuid.uuid4().hex,
        "source_commit": source["commit"],
        "tag": QUALIFIED_TAG,
        "tag_object": source["tag_object"],
        "created_at": utc_now(),
    }
    try:
        with CANONICAL_CLAIM.open("xb") as handle:
            claim_bytes = _canonical_json_bytes(claim)
            handle.write(claim_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        smoke_report = smoke_runner(
            repository=repository,
            require_clean_source=True,
            seed_access=smoke_base.seed_access(),
            forbidden_layout_hashes=smoke_mapping,
        )
        if smoke_report.get("guard_mapping_sha256") != smoke_guards["applied_mapping_sha256"]:
            raise ActionEffectQualificationError(
                "v0.3 disposable smoke used a different static history guard"
            )
        expected, _base, _mapping = _expected_report(
            repository,
            source=source,
            claim=claim,
            smoke_report=smoke_report,
        )
        expected["smoke_evidence"]["_full_report"] = json.loads(json.dumps(smoke_report))
        expected["storage_preflight"] = storage_preflight
        expected["created_at"] = utc_now()
        if set(expected) != _REPORT_FIELDS:
            raise ActionEffectQualificationError("internal v0.3 report schema is incomplete")
        report_bytes = _canonical_json_bytes(expected)
        with report_path.open("xb") as handle:
            handle.write(report_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(report_bytes).hexdigest()
        with CANONICAL_CHECKSUM.open("x", encoding="ascii") as handle:
            handle.write(f"{digest}  {report_path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The durable claim is intentionally retained.  This attempt identity
        # is terminal and cannot silently be retried under the same root.
        raise
    return verify_action_effect_qualification(
        report_path,
        expected_source_commit=str(source["commit"]),
        expected_tag_object=str(source["tag_object"]),
        repository=repository,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, default=CANONICAL_REPORT)
    parser.add_argument("--tag-payload-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = args.repository.expanduser().resolve()
    if args.tag_payload_only:
        head = _require_git_object(
            _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"]),
            "v0.3 source commit",
        )
        if _git(
            repository,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ):
            raise SystemExit("v0.3 tag payload requires a clean committed repository")
        print(
            json.dumps(
                build_action_effect_tag_payload(
                    repository,
                    source_commit=head,
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return
    evidence = collect_action_effect_qualification(
        repository=repository,
        output=args.output,
    )
    print(json.dumps(evidence.public_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
