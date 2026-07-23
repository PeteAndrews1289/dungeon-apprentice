from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from dungeon_apprentice import u2_qualification_anchor as anchor


def _evidence() -> SimpleNamespace:
    public = {
        "report": str(anchor.CANONICAL_REPORT),
        "report_sha256": "a" * 64,
        "report_byte_length": 123_456,
        "attempt_id": "u2-preflight-v0.2-u2-20260723-attempt-1",
        "attempt_sha256": "b" * 64,
        "claim_id": "c" * 64,
        "claim_sha256": "d" * 64,
        "source_commit": "e" * 40,
        "generator_profile": "v0.2-u2-separated-v1",
        "generator_profile_version": 1,
    }
    return SimpleNamespace(public_dict=lambda: public)


def test_anchor_message_is_deterministic_strict_one_line() -> None:
    message = anchor.anchor_message(_evidence(), source_commit="e" * 40)
    assert "\n" not in message
    payload = json.loads(message)
    assert set(payload) == anchor.ANCHOR_FIELDS
    assert payload["tag"] == anchor.ANCHOR_TAG
    assert payload["report_sha256"] == "a" * 64
    assert payload["attempt_sha256"] == "b" * 64
    assert payload["claim_id"] == "c" * 64


def test_anchor_builder_rejects_noncanonical_report_and_source() -> None:
    public = _evidence().public_dict()
    public["report"] = "/tmp/forged-report.json"
    with pytest.raises(anchor.U2QualificationAnchorError, match="canonical"):
        anchor.build_anchor_payload(public, source_commit="e" * 40)

    with pytest.raises(anchor.U2QualificationAnchorError, match="commits differ"):
        anchor.build_anchor_payload(_evidence(), source_commit="f" * 40)


def _runner(
    *,
    message: str,
    head: str = "e" * 40,
    remote_object: str = "1" * 40,
    tag_type: str = "tag",
    peeled: str = "e" * 40,
    remote_url: str = anchor.EXPECTED_ORIGIN_URL,
):
    outputs = {
        ("rev-parse", "--verify", "HEAD^{commit}"): head,
        ("cat-file", "-t", f"refs/tags/{anchor.ANCHOR_TAG}"): tag_type,
        (
            "rev-parse",
            "--verify",
            f"refs/tags/{anchor.ANCHOR_TAG}",
        ): "1" * 40,
        ("rev-list", "-n", "1", f"refs/tags/{anchor.ANCHOR_TAG}"): peeled,
        ("remote", "get-url", anchor.ANCHOR_REMOTE): remote_url,
        (
            "ls-remote",
            "--tags",
            anchor.ANCHOR_REMOTE,
            f"refs/tags/{anchor.ANCHOR_TAG}",
        ): f"{remote_object}\trefs/tags/{anchor.ANCHOR_TAG}",
        (
            "for-each-ref",
            "--format=%(contents)",
            f"refs/tags/{anchor.ANCHOR_TAG}",
        ): message,
    }

    def run(command, **_kwargs):
        assert command[0] == "git"
        key = tuple(command[1:])
        if key not in outputs:
            raise AssertionError(f"unexpected git call: {command}")
        return subprocess.CompletedProcess(command, 0, outputs[key] + "\n", "")

    return run


def test_external_anchor_requires_same_annotated_object_on_origin(
    tmp_path: Path,
) -> None:
    message = anchor.anchor_message(_evidence(), source_commit="e" * 40)
    verified = anchor.verify_external_anchor(
        tmp_path,
        runner=_runner(message=message),
    )
    assert verified.tag_object == "1" * 40
    assert verified.source_commit == "e" * 40
    assert verified.report_sha256 == "a" * 64
    assert verified.attempt_id.endswith("attempt-1")

    historical = anchor.verify_external_anchor(
        tmp_path,
        expected_source_commit="e" * 40,
        runner=_runner(message=message, head="f" * 40),
    )
    assert historical.source_commit == "e" * 40

    with pytest.raises(anchor.U2QualificationAnchorError, match="annotated"):
        anchor.verify_external_anchor(
            tmp_path,
            runner=_runner(message=message, tag_type="commit"),
        )
    with pytest.raises(anchor.U2QualificationAnchorError, match="origin"):
        anchor.verify_external_anchor(
            tmp_path,
            runner=_runner(message=message, remote_object="2" * 40),
        )


def test_external_anchor_rejects_wrong_commit_remote_and_unknown_fields(
    tmp_path: Path,
) -> None:
    message = anchor.anchor_message(_evidence(), source_commit="e" * 40)
    with pytest.raises(anchor.U2QualificationAnchorError, match="active source"):
        anchor.verify_external_anchor(
            tmp_path,
            runner=_runner(message=message, peeled="f" * 40),
        )
    with pytest.raises(anchor.U2QualificationAnchorError, match="frozen GitHub"):
        anchor.verify_external_anchor(
            tmp_path,
            runner=_runner(message=message, remote_url="https://example.invalid/repo"),
        )

    payload = json.loads(message)
    payload["extra"] = True
    with pytest.raises(anchor.U2QualificationAnchorError, match="fields"):
        anchor.verify_external_anchor(
            tmp_path,
            runner=_runner(
                message=json.dumps(payload, sort_keys=True, separators=(",", ":"))
            ),
        )


def test_bootstrap_verifier_binds_measured_bytes_and_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dungeon_apprentice import v02_u2_qualify as qualify

    directory = tmp_path / "qualification"
    directory.mkdir()
    report = directory / "report.json"
    report.write_bytes(b'{"result":"passed"}\n')
    attempt = directory / "attempt.json"
    attempt.write_text(
        json.dumps(
            {
                "attempt_id": qualify.QUALIFICATION_ATTEMPT_ID,
                "claim_id": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "directory": directory,
        "report": report,
        "checksum": directory / "report.json.sha256",
        "attempt": attempt,
        "claim": directory / "launch-claim.json",
    }
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(
        anchor,
        "_git",
        lambda _repo, arguments, **_kwargs: (
            "e" * 40 if arguments[0] == "rev-parse" else ""
        ),
    )
    monkeypatch.setattr(qualify, "canonical_qualification_paths", lambda: paths)

    def verify(path: Path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(qualify, "verify_u2_qualification_report", verify)
    evidence = anchor.qualification_evidence_for_anchor(
        tmp_path,
        expected_source_commit="e" * 40,
    )

    assert evidence is sentinel
    assert captured["path"] == report
    assert captured["expected_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert captured["expected_attempt_id"] == qualify.QUALIFICATION_ATTEMPT_ID
    assert captured["expected_claim_id"] == "c" * 64


def test_publisher_creates_pushes_and_verifies_fixed_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    remote_exists = False
    verified = object()
    message = anchor.anchor_message(_evidence(), source_commit="e" * 40)

    def fake_git(_repo, arguments, **_kwargs):
        nonlocal remote_exists
        command = tuple(arguments)
        calls.append(command)
        if command[0] == "rev-parse":
            return "e" * 40
        if command[0] == "status":
            return ""
        if command[:2] == ("remote", "get-url"):
            return anchor.EXPECTED_ORIGIN_URL
        if command[0] == "ls-remote":
            return (
                f"{'1' * 40}\trefs/tags/{anchor.ANCHOR_TAG}"
                if remote_exists
                else ""
            )
        if command[0] == "tag":
            assert message in command
            return ""
        if command[0] == "push":
            remote_exists = True
            return ""
        raise AssertionError(command)

    def runner(command, **_kwargs):
        assert command[:3] == ["git", "show-ref", "--verify"]
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(anchor, "_git", fake_git)
    monkeypatch.setattr(
        anchor,
        "verify_external_anchor",
        lambda _repo, **_kwargs: verified,
    )

    result = anchor.publish_external_anchor(
        tmp_path,
        _evidence(),
        source_commit="e" * 40,
        runner=runner,
    )

    assert result is verified
    assert any(command[0] == "tag" for command in calls)
    assert any(command[0] == "push" for command in calls)
