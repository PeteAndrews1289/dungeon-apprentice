from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice.action_streak import (
    ACTION_COUNT,
    IMAGE_KEY,
    INEFFECTIVE_TRACE_CAP,
    INEFFECTIVE_TRACE_DIM,
    INEFFECTIVE_TRACE_KEY,
    IneffectiveTraceMode,
    IneffectiveTraceObservation,
    IneffectiveTracePredictAdapter,
    advance_ineffective_trace,
    assert_zero_context_equivalence,
    empty_ineffective_trace_vector,
    encode_ineffective_trace,
    ineffective_trace_vector,
    transplant_legacy_model_state,
)


class _SequenceEnv(gym.Env[np.ndarray, int]):
    metadata: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        observations: list[np.ndarray],
        *,
        terminate_at: int | None = None,
        info_marker: str = "ignored",
    ) -> None:
        self.observations = [np.array(value, copy=True) for value in observations]
        self.observation_space = gym.spaces.Box(
            0,
            255,
            shape=self.observations[0].shape,
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.terminate_at = terminate_at
        self.info_marker = info_marker
        self.index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        self.index = 0
        return np.array(self.observations[0], copy=True), {
            "coordinates": (999, 999),
            "marker": self.info_marker,
        }

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        self.index += 1
        observation = self.observations[
            min(self.index, len(self.observations) - 1)
        ]
        return (
            np.array(observation, copy=True),
            0.125,
            self.terminate_at is not None and self.index >= self.terminate_at,
            False,
            {
                "coordinates": (-999, -999),
                "milestones": {"success": True},
                "marker": self.info_marker,
            },
        )


def test_trace_encoding_is_bounded_and_reset_is_distinct() -> None:
    assert not empty_ineffective_trace_vector().any()
    assert encode_ineffective_trace(0).tolist() == [0.0]
    assert encode_ineffective_trace(1).tolist() == pytest.approx([1 / 9])
    assert encode_ineffective_trace(INEFFECTIVE_TRACE_CAP).tolist() == [1.0]
    assert encode_ineffective_trace(INEFFECTIVE_TRACE_CAP + 100).tolist() == [1.0]
    assert ineffective_trace_vector(
        7,
        mode=IneffectiveTraceMode.ZERO_TRACE,
    ).tolist() == [0.0]
    with pytest.raises(ValueError, match="non-negative integer"):
        encode_ineffective_trace(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        encode_ineffective_trace(1.5)  # type: ignore[arg-type]


def test_trace_counter_requires_same_action_and_unchanged_pixels() -> None:
    assert (
        advance_ineffective_trace(
            0,
            preceding_action=None,
            selected_action=4,
            visible_changed=False,
        )
        == 1
    )
    assert (
        advance_ineffective_trace(
            0,
            preceding_action=4,
            selected_action=4,
            visible_changed=False,
        )
        == 1
    )
    assert (
        advance_ineffective_trace(
            6,
            preceding_action=4,
            selected_action=4,
            visible_changed=False,
        )
        == 7
    )
    assert (
        advance_ineffective_trace(
            INEFFECTIVE_TRACE_CAP,
            preceding_action=4,
            selected_action=4,
            visible_changed=False,
        )
        == INEFFECTIVE_TRACE_CAP
    )
    assert (
        advance_ineffective_trace(
            6,
            preceding_action=4,
            selected_action=3,
            visible_changed=False,
        )
        == 1
    )
    assert (
        advance_ineffective_trace(
            6,
            preceding_action=4,
            selected_action=4,
            visible_changed=True,
        )
        == 0
    )
    with pytest.raises(ValueError, match="selected action"):
        advance_ineffective_trace(
            0,
            preceding_action=4,
            selected_action=7,
            visible_changed=False,
        )


def test_candidate_wrapper_resets_on_action_switch_and_visible_change() -> None:
    zeros = np.zeros((56, 56, 3), dtype=np.uint8)
    ones = np.ones((56, 56, 3), dtype=np.uint8)
    wrapped = IneffectiveTraceObservation(
        _SequenceEnv([zeros, zeros, zeros, zeros, ones, ones]),
        mode=IneffectiveTraceMode.STREAK_TRACE,
    )

    observation, info = wrapped.reset()
    assert set(observation) == {IMAGE_KEY, INEFFECTIVE_TRACE_KEY}
    assert observation[IMAGE_KEY].shape == (56, 56, 3)
    assert observation[IMAGE_KEY].dtype == np.uint8
    assert observation[INEFFECTIVE_TRACE_KEY].shape == (INEFFECTIVE_TRACE_DIM,)
    assert observation[INEFFECTIVE_TRACE_KEY].dtype == np.float32
    assert not observation[INEFFECTIVE_TRACE_KEY].any()
    assert info["coordinates"] == (999, 999)

    first, reward, terminated, truncated, first_info = wrapped.step(2)
    assert reward == pytest.approx(0.125)
    assert not terminated and not truncated
    assert first[INEFFECTIVE_TRACE_KEY].tolist() == pytest.approx([1 / 9])
    assert first_info["ineffective_trace_raw_count"] == 1

    repeated, *_rest, repeated_info = wrapped.step(2)
    assert repeated[INEFFECTIVE_TRACE_KEY].tolist() == pytest.approx([2 / 9])
    assert repeated_info["ineffective_trace_raw_count"] == 2

    switched, *_rest, switched_info = wrapped.step(3)
    assert switched[INEFFECTIVE_TRACE_KEY].tolist() == pytest.approx([1 / 9])
    assert switched_info["ineffective_trace_raw_count"] == 1

    changed, *_rest, changed_info = wrapped.step(3)
    assert changed[INEFFECTIVE_TRACE_KEY].tolist() == [0.0]
    assert changed_info["ineffective_trace_visible_changed"] is True
    assert changed_info["ineffective_trace_raw_count"] == 0

    restarted, *_rest, restarted_info = wrapped.step(3)
    assert restarted[INEFFECTIVE_TRACE_KEY].tolist() == pytest.approx([1 / 9])
    assert restarted_info["ineffective_trace_raw_count"] == 1


def test_control_computes_trace_but_emits_exact_zero() -> None:
    zeros = np.zeros((56, 56, 3), dtype=np.uint8)
    control = IneffectiveTraceObservation(
        _SequenceEnv([zeros, zeros, zeros]),
        mode=IneffectiveTraceMode.ZERO_TRACE,
    )
    candidate = IneffectiveTraceObservation(
        _SequenceEnv([zeros, zeros, zeros]),
        mode=IneffectiveTraceMode.STREAK_TRACE,
    )
    control.reset()
    candidate.reset()
    control.step(5)
    candidate.step(5)
    control_second, *_rest, control_info = control.step(5)
    candidate_second, *_rest, candidate_info = candidate.step(5)

    assert control_second[INEFFECTIVE_TRACE_KEY].tolist() == [0.0]
    assert candidate_second[INEFFECTIVE_TRACE_KEY].tolist() == pytest.approx(
        [2 / 9]
    )
    assert control_info["ineffective_trace_raw_count"] == 2
    assert candidate_info["ineffective_trace_raw_count"] == 2
    assert control_info["ineffective_trace_context_active"] is False
    assert candidate_info["ineffective_trace_context_active"] is True


def test_wrapper_trace_and_raw_counter_saturate_at_frozen_cap() -> None:
    zeros = np.zeros((56, 56, 3), dtype=np.uint8)
    wrapped = IneffectiveTraceObservation(
        _SequenceEnv([zeros]),
        mode=IneffectiveTraceMode.STREAK_TRACE,
    )
    wrapped.reset()
    latest: dict[str, np.ndarray] | None = None
    info: dict[str, Any] = {}
    for _ in range(INEFFECTIVE_TRACE_CAP + 5):
        latest, *_rest, info = wrapped.step(1)

    assert latest is not None
    assert latest[INEFFECTIVE_TRACE_KEY].tolist() == [1.0]
    assert info["ineffective_trace_raw_count"] == INEFFECTIVE_TRACE_CAP


def test_episode_reset_clears_trace_and_terminal_observation_retains_it() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    zeros = np.zeros((56, 56, 3), dtype=np.uint8)

    def factory() -> IneffectiveTraceObservation:
        return IneffectiveTraceObservation(
            _SequenceEnv([zeros, zeros, zeros], terminate_at=2),
            mode=IneffectiveTraceMode.STREAK_TRACE,
        )

    vector = VecTransposeImage(DummyVecEnv([factory]))
    reset = vector.reset()
    assert reset[IMAGE_KEY].shape == (1, 3, 56, 56)
    assert reset[INEFFECTIVE_TRACE_KEY].shape == (1, INEFFECTIVE_TRACE_DIM)
    assert not reset[INEFFECTIVE_TRACE_KEY].any()
    vector.step(np.array([6]))
    observation, rewards, dones, infos = vector.step(np.array([6]))

    assert rewards.tolist() == pytest.approx([0.125])
    assert dones.tolist() == [True]
    assert not observation[INEFFECTIVE_TRACE_KEY].any()
    terminal = infos[0]["terminal_observation"]
    assert terminal[IMAGE_KEY].shape == (3, 56, 56)
    assert terminal[INEFFECTIVE_TRACE_KEY].tolist() == pytest.approx([2 / 9])


def test_privileged_info_cannot_change_trace_observation() -> None:
    zeros = np.zeros((56, 56, 3), dtype=np.uint8)
    left = IneffectiveTraceObservation(
        _SequenceEnv([zeros], info_marker="left"),
        mode=IneffectiveTraceMode.STREAK_TRACE,
    )
    right = IneffectiveTraceObservation(
        _SequenceEnv([zeros], info_marker="right"),
        mode=IneffectiveTraceMode.STREAK_TRACE,
    )
    left.reset()
    right.reset()
    left.step(4)
    right.step(4)
    left_observation, *_ = left.step(4)
    right_observation, *_ = right.step(4)

    assert np.array_equal(
        left_observation[IMAGE_KEY],
        right_observation[IMAGE_KEY],
    )
    assert np.array_equal(
        left_observation[INEFFECTIVE_TRACE_KEY],
        right_observation[INEFFECTIVE_TRACE_KEY],
    )


class _RecordingModel:
    def __init__(self, actions: list[int]) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                IMAGE_KEY: gym.spaces.Box(
                    0,
                    255,
                    shape=(3, 2, 2),
                    dtype=np.uint8,
                ),
                INEFFECTIVE_TRACE_KEY: gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(INEFFECTIVE_TRACE_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.actions = iter(actions)
        self.inputs: list[dict[str, np.ndarray]] = []

    def predict(
        self,
        observation: dict[str, np.ndarray],
        **_kwargs: Any,
    ) -> tuple[int, str]:
        self.inputs.append(
            {
                key: np.array(value, copy=True)
                for key, value in observation.items()
            }
        )
        return next(self.actions), "state"


def test_evaluation_adapter_reconstructs_wrapper_trace_exactly() -> None:
    zeros = np.zeros((3, 2, 2), dtype=np.uint8)
    ones = np.ones((3, 2, 2), dtype=np.uint8)
    images = [zeros, zeros, zeros, zeros, ones, ones, ones]
    actions = [2, 2, 3, 3, 3, 3, 3]
    model = _RecordingModel(actions)
    adapter = IneffectiveTracePredictAdapter(
        model,
        mode=IneffectiveTraceMode.STREAK_TRACE,
    )

    for index, image in enumerate(images):
        adapter.predict(
            image,
            episode_start=np.asarray([index == 0]),
            deterministic=True,
        )

    expected = [0, 1 / 9, 2 / 9, 1 / 9, 0, 1 / 9, 2 / 9]
    observed = [
        item[INEFFECTIVE_TRACE_KEY].tolist()[0]
        for item in model.inputs
    ]
    assert observed == pytest.approx(expected)


def test_evaluation_adapter_reset_and_control_are_exact_zero() -> None:
    pixels = np.zeros((3, 2, 2), dtype=np.uint8)
    model = _RecordingModel([1, 1, 1, 1])
    adapter = IneffectiveTracePredictAdapter(
        model,
        mode=IneffectiveTraceMode.ZERO_TRACE,
    )
    adapter.predict(pixels, episode_start=np.asarray([True]))
    adapter.predict(pixels, episode_start=np.asarray([False]))
    adapter.predict(pixels, episode_start=np.asarray([False]))
    adapter.predict(pixels, episode_start=np.asarray([True]))

    assert all(
        item[INEFFECTIVE_TRACE_KEY].tolist() == [0.0]
        for item in model.inputs
    )


def test_ineffective_trace_encoder_is_one_by_512_and_zero_initialized() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    import torch as th

    from dungeon_apprentice.action_streak import IneffectiveTraceNatureCNN

    observation_space = gym.spaces.Dict(
        {
            IMAGE_KEY: gym.spaces.Box(
                0,
                255,
                shape=(3, 56, 56),
                dtype=np.uint8,
            ),
            INEFFECTIVE_TRACE_KEY: gym.spaces.Box(
                0.0,
                1.0,
                shape=(INEFFECTIVE_TRACE_DIM,),
                dtype=np.float32,
            ),
        }
    )
    extractor = IneffectiveTraceNatureCNN(observation_space)
    encoder = extractor.ineffective_trace_encoder

    assert encoder.weight.shape == (512, 1)
    assert encoder.bias is None
    assert not encoder.weight.detach().count_nonzero()
    images = th.rand((2, 3, 56, 56), dtype=th.float32)
    zero = extractor(
        {
            IMAGE_KEY: images,
            INEFFECTIVE_TRACE_KEY: th.zeros((2, 1)),
        }
    )
    one = extractor(
        {
            IMAGE_KEY: images,
            INEFFECTIVE_TRACE_KEY: th.ones((2, 1)),
        }
    )
    assert th.equal(zero, one)


class _PixelEnv(gym.Env[np.ndarray, int]):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            0,
            255,
            shape=(56, 56, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        return np.zeros((56, 56, 3), dtype=np.uint8), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        return (
            np.zeros((56, 56, 3), dtype=np.uint8),
            0.0,
            False,
            False,
            {},
        )


def test_named_transplant_and_zero_context_equivalence() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import RecurrentPPO
    from sb3_contrib.common.recurrent.policies import (
        RecurrentMultiInputActorCriticPolicy,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    from dungeon_apprentice.action_streak import IneffectiveTraceNatureCNN

    legacy_env = VecTransposeImage(DummyVecEnv([_PixelEnv]))
    legacy = RecurrentPPO(
        "CnnLstmPolicy",
        legacy_env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={
            "net_arch": [],
            "ortho_init": False,
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
        },
        seed=123,
        device="cpu",
    )

    def trace_factory() -> IneffectiveTraceObservation:
        return IneffectiveTraceObservation(
            _PixelEnv(),
            mode=IneffectiveTraceMode.STREAK_TRACE,
        )

    trace_env = VecTransposeImage(DummyVecEnv([trace_factory]))
    trace = RecurrentPPO(
        RecurrentMultiInputActorCriticPolicy,
        trace_env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": IneffectiveTraceNatureCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [],
            "ortho_init": False,
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
        },
        seed=999,
        device="cpu",
    )
    evidence = transplant_legacy_model_state(legacy, trace)

    assert evidence.new_parameter_names == (
        "features_extractor.ineffective_trace_encoder.weight",
    )
    assert evidence.trace_encoder_zero is True
    assert evidence.inherited_timesteps == legacy.num_timesteps
    assert evidence.inherited_optimizer_updates == legacy._n_updates
    assert "trace_encoder_zero" in evidence.public_dict()
    corpus = np.random.default_rng(7).integers(
        0,
        256,
        size=(3, 3, 56, 56),
        dtype=np.uint8,
    )
    result = assert_zero_context_equivalence(legacy, trace, corpus)
    assert all(
        value is True
        for key, value in result.items()
        if key != "batch"
    )
