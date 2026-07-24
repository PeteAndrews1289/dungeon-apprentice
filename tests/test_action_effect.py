from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice.action_effect import (
    ACTION_COUNT,
    ACTION_EFFECT_DIM,
    ACTION_EFFECT_KEY,
    IMAGE_KEY,
    ActionEffectError,
    ActionEffectMode,
    ActionEffectObservation,
    ActionEffectPredictAdapter,
    action_effect_vector,
    assert_zero_context_equivalence,
    empty_action_effect_vector,
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


def test_action_effect_vector_has_explicit_changed_and_unchanged_outcomes() -> None:
    changed = action_effect_vector(4, visible_changed=True)
    unchanged = action_effect_vector(4, visible_changed=False)

    assert changed.shape == unchanged.shape == (ACTION_EFFECT_DIM,)
    assert changed.dtype == unchanged.dtype == np.float32
    assert changed[:ACTION_COUNT].tolist() == unchanged[:ACTION_COUNT].tolist()
    assert changed[:ACTION_COUNT].tolist() == [0, 0, 0, 0, 1, 0, 0]
    assert changed[ACTION_COUNT:].tolist() == [1, 0]
    assert unchanged[ACTION_COUNT:].tolist() == [0, 1]
    assert not empty_action_effect_vector().any()
    with pytest.raises(ValueError):
        action_effect_vector(7, visible_changed=True)


def test_wrapper_schema_reset_and_causal_context() -> None:
    zeros = np.zeros((56, 56, 3), dtype=np.uint8)
    ones = np.ones((56, 56, 3), dtype=np.uint8)
    wrapped = ActionEffectObservation(
        _SequenceEnv([zeros, zeros, ones]),
        mode=ActionEffectMode.ACTION_EFFECT,
    )

    observation, info = wrapped.reset()
    assert set(observation) == {IMAGE_KEY, ACTION_EFFECT_KEY}
    assert observation[IMAGE_KEY].shape == (56, 56, 3)
    assert observation[IMAGE_KEY].dtype == np.uint8
    assert observation[ACTION_EFFECT_KEY].shape == (ACTION_EFFECT_DIM,)
    assert observation[ACTION_EFFECT_KEY].dtype == np.float32
    assert not observation[ACTION_EFFECT_KEY].any()
    assert info["coordinates"] == (999, 999)

    observation, reward, terminated, truncated, info = wrapped.step(5)
    assert reward == pytest.approx(0.125)
    assert not terminated and not truncated
    assert observation[ACTION_EFFECT_KEY].tolist() == (
        action_effect_vector(5, visible_changed=False).tolist()
    )
    assert info["action_effect_visible_changed"] is False

    observation, *_rest, info = wrapped.step(2)
    assert observation[ACTION_EFFECT_KEY].tolist() == (
        action_effect_vector(2, visible_changed=True).tolist()
    )
    assert info["action_effect_visible_changed"] is True


def test_sham_performs_comparison_but_never_exposes_context() -> None:
    zeros = np.zeros((56, 56, 3), dtype=np.uint8)
    ones = np.ones((56, 56, 3), dtype=np.uint8)
    wrapped = ActionEffectObservation(
        _SequenceEnv([zeros, ones, ones]),
        mode=ActionEffectMode.SHAM,
    )
    wrapped.reset()
    first, first_reward, *_rest, first_info = wrapped.step(1)
    second, second_reward, *_rest, second_info = wrapped.step(3)

    assert not first[ACTION_EFFECT_KEY].any()
    assert not second[ACTION_EFFECT_KEY].any()
    assert first_reward == second_reward == pytest.approx(0.125)
    assert first_info["action_effect_visible_changed"] is True
    assert second_info["action_effect_visible_changed"] is False
    assert first_info["action_effect_context_active"] is False


def test_privileged_info_cannot_change_action_effect_observation() -> None:
    pixels = [
        np.zeros((56, 56, 3), dtype=np.uint8),
        np.ones((56, 56, 3), dtype=np.uint8),
    ]
    left = ActionEffectObservation(
        _SequenceEnv(pixels, info_marker="left"),
        mode=ActionEffectMode.ACTION_EFFECT,
    )
    right = ActionEffectObservation(
        _SequenceEnv(pixels, info_marker="right"),
        mode=ActionEffectMode.ACTION_EFFECT,
    )
    left.reset()
    right.reset()
    left_observation, *_ = left.step(4)
    right_observation, *_ = right.step(4)

    assert np.array_equal(
        left_observation[IMAGE_KEY],
        right_observation[IMAGE_KEY],
    )
    assert np.array_equal(
        left_observation[ACTION_EFFECT_KEY],
        right_observation[ACTION_EFFECT_KEY],
    )


def test_vec_transpose_changes_only_image_and_preserves_terminal_effect() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    zeros = np.zeros((56, 56, 3), dtype=np.uint8)
    ones = np.ones((56, 56, 3), dtype=np.uint8)

    def factory() -> ActionEffectObservation:
        return ActionEffectObservation(
            _SequenceEnv([zeros, ones], terminate_at=1),
            mode=ActionEffectMode.ACTION_EFFECT,
        )

    vector = VecTransposeImage(DummyVecEnv([factory]))
    reset = vector.reset()
    assert reset[IMAGE_KEY].shape == (1, 3, 56, 56)
    assert reset[ACTION_EFFECT_KEY].shape == (1, ACTION_EFFECT_DIM)
    assert not reset[ACTION_EFFECT_KEY].any()

    observation, rewards, dones, infos = vector.step(np.array([6]))
    assert rewards.tolist() == pytest.approx([0.125])
    assert dones.tolist() == [True]
    # DummyVecEnv has already auto-reset the live observation.
    assert not observation[ACTION_EFFECT_KEY].any()
    terminal = infos[0]["terminal_observation"]
    assert terminal[IMAGE_KEY].shape == (3, 56, 56)
    assert terminal[ACTION_EFFECT_KEY].tolist() == (
        action_effect_vector(6, visible_changed=True).tolist()
    )


class _RecordingModel:
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                IMAGE_KEY: gym.spaces.Box(
                    0,
                    255,
                    shape=(3, 2, 2),
                    dtype=np.uint8,
                ),
                ACTION_EFFECT_KEY: gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(ACTION_EFFECT_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.inputs: list[dict[str, np.ndarray]] = []

    def predict(self, observation: dict[str, np.ndarray], **_kwargs: Any) -> tuple[int, str]:
        self.inputs.append(
            {
                key: np.array(value, copy=True)
                for key, value in observation.items()
            }
        )
        return 4, "state"


def test_evaluation_adapter_resets_context_for_every_episode() -> None:
    model = _RecordingModel()
    adapter = ActionEffectPredictAdapter(
        model,
        mode=ActionEffectMode.ACTION_EFFECT,
    )
    zeros = np.zeros((3, 2, 2), dtype=np.uint8)
    ones = np.ones((3, 2, 2), dtype=np.uint8)
    adapter.predict(
        zeros,
        episode_start=np.ones((1,), dtype=bool),
        deterministic=True,
    )
    adapter.predict(
        ones,
        episode_start=np.zeros((1,), dtype=bool),
        deterministic=True,
    )
    adapter.predict(
        ones,
        episode_start=np.ones((1,), dtype=bool),
        deterministic=True,
    )

    assert not model.inputs[0][ACTION_EFFECT_KEY].any()
    assert model.inputs[1][ACTION_EFFECT_KEY].tolist() == (
        action_effect_vector(4, visible_changed=True).tolist()
    )
    assert not model.inputs[2][ACTION_EFFECT_KEY].any()


class _LearningPixelEnv(gym.Env[np.ndarray, int]):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            0,
            255,
            shape=(56, 56, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        self.step_count = 0
        return np.zeros((56, 56, 3), dtype=np.uint8), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        value = (self.step_count + int(action)) % 256
        observation = np.full(
            (56, 56, 3),
            value,
            dtype=np.uint8,
        )
        return observation, float(action == 0), self.step_count >= 4, False, {}


@pytest.fixture
def local_model_pair() -> tuple[Any, Any]:
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import RecurrentPPO
    from sb3_contrib.common.recurrent.policies import (
        RecurrentMultiInputActorCriticPolicy,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    from dungeon_apprentice.action_effect import ActionEffectNatureCNN

    legacy_env = VecTransposeImage(DummyVecEnv([_LearningPixelEnv]))
    legacy = RecurrentPPO(
        "CnnLstmPolicy",
        legacy_env,
        learning_rate=2.5e-4,
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
    legacy.learn(total_timesteps=8, progress_bar=False)

    def effect_factory() -> ActionEffectObservation:
        return ActionEffectObservation(
            _LearningPixelEnv(),
            mode=ActionEffectMode.ACTION_EFFECT,
        )

    effect_env = VecTransposeImage(DummyVecEnv([effect_factory]))
    effect = RecurrentPPO(
        RecurrentMultiInputActorCriticPolicy,
        effect_env,
        learning_rate=2.5e-4,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": ActionEffectNatureCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [],
            "ortho_init": False,
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
        },
        seed=999,
        device="cpu",
    )
    return legacy, effect


def test_named_transplant_preserves_topology_state_and_behavior(
    local_model_pair: tuple[Any, Any],
) -> None:
    legacy, effect = local_model_pair
    evidence = transplant_legacy_model_state(legacy, effect)

    assert evidence.new_parameter_names == (
        "features_extractor.action_effect_encoder.weight",
    )
    assert not evidence.missing_optimizer_state_names
    assert len(evidence.inherited_optimizer_state_names) == len(
        dict(legacy.policy.named_parameters())
    )
    assert effect.policy.lstm_actor.input_size == 512
    assert effect.policy.lstm_actor.hidden_size == 256
    assert len(effect.policy.mlp_extractor.policy_net) == 0
    assert len(effect.policy.mlp_extractor.value_net) == 0
    assert len(effect.policy.optimizer.state) == len(
        legacy.policy.optimizer.state
    )
    new_parameter = (
        effect.policy.features_extractor.action_effect_encoder.weight
    )
    assert new_parameter not in effect.policy.optimizer.state
    assert not new_parameter.detach().count_nonzero()
    corpus = np.random.default_rng(7).integers(
        0,
        256,
        size=(3, 3, 56, 56),
        dtype=np.uint8,
    )
    result = assert_zero_context_equivalence(legacy, effect, corpus)
    assert all(
        value is True
        for key, value in result.items()
        if key != "batch"
    )


def test_zero_effect_projection_receives_a_learning_gradient(
    local_model_pair: tuple[Any, Any],
) -> None:
    import torch as th

    legacy, effect = local_model_pair
    transplant_legacy_model_state(legacy, effect)
    effect.policy.optimizer.zero_grad()
    images = th.ones((2, 3, 56, 56), dtype=th.uint8)
    contexts = th.as_tensor(
        np.stack(
            [
                action_effect_vector(4, visible_changed=False),
                action_effect_vector(5, visible_changed=True),
            ]
        )
    )
    features = effect.policy.extract_features(
        {
            IMAGE_KEY: images,
            ACTION_EFFECT_KEY: contexts,
        }
    )
    features.sum().backward()
    gradient = (
        effect.policy.features_extractor.action_effect_encoder.weight.grad
    )
    assert gradient is not None
    assert gradient.detach().count_nonzero()


def test_saved_transplanted_archive_loads_without_custom_objects(
    local_model_pair: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    from sb3_contrib import RecurrentPPO

    legacy, effect = local_model_pair
    transplant_legacy_model_state(legacy, effect)
    path = tmp_path / "effect-model.zip"
    effect.save(path)
    loaded = RecurrentPPO.load(path, device="cpu")

    assert type(loaded.policy.features_extractor).__name__ == (
        "ActionEffectNatureCNN"
    )
    assert set(loaded.observation_space.spaces) == {
        IMAGE_KEY,
        ACTION_EFFECT_KEY,
    }
    original = dict(effect.policy.named_parameters())
    restored = dict(loaded.policy.named_parameters())
    assert set(original) == set(restored)
    assert all(
        np.array_equal(
            original[name].detach().cpu().numpy(),
            restored[name].detach().cpu().numpy(),
        )
        for name in original
    )


def test_transplant_rejects_an_unexpected_new_parameter(
    local_model_pair: tuple[Any, Any],
) -> None:
    import torch

    legacy, effect = local_model_pair
    effect.policy.register_parameter(
        "unexpected_architecture_weight",
        torch.nn.Parameter(torch.zeros(1)),
    )
    with pytest.raises(ActionEffectError, match="unexpected parameter delta"):
        transplant_legacy_model_state(legacy, effect)


def test_action_effect_archive_reloads_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    archive = tmp_path / "cross-process.zip"
    create = r"""
import sys
import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from dungeon_apprentice.action_effect import (
    ACTION_EFFECT_DIM,
    ACTION_EFFECT_KEY,
    IMAGE_KEY,
    ActionEffectNatureCNN,
)

class Env(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            IMAGE_KEY: gym.spaces.Box(0, 255, (56, 56, 3), np.uint8),
            ACTION_EFFECT_KEY: gym.spaces.Box(
                0.0, 1.0, (ACTION_EFFECT_DIM,), np.float32
            ),
        })
        self.action_space = gym.spaces.Discrete(7)
    def reset(self, *, seed=None, options=None):
        return {
            IMAGE_KEY: np.zeros((56, 56, 3), np.uint8),
            ACTION_EFFECT_KEY: np.zeros((ACTION_EFFECT_DIM,), np.float32),
        }, {}
    def step(self, action):
        return self.reset()[0], 0.0, False, False, {}

env = VecTransposeImage(DummyVecEnv([Env]))
model = RecurrentPPO(
    RecurrentMultiInputActorCriticPolicy,
    env,
    n_steps=8,
    batch_size=8,
    n_epochs=1,
    policy_kwargs={
        "features_extractor_class": ActionEffectNatureCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": [],
        "ortho_init": False,
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
    },
    device="cpu",
)
model.save(sys.argv[1])
"""
    load = r"""
import sys
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load(sys.argv[1], device="cpu")
assert type(model.policy.features_extractor).__module__ == (
    "dungeon_apprentice.action_effect"
)
assert type(model.policy.features_extractor).__name__ == "ActionEffectNatureCNN"
print("fresh-process-load-ok")
"""
    subprocess.run(
        [sys.executable, "-c", create, str(archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [sys.executable, "-c", load, str(archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "fresh-process-load-ok"
