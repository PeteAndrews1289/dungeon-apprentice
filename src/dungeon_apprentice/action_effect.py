"""Experience-only action-effect context for recurrent visual policies.

The v0.3 architecture adds one deliberately small information channel to the
existing pixels-only learner:

* which primitive action the policy selected on the preceding transition; and
* whether the next policy-visible RGB observation changed or stayed identical.

Both facts are available to an embodied agent from its own sensorimotor
experience.  The wrapper never reads coordinates, objects, inventory, oracle
state, reward, or trainer-only ``info`` fields.

The feature extractor preserves the legacy NatureCNN parameter names and its
512-dimensional output.  A zero-initialized linear residual embeds the
action-effect context into that output.  Consequently a freshly transplanted
model is bit-for-bit behavior-equivalent to its legacy parent, while PPO can
subsequently learn how to use the new context.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import gymnasium as gym
import numpy as np

ACTION_COUNT = 7
EFFECT_OUTCOME_COUNT = 2
ACTION_EFFECT_DIM = ACTION_COUNT + EFFECT_OUTCOME_COUNT
IMAGE_KEY = "image"
ACTION_EFFECT_KEY = "action_effect"


class ActionEffectError(RuntimeError):
    """Raised when the action-effect boundary or transplant fails closed."""


class ActionEffectMode(StrEnum):
    """Matched v0.3 observation interventions."""

    SHAM = "sham"
    ACTION_EFFECT = "action-effect"


def observations_equal(before: Any, after: Any) -> bool:
    """Return whether two policy-visible observations are byte-identical."""

    left = np.asarray(before)
    right = np.asarray(after)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def action_effect_vector(
    action: int,
    *,
    visible_changed: bool,
) -> np.ndarray:
    """Encode one known transition as action one-hot plus outcome one-hot."""

    selected = int(action)
    if not 0 <= selected < ACTION_COUNT:
        raise ValueError(f"action must be in [0, {ACTION_COUNT})")
    value = np.zeros((ACTION_EFFECT_DIM,), dtype=np.float32)
    value[selected] = 1.0
    # The two outcome coordinates are explicit so reset (all zeros), changed,
    # and unchanged are three distinct inputs without relying on a magic zero.
    value[ACTION_COUNT + int(not visible_changed)] = 1.0
    return value


def empty_action_effect_vector() -> np.ndarray:
    """Return the episode-start sentinel."""

    return np.zeros((ACTION_EFFECT_DIM,), dtype=np.float32)


class ActionEffectObservation(gym.Wrapper):
    """Expose the immediately preceding visible action effect.

    ``SHAM`` performs the same visible comparison work but emits zero context.
    This keeps the observation space, module bytes, wrapper stack, and
    non-random computation matched between study arms.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: ActionEffectMode | str,
    ) -> None:
        super().__init__(env)
        selected = ActionEffectMode(mode)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("action-effect wrapper requires one RGB Box observation")
        if env.observation_space.dtype != np.uint8:
            raise TypeError("action-effect image observation must use uint8 pixels")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("action-effect wrapper requires a discrete action space")
        if int(env.action_space.n) != ACTION_COUNT:
            raise ValueError("action-effect wrapper requires exactly seven actions")
        self.mode = selected
        self.observation_space = gym.spaces.Dict(
            {
                IMAGE_KEY: env.observation_space,
                ACTION_EFFECT_KEY: gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(ACTION_EFFECT_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self._previous_observation: np.ndarray | None = None

    @staticmethod
    def _value(image: Any, context: np.ndarray) -> dict[str, np.ndarray]:
        return {
            IMAGE_KEY: np.array(image, copy=True),
            ACTION_EFFECT_KEY: np.array(context, dtype=np.float32, copy=True),
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._previous_observation = np.array(observation, copy=True)
        return self._value(observation, empty_action_effect_vector()), dict(info)

    def step(
        self,
        action: Any,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        selected = int(np.asarray(action).item())
        observation, reward, terminated, truncated, info = self.env.step(selected)
        if self._previous_observation is None:
            raise ActionEffectError("action-effect environment stepped before reset")
        changed = not observations_equal(self._previous_observation, observation)
        context = (
            action_effect_vector(selected, visible_changed=changed)
            if self.mode is ActionEffectMode.ACTION_EFFECT
            else empty_action_effect_vector()
        )
        result_info = dict(info)
        result_info.update(
            {
                "action_effect_mode": self.mode.value,
                "action_effect_visible_changed": changed,
                "action_effect_context_active": (
                    self.mode is ActionEffectMode.ACTION_EFFECT
                ),
            }
        )
        self._previous_observation = np.array(observation, copy=True)
        if terminated or truncated:
            # DummyVecEnv retains this transition's Dict observation as
            # terminal_observation before asking for the next reset.
            self._previous_observation = None
        return (
            self._value(observation, context),
            float(reward),
            bool(terminated),
            bool(truncated),
            result_info,
        )


def make_action_effect_extractor() -> type[Any]:
    """Import and return the SB3 feature extractor lazily.

    The project keeps training dependencies optional for environment-only use.
    Returning the class from this function lets callers and tests fail with a
    focused installation message rather than making every package import
    require Torch.
    """

    try:
        import torch as th
        from stable_baselines3.common.torch_layers import NatureCNN
        from torch import nn
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    class ActionEffectNatureCNN(NatureCNN):
        """NatureCNN plus a zero-initialized context residual.

        Inheriting directly from ``NatureCNN`` intentionally retains the
        legacy ``cnn`` and ``linear`` state-dict paths.  That makes named
        parameter and Adam-moment transplantation explicit and auditable.
        """

        def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim: int = 512,
            normalized_image: bool = False,
        ) -> None:
            if not isinstance(observation_space, gym.spaces.Dict):
                raise TypeError("action-effect extractor requires a Dict space")
            if set(observation_space.spaces) != {IMAGE_KEY, ACTION_EFFECT_KEY}:
                raise ValueError("action-effect Dict observation keys changed")
            image_space = observation_space.spaces[IMAGE_KEY]
            context_space = observation_space.spaces[ACTION_EFFECT_KEY]
            if (
                not isinstance(context_space, gym.spaces.Box)
                or context_space.shape != (ACTION_EFFECT_DIM,)
            ):
                raise ValueError("action-effect context shape changed")
            super().__init__(
                image_space,
                features_dim=features_dim,
                normalized_image=normalized_image,
            )
            self.action_effect_encoder = nn.Linear(
                ACTION_EFFECT_DIM,
                features_dim,
                bias=False,
            )
            nn.init.zeros_(self.action_effect_encoder.weight)

        def forward(self, observations: Mapping[str, th.Tensor]) -> th.Tensor:
            visual = super().forward(observations[IMAGE_KEY])
            context = self.action_effect_encoder(
                observations[ACTION_EFFECT_KEY]
            )
            return visual + context

    # A stable module/name helps SB3 archives created in one process load in a
    # later process even though the optional dependency is imported lazily.
    ActionEffectNatureCNN.__module__ = __name__
    ActionEffectNatureCNN.__qualname__ = "ActionEffectNatureCNN"
    return ActionEffectNatureCNN


# Expose one stable top-level class when training dependencies are installed.
# On an environment-only installation the placeholder is resolved only when a
# caller actually requests training.
try:  # pragma: no branch - one branch per installed extra
    ActionEffectNatureCNN = make_action_effect_extractor()
except ActionEffectError:  # pragma: no cover - environment-only install
    ActionEffectNatureCNN = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class TransplantEvidence:
    """Audit record for a legacy-to-action-effect model transplant."""

    inherited_parameter_names: tuple[str, ...]
    new_parameter_names: tuple[str, ...]
    inherited_optimizer_state_names: tuple[str, ...]
    missing_optimizer_state_names: tuple[str, ...]
    inherited_timesteps: int
    inherited_optimizer_updates: int
    effect_encoder_zero: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "inherited_parameter_names": list(self.inherited_parameter_names),
            "new_parameter_names": list(self.new_parameter_names),
            "inherited_optimizer_state_names": list(
                self.inherited_optimizer_state_names
            ),
            "missing_optimizer_state_names": list(
                self.missing_optimizer_state_names
            ),
            "inherited_timesteps": self.inherited_timesteps,
            "inherited_optimizer_updates": self.inherited_optimizer_updates,
            "effect_encoder_zero": self.effect_encoder_zero,
        }


def transplant_legacy_model_state(
    legacy_model: Any,
    action_effect_model: Any,
) -> TransplantEvidence:
    """Copy every legacy policy tensor and named Adam moment exactly.

    Loading an optimizer state dict positionally would silently associate
    moments with the wrong parameters after inserting the new context weight.
    This routine instead joins old and new parameters by stable state-dict
    name, checks every shape, and leaves only the new weight without moments.
    """

    try:
        import torch as th
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    legacy_named = dict(legacy_model.policy.named_parameters())
    new_named = dict(action_effect_model.policy.named_parameters())
    missing = sorted(set(legacy_named) - set(new_named))
    if missing:
        raise ActionEffectError(
            f"action-effect policy lost legacy parameters: {missing}"
        )
    new_only = tuple(sorted(set(new_named) - set(legacy_named)))
    expected_new = ("features_extractor.action_effect_encoder.weight",)
    if new_only != expected_new:
        raise ActionEffectError(
            "action-effect policy has an unexpected parameter delta: "
            f"{list(new_only)}"
        )

    with th.no_grad():
        for name, source in legacy_named.items():
            destination = new_named[name]
            if source.shape != destination.shape:
                raise ActionEffectError(
                    f"legacy parameter shape changed for {name}: "
                    f"{tuple(source.shape)} != {tuple(destination.shape)}"
                )
            destination.copy_(
                source.detach().to(
                    device=destination.device,
                    dtype=destination.dtype,
                )
            )
        new_named[expected_new[0]].zero_()

    legacy_optimizer = legacy_model.policy.optimizer
    new_optimizer = action_effect_model.policy.optimizer
    if len(legacy_optimizer.param_groups) != 1 or len(new_optimizer.param_groups) != 1:
        raise ActionEffectError("action-effect transplant requires one optimizer group")
    for key, value in legacy_optimizer.param_groups[0].items():
        if key != "params":
            new_optimizer.param_groups[0][key] = copy.deepcopy(value)

    inherited_states: list[str] = []
    missing_states: list[str] = []
    new_optimizer.state.clear()
    for name, source in legacy_named.items():
        source_state = legacy_optimizer.state.get(source)
        if source_state is None:
            missing_states.append(name)
            continue
        destination = new_named[name]
        copied: dict[str, Any] = {}
        for key, value in source_state.items():
            copied[key] = (
                value.detach().clone().to(destination.device)
                if th.is_tensor(value)
                else copy.deepcopy(value)
            )
        new_optimizer.state[destination] = copied
        inherited_states.append(name)

    action_effect_model.num_timesteps = int(legacy_model.num_timesteps)
    action_effect_model._n_updates = int(legacy_model._n_updates)
    effect_zero = bool(
        th.count_nonzero(new_named[expected_new[0]].detach()).item() == 0
    )
    if not effect_zero:
        raise ActionEffectError("action-effect residual is not zero after transplant")
    return TransplantEvidence(
        inherited_parameter_names=tuple(sorted(legacy_named)),
        new_parameter_names=new_only,
        inherited_optimizer_state_names=tuple(sorted(inherited_states)),
        missing_optimizer_state_names=tuple(sorted(missing_states)),
        inherited_timesteps=int(action_effect_model.num_timesteps),
        inherited_optimizer_updates=int(action_effect_model._n_updates),
        effect_encoder_zero=effect_zero,
    )


def assert_zero_context_equivalence(
    legacy_model: Any,
    action_effect_model: Any,
    images: np.ndarray,
) -> dict[str, Any]:
    """Prove exact deterministic behavior at the transplant boundary."""

    try:
        import torch as th
        from sb3_contrib.common.recurrent.type_aliases import RNNStates
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    values = np.asarray(images)
    if values.ndim != 4 or tuple(values.shape[1:]) != (3, 56, 56):
        raise ValueError("equivalence images must have shape (batch, 3, 56, 56)")
    batch = int(values.shape[0])
    image_tensor = th.as_tensor(values, device=legacy_model.device)
    context_tensor = th.zeros(
        (batch, ACTION_EFFECT_DIM),
        dtype=th.float32,
        device=action_effect_model.device,
    )
    if legacy_model.device != action_effect_model.device:
        raise ActionEffectError("equivalence models must use the same device")
    hidden_size = int(legacy_model.policy.lstm_actor.hidden_size)
    layers = int(legacy_model.policy.lstm_actor.num_layers)

    def state() -> tuple[th.Tensor, th.Tensor]:
        shape = (layers, batch, hidden_size)
        return (
            th.zeros(shape, device=legacy_model.device),
            th.zeros(shape, device=legacy_model.device),
        )

    states = RNNStates(state(), state())
    episode_starts = th.zeros(
        (batch,),
        dtype=th.float32,
        device=legacy_model.device,
    )
    episode_starts[0] = 1.0
    with th.no_grad():
        legacy_features = legacy_model.policy.extract_features(image_tensor)
        action_effect_features = action_effect_model.policy.extract_features(
            {
                IMAGE_KEY: image_tensor,
                ACTION_EFFECT_KEY: context_tensor,
            }
        )
        legacy_output = legacy_model.policy.forward(
            image_tensor,
            states,
            episode_starts,
            deterministic=True,
        )
        action_effect_output = action_effect_model.policy.forward(
            {
                IMAGE_KEY: image_tensor,
                ACTION_EFFECT_KEY: context_tensor,
            },
            states,
            episode_starts,
            deterministic=True,
        )
    tensors = (
        ("features", legacy_features, action_effect_features),
        ("actions", legacy_output[0], action_effect_output[0]),
        ("values", legacy_output[1], action_effect_output[1]),
        ("log_probabilities", legacy_output[2], action_effect_output[2]),
    )
    mismatched = [label for label, left, right in tensors if not th.equal(left, right)]
    legacy_states = tuple(
        tensor
        for branch in legacy_output[3]
        for tensor in branch
    )
    action_effect_states = tuple(
        tensor
        for branch in action_effect_output[3]
        for tensor in branch
    )
    if any(
        not th.equal(left, right)
        for left, right in zip(
            legacy_states,
            action_effect_states,
            strict=True,
        )
    ):
        mismatched.append("recurrent_states")
    if mismatched:
        raise ActionEffectError(
            "zero-context transplant changed parent behavior: "
            f"{mismatched}"
        )
    return {
        "batch": batch,
        "features_exact": True,
        "actions_exact": True,
        "values_exact": True,
        "log_probabilities_exact": True,
        "recurrent_states_exact": True,
    }


class ActionEffectPredictAdapter:
    """Present a raw-pixel evaluation interface around a Dict policy.

    Frozen evaluators pass one observation at a time and retain the recurrent
    state externally.  This adapter derives the same previous-transition
    context as :class:`ActionEffectObservation` without changing the frozen
    lesson mechanics or letting evaluation update the policy.
    """

    def __init__(
        self,
        model: Any,
        *,
        mode: ActionEffectMode | str,
    ) -> None:
        if not isinstance(model.observation_space, gym.spaces.Dict):
            raise TypeError("action-effect evaluation model must use Dict observations")
        self.model = model
        self.mode = ActionEffectMode(mode)
        self.observation_space = model.observation_space.spaces[IMAGE_KEY]
        self.action_space = model.action_space
        self._previous_observation: np.ndarray | None = None
        self._previous_action: int | None = None

    def predict(
        self,
        observation: Any,
        *,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[Any, Any]:
        image = np.asarray(observation)
        starting = bool(
            episode_start is not None
            and np.asarray(episode_start, dtype=bool).reshape(-1)[0]
        )
        if starting:
            self._previous_observation = None
            self._previous_action = None
        if self._previous_observation is None or self._previous_action is None:
            context = empty_action_effect_vector()
        else:
            changed = not observations_equal(
                self._previous_observation,
                image,
            )
            context = (
                action_effect_vector(
                    self._previous_action,
                    visible_changed=changed,
                )
                if self.mode is ActionEffectMode.ACTION_EFFECT
                else empty_action_effect_vector()
            )
        action, next_state = self.model.predict(
            {
                IMAGE_KEY: image,
                ACTION_EFFECT_KEY: context,
            },
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        self._previous_observation = np.array(image, copy=True)
        self._previous_action = int(np.asarray(action).item())
        return action, next_state

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)
