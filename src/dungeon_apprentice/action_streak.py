"""Policy-visible ineffective-action trace for recurrent visual policies.

The v0.4 architecture adds one scalar to the legacy pixel observation: the
length of the current run in which one selected primitive action has left the
visible RGB observation unchanged.  An unchanged transition starts a run at
one; the same action increments it, and an action switch starts a new run at
one.  The trace uses only the policy's own actions and consecutive
byte-identical sensor frames. It never reads reward, trainer ``info``,
coordinates, objects, or other privileged state.

The matched control computes the same trace but always emits zero.  The
candidate exposes ``min(count, 9) / 9``.  A zero-initialized, bias-free linear
residual preserves exact legacy behavior at transplantation while adding only
512 trainable parameters.
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
IMAGE_KEY = "image"
INEFFECTIVE_TRACE_KEY = "ineffective_trace"
INEFFECTIVE_TRACE_DIM = 1
INEFFECTIVE_TRACE_CAP = 9


class IneffectiveTraceError(RuntimeError):
    """Raised when the ineffective-trace boundary or transplant fails closed."""


class IneffectiveTraceMode(StrEnum):
    """Matched v0.4 observation interventions."""

    ZERO_TRACE = "trace-sham"
    STREAK_TRACE = "ineffective-trace"


def observations_equal(before: Any, after: Any) -> bool:
    """Return whether two policy-visible observations are byte-identical."""

    left = np.asarray(before)
    right = np.asarray(after)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def empty_ineffective_trace_vector() -> np.ndarray:
    """Return the episode-start and control sentinel."""

    return np.zeros((INEFFECTIVE_TRACE_DIM,), dtype=np.float32)


def encode_ineffective_trace(count: int) -> np.ndarray:
    """Encode a non-negative raw trace count as one bounded float."""

    value = int(count)
    if value != count or value < 0:
        raise ValueError("ineffective trace count must be a non-negative integer")
    normalized = min(value, INEFFECTIVE_TRACE_CAP) / INEFFECTIVE_TRACE_CAP
    return np.asarray([normalized], dtype=np.float32)


def ineffective_trace_vector(
    count: int,
    *,
    mode: IneffectiveTraceMode | str,
) -> np.ndarray:
    """Return the mode-specific trace observation."""

    selected = IneffectiveTraceMode(mode)
    if selected is IneffectiveTraceMode.ZERO_TRACE:
        # Validate the count in both arms even though the control emits zero.
        encode_ineffective_trace(count)
        return empty_ineffective_trace_vector()
    return encode_ineffective_trace(count)


def advance_ineffective_trace(
    count: int,
    *,
    preceding_action: int | None,
    selected_action: int,
    visible_changed: bool,
) -> int:
    """Advance the raw count for one policy-visible transition.

    A visible change resets the count to zero.  The first unchanged action
    after reset and an unchanged action switch both start a new run at one.
    Otherwise the same, visibly ineffective action extends the run by one.
    """

    current = int(count)
    selected = int(selected_action)
    if current != count or current < 0:
        raise ValueError("ineffective trace count must be a non-negative integer")
    if not 0 <= selected < ACTION_COUNT:
        raise ValueError(f"selected action must be in [0, {ACTION_COUNT})")
    if visible_changed:
        return 0
    if preceding_action is None:
        return 1
    preceding = int(preceding_action)
    if not 0 <= preceding < ACTION_COUNT:
        raise ValueError(f"preceding action must be in [0, {ACTION_COUNT})")
    if selected != preceding:
        return 1
    return min(INEFFECTIVE_TRACE_CAP, current + 1)


class IneffectiveTraceObservation(gym.Wrapper):
    """Expose a bounded repeated-action/no-visible-effect trace."""

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: IneffectiveTraceMode | str,
    ) -> None:
        super().__init__(env)
        selected = IneffectiveTraceMode(mode)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("ineffective-trace wrapper requires one RGB Box observation")
        if env.observation_space.dtype != np.uint8:
            raise TypeError("ineffective-trace image observation must use uint8 pixels")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("ineffective-trace wrapper requires a discrete action space")
        if int(env.action_space.n) != ACTION_COUNT:
            raise ValueError("ineffective-trace wrapper requires exactly seven actions")
        self.mode = selected
        self.observation_space = gym.spaces.Dict(
            {
                IMAGE_KEY: env.observation_space,
                INEFFECTIVE_TRACE_KEY: gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(INEFFECTIVE_TRACE_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self._previous_observation: np.ndarray | None = None
        self._previous_action: int | None = None
        self._trace_count = 0

    @staticmethod
    def _value(image: Any, trace: np.ndarray) -> dict[str, np.ndarray]:
        return {
            IMAGE_KEY: np.array(image, copy=True),
            INEFFECTIVE_TRACE_KEY: np.array(trace, dtype=np.float32, copy=True),
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._previous_observation = np.array(observation, copy=True)
        self._previous_action = None
        self._trace_count = 0
        return self._value(
            observation,
            empty_ineffective_trace_vector(),
        ), dict(info)

    def step(
        self,
        action: Any,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        selected = int(np.asarray(action).item())
        observation, reward, terminated, truncated, info = self.env.step(selected)
        if self._previous_observation is None:
            raise IneffectiveTraceError(
                "ineffective-trace environment stepped before reset"
            )
        changed = not observations_equal(self._previous_observation, observation)
        self._trace_count = advance_ineffective_trace(
            self._trace_count,
            preceding_action=self._previous_action,
            selected_action=selected,
            visible_changed=changed,
        )
        trace = ineffective_trace_vector(self._trace_count, mode=self.mode)
        result_info = dict(info)
        result_info.update(
            {
                "ineffective_trace_mode": self.mode.value,
                "ineffective_trace_visible_changed": changed,
                "ineffective_trace_raw_count": self._trace_count,
                "ineffective_trace_context_active": (
                    self.mode is IneffectiveTraceMode.STREAK_TRACE
                ),
            }
        )
        self._previous_observation = np.array(observation, copy=True)
        self._previous_action = selected
        if terminated or truncated:
            # DummyVecEnv retains this transition's Dict observation as
            # terminal_observation before asking for the next reset.
            self._previous_observation = None
            self._previous_action = None
            self._trace_count = 0
        return (
            self._value(observation, trace),
            float(reward),
            bool(terminated),
            bool(truncated),
            result_info,
        )


def make_ineffective_trace_extractor() -> type[Any]:
    """Import and return the SB3 feature extractor lazily."""

    try:
        import torch as th
        from stable_baselines3.common.torch_layers import NatureCNN
        from torch import nn
    except ImportError as error:  # pragma: no cover - dependency message
        raise IneffectiveTraceError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    class IneffectiveTraceNatureCNN(NatureCNN):
        """NatureCNN plus a zero-initialized one-scalar residual."""

        def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim: int = 512,
            normalized_image: bool = False,
        ) -> None:
            if not isinstance(observation_space, gym.spaces.Dict):
                raise TypeError("ineffective-trace extractor requires a Dict space")
            if set(observation_space.spaces) != {
                IMAGE_KEY,
                INEFFECTIVE_TRACE_KEY,
            }:
                raise ValueError("ineffective-trace Dict observation keys changed")
            image_space = observation_space.spaces[IMAGE_KEY]
            trace_space = observation_space.spaces[INEFFECTIVE_TRACE_KEY]
            if (
                not isinstance(trace_space, gym.spaces.Box)
                or trace_space.shape != (INEFFECTIVE_TRACE_DIM,)
            ):
                raise ValueError("ineffective-trace context shape changed")
            super().__init__(
                image_space,
                features_dim=features_dim,
                normalized_image=normalized_image,
            )
            self.ineffective_trace_encoder = nn.Linear(
                INEFFECTIVE_TRACE_DIM,
                features_dim,
                bias=False,
            )
            nn.init.zeros_(self.ineffective_trace_encoder.weight)

        def forward(self, observations: Mapping[str, th.Tensor]) -> th.Tensor:
            visual = super().forward(observations[IMAGE_KEY])
            trace = self.ineffective_trace_encoder(
                observations[INEFFECTIVE_TRACE_KEY]
            )
            return visual + trace

    # Stable module/name metadata lets an SB3 archive load in a fresh process.
    IneffectiveTraceNatureCNN.__module__ = __name__
    IneffectiveTraceNatureCNN.__qualname__ = "IneffectiveTraceNatureCNN"
    return IneffectiveTraceNatureCNN


try:  # pragma: no branch - one branch per installed extra
    IneffectiveTraceNatureCNN = make_ineffective_trace_extractor()
except IneffectiveTraceError:  # pragma: no cover - environment-only install
    IneffectiveTraceNatureCNN = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class TransplantEvidence:
    """Audit record for a legacy-to-ineffective-trace transplant."""

    inherited_parameter_names: tuple[str, ...]
    new_parameter_names: tuple[str, ...]
    inherited_optimizer_state_names: tuple[str, ...]
    missing_optimizer_state_names: tuple[str, ...]
    inherited_timesteps: int
    inherited_optimizer_updates: int
    trace_encoder_zero: bool

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
            "trace_encoder_zero": self.trace_encoder_zero,
        }


def transplant_legacy_model_state(
    legacy_model: Any,
    ineffective_trace_model: Any,
) -> TransplantEvidence:
    """Copy every legacy policy tensor and named Adam moment exactly."""

    try:
        import torch as th
    except ImportError as error:  # pragma: no cover - dependency message
        raise IneffectiveTraceError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    legacy_named = dict(legacy_model.policy.named_parameters())
    new_named = dict(ineffective_trace_model.policy.named_parameters())
    missing = sorted(set(legacy_named) - set(new_named))
    if missing:
        raise IneffectiveTraceError(
            f"ineffective-trace policy lost legacy parameters: {missing}"
        )
    new_only = tuple(sorted(set(new_named) - set(legacy_named)))
    expected_new = ("features_extractor.ineffective_trace_encoder.weight",)
    if new_only != expected_new:
        raise IneffectiveTraceError(
            "ineffective-trace policy has an unexpected parameter delta: "
            f"{list(new_only)}"
        )

    with th.no_grad():
        for name, source in legacy_named.items():
            destination = new_named[name]
            if source.shape != destination.shape:
                raise IneffectiveTraceError(
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
    new_optimizer = ineffective_trace_model.policy.optimizer
    if len(legacy_optimizer.param_groups) != 1 or len(new_optimizer.param_groups) != 1:
        raise IneffectiveTraceError(
            "ineffective-trace transplant requires one optimizer group"
        )
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

    ineffective_trace_model.num_timesteps = int(legacy_model.num_timesteps)
    ineffective_trace_model._n_updates = int(legacy_model._n_updates)
    trace_zero = bool(
        th.count_nonzero(new_named[expected_new[0]].detach()).item() == 0
    )
    if not trace_zero:
        raise IneffectiveTraceError(
            "ineffective-trace residual is not zero after transplant"
        )
    return TransplantEvidence(
        inherited_parameter_names=tuple(sorted(legacy_named)),
        new_parameter_names=new_only,
        inherited_optimizer_state_names=tuple(sorted(inherited_states)),
        missing_optimizer_state_names=tuple(sorted(missing_states)),
        inherited_timesteps=int(ineffective_trace_model.num_timesteps),
        inherited_optimizer_updates=int(ineffective_trace_model._n_updates),
        trace_encoder_zero=trace_zero,
    )


def assert_zero_context_equivalence(
    legacy_model: Any,
    ineffective_trace_model: Any,
    images: np.ndarray,
) -> dict[str, Any]:
    """Prove exact deterministic behavior at the transplant boundary."""

    try:
        import torch as th
        from sb3_contrib.common.recurrent.type_aliases import RNNStates
    except ImportError as error:  # pragma: no cover - dependency message
        raise IneffectiveTraceError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    values = np.asarray(images)
    if values.ndim != 4 or tuple(values.shape[1:]) != (3, 56, 56):
        raise ValueError("equivalence images must have shape (batch, 3, 56, 56)")
    batch = int(values.shape[0])
    image_tensor = th.as_tensor(values, device=legacy_model.device)
    trace_tensor = th.zeros(
        (batch, INEFFECTIVE_TRACE_DIM),
        dtype=th.float32,
        device=ineffective_trace_model.device,
    )
    if legacy_model.device != ineffective_trace_model.device:
        raise IneffectiveTraceError(
            "equivalence models must use the same device"
        )
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
        trace_features = ineffective_trace_model.policy.extract_features(
            {
                IMAGE_KEY: image_tensor,
                INEFFECTIVE_TRACE_KEY: trace_tensor,
            }
        )
        legacy_output = legacy_model.policy.forward(
            image_tensor,
            states,
            episode_starts,
            deterministic=True,
        )
        trace_output = ineffective_trace_model.policy.forward(
            {
                IMAGE_KEY: image_tensor,
                INEFFECTIVE_TRACE_KEY: trace_tensor,
            },
            states,
            episode_starts,
            deterministic=True,
        )
    tensors = (
        ("features", legacy_features, trace_features),
        ("actions", legacy_output[0], trace_output[0]),
        ("values", legacy_output[1], trace_output[1]),
        ("log_probabilities", legacy_output[2], trace_output[2]),
    )
    mismatched = [label for label, left, right in tensors if not th.equal(left, right)]
    legacy_states = tuple(tensor for branch in legacy_output[3] for tensor in branch)
    trace_states = tuple(tensor for branch in trace_output[3] for tensor in branch)
    if any(
        not th.equal(left, right)
        for left, right in zip(legacy_states, trace_states, strict=True)
    ):
        mismatched.append("recurrent_states")
    if mismatched:
        raise IneffectiveTraceError(
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


class IneffectiveTracePredictAdapter:
    """Present a raw-pixel evaluation interface around a Dict policy."""

    def __init__(
        self,
        model: Any,
        *,
        mode: IneffectiveTraceMode | str,
    ) -> None:
        if not isinstance(model.observation_space, gym.spaces.Dict):
            raise TypeError(
                "ineffective-trace evaluation model must use Dict observations"
            )
        if set(model.observation_space.spaces) != {
            IMAGE_KEY,
            INEFFECTIVE_TRACE_KEY,
        }:
            raise ValueError(
                "ineffective-trace evaluation observation keys changed"
            )
        trace_space = model.observation_space.spaces[INEFFECTIVE_TRACE_KEY]
        if (
            not isinstance(trace_space, gym.spaces.Box)
            or trace_space.shape != (INEFFECTIVE_TRACE_DIM,)
        ):
            raise ValueError(
                "ineffective-trace evaluation context shape changed"
            )
        self.model = model
        self.mode = IneffectiveTraceMode(mode)
        self.observation_space = model.observation_space.spaces[IMAGE_KEY]
        self.action_space = model.action_space
        self._previous_observation: np.ndarray | None = None
        self._pending_action: int | None = None
        self._preceding_action: int | None = None
        self._trace_count = 0

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
            self._pending_action = None
            self._preceding_action = None
            self._trace_count = 0
        if self._previous_observation is None or self._pending_action is None:
            self._trace_count = 0
        else:
            changed = not observations_equal(
                self._previous_observation,
                image,
            )
            self._trace_count = advance_ineffective_trace(
                self._trace_count,
                preceding_action=self._preceding_action,
                selected_action=self._pending_action,
                visible_changed=changed,
            )
        trace = ineffective_trace_vector(self._trace_count, mode=self.mode)
        action, next_state = self.model.predict(
            {
                IMAGE_KEY: image,
                INEFFECTIVE_TRACE_KEY: trace,
            },
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        self._previous_observation = np.array(image, copy=True)
        self._preceding_action = self._pending_action
        self._pending_action = int(np.asarray(action).item())
        return action, next_state

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)
