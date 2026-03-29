from __future__ import annotations

import sys
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

if sys.version_info < (3, 14):
    try:  # Prefer official OpenEnv base types when available.
        from openenv.core.env_server.types import (  # type: ignore
            Action as OpenEnvAction,
            Observation as OpenEnvObservation,
            State as OpenEnvState,
        )
    except Exception:  # pragma: no cover - local fallback for unsupported runtimes
        OpenEnvAction = None
        OpenEnvObservation = None
        OpenEnvState = None
else:  # pragma: no cover - Python 3.14 currently breaks transitive openenv imports
    OpenEnvAction = None
    OpenEnvObservation = None
    OpenEnvState = None

if OpenEnvAction is None or OpenEnvObservation is None or OpenEnvState is None:
    class OpenEnvAction(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        metadata: Dict[str, Any] = Field(
            default_factory=dict,
            description="Additional metadata for the action",
        )

    class OpenEnvObservation(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        done: bool = Field(default=False, description="Whether the episode ended")
        reward: Optional[float] = Field(
            default=None,
            description="Reward emitted by the previous transition",
        )
        metadata: Dict[str, Any] = Field(
            default_factory=dict,
            description="Additional metadata for the observation",
        )

    class OpenEnvState(BaseModel):
        model_config = ConfigDict(
            extra="allow",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        episode_id: Optional[str] = Field(
            default=None,
            description="Unique identifier for the active episode",
        )
        step_count: int = Field(
            default=0,
            ge=0,
            description="Number of completed environment steps",
        )


ContentType = Literal["text", "visual", "audio", "mixed"]
Pace = Literal["slow", "normal", "fast"]
ContrastMode = Literal["normal", "high", "low"]
AnimationSpeed = Literal["none", "slow", "normal", "fast"]
Subject = Literal["math", "reading", "science", "life_skills"]


class Action(OpenEnvAction):
    """Action chosen by the agent to adapt a live tutoring session."""

    content_type: ContentType = Field(
        ...,
        description="Primary delivery modality for the next learning item.",
    )
    difficulty: int = Field(
        ...,
        ge=1,
        le=5,
        description="Difficulty target for the next question.",
    )
    pace: Pace = Field(
        ...,
        description="Presentation pace for the next interaction.",
    )
    hint_level: int = Field(
        ...,
        ge=0,
        le=3,
        description="Amount of scaffolding provided to the learner.",
    )
    take_break: bool = Field(
        ...,
        description="Whether to insert a calming sensory break before continuing.",
    )
    font_size: int = Field(
        default=16,
        ge=12,
        le=24,
        description="Accessible text size for the current activity.",
    )
    contrast_mode: ContrastMode = Field(
        default="normal",
        description="Display contrast profile used for the current activity.",
    )
    animation_speed: AnimationSpeed = Field(
        default="slow",
        description="Maximum animation speed allowed in the interface.",
    )
    subject: Optional[Subject] = Field(
        default=None,
        description="Curriculum area to emphasize for the next question.",
    )


class Observation(OpenEnvObservation):
    """Observable signals available to the tutoring agent after each step."""

    stress_signal: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Noisy estimate of the learner's current stress level.",
    )
    engagement: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated engagement with the current tutoring configuration.",
    )
    response_time_norm: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized response time where lower values indicate faster responses.",
    )
    error_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rolling error rate over the most recent interactions.",
    )
    last_correct: Optional[bool] = Field(
        default=None,
        description="Whether the learner answered the previous question correctly.",
    )
    step: int = Field(
        ...,
        ge=0,
        description="Current step index within the episode.",
    )
    profile_hint: str = Field(
        ...,
        description="Partial sensory profile information visible to the agent.",
    )


class HiddenState(BaseModel):
    """Internal learner state used by the simulator and graders."""

    true_stress: float = Field(..., ge=0.0, le=1.0)
    fatigue: float = Field(..., ge=0.0, le=1.0)
    preference: ContentType
    sensory_overload_count: int = Field(default=0, ge=0)
    current_context: str = Field(
        default="warmup",
        description="Current classroom context, hidden from the agent except through signals.",
    )


class State(OpenEnvState):
    """Complete episode state returned by `/state`."""

    task_id: str = Field(..., description="Current task identifier.")
    max_steps: int = Field(..., ge=1, description="Maximum steps for the task.")
    hidden: HiddenState
    total_learning: float = Field(default=0.0, description="Accumulated learning gain.")
    total_reward: float = Field(default=0.0, description="Accumulated shaped reward.")
    history: List[Action] = Field(
        default_factory=list,
        description="All actions taken so far in the episode.",
    )
    trajectory: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed transition log used for deterministic grading.",
    )
    seed: int = Field(default=0, description="Seed used to initialize the episode.")


class StepResult(BaseModel):
    """HTTP response shape used by `/reset` and `/step`."""

    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: Optional[float] = Field(
        default=None,
        description="Reward assigned to the transition.",
    )
    done: bool = Field(default=False, description="Whether the episode is complete.")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional transition metadata.",
    )


class Task(BaseModel):
    """Task metadata returned by `/tasks`."""

    task_id: str
    description: str
    difficulty: str
    max_steps: int
    action_schema: Dict[str, Any]


class GraderResult(BaseModel):
    """Deterministic grading outcome for a completed episode."""

    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float | int | bool]
    passed: bool


class BaselineResult(BaseModel):
    """Aggregate scores reported by the built-in baseline."""

    easy: float = Field(..., ge=0.0, le=1.0)
    medium: float = Field(..., ge=0.0, le=1.0)
    hard: float = Field(..., ge=0.0, le=1.0)
    average: float = Field(..., ge=0.0, le=1.0)
