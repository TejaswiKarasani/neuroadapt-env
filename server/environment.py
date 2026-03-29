from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from ..models import Action, HiddenState, Observation, State
    from .curriculum import CURRICULUM, Question
    from .profiles import AutismSensoryProfile, PROFILES
except ImportError:  # pragma: no cover - local script fallback
    from models import Action, HiddenState, Observation, State
    from server.curriculum import CURRICULUM, Question
    from server.profiles import AutismSensoryProfile, PROFILES

ANIMATION_ORDER = {"none": 0, "slow": 1, "normal": 2, "fast": 3}
PACE_ORDER = {"slow": 0, "normal": 1, "fast": 2}
VALID_SUBJECTS = {"math", "reading", "science", "life_skills"}
TASK_MAX_STEPS = {"easy": 4, "medium": 7, "hard": 12}


@dataclass(frozen=True)
class SessionContext:
    label: str
    note: str
    stress_pressure: float
    fatigue_pressure: float


class NeuroAdaptEnv:
    """Session-isolated tutoring environment with deterministic seeded behavior."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state: Optional[State] = None
        self.profile: Optional[AutismSensoryProfile] = None
        self.current_question: Optional[Question] = None
        self.episode_id: str = ""
        self._rng = random.Random()

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **_: Any,
    ) -> Observation:
        """Reset the environment and return the initial observation."""
        if task_id not in TASK_MAX_STEPS:
            raise ValueError("task_id must be 'easy', 'medium', or 'hard'")

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        self._rng = random.Random(seed)
        self.profile = self._rng.choice(PROFILES)
        self.episode_id = episode_id or str(uuid.UUID(int=self._rng.getrandbits(128)))[:8]
        self.current_question = self._get_question(
            difficulty=self.profile.optimal_difficulty,
            subject=self._recommended_subject(task_id, step_count=0),
        )

        self._state = State(
            episode_id=self.episode_id,
            step_count=0,
            task_id=task_id,
            max_steps=TASK_MAX_STEPS[task_id],
            hidden=HiddenState(
                true_stress=self.profile.baseline_stress,
                fatigue=0.15,
                preference=self.profile.preference,
                sensory_overload_count=0,
                current_context="warmup",
            ),
            total_learning=0.0,
            total_reward=0.0,
            history=[],
            trajectory=[],
            seed=seed,
        )
        return self._make_observation(correct=None, reward=0.0, done=False)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> Observation:
        """Advance the environment by one step and return the resulting observation."""
        del timeout_s  # The environment is deterministic and does not use timeouts.

        if self._state is None or self.profile is None:
            raise ValueError("Call reset() before step()")

        state = self._state
        hidden = state.hidden
        profile = self.profile
        previous_stress = hidden.true_stress
        context = self._session_context(state.task_id, state.step_count)
        hidden.current_context = context.label

        stress_delta = context.stress_pressure
        fatigue_delta = context.fatigue_pressure
        engagement_adjustment = 0.0
        learning_multiplier = 1.0

        if action.content_type != profile.preference:
            stress_delta += 0.06 * profile.sensory_sensitivity
        if action.content_type in ("audio", "mixed"):
            stress_delta += 0.04 * profile.noise_sensitivity
        if action.content_type in ("visual", "mixed"):
            anim_value = ANIMATION_ORDER[action.animation_speed]
            max_anim = ANIMATION_ORDER[profile.max_animation_speed]
            if anim_value > max_anim:
                stress_delta += 0.07 * profile.visual_sensitivity
        if PACE_ORDER[action.pace] > PACE_ORDER["normal"]:
            stress_delta += 0.05 * profile.sensory_sensitivity
            fatigue_delta += 0.02
        if action.font_size < profile.preferred_font_size - 2:
            stress_delta += 0.03
        if action.contrast_mode != profile.preferred_contrast:
            stress_delta += 0.025 * profile.visual_sensitivity
        if action.take_break:
            stress_delta -= 0.16
            fatigue_delta -= 0.10
            engagement_adjustment += 0.04

        if context.label in {"hallway_transition", "task_switch"}:
            if action.pace == "slow":
                stress_delta -= 0.03
            if action.animation_speed == "none":
                stress_delta -= 0.02
            if action.take_break:
                stress_delta -= 0.03
        if context.label == "fatigue_peak" and action.hint_level >= 2:
            fatigue_delta -= 0.02
            learning_multiplier *= 1.05

        consecutive_breaks = 0
        for past_action in reversed(state.history[-3:]):
            if past_action.take_break:
                consecutive_breaks += 1
            else:
                break
        if consecutive_breaks >= 2:
            learning_multiplier *= max(0.10, 1.0 - consecutive_breaks * 0.35)
            engagement_adjustment -= 0.05

        chosen_subject = (
            action.subject if action.subject is not None and action.subject in VALID_SUBJECTS else None
        )
        self.current_question = self._get_question(
            difficulty=action.difficulty,
            subject=chosen_subject or self._recommended_subject(state.task_id, state.step_count),
        )
        question = self.current_question

        subject_bonus = self._subject_adaptation_bonus(action.subject, state)
        engagement_adjustment += subject_bonus["engagement"]
        stress_delta += subject_bonus["stress"]
        learning_multiplier *= subject_bonus["learning"]

        fatigue_delta += 0.04 * (1.0 - profile.attention_span)
        hidden.fatigue = self._clamp(hidden.fatigue + fatigue_delta)

        if stress_delta > 0.08:
            hidden.sensory_overload_count += 1
        hidden.true_stress = self._clamp(hidden.true_stress + stress_delta)

        difficulty_gap = abs(action.difficulty - profile.optimal_difficulty)
        if difficulty_gap == 0:
            base_learning = 0.12
            correct = True
        elif difficulty_gap == 1:
            base_learning = 0.07
            correct = True
        elif difficulty_gap == 2:
            base_learning = 0.03
            correct = False
        else:
            base_learning = 0.005
            correct = False

        if action.content_type == question.preferred_modality:
            base_learning *= 1.25
        if not correct and action.hint_level >= 2:
            base_learning *= 1.40
            correct = True
        if action.take_break:
            base_learning *= 0.25
            correct = None

        if hidden.true_stress > 0.75:
            learning_multiplier *= 0.15
        elif hidden.true_stress > 0.55:
            learning_multiplier *= 0.50
        elif hidden.true_stress > 0.40:
            learning_multiplier *= 0.75

        learning_multiplier *= max(0.20, 1.0 - hidden.fatigue * 0.5)
        learning = base_learning * learning_multiplier * profile.learning_rate
        engagement = self._clamp(
            1.0
            - hidden.true_stress * 0.72
            - hidden.fatigue * 0.35
            + engagement_adjustment
        )

        regulation_bonus = max(-0.08, min(0.10, previous_stress - hidden.true_stress))
        correct_bonus = 1.0 if correct is True else (0.5 if correct is None else 0.0)
        reward = (
            0.38 * learning
            + 0.24 * (1.0 - hidden.true_stress)
            + 0.18 * engagement
            + 0.10 * correct_bonus
            + 0.10 * regulation_bonus
            - 0.15 * (hidden.true_stress ** 1.5)
        )

        if len(state.history) >= 3:
            last_three = state.history[-3:]
            if all(
                prior.content_type == action.content_type and prior.difficulty == action.difficulty
                for prior in last_three
            ):
                reward -= 0.06

        if consecutive_breaks >= 2:
            reward -= 0.08 * (consecutive_breaks - 1)

        reward = round(max(-1.0, min(1.0, reward)), 4)

        state.total_learning += learning
        state.total_reward += reward
        state.history.append(action)
        state.step_count += 1

        done = state.step_count >= state.max_steps or hidden.true_stress >= 0.95
        state.trajectory.append(
            {
                "step": state.step_count,
                "action": action.model_dump(),
                "reward": reward,
                "stress": round(hidden.true_stress, 4),
                "engagement": round(engagement, 4),
                "learning": round(learning, 4),
                "correct": correct,
                "fatigue": round(hidden.fatigue, 4),
                "question_id": question.id,
                "subject": question.subject,
                "context": context.label,
            }
        )

        return self._make_observation(correct=correct, reward=reward, done=done)

    @property
    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Call reset() before requesting state")
        return self._state

    def close(self) -> None:
        """Compatibility no-op for container and client helpers."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "neuroadapt-env",
            "description": (
                "Adaptive tutoring environment that simulates sensory-aware lesson"
                " adjustments for autistic learners."
            ),
            "version": "1.0.0",
            "author": "NeuroAdapt contributors",
        }

    def _get_question(self, difficulty: int, subject: Optional[str] = None) -> Question:
        pool = [question for question in CURRICULUM if question.difficulty == difficulty]
        if subject:
            matching = [question for question in pool if question.subject == subject]
            if matching:
                pool = matching
        if not pool:
            pool = sorted(CURRICULUM, key=lambda item: abs(item.difficulty - difficulty))[:3]
        return self._rng.choice(pool)

    def _make_observation(
        self,
        correct: Optional[bool],
        reward: float,
        done: bool,
    ) -> Observation:
        if self._state is None or self.profile is None:
            raise RuntimeError("Call reset() before requesting observations")

        state = self._state
        hidden = state.hidden
        profile = self.profile
        question = self.current_question
        noise = self._rng.gauss(0, 0.025)
        observed_stress = round(self._clamp(hidden.true_stress + noise), 3)
        response_time = round(
            min(1.0, 0.25 + hidden.true_stress * 0.45 + hidden.fatigue * 0.25),
            3,
        )
        recent = state.trajectory[-3:] if state.trajectory else []
        error_rate = round(
            sum(1 for transition in recent if transition.get("correct") is False)
            / max(1, len(recent)),
            3,
        )

        profile_hint = (
            f"attention:{profile.attention_span:.1f},"
            f"processing_speed:{profile.processing_speed:.1f},"
            f"noise_sensitivity:{profile.noise_sensitivity:.1f}"
        )
        context = self._session_context(state.task_id, state.step_count)
        termination_reason = None
        if done:
            termination_reason = (
                "overwhelmed" if hidden.true_stress >= 0.95 else "max_steps_reached"
            )

        metadata: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "task_id": state.task_id,
            "seed": state.seed,
            "current_subject": question.subject if question else "unknown",
            "current_question": question.text if question else "",
            "question_preferred_modality": question.preferred_modality if question else "visual",
            "question_hint": question.hint if question else "",
            "available_subjects": sorted(VALID_SUBJECTS),
            "max_steps": state.max_steps,
            "steps_remaining": max(0, state.max_steps - state.step_count),
            "termination_threshold": 0.95,
            "session_phase": context.label,
            "context_note": context.note,
        }
        if termination_reason is not None:
            metadata["termination_reason"] = termination_reason

        return Observation(
            stress_signal=observed_stress,
            engagement=round(
                self._clamp(1.0 - hidden.true_stress * 0.78 - hidden.fatigue * 0.10),
                3,
            ),
            response_time_norm=response_time,
            error_rate=error_rate,
            last_correct=correct,
            step=state.step_count,
            done=done,
            reward=reward,
            profile_hint=profile_hint,
            metadata=metadata,
        )

    def _recommended_subject(self, task_id: str, step_count: int) -> str:
        sequence = ["math", "reading", "science", "life_skills"]
        if task_id == "easy":
            return sequence[step_count % 2]
        return sequence[step_count % len(sequence)]

    def _session_context(self, task_id: str, step_count: int) -> SessionContext:
        if task_id == "easy":
            if step_count == 0:
                return SessionContext(
                    label="warmup",
                    note="The learner is settling into the session.",
                    stress_pressure=0.0,
                    fatigue_pressure=0.0,
                )
            return SessionContext(
                label="guided_practice",
                note="Routine practice with stable classroom conditions.",
                stress_pressure=0.01,
                fatigue_pressure=0.0,
            )

        if task_id == "medium":
            if step_count < 2:
                return SessionContext(
                    label="warmup",
                    note="The session starts calmly before a routine transition.",
                    stress_pressure=0.0,
                    fatigue_pressure=0.0,
                )
            if step_count == 2:
                return SessionContext(
                    label="hallway_transition",
                    note="A noisy hallway transition briefly raises sensory load.",
                    stress_pressure=0.12,
                    fatigue_pressure=0.02,
                )
            return SessionContext(
                label="recovery_window",
                note="The learner is recovering from a disruption while trying to stay on task.",
                stress_pressure=0.03,
                fatigue_pressure=0.01,
            )

        if step_count < 4:
            return SessionContext(
                label="warmup",
                note="The learner begins with moderate focus.",
                stress_pressure=0.0,
                fatigue_pressure=0.0,
            )
        if step_count < 8:
            return SessionContext(
                label="task_switch",
                note="The therapist introduces new topics to avoid fixation and boredom.",
                stress_pressure=0.05,
                fatigue_pressure=0.01,
            )
        if step_count < 10:
            return SessionContext(
                label="fatigue_peak",
                note="Late-session fatigue makes pacing and breaks especially important.",
                stress_pressure=0.07,
                fatigue_pressure=0.03,
            )
        return SessionContext(
            label="closing_session",
            note="The session is winding down and the learner benefits from gentle closure.",
            stress_pressure=0.03,
            fatigue_pressure=0.02,
        )

    def _subject_adaptation_bonus(self, subject: Optional[str], state: State) -> Dict[str, float]:
        if subject is None:
            return {"stress": 0.02 if state.task_id == "hard" else 0.0, "engagement": 0.0, "learning": 0.96}

        recent_subjects = [
            transition["action"].get("subject")
            for transition in state.trajectory[-3:]
            if transition["action"].get("subject") is not None
        ]
        if state.task_id == "hard":
            if subject not in recent_subjects:
                return {"stress": -0.01, "engagement": 0.05, "learning": 1.08}
            if recent_subjects.count(subject) >= 2:
                return {"stress": 0.04, "engagement": -0.06, "learning": 0.92}

        if recent_subjects and recent_subjects[-1] == subject and state.task_id != "easy":
            return {"stress": 0.01, "engagement": -0.02, "learning": 0.98}

        return {"stress": 0.0, "engagement": 0.02, "learning": 1.02}

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
