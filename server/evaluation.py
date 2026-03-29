from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

try:
    from ..models import Action, BaselineResult, Observation, State
    from .environment import NeuroAdaptEnv
    from .graders import Grader
except ImportError:  # pragma: no cover - local script fallback
    from models import Action, BaselineResult, Observation, State
    from server.environment import NeuroAdaptEnv
    from server.graders import Grader

TASK_SEEDS: Dict[str, int] = {"easy": 101, "medium": 202, "hard": 303}
SUBJECT_SEQUENCE = ["math", "reading", "science", "life_skills"]
SAFE_VARIANTS = {
    "visual": "text",
    "text": "visual",
    "audio": "visual",
    "mixed": "visual",
}


def infer_preferred_modality(profile: Dict[str, float]) -> str:
    attention = profile.get("attention", 0.6)
    processing_speed = profile.get("processing_speed", 0.9)
    noise_sensitivity = profile.get("noise_sensitivity", 1.0)

    if noise_sensitivity >= 1.5:
        return "visual"
    if processing_speed <= 0.75 and noise_sensitivity <= 0.8:
        return "audio"
    if attention <= 0.45 and processing_speed >= 1.1:
        return "mixed"
    return "visual"


def parse_profile_hint(profile_hint: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for part in profile_hint.split(","):
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        try:
            values[key.strip()] = float(raw_value.strip())
        except ValueError:
            continue
    return values


def heuristic_action(
    task_id: str,
    observation: Observation,
    step_num: int,
    history: List[Observation],
) -> Action:
    profile = parse_profile_hint(observation.profile_hint)
    attention = profile.get("attention", 0.6)
    processing_speed = profile.get("processing_speed", 0.9)
    noise_sensitivity = profile.get("noise_sensitivity", 1.0)

    metadata = observation.metadata or {}
    question_modality = str(metadata.get("question_preferred_modality", "visual"))
    session_phase = str(metadata.get("session_phase", "warmup"))
    context_note = str(metadata.get("context_note", ""))
    stress = observation.stress_signal
    error_rate = observation.error_rate

    previous_stress = history[-1].stress_signal if history else stress
    stress_rising = stress - previous_stress > 0.08
    learner_modality = infer_preferred_modality(profile)

    subject = SUBJECT_SEQUENCE[step_num % len(SUBJECT_SEQUENCE)]
    if task_id == "easy":
        subject = SUBJECT_SEQUENCE[step_num % 2]

    content_type = learner_modality
    if task_id == "easy" and question_modality in SAFE_VARIANTS:
        content_type = question_modality
    elif task_id == "medium" and stress < 0.45 and question_modality in SAFE_VARIANTS:
        content_type = question_modality
    if noise_sensitivity > 1.35 and content_type in {"audio", "mixed"}:
        content_type = "visual"
    if stress < 0.38 and task_id == "easy" and step_num % 2 == 1:
        content_type = SAFE_VARIANTS.get(content_type, "visual")

    take_break = (
        stress >= 0.68
        or (task_id == "medium" and session_phase == "hallway_transition")
        or (task_id == "hard" and session_phase == "task_switch" and stress >= 0.42)
        or (task_id == "hard" and session_phase == "fatigue_peak" and stress >= 0.30)
        or (task_id == "hard" and session_phase == "closing_session" and stress >= 0.48)
        or (stress_rising and stress >= 0.55)
    )

    pace = "normal"
    if (
        stress >= 0.45
        or processing_speed < 0.85
        or session_phase in {"hallway_transition", "recovery_window", "fatigue_peak"}
    ):
        pace = "slow"

    difficulty = 4 if processing_speed >= 1.05 else 3
    if processing_speed < 0.8:
        difficulty = 2
    if error_rate > 0.60 or stress > 0.70:
        difficulty = max(1, difficulty - 1)
    if task_id == "hard" and step_num >= 6 and stress < 0.32 and error_rate < 0.20:
        difficulty = min(5, difficulty + 1)

    hint_level = 1
    if processing_speed < 0.85 or error_rate > 0.34:
        hint_level = 2
    if error_rate > 0.60 or stress > 0.65 or "disruption" in context_note:
        hint_level = 3
    if task_id == "easy" and stress < 0.30:
        hint_level = 1

    font_size = 16
    if stress > 0.45 or processing_speed < 0.8:
        font_size = 18
    if stress > 0.62:
        font_size = 20

    if noise_sensitivity > 1.55:
        contrast_mode = "high"
    elif processing_speed < 0.75:
        contrast_mode = "low"
    else:
        contrast_mode = "normal"

    animation_speed = "slow"
    if stress > 0.44 or session_phase in {"hallway_transition", "task_switch", "fatigue_peak"}:
        animation_speed = "none"
    elif attention < 0.45 and noise_sensitivity < 0.8 and task_id == "hard":
        animation_speed = "normal"

    return Action(
        content_type=content_type,
        difficulty=difficulty,
        pace=pace,
        hint_level=hint_level,
        take_break=take_break,
        font_size=font_size,
        contrast_mode=contrast_mode,
        animation_speed=animation_speed,
        subject=subject,
    )


def run_episode(
    task_id: str,
    seed: int,
    policy: Optional[Callable[[str, Observation, int, List[Observation]], Action]] = None,
) -> Tuple[State, float]:
    env = NeuroAdaptEnv()
    grader = Grader()
    observation = env.reset(task_id=task_id, seed=seed)
    history: List[Observation] = []
    step_num = 0
    decision_policy = policy or heuristic_action

    while not observation.done:
        action = decision_policy(task_id, observation, step_num, history)
        history.append(observation)
        observation = env.step(action)
        step_num += 1

    final_state = env.state
    score = grader.grade(task_id, final_state).score
    return final_state, score


def run_baseline_suite(
    policy: Optional[Callable[[str, Observation, int, List[Observation]], Action]] = None,
) -> BaselineResult:
    scores: Dict[str, float] = {}
    for task_id, seed in TASK_SEEDS.items():
        _, score = run_episode(task_id=task_id, seed=seed, policy=policy)
        scores[task_id] = round(score, 4)

    average = round(sum(scores.values()) / len(scores), 4)
    return BaselineResult(
        easy=scores["easy"],
        medium=scores["medium"],
        hard=scores["hard"],
        average=average,
    )
