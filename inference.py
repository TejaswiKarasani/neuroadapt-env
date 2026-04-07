from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from models import Action, Observation

# Required env configuration for submission validators.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")  # set your active endpoint
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")             # set your active model
HF_TOKEN = os.getenv("HF_TOKEN")                                               # optional if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")                               # optional: local Docker image name
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY")

# URL of the running NeuroAdapt environment server (HF Space or local Docker).
# PING_URL is injected by the OpenEnv submission validator; ENV_URL for manual runs.
ENV_URL = os.getenv("PING_URL") or os.getenv("ENV_URL") or "http://localhost:7860"

TASK_NAME = os.getenv("MY_ENV_V4_TASK") or os.getenv("TASK_NAME") or ""
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "neuroadapt-env")
TEMPERATURE = 0.0
MAX_TOKENS = 220

ACTION_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
VALID_TASKS = ("easy", "medium", "hard")

TASK_INSTRUCTIONS = {
    "easy": "Keep engagement high and stress low during a short stable session.",
    "medium": "Handle a noisy transition, reduce stress quickly, and keep learning going.",
    "hard": "Sustain learning across a long session while varying subjects and preventing overload.",
}

SYSTEM_PROMPT = (
    "You are controlling a sensory-aware tutoring system. Return exactly one JSON object "
    "matching this schema: "
    "{content_type, difficulty, pace, hint_level, take_break, font_size, contrast_mode, "
    "animation_speed, subject}. Use only allowed enum values. No markdown or explanation."
)

TASK_SEEDS: Dict[str, int] = {"easy": 101, "medium": 202, "hard": 303}


# ---------------------------------------------------------------------------
# Lightweight HTTP client (no openenv-core dependency required)
# ---------------------------------------------------------------------------

class _StepResult:
    def __init__(self, observation: Observation, reward: Optional[float], done: bool) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done


class EnvClient:
    """Thin HTTP client that talks to the running NeuroAdapt server."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> _StepResult:
        payload: Dict[str, Any] = {"task_id": task_id, "session_id": "default"}
        if seed is not None:
            payload["seed"] = seed
        resp = self._session.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return _StepResult(
            observation=Observation.model_validate(data["observation"]),
            reward=data.get("reward"),
            done=bool(data.get("done", False)),
        )

    def step(self, action: Action) -> _StepResult:
        resp = self._session.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump(exclude_none=True, exclude={"metadata"}), "session_id": "default"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return _StepResult(
            observation=Observation.model_validate(data["observation"]),
            reward=data.get("reward"),
            done=bool(data.get("done", False)),
        )

    def grader(self) -> Dict[str, Any]:
        resp = self._session.get(
            f"{self.base_url}/grader", params={"session_id": "default"}, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._session.close()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _inline(value: Optional[str]) -> str:
    if not value:
        return "null"
    return value.replace("\r", " ").replace("\n", " ").strip() or "null"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={_inline(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Heuristic fallback (mirrors server/evaluation.py logic without import)
# ---------------------------------------------------------------------------

_SUBJECT_SEQUENCE = ["math", "reading", "science", "life_skills"]
_SAFE_VARIANTS = {"visual": "text", "text": "visual", "audio": "visual", "mixed": "visual"}
_PACE_ORDER = {"slow": 0, "normal": 1, "fast": 2}


def _parse_profile(hint: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for part in hint.split(","):
        if ":" not in part:
            continue
        key, raw = part.split(":", 1)
        try:
            values[key.strip()] = float(raw.strip())
        except ValueError:
            continue
    return values


def heuristic_action(
    task_id: str,
    observation: Observation,
    step_num: int,
    history: List[Observation],
) -> Action:
    profile = _parse_profile(observation.profile_hint)
    attention = profile.get("attention", 0.6)
    processing_speed = profile.get("processing_speed", 0.9)
    noise_sensitivity = profile.get("noise_sensitivity", 1.0)

    metadata = observation.metadata or {}
    question_modality = str(metadata.get("question_preferred_modality", "visual"))
    session_phase = str(metadata.get("session_phase", "warmup"))
    stress = observation.stress_signal
    error_rate = observation.error_rate
    previous_stress = history[-1].stress_signal if history else stress
    stress_rising = stress - previous_stress > 0.08

    if noise_sensitivity >= 1.5:
        learner_modality = "visual"
    elif processing_speed <= 0.75 and noise_sensitivity <= 0.8:
        learner_modality = "audio"
    elif attention <= 0.45 and processing_speed >= 1.1:
        learner_modality = "mixed"
    else:
        learner_modality = "visual"

    subject = _SUBJECT_SEQUENCE[step_num % len(_SUBJECT_SEQUENCE)]
    if task_id == "easy":
        subject = _SUBJECT_SEQUENCE[step_num % 2]

    content_type = learner_modality
    if task_id == "easy" and question_modality in _SAFE_VARIANTS:
        content_type = question_modality
    elif task_id == "medium" and stress < 0.45 and question_modality in _SAFE_VARIANTS:
        content_type = question_modality
    if noise_sensitivity > 1.35 and content_type in {"audio", "mixed"}:
        content_type = "visual"
    if stress < 0.38 and task_id == "easy" and step_num % 2 == 1:
        content_type = _SAFE_VARIANTS.get(content_type, "visual")

    take_break = (
        stress >= 0.68
        or (task_id == "medium" and session_phase == "hallway_transition")
        or (task_id == "hard" and session_phase == "task_switch" and stress >= 0.42)
        or (task_id == "hard" and session_phase == "fatigue_peak" and stress >= 0.30)
        or (task_id == "hard" and session_phase == "closing_session" and stress >= 0.48)
        or (stress_rising and stress >= 0.55)
    )

    pace = "normal"
    if stress >= 0.45 or processing_speed < 0.85 or session_phase in {
        "hallway_transition", "recovery_window", "fatigue_peak"
    }:
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
    if error_rate > 0.60 or stress > 0.65:
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


# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------

def _history_lines(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "None"
    lines: List[str] = []
    for item in history[-4:]:
        lines.append(
            f"step={item['step']} stress={item['stress']:.3f} "
            f"error_rate={item['error_rate']:.3f} "
            f"action={json.dumps(item['action'], sort_keys=True, separators=(',', ':'))}"
        )
    return "\n".join(lines)


def _parse_model_action(response_text: str) -> Action:
    match = ACTION_JSON_RE.search(response_text.strip())
    if not match:
        raise ValueError("Model did not return a JSON object")
    payload = json.loads(match.group(0))
    return Action.model_validate(payload)


def llm_action(
    client: OpenAI,
    task_id: str,
    observation: Observation,
    history: List[Dict[str, Any]],
    step_num: int,
) -> Action:
    user_prompt = (
        f"Task: {task_id}\n"
        f"Objective: {TASK_INSTRUCTIONS[task_id]}\n"
        f"Step: {step_num}\n"
        f"Observation: {json.dumps(observation.model_dump(mode='json'), sort_keys=True)}\n"
        f"Recent history:\n{_history_lines(history)}\n"
        "Choose the next tutoring adaptation."
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    response_text = completion.choices[0].message.content or ""
    return _parse_model_action(response_text)


def _choose_action(
    mode: str,
    client: Optional[OpenAI],
    task_id: str,
    observation: Observation,
    llm_history: List[Dict[str, Any]],
    observation_history: List[Observation],
    step_num: int,
) -> Action:
    if mode == "llm" and client is not None:
        try:
            return llm_action(client, task_id, observation, llm_history, step_num)
        except Exception:
            pass
    return heuristic_action(task_id, observation, step_num, observation_history)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, mode: str) -> None:
    model_label = MODEL_NAME if mode == "llm" else "heuristic"
    log_start(task=task_id, env=BENCHMARK, model=model_label)

    env = EnvClient(base_url=ENV_URL)
    client: Optional[OpenAI] = None

    llm_history: List[Dict[str, Any]] = []
    observation_history: List[Observation] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        if mode == "llm":
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        result = env.reset(task_id=task_id, seed=TASK_SEEDS[task_id])
        observation = result.observation

        while not observation.done:
            action = _choose_action(
                mode=mode,
                client=client,
                task_id=task_id,
                observation=observation,
                llm_history=llm_history,
                observation_history=observation_history,
                step_num=steps_taken,
            )
            action_payload = action.model_dump(exclude_none=True, exclude={"metadata"})
            action_str = json.dumps(action_payload, sort_keys=True, separators=(",", ":"))

            result = env.step(action)
            next_observation = result.observation
            reward = float(next_observation.reward or 0.0)
            done = bool(next_observation.done)
            error = next_observation.metadata.get("last_action_error")

            steps_taken += 1
            rewards.append(reward)
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=error if isinstance(error, str) else None,
            )

            llm_history.append({
                "step": steps_taken,
                "stress": observation.stress_signal,
                "error_rate": observation.error_rate,
                "action": action_payload,
            })
            observation_history.append(observation)
            observation = next_observation

        grader_result = env.grader()
        score = max(0.0, min(1.0, float(grader_result.get("score", 0.0))))
        success = bool(grader_result.get("passed", False))

    except Exception:
        score = 0.0
        success = False
    finally:
        try:
            env.close()
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NeuroAdapt inference with structured stdout logs."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "heuristic", "llm"],
        default="auto",
        help="Inference mode. 'auto' prefers llm when credentials are configured.",
    )
    parser.add_argument(
        "--task",
        choices=list(VALID_TASKS) + ["all"],
        default=TASK_NAME if TASK_NAME in VALID_TASKS else "all",
        help="Task to run, or 'all' to run all three tasks (default when no TASK_NAME is set).",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "llm" if API_KEY and MODEL_NAME else "heuristic"
    if mode == "llm" and (not API_KEY or not MODEL_NAME):
        mode = "heuristic"

    if args.task == "all":
        for task_id in VALID_TASKS:
            run_task(task_id=task_id, mode=mode)
    else:
        run_task(task_id=args.task, mode=mode)


if __name__ == "__main__":
    main()
