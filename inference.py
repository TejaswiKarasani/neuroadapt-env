from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI

from models import Action, Observation
from server.environment import NeuroAdaptEnv
from server.evaluation import TASK_SEEDS, heuristic_action
from server.graders import Grader

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ACTION_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

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


def _history_lines(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "None"
    lines = []
    for item in history[-4:]:
        lines.append(
            f"step={item['step']} stress={item['stress']:.3f} error_rate={item['error_rate']:.3f} "
            f"action={json.dumps(item['action'], sort_keys=True)}"
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
        temperature=0.0,
        max_tokens=220,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    response_text = completion.choices[0].message.content or ""
    return _parse_model_action(response_text)


def run_task(task_id: str, mode: str) -> float:
    env = NeuroAdaptEnv()
    grader = Grader()
    observation = env.reset(task_id=task_id, seed=TASK_SEEDS[task_id])
    history: List[Dict[str, Any]] = []
    client = None

    if mode == "llm":
        if not MODEL_NAME:
            raise RuntimeError("MODEL_NAME is required for llm mode")
        if not API_KEY:
            raise RuntimeError("HF_TOKEN or OPENAI_API_KEY is required for llm mode")
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    step_num = 0
    while not observation.done:
        try:
            if client is not None:
                action = llm_action(client, task_id, observation, history, step_num)
            else:
                action = heuristic_action(task_id, observation, step_num, [item["observation"] for item in history])
        except Exception:
            action = heuristic_action(task_id, observation, step_num, [item["observation"] for item in history])

        history.append(
            {
                "step": step_num,
                "stress": observation.stress_signal,
                "error_rate": observation.error_rate,
                "observation": observation,
                "action": action.model_dump(exclude_none=True),
            }
        )
        observation = env.step(action)
        step_num += 1

    return grader.grade(task_id, env.state).score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuroAdapt baseline inference.")
    parser.add_argument(
        "--mode",
        choices=["auto", "heuristic", "llm"],
        default="auto",
        help="Inference mode. 'auto' prefers llm when credentials are configured.",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "llm" if MODEL_NAME and API_KEY else "heuristic"

    scores: Dict[str, float] = {}
    for task_id in ("easy", "medium", "hard"):
        scores[task_id] = round(run_task(task_id, mode), 4)

    average = round(sum(scores.values()) / len(scores), 4)
    print(json.dumps({**scores, "average": average, "mode": mode}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
