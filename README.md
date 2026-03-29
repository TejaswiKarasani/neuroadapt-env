---
title: neuroadapt-env
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - education
  - accessibility
  - autism
  - adaptive-learning
  - reinforcement-learning
license: mit
---

# NeuroAdapt

NeuroAdapt is a real-world OpenEnv environment for adaptive tutoring. The agent acts like a classroom support system for an autistic learner: it must choose content modality, task difficulty, pace, scaffolding, accessibility settings, breaks, and subject rotation to maximize learning without triggering sensory overload.

This is not a toy game. It models a real task that teachers, therapists, and assistive systems actually face: keeping instruction productive while reacting to stress, fatigue, and sensory sensitivity in real time.

## Why this environment matters

Autistic learners often need active adjustment of sensory conditions during instruction. A lesson that is academically appropriate can still fail if the delivery is too loud, too visually busy, too fast, or too rigid. NeuroAdapt turns that challenge into a deterministic benchmark for agent training and evaluation.

The environment is designed to test whether an agent can:

- infer learner preferences from partial signals
- recover from distress during routine classroom transitions
- sustain a long session while varying subjects and managing fatigue

## Environment design

Each episode simulates a one-on-one tutoring session with a learner sampled from clinically-inspired sensory profiles. The agent never sees the full hidden profile. Instead, it observes noisy stress signals, engagement, recent accuracy, response speed, and a partial profile hint.

The environment exposes the standard OpenEnv surface:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`
- `WS /ws`

Additional benchmarking endpoints are included for hackathon submission workflows:

- `GET /tasks`
- `GET /grader`
- `GET /baseline`
- `GET /health`

## Action space

The agent chooses a typed action with these fields:

| Field | Type | Values | Meaning |
|---|---|---|---|
| `content_type` | string | `text`, `visual`, `audio`, `mixed` | Delivery modality for the next item |
| `difficulty` | int | `1`-`5` | Difficulty target for the next question |
| `pace` | string | `slow`, `normal`, `fast` | Presentation speed |
| `hint_level` | int | `0`-`3` | Scaffolding intensity |
| `take_break` | bool | `true` or `false` | Insert a calming break |
| `font_size` | int | `12`-`24` | Accessibility text size |
| `contrast_mode` | string | `normal`, `high`, `low` | Display contrast profile |
| `animation_speed` | string | `none`, `slow`, `normal`, `fast` | Motion allowed in the interface |
| `subject` | string or null | `math`, `reading`, `science`, `life_skills` | Curriculum area for the next item |

## Observation space

The environment returns a typed observation after every reset and step:

| Field | Type | Meaning |
|---|---|---|
| `stress_signal` | float `0.0-1.0` | Noisy estimate of learner stress |
| `engagement` | float `0.0-1.0` | Estimated engagement |
| `response_time_norm` | float `0.0-1.0` | Slower responses trend higher |
| `error_rate` | float `0.0-1.0` | Recent rolling error rate |
| `last_correct` | bool or null | Whether the last question was answered correctly |
| `step` | int | Current step number |
| `done` | bool | Episode termination flag |
| `reward` | float or null | Shaped reward for the last transition |
| `profile_hint` | string | Partial learner profile summary |
| `metadata` | object | Current question, preferred modality, session phase, context note, steps remaining, and termination details |

## Tasks and graders

NeuroAdapt ships with three deterministic tasks and programmatic graders that all return scores in `[0.0, 1.0]`.

### 1. Easy: Profile Matching

Goal: keep engagement high and stress low during a short stable session by adapting to the profile hint.

Why it is easy:

- the context is stable
- the main challenge is inferring safe modality and accessibility settings

Grader components:

- engagement score
- low stress score
- correctness rate
- content variety

Pass threshold: `0.50`

### 2. Medium: Distress Response

Goal: recover from a noisy mid-session transition while maintaining learning progress.

Why it is medium:

- the learner experiences an exogenous stressor
- the agent must react quickly without resorting to empty break loops

Grader components:

- prevention score
- control score
- recovery bonus
- recovery speed bonus
- learning score

Pass threshold: `0.45`

### 3. Hard: Full Session Optimization

Goal: manage a full tutoring session with subject switching, fatigue buildup, and late-session stress while sustaining learning.

Why it is hard:

- the agent must balance long-horizon reward tradeoffs
- subject variety matters
- stress must stay low across the full trajectory, not just at the end

Grader components:

- learning score
- stress score
- engagement score
- adaptation score
- consistency score
- subject variety score
- overload penalty

Pass threshold: `0.40`

## Reward shaping

Reward is dense and emitted at every step:

```text
reward = 0.38 * learning_delta
       + 0.24 * (1 - stress)
       + 0.18 * engagement
       + 0.10 * correctness_bonus
       + 0.10 * stress_regulation_bonus
       - 0.15 * (stress ^ 1.5)
       - repetition_penalty
       - break_spam_penalty
```

This gives the agent useful partial credit over the full trajectory instead of a single sparse terminal signal.

## Project structure

```text
neuroadapt-env/
|-- Dockerfile
|-- README.md
|-- baseline.py
|-- client.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- server/
|   |-- Dockerfile
|   |-- app.py
|   |-- curriculum.py
|   |-- environment.py
|   |-- evaluation.py
|   |-- graders.py
|   |-- profiles.py
|   `-- requirements.txt
`-- tests/
    `-- test_env.py
```

## Local setup

### Run the API server

```bash
python -m server.app --port 7860
```

OpenAPI docs will be available at `http://localhost:7860/docs`.

### Run tests

```bash
python -m unittest discover -s tests -v
```

### Build and run with Docker

```bash
docker build -t neuroadapt-env .
docker run -p 7860:7860 neuroadapt-env
```

### Validate with OpenEnv

From a compatible Python 3.11 environment:

```bash
openenv validate
openenv validate --url http://localhost:7860
```

## Baseline inference

The submission baseline lives at the repository root as `inference.py`.

Supported modes:

- `heuristic`: deterministic rule-based baseline with reproducible scores
- `llm`: model-driven baseline using the OpenAI client
- `auto`: uses `llm` when the required credentials are configured, otherwise falls back to `heuristic`

### Required environment variables for LLM mode

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=your-model-name
HF_TOKEN=your-hugging-face-token
```

### Run the baseline

```bash
python inference.py --mode heuristic
python inference.py --mode llm
python inference.py --mode auto
```

## Reproducible baseline scores

The deterministic heuristic baseline uses fixed seeds per task and currently produces:

| Task | Score | Passed |
|---|---:|---|
| Easy | `0.8625` | yes |
| Medium | `1.0000` | yes |
| Hard | `0.4875` | yes |
| Average | `0.7833` | |

## Example usage

### Direct Python client

```python
from client import NeuroAdaptEnv
from models import Action

client = NeuroAdaptEnv(base_url="http://localhost:7860")

try:
    result = client.reset(task_id="hard", seed=303)
    while not result.done:
        action = Action(
            content_type="audio",
            difficulty=2,
            pace="slow",
            hint_level=2,
            take_break=False,
            font_size=18,
            contrast_mode="low",
            animation_speed="none",
            subject="math",
        )
        result = client.step(action)

    final_state = client.state()
    print(final_state.step_count)
finally:
    client.close()
```

### Raw HTTP

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "content_type": "visual",
      "difficulty": 3,
      "pace": "slow",
      "hint_level": 2,
      "take_break": false,
      "font_size": 16,
      "contrast_mode": "normal",
      "animation_speed": "none",
      "subject": "reading"
    }
  }'
```

## Hugging Face Space deployment

This repository includes:

- a root `Dockerfile` for Hugging Face Spaces
- `openenv.yaml` metadata
- a root `inference.py`
- deterministic graders and a reproducible baseline

For deployment, create a Docker Space, push this repository, and tag the Space with `openenv`.

Current Space URL:

- https://huggingface.co/spaces/TejaswiKarasani/neuroadapt-env

Expected runtime base URL:

- https://tejaswikarasani-neuroadapt-env.hf.space
