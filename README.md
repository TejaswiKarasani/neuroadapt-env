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

**A real-world OpenEnv environment for sensory-aware adaptive tutoring.**

NeuroAdapt places an AI agent in the role of a classroom support system for an autistic learner. At every step the agent must decide: what content modality to use, how hard to make the next question, how fast to pace the session, how much scaffolding to offer, whether to call a sensory break, and which accessibility settings to apply. Getting this wrong raises the learner's stress and shuts down learning. Getting it right keeps the session productive.

This is not a toy game. Teachers, therapists, and assistive technology systems face exactly this problem every day. NeuroAdapt turns it into a deterministic, reproducible benchmark for agent training and evaluation.

---

## Why this environment matters

Autistic learners frequently require active, real-time adjustment of sensory conditions during instruction. A lesson that is academically appropriate can still fail if delivery is too loud, too visually busy, too fast-paced, or too rigid. Stress and sensory overload suppress working memory and learning almost immediately.

Current RL benchmarks do not model this. NeuroAdapt fills that gap: it provides a rigorous, clinically-grounded simulation that tests whether an agent can

- infer hidden learner preferences from partial, noisy signals
- react appropriately to exogenous stressors like hallway transitions
- sustain a productive long session while managing fatigue and subject switching

The environment is designed to be challenging for state-of-the-art LLMs. A random policy scores below 0.10 on the medium and hard tasks. A well-tuned heuristic reaches 0.86 / 1.00 / 0.49. Frontier models are expected to close the gap on the hard task through multi-step reasoning about the learner profile.

---

## Environment design

Each episode simulates a one-on-one tutoring session. The learner is drawn from one of four clinically-inspired sensory profiles at reset time. The agent never sees the full profile — it must infer preferences from noisy observations and a partial profile hint.

**Sensory profiles (hidden from agent)**

| Profile | Preference | Baseline stress | Key sensitivity |
|---|---|---|---|
| `hypersensitive_visual` | audio / text | 0.35 | visual motion, bright contrast |
| `hypersensitive_auditory` | visual | 0.30 | noise, audio content |
| `hyposensitive_seeking` | mixed | 0.18 | craves stimulation, bores easily |
| `mixed_pattern` | visual | 0.25 | moderate across modalities |

Profiles are based on DSM-5 sensory processing criteria, the Sensory Processing Measure (SPM), and peer-reviewed research on sensory modulation in ASD (Marco et al. 2011; Baranek et al. 2006; Green et al. 2016).

The environment exposes the full OpenEnv HTTP surface:

| Endpoint | Method | Purpose |
|---|---|---|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit an action and receive an observation |
| `/state` | GET | Full current episode state |
| `/metadata` | GET | Environment metadata and README |
| `/schema` | GET | JSON schemas for action / observation / state |
| `/tasks` | GET | Task listing with difficulty and max_steps |
| `/grader` | GET | Score the current episode |
| `/baseline` | GET | Run the built-in heuristic baseline |
| `/health` | GET | Liveness check |
| `/mcp` | POST | MCP JSON-RPC interface |
| `/ws` | WS | WebSocket streaming interface |

---

## Action space

The agent submits a typed JSON action at every step.

| Field | Type | Values | Meaning |
|---|---|---|---|
| `content_type` | string | `text`, `visual`, `audio`, `mixed` | Delivery modality for the next learning item |
| `difficulty` | int | `1`–`5` | Target difficulty for the next question |
| `pace` | string | `slow`, `normal`, `fast` | Presentation speed |
| `hint_level` | int | `0`–`3` | Scaffolding / hint intensity |
| `take_break` | bool | `true` / `false` | Insert a calming sensory break before continuing |
| `font_size` | int | `12`–`24` | Accessible text size |
| `contrast_mode` | string | `normal`, `high`, `low` | Display contrast profile |
| `animation_speed` | string | `none`, `slow`, `normal`, `fast` | Maximum interface motion allowed |
| `subject` | string or null | `math`, `reading`, `science`, `life_skills` | Curriculum area to emphasize |

---

## Observation space

The environment returns a typed observation after every `reset` and `step`.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `stress_signal` | float | `0.0`–`1.0` | Noisy estimate of current learner stress |
| `engagement` | float | `0.0`–`1.0` | Estimated engagement with the current configuration |
| `response_time_norm` | float | `0.0`–`1.0` | Normalized response time — higher means slower |
| `error_rate` | float | `0.0`–`1.0` | Rolling error rate over recent interactions |
| `last_correct` | bool or null | — | Whether the previous question was answered correctly |
| `step` | int | — | Current step index within the episode |
| `done` | bool | — | Whether the episode has ended |
| `reward` | float or null | `−1.0`–`1.0` | Shaped reward for the last transition |
| `profile_hint` | string | — | Partial learner profile: `attention`, `processing_speed`, `noise_sensitivity` |
| `metadata` | object | — | Current question text, preferred modality, session phase, context note, steps remaining, termination reason |

The stress signal has Gaussian noise (`σ = 0.025`). The agent must reason about trend, not just the latest reading.

---

## Tasks and graders

NeuroAdapt ships three deterministic tasks with programmatic graders. All scores are in `[0.0, 1.0]`. Graders are trajectory-only — they use no hidden state — making them auditable and reproducible.

### Task 1 — Easy: Profile Matching (4 steps)

**Objective:** keep engagement high and stress low during a short, stable session by correctly inferring the learner's sensory profile from the partial hint.

**Why it is easy:** the session context is stable; the main challenge is reading the profile hint and choosing safe modality, contrast, and animation settings.

**Grader components:**

| Component | Weight | Criterion |
|---|---|---|
| Engagement score | 35% | Steps where `engagement >= 0.55` |
| Low stress score | 35% | Steps where `stress <= 0.50` |
| Correctness rate | 20% | Steps where last answer was correct |
| Content variety | 10% | 2+ distinct `content_type` values used |

**Pass threshold:** `0.50` | **Max steps:** 4

---

### Task 2 — Medium: Distress Response (7 steps)

**Objective:** detect and recover from a noisy mid-session hallway transition while maintaining meaningful learning progress. Prevention is rewarded more than cure.

**Why it is medium:** at step 3, an exogenous stressor (`hallway_transition`) raises sensory load by +0.12. The agent must respond within 1–2 steps without spamming breaks, which suppresses learning.

**Grader components:**

| Component | Weight | Criterion |
|---|---|---|
| Prevention score | 35% | Average stress kept near 0.35 (full) vs 0.55 (zero) |
| Control score | 25% | Steps where `stress <= 0.50` |
| Recovery bonus | 20% | If spike occurred, how much stress dropped; if no spike, full credit |
| Speed bonus | 10% | Steps to recover after a spike |
| Learning score | 10% | Average learning per step (anti break-spam) |

A learning gate applies: sessions with near-zero learning (`avg < 0.04`) are penalized by 45%, regardless of stress.

**Pass threshold:** `0.45` | **Max steps:** 7

---

### Task 3 — Hard: Full Session Optimization (12 steps)

**Objective:** sustain learning across a full session while managing fatigue buildup, deliberate subject switching, late-session stress, and varying session phases.

**Why it is hard for frontier models:** the agent must track multi-step trends (is stress rising or stable?), rotate subjects deliberately (the grader scores `action.subject` choices, not question randomness), and balance short-term stress relief against long-term learning accumulation. A constant policy cannot pass.

**Grader components:**

| Component | Weight | Criterion |
|---|---|---|
| Learning score | 28% | Total learning normalized by theoretical max |
| Stress score | 22% | Strict: avg stress > 0.40 is heavily penalized |
| Engagement score | 18% | Average engagement across all steps |
| Adaptation score | 15% | Second-half stress must improve on first-half |
| Consistency score | 10% | Low step-to-step stress volatility |
| Subject variety | 7% | Agent must deliberately choose 3+ distinct subjects |
| Overload penalty | −30% max | Deducted for stress spikes > 0.08 per step |

**Pass threshold:** `0.40` | **Max steps:** 12

---

## Reward function

Reward is dense — emitted at every step, never just at episode end.

```
reward = 0.38 * learning_delta
       + 0.24 * (1 − stress)
       + 0.18 * engagement
       + 0.10 * correctness_bonus
       + 0.10 * stress_regulation_bonus
       − 0.15 * (stress ^ 1.5)
       − repetition_penalty          (−0.06 if same content_type + difficulty for 4+ steps)
       − break_spam_penalty          (−0.08 per step after 2 consecutive breaks)
```

`stress_regulation_bonus = max(−0.08, min(0.10, previous_stress − current_stress))` — the agent is rewarded for actively reducing stress, not just for low stress.

The `stress ^ 1.5` term creates nonlinear pressure: mild stress is tolerable but high stress is catastrophically penalized. This incentivizes prevention over cure.

---

## Baseline scores

Scores are fully deterministic. Fixed seeds are used per task (`easy=101`, `medium=202`, `hard=303`).

| Task | Random policy | Heuristic baseline | Gap (LLM opportunity) |
|---|---:|---:|---|
| Easy | 0.626 | 0.863 | small |
| Medium | 0.094 | 1.000 | large |
| Hard | 0.059 | 0.488 | large |
| **Average** | **0.260** | **0.784** | |

The heuristic baseline uses rule-based logic derived from the profile hint. A good LLM agent is expected to match or exceed the heuristic on easy/medium and close the gap significantly on hard by reasoning about multi-step stress trends and subject rotation.

---

## Project structure

```
neuroadapt-env/
├── Dockerfile              # Root Dockerfile for HF Space deployment
├── README.md
├── inference.py            # Submission baseline script
├── baseline.py             # Thin wrapper around inference.py
├── client.py               # HTTP client for the running server
├── models.py               # Pydantic models: Action, Observation, State
├── openenv.yaml            # OpenEnv spec metadata
├── pyproject.toml
├── server/
│   ├── app.py              # FastAPI application
│   ├── curriculum.py       # Question bank
│   ├── environment.py      # Core NeuroAdaptEnv simulation
│   ├── evaluation.py       # Heuristic policy and baseline runner
│   ├── graders.py          # Deterministic trajectory graders
│   ├── profiles.py         # Clinically-inspired sensory profiles
│   ├── requirements.txt
│   └── Dockerfile
└── tests/
    └── test_env.py         # 11 smoke and contract tests
```

---

## Setup and usage

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)

### Option 1 — Run the server directly

```bash
# Install dependencies
pip install fastapi uvicorn pydantic requests openai websockets

# Start the API server
python -m server.app --port 7860
```

The OpenAPI docs will be available at `http://localhost:7860/docs`.

### Option 2 — Run with uv

```bash
uv sync
uv run server --port 7860
```

### Option 3 — Docker

```bash
docker build -t neuroadapt-env .
docker run -p 7860:7860 neuroadapt-env
```

### Run tests

```bash
python -m unittest discover -s tests -v
```

### Validate with OpenEnv CLI

```bash
openenv validate                           # local structure check
openenv validate --url http://localhost:7860  # runtime API compliance
```

---

## Baseline inference script

`inference.py` is the submission baseline. It connects to the running environment server via HTTP and emits structured stdout logs in the required OpenEnv format.

### Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `PING_URL` / `ENV_URL` | for validator | `http://localhost:7860` | URL of the running environment server |
| `API_BASE_URL` | for LLM mode | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | for LLM mode | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` / `OPENAI_API_KEY` | for LLM mode | — | API key |

### Run modes

```bash
# Deterministic heuristic (no API key needed)
python inference.py --mode heuristic --task easy
python inference.py --mode heuristic --task medium
python inference.py --mode heuristic --task hard

# LLM-driven (requires API credentials)
export HF_TOKEN=your-token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py --mode llm --task hard

# Auto: uses LLM when credentials are set, heuristic otherwise
python inference.py --mode auto --task easy
```

### Stdout format

```
[START] task=easy env=neuroadapt-env model=heuristic
[STEP] step=1 action={...} reward=0.36 done=false error=null
[STEP] step=2 action={...} reward=0.36 done=false error=null
[STEP] step=3 action={...} reward=0.29 done=false error=null
[STEP] step=4 action={...} reward=0.29 done=true error=null
[END] success=true steps=4 score=0.86 rewards=0.36,0.36,0.29,0.29
```

---

## HTTP usage examples

### Python client

```python
from client import NeuroAdaptEnv
from models import Action

client = NeuroAdaptEnv(base_url="http://localhost:7860")

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

score = client.grader()
print(score)
client.close()
```

### Raw curl

```bash
# Start a new episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "medium", "seed": 202}'

# Submit an action
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

# Grade the completed episode
curl http://localhost:7860/grader
```

---

## Deployment

This repository deploys as a Docker-based Hugging Face Space.

Live Space: https://huggingface.co/spaces/TejaswiKarasani/neuroadapt-env

Runtime base URL: `https://tejaswikarasani-neuroadapt-env.hf.space`

To deploy your own copy:

```bash
# Push to a Docker Space tagged with `openenv`
git clone https://huggingface.co/spaces/<your-username>/neuroadapt-env hf-space
cp -r . hf-space/
cd hf-space
git add .
git commit -m "Deploy NeuroAdapt"
git push
```

---

## References

- Marco, E.J. et al. (2011). Sensory processing in autism: A review of neurophysiologic findings. *Pediatric Research*, 69, 48–54.
- Baranek, G.T. et al. (2006). Sensory processing subtypes in autism. *Journal of Autism and Developmental Disorders*, 36, 591–605.
- Green, S.A. et al. (2016). Sensory over-responsivity in ASD. *Journal of Autism and Developmental Disorders*, 46, 3232–3241.
