from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from ..models import Action, BaselineResult, GraderResult, Observation, State, StepResult, Task
    from .environment import NeuroAdaptEnv
    from .evaluation import run_baseline_suite
    from .graders import Grader
except ImportError:  # pragma: no cover - local script fallback
    from models import Action, BaselineResult, GraderResult, Observation, State, StepResult, Task
    from server.environment import NeuroAdaptEnv
    from server.evaluation import run_baseline_suite
    from server.graders import Grader

APP_VERSION = "1.0.0"
VALID_TASK_IDS = ("easy", "medium", "hard")
README_PATH = Path(__file__).parent.parent / "README.md"
STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(
    title="NeuroAdapt OpenEnv Server",
    version=APP_VERSION,
    description=(
        "Real-world adaptive tutoring environment for sensory-aware lesson planning. "
        "The API exposes standard OpenEnv reset/step/state endpoints plus task, "
        "grader, baseline, and lightweight MCP helpers."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_sessions: Dict[str, Dict[str, Any]] = {}
grader = Grader()


def _create_env_session(session_id: str, task_id: str) -> NeuroAdaptEnv:
    env = NeuroAdaptEnv()
    _sessions[session_id] = {"env": env, "task_id": task_id}
    return env


def _get_session(session_id: str) -> Tuple[NeuroAdaptEnv, str]:
    if session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Unknown session_id. Call /reset first.")
    entry = _sessions[session_id]
    return entry["env"], entry["task_id"]


def _require_task_id(task_id: str) -> str:
    if task_id not in VALID_TASK_IDS:
        raise HTTPException(status_code=400, detail="task_id must be 'easy', 'medium', or 'hard'")
    return task_id


def _step_result(observation: Observation, info: Dict[str, Any] | None = None) -> StepResult:
    return StepResult(
        observation=observation,
        reward=observation.reward,
        done=observation.done,
        info=info or {},
    )


def _transition_info(env: NeuroAdaptEnv, task_id: str, session_id: str) -> Dict[str, Any]:
    state = env.state
    info: Dict[str, Any] = {
        "session_id": session_id,
        "episode_id": state.episode_id,
        "task_id": task_id,
        "seed": state.seed,
        "step_count": state.step_count,
        "steps_remaining": max(0, state.max_steps - state.step_count),
    }
    termination_reason = env.state.trajectory[-1]["context"] if state.trajectory else None
    if env.state.trajectory and state.step_count >= state.max_steps and env.state.hidden.true_stress < 0.95:
        info["termination_reason"] = "max_steps_reached"
    elif env.state.hidden.true_stress >= 0.95:
        info["termination_reason"] = "overwhelmed"
    elif termination_reason:
        info["latest_context"] = termination_reason
    return info


ACTION_SCHEMA = Action.model_json_schema()
TASKS = [
    Task(
        task_id="easy",
        description=(
            "Profile Matching. Keep engagement high while staying inside safe sensory "
            "bounds during a short stable tutoring warmup."
        ),
        difficulty="easy",
        max_steps=4,
        action_schema=ACTION_SCHEMA,
    ),
    Task(
        task_id="medium",
        description=(
            "Distress Response. Handle a noisy mid-session transition, reduce stress, "
            "and keep learning moving without break-spamming."
        ),
        difficulty="medium",
        max_steps=7,
        action_schema=ACTION_SCHEMA,
    ),
    Task(
        task_id="hard",
        description=(
            "Full Session Optimization. Manage fatigue across a full session, vary "
            "subjects deliberately, and sustain learning while avoiding overload."
        ),
        difficulty="hard",
        max_steps=12,
        action_schema=ACTION_SCHEMA,
    ),
]


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "neuroadapt-env",
        "version": APP_VERSION,
        "description": "Adaptive tutoring environment for sensory-aware lesson planning.",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "metadata": "GET /metadata",
            "schema": "GET /schema",
            "mcp": "POST /mcp",
            "ws": "WS /ws",
            "tasks": "GET /tasks",
            "grader": "GET /grader",
            "baseline": "GET /baseline",
            "health": "GET /health",
        },
    }


@app.get("/ui", include_in_schema=False, response_model=None)
def ui() -> FileResponse | Dict[str, str]:
    ui_file = STATIC_DIR / "index.html"
    if ui_file.exists():
        return FileResponse(str(ui_file), media_type="text/html")
    return {"error": "UI not found. Expected static/index.html."}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    readme_content = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else None
    return {
        "name": "neuroadapt-env",
        "description": (
            "Adaptive tutoring simulator where an agent tunes modality, pace, breaks, "
            "and accessibility settings for autistic learners."
        ),
        "version": APP_VERSION,
        "author": "NeuroAdapt contributors",
        "readme_content": readme_content,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    }


@app.post("/reset", response_model=StepResult)
def reset(request: Dict[str, Any] = Body(default_factory=dict)) -> StepResult:
    task_id = _require_task_id(str(request.get("task_id", "easy")))
    session_id = str(request.get("session_id", "default"))
    seed = request.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise HTTPException(status_code=400, detail="seed must be an integer")

    env = _create_env_session(session_id=session_id, task_id=task_id)
    observation = env.reset(task_id=task_id, seed=seed, episode_id=request.get("episode_id"))
    return _step_result(
        observation,
        info=_transition_info(env=env, task_id=task_id, session_id=session_id),
    )


@app.post("/step", response_model=StepResult)
def step(request: Dict[str, Any] = Body(...)) -> StepResult:
    session_id = str(request.get("session_id", "default"))
    env, task_id = _get_session(session_id)

    action_payload = request.get("action", request)
    if not isinstance(action_payload, dict):
        raise HTTPException(status_code=400, detail="Action payload must be an object")

    action_payload = {k: v for k, v in action_payload.items() if k != "session_id"}
    try:
        action = Action.model_validate(action_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    observation = env.step(action)
    return _step_result(
        observation,
        info=_transition_info(env=env, task_id=task_id, session_id=session_id),
    )


@app.get("/state")
def state(session_id: str = Query(default="default")) -> State:
    env, _task_id = _get_session(session_id)
    return env.state


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {"tasks": [task.model_dump() for task in TASKS]}


@app.get("/grader", response_model=GraderResult)
def grade(session_id: str = Query(default="default")) -> GraderResult:
    env, task_id = _get_session(session_id)
    return grader.grade(task_id, env.state)


@app.get("/baseline", response_model=BaselineResult)
def baseline() -> BaselineResult:
    return run_baseline_suite()


def _mcp_tools() -> list[Dict[str, Any]]:
    return [
        {
            "name": "reset",
            "description": "Reset an episode for one of the configured tasks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "enum": list(VALID_TASK_IDS)},
                    "session_id": {"type": "string"},
                    "seed": {"type": "integer"},
                },
            },
        },
        {
            "name": "step",
            "description": "Execute one tutoring adaptation action.",
            "inputSchema": Action.model_json_schema(),
        },
        {
            "name": "state",
            "description": "Return the full current environment state for a session.",
            "inputSchema": {"type": "object", "properties": {"session_id": {"type": "string"}}},
        },
        {
            "name": "grader",
            "description": "Grade the current episode and return a score in [0.0, 1.0].",
            "inputSchema": {"type": "object", "properties": {"session_id": {"type": "string"}}},
        },
    ]


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        )

    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params", {}) or {}

    if method == "tools/list":
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {"tools": _mcp_tools()}})

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        try:
            if tool_name == "reset":
                result = reset(arguments).model_dump()
            elif tool_name == "step":
                result = step({"action": arguments, "session_id": arguments.get("session_id", "default")}).model_dump()
            elif tool_name == "state":
                result = state(arguments.get("session_id", "default")).model_dump()
            elif tool_name == "grader":
                result = grade(arguments.get("session_id", "default")).model_dump()
            else:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
        except HTTPException as exc:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": exc.detail},
                }
            )

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
            }
        )

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "Method not found"},
        }
    )


async def _send_ws_observation(
    websocket: WebSocket,
    observation: Observation,
    env: NeuroAdaptEnv,
    task_id: str,
) -> None:
    await websocket.send_text(
        json.dumps(
            {
                "type": "observation",
                "data": _step_result(
                    observation,
                    info=_transition_info(env=env, task_id=task_id, session_id="websocket"),
                ).model_dump(),
            }
        )
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    env = NeuroAdaptEnv()
    task_id = "easy"
    initialized = False

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "data": {"message": "Invalid JSON", "code": "INVALID_JSON"},
                        }
                    )
                )
                continue

            if "type" in message:
                message_type = message.get("type")
                data = message.get("data", {}) or {}
            else:
                message_type = message.get("action")
                data = {key: value for key, value in message.items() if key != "action"}

            try:
                if message_type == "reset":
                    task_id = _require_task_id(str(data.get("task_id", "easy")))
                    observation = env.reset(
                        task_id=task_id,
                        seed=data.get("seed"),
                        episode_id=data.get("episode_id"),
                    )
                    initialized = True
                    await _send_ws_observation(websocket, observation, env, task_id)
                elif message_type == "step":
                    if not initialized:
                        raise ValueError("Call reset first")
                    action_payload = data.get("action", data)
                    action = Action.model_validate(action_payload)
                    observation = env.step(action)
                    await _send_ws_observation(websocket, observation, env, task_id)
                elif message_type == "state":
                    if not initialized:
                        raise ValueError("Call reset first")
                    await websocket.send_text(
                        json.dumps({"type": "state", "data": env.state.model_dump()})
                    )
                elif message_type == "grader":
                    if not initialized:
                        raise ValueError("Call reset first")
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "grader",
                                "data": grader.grade(task_id, env.state).model_dump(),
                            }
                        )
                    )
                elif message_type == "close":
                    break
                else:
                    raise ValueError(
                        "Unknown message type. Use reset, step, state, grader, or close."
                    )
            except Exception as exc:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "data": {"message": str(exc), "code": "EXECUTION_ERROR"},
                        }
                    )
                )
    except WebSocketDisconnect:
        pass
    finally:
        env.close()


def main(host: str | None = None, port: int | None = None) -> None:
    if host is None or port is None:
        parser = argparse.ArgumentParser(description="Run the NeuroAdapt OpenEnv server.")
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=7860)
        args = parser.parse_args()
        host = args.host
        port = args.port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
