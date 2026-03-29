from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Dict, Optional

import requests

from models import Action, BaselineResult, GraderResult, Observation, State

if sys.version_info < (3, 14):
    try:
        from openenv.core import EnvClient as OpenEnvClient  # type: ignore
        from openenv.core.client_types import StepResult as OpenEnvStepResult  # type: ignore
    except Exception:  # pragma: no cover - local fallback on unsupported runtimes
        OpenEnvClient = None
        OpenEnvStepResult = None
else:  # pragma: no cover - Python 3.14 currently breaks transitive openenv imports
    OpenEnvClient = None
    OpenEnvStepResult = None


if OpenEnvClient is not None and OpenEnvStepResult is not None:

    class NeuroAdaptEnv(OpenEnvClient[Action, Observation, State]):
        """OpenEnv WebSocket client for NeuroAdapt."""

        def _step_payload(self, action: Action) -> Dict[str, Any]:
            return action.model_dump(exclude_none=True)

        def _parse_result(self, payload: Dict[str, Any]) -> OpenEnvStepResult[Observation]:
            observation = Observation.model_validate(payload.get("observation", {}))
            return OpenEnvStepResult(
                observation=observation,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: Dict[str, Any]) -> State:
            return State.model_validate(payload)


else:

    @dataclass
    class StepResult:
        observation: Observation
        reward: Optional[float] = None
        done: bool = False


    class NeuroAdaptEnv:
        """HTTP fallback client for environments where openenv cannot be imported."""

        def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
            self.base_url = base_url.rstrip("/")
            self.timeout = timeout
            self.session = requests.Session()

        def __enter__(self) -> "NeuroAdaptEnv":
            return self

        def __exit__(self, *_: Any) -> None:
            self.close()

        def reset(
            self,
            task_id: str = "easy",
            seed: Optional[int] = None,
            session_id: str = "default",
        ) -> StepResult:
            payload: Dict[str, Any] = {"task_id": task_id, "session_id": session_id}
            if seed is not None:
                payload["seed"] = seed
            response = self.session.post(
                f"{self.base_url}/reset",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return StepResult(
                observation=Observation.model_validate(data["observation"]),
                reward=data.get("reward"),
                done=data.get("done", False),
            )

        def step(self, action: Action, session_id: str = "default") -> StepResult:
            response = self.session.post(
                f"{self.base_url}/step",
                json={"action": action.model_dump(exclude_none=True), "session_id": session_id},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return StepResult(
                observation=Observation.model_validate(data["observation"]),
                reward=data.get("reward"),
                done=data.get("done", False),
            )

        def state(self, session_id: str = "default") -> State:
            response = self.session.get(
                f"{self.base_url}/state",
                params={"session_id": session_id},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return State.model_validate(response.json())

        def grader(self, session_id: str = "default") -> GraderResult:
            response = self.session.get(
                f"{self.base_url}/grader",
                params={"session_id": session_id},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return GraderResult.model_validate(response.json())

        def tasks(self) -> Dict[str, Any]:
            response = self.session.get(f"{self.base_url}/tasks", timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        def baseline(self) -> BaselineResult:
            response = self.session.get(f"{self.base_url}/baseline", timeout=self.timeout)
            response.raise_for_status()
            return BaselineResult.model_validate(response.json())

        def health(self) -> Dict[str, Any]:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        def close(self) -> None:
            self.session.close()


NeuroAdaptClient = NeuroAdaptEnv
