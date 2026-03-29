"""Smoke and contract tests for NeuroAdapt."""

from __future__ import annotations

import os
import sys
import unittest

from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import server.app as app_module

from models import Action, GraderResult, Observation
from server.environment import NeuroAdaptEnv
from server.evaluation import run_baseline_suite
from server.graders import Grader


def make_action(**overrides: object) -> Action:
    payload = {
        "content_type": "visual",
        "difficulty": 3,
        "pace": "normal",
        "hint_level": 1,
        "take_break": False,
        "font_size": 16,
        "contrast_mode": "normal",
        "animation_speed": "slow",
        "subject": "math",
    }
    payload.update(overrides)
    return Action(**payload)


class EnvironmentTests(unittest.TestCase):
    def test_reset_returns_initial_observation(self) -> None:
        env = NeuroAdaptEnv()
        observation = env.reset(task_id="easy", seed=11)

        self.assertIsInstance(observation, Observation)
        self.assertEqual(observation.step, 0)
        self.assertFalse(observation.done)
        self.assertGreaterEqual(observation.stress_signal, 0.0)
        self.assertLessEqual(observation.stress_signal, 1.0)

    def test_seed_reproducibility(self) -> None:
        env_a = NeuroAdaptEnv()
        env_b = NeuroAdaptEnv()
        first_obs_a = env_a.reset(task_id="medium", seed=77)
        first_obs_b = env_b.reset(task_id="medium", seed=77)

        self.assertEqual(first_obs_a.model_dump(), first_obs_b.model_dump())

        action = make_action(subject="science")
        second_obs_a = env_a.step(action)
        second_obs_b = env_b.step(action)
        self.assertEqual(second_obs_a.model_dump(), second_obs_b.model_dump())

    def test_medium_task_has_transition_context(self) -> None:
        env = NeuroAdaptEnv()
        observation = env.reset(task_id="medium", seed=101)
        while not observation.done and env.state.step_count < 3:
            observation = env.step(make_action())

        contexts = [transition["context"] for transition in env.state.trajectory]
        self.assertIn("hallway_transition", contexts)

    def test_episode_terminates_within_max_steps(self) -> None:
        env = NeuroAdaptEnv()
        observation = env.reset(task_id="hard", seed=5)
        steps = 0
        while not observation.done:
            observation = env.step(make_action())
            steps += 1
            self.assertLessEqual(steps, env.state.max_steps)

        self.assertTrue(observation.done)

    def test_graders_return_bounded_scores(self) -> None:
        grader = Grader()
        for task_id in ("easy", "medium", "hard"):
            env = NeuroAdaptEnv()
            observation = env.reset(task_id=task_id, seed=13)
            while not observation.done:
                observation = env.step(make_action(subject="reading"))
            result = grader.grade(task_id, env.state)
            self.assertIsInstance(result, GraderResult)
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)

    def test_constant_policy_stays_limited_on_hard(self) -> None:
        grader = Grader()
        scores = []
        for seed in (1, 2, 3):
            env = NeuroAdaptEnv()
            observation = env.reset(task_id="hard", seed=seed)
            while not observation.done:
                observation = env.step(make_action(subject="math"))
            scores.append(grader.grade("hard", env.state).score)

        self.assertLess(sum(scores) / len(scores), 0.75)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app_module._sessions.clear()
        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        self.client.close()

    def test_health_metadata_and_schema_endpoints(self) -> None:
        health = self.client.get("/health")
        metadata = self.client.get("/metadata")
        schema = self.client.get("/schema")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")
        self.assertEqual(metadata.status_code, 200)
        self.assertIn("name", metadata.json())
        self.assertIn("description", metadata.json())
        self.assertEqual(schema.status_code, 200)
        self.assertIn("action", schema.json())
        self.assertIn("observation", schema.json())
        self.assertIn("state", schema.json())

    def test_http_reset_step_and_state(self) -> None:
        reset = self.client.post("/reset", json={"task_id": "easy", "seed": 123})
        self.assertEqual(reset.status_code, 200)
        reset_payload = reset.json()
        self.assertIn("observation", reset_payload)
        self.assertFalse(reset_payload["done"])

        step = self.client.post("/step", json={"action": make_action().model_dump()})
        self.assertEqual(step.status_code, 200)
        step_payload = step.json()
        self.assertIn("reward", step_payload)
        self.assertIn("done", step_payload)

        state = self.client.get("/state")
        self.assertEqual(state.status_code, 200)
        self.assertEqual(state.json()["step_count"], 1)

    def test_mcp_endpoint_returns_jsonrpc_payload(self) -> None:
        response = self.client.post("/mcp", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["jsonrpc"], "2.0")

    def test_websocket_protocol(self) -> None:
        with self.client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "reset", "data": {"task_id": "easy", "seed": 9}})
            first = websocket.receive_json()
            self.assertEqual(first["type"], "observation")

            websocket.send_json({"type": "step", "data": make_action().model_dump()})
            second = websocket.receive_json()
            self.assertEqual(second["type"], "observation")

            websocket.send_json({"type": "state"})
            state_response = websocket.receive_json()
            self.assertEqual(state_response["type"], "state")
            self.assertEqual(state_response["data"]["step_count"], 1)

    def test_baseline_endpoint_and_runner(self) -> None:
        response = self.client.get("/baseline")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        for key in ("easy", "medium", "hard", "average"):
            self.assertGreaterEqual(payload[key], 0.0)
            self.assertLessEqual(payload[key], 1.0)

        direct = run_baseline_suite()
        self.assertEqual(payload, direct.model_dump())


if __name__ == "__main__":
    unittest.main()
