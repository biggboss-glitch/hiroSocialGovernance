"""Tests for the Hiro Social Governance environment."""

import pytest

from src.environment import HiroSocialEnv
from src.models import Action, ActionType, Observation


class TestEnvironment:
    """Test core environment functionality."""

    def test_environment_initialization(self):
        """Test environment can be initialized."""
        env = HiroSocialEnv()
        assert env is not None
        assert not env._is_initialized

    def test_reset_returns_observation(self):
        """reset() must return Observation."""
        env = HiroSocialEnv()
        obs = env.reset(task="easy")

        assert obs is not None
        assert isinstance(obs, Observation)
        assert hasattr(obs, 'step')
        assert hasattr(obs, 'network_toxicity')
        assert hasattr(obs, 'num_agents')

    def test_reset_with_invalid_task_raises(self):
        """reset() with invalid task must raise ValueError."""
        env = HiroSocialEnv()

        with pytest.raises(ValueError):
            env.reset(task="invalid_task")

    def test_step_returns_correct_tuple(self):
        """step() must return (obs, reward, done, info)."""
        env = HiroSocialEnv()
        env.reset(task="easy")

        action = Action(action_type=ActionType.NO_ACTION)
        obs, reward, done, info = env.step(action)

        assert obs is None or isinstance(obs, Observation)
        assert isinstance(reward.total, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_before_reset_raises(self):
        """step() before reset() must raise RuntimeError."""
        env = HiroSocialEnv()
        action = Action(action_type=ActionType.NO_ACTION)

        with pytest.raises(RuntimeError):
            env.step(action)

    def test_reward_in_valid_range(self):
        """Reward must be in [-1, 1]."""
        env = HiroSocialEnv()
        env.reset(task="easy")

        for _ in range(5):
            action = Action(action_type=ActionType.NO_ACTION)
            _, reward, done, _ = env.step(action)

            assert -1.0 <= reward.total <= 1.0

            if done:
                break

    def test_state_returns_dict(self):
        """state() must return dictionary."""
        env = HiroSocialEnv()
        env.reset(task="easy")

        state = env.state()

        assert isinstance(state, dict)
        assert "step_count" in state or "is_initialized" in state

    def test_episode_terminates(self):
        """Episode must eventually terminate."""
        env = HiroSocialEnv()
        env.reset(task="easy")

        done = False
        steps = 0
        max_steps = 200

        while not done and steps < max_steps:
            action = Action(action_type=ActionType.NO_ACTION)
            _, _, done, _ = env.step(action)
            steps += 1

        assert done, "Episode did not terminate"
        assert steps < max_steps, "Episode exceeded max steps"

    def test_get_task_grade(self):
        """get_task_grade() must return score in [0, 1]."""
        env = HiroSocialEnv()
        env.reset(task="easy")

        # Run episode
        done = False
        while not done:
            action = Action(action_type=ActionType.NO_ACTION)
            _, _, done, _ = env.step(action)

        grade = env.get_task_grade()

        assert isinstance(grade, float)
        assert 0.0 <= grade <= 1.0

    def test_get_available_tasks(self):
        """get_available_tasks() must return task list."""
        env = HiroSocialEnv()
        tasks = env.get_available_tasks()

        assert isinstance(tasks, dict)
        assert "easy" in tasks
        assert "medium" in tasks
        assert "hard" in tasks

    def test_all_tasks_runnable(self):
        """All tasks must be runnable."""
        env = HiroSocialEnv()
        tasks = ["easy", "medium", "hard"]

        for task_id in tasks:
            obs = env.reset(task=task_id)
            assert obs is not None

            # Take one step
            action = Action(action_type=ActionType.NO_ACTION)
            obs, reward, done, info = env.step(action)
            assert isinstance(reward.total, float)

    def test_seed_reproducibility(self):
        """Same seed should produce same initial observation."""
        env = HiroSocialEnv()

        obs1 = env.reset(task="easy", seed=42)
        obs2 = env.reset(task="easy", seed=42)

        assert obs1.num_agents == obs2.num_agents
        assert obs1.network_toxicity == obs2.network_toxicity


class TestActions:
    """Test moderation actions."""

    def test_remove_post(self):
        """Test remove_post action."""
        env = HiroSocialEnv()
        env.reset(task="easy")

        # Generate some posts first
        for _ in range(3):
            env.step(Action(action_type=ActionType.NO_ACTION))

        # Try to remove (may or may not have posts)
        action = Action(action_type=ActionType.REMOVE_POST, target_id="post_001")
        obs, reward, done, info = env.step(action)

        assert isinstance(reward.total, float)

    def test_suspend_user(self):
        """Test suspend_user action."""
        env = HiroSocialEnv()
        env.reset(task="easy")

        action = Action(action_type=ActionType.SUSPEND_USER, target_id="agent_001")
        obs, reward, done, info = env.step(action)

        assert isinstance(reward.total, float)

    def test_inject_counter_info(self):
        """Test inject_counter_info action."""
        env = HiroSocialEnv()
        env.reset(task="easy")

        action = Action(
            action_type=ActionType.INJECT_COUNTER_INFO,
            content="Fact-check: Verify information before sharing."
        )
        obs, reward, done, info = env.step(action)

        assert isinstance(reward.total, float)

    def test_flag_post(self):
        """Test flag_post action."""
        env = HiroSocialEnv()
        env.reset(task="easy", seed=42)

        # Generate posts
        env.step(Action(action_type=ActionType.NO_ACTION))

        action = Action(action_type=ActionType.FLAG_POST, target_id="test_post")
        obs, reward, done, info = env.step(action)
        assert isinstance(reward.total, float)

    def test_downrank_post(self):
        """Test downrank_post action."""
        env = HiroSocialEnv()
        env.reset(task="easy", seed=42)

        action = Action(action_type=ActionType.DOWNRANK_POST, target_id="test_post")
        obs, reward, done, info = env.step(action)
        assert isinstance(reward.total, float)

    def test_boost_post(self):
        """Test boost_post action."""
        env = HiroSocialEnv()
        env.reset(task="easy", seed=42)

        action = Action(action_type=ActionType.BOOST_POST, target_id="test_post")
        obs, reward, done, info = env.step(action)
        assert isinstance(reward.total, float)

    def test_warn_user(self):
        """Test warn_user action."""
        env = HiroSocialEnv()
        env.reset(task="easy", seed=42)

        action = Action(action_type=ActionType.WARN_USER, target_id="agent_000")
        obs, reward, done, info = env.step(action)
        assert isinstance(reward.total, float)


class TestGraders:
    """Test task graders."""

    def test_easy_task_grader(self):
        """Test easy task grader."""
        env = HiroSocialEnv()
        env.reset(task="easy", seed=42)

        done = False
        while not done:
            action = Action(action_type=ActionType.NO_ACTION)
            _, _, done, _ = env.step(action)

        grade = env.get_task_grade()
        assert 0.0 <= grade <= 1.0

    def test_medium_task_grader(self):
        """Test medium task grader."""
        env = HiroSocialEnv()
        env.reset(task="medium", seed=42)

        # Run partial episode
        for _ in range(10):
            action = Action(action_type=ActionType.NO_ACTION)
            env.step(action)

        grade = env.get_task_grade()
        assert 0.0 <= grade <= 1.0

    def test_hard_task_grader(self):
        """Test hard task grader."""
        env = HiroSocialEnv()
        env.reset(task="hard", seed=42)

        # Run partial episode
        for _ in range(10):
            action = Action(action_type=ActionType.NO_ACTION)
            env.step(action)

        grade = env.get_task_grade()
        assert 0.0 <= grade <= 1.0

    def test_full_easy_episode_grade(self):
        """Full easy episode should produce reasonable grade."""
        env = HiroSocialEnv()
        env.reset(task="easy", seed=42)

        done = False
        steps = 0
        while not done:
            action = Action(action_type=ActionType.NO_ACTION)
            _, _, done, _ = env.step(action)
            steps += 1

        grade = env.get_task_grade()
        assert steps == 50, "Easy task should run for 50 steps"
        assert 0.3 <= grade <= 0.8, f"Easy no-action grade should be reasonable, got {grade}"


class TestHardTaskOutbreak:
    """Test Hard task viral outbreak feature."""

    def test_outbreak_triggers(self):
        """Hard task outbreak should trigger at step 50."""
        env = HiroSocialEnv()
        env.reset(task="hard", seed=42)

        task = env._task
        assert hasattr(task, 'outbreak_triggered')
        assert not task.outbreak_triggered

        # Run up to step 50
        for _ in range(51):
            action = Action(action_type=ActionType.NO_ACTION)
            env.step(action)

        assert task.outbreak_triggered, "Outbreak should have triggered by step 51"

    def test_hard_full_episode(self):
        """Hard task should complete with valid grade."""
        env = HiroSocialEnv()
        env.reset(task="hard", seed=42)

        done = False
        steps = 0
        while not done:
            action = Action(action_type=ActionType.NO_ACTION)
            _, _, done, _ = env.step(action)
            steps += 1

        grade = env.get_task_grade()
        assert steps == 150, "Hard task should run for 150 steps"
        assert 0.0 <= grade <= 1.0


class TestAPIServer:
    """Test FastAPI API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from httpx import ASGITransport, AsyncClient
        from api.server import app
        import asyncio

        transport = ASGITransport(app=app)
        # Use sync test via asyncio
        return transport, app

    def test_health_endpoint(self):
        """Test /health endpoint."""
        from fastapi.testclient import TestClient
        from api.server import app

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    def test_root_endpoint(self):
        """Test / root endpoint."""
        from fastapi.testclient import TestClient
        from api.server import app

        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "endpoints" in data

    def test_tasks_endpoint(self):
        """Test /tasks endpoint."""
        from fastapi.testclient import TestClient
        from api.server import app

        with TestClient(app) as client:
            response = client.get("/tasks")
            assert response.status_code == 200
            data = response.json()
            assert "tasks" in data
            assert "easy" in data["tasks"]

    def test_reset_and_step_flow(self):
        """Test /reset -> /step -> /grade flow."""
        from fastapi.testclient import TestClient
        from api.server import app

        with TestClient(app) as client:
            # Reset
            response = client.post("/reset", json={"task": "easy", "seed": 42})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

            # Step
            response = client.post("/step", json={
                "action": {
                    "action_type": "no_action"
                }
            })
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "reward" in data
            assert isinstance(data["done"], bool)

            # State
            response = client.get("/state")
            assert response.status_code == 200

            # Grade
            response = client.get("/grade")
            assert response.status_code == 200
            data = response.json()
            assert "grade" in data

    def test_reset_invalid_task(self):
        """Test /reset with invalid task returns 400."""
        from fastapi.testclient import TestClient
        from api.server import app

        with TestClient(app) as client:
            response = client.post("/reset", json={"task": "nonexistent"})
            assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
