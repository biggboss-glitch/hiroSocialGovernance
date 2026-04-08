"""Main Hiro Social Governance OpenEnv environment."""

from typing import Any, Dict, List, Optional, Tuple

from src.models import Action, Observation, Reward, TaskConfig  # type: ignore
from src.reward import Grader  # type: ignore
from src.tasks import TaskRegistry  # type: ignore
from src.tasks.base import BaseTask  # type: ignore


class HiroSocialEnv:
    """
    Hiro: Multi-Agent Social Governance Environment.
    
    Simulates a social media platform where an AI agent must govern
    by balancing content safety, misinformation control, engagement,
    and platform trust.
    
    Implements OpenEnv specification:
    - reset(task, seed) -> Observation
    - step(action) -> (Observation, Reward, done, info)
    - state() -> dict
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "easy": TaskConfig(
            task_id="easy",
            name="Small Network Governance",
            description="Manage a small social network and control toxic content",
            difficulty="easy",
            num_agents=10,
            max_steps=50
        ),
        "medium": TaskConfig(
            task_id="medium",
            name="Balanced Network Governance",
            description="Manage a medium network balancing moderation and engagement",
            difficulty="medium",
            num_agents=25,
            max_steps=100
        ),
        "hard": TaskConfig(
            task_id="hard",
            name="Crisis Network Governance",
            description="Manage a large network during viral misinformation outbreak",
            difficulty="hard",
            num_agents=50,
            max_steps=150,
            troll_ratio=0.20,
            bot_ratio=0.20,
            influencer_ratio=0.10,
            viral_outbreak=True,
            outbreak_step=50
        )
    }
    
    def __init__(
        self,
        task: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize environment.
        
        Args:
            task: Task identifier (easy, medium, hard)
            config: Optional configuration overrides
        """
        self._task_id = task
        self._task: Optional[BaseTask] = None
        self._config_override = config
        self._is_initialized = False
    
    def reset(
        self,
        task: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Observation:
        """
        Reset environment and start new episode.
        
        Args:
            task: Task to run (if not specified in __init__)
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation
            
        Raises:
            ValueError: If task is invalid or not specified
        """
        # Determine task
        task_id = task or self._task_id
        if task_id is None:
            raise ValueError("Task must be specified in reset() or __init__()")
        
        # Create task instance
        task_class = TaskRegistry.get(task_id)
        if task_class is None:
            available = TaskRegistry.list_tasks()
            raise ValueError(f"Unknown task: {task_id}. Available: {available}")
        
        # Get task config
        config = self.DEFAULT_CONFIG.get(task_id)
        if self._config_override:
            # Apply overrides
            for key, value in self._config_override.items():
                setattr(config, key, value)
        
        self._task = task_class(config)
        
        # Reset task and get initial observation
        observation = self._task.reset(seed=seed)
        
        self._is_initialized = True
        self._task_id = task_id
        
        return observation
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Agent action (must be valid Action type)
            
        Returns:
            Tuple of (observation, reward, done, info)
            
        Raises:
            RuntimeError: If called before reset()
            TypeError: If action is not Action type
        """
        # Validate initialization
        if not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Validate action type
        if not isinstance(action, Action):
            raise TypeError(f"Action must be Action type, got {type(action)}")
        
        # Execute task step
        obs, reward_value, done, info = self._task.step(action)
        
        # Create Reward object
        reward = Reward(
            total=reward_value,
            toxicity_reduction=info.get("reward_components", {}).get("toxicity_reduction", 0.0),
            misinformation_control=info.get("reward_components", {}).get("misinformation_control", 0.0),
            engagement=info.get("reward_components", {}).get("engagement", 0.0),
            trust_score=info.get("reward_components", {}).get("trust_score", 0.0),
            over_moderation_penalty=info.get("reward_components", {}).get("over_moderation_penalty", 0.0),
            feedback=info.get("reward_components", {}).get("feedback", "")
        )
        
        # Add environment info
        info.update({
            "task_id": self._task_id,
            "step_count": self._task._step_count,
            "max_steps": self._task.config.max_steps
        })
        
        return obs, reward, done, info
    
    def state(self) -> Dict[str, Any]:
        """
        Get current environment state.
        
        Returns:
            Dictionary containing:
            - task_id: Current task
            - step_count: Current step number
            - max_steps: Maximum steps
            - metrics: Network metrics
            - task_metrics: Task-specific metrics
            - is_initialized: Whether environment is initialized
        """
        if not self._is_initialized:
            return {
                "is_initialized": False,
                "message": "Environment not initialized. Call reset() first."
            }
        
        task_metrics = self._task.get_metrics()
        
        return {
            "task_id": self._task_id,
            "step_count": task_metrics.get("step_count", 0),
            "max_steps": task_metrics.get("max_steps", 0),
            "progress": task_metrics.get("progress", 0.0),
            "is_done": task_metrics.get("is_done", False),
            "is_initialized": self._is_initialized,
            "task_metrics": task_metrics
        }
    
    def render(self, mode: str = "human") -> str:
        """
        Render environment state as string.
        
        Args:
            mode: "human" (readable) or "json" (machine-readable)
            
        Returns:
            Formatted state representation
        """
        state = self.state()
        
        if mode == "human":
            lines = [
                "=" * 60,
                "Hiro Social Governance Environment",
                "=" * 60,
                f"Task: {state.get('task_id', 'N/A')}",
                f"Step: {state.get('step_count', 0)} / {state.get('max_steps', 0)}",
                f"Progress: {state.get('progress', 0):.1%}",
                "-" * 60
            ]
            return "\n".join(lines)
        
        else:  # json mode
            import json
            return json.dumps(state, indent=2, default=str)
    
    def close(self) -> None:
        """Clean up resources."""
        self._task = None
        self._is_initialized = False
    
    def get_task_grade(self) -> float:
        """
        Get final task grade.
        
        Returns:
            Score between 0.0 and 1.0
            
        Raises:
            RuntimeError: If called before reset()
        """
        if not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        grade = self._task.grade()
        # Safety: ensure score is strictly in (0, 1) — validator rejects 0.0 and 1.0
        grade = max(0.001, min(0.999, float(grade)))
        return round(grade, 4)
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get information about current task.
        
        Returns:
            Task information dictionary
            
        Raises:
            RuntimeError: If called before reset()
        """
        if not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        config = self._task.get_config()
        return {
            "task_id": config.task_id,
            "name": config.name,
            "description": config.description,
            "difficulty": config.difficulty,
            "num_agents": config.num_agents,
            "max_steps": config.max_steps
        }
    
    def get_available_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of available tasks.
        
        Returns:
            Dictionary mapping task IDs to task info
        """
        tasks = {}
        
        for task_id in TaskRegistry.list_tasks():
            task_class = TaskRegistry.get(task_id)
            if task_class:
                tasks[task_id] = {
                    "name": task_class.name,
                    "difficulty": task_class.difficulty,
                    "description": task_class.description
                }
        
        return tasks
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Get detailed results for the current episode.
        
        Returns:
            Dictionary with detailed results
        """
        if not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        raw_grade = self._task.grade()
        clamped_grade = max(0.001, min(0.999, float(raw_grade)))
        return {
            "task_id": self._task_id,
            "final_grade": round(clamped_grade, 4),
            "total_steps": self._task._step_count,
            "max_steps": self._task.config.max_steps,
            "history": self._task._history
        }
