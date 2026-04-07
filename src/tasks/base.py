"""Base task class for Hiro Social Governance Environment."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.models import Action, Observation, TaskConfig


class BaseTask(ABC):
    """
    Abstract base class for all governance tasks.
    
    All tasks must implement reset(), step(), and grade() methods.
    """
    
    task_id: str = "base"
    name: str = "Base Task"
    difficulty: str = "easy"
    description: str = "Base task description"
    
    def __init__(self, config: Optional[TaskConfig] = None):
        """Initialize task with configuration."""
        self.config = config or self._default_config()
        self._step_count = 0
        self._is_done = False
        self._history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset task state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Args:
            action: Agent action
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def grade(self) -> float:
        """
        Calculate final task score.
        
        Returns:
            Score between 0.0 and 1.0
        """
        pass
    
    def get_config(self) -> TaskConfig:
        """Get task configuration."""
        return self.config
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get task-specific metrics."""
        return {
            "step_count": self._step_count,
            "max_steps": self.config.max_steps,
            "progress": self._step_count / self.config.max_steps,
            "is_done": self._is_done
        }
    
    def _default_config(self) -> TaskConfig:
        """Get default task configuration."""
        return TaskConfig(
            task_id=self.task_id,
            name=self.name,
            description=self.description,
            difficulty=self.difficulty,
            num_agents=10,
            max_steps=50
        )


class TaskRegistry:
    """Registry for available tasks."""
    
    _tasks: Dict[str, type] = {}
    
    @classmethod
    def register(cls, task_class: type) -> type:
        """Register a task class."""
        cls._tasks[task_class.task_id] = task_class
        return task_class
    
    @classmethod
    def get(cls, task_id: str) -> Optional[type]:
        """Get task class by ID."""
        return cls._tasks.get(task_id)
    
    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered task IDs."""
        return list(cls._tasks.keys())
    
    @classmethod
    def create_task(cls, task_id: str, config: Optional[TaskConfig] = None) -> Optional[BaseTask]:
        """Create a task instance."""
        task_class = cls.get(task_id)
        if task_class:
            return task_class(config)
        return None
