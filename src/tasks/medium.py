"""Medium Task: Medium network with balanced moderation."""

from src.models import TaskConfig
from src.tasks.easy import EasyTask
from src.tasks.base import TaskRegistry


@TaskRegistry.register
class MediumTask(EasyTask):
    """
    Medium Task: Medium network (25 agents) with balanced moderation.
    
    Focus: Balance content moderation with user engagement and trust.
    
    Configuration:
    - 25 agents
    - 15% trolls
    - 15% bots
    - 15% influencers
    - 100 max steps
    - More complex dynamics
    """
    
    task_id = "medium"
    name = "Balanced Network Governance"
    difficulty = "medium"
    description = "Manage a medium-sized network balancing safety, engagement, and trust"
    
    def __init__(self, config: TaskConfig = None):
        """Initialize medium task."""
        if config is None:
            config = TaskConfig(
                task_id=self.task_id,
                name=self.name,
                description=self.description,
                difficulty=self.difficulty,
                num_agents=25,
                max_steps=100,
                troll_ratio=0.15,
                bot_ratio=0.15,
                influencer_ratio=0.15,
                viral_outbreak=False
            )
        super().__init__(config)
