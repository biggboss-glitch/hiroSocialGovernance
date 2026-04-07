"""Hard Task: Large network with viral misinformation outbreak."""

import random
from typing import Dict

from src.dynamics import OutbreakSimulator
from src.models import TaskConfig
from src.tasks.easy import EasyTask
from src.tasks.base import TaskRegistry


@TaskRegistry.register
class HardTask(EasyTask):
    """
    Hard Task: Large network (50 agents) with viral misinformation outbreak.
    
    Focus: Handle a sudden viral misinformation campaign while maintaining trust.
    
    Configuration:
    - 50 agents
    - 20% trolls
    - 20% bots
    - 10% influencers
    - 150 max steps
    - Viral outbreak at step 50
    """
    
    task_id = "hard"
    name = "Crisis Network Governance"
    difficulty = "hard"
    description = "Manage a large network during a viral misinformation outbreak"
    
    def __init__(self, config: TaskConfig = None):
        """Initialize hard task."""
        if config is None:
            config = TaskConfig(
                task_id=self.task_id,
                name=self.name,
                description=self.description,
                difficulty=self.difficulty,
                num_agents=50,
                max_steps=150,
                troll_ratio=0.20,
                bot_ratio=0.20,
                influencer_ratio=0.10,
                viral_outbreak=True,
                outbreak_step=50
            )
        super().__init__(config)
        
        self.outbreak_triggered = False
        self.outbreak_simulator = OutbreakSimulator()
    
    def step(self, action):
        """Execute one step with potential outbreak."""
        # Check for outbreak
        if (self.config.viral_outbreak and 
            self.config.outbreak_step and 
            self._step_count == self.config.outbreak_step and 
            not self.outbreak_triggered):
            
            self._trigger_outbreak()
            self.outbreak_triggered = True
        
        # Continue with normal step
        return super().step(action)
    
    def _trigger_outbreak(self):
        """Trigger viral misinformation outbreak."""
        # Create viral posts
        viral_posts = self.outbreak_simulator.trigger_outbreak(
            self.agents, self.posts, intensity=0.9
        )
        self.posts.extend(viral_posts)
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate metrics with outbreak indicator."""
        metrics = super()._calculate_metrics()
        metrics["outbreak_active"] = 1.0 if self.outbreak_triggered else 0.0
        return metrics
