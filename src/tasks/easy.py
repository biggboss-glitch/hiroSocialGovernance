"""Easy Task: Small network with basic toxicity detection."""

import random
from typing import Any, Dict, List, Optional, Tuple

from src.agents import AgentBehavior, AgentFactory
from src.dynamics import (
    BeliefDynamics,
    NetworkMetrics,
    OutbreakSimulator,
    ViralityModel,
)
from src.models import Action, Agent, AgentType, Observation, Post, TaskConfig
from src.reward import Grader, RewardCalculator
from src.tasks.base import BaseTask, TaskRegistry


@TaskRegistry.register
class EasyTask(BaseTask):
    """
    Easy Task: Small network (10 agents) with basic toxicity detection.
    
    Focus: Learn to identify and remove toxic content.
    
    Configuration:
    - 10 agents
    - 20% trolls
    - 10% bots
    - 10% influencers
    - 50 max steps
    """
    
    task_id = "easy"
    name = "Small Network Governance"
    difficulty = "easy"
    description = "Manage a small social network and control toxic content"
    
    def __init__(self, config: Optional[TaskConfig] = None):
        """Initialize easy task."""
        if config is None:
            config = TaskConfig(
                task_id=self.task_id,
                name=self.name,
                description=self.description,
                difficulty=self.difficulty,
                num_agents=10,
                max_steps=50,
                troll_ratio=0.2,
                bot_ratio=0.1,
                influencer_ratio=0.1,
                viral_outbreak=False
            )
        super().__init__(config)
        
        self.agents: List[Agent] = []
        self.posts: List[Post] = []
        self.reward_calculator: Optional[RewardCalculator] = None
        self.virality_model = ViralityModel()
        self.belief_dynamics = BeliefDynamics()
        self.network_metrics = NetworkMetrics()
    
    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset task and return initial observation."""
        if seed is not None:
            random.seed(seed)
        
        # Create agent network
        self.agents = AgentFactory.create_network(
            num_agents=self.config.num_agents,
            troll_ratio=self.config.troll_ratio,
            bot_ratio=self.config.bot_ratio,
            influencer_ratio=self.config.influencer_ratio,
            seed=seed
        )
        
        # Initialize posts
        self.posts = []
        
        # Reset state
        self._step_count = 0
        self._is_done = False
        self._history = []
        
        # Calculate baseline metrics for reward
        baseline_metrics = self._calculate_metrics()
        self.reward_calculator = RewardCalculator(
            baseline_toxicity=baseline_metrics["network_toxicity"],
            baseline_misinfo=baseline_metrics["misinformation_index"],
            baseline_engagement=baseline_metrics["engagement_score"],
            baseline_trust=baseline_metrics["avg_trust_score"]
        )
        
        # Generate initial posts
        self._generate_posts()
        
        return self._create_observation()
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute one step."""
        self._step_count += 1
        
        # Apply moderation action
        self._apply_action(action)
        
        # Generate new posts from agents
        self._generate_posts()
        
        # Simulate content spread
        self._simulate_spread()
        
        # Update agent beliefs
        self._update_beliefs()
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Calculate reward
        posts_removed = sum(1 for p in self.posts if p.is_removed)
        users_suspended = sum(1 for a in self.agents if a.is_suspended)
        
        reward_components = self.reward_calculator.calculate(
            metrics,
            action.action_type.value,
            posts_removed,
            users_suspended
        )
        
        reward = reward_components["total"]
        
        # Check if done
        self._is_done = self._step_count >= self.config.max_steps
        
        # Create observation
        observation = self._create_observation()
        
        # Build info
        info = {
            "metrics": metrics,
            "reward_components": reward_components,
            "action": action.action_type.value,
            "posts_removed": posts_removed,
            "users_suspended": users_suspended
        }
        
        # Record history
        self._history.append({
            "step": self._step_count,
            "action": action.action_type.value,
            "reward": reward,
            "metrics": metrics
        })
        
        return observation, reward, self._is_done, info
    
    def grade(self) -> float:
        """Calculate final grade — always returns strictly in (0, 1)."""
        import math
        final_metrics = self._calculate_metrics()
        raw = Grader.grade(final_metrics, self._step_count, self.config.max_steps)
        raw = float(raw)
        if not math.isfinite(raw):
            raw = 0.5
        return max(0.01, min(0.99, raw))
    
    def _generate_posts(self):
        """Generate posts from agents."""
        for agent in self.agents:
            post = AgentBehavior.generate_post(agent)
            if post:
                self.posts.append(post)
    
    def _simulate_spread(self):
        """Simulate content spread through network."""
        avg_belief = sum(a.belief for a in self.agents) / len(self.agents)
        
        for post in self.posts:
            if not post.is_removed:
                viewers = [a for a in self.agents if a.id != post.author_id]
                self.virality_model.simulate_engagement(post, viewers, avg_belief)
    
    def _update_beliefs(self):
        """Update agent beliefs based on content."""
        for agent in self.agents:
            # Get posts this agent viewed (simplified)
            viewed_posts = [p for p in self.posts[-10:] if not p.is_removed]
            agent.belief = self.belief_dynamics.update_belief_from_content(
                agent, viewed_posts
            )
    
    def _apply_action(self, action: Action):
        """Apply moderation action."""
        if action.action_type.value == "no_action":
            return
        
        if action.action_type.value == "remove_post" and action.target_id:
            for post in self.posts:
                if post.id == action.target_id:
                    post.is_removed = True
                    break
        
        elif action.action_type.value == "flag_post" and action.target_id:
            for post in self.posts:
                if post.id == action.target_id:
                    post.is_flagged = True
                    break
        
        elif action.action_type.value == "downrank_post" and action.target_id:
            for post in self.posts:
                if post.id == action.target_id:
                    post.visibility = max(0.0, post.visibility - 0.5)
                    break
        
        elif action.action_type.value == "boost_post" and action.target_id:
            for post in self.posts:
                if post.id == action.target_id:
                    post.visibility = min(1.0, post.visibility + 0.3)
                    break
        
        elif action.action_type.value == "warn_user" and action.target_id:
            for agent in self.agents:
                if agent.id == action.target_id:
                    agent.warning_count += 1
                    agent.trust_in_platform = max(0.0, agent.trust_in_platform - 0.15)
                    break
        
        elif action.action_type.value == "suspend_user" and action.target_id:
            for agent in self.agents:
                if agent.id == action.target_id:
                    agent.is_suspended = True
                    agent.trust_in_platform = max(0.0, agent.trust_in_platform - 0.4)
                    break
        
        elif action.action_type.value == "inject_counter_info":
            # Create a counter-information post
            counter_post = Post(
                id=f"counter_{self._step_count}",
                author_id="platform",
                content=action.content or "Fact-check: Recent claims need verification.",
                toxicity=0.0,
                misinformation=0.0,
                emotional_intensity=0.3,
                visibility=0.8
            )
            self.posts.append(counter_post)
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate network metrics."""
        return {
            "network_toxicity": self.network_metrics.calculate_network_toxicity(
                self.agents, self.posts
            ),
            "misinformation_index": self.network_metrics.calculate_misinformation_index(
                self.agents, self.posts
            ),
            "engagement_score": self.network_metrics.calculate_engagement_score(self.posts),
            "avg_trust_score": self.network_metrics.calculate_avg_trust(self.agents),
            "belief_polarization": self.belief_dynamics.calculate_polarization(self.agents),
            "total_posts": len(self.posts)
        }
    
    def _create_observation(self) -> Observation:
        """Create observation from current state."""
        metrics = self._calculate_metrics()
        
        # Get recent posts (last 20)
        recent_posts = [p for p in self.posts[-20:] if not p.is_removed]
        
        # Get flagged posts
        flagged_posts = [p for p in self.posts if p.is_flagged and not p.is_removed]
        
        return Observation(
            step=self._step_count,
            max_steps=self.config.max_steps,
            recent_posts=recent_posts,
            flagged_posts=flagged_posts,
            num_agents=len(self.agents),
            active_agents=sum(1 for a in self.agents if not a.is_suspended),
            network_toxicity=metrics["network_toxicity"],
            misinformation_index=metrics["misinformation_index"],
            engagement_score=metrics["engagement_score"],
            total_posts=len(self.posts),
            total_interactions=sum(
                p.likes + p.shares + p.comments for p in self.posts
            ),
            avg_trust_score=metrics["avg_trust_score"],
            belief_polarization=metrics["belief_polarization"],
            posts_removed=sum(1 for p in self.posts if p.is_removed),
            users_warned=sum(1 for a in self.agents if a.warning_count > 0),
            users_suspended=sum(1 for a in self.agents if a.is_suspended),
            counter_info_injected=sum(
                1 for p in self.posts if p.author_id == "platform"
            )
        )
