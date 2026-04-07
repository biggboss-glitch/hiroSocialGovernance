"""Typed Pydantic models for Hiro Social Governance Environment."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentType(str, Enum):
    """Types of social media agents."""
    NORMAL = "normal"
    TROLL = "troll"
    INFLUENCER = "influencer"
    BOT = "bot"


class ActionType(str, Enum):
    """Types of moderation actions."""
    REMOVE_POST = "remove_post"
    FLAG_POST = "flag_post"
    DOWNRANK_POST = "downrank_post"
    BOOST_POST = "boost_post"
    WARN_USER = "warn_user"
    SUSPEND_USER = "suspend_user"
    INJECT_COUNTER_INFO = "inject_counter_info"
    NO_ACTION = "no_action"


class Post(BaseModel):
    """A social media post."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "post_001",
                "author_id": "agent_01",
                "content": "This is a sample post",
                "toxicity": 0.2,
                "misinformation": 0.1,
                "emotional_intensity": 0.5,
                "views": 100,
                "likes": 10
            }
        }
    )

    id: str = Field(..., description="Unique post identifier")
    author_id: str = Field(..., description="ID of the author agent")
    content: str = Field(..., description="Post content")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Content metrics
    toxicity: float = Field(0.0, ge=0.0, le=1.0, description="Toxicity score")
    misinformation: float = Field(0.0, ge=0.0, le=1.0, description="Misinformation score")
    emotional_intensity: float = Field(0.5, ge=0.0, le=1.0, description="Emotional intensity")

    # Engagement metrics
    views: int = Field(0, ge=0, description="Number of views")
    likes: int = Field(0, ge=0, description="Number of likes")
    shares: int = Field(0, ge=0, description="Number of shares")
    comments: int = Field(0, ge=0, description="Number of comments")

    # Moderation status
    is_removed: bool = Field(False, description="Whether post was removed")
    is_flagged: bool = Field(False, description="Whether post was flagged")
    visibility: float = Field(1.0, ge=0.0, le=1.0, description="Visibility score (downranking)")


class Agent(BaseModel):
    """A social media user agent."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "agent_01",
                "name": "User123",
                "agent_type": "normal",
                "belief": 0.3,
                "toxicity": 0.1,
                "credibility": 0.7,
                "influence": 2.5,
                "trust_in_platform": 0.6
            }
        }
    )

    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent display name")
    agent_type: AgentType = Field(AgentType.NORMAL, description="Type of agent")

    # Core attributes
    belief: float = Field(0.0, ge=-1.0, le=1.0, description="Political/social belief (-1 to +1)")
    toxicity: float = Field(0.0, ge=0.0, le=1.0, description="Base toxicity tendency")
    credibility: float = Field(0.5, ge=0.0, le=1.0, description="Information credibility")
    influence: float = Field(1.0, ge=0.0, description="Social influence/power")
    trust_in_platform: float = Field(0.5, ge=0.0, le=1.0, description="Trust in platform")

    # State
    is_suspended: bool = Field(False, description="Whether agent is suspended")
    warning_count: int = Field(0, ge=0, description="Number of warnings received")
    posts_today: int = Field(0, ge=0, description="Posts created today")


class Observation(BaseModel):
    """Environment observation sent to the governance agent."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step": 10,
                "max_steps": 100,
                "recent_posts": [],
                "network_toxicity": 0.3,
                "misinformation_index": 0.2,
                "engagement_score": 150.5,
                "avg_trust_score": 0.6,
                "belief_polarization": 0.4
            }
        }
    )

    step: int = Field(..., ge=0, description="Current simulation step")
    max_steps: int = Field(..., ge=1, description="Maximum steps for episode")

    # Recent content
    recent_posts: List[Post] = Field(default_factory=list, description="Recent posts")
    flagged_posts: List[Post] = Field(default_factory=list, description="Flagged posts")

    # Network metrics
    num_agents: int = Field(..., ge=0, description="Total number of agents")
    active_agents: int = Field(..., ge=0, description="Number of non-suspended agents")

    # Content quality metrics
    network_toxicity: float = Field(0.0, ge=0.0, le=1.0, description="Overall network toxicity")
    misinformation_index: float = Field(0.0, ge=0.0, le=1.0, description="Misinformation spread level")

    # Engagement metrics
    engagement_score: float = Field(0.0, ge=0.0, description="Platform engagement score")
    total_posts: int = Field(0, ge=0, description="Total posts in simulation")
    total_interactions: int = Field(0, ge=0, description="Total likes + shares + comments")

    # Trust and polarization
    avg_trust_score: float = Field(0.5, ge=0.0, le=1.0, description="Average agent trust")
    belief_polarization: float = Field(0.0, ge=0.0, le=1.0, description="Belief polarization (0=unified, 1=polarized)")

    # Moderation metrics
    posts_removed: int = Field(0, ge=0, description="Posts removed by moderator")
    users_warned: int = Field(0, ge=0, description="Users warned")
    users_suspended: int = Field(0, ge=0, description="Users suspended")
    counter_info_injected: int = Field(0, ge=0, description="Counter-information posts injected")


class Action(BaseModel):
    """Moderation action taken by the governance agent."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action_type": "remove_post",
                "target_id": "post_001",
                "reason": "High toxicity"
            }
        }
    )

    action_type: ActionType = Field(..., description="Type of action")
    target_id: Optional[str] = Field(None, description="Target post or user ID")
    reason: Optional[str] = Field(None, description="Reason for action")

    # For counter-information injection
    content: Optional[str] = Field(None, description="Content for counter-info post")


class Reward(BaseModel):
    """Reward signal returned by the environment."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 0.45,
                "toxicity_reduction": 0.15,
                "misinformation_control": 0.10,
                "engagement": 0.12,
                "trust_score": 0.08,
                "over_moderation_penalty": -0.05,
                "feedback": "Good toxicity control, but slightly over-moderating"
            }
        }
    )

    total: float = Field(..., ge=-1.0, le=1.0, description="Total reward value")

    # Component breakdown
    toxicity_reduction: float = Field(0.0, description="Reward for reducing toxicity")
    misinformation_control: float = Field(0.0, description="Reward for controlling misinformation")
    engagement: float = Field(0.0, description="Reward for engagement")
    trust_score: float = Field(0.0, description="Reward for maintaining trust")
    over_moderation_penalty: float = Field(0.0, description="Penalty for over-moderation")

    # Metadata
    feedback: str = Field("", description="Human-readable explanation")


class TaskConfig(BaseModel):
    """Configuration for a task."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "easy",
                "name": "Small Network Governance",
                "num_agents": 10,
                "max_steps": 50
            }
        }
    )

    task_id: str = Field(..., description="Task identifier")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    difficulty: str = Field(..., description="Task difficulty")

    # Network parameters
    num_agents: int = Field(10, ge=1, description="Number of agents")
    max_steps: int = Field(100, ge=1, description="Maximum simulation steps")

    # Agent composition
    troll_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Ratio of troll agents")
    bot_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Ratio of bot agents")
    influencer_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Ratio of influencer agents")

    # Special conditions
    viral_outbreak: bool = Field(False, description="Whether to simulate viral misinformation")
    outbreak_step: Optional[int] = Field(None, description="Step when outbreak occurs")


class EnvironmentState(BaseModel):
    """Complete environment state for saving/loading."""
    step: int = Field(0, description="Current step")
    agents: List[Agent] = Field(default_factory=list, description="All agents")
    posts: List[Post] = Field(default_factory=list, description="All posts")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Action history")

    # Baseline metrics for reward calculation
    baseline_toxicity: float = Field(0.0, description="Initial toxicity level")
    baseline_misinfo: float = Field(0.0, description="Initial misinformation level")
    baseline_engagement: float = Field(0.0, description="Initial engagement")
    baseline_trust: float = Field(0.5, description="Initial trust level")
