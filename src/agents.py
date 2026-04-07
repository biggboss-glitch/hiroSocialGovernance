"""Agent system with rule-based behavior for social simulation."""

import random
import uuid
from typing import List, Optional

from src.models import Agent, AgentType, Post


class AgentBehavior:
    """Rule-based behavior patterns for different agent types."""
    
    @staticmethod
    def generate_post(agent: Agent, current_belief_climate: float = 0.0) -> Optional[Post]:
        """
        Generate a post based on agent type and state.
        
        Args:
            agent: The agent creating the post
            current_belief_climate: Average belief in network
            
        Returns:
            Post object or None if agent doesn't post
        """
        if agent.is_suspended:
            return None
        
        # Post frequency based on agent type
        post_probability = {
            AgentType.NORMAL: 0.3,
            AgentType.TROLL: 0.6,
            AgentType.INFLUENCER: 0.5,
            AgentType.BOT: 0.8
        }.get(agent.agent_type, 0.3)
        
        if random.random() > post_probability:
            return None
        
        # Generate content characteristics based on agent type
        toxicity = AgentBehavior._calculate_toxicity(agent)
        misinformation = AgentBehavior._calculate_misinformation(agent)
        emotional_intensity = AgentBehavior._calculate_emotion(agent)
        
        # Content generation based on type
        content = AgentBehavior._generate_content(agent, toxicity, misinformation)
        
        return Post(
            id=f"post_{uuid.uuid4().hex[:8]}",
            author_id=agent.id,
            content=content,
            toxicity=toxicity,
            misinformation=misinformation,
            emotional_intensity=emotional_intensity,
            views=0,
            likes=0,
            shares=0,
            comments=0,
            visibility=1.0
        )
    
    @staticmethod
    def _calculate_toxicity(agent: Agent) -> float:
        """Calculate post toxicity based on agent."""
        base_toxicity = agent.toxicity
        
        if agent.agent_type == AgentType.TROLL:
            base_toxicity = min(1.0, base_toxicity * 1.5 + 0.2)
        elif agent.agent_type == AgentType.BOT:
            base_toxicity = min(1.0, base_toxicity * 1.2)
        
        # Add some randomness
        noise = random.gauss(0, 0.1)
        return max(0.0, min(1.0, base_toxicity + noise))
    
    @staticmethod
    def _calculate_misinformation(agent: Agent) -> float:
        """Calculate misinformation level based on agent."""
        # Inversely related to credibility
        base_misinfo = 1.0 - agent.credibility
        
        if agent.agent_type == AgentType.BOT:
            base_misinfo = min(1.0, base_misinfo * 1.5 + 0.3)
        elif agent.agent_type == AgentType.TROLL:
            base_misinfo = min(1.0, base_misinfo * 1.3)
        
        noise = random.gauss(0, 0.1)
        return max(0.0, min(1.0, base_misinfo + noise))
    
    @staticmethod
    def _calculate_emotion(agent: Agent) -> float:
        """Calculate emotional intensity."""
        base_emotion = 0.5
        
        if agent.agent_type == AgentType.TROLL:
            base_emotion = 0.8
        elif agent.agent_type == AgentType.INFLUENCER:
            base_emotion = 0.7
        
        noise = random.gauss(0, 0.15)
        return max(0.0, min(1.0, base_emotion + noise))
    
    @staticmethod
    def _generate_content(agent: Agent, toxicity: float, misinfo: float) -> str:
        """Generate post content based on agent characteristics."""
        templates = {
            AgentType.NORMAL: [
                "Just sharing my thoughts on this.",
                "Interesting perspective to consider.",
                "What do you all think about this?",
                "I found this quite informative.",
                "Here's my take on the situation."
            ],
            AgentType.TROLL: [
                "This is absolutely ridiculous!",
                "Can't believe people fall for this garbage.",
                "You're all wrong and I can prove it!",
                "This platform is full of idiots.",
                "Wake up sheeple!"
            ],
            AgentType.INFLUENCER: [
                "My followers need to hear this!",
                "This is game-changing information.",
                "I've been saying this for years.",
                "You won't believe what I just discovered!",
                "This affects all of us."
            ],
            AgentType.BOT: [
                "Check out this amazing opportunity!",
                "Breaking news you need to see!",
                "This changes everything!",
                "Don't miss out on this!",
                "Exclusive information inside!"
            ]
        }
        
        type_templates = templates.get(agent.agent_type, templates[AgentType.NORMAL])
        content = random.choice(type_templates)
        
        # Add toxicity markers
        if toxicity > 0.7:
            content += " " + random.choice(["😡", "🤬", "💩", "🖕", "Trash!"])
        
        # Add misinformation markers
        if misinfo > 0.7:
            content = "BREAKING: " + content + " Share before it's deleted!"
        
        return content
    
    @staticmethod
    def update_agent_state(
        agent: Agent,
        moderation_action: Optional[str] = None,
        post_engagement: float = 0.0
    ) -> Agent:
        """
        Update agent state based on moderation and engagement.
        
        Args:
            agent: Agent to update
            moderation_action: Type of moderation action taken
            post_engagement: Engagement received on posts
            
        Returns:
            Updated agent
        """
        # Trust dynamics
        if moderation_action == "warn_user":
            agent.trust_in_platform = max(0.0, agent.trust_in_platform - 0.15)
        elif moderation_action == "suspend_user":
            agent.trust_in_platform = max(0.0, agent.trust_in_platform - 0.4)
        elif moderation_action == "post_removed":
            agent.trust_in_platform = max(0.0, agent.trust_in_platform - 0.1)
        elif post_engagement > 10:
            # Positive engagement increases trust slightly
            agent.trust_in_platform = min(1.0, agent.trust_in_platform + 0.02)
        
        # Belief drift (agents slowly move toward network average)
        # This is handled at network level
        
        return agent


class AgentFactory:
    """Factory for creating agents with different configurations."""
    
    @staticmethod
    def create_agent(agent_type: AgentType, agent_id: str, belief_bias: float = 0.0) -> Agent:
        """
        Create an agent of specified type.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Unique identifier
            belief_bias: Bias toward positive or negative belief
            
        Returns:
            Configured Agent
        """
        configs = {
            AgentType.NORMAL: {
                "toxicity": random.uniform(0.0, 0.3),
                "credibility": random.uniform(0.5, 0.8),
                "influence": random.uniform(0.5, 2.0),
                "trust_in_platform": random.uniform(0.4, 0.7)
            },
            AgentType.TROLL: {
                "toxicity": random.uniform(0.5, 0.9),
                "credibility": random.uniform(0.1, 0.4),
                "influence": random.uniform(0.3, 1.5),
                "trust_in_platform": random.uniform(0.1, 0.3)
            },
            AgentType.INFLUENCER: {
                "toxicity": random.uniform(0.0, 0.2),
                "credibility": random.uniform(0.6, 0.9),
                "influence": random.uniform(3.0, 8.0),
                "trust_in_platform": random.uniform(0.5, 0.8)
            },
            AgentType.BOT: {
                "toxicity": random.uniform(0.2, 0.6),
                "credibility": random.uniform(0.0, 0.3),
                "influence": random.uniform(0.5, 2.0),
                "trust_in_platform": random.uniform(0.0, 0.2)
            }
        }
        
        config = configs.get(agent_type, configs[AgentType.NORMAL])
        
        # Generate name based on type
        names = {
            AgentType.NORMAL: [f"User{i}" for i in range(1000)],
            AgentType.TROLL: [f"Anon{i}" for i in range(1000)],
            AgentType.INFLUENCER: [f"Influencer{i}" for i in range(100)],
            AgentType.BOT: [f"Bot{i}" for i in range(1000)]
        }
        
        name = random.choice(names.get(agent_type, names[AgentType.NORMAL]))
        
        # Belief with bias
        belief = random.uniform(-0.5, 0.5) + belief_bias
        belief = max(-1.0, min(1.0, belief))
        
        return Agent(
            id=agent_id,
            name=name,
            agent_type=agent_type,
            belief=belief,
            toxicity=config["toxicity"],
            credibility=config["credibility"],
            influence=config["influence"],
            trust_in_platform=config["trust_in_platform"]
        )
    
    @staticmethod
    def create_network(
        num_agents: int,
        troll_ratio: float = 0.1,
        bot_ratio: float = 0.1,
        influencer_ratio: float = 0.1,
        seed: Optional[int] = None
    ) -> List[Agent]:
        """
        Create a network of agents with specified composition.
        
        Args:
            num_agents: Total number of agents
            troll_ratio: Fraction of trolls
            bot_ratio: Fraction of bots
            influencer_ratio: Fraction of influencers
            seed: Random seed
            
        Returns:
            List of agents
        """
        if seed is not None:
            random.seed(seed)
        
        agents = []
        
        # Calculate counts
        num_trolls = int(num_agents * troll_ratio)
        num_bots = int(num_agents * bot_ratio)
        num_influencers = int(num_agents * influencer_ratio)
        num_normal = num_agents - num_trolls - num_bots - num_influencers
        
        # Create agents
        for i in range(num_normal):
            agents.append(AgentFactory.create_agent(
                AgentType.NORMAL, f"agent_{i:03d}"
            ))
        
        for i in range(num_trolls):
            agents.append(AgentFactory.create_agent(
                AgentType.TROLL, f"troll_{i:03d}", belief_bias=random.choice([-0.5, 0.5])
            ))
        
        for i in range(num_bots):
            agents.append(AgentFactory.create_agent(
                AgentType.BOT, f"bot_{i:03d}"
            ))
        
        for i in range(num_influencers):
            agents.append(AgentFactory.create_agent(
                AgentType.INFLUENCER, f"influencer_{i:03d}"
            ))
        
        random.shuffle(agents)
        return agents
