"""Social dynamics engine for post spread, virality, and belief updates."""

import random
from typing import List, Tuple

from src.models import Agent, AgentType, Post


class ViralityModel:
    """Models how posts spread through the network."""
    
    @staticmethod
    def calculate_spread_probability(
        post: Post,
        viewer: Agent,
        network_avg_belief: float
    ) -> float:
        """
        Calculate probability of a viewer engaging with a post.
        
        Args:
            post: The post being viewed
            viewer: The agent viewing the post
            network_avg_belief: Average belief in network
            
        Returns:
            Probability of engagement (0-1)
        """
        if post.is_removed or post.visibility <= 0:
            return 0.0
        
        # Base probability
        base_prob = 0.1
        
        # Emotional intensity increases engagement
        emotion_factor = post.emotional_intensity * 0.3
        
        # Belief alignment (confirmation bias)
        author_belief = 0.0  # Simplified - would need author reference
        belief_diff = abs(viewer.belief - author_belief)
        belief_factor = (1 - belief_diff) * 0.2
        
        # Influencer effect (if we knew the author's influence)
        # Simplified: higher visibility = more influential
        influence_factor = post.visibility * 0.15
        
        # Toxicity can increase or decrease engagement
        if viewer.toxicity > 0.5:
            toxicity_factor = post.toxicity * 0.15
        else:
            toxicity_factor = -post.toxicity * 0.1
        
        # Misinformation spreads faster with some viewers
        if viewer.credibility < 0.3:
            misinfo_factor = post.misinformation * 0.1
        else:
            misinfo_factor = -post.misinformation * 0.05
        
        total_prob = base_prob + emotion_factor + belief_factor + influence_factor + toxicity_factor + misinfo_factor
        return max(0.0, min(1.0, total_prob))
    
    @staticmethod
    def simulate_engagement(
        post: Post,
        viewers: List[Agent],
        network_avg_belief: float
    ) -> Post:
        """
        Simulate engagement on a post from viewers.
        
        Args:
            post: The post
            viewers: List of potential viewers
            network_avg_belief: Network belief average
            
        Returns:
            Updated post with engagement metrics
        """
        views = 0
        likes = 0
        shares = 0
        comments = 0
        
        for viewer in viewers:
            if viewer.is_suspended:
                continue
            
            prob = ViralityModel.calculate_spread_probability(
                post, viewer, network_avg_belief
            )
            
            if random.random() < prob:
                views += 1
                
                # Like probability
                if random.random() < 0.4:
                    likes += 1
                
                # Share probability (lower)
                if random.random() < 0.15:
                    shares += 1
                
                # Comment probability (lowest)
                if random.random() < 0.08:
                    comments += 1
        
        post.views += views
        post.likes += likes
        post.shares += shares
        post.comments += comments
        
        return post


class BeliefDynamics:
    """Models how agent beliefs change over time."""
    
    @staticmethod
    def update_belief_from_content(
        agent: Agent,
        posts: List[Post],
        learning_rate: float = 0.05
    ) -> float:
        """
        Update agent belief based on content consumed.
        
        Args:
            agent: The agent
            posts: Posts the agent viewed
            learning_rate: How much beliefs shift
            
        Returns:
            New belief value
        """
        if not posts or agent.credibility > 0.8:
            # High credibility agents are resistant to belief change
            return agent.belief
        
        # Calculate weighted average of post influences
        total_influence = 0.0
        weighted_belief_sum = 0.0
        
        for post in posts:
            if post.is_removed:
                continue
            
            # Post influence based on engagement
            influence = (post.likes + post.shares * 2 + post.comments) * post.visibility
            
            # Misinformation has different effect based on agent credibility
            if post.misinformation > 0.5 and agent.credibility < 0.5:
                # Low credibility agents more susceptible to misinformation
                influence *= 1.5
            
            # Toxic content can polarize
            if post.toxicity > 0.7:
                # Push belief further in its direction
                post_belief = 1.0 if random.random() > 0.5 else -1.0
            else:
                post_belief = 0.0
            
            weighted_belief_sum += post_belief * influence
            total_influence += influence
        
        if total_influence == 0:
            return agent.belief
        
        # New belief with learning
        content_belief = weighted_belief_sum / total_influence
        new_belief = agent.belief + (content_belief - agent.belief) * learning_rate
        
        return max(-1.0, min(1.0, new_belief))
    
    @staticmethod
    def calculate_polarization(agents: List[Agent]) -> float:
        """
        Calculate belief polarization in the network.
        
        Args:
            agents: List of all agents
            
        Returns:
            Polarization score (0=unified, 1=highly polarized)
        """
        if not agents:
            return 0.0
        
        beliefs = [a.belief for a in agents if not a.is_suspended]
        
        if len(beliefs) < 2:
            return 0.0
        
        # Check for bimodal distribution
        negative_beliefs = [b for b in beliefs if b < -0.2]
        positive_beliefs = [b for b in beliefs if b > 0.2]
        neutral_beliefs = [b for b in beliefs if -0.2 <= b <= 0.2]
        
        # Polarization is high when we have significant groups at extremes
        total = len(beliefs)
        neg_ratio = len(negative_beliefs) / total
        pos_ratio = len(positive_beliefs) / total
        neutral_ratio = len(neutral_beliefs) / total
        
        # High polarization = significant groups at both extremes
        polarization = (neg_ratio * pos_ratio * 4) + (1 - neutral_ratio) * 0.2
        
        return min(1.0, polarization)


class NetworkMetrics:
    """Calculate network-wide metrics."""
    
    @staticmethod
    def calculate_network_toxicity(agents: List[Agent], posts: List[Post]) -> float:
        """Calculate overall network toxicity."""
        if not posts:
            return 0.0
        
        active_posts = [p for p in posts if not p.is_removed]
        if not active_posts:
            return 0.0
        
        # Weight by engagement
        total_engagement = sum(p.likes + p.shares + p.comments for p in active_posts)
        if total_engagement == 0:
            return sum(p.toxicity for p in active_posts) / len(active_posts)
        
        weighted_toxicity = sum(
            p.toxicity * (p.likes + p.shares + p.comments + 1)
            for p in active_posts
        ) / sum(p.likes + p.shares + p.comments + 1 for p in active_posts)
        
        return weighted_toxicity
    
    @staticmethod
    def calculate_misinformation_index(agents: List[Agent], posts: List[Post]) -> float:
        """Calculate misinformation spread level."""
        if not posts:
            return 0.0
        
        active_posts = [p for p in posts if not p.is_removed]
        if not active_posts:
            return 0.0
        
        # Weight by views
        total_views = sum(p.views for p in active_posts)
        if total_views == 0:
            return sum(p.misinformation for p in active_posts) / len(active_posts)
        
        weighted_misinfo = sum(
            p.misinformation * p.views for p in active_posts
        ) / total_views
        
        return weighted_misinfo
    
    @staticmethod
    def calculate_engagement_score(posts: List[Post]) -> float:
        """Calculate platform engagement score."""
        if not posts:
            return 0.0
        
        total_interactions = sum(
            p.likes + p.shares * 2 + p.comments
            for p in posts if not p.is_removed
        )
        
        # Normalize by number of posts
        return total_interactions / max(len(posts), 1)
    
    @staticmethod
    def calculate_avg_trust(agents: List[Agent]) -> float:
        """Calculate average agent trust in platform."""
        if not agents:
            return 0.5
        
        active_agents = [a for a in agents if not a.is_suspended]
        if not active_agents:
            return 0.0
        
        return sum(a.trust_in_platform for a in active_agents) / len(active_agents)


class OutbreakSimulator:
    """Simulate viral misinformation outbreaks."""
    
    @staticmethod
    def trigger_outbreak(
        agents: List[Agent],
        posts: List[Post],
        intensity: float = 0.8
    ) -> List[Post]:
        """
        Trigger a viral misinformation outbreak.
        
        Args:
            agents: All agents
            posts: Existing posts
            intensity: Outbreak intensity (0-1)
            
        Returns:
            New viral posts
        """
        viral_posts = []
        
        # Select bot and troll agents to spread
        spreaders = [a for a in agents if a.agent_type in (AgentType.BOT, AgentType.TROLL) and not a.is_suspended]
        
        if not spreaders:
            return viral_posts
        
        # Create viral posts
        num_viral = int(len(spreaders) * intensity * 0.5)
        
        for i in range(num_viral):
            spreader = random.choice(spreaders)
            
            post = Post(
                id=f"viral_{i:03d}",
                author_id=spreader.id,
                content="BREAKING: Major revelation that changes everything! Share immediately!",
                toxicity=random.uniform(0.3, 0.7),
                misinformation=intensity,
                emotional_intensity=0.9,
                views=int(1000 * intensity),
                likes=int(500 * intensity),
                shares=int(300 * intensity),
                comments=int(200 * intensity),
                visibility=1.0
            )
            
            viral_posts.append(post)
        
        return viral_posts
