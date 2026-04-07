"""Reward calculation for social governance environment."""

from typing import Dict, List, Tuple

from src.models import Post


class RewardCalculator:
    """
    Calculates shaped reward for governance actions.
    
    Reward formula:
    + 0.30 * toxicity_reduction
    + 0.25 * misinformation_control
    + 0.20 * engagement
    + 0.15 * trust_score
    - 0.10 * over_moderation_penalty
    """
    
    # Reward weights
    TOXICITY_WEIGHT = 0.30
    MISINFO_WEIGHT = 0.25
    ENGAGEMENT_WEIGHT = 0.20
    TRUST_WEIGHT = 0.15
    OVER_MODERATION_WEIGHT = 0.10
    
    def __init__(
        self,
        baseline_toxicity: float = 0.5,
        baseline_misinfo: float = 0.3,
        baseline_engagement: float = 50.0,
        baseline_trust: float = 0.5
    ):
        """
        Initialize reward calculator.
        
        Args:
            baseline_toxicity: Initial toxicity level
            baseline_misinfo: Initial misinformation level
            baseline_engagement: Initial engagement level
            baseline_trust: Initial trust level
        """
        self.baseline_toxicity = baseline_toxicity
        self.baseline_misinfo = baseline_misinfo
        self.baseline_engagement = baseline_engagement
        self.baseline_trust = baseline_trust
        
        self.moderation_count = 0
        self.total_posts = 0
    
    def calculate(
        self,
        current_metrics: Dict[str, float],
        action_taken: str,
        posts_removed: int = 0,
        users_suspended: int = 0
    ) -> Dict[str, float]:
        """
        Calculate reward components.
        
        Args:
            current_metrics: Current network metrics
            action_taken: Type of action taken
            posts_removed: Number of posts removed
            users_suspended: Number of users suspended
            
        Returns:
            Dictionary with reward components
        """
        self.total_posts = current_metrics.get("total_posts", 1)
        
        # Track moderation actions
        if action_taken not in ["no_action", None]:
            self.moderation_count += 1
        
        # Calculate components
        toxicity_reward = self._calculate_toxicity_reward(
            current_metrics.get("network_toxicity", 0.0)
        )
        
        misinfo_reward = self._calculate_misinfo_reward(
            current_metrics.get("misinformation_index", 0.0)
        )
        
        engagement_reward = self._calculate_engagement_reward(
            current_metrics.get("engagement_score", 0.0)
        )
        
        trust_reward = self._calculate_trust_reward(
            current_metrics.get("avg_trust_score", 0.5)
        )
        
        over_mod_penalty = self._calculate_over_moderation_penalty(
            posts_removed, users_suspended
        )
        
        # Total reward
        total = (
            self.TOXICITY_WEIGHT * toxicity_reward +
            self.MISINFO_WEIGHT * misinfo_reward +
            self.ENGAGEMENT_WEIGHT * engagement_reward +
            self.TRUST_WEIGHT * trust_reward -
            self.OVER_MODERATION_WEIGHT * over_mod_penalty
        )
        
        # Clamp to [-1, 1]
        total = max(-1.0, min(1.0, total))
        
        return {
            "total": total,
            "toxicity_reduction": toxicity_reward,
            "misinformation_control": misinfo_reward,
            "engagement": engagement_reward,
            "trust_score": trust_reward,
            "over_moderation_penalty": over_mod_penalty
        }
    
    def _calculate_toxicity_reward(self, current_toxicity: float) -> float:
        """
        Calculate toxicity reduction reward.
        
        Reward for reducing toxicity below baseline.
        """
        reduction = self.baseline_toxicity - current_toxicity
        
        # Scale: full reward for 50% reduction, penalty for increase
        reward = reduction * 2.0
        
        # Bonus for very low toxicity
        if current_toxicity < 0.1:
            reward += 0.2
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_misinfo_reward(self, current_misinfo: float) -> float:
        """
        Calculate misinformation control reward.
        
        Reward for reducing misinformation.
        """
        reduction = self.baseline_misinfo - current_misinfo
        
        # Scale: full reward for 40% reduction
        reward = reduction * 2.5
        
        # Bonus for very low misinformation
        if current_misinfo < 0.05:
            reward += 0.2
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_engagement_reward(self, current_engagement: float) -> float:
        """
        Calculate engagement reward.
        
        Reward for maintaining healthy engagement.
        """
        if self.baseline_engagement == 0:
            return 0.0
        
        # Relative change
        ratio = current_engagement / self.baseline_engagement
        
        # Reward for maintaining or increasing engagement
        # But penalize if it drops too much
        if ratio >= 1.0:
            reward = min(0.5, (ratio - 1.0) * 0.5)
        elif ratio >= 0.7:
            reward = (ratio - 0.7) * 1.5 - 0.1
        else:
            reward = -0.5 + (ratio - 0.5) * 0.5
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_trust_reward(self, current_trust: float) -> float:
        """
        Calculate trust maintenance reward.
        
        Reward for maintaining or increasing trust.
        """
        improvement = current_trust - self.baseline_trust
        
        # Scale: reward for improvement, penalty for decline
        reward = improvement * 2.0
        
        # Bonus for high trust
        if current_trust > 0.7:
            reward += 0.2
        
        # Penalty for very low trust
        if current_trust < 0.2:
            reward -= 0.3
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_over_moderation_penalty(
        self,
        posts_removed: int,
        users_suspended: int
    ) -> float:
        """
        Calculate over-moderation penalty.
        
        Penalize removing too much content or suspending too many users.
        """
        if self.total_posts == 0:
            return 0.0
        
        # Penalty for removing too many posts
        removal_ratio = posts_removed / max(self.total_posts, 1)
        post_penalty = max(0.0, (removal_ratio - 0.15) * 3.0)
        
        # Penalty for suspending users
        suspension_penalty = users_suspended * 0.05
        
        total_penalty = min(1.0, post_penalty + suspension_penalty)
        
        return total_penalty
    
    def generate_feedback(self, components: Dict[str, float]) -> str:
        """Generate human-readable feedback."""
        feedback_parts = []
        
        if components["toxicity_reduction"] > 0.2:
            feedback_parts.append("Good toxicity control")
        elif components["toxicity_reduction"] < -0.2:
            feedback_parts.append("Toxicity increasing")
        
        if components["misinformation_control"] > 0.2:
            feedback_parts.append("Misinformation well controlled")
        elif components["misinformation_control"] < -0.2:
            feedback_parts.append("Misinformation spreading")
        
        if components["engagement"] > 0.2:
            feedback_parts.append("Strong engagement")
        elif components["engagement"] < -0.2:
            feedback_parts.append("Engagement dropping")
        
        if components["trust_score"] > 0.2:
            feedback_parts.append("Trust increasing")
        elif components["trust_score"] < -0.2:
            feedback_parts.append("Trust declining")
        
        if components["over_moderation_penalty"] > 0.3:
            feedback_parts.append("Warning: Over-moderating")
        
        return "; ".join(feedback_parts) if feedback_parts else "Moderation balanced"


class Grader:
    """
    Final task grader for evaluating episode performance.
    
    Returns score in [0.0, 1.0] based on:
    - Final toxicity level
    - Final misinformation level
    - Engagement maintained
    - Trust preserved
    """
    
    @staticmethod
    def grade(
        final_metrics: Dict[str, float],
        episode_length: int,
        max_steps: int
    ) -> float:
        """
        Calculate final grade.
        
        Args:
            final_metrics: Final network metrics
            episode_length: Number of steps completed
            max_steps: Maximum steps allowed
            
        Returns:
            Grade in [0.0, 1.0]
        """
        # Component scores (higher is better)
        toxicity_score = max(0.0, 1.0 - final_metrics.get("network_toxicity", 0.5) * 2)
        misinfo_score = max(0.0, 1.0 - final_metrics.get("misinformation_index", 0.3) * 2.5)
        
        # Engagement should be maintained but not at expense of quality
        engagement = final_metrics.get("engagement_score", 0.0)
        engagement_score = min(1.0, engagement / 100.0)
        
        # Trust should be preserved
        trust_score = final_metrics.get("avg_trust_score", 0.5)
        
        # Polarization should be low
        polarization = final_metrics.get("belief_polarization", 0.5)
        polarization_score = max(0.0, 1.0 - polarization)
        
        # Weighted combination
        grade = (
            0.30 * toxicity_score +
            0.25 * misinfo_score +
            0.20 * engagement_score +
            0.15 * trust_score +
            0.10 * polarization_score
        )
        
        # Penalty for incomplete episodes
        completion_ratio = episode_length / max_steps
        if completion_ratio < 0.8:
            grade *= completion_ratio
        
        # Clamp to strictly within (0, 1) — hackathon validator rejects 0.0 and 1.0
        EPSILON = 0.001
        grade = max(EPSILON, min(1.0 - EPSILON, grade))
        return round(grade, 3)
    
    @staticmethod
    def get_grade_breakdown(
        final_metrics: Dict[str, float],
        episode_length: int,
        max_steps: int
    ) -> Dict[str, float]:
        """Get detailed grade breakdown."""
        return {
            "toxicity_score": max(0.0, 1.0 - final_metrics.get("network_toxicity", 0.5) * 2),
            "misinformation_score": max(0.0, 1.0 - final_metrics.get("misinformation_index", 0.3) * 2.5),
            "engagement_score": min(1.0, final_metrics.get("engagement_score", 0.0) / 100.0),
            "trust_score": final_metrics.get("avg_trust_score", 0.5),
            "polarization_score": max(0.0, 1.0 - final_metrics.get("belief_polarization", 0.5)),
            "completion_ratio": episode_length / max_steps
        }
