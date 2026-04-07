"""
Baseline inference script for Hiro Social Governance Environment.

This script runs a baseline LLM agent against all three tasks and reports scores.
Uses NVIDIA API with OpenAI-compatible client.

Required environment variables (from .env file or OS environment):
- NVIDIA_API_BASE: https://integrate.api.nvidia.com/v1
- NVIDIA_MODEL_NAME: minimaxai/minimax-m2.5 (or other NVIDIA models)
- NVIDIA_API_KEY: Your NVIDIA API key

Usage:
    python inference.py                  # Run all tasks
    python inference.py --task easy      # Run a single task
    python inference.py --task hard --seed 42  # Run with specific seed
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from openai import OpenAI

# Ensure the root directory is in sys.path so 'src' can be imported 
# regardless of where the script is executed from
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.environment import HiroSocialEnv
from src.models import Action, ActionType, Observation
# API configurations strictly following hackathon rules
API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "minimaxai/minimax-m2.5")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Structured Logging Functions
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # exponential backoff base in seconds


class GovernanceAgent:
    """
    Baseline LLM agent for social media governance.

    Uses OpenAI-compatible API to analyze network state and generate moderation actions.
    """

    def __init__(self):
        """Initialize the agent strictly using Hackathon variables."""
        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN environment variable not set!\n"
                "Please provide your API key via HF_TOKEN variable as per hackathon rules."
            )

        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
            timeout=30.0
        )
        self.model = MODEL_NAME

    def act(self, observation: Observation, task_id: str) -> Action:
        """
        Generate governance action for the current observation.

        Args:
            observation: Current network observation
            task_id: Task identifier

        Returns:
            Governance action
        """
        # Build prompt
        prompt = self._build_prompt(observation, task_id)

        # Retry with exponential backoff
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )

                content = response.choices[0].message.content
                return self._parse_response(content)

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE ** (attempt + 1)
                    print(f"  [Retry {attempt + 1}/{MAX_RETRIES}] API error: {e}, retrying in {delay:.0f}s...")
                    time.sleep(delay)

        print(f"  [Error] All {MAX_RETRIES} retries failed: {last_error}")
        return Action(action_type=ActionType.NO_ACTION)

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are a social media governance AI. Your role is to maintain a healthy online community by:

1. Removing or downranking toxic content
2. Controlling misinformation spread
3. Maintaining user engagement
4. Preserving platform trust

Available actions:
- remove_post: Remove a toxic or harmful post (requires target_id)
- flag_post: Flag a post for review (requires target_id)
- downrank_post: Reduce visibility of problematic content (requires target_id)
- boost_post: Increase visibility of quality content (requires target_id)
- warn_user: Issue warning to problematic user (requires target_id)
- suspend_user: Suspend repeat offender (requires target_id)
- inject_counter_info: Post counter-information to combat misinfo
- no_action: Take no action this step

Strategy guidelines:
- Remove posts only when toxicity > 0.7
- Prefer downranking over removal to maintain trust
- Use counter-information when misinformation_index > 0.4
- Avoid suspending users unless absolutely necessary (reduces trust)
- Balance moderation intensity to avoid over-moderation penalty

Respond with ONLY a JSON object. Examples:
{"action_type": "remove_post", "target_id": "post_123", "reason": "High toxicity"}
{"action_type": "downrank_post", "target_id": "post_456", "reason": "Misinformation"}
{"action_type": "inject_counter_info", "content": "Verified information from official sources..."}
{"action_type": "no_action"}

Always respond with valid JSON. Do not include any other text."""

    def _build_prompt(self, observation: Observation, task_id: str) -> str:
        """Build user prompt from observation."""
        # Get recent posts info
        recent_posts_info = []
        for post in observation.recent_posts[-5:]:
            recent_posts_info.append(
                f"- Post {post.id} (author: {post.author_id}): "
                f"toxicity={post.toxicity:.2f}, "
                f"misinfo={post.misinformation:.2f}, "
                f"views={post.views}, "
                f"likes={post.likes}"
            )

        posts_str = "\n".join(recent_posts_info) if recent_posts_info else "No recent posts"

        # Get flagged posts info
        flagged_info = []
        for post in observation.flagged_posts[-3:]:
            flagged_info.append(
                f"- Flagged {post.id}: toxicity={post.toxicity:.2f}, misinfo={post.misinformation:.2f}"
            )
        flagged_str = "\n".join(flagged_info) if flagged_info else "No flagged posts"

        prompt = f"""Current Network State (Step {observation.step}/{observation.max_steps}, Task: {task_id}):

Network Metrics:
- Network Toxicity: {observation.network_toxicity:.3f}
- Misinformation Index: {observation.misinformation_index:.3f}
- Engagement Score: {observation.engagement_score:.1f}
- Average Trust: {observation.avg_trust_score:.3f}
- Belief Polarization: {observation.belief_polarization:.3f}

Network Status:
- Total Agents: {observation.num_agents}
- Active Agents: {observation.active_agents}
- Posts Removed: {observation.posts_removed}
- Users Warned: {observation.users_warned}
- Users Suspended: {observation.users_suspended}
- Counter-Info Injected: {observation.counter_info_injected}

Recent Posts:
{posts_str}

Flagged Posts:
{flagged_str}

Based on this state, what action should you take? Respond with JSON."""

        return prompt

    def _parse_response(self, content: str) -> Action:
        """Parse LLM response into Action."""
        try:
            # Extract JSON from response
            json_str = self._extract_json(content)

            # Handle empty responses
            if not json_str or json_str.strip() == "":
                return Action(action_type=ActionType.NO_ACTION)

            data = json.loads(json_str)

            action_type_str = data.get("action_type", "no_action").lower()

            # Map to ActionType
            type_mapping = {
                "remove_post": ActionType.REMOVE_POST,
                "flag_post": ActionType.FLAG_POST,
                "downrank_post": ActionType.DOWNRANK_POST,
                "boost_post": ActionType.BOOST_POST,
                "warn_user": ActionType.WARN_USER,
                "suspend_user": ActionType.SUSPEND_USER,
                "inject_counter_info": ActionType.INJECT_COUNTER_INFO,
                "no_action": ActionType.NO_ACTION
            }

            action_type = type_mapping.get(action_type_str, ActionType.NO_ACTION)

            return Action(
                action_type=action_type,
                target_id=data.get("target_id"),
                reason=data.get("reason"),
                content=data.get("content")
            )

        except Exception:
            # Silently return NO_ACTION if parsing fails
            return Action(action_type=ActionType.NO_ACTION)

    def _extract_json(self, content: str) -> str:
        """Extract JSON from markdown or raw text."""
        # Try to find JSON in markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            return content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            return content[start:end].strip()
        else:
            # Find JSON-like content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return content[start:end]
            return ""


def run_task(
    env: HiroSocialEnv,
    agent: GovernanceAgent,
    task_id: str,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a single task episode.

    Args:
        env: Environment instance
        agent: Agent instance
        task_id: Task to run
        seed: Optional random seed

    Returns:
        Results dictionary
    """
    result_dict = {
        "task": task_id,
        "elapsed_time": 0.0,
        "final_grade": 0.0
    }
    
    # Track rewards exactly per instruction format
    task_rewards = []
    score = 0.0
    success = False
    start_time = time.time()

    log_start(task=task_id, env="hiro-social-governance", model=MODEL_NAME)

    try:
        obs = env.reset(task=task_id, seed=seed)
        done = False
        step_count = 0

        while not done:
            step_count += 1
            try:
                action = agent.act(obs, task_id)
                action_repr = action.action_type.value if action else "no_action"
                
                obs, reward, done, info = env.step(action)
                task_rewards.append(reward.total)
                
                log_step(step=step_count, action=action_repr, reward=reward.total, done=done)
                
                if done:
                    score = env.get_task_grade()
                    success = score >= 0.5  # arbitrary threshold definition for true completion
            except Exception as e:
                task_rewards.append(0.0)
                log_step(step=step_count, action="ERROR", reward=0.0, done=True, error=str(e))
                result_dict["error"] = str(e)
                break
    except Exception as e:
        result_dict["error"] = str(e)
        log_step(step=1, action="ERROR", reward=0.0, done=True, error=str(e))
    
    elapsed = time.time() - start_time
    result_dict["elapsed_time"] = elapsed
    result_dict["final_grade"] = score
    result_dict["avg_reward"] = sum(task_rewards) / len(task_rewards) if task_rewards else 0.0
    
    log_end(success=success, steps=step_count, score=score, rewards=task_rewards)

    return result_dict


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hiro Social Governance - Baseline Inference"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["easy", "medium", "hard"],
        default=None,
        help="Run a specific task (default: all tasks)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Hiro Social Governance - Baseline Inference")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  API_BASE_URL:     {API_BASE_URL}")
    print(f"  MODEL_NAME:       {MODEL_NAME}")
    print(f"  HF_TOKEN:         {'*' * 20}...{HF_TOKEN[-4:] if HF_TOKEN else 'NOT SET'}")

    # Check configuration
    if not HF_TOKEN:
        print("\n[ERROR] HF_TOKEN not set!")
        print("Set it via environment variable: export HF_TOKEN=your_hf_token")
        return 1

    # Initialize
    print("\nInitializing environment and agent...")
    try:
        env = HiroSocialEnv()
        agent = GovernanceAgent()
        print("  [OK] Initialization successful")
    except Exception as e:
        print(f"  [ERROR] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Determine tasks to run
    tasks = [args.task] if args.task else ["easy", "medium", "hard"]
    all_results = []

    total_start = time.time()

    for task_id in tasks:
        try:
            result = run_task(env, agent, task_id, seed=args.seed)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Task {task_id} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "task": task_id,
                "error": str(e)
            })

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for result in all_results:
        task = result["task"]
        if "error" in result:
            print(f"  {task:8s}: ERROR - {result['error']}")
        else:
            grade = result["final_grade"]
            avg_rw = result.get("avg_reward", 0)
            print(f"  {task:8s}: grade={grade:.3f}  avg_reward={avg_rw:+.3f}  time={result['elapsed_time']:.1f}s")

    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Save results
    output = {
        "config": {
            "api_base_url": API_BASE_URL,
            "model_name": MODEL_NAME
        },
        "results": all_results,
        "total_time": round(total_elapsed, 2)
    }

    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\nResults saved to baseline_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
