"""
Baseline inference script for Hiro Social Governance Environment.

Runs a baseline LLM agent against all three tasks IN PARALLEL and reports scores.
Uses NVIDIA API with async OpenAI-compatible client for non-blocking I/O.

Required environment variables:
- HF_TOKEN: Your API key
- API_BASE_URL: https://integrate.api.nvidia.com/v1
- MODEL_NAME: minimaxai/minimax-m2.5

Usage:
    python inference.py                  # Run all tasks in parallel
    python inference.py --task easy      # Run a single task
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from openai import AsyncOpenAI

# Ensure the root directory is in sys.path so 'src' can be imported
# regardless of where the script is executed from
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.environment import HiroSocialEnv  # type: ignore
from src.models import Action, ActionType, Observation  # type: ignore

# API configurations strictly following hackathon rules
API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "minimaxai/minimax-m2.5")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Performance tuning constants ---
LLM_TIMEOUT = 10.0        # seconds per LLM call
MAX_RETRIES = 1            # single retry on failure
RETRY_DELAY = 1.0          # fixed 1s retry delay
TASK_TIME_LIMIT = 1500.0   # 25 min cap per task (tasks run in parallel, so total ~11 min)
MAX_TOKENS = 150           # keep responses short and fast


# Structured Logging Functions
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


# ── Fast rule-based fallback (zero latency) ────────────────────────
def rule_based_action(observation: Observation) -> Action:
    """
    Instant heuristic action when LLM is too slow or unavailable.
    Scans recent posts and acts on the worst offender.
    """
    # Check flagged posts first (highest priority)
    for post in observation.flagged_posts[:3]:
        if post.toxicity > 0.7:
            return Action(action_type=ActionType.REMOVE_POST, target_id=post.id,
                          reason="High toxicity flagged post")
        if post.misinformation > 0.6:
            return Action(action_type=ActionType.DOWNRANK_POST, target_id=post.id,
                          reason="Misinformation flagged post")

    # Check recent posts
    for post in observation.recent_posts[-5:]:
        if post.toxicity > 0.7:
            return Action(action_type=ActionType.REMOVE_POST, target_id=post.id,
                          reason="High toxicity")
        if post.misinformation > 0.5:
            return Action(action_type=ActionType.DOWNRANK_POST, target_id=post.id,
                          reason="Misinformation")

    # Inject counter-info when misinformation is elevated
    if observation.misinformation_index > 0.4:
        return Action(action_type=ActionType.INJECT_COUNTER_INFO,
                      content="Verified information from official sources.")

    return Action(action_type=ActionType.NO_ACTION)


class GovernanceAgent:
    """Async LLM agent for social media governance with fast fallback."""

    SYSTEM_PROMPT = (
        "You are a social media governance AI. Respond with ONLY a JSON object.\n"
        "Actions: remove_post, flag_post, downrank_post, boost_post, "
        "warn_user, suspend_user, inject_counter_info, no_action.\n"
        "Rules: remove if toxicity>0.7, downrank if misinfo>0.5, "
        "inject_counter_info if misinfo_index>0.4, prefer downrank over remove.\n"
        "Format: {\"action_type\":\"...\",\"target_id\":\"...\",\"reason\":\"...\"}"
    )

    def __init__(self):
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable not set!")
        self.client = AsyncOpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
            timeout=LLM_TIMEOUT,
        )
        self.model = MODEL_NAME

    async def act(self, observation: Observation, task_id: str) -> Action:
        """Generate governance action; falls back to rules if LLM is slow."""
        prompt = self._build_prompt(observation, task_id)

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                        max_tokens=MAX_TOKENS,
                    ),
                    timeout=LLM_TIMEOUT,
                )
                content = response.choices[0].message.content
                return self._parse_response(content)

            except Exception as e:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"  [Fallback] LLM failed ({e}), using rules", flush=True)
                    return rule_based_action(observation)

        return rule_based_action(observation)

    # ── Prompt building (compact to save tokens) ───────────────────
    def _build_prompt(self, obs: Observation, task_id: str) -> str:
        posts = "\n".join(
            f"- {p.id}: tox={p.toxicity:.2f} mis={p.misinformation:.2f} v={p.views}"
            for p in obs.recent_posts[-5:]
        ) or "None"
        flagged = "\n".join(
            f"- {p.id}: tox={p.toxicity:.2f} mis={p.misinformation:.2f}"
            for p in obs.flagged_posts[-3:]
        ) or "None"

        return (
            f"Step {obs.step}/{obs.max_steps} Task:{task_id}\n"
            f"Toxicity:{obs.network_toxicity:.3f} Misinfo:{obs.misinformation_index:.3f} "
            f"Engage:{obs.engagement_score:.1f} Trust:{obs.avg_trust_score:.3f}\n"
            f"Agents:{obs.num_agents} Active:{obs.active_agents} "
            f"Removed:{obs.posts_removed} Warned:{obs.users_warned}\n"
            f"Posts:\n{posts}\nFlagged:\n{flagged}\nAction?"
        )

    # ── Response parsing ───────────────────────────────────────────
    def _parse_response(self, content: str) -> Action:
        try:
            json_str = self._extract_json(content)
            if not json_str:
                return Action(action_type=ActionType.NO_ACTION)
            data = json.loads(json_str)
            mapping = {
                "remove_post": ActionType.REMOVE_POST,
                "flag_post": ActionType.FLAG_POST,
                "downrank_post": ActionType.DOWNRANK_POST,
                "boost_post": ActionType.BOOST_POST,
                "warn_user": ActionType.WARN_USER,
                "suspend_user": ActionType.SUSPEND_USER,
                "inject_counter_info": ActionType.INJECT_COUNTER_INFO,
                "no_action": ActionType.NO_ACTION,
            }
            return Action(
                action_type=mapping.get(data.get("action_type", "").lower(), ActionType.NO_ACTION),
                target_id=data.get("target_id"),
                reason=data.get("reason"),
                content=data.get("content"),
            )
        except Exception:
            return Action(action_type=ActionType.NO_ACTION)

    @staticmethod
    def _extract_json(content: str) -> str:
        if "```json" in content:
            s = content.find("```json") + 7
            e = content.find("```", s)
            return content[s:e].strip()
        if "```" in content:
            s = content.find("```") + 3
            e = content.find("```", s)
            return content[s:e].strip()
        s = content.find("{")
        e = content.rfind("}") + 1
        return content[s:e] if s >= 0 and e > s else ""


# ── Task runner (each task gets its own env + agent) ───────────────
async def run_task(task_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Run a single task with a hard time cap."""
    env = HiroSocialEnv()
    agent = GovernanceAgent()
    result = {"task": task_id, "elapsed_time": 0.0, "final_grade": 0.0}
    rewards: List[float] = []
    score = 0.0
    step_count = 0
    start = time.time()

    log_start(task=task_id, env="hiro-social-governance", model=MODEL_NAME)

    try:
        obs = env.reset(task=task_id, seed=seed)
        done = False

        while not done:
            # Hard time cap — bail out gracefully
            if time.time() - start > TASK_TIME_LIMIT:
                print(f"  [TimeLimit] {task_id} hit {TASK_TIME_LIMIT}s cap", flush=True)
                break

            step_count += 1
            try:
                action = await agent.act(obs, task_id)
                action_repr = action.action_type.value if action else "no_action"
                obs, reward, done, info = env.step(action)
                rewards.append(reward.total)
                log_step(step=step_count, action=action_repr, reward=reward.total, done=done)

                if done:
                    score = env.get_task_grade()
            except Exception as e:
                rewards.append(0.0)
                log_step(step=step_count, action="ERROR", reward=0.0, done=True, error=str(e))
                result["error"] = str(e)
                break
    except Exception as e:
        result["error"] = str(e)
        log_step(step=1, action="ERROR", reward=0.0, done=True, error=str(e))

    # Always retrieve grade — even on early exit or error
    try:
        score = env.get_task_grade()
    except Exception:
        pass  # score stays at its last value (or 0.0)

    # Nuclear clamp — strictly (0, 1) with NaN/inf protection
    import math
    if not math.isfinite(score):
        score = 0.5
    EPSILON = 0.01
    score = max(EPSILON, min(1.0 - EPSILON, score))

    elapsed = time.time() - start
    result["elapsed_time"] = elapsed
    result["final_grade"] = score
    result["avg_reward"] = sum(rewards) / len(rewards) if rewards else 0.0
    log_end(success=score >= 0.5, steps=step_count, score=score, rewards=rewards)
    return result


# ── Main (async, parallel) ────────────────────────────────────────
async def amain():
    parser = argparse.ArgumentParser(description="Hiro Social Governance - Baseline Inference")
    parser.add_argument("--task", type=str, choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Hiro Social Governance - Baseline Inference (ASYNC)")
    print("=" * 60)
    print(f"  API_BASE_URL: {API_BASE_URL}")
    print(f"  MODEL_NAME:   {MODEL_NAME}")
    print(f"  HF_TOKEN:     {'*' * 20}...{HF_TOKEN[-4:] if HF_TOKEN else 'NOT SET'}")

    if not HF_TOKEN:
        print("\n[ERROR] HF_TOKEN not set!")
        return 1

    tasks = [args.task] if args.task else ["easy", "medium", "hard"]
    total_start = time.time()

    if len(tasks) > 1:
        # Run ALL tasks in parallel
        print(f"\n>> Launching {len(tasks)} tasks in PARALLEL...\n")
        all_results = await asyncio.gather(
            *(run_task(t, seed=args.seed) for t in tasks),
            return_exceptions=True,
        )
        # Convert exceptions to error dicts
        cleaned = []
        for i, r in enumerate(all_results):
            if isinstance(r, Exception):
                cleaned.append({"task": tasks[i], "error": str(r)})
            else:
                cleaned.append(r)
        all_results = cleaned
    else:
        all_results = [await run_task(tasks[0], seed=args.seed)]

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in all_results:
        t = r["task"]
        if "error" in r:
            print(f"  {t:8s}: ERROR - {r['error']}")
        else:
            print(f"  {t:8s}: grade={r['final_grade']:.3f}  avg_reward={r.get('avg_reward',0):+.3f}  time={r['elapsed_time']:.1f}s")

    print(f"\nTotal wall-clock time: {total_elapsed:.1f}s")

    # Save results
    output = {
        "config": {"api_base_url": API_BASE_URL, "model_name": MODEL_NAME},
        "results": all_results,
        "total_time": round(total_elapsed, 2),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Results saved to baseline_results.json")
    return 0


def main():
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
