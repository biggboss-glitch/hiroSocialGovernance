"""
OASIS 双平台并行模拟预设脚本
同时运行Twitter和Reddit模拟，读取相同的配置文件

使用方式:
    python run_parallel_simulation.py --config simulation_config.json [--action-log actions.jsonl]
"""

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_logger import ActionLogger

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph,
        generate_reddit_agent_graph
    )
except ImportError as e:
    print(f"错误: 缺少依赖 {e}")
    print("请先安装: pip install oasis-ai camel-ai")
    sys.exit(1)


# Twitter可用动作
TWITTER_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
    ActionType.QUOTE_POST,
]

# Reddit可用动作
REDDIT_ACTIONS = [
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.SEARCH_POSTS,
    ActionType.SEARCH_USER,
    ActionType.TREND,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
    ActionType.FOLLOW,
    ActionType.MUTE,
]


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_model(config: Dict[str, Any]):
    """
    创建LLM模型
    
    OASIS使用camel-ai的ModelFactory，配置方式：
    - 标准OpenAI: 只需设置 OPENAI_API_KEY 环境变量
    - 自定义API: 设置 OPENAI_API_KEY 和 OPENAI_API_BASE_URL 环境变量
    """
    llm_model = config.get("llm_model", "gpt-4o-mini")
    llm_base_url = config.get("llm_base_url", "")
    
    # 如果配置了base_url，设置环境变量
    if llm_base_url:
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url
    
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=llm_model,
    )


def get_active_agents_for_round(
    env,
    config: Dict[str, Any],
    current_hour: int,
    round_num: int
) -> List:
    """根据时间和配置决定本轮激活哪些Agent"""
    time_config = config.get("time_config", {})
    agent_configs = config.get("agent_configs", [])
    
    base_min = time_config.get("agents_per_hour_min", 5)
    base_max = time_config.get("agents_per_hour_max", 20)
    
    peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
    off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
    
    if current_hour in peak_hours:
        multiplier = time_config.get("peak_activity_multiplier", 1.5)
    elif current_hour in off_peak_hours:
        multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
    else:
        multiplier = 1.0
    
    target_count = int(random.uniform(base_min, base_max) * multiplier)
    
    candidates = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        active_hours = cfg.get("active_hours", list(range(8, 23)))
        activity_level = cfg.get("activity_level", 0.5)
        
        if current_hour not in active_hours:
            continue
        
        if random.random() < activity_level:
            candidates.append(agent_id)
    
    selected_ids = random.sample(
        candidates, 
        min(target_count, len(candidates))
    ) if candidates else []
    
    active_agents = []
    for agent_id in selected_ids:
        try:
            agent = env.agent_graph.get_agent(agent_id)
            active_agents.append((agent_id, agent))
        except Exception:
            pass
    
    return active_agents


async def run_twitter_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[ActionLogger] = None
):
    """运行Twitter模拟"""
    print("[Twitter] 初始化...")
    
    model = create_model(config)
    
    # OASIS Twitter使用CSV格式
    profile_path = os.path.join(simulation_dir, "twitter_profiles.csv")
    if not os.path.exists(profile_path):
        print(f"[Twitter] 错误: Profile文件不存在: {profile_path}")
        return
    
    agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=TWITTER_ACTIONS,
    )
    
    # 获取Agent名称映射
    agent_names = {}
    for agent_id, agent in agent_graph.get_agents():
        agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "twitter_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
    )
    
    await env.reset()
    print("[Twitter] 环境已启动")
    
    if action_logger:
        action_logger.log_simulation_start("twitter", config)
    
    total_actions = 0
    
    # 执行初始事件
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = env.agent_graph.get_agent(agent_id)
                initial_actions[agent] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        platform="twitter",
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content[:100] + "..." if len(content) > 100 else content}
                    )
                    total_actions += 1
            except Exception:
                pass
        
        if initial_actions:
            await env.step(initial_actions)
            print(f"[Twitter] 已发布 {len(initial_actions)} 条初始帖子")
    
    # 主模拟循环
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    start_time = datetime.now()
    
    for round_num in range(total_rounds):
        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1
        
        active_agents = get_active_agents_for_round(
            env, config, simulated_hour, round_num
        )
        
        if not active_agents:
            continue
        
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour, "twitter")
        
        actions = {agent: LLMAction() for _, agent in active_agents}
        await env.step(actions)
        
        # 记录动作
        for agent_id, agent in active_agents:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    platform="twitter",
                    agent_id=agent_id,
                    agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                    action_type="LLM_ACTION",
                    action_args={}
                )
                total_actions += 1
        
        if action_logger:
            action_logger.log_round_end(round_num + 1, len(active_agents), "twitter")
        
        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            print(f"[Twitter] Day {simulated_day}, {simulated_hour:02d}:00 "
                  f"- Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")
    
    await env.close()
    
    if action_logger:
        action_logger.log_simulation_end("twitter", total_rounds, total_actions)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[Twitter] 模拟完成! 耗时: {elapsed:.1f}秒, 总动作: {total_actions}")


async def run_reddit_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[ActionLogger] = None
):
    """运行Reddit模拟"""
    print("[Reddit] 初始化...")
    
    model = create_model(config)
    
    profile_path = os.path.join(simulation_dir, "reddit_profiles.json")
    if not os.path.exists(profile_path):
        print(f"[Reddit] 错误: Profile文件不存在: {profile_path}")
        return
    
    agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=REDDIT_ACTIONS,
    )
    
    # 获取Agent名称映射
    agent_names = {}
    for agent_id, agent in agent_graph.get_agents():
        agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "reddit_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )
    
    await env.reset()
    print("[Reddit] 环境已启动")
    
    if action_logger:
        action_logger.log_simulation_start("reddit", config)
    
    total_actions = 0
    
    # 执行初始事件
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = env.agent_graph.get_agent(agent_id)
                if agent in initial_actions:
                    if not isinstance(initial_actions[agent], list):
                        initial_actions[agent] = [initial_actions[agent]]
                    initial_actions[agent].append(ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    ))
                else:
                    initial_actions[agent] = ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        platform="reddit",
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content[:100] + "..." if len(content) > 100 else content}
                    )
                    total_actions += 1
            except Exception:
                pass
        
        if initial_actions:
            await env.step(initial_actions)
            print(f"[Reddit] 已发布 {len(initial_actions)} 条初始帖子")
    
    # 主模拟循环
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    start_time = datetime.now()
    
    for round_num in range(total_rounds):
        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1
        
        active_agents = get_active_agents_for_round(
            env, config, simulated_hour, round_num
        )
        
        if not active_agents:
            continue
        
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour, "reddit")
        
        actions = {agent: LLMAction() for _, agent in active_agents}
        await env.step(actions)
        
        # 记录动作
        for agent_id, agent in active_agents:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    platform="reddit",
                    agent_id=agent_id,
                    agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                    action_type="LLM_ACTION",
                    action_args={}
                )
                total_actions += 1
        
        if action_logger:
            action_logger.log_round_end(round_num + 1, len(active_agents), "reddit")
        
        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            print(f"[Reddit] Day {simulated_day}, {simulated_hour:02d}:00 "
                  f"- Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")
    
    await env.close()
    
    if action_logger:
        action_logger.log_simulation_end("reddit", total_rounds, total_actions)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[Reddit] 模拟完成! 耗时: {elapsed:.1f}秒, 总动作: {total_actions}")


async def main():
    parser = argparse.ArgumentParser(description='OASIS双平台并行模拟')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='配置文件路径 (simulation_config.json)'
    )
    parser.add_argument(
        '--twitter-only',
        action='store_true',
        help='只运行Twitter模拟'
    )
    parser.add_argument(
        '--reddit-only',
        action='store_true',
        help='只运行Reddit模拟'
    )
    parser.add_argument(
        '--action-log',
        type=str,
        default='actions.jsonl',
        help='动作日志文件路径 (默认: actions.jsonl)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    simulation_dir = os.path.dirname(args.config) or "."
    
    # 创建动作日志记录器
    action_log_path = os.path.join(simulation_dir, args.action_log)
    action_logger = ActionLogger(action_log_path)
    
    print("=" * 60)
    print("OASIS 双平台并行模拟")
    print(f"配置文件: {args.config}")
    print(f"模拟ID: {config.get('simulation_id', 'unknown')}")
    print(f"动作日志: {action_log_path}")
    print("=" * 60)
    
    time_config = config.get("time_config", {})
    print(f"\n模拟参数:")
    print(f"  - 总模拟时长: {time_config.get('total_simulation_hours', 72)}小时")
    print(f"  - 每轮时间: {time_config.get('minutes_per_round', 30)}分钟")
    print(f"  - Agent数量: {len(config.get('agent_configs', []))}")
    
    # LLM推理说明
    reasoning = config.get("generation_reasoning", "")
    if reasoning:
        print(f"\nLLM配置推理:")
        print(f"  {reasoning[:500]}..." if len(reasoning) > 500 else f"  {reasoning}")
    
    print("\n" + "=" * 60)
    
    start_time = datetime.now()
    
    if args.twitter_only:
        await run_twitter_simulation(config, simulation_dir, action_logger)
    elif args.reddit_only:
        await run_reddit_simulation(config, simulation_dir, action_logger)
    else:
        # 并行运行（共享同一个action_logger）
        await asyncio.gather(
            run_twitter_simulation(config, simulation_dir, action_logger),
            run_reddit_simulation(config, simulation_dir, action_logger),
        )
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print(f"全部模拟完成! 总耗时: {total_elapsed:.1f}秒")
    print(f"动作日志已保存到: {action_log_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

