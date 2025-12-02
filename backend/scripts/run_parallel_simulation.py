"""
OASIS 双平台并行模拟预设脚本
同时运行Twitter和Reddit模拟，读取相同的配置文件

使用方式:
    python run_parallel_simulation.py --config simulation_config.json [--action-log actions.jsonl]
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加 backend 目录到路径
# 脚本固定位于 backend/scripts/ 目录
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# 加载项目根目录的 .env 文件（包含 LLM_API_KEY 等配置）
from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print(f"已加载环境配置: {_env_file}")
else:
    # 尝试加载 backend/.env
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"已加载环境配置: {_backend_env}")


class UnicodeFormatter(logging.Formatter):
    """
    自定义格式化器，将 Unicode 转义序列（如 \\uXXXX）转换为可读字符
    """
    
    # 匹配 \uXXXX 形式的 Unicode 转义序列
    UNICODE_ESCAPE_PATTERN = None
    
    @classmethod
    def _get_pattern(cls):
        if cls.UNICODE_ESCAPE_PATTERN is None:
            import re
            cls.UNICODE_ESCAPE_PATTERN = re.compile(r'\\u([0-9a-fA-F]{4})')
        return cls.UNICODE_ESCAPE_PATTERN
    
    def format(self, record):
        # 先获取原始格式化结果
        result = super().format(record)
        # 使用正则表达式替换 Unicode 转义序列
        pattern = self._get_pattern()
        
        def replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return match.group(0)
        
        return pattern.sub(replace_unicode, result)


def setup_oasis_logging(log_dir: str):
    """
    配置 OASIS 的日志，覆盖默认的带时间戳日志文件
    
    Args:
        log_dir: 日志目录路径
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 清理旧的日志文件
    for f in os.listdir(log_dir):
        old_log = os.path.join(log_dir, f)
        if os.path.isfile(old_log) and f.endswith('.log'):
            try:
                os.remove(old_log)
            except OSError:
                pass
    
    # 创建自定义格式化器（支持 Unicode 解码）
    formatter = UnicodeFormatter(
        "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
    )
    
    # 重新配置 OASIS 使用的日志器，使用固定名称（不带时间戳）
    loggers_config = {
        "social.agent": os.path.join(log_dir, "social.agent.log"),
        "social.twitter": os.path.join(log_dir, "social.twitter.log"),
        "social.rec": os.path.join(log_dir, "social.rec.log"),
        "oasis.env": os.path.join(log_dir, "oasis.env.log"),
        "table": os.path.join(log_dir, "table.log"),
    }
    
    for logger_name, log_file in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # 清除 OASIS 添加的现有处理器（带时间戳的日志文件）
        logger.handlers.clear()
        # 添加新的文件处理器（使用 UTF-8 编码，固定文件名）
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # 防止日志向上传播（避免重复）
        logger.propagate = False
    
    print(f"日志配置完成，日志目录: {log_dir}")


def init_logging_for_simulation(simulation_dir: str):
    """初始化模拟的日志配置"""
    log_dir = os.path.join(simulation_dir, "log")
    setup_oasis_logging(log_dir)


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
    
    统一使用项目根目录 .env 文件中的配置（优先级最高）：
    - LLM_API_KEY: API密钥
    - LLM_BASE_URL: API基础URL
    - LLM_MODEL_NAME: 模型名称
    
    OASIS使用camel-ai的ModelFactory，需要设置 OPENAI_API_KEY 和 OPENAI_API_BASE_URL 环境变量
    """
    # 优先从 .env 读取配置
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_base_url = os.environ.get("LLM_BASE_URL", "")
    llm_model = os.environ.get("LLM_MODEL_NAME", "")
    
    # 如果 .env 中没有，则使用 config 作为备用
    if not llm_model:
        llm_model = config.get("llm_model", "gpt-4o-mini")
    
    # 设置 camel-ai 所需的环境变量
    if llm_api_key:
        os.environ["OPENAI_API_KEY"] = llm_api_key
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("缺少 API Key 配置，请在项目根目录 .env 文件中设置 LLM_API_KEY")
    
    if llm_base_url:
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url
    
    print(f"LLM配置: model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else '默认'}...")
    
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
    
    # 初始化日志配置（清理旧日志文件，使用固定名称）
    init_logging_for_simulation(simulation_dir)
    
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

