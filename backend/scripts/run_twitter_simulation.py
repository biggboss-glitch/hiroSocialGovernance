"""
OASIS Twitter模拟预设脚本
此脚本读取配置文件中的参数来执行模拟，实现全程自动化

使用方式:
    python run_twitter_simulation.py --config /path/to/simulation_config.json
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, Any, List

# 添加项目路径
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
else:
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)


import re


class UnicodeFormatter(logging.Formatter):
    """自定义格式化器，将 Unicode 转义序列转换为可读字符"""
    
    UNICODE_ESCAPE_PATTERN = re.compile(r'\\u([0-9a-fA-F]{4})')
    
    def format(self, record):
        result = super().format(record)
        
        def replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return match.group(0)
        
        return self.UNICODE_ESCAPE_PATTERN.sub(replace_unicode, result)


def setup_oasis_logging(log_dir: str):
    """配置 OASIS 的日志，使用固定名称的日志文件"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 清理旧的日志文件
    for f in os.listdir(log_dir):
        old_log = os.path.join(log_dir, f)
        if os.path.isfile(old_log) and f.endswith('.log'):
            try:
                os.remove(old_log)
            except OSError:
                pass
    
    formatter = UnicodeFormatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
    
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
        logger.handlers.clear()
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False


try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph
    )
except ImportError as e:
    print(f"错误: 缺少依赖 {e}")
    print("请先安装: pip install oasis-ai camel-ai")
    sys.exit(1)


class TwitterSimulationRunner:
    """Twitter模拟运行器"""
    
    # Twitter可用动作
    AVAILABLE_ACTIONS = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST,
    ]
    
    def __init__(self, config_path: str):
        """
        初始化模拟运行器
        
        Args:
            config_path: 配置文件路径 (simulation_config.json)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.simulation_dir = os.path.dirname(config_path)
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_profile_path(self) -> str:
        """获取Profile文件路径（OASIS Twitter使用CSV格式）"""
        return os.path.join(self.simulation_dir, "twitter_profiles.csv")
    
    def _get_db_path(self) -> str:
        """获取数据库路径"""
        return os.path.join(self.simulation_dir, "twitter_simulation.db")
    
    def _create_model(self):
        """
        创建LLM模型
        
        统一使用项目根目录 .env 文件中的配置（优先级最高）：
        - LLM_API_KEY: API密钥
        - LLM_BASE_URL: API基础URL
        - LLM_MODEL_NAME: 模型名称
        """
        # 优先从 .env 读取配置
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        
        # 如果 .env 中没有，则使用 config 作为备用
        if not llm_model:
            llm_model = self.config.get("llm_model", "gpt-4o-mini")
        
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
    
    def _get_active_agents_for_round(
        self, 
        env, 
        current_hour: int,
        round_num: int
    ) -> List:
        """
        根据时间和配置决定本轮激活哪些Agent
        
        Args:
            env: OASIS环境
            current_hour: 当前模拟小时（0-23）
            round_num: 当前轮数
            
        Returns:
            激活的Agent列表
        """
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])
        
        # 基础激活数量
        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)
        
        # 根据时段调整
        peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
        off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
        
        if current_hour in peak_hours:
            multiplier = time_config.get("peak_activity_multiplier", 1.5)
        elif current_hour in off_peak_hours:
            multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
        else:
            multiplier = 1.0
        
        target_count = int(random.uniform(base_min, base_max) * multiplier)
        
        # 根据每个Agent的配置计算激活概率
        candidates = []
        for cfg in agent_configs:
            agent_id = cfg.get("agent_id", 0)
            active_hours = cfg.get("active_hours", list(range(8, 23)))
            activity_level = cfg.get("activity_level", 0.5)
            
            # 检查是否在活跃时间
            if current_hour not in active_hours:
                continue
            
            # 根据活跃度计算概率
            if random.random() < activity_level:
                candidates.append(agent_id)
        
        # 随机选择
        selected_ids = random.sample(
            candidates, 
            min(target_count, len(candidates))
        ) if candidates else []
        
        # 转换为Agent对象
        active_agents = []
        for agent_id in selected_ids:
            try:
                agent = env.agent_graph.get_agent(agent_id)
                active_agents.append((agent_id, agent))
            except Exception:
                pass
        
        return active_agents
    
    async def run(self):
        """运行Twitter模拟"""
        print("=" * 60)
        print("OASIS Twitter模拟")
        print(f"配置文件: {self.config_path}")
        print(f"模拟ID: {self.config.get('simulation_id', 'unknown')}")
        print("=" * 60)
        
        # 加载时间配置
        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        
        # 计算总轮数
        total_rounds = (total_hours * 60) // minutes_per_round
        
        print(f"\n模拟参数:")
        print(f"  - 总模拟时长: {total_hours}小时")
        print(f"  - 每轮时间: {minutes_per_round}分钟")
        print(f"  - 总轮数: {total_rounds}")
        print(f"  - Agent数量: {len(self.config.get('agent_configs', []))}")
        
        # 创建模型
        print("\n初始化LLM模型...")
        model = self._create_model()
        
        # 加载Agent图
        print("加载Agent Profile...")
        profile_path = self._get_profile_path()
        if not os.path.exists(profile_path):
            print(f"错误: Profile文件不存在: {profile_path}")
            return
        
        agent_graph = await generate_twitter_agent_graph(
            profile_path=profile_path,
            model=model,
            available_actions=self.AVAILABLE_ACTIONS,
        )
        
        # 数据库路径
        db_path = self._get_db_path()
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"已删除旧数据库: {db_path}")
        
        # 创建环境
        print("创建OASIS环境...")
        env = oasis.make(
            agent_graph=agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
        )
        
        await env.reset()
        print("环境初始化完成\n")
        
        # 执行初始事件
        event_config = self.config.get("event_config", {})
        initial_posts = event_config.get("initial_posts", [])
        
        if initial_posts:
            print(f"执行初始事件 ({len(initial_posts)}条初始帖子)...")
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
                except Exception as e:
                    print(f"  警告: 无法为Agent {agent_id}创建初始帖子: {e}")
            
            if initial_actions:
                await env.step(initial_actions)
                print(f"  已发布 {len(initial_actions)} 条初始帖子")
        
        # 主模拟循环
        print("\n开始模拟循环...")
        start_time = datetime.now()
        
        for round_num in range(total_rounds):
            # 计算当前模拟时间
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24
            simulated_day = simulated_minutes // (60 * 24) + 1
            
            # 获取本轮激活的Agent
            active_agents = self._get_active_agents_for_round(
                env, simulated_hour, round_num
            )
            
            if not active_agents:
                continue
            
            # 构建动作
            actions = {
                agent: LLMAction()
                for _, agent in active_agents
            }
            
            # 执行动作
            await env.step(actions)
            
            # 打印进度
            if (round_num + 1) % 10 == 0 or round_num == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (round_num + 1) / total_rounds * 100
                print(f"  [Day {simulated_day}, {simulated_hour:02d}:00] "
                      f"Round {round_num + 1}/{total_rounds} ({progress:.1f}%) "
                      f"- {len(active_agents)} agents active "
                      f"- elapsed: {elapsed:.1f}s")
        
        # 关闭环境
        await env.close()
        
        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n模拟完成!")
        print(f"  - 总耗时: {total_elapsed:.1f}秒")
        print(f"  - 数据库: {db_path}")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='OASIS Twitter模拟')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='配置文件路径 (simulation_config.json)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 初始化日志配置（使用固定文件名，清理旧日志）
    simulation_dir = os.path.dirname(args.config) or "."
    setup_oasis_logging(os.path.join(simulation_dir, "log"))
    
    runner = TwitterSimulationRunner(args.config)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())

