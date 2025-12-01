"""
模拟配置智能生成器
使用LLM根据模拟需求、文档内容、图谱信息自动生成细致的模拟参数
实现全程自动化，无需人工设置参数
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.simulation_config')


@dataclass
class AgentActivityConfig:
    """单个Agent的活动配置"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    # 活跃度配置 (0.0-1.0)
    activity_level: float = 0.5  # 整体活跃度
    
    # 发言频率（每小时预期发言次数）
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    
    # 活跃时间段（24小时制，0-23）
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    
    # 响应速度（对热点事件的反应延迟，单位：模拟分钟）
    response_delay_min: int = 5
    response_delay_max: int = 60
    
    # 情感倾向 (-1.0到1.0，负面到正面)
    sentiment_bias: float = 0.0
    
    # 立场（对特定话题的态度）
    stance: str = "neutral"  # supportive, opposing, neutral, observer
    
    # 影响力权重（决定其发言被其他Agent看到的概率）
    influence_weight: float = 1.0


@dataclass  
class TimeSimulationConfig:
    """时间模拟配置"""
    # 模拟总时长（模拟小时数）
    total_simulation_hours: int = 72  # 默认模拟72小时（3天）
    
    # 每轮代表的时间（模拟分钟）
    minutes_per_round: int = 30
    
    # 每小时激活的Agent数量范围
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    
    # 高峰时段（活跃度提升）
    peak_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 14, 15, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    
    # 低谷时段（活跃度降低）
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    off_peak_activity_multiplier: float = 0.3


@dataclass
class EventConfig:
    """事件配置"""
    # 初始事件（模拟开始时的触发事件）
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    # 定时事件（在特定时间触发的事件）
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # 热点话题关键词
    hot_topics: List[str] = field(default_factory=list)
    
    # 舆论引导方向
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """平台特定配置"""
    platform: str  # twitter or reddit
    
    # 推荐算法权重
    recency_weight: float = 0.4  # 时间新鲜度
    popularity_weight: float = 0.3  # 热度
    relevance_weight: float = 0.3  # 相关性
    
    # 病毒传播阈值（达到多少互动后触发扩散）
    viral_threshold: int = 10
    
    # 回声室效应强度（相似观点聚集程度）
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """完整的模拟参数配置"""
    # 基础信息
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str
    
    # 时间配置
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    
    # Agent配置列表
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    
    # 事件配置
    event_config: EventConfig = field(default_factory=EventConfig)
    
    # 平台配置
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    
    # LLM配置
    llm_model: str = ""
    llm_base_url: str = ""
    
    # 生成元数据
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM的推理说明
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": asdict(self.time_config),
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    模拟配置智能生成器
    
    使用LLM分析模拟需求、文档内容、图谱实体信息，
    自动生成最佳的模拟参数配置
    """
    
    # 上下文最大字符数
    MAX_CONTEXT_LENGTH = 50000
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
    ) -> SimulationParameters:
        """
        智能生成完整的模拟配置
        
        Args:
            simulation_id: 模拟ID
            project_id: 项目ID
            graph_id: 图谱ID
            simulation_requirement: 模拟需求描述
            document_text: 原始文档内容
            entities: 过滤后的实体列表
            enable_twitter: 是否启用Twitter
            enable_reddit: 是否启用Reddit
            
        Returns:
            SimulationParameters: 完整的模拟参数
        """
        logger.info(f"开始智能生成模拟配置: simulation_id={simulation_id}")
        
        # 1. 构建上下文信息（截断到50000字符）
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        # 2. 调用LLM生成配置
        llm_result = self._generate_config_with_llm(
            context=context,
            entities=entities,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit
        )
        
        # 3. 构建SimulationParameters对象
        params = self._build_parameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            entities=entities,
            llm_result=llm_result,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit
        )
        
        logger.info(f"模拟配置生成完成: {len(params.agent_configs)} 个Agent配置")
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """构建LLM上下文，截断到最大长度"""
        
        # 实体摘要
        entity_summary = self._summarize_entities(entities)
        
        # 构建上下文
        context_parts = [
            f"## 模拟需求\n{simulation_requirement}",
            f"\n## 实体信息 ({len(entities)}个)\n{entity_summary}",
        ]
        
        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  # 留500字符余量
        
        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(文档已截断)"
            context_parts.append(f"\n## 原始文档内容\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """生成实体摘要"""
        lines = []
        
        # 按类型分组
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)}个)")
            for e in type_entities[:10]:  # 每类最多显示10个
                summary_preview = (e.summary[:100] + "...") if len(e.summary) > 100 else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > 10:
                lines.append(f"  ... 还有 {len(type_entities) - 10} 个")
        
        return "\n".join(lines)
    
    def _generate_config_with_llm(
        self,
        context: str,
        entities: List[EntityNode],
        enable_twitter: bool,
        enable_reddit: bool
    ) -> Dict[str, Any]:
        """调用LLM生成配置"""
        
        # 构建实体列表用于Agent配置
        entity_list = []
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": i,
                "entity_uuid": e.uuid,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:200] if e.summary else ""
            })
        
        prompt = f"""你是一个社交媒体舆论模拟专家。请根据以下信息，生成详细的模拟参数配置。

{context}

## 实体列表（需要为每个实体生成活动配置）
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## 任务
请生成一个JSON配置，包含以下部分：

1. **time_config** - 时间模拟配置
   - total_simulation_hours: 模拟总时长（小时），根据事件性质决定（短期热点24-72小时，长期舆论168-336小时）
   - minutes_per_round: 每轮代表的时间（分钟），建议15-60
   - agents_per_hour_min/max: 每小时激活的Agent数量范围
   - peak_hours: 高峰时段列表（0-23）
   - off_peak_hours: 低谷时段列表

2. **agent_configs** - 每个Agent的活动配置（必须为每个实体生成）
   对于每个agent_id，设置：
   - activity_level: 活跃度(0.0-1.0)，官方机构通常0.1-0.3，媒体0.3-0.5，个人0.5-0.9
   - posts_per_hour: 每小时发帖频率，官方机构0.05-0.2，媒体0.5-2，个人0.1-1
   - comments_per_hour: 每小时评论频率
   - active_hours: 活跃时间段列表，官方通常工作时间，个人更分散
   - response_delay_min/max: 响应延迟（模拟分钟），官方较慢(30-180)，个人较快(1-30)
   - sentiment_bias: 情感倾向(-1到1)，根据实体立场设置
   - stance: 立场(supportive/opposing/neutral/observer)
   - influence_weight: 影响力权重，知名人物和媒体较高

3. **event_config** - 事件配置
   - initial_posts: 初始帖子列表，包含content和poster_agent_id
   - hot_topics: 热点话题关键词列表
   - narrative_direction: 舆论发展方向描述

4. **platform_configs** - 平台配置（如果启用）
   - viral_threshold: 病毒传播阈值
   - echo_chamber_strength: 回声室效应强度(0-1)

5. **reasoning** - 你的推理说明，解释为什么这样设置参数

## 重要原则
- 官方机构（University、GovernmentAgency）发言频率低但影响力大
- 媒体（MediaOutlet）发言频率中等，传播速度快
- 个人（Student、PublicFigure）发言频率高但影响力分散
- 根据模拟需求判断各实体的立场和情感倾向
- 时间配置要符合真实社交媒体的使用规律

请返回JSON格式，不要包含markdown代码块标记。"""

        try:
            # 使用重试机制调用LLM API
            from ..utils.retry import RetryableAPIClient
            
            retry_client = RetryableAPIClient(max_retries=3, initial_delay=2.0, max_delay=60.0)
            
            def call_llm():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "你是社交媒体舆论模拟专家，擅长设计真实的模拟参数。返回纯JSON格式，不要markdown。"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=8000
                )
            
            response = retry_client.call_with_retry(call_llm)
            result = json.loads(response.choices[0].message.content)
            logger.info(f"LLM配置生成成功")
            return result
            
        except Exception as e:
            logger.error(f"LLM配置生成失败（已重试）: {str(e)}")
            # 返回默认配置
            return self._generate_default_config(entities)
    
    def _generate_default_config(self, entities: List[EntityNode]) -> Dict[str, Any]:
        """生成默认配置（LLM失败时的fallback）"""
        agent_configs = []
        
        for i, e in enumerate(entities):
            entity_type = (e.get_entity_type() or "Unknown").lower()
            
            # 根据实体类型设置默认参数
            if entity_type in ["university", "governmentagency", "ngo"]:
                config = {
                    "agent_id": i,
                    "activity_level": 0.2,
                    "posts_per_hour": 0.1,
                    "comments_per_hour": 0.05,
                    "active_hours": list(range(9, 18)),
                    "response_delay_min": 60,
                    "response_delay_max": 240,
                    "sentiment_bias": 0.0,
                    "stance": "neutral",
                    "influence_weight": 3.0
                }
            elif entity_type in ["mediaoutlet"]:
                config = {
                    "agent_id": i,
                    "activity_level": 0.6,
                    "posts_per_hour": 1.0,
                    "comments_per_hour": 0.5,
                    "active_hours": list(range(6, 24)),
                    "response_delay_min": 5,
                    "response_delay_max": 30,
                    "sentiment_bias": 0.0,
                    "stance": "observer",
                    "influence_weight": 2.5
                }
            elif entity_type in ["publicfigure", "expert"]:
                config = {
                    "agent_id": i,
                    "activity_level": 0.5,
                    "posts_per_hour": 0.3,
                    "comments_per_hour": 0.5,
                    "active_hours": list(range(8, 23)),
                    "response_delay_min": 10,
                    "response_delay_max": 60,
                    "sentiment_bias": 0.0,
                    "stance": "neutral",
                    "influence_weight": 2.0
                }
            else:  # Student, Person, etc.
                config = {
                    "agent_id": i,
                    "activity_level": 0.7,
                    "posts_per_hour": 0.5,
                    "comments_per_hour": 1.0,
                    "active_hours": list(range(7, 24)),
                    "response_delay_min": 1,
                    "response_delay_max": 20,
                    "sentiment_bias": 0.0,
                    "stance": "neutral",
                    "influence_weight": 1.0
                }
            
            agent_configs.append(config)
        
        return {
            "time_config": {
                "total_simulation_hours": 72,
                "minutes_per_round": 30,
                "agents_per_hour_min": max(1, len(entities) // 10),
                "agents_per_hour_max": max(5, len(entities) // 3),
                "peak_hours": [9, 10, 11, 14, 15, 20, 21, 22],
                "off_peak_hours": [0, 1, 2, 3, 4, 5]
            },
            "agent_configs": agent_configs,
            "event_config": {
                "initial_posts": [],
                "hot_topics": [],
                "narrative_direction": ""
            },
            "reasoning": "使用默认配置（LLM生成失败）"
        }
    
    def _build_parameters(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        entities: List[EntityNode],
        llm_result: Dict[str, Any],
        enable_twitter: bool,
        enable_reddit: bool
    ) -> SimulationParameters:
        """根据LLM结果构建SimulationParameters对象"""
        
        # 时间配置
        time_cfg = llm_result.get("time_config", {})
        time_config = TimeSimulationConfig(
            total_simulation_hours=time_cfg.get("total_simulation_hours", 72),
            minutes_per_round=time_cfg.get("minutes_per_round", 30),
            agents_per_hour_min=time_cfg.get("agents_per_hour_min", 5),
            agents_per_hour_max=time_cfg.get("agents_per_hour_max", 20),
            peak_hours=time_cfg.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22]),
            off_peak_hours=time_cfg.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            peak_activity_multiplier=time_cfg.get("peak_activity_multiplier", 1.5),
            off_peak_activity_multiplier=time_cfg.get("off_peak_activity_multiplier", 0.3)
        )
        
        # Agent配置
        agent_configs = []
        llm_agent_configs = {cfg["agent_id"]: cfg for cfg in llm_result.get("agent_configs", [])}
        
        for i, entity in enumerate(entities):
            cfg = llm_agent_configs.get(i, {})
            
            agent_config = AgentActivityConfig(
                agent_id=i,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(8, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            agent_configs.append(agent_config)
        
        # 事件配置
        event_cfg = llm_result.get("event_config", {})
        event_config = EventConfig(
            initial_posts=event_cfg.get("initial_posts", []),
            scheduled_events=event_cfg.get("scheduled_events", []),
            hot_topics=event_cfg.get("hot_topics", []),
            narrative_direction=event_cfg.get("narrative_direction", "")
        )
        
        # 平台配置
        twitter_config = None
        reddit_config = None
        
        platform_cfgs = llm_result.get("platform_configs", {})
        
        if enable_twitter:
            tw_cfg = platform_cfgs.get("twitter", {})
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=tw_cfg.get("recency_weight", 0.4),
                popularity_weight=tw_cfg.get("popularity_weight", 0.3),
                relevance_weight=tw_cfg.get("relevance_weight", 0.3),
                viral_threshold=tw_cfg.get("viral_threshold", 10),
                echo_chamber_strength=tw_cfg.get("echo_chamber_strength", 0.5)
            )
        
        if enable_reddit:
            rd_cfg = platform_cfgs.get("reddit", {})
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=rd_cfg.get("recency_weight", 0.3),
                popularity_weight=rd_cfg.get("popularity_weight", 0.4),
                relevance_weight=rd_cfg.get("relevance_weight", 0.3),
                viral_threshold=rd_cfg.get("viral_threshold", 15),
                echo_chamber_strength=rd_cfg.get("echo_chamber_strength", 0.6)
            )
        
        return SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=llm_result.get("reasoning", "")
        )


