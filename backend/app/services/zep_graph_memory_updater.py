"""
Zep图谱记忆更新服务
将模拟中的Agent活动动态更新到Zep图谱中
"""

import os
import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.zep_graph_memory_updater')


@dataclass
class AgentActivity:
    """Agent活动记录"""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str
    
    def to_episode_text(self) -> str:
        """
        将活动转换为可以发送给Zep的文本描述
        
        采用自然语言描述格式，让Zep能够从中提取实体和关系
        不添加模拟相关的前缀，避免误导图谱更新
        """
        # 根据不同的动作类型生成不同的描述
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }
        
        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()
        
        # 直接返回 "agent名称: 活动描述" 格式，不添加模拟前缀
        return f"{self.agent_name}: {description}"
    
    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f"发布了一条帖子：「{content}」"
        return "发布了一条帖子"
    
    def _describe_like_post(self) -> str:
        post_id = self.action_args.get("post_id", "")
        return f"点赞了帖子#{post_id}" if post_id else "点赞了一条帖子"
    
    def _describe_dislike_post(self) -> str:
        post_id = self.action_args.get("post_id", "")
        return f"踩了帖子#{post_id}" if post_id else "踩了一条帖子"
    
    def _describe_repost(self) -> str:
        post_id = self.action_args.get("post_id", "")
        return f"转发了帖子#{post_id}" if post_id else "转发了一条帖子"
    
    def _describe_quote_post(self) -> str:
        quoted_id = self.action_args.get("quoted_id", "")
        content = self.action_args.get("content", "")
        if quoted_id:
            if content:
                return f"引用帖子#{quoted_id}并评论：「{content}」"
            return f"引用了帖子#{quoted_id}"
        return "引用了一条帖子"
    
    def _describe_follow(self) -> str:
        target_id = self.action_args.get("user_id", "") or self.action_args.get("target_id", "")
        return f"关注了用户#{target_id}" if target_id else "关注了一个用户"
    
    def _describe_create_comment(self) -> str:
        content = self.action_args.get("content", "")
        post_id = self.action_args.get("post_id", "")
        if content:
            base = f"评论道：「{content}」"
            if post_id:
                base = f"在帖子#{post_id}下{base}"
            return base
        return f"在帖子#{post_id}下发表了评论" if post_id else "发表了评论"
    
    def _describe_like_comment(self) -> str:
        comment_id = self.action_args.get("comment_id", "")
        return f"点赞了评论#{comment_id}" if comment_id else "点赞了一条评论"
    
    def _describe_dislike_comment(self) -> str:
        comment_id = self.action_args.get("comment_id", "")
        return f"踩了评论#{comment_id}" if comment_id else "踩了一条评论"
    
    def _describe_search(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"搜索了「{query}」" if query else "进行了搜索"
    
    def _describe_search_user(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f"搜索了用户「{query}」" if query else "搜索了用户"
    
    def _describe_mute(self) -> str:
        target_id = self.action_args.get("user_id", "") or self.action_args.get("target_id", "")
        return f"屏蔽了用户#{target_id}" if target_id else "屏蔽了一个用户"
    
    def _describe_generic(self) -> str:
        # 对于未知的动作类型，生成通用描述
        return f"执行了{self.action_type}操作"


class ZepGraphMemoryUpdater:
    """
    Zep图谱记忆更新器
    
    监控模拟的actions日志文件，将新的agent活动实时更新到Zep图谱中。
    每条活动单独发送到Zep，确保图谱能正确解析实体和关系。
    """
    
    # 发送间隔（秒），避免请求过快
    SEND_INTERVAL = 0.5
    
    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # 秒
    
    def __init__(self, graph_id: str, api_key: Optional[str] = None):
        """
        初始化更新器
        
        Args:
            graph_id: Zep图谱ID
            api_key: Zep API Key（可选，默认从配置读取）
        """
        self.graph_id = graph_id
        self.api_key = api_key or Config.ZEP_API_KEY
        
        if not self.api_key:
            raise ValueError("ZEP_API_KEY未配置")
        
        self.client = Zep(api_key=self.api_key)
        
        # 活动队列
        self._activity_queue: Queue = Queue()
        
        # 控制标志
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # 统计
        self._total_activities = 0
        self._total_sent = 0
        self._failed_count = 0
        
        logger.info(f"ZepGraphMemoryUpdater 初始化完成: graph_id={graph_id}")
    
    def start(self):
        """启动后台工作线程"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"ZepMemoryUpdater-{self.graph_id[:8]}"
        )
        self._worker_thread.start()
        logger.info(f"ZepGraphMemoryUpdater 已启动: graph_id={self.graph_id}")
    
    def stop(self):
        """停止后台工作线程"""
        self._running = False
        
        # 发送剩余的活动
        self._flush_remaining()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        
        logger.info(f"ZepGraphMemoryUpdater 已停止: graph_id={self.graph_id}, "
                   f"total_activities={self._total_activities}, "
                   f"total_sent={self._total_sent}, "
                   f"failed={self._failed_count}")
    
    def add_activity(self, activity: AgentActivity):
        """
        添加一个agent活动到队列
        
        Args:
            activity: Agent活动记录
        """
        # 跳过DO_NOTHING类型的活动
        if activity.action_type == "DO_NOTHING":
            return
        
        self._activity_queue.put(activity)
        self._total_activities += 1
    
    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """
        从字典数据添加活动
        
        Args:
            data: 从actions.jsonl解析的字典数据
            platform: 平台名称 (twitter/reddit)
        """
        # 跳过事件类型的条目
        if "event_type" in data:
            return
        
        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        
        self.add_activity(activity)
    
    def _worker_loop(self):
        """后台工作循环 - 逐条发送活动到Zep"""
        while self._running or not self._activity_queue.empty():
            try:
                # 尝试从队列获取活动（超时1秒）
                try:
                    activity = self._activity_queue.get(timeout=1)
                    # 立即发送单条活动
                    self._send_single_activity(activity)
                    # 发送间隔，避免请求过快
                    time.sleep(self.SEND_INTERVAL)
                except Empty:
                    pass
                    
            except Exception as e:
                logger.error(f"工作循环异常: {e}")
                time.sleep(1)
    
    def _send_single_activity(self, activity: AgentActivity):
        """
        发送单条活动到Zep图谱
        
        Args:
            activity: 单条Agent活动
        """
        episode_text = activity.to_episode_text()
        
        # 带重试的发送
        for attempt in range(self.MAX_RETRIES):
            try:
                self.client.graph.add(
                    graph_id=self.graph_id,
                    type="text",
                    data=episode_text
                )
                
                self._total_sent += 1
                logger.debug(f"成功发送活动到图谱 {self.graph_id}: {episode_text[:50]}...")
                return
                
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"发送到Zep失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"发送到Zep失败，已重试{self.MAX_RETRIES}次: {e}")
                    self._failed_count += 1
    
    def _flush_remaining(self):
        """发送队列中剩余的活动（逐条发送）"""
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                self._send_single_activity(activity)
            except Empty:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "graph_id": self.graph_id,
            "total_activities": self._total_activities,
            "total_sent": self._total_sent,
            "failed_count": self._failed_count,
            "queue_size": self._activity_queue.qsize(),
            "running": self._running,
        }


class ZepGraphMemoryManager:
    """
    管理多个模拟的Zep图谱记忆更新器
    
    每个模拟可以有自己的更新器实例
    """
    
    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()
    
    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> ZepGraphMemoryUpdater:
        """
        为模拟创建图谱记忆更新器
        
        Args:
            simulation_id: 模拟ID
            graph_id: Zep图谱ID
            
        Returns:
            ZepGraphMemoryUpdater实例
        """
        with cls._lock:
            # 如果已存在，先停止旧的
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
            
            updater = ZepGraphMemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater
            
            logger.info(f"创建图谱记忆更新器: simulation_id={simulation_id}, graph_id={graph_id}")
            return updater
    
    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        """获取模拟的更新器"""
        return cls._updaters.get(simulation_id)
    
    @classmethod
    def stop_updater(cls, simulation_id: str):
        """停止并移除模拟的更新器"""
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]
                logger.info(f"已停止图谱记忆更新器: simulation_id={simulation_id}")
    
    @classmethod
    def stop_all(cls):
        """停止所有更新器"""
        with cls._lock:
            for simulation_id, updater in list(cls._updaters.items()):
                try:
                    updater.stop()
                except Exception as e:
                    logger.error(f"停止更新器失败: simulation_id={simulation_id}, error={e}")
            cls._updaters.clear()
            logger.info("已停止所有图谱记忆更新器")
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有更新器的统计信息"""
        return {
            sim_id: updater.get_stats() 
            for sim_id, updater in cls._updaters.items()
        }
