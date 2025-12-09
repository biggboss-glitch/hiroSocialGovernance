"""
Report Agent服务
使用LangChain + Zep实现ReACT模式的模拟报告生成

功能：
1. 根据模拟需求和Zep图谱信息生成报告
2. 先规划目录结构，然后分段生成
3. 每段采用ReACT多轮思考与反思模式
4. 支持与用户对话，在对话中自主调用检索工具
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import ZepToolsService, SearchResult

logger = get_logger('mirofish.report_agent')


class ReportStatus(str, Enum):
    """报告状态"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str = ""
    subsections: List['ReportSection'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections]
        }
    
    def to_markdown(self, level: int = 2) -> str:
        """转换为Markdown格式"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        for sub in self.subsections:
            md += sub.to_markdown(level + 1)
        return md


@dataclass
class ReportOutline:
    """报告大纲"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """转换为Markdown格式"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """完整报告"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


class ReportAgent:
    """
    Report Agent - 模拟报告生成Agent
    
    采用ReACT（Reasoning + Acting）模式：
    1. 规划阶段：分析模拟需求，规划报告目录结构
    2. 生成阶段：逐章节生成内容，每章节可多次调用工具获取信息
    3. 反思阶段：检查内容完整性和准确性
    
    工具（MCP封装）：
    - search_graph: 图谱语义搜索
    - get_graph_statistics: 获取图谱统计
    - get_entity_summary: 获取实体摘要
    - get_simulation_context: 获取模拟上下文
    """
    
    # 最大工具调用次数（每个章节）
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # 最大反思轮数
    MAX_REFLECTION_ROUNDS = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        初始化Report Agent
        
        Args:
            graph_id: 图谱ID
            simulation_id: 模拟ID
            simulation_requirement: 模拟需求描述
            llm_client: LLM客户端（可选）
            zep_tools: Zep工具服务（可选）
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # 工具定义
        self.tools = self._define_tools()
        
        logger.info(f"ReportAgent 初始化完成: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """定义可用工具"""
        return {
            "search_graph": {
                "name": "search_graph",
                "description": "在知识图谱中搜索相关信息。输入搜索查询，返回与查询相关的事实和关系。",
                "parameters": {
                    "query": "搜索查询字符串",
                    "limit": "返回结果数量（可选，默认10）"
                }
            },
            "get_graph_statistics": {
                "name": "get_graph_statistics",
                "description": "获取知识图谱的统计信息，包括节点数量、边数量、实体类型分布等。",
                "parameters": {}
            },
            "get_entity_summary": {
                "name": "get_entity_summary",
                "description": "获取指定实体的详细信息和关系摘要。",
                "parameters": {
                    "entity_name": "实体名称"
                }
            },
            "get_simulation_context": {
                "name": "get_simulation_context",
                "description": "获取与模拟需求相关的上下文信息，包括相关事实、实体列表等。",
                "parameters": {
                    "query": "额外的查询条件（可选）"
                }
            },
            "get_entities_by_type": {
                "name": "get_entities_by_type",
                "description": "按类型获取实体列表，如获取所有Student类型或PublicFigure类型的实体。",
                "parameters": {
                    "entity_type": "实体类型名称"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        执行工具调用
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            
        Returns:
            工具执行结果（文本格式）
        """
        logger.info(f"执行工具: {tool_name}, 参数: {parameters}")
        
        try:
            if tool_name == "search_graph":
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                result = self.zep_tools.search_graph(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                query = parameters.get("query", self.simulation_requirement)
                result = self.zep_tools.get_simulation_context(
                    graph_id=self.graph_id,
                    simulation_requirement=query
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"未知工具: {tool_name}"
                
        except Exception as e:
            logger.error(f"工具执行失败: {tool_name}, 错误: {str(e)}")
            return f"工具执行失败: {str(e)}"
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        从LLM响应中解析工具调用
        
        支持的格式：
        <tool_call>
        {"name": "tool_name", "parameters": {"param1": "value1"}}
        </tool_call>
        
        或者：
        [TOOL_CALL] tool_name(param1="value1", param2="value2")
        """
        tool_calls = []
        
        # 格式1: XML风格
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass
        
        # 格式2: 函数调用风格
        func_pattern = r'\[TOOL_CALL\]\s*(\w+)\s*\((.*?)\)'
        for match in re.finditer(func_pattern, response, re.DOTALL):
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # 解析参数
            params = {}
            for param_match in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', params_str):
                params[param_match.group(1)] = param_match.group(2)
            
            tool_calls.append({
                "name": tool_name,
                "parameters": params
            })
        
        return tool_calls
    
    def _get_tools_description(self) -> str:
        """生成工具描述文本"""
        desc_parts = ["可用工具："]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  参数: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        规划报告大纲
        
        使用LLM分析模拟需求，规划报告的目录结构
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            ReportOutline: 报告大纲
        """
        logger.info("开始规划报告大纲...")
        
        if progress_callback:
            progress_callback("planning", 0, "正在分析模拟需求...")
        
        # 首先获取模拟上下文
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "正在生成报告大纲...")
        
        # 构建规划prompt
        system_prompt = """你是一个专业的舆情分析报告撰写专家。你需要根据用户的模拟需求和已有的知识图谱信息，规划一份精炼的模拟分析报告大纲。

【重要】报告章节数量限制：
- 报告最多包含5个主章节
- 每个章节可以有0-2个子章节
- 内容要精炼，避免冗余

报告应聚焦以下核心内容（选择最相关的3-5项）：
1. 执行摘要 - 简要总结模拟结果和关键发现
2. 模拟背景 - 描述模拟的初始条件和场景设定
3. 关键发现 - 分析模拟中的重要发现和趋势
4. 舆情分析 - 分析舆论走向、情绪变化、关键意见领袖等
5. 建议与展望 - 基于分析结果提出建议

请输出JSON格式的报告大纲，格式如下：
{
    "title": "报告标题",
    "summary": "报告摘要（一句话概括）",
    "sections": [
        {
            "title": "章节标题",
            "description": "章节内容描述",
            "subsections": [
                {"title": "子章节标题", "description": "子章节描述"}
            ]
        }
    ]
}

注意：sections数组最多包含5个元素！"""

        user_prompt = f"""模拟需求：
{self.simulation_requirement}

已有的知识图谱信息：
- 总节点数: {context.get('graph_statistics', {}).get('total_nodes', 0)}
- 总边数: {context.get('graph_statistics', {}).get('total_edges', 0)}
- 实体类型: {list(context.get('graph_statistics', {}).get('entity_types', {}).keys())}
- 实体数量: {context.get('total_entities', 0)}

相关事实：
{json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2)}

请根据以上信息，生成一份针对此模拟场景的报告大纲。

【再次提醒】报告必须控制在最多5个章节以内，内容要精炼聚焦。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "正在解析大纲结构...")
            
            # 解析大纲
            sections = []
            for section_data in response.get("sections", []):
                subsections = []
                for sub_data in section_data.get("subsections", []):
                    subsections.append(ReportSection(
                        title=sub_data.get("title", ""),
                        content=""
                    ))
                
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content="",
                    subsections=subsections
                ))
            
            outline = ReportOutline(
                title=response.get("title", "模拟分析报告"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "大纲规划完成")
            
            logger.info(f"大纲规划完成: {len(sections)} 个章节")
            return outline
            
        except Exception as e:
            logger.error(f"大纲规划失败: {str(e)}")
            # 返回默认大纲（5个章节）
            return ReportOutline(
                title="模拟分析报告",
                summary="基于模拟结果的分析报告",
                sections=[
                    ReportSection(title="执行摘要"),
                    ReportSection(title="模拟背景与场景设定"),
                    ReportSection(title="关键发现与趋势分析"),
                    ReportSection(title="舆情走向与情绪演化"),
                    ReportSection(title="总结与建议")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        使用ReACT模式生成单个章节内容
        
        ReACT循环：
        1. Thought（思考）- 分析需要什么信息
        2. Action（行动）- 调用工具获取信息
        3. Observation（观察）- 分析工具返回结果
        4. 重复直到信息足够或达到最大次数
        5. Final Answer（最终回答）- 生成章节内容
        
        Args:
            section: 要生成的章节
            outline: 完整大纲
            previous_sections: 之前章节的内容（用于保持连贯性）
            progress_callback: 进度回调
            
        Returns:
            章节内容（Markdown格式）
        """
        logger.info(f"ReACT生成章节: {section.title}")
        
        # 构建系统prompt
        system_prompt = f"""你是一个专业的舆情分析报告撰写专家，正在撰写报告的一个章节。

报告标题: {outline.title}
报告摘要: {outline.summary}
模拟需求: {self.simulation_requirement}

当前要撰写的章节: {section.title}

你可以使用以下工具来获取信息，每次最多调用{self.MAX_TOOL_CALLS_PER_SECTION}次：

{self._get_tools_description()}

请按照以下ReACT格式进行思考和行动：

Thought: [分析当前需要什么信息来撰写这个章节]
Action: [如果需要信息，调用工具]
<tool_call>
{{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}
</tool_call>

当收集到足够信息后，输出：
Final Answer:
[章节的完整Markdown内容]

注意：
1. 内容要专业、客观、有深度
2. 引用具体的数据和事实
3. 保持与其他章节的逻辑连贯性
4. 使用适当的Markdown格式（列表、强调等）
5. 不要重复前面章节已经详细描述的内容"""

        # 构建用户prompt
        previous_content = "\n\n".join(previous_sections) if previous_sections else "（这是第一个章节）"
        user_prompt = f"""已完成的章节内容：
{previous_content[:2000]}

现在请撰写章节: {section.title}

首先思考需要什么信息，然后调用工具获取，最后生成内容。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT循环
        tool_calls_count = 0
        max_iterations = self.MAX_TOOL_CALLS_PER_SECTION + 2
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"思考与行动中 ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # 调用LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )
            
            logger.debug(f"LLM响应: {response[:200]}...")
            
            # 检查是否有最终答案
            if "Final Answer:" in response:
                # 提取最终答案
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"章节 {section.title} 生成完成")
                return final_answer
            
            # 解析工具调用
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # 没有工具调用也没有最终答案，提示生成最终答案
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user", 
                    "content": "请基于已有信息，输出 Final Answer: 并生成章节内容。"
                })
                continue
            
            # 执行工具调用
            tool_results = []
            for call in tool_calls:
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    break
                
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append(f"工具 {call['name']} 返回:\n{result}")
                tool_calls_count += 1
            
            # 将结果添加到消息
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation:\n" + "\n\n".join(tool_results) + "\n\n请继续思考或输出 Final Answer:"
            })
        
        # 达到最大迭代次数，强制生成内容
        logger.warning(f"章节 {section.title} 达到最大迭代次数，强制生成")
        messages.append({
            "role": "user",
            "content": "已达到工具调用限制，请直接输出 Final Answer: 并生成章节内容。"
        })
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )
        
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()
        
        return response
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None
    ) -> Report:
        """
        生成完整报告（分章节实时输出）
        
        每个章节生成完成后立即保存到文件夹，不需要等待整个报告完成。
        文件结构：
        reports/{report_id}/
            meta.json       - 报告元信息
            outline.json    - 报告大纲
            progress.json   - 生成进度
            section_01.md   - 第1章节
            section_02.md   - 第2章节
            ...
            full_report.md  - 完整报告
        
        Args:
            progress_callback: 进度回调函数 (stage, progress, message)
            
        Returns:
            Report: 完整报告
        """
        import uuid
        
        report_id = f"report_{uuid.uuid4().hex[:12]}"
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # 已完成的章节标题列表（用于进度追踪）
        completed_section_titles = []
        
        try:
            # 初始化：创建报告文件夹并保存初始状态
            ReportManager._ensure_report_folder(report_id)
            ReportManager.update_progress(
                report_id, "pending", 0, "初始化报告...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # 阶段1: 规划大纲
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "开始规划报告大纲...",
                completed_sections=[]
            )
            
            if progress_callback:
                progress_callback("planning", 0, "开始规划报告大纲...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # 保存大纲到文件
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"大纲规划完成，共{len(outline.sections)}个章节",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"大纲已保存到文件: {report_id}/outline.json")
            
            # 阶段2: 逐章节生成（分章节保存）
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # 保存内容用于上下文
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # 更新进度
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"正在生成章节: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"正在生成章节: {section.title} ({section_num}/{total_sections})"
                    )
                
                # 生成章节内容
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")
                
                # 【关键】立即保存章节到文件
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)
                
                logger.info(f"章节已保存: {report_id}/section_{section_num:02d}.md")
                
                # 更新进度
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"章节 {section.title} 已完成",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
                
                # 生成并保存子章节
                for j, subsection in enumerate(section.subsections):
                    subsection_num = j + 1
                    
                    if progress_callback:
                        progress_callback(
                            "generating",
                            base_progress + int(((j + 1) / len(section.subsections)) * 5),
                            f"正在生成子章节: {subsection.title}"
                        )
                    
                    ReportManager.update_progress(
                        report_id, "generating",
                        base_progress + int(((j + 1) / len(section.subsections)) * 5),
                        f"正在生成子章节: {subsection.title}",
                        current_section=subsection.title,
                        completed_sections=completed_section_titles
                    )
                    
                    subsection_content = self._generate_section_react(
                        section=subsection,
                        outline=outline,
                        previous_sections=generated_sections,
                        progress_callback=None
                    )
                    subsection.content = subsection_content
                    generated_sections.append(f"### {subsection.title}\n\n{subsection_content}")
                    
                    # 【关键】立即保存子章节到文件
                    ReportManager.save_section(
                        report_id, subsection_num, subsection,
                        is_subsection=True, parent_index=section_num
                    )
                    completed_section_titles.append(f"  └─ {subsection.title}")
                    
                    logger.info(f"子章节已保存: {report_id}/section_{section_num:02d}_{subsection_num:02d}.md")
            
            # 阶段3: 组装完整报告
            if progress_callback:
                progress_callback("generating", 95, "正在组装完整报告...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "正在组装完整报告...",
                completed_sections=completed_section_titles
            )
            
            # 使用ReportManager组装完整报告
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # 保存最终报告
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "报告生成完成",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "报告生成完成")
            
            logger.info(f"报告生成完成: {report_id}")
            return report
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # 保存失败状态
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"报告生成失败: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # 忽略保存失败的错误
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        与Report Agent对话
        
        在对话中Agent可以自主调用检索工具来回答问题
        
        Args:
            message: 用户消息
            chat_history: 对话历史
            
        Returns:
            {
                "response": "Agent回复",
                "tool_calls": [调用的工具列表],
                "sources": [信息来源]
            }
        """
        logger.info(f"Report Agent对话: {message[:50]}...")
        
        chat_history = chat_history or []
        
        system_prompt = f"""你是一个专业的舆情分析助手，负责回答关于模拟分析报告的问题。

模拟需求: {self.simulation_requirement}
图谱ID: {self.graph_id}

你可以使用以下工具来获取信息：

{self._get_tools_description()}

工具调用格式：
<tool_call>
{{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}
</tool_call>

回答要求：
1. 基于事实和数据回答
2. 引用具体信息来源
3. 如果不确定，说明信息限制
4. 保持专业和客观"""

        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史对话
        for h in chat_history[-10:]:  # 限制历史长度
            messages.append(h)
        
        messages.append({"role": "user", "content": message})
        
        # ReACT循环
        tool_calls_made = []
        max_iterations = 3
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=2048
            )
            
            # 解析工具调用
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # 没有工具调用，返回响应
                # 清理响应中的工具调用标记
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": []
                }
            
            # 执行工具调用
            tool_results = []
            for call in tool_calls:
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1000]  # 限制长度
                })
                tool_calls_made.append(call)
            
            # 将结果添加到消息
            messages.append({"role": "assistant", "content": response})
            observation = "工具调用结果:\n" + "\n\n".join([
                f"[{r['tool']}]: {r['result']}" for r in tool_results
            ])
            messages.append({"role": "user", "content": observation + "\n\n请基于以上信息回答问题。"})
        
        # 达到最大迭代，获取最终响应
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=2048
        )
        
        return {
            "response": final_response,
            "tool_calls": tool_calls_made,
            "sources": []
        }


class ReportManager:
    """
    报告管理器
    
    负责报告的持久化存储和检索
    
    文件结构（分章节输出）：
    reports/
      {report_id}/
        meta.json          - 报告元信息和状态
        outline.json       - 报告大纲
        progress.json      - 生成进度
        section_01.md      - 第1章节
        section_02.md      - 第2章节
        ...
        full_report.md     - 完整报告
    """
    
    # 报告存储目录
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """确保报告根目录存在"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """获取报告文件夹路径"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """确保报告文件夹存在并返回路径"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """获取报告元信息文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """获取完整报告Markdown文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """获取大纲文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """获取进度文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """获取章节Markdown文件路径"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        保存报告大纲
        
        在规划阶段完成后立即调用
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"大纲已保存: {report_id}")
    
    @classmethod
    def save_section(
        cls, 
        report_id: str, 
        section_index: int, 
        section: ReportSection,
        is_subsection: bool = False,
        parent_index: int = None
    ) -> str:
        """
        保存单个章节
        
        在每个章节生成完成后立即调用，实现分章节输出
        
        Args:
            report_id: 报告ID
            section_index: 章节索引（从1开始）
            section: 章节对象
            is_subsection: 是否是子章节
            parent_index: 父章节索引（子章节时使用）
            
        Returns:
            保存的文件路径
        """
        cls._ensure_report_folder(report_id)
        
        # 确定章节级别和标题格式
        if is_subsection and parent_index is not None:
            level = "###"
            file_suffix = f"section_{parent_index:02d}_{section_index:02d}.md"
        else:
            level = "##"
            file_suffix = f"section_{section_index:02d}.md"
        
        # 构建章节Markdown内容
        md_content = f"{level} {section.title}\n\n"
        if section.content:
            md_content += f"{section.content}\n\n"
        
        # 保存文件
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"章节已保存: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        更新报告生成进度
        
        前端可以通过读取progress.json获取实时进度
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """获取报告生成进度"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        获取已生成的章节列表
        
        返回所有已保存的章节文件信息
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 从文件名解析章节索引
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])
                subsection_index = int(parts[2]) if len(parts) > 2 else None
                
                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "subsection_index": subsection_index,
                    "content": content,
                    "is_subsection": subsection_index is not None
                })
        
        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        组装完整报告
        
        从已保存的章节文件组装完整报告
        """
        folder = cls._get_report_folder(report_id)
        
        # 构建报告头部
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # 按顺序读取所有章节文件
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # 保存完整报告
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"完整报告已组装: {report_id}")
        return md_content
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """保存报告元信息和完整报告"""
        cls._ensure_report_folder(report.report_id)
        
        # 保存元信息JSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 保存大纲
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # 保存完整Markdown报告
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"报告已保存: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """获取报告"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # 兼容旧格式：检查直接存储在reports目录下的文件
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建Report对象
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                subsections = [
                    ReportSection(title=sub['title'], content=sub.get('content', ''))
                    for sub in s.get('subsections', [])
                ]
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', ''),
                    subsections=subsections
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # 如果markdown_content为空，尝试从full_report.md读取
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """根据模拟ID获取报告"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新格式：文件夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # 兼容旧格式：JSON文件
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """列出报告"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新格式：文件夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # 兼容旧格式：JSON文件
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # 按创建时间倒序
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """删除报告（整个文件夹）"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # 新格式：删除整个文件夹
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"报告文件夹已删除: {report_id}")
            return True
        
        # 兼容旧格式：删除单独的文件
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
