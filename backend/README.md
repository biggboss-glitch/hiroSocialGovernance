# MiroFish Backend - 详细技术文档

## 目录

- [项目简介](#项目简介)
- [技术架构](#技术架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [核心功能模块](#核心功能模块)
- [API接口文档](#api接口文档)
- [数据模型](#数据模型)
- [服务层详解](#服务层详解)
- [工具类](#工具类)
- [配置说明](#配置说明)
- [运行指南](#运行指南)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

---

## 项目简介

**MiroFish Backend** 是一个基于 Flask 的后端服务,用于社交媒体舆论模拟。系统核心功能包括:

1. **知识图谱构建**: 从文档中提取实体和关系,使用 Zep Cloud 构建知识图谱
2. **本体生成**: 使用 LLM 自动分析文档并生成适合舆论模拟的实体类型和关系类型
3. **Agent人设生成**: 基于图谱实体,使用 LLM 生成详细的社交媒体用户人设
4. **模拟配置智能生成**: 使用 LLM 根据需求自动生成模拟参数(时间、活跃度、事件等)
5. **双平台模拟**: 支持 Twitter 和 Reddit 双平台并行舆论模拟(基于 OASIS 框架)
6. **图谱记忆动态更新**: 可选功能,将模拟中Agent的活动实时更新到Zep图谱,让图谱"记住"模拟过程
7. **智能报告生成**: 使用 LangChain + Zep 实现 ReACT 模式的模拟分析报告自动生成
8. **Report Agent对话**: 报告生成后可与Report Agent对话,自主调用检索工具回答问题

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        MiroFish Backend                       │
├─────────────────────────────────────────────────────────────┤
│  Flask Web Framework + CORS                                  │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  API层         │  │  服务层      │  │  模型层         │ │
│  │  - graph.py    │→ │  - 本体生成  │→ │  - Project      │ │
│  │  - simulation  │  │  - 图谱构建  │  │  - Task         │ │
│  │  - report.py   │  │  - 实体读取  │  │  - Report       │ │
│  └────────────────┘  │  - 人设生成  │  └─────────────────┘ │
│                      │  - 配置生成  │                        │
│                      │  - 模拟运行  │                        │
│                      │  - 报告生成  │                        │
│                      └──────────────┘                        │
├─────────────────────────────────────────────────────────────┤
│  外部服务集成                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Zep Cloud│  │ LLM API  │  │  OASIS   │  │  文件系统│   │
│  │ 知识图谱 │  │ (OpenAI) │  │  社交模拟│  │  存储    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 核心流程

1. **图谱构建流程**:
   ```
   上传文档 → 提取文本 → LLM生成本体 → 文本分块 → Zep构建图谱
   ```

2. **模拟准备流程**:
   ```
   创建模拟 → 读取图谱实体 → LLM生成人设 → LLM生成配置 → 准备完成
   ```

3. **模拟运行流程**:
   ```
   启动模拟 → 运行OASIS脚本 → 实时监控 → 记录动作 → (可选)更新Zep图谱记忆 → 状态查询
   ```

4. **Interview采访流程**:
   ```
   模拟完成 → 环境进入等待模式 → 发送Interview命令 → Agent回答 → 获取结果 → (可选)关闭环境
   ```

5. **报告生成流程**:
   ```
   模拟完成 → 调用Report API → ReACT规划大纲 → 逐章节生成(多次工具调用) → 生成Markdown报告 → 解锁Interview功能
   ```

6. **Report Agent对话流程**:
   ```
   用户提问 → Agent分析 → 调用Zep检索工具 → 整合信息 → 返回回答
   ```

---

## 技术栈

### 核心框架
- **Flask 3.0+**: Web 框架
- **Flask-CORS**: 跨域支持

### AI & 知识图谱
- **Zep Cloud SDK 2.0+**: 知识图谱构建与管理
- **OpenAI SDK 1.0+**: LLM 调用(支持 OpenAI 兼容接口)
- **LangChain 0.2+**: Report Agent框架(ReACT模式)
- **OASIS-AI**: 社交媒体模拟框架
- **CAMEL-AI**: Agent 行为模拟

### 数据处理
- **PyMuPDF (fitz)**: PDF 文本提取
- **Pydantic 2.0+**: 数据验证
- **Python-dotenv**: 环境变量管理

### 文件处理
- **Werkzeug 3.0+**: 文件上传处理

---

## 项目结构

```
backend/
├── run.py                      # 启动入口
├── requirements.txt            # Python依赖
├── .env                        # 环境配置(需创建)
├── logs/                       # 日志文件
│   └── YYYY-MM-DD.log
├── uploads/                    # 数据存储
│   ├── projects/               # 项目数据
│   │   └── proj_xxx/
│   │       ├── project.json    # 项目元数据
│   │       ├── files/          # 上传的文件
│   │       └── extracted_text.txt  # 提取的文本
│   ├── reports/                # 报告数据
│   │   └── report_xxx/
│   │       ├── report_xxx.json # 报告元数据
│   │       └── report_xxx.md   # Markdown报告
│   └── simulations/            # 模拟数据
│       └── sim_xxx/
│           ├── state.json      # 模拟状态
│           ├── simulation_config.json  # 模拟配置
│           ├── reddit_profiles.json    # Reddit人设
│           ├── twitter_profiles.csv    # Twitter人设
│           ├── run_state.json  # 运行状态
│           ├── simulation.log  # 主日志
│           ├── twitter/        # Twitter数据
│           │   ├── actions.jsonl
│           │   └── twitter_simulation.db
│           └── reddit/         # Reddit数据
│               ├── actions.jsonl
│               └── reddit_simulation.db
├── scripts/                    # 模拟运行脚本
│   ├── run_twitter_simulation.py
│   ├── run_reddit_simulation.py
│   ├── run_parallel_simulation.py
│   └── action_logger.py
└── app/
    ├── __init__.py            # Flask应用工厂
    ├── config.py              # 配置管理
    ├── api/                   # API路由
    │   ├── __init__.py
    │   ├── graph.py           # 图谱相关接口
    │   ├── simulation.py      # 模拟相关接口
    │   └── report.py          # 报告相关接口
    ├── models/                # 数据模型
    │   ├── __init__.py
    │   ├── project.py         # 项目模型
    │   └── task.py            # 任务模型
    ├── services/              # 业务服务
    │   ├── __init__.py
    │   ├── ontology_generator.py          # 本体生成
    │   ├── graph_builder.py               # 图谱构建
    │   ├── text_processor.py              # 文本处理
    │   ├── zep_entity_reader.py           # 实体读取
    │   ├── zep_tools.py                   # Zep检索工具服务
    │   ├── oasis_profile_generator.py     # 人设生成
    │   ├── simulation_config_generator.py # 配置生成
    │   ├── simulation_manager.py          # 模拟管理
    │   ├── simulation_runner.py           # 模拟运行
    │   ├── simulation_ipc.py              # 模拟IPC通信（Interview功能）
    │   ├── zep_graph_memory_updater.py    # 图谱记忆动态更新
    │   └── report_agent.py                # 报告生成Agent（ReACT模式）
    └── utils/                 # 工具类
        ├── __init__.py
        ├── file_parser.py     # 文件解析
        ├── llm_client.py      # LLM客户端
        ├── logger.py          # 日志配置
        └── retry.py           # 重试机制
```

---

## 核心功能模块

### 1. 图谱构建模块

**功能**: 从文档构建知识图谱

**流程**:
1. 上传文档(PDF/TXT/MD)
2. 提取文本内容
3. LLM分析生成本体(实体类型+关系类型)
4. 文本分块(chunk_size=500, overlap=50)
5. 调用 Zep API 构建图谱
6. 等待 Zep 处理完成
7. 返回图谱ID和统计信息

**核心服务**:
- `OntologyGenerator`: 本体生成
- `GraphBuilderService`: 图谱构建
- `TextProcessor`: 文本处理

### 2. 模拟准备模块

**功能**: 准备舆论模拟所需的所有数据

**流程**:
1. 创建模拟(指定project_id和graph_id)
2. 从 Zep 图谱读取并过滤实体
3. 为每个实体生成 OASIS Agent Profile(支持并行)
4. 使用 LLM 智能生成模拟配置(时间/活跃度/事件)
5. 保存配置文件和人设文件

**核心服务**:
- `ZepEntityReader`: 实体读取与过滤
- `OasisProfileGenerator`: Agent人设生成
- `SimulationConfigGenerator`: 模拟配置生成
- `SimulationManager`: 模拟管理

### 3. 模拟运行模块

**功能**: 运行 Twitter/Reddit 双平台舆论模拟

**流程**:
1. 检查模拟准备状态
2. 启动 OASIS 模拟进程(subprocess)
3. 监控进程运行状态
4. 解析动作日志(actions.jsonl)
5. (可选)将Agent活动实时更新到Zep图谱
6. 实时更新运行状态
7. 模拟完成后进入等待命令模式
8. 支持停止/暂停/恢复

**核心服务**:
- `SimulationRunner`: 模拟运行器
- `ZepGraphMemoryUpdater`: 图谱记忆动态更新器

### 4. Agent采访(Interview)模块

**功能**: 在模拟完成后对Agent进行采访

**特点**:
- **模拟状态持久化**: 模拟完成后环境不立即关闭，进入等待命令模式
- **IPC通信机制**: 通过文件系统在Flask后端和模拟脚本之间通信
- **单个采访**: 对指定Agent提问并获取回答
- **批量采访**: 同时对多个Agent提不同问题
- **全局采访**: 使用相同问题采访所有Agent
- **采访历史**: 从数据库读取所有Interview记录

**核心服务**:
- `SimulationIPCClient`: IPC客户端（Flask端使用）
- `SimulationIPCServer`: IPC服务器（模拟脚本端使用）

### 5. Report Agent模块（报告生成）

**功能**: 模拟完成后自动生成分析报告，支持与用户对话

**特点**:
- **ReACT模式**: Reasoning + Acting，多轮思考与工具调用
- **大纲规划**: LLM分析模拟需求，自动规划报告目录结构
- **分段生成**: 逐章节生成，每章节可多次调用Zep检索工具
- **Markdown输出**: 生成专业的Markdown格式报告
- **对话功能**: 报告完成后可与Report Agent对话，自主调用工具回答问题

**工具（MCP封装）**:
- `search_graph`: 图谱语义搜索
- `get_graph_statistics`: 获取图谱统计信息
- `get_entity_summary`: 获取实体关系摘要
- `get_simulation_context`: 获取模拟上下文
- `get_entities_by_type`: 按类型获取实体

**核心服务**:
- `ZepToolsService`: Zep检索工具封装
- `ReportAgent`: 报告生成Agent（ReACT模式）
- `ReportManager`: 报告持久化管理

**工作原理**:
```
┌─────────────────────────────────────────────────────────────┐
│                    Report Agent (ReACT)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 规划阶段                                                 │
│     ┌─────────────────────────────────────────────────┐     │
│     │ LLM分析模拟需求 → 获取图谱上下文 → 生成报告大纲  │     │
│     └─────────────────────────────────────────────────┘     │
│                                                              │
│  2. 生成阶段 (每章节)                                        │
│     ┌─────────────────────────────────────────────────┐     │
│     │  Thought → Action → Observation → ... → Final   │     │
│     │    ↓         ↓          ↓                       │     │
│     │  分析需求  调用工具   分析结果      生成内容     │     │
│     └─────────────────────────────────────────────────┘     │
│                                                              │
│  3. 输出阶段                                                 │
│     ┌─────────────────────────────────────────────────┐     │
│     │ 组装章节 → 生成Markdown → 保存JSON/MD文件       │     │
│     └─────────────────────────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**工作原理**:
```
Flask后端                    模拟脚本
    │                           │
    │ 写入命令文件               │
    │ ─────────────────────────→│
    │                           │ 轮询命令目录
    │                           │ 执行Interview
    │                           │ 写入响应文件
    │←───────────────────────── │
    │ 读取响应文件               │
    │                           │
```

---

## API接口文档

### 图谱管理接口

#### 1. 生成本体

**接口**: `POST /api/graph/ontology/generate`

**请求类型**: `multipart/form-data`

**请求参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | File[] | 是 | 上传的文档(PDF/MD/TXT) |
| simulation_requirement | String | 是 | 模拟需求描述 |
| project_name | String | 否 | 项目名称 |
| additional_context | String | 否 | 额外说明 |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "project_id": "proj_33469c670f56",
    "project_name": "学术不端事件模拟",
    "ontology": {
      "entity_types": [
        {
          "name": "Student",
          "description": "Students involved in the event",
          "attributes": [
            {"name": "full_name", "type": "text", "description": "Student full name"},
            {"name": "major", "type": "text", "description": "Major field"}
          ],
          "examples": ["张三", "李四"]
        },
        {
          "name": "Professor",
          "description": "Faculty members",
          "attributes": [...]
        },
        ...
        {
          "name": "Person",
          "description": "Any individual person not fitting other specific person types",
          "attributes": [...]
        },
        {
          "name": "Organization",
          "description": "Any organization not fitting other specific types",
          "attributes": [...]
        }
      ],
      "edge_types": [
        {
          "name": "STUDIES_AT",
          "description": "Student studies at university",
          "source_targets": [
            {"source": "Student", "target": "University"}
          ],
          "attributes": []
        },
        ...
      ]
    },
    "analysis_summary": "文档涉及学术不端事件...",
    "files": [
      {"filename": "document.pdf", "size": 102400}
    ],
    "total_text_length": 12345
  }
}
```

**说明**:
- 本体设计必须包含10个实体类型,最后2个为兜底类型(`Person`和`Organization`)
- 实体类型必须是现实中可以发声的主体
- 属性名不能使用保留字(`name`, `uuid`, `group_id`, `created_at`, `summary`)

---

#### 2. 构建图谱

**接口**: `POST /api/graph/build`

**请求类型**: `application/json`

**请求参数**:
```json
{
  "project_id": "proj_33469c670f56",
  "graph_name": "学术不端事件图谱",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "force": false
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| project_id | String | 是 | - | 项目ID(来自接口1) |
| graph_name | String | 否 | 项目名称 | 图谱名称 |
| chunk_size | Integer | 否 | 500 | 文本块大小 |
| chunk_overlap | Integer | 否 | 50 | 块重叠大小 |
| force | Boolean | 否 | false | 强制重新构建 |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "project_id": "proj_33469c670f56",
    "task_id": "a1b2c3d4-e5f6-...",
    "message": "图谱构建任务已启动,请通过 /task/{task_id} 查询进度"
  }
}
```

**异步任务**: 此接口立即返回task_id,实际构建在后台进行

---

#### 3. 查询任务状态

**接口**: `GET /api/graph/task/{task_id}`

**返回示例**:
```json
{
  "success": true,
  "data": {
    "task_id": "a1b2c3d4-e5f6-...",
    "task_type": "graph_build",
    "status": "processing",
    "created_at": "2025-12-02T10:00:00",
    "updated_at": "2025-12-02T10:05:00",
    "progress": 45,
    "message": "Zep处理中... 10/30 完成",
    "result": null,
    "error": null,
    "metadata": {
      "project_id": "proj_33469c670f56"
    }
  }
}
```

**状态值**:
- `pending`: 等待中
- `processing`: 处理中
- `completed`: 已完成
- `failed`: 失败

---

#### 4. 获取图谱数据

**接口**: `GET /api/graph/data/{graph_id}`

**返回示例**:
```json
{
  "success": true,
  "data": {
    "graph_id": "mirofish_abc123",
    "nodes": [
      {
        "uuid": "node-uuid-1",
        "name": "张三",
        "labels": ["Entity", "Student"],
        "summary": "某大学计算机专业学生",
        "attributes": {
          "full_name": "张三",
          "major": "计算机科学"
        }
      },
      ...
    ],
    "edges": [
      {
        "uuid": "edge-uuid-1",
        "name": "STUDIES_AT",
        "fact": "张三就读于某大学",
        "source_node_uuid": "node-uuid-1",
        "target_node_uuid": "node-uuid-2",
        "attributes": {}
      },
      ...
    ],
    "node_count": 50,
    "edge_count": 120
  }
}
```

---

#### 5. 项目管理接口

**获取项目**: `GET /api/graph/project/{project_id}`

**列出项目**: `GET /api/graph/project/list?limit=50`

**删除项目**: `DELETE /api/graph/project/{project_id}`

**重置项目**: `POST /api/graph/project/{project_id}/reset`

---

### 模拟管理接口

#### 1. 创建模拟

**接口**: `POST /api/simulation/create`

**请求参数**:
```json
{
  "project_id": "proj_33469c670f56",
  "graph_id": "mirofish_abc123",
  "enable_twitter": true,
  "enable_reddit": true
}
```

**返回示例**:
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_10b494550540",
    "project_id": "proj_33469c670f56",
    "graph_id": "mirofish_abc123",
    "status": "created",
    "enable_twitter": true,
    "enable_reddit": true,
    "created_at": "2025-12-02T10:00:00"
  }
}
```

---

#### 2. 准备模拟

**接口**: `POST /api/simulation/prepare`

**请求参数**:
```json
{
  "simulation_id": "sim_10b494550540",
  "entity_types": ["Student", "Professor"],
  "use_llm_for_profiles": true,
  "parallel_profile_count": 5,
  "force_regenerate": false
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| entity_types | String[] | 否 | null | 指定实体类型(为空则全部) |
| use_llm_for_profiles | Boolean | 否 | true | 是否用LLM生成详细人设 |
| parallel_profile_count | Integer | 否 | 5 | 并行生成人设数量 |
| force_regenerate | Boolean | 否 | false | 强制重新生成 |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_10b494550540",
    "task_id": "task_xyz789",
    "status": "preparing",
    "message": "准备任务已启动",
    "already_prepared": false
  }
}
```

**特性**:
- 自动检测已完成的准备工作,避免重复生成
- 支持并行生成人设(默认5个并发)
- 支持强制重新生成

---

#### 3. 查询准备进度

**接口**: `POST /api/simulation/prepare/status`

**请求参数**:
```json
{
  "task_id": "task_xyz789",
  "simulation_id": "sim_10b494550540"
}
```

**返回示例**:
```json
{
  "success": true,
  "data": {
    "task_id": "task_xyz789",
    "status": "processing",
    "progress": 45,
    "message": "[2/4] 生成Agent配置: 5/15 - 已完成 Student: 张三",
    "progress_detail": {
      "current_stage": "generating_profiles",
      "current_stage_name": "生成Agent人设",
      "stage_index": 2,
      "total_stages": 4,
      "stage_progress": 33,
      "current_item": 5,
      "total_items": 15,
      "item_description": "已完成 Student: 张三"
    },
    "already_prepared": false
  }
}
```

**进度阶段**:
1. `reading`: 读取图谱实体 (0-20%)
2. `generating_profiles`: 生成Agent人设 (20-70%)
3. `generating_config`: 生成模拟配置 (70-90%)
4. `copying_scripts`: 准备模拟脚本 (90-100%)

---

#### 4. 启动模拟

**接口**: `POST /api/simulation/start`

**请求参数**:
```json
{
  "simulation_id": "sim_10b494550540",
  "platform": "parallel",
  "max_rounds": 100,
  "enable_graph_memory_update": false
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| platform | String | 否 | parallel | 运行平台: twitter/reddit/parallel |
| max_rounds | Integer | 否 | - | 最大模拟轮数，用于截断过长的模拟 |
| enable_graph_memory_update | Boolean | 否 | false | 是否将Agent活动动态更新到Zep图谱 |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_10b494550540",
    "runner_status": "running",
    "process_pid": 12345,
    "twitter_running": true,
    "reddit_running": true,
    "started_at": "2025-12-02T11:00:00",
    "total_rounds": 100,
    "max_rounds_applied": 100,
    "graph_memory_update_enabled": true,
    "graph_id": "mirofish_abc123"
  }
}
```

> **说明**: 
> - `max_rounds_applied` 字段仅在指定了 `max_rounds` 参数时返回
> - `graph_memory_update_enabled` 和 `graph_id` 字段在启用图谱记忆更新时返回

**图谱记忆更新功能说明**:

启用 `enable_graph_memory_update` 后:
- 模拟中所有Agent的活动(发帖、评论、点赞、转发等)会实时更新到Zep图谱
- 每条活动单独发送,确保Zep能正确解析实体和关系
- 活动会被转换为自然语言描述,例如:`张三: 发布了一条帖子：「...」`
- Zep会自动从文本中提取实体和关系,丰富图谱知识
- 需要项目已构建有效的图谱(graph_id)

---

#### 5. 停止模拟

**接口**: `POST /api/simulation/stop`

**请求参数**:
```json
{
  "simulation_id": "sim_10b494550540"
}
```

**返回示例**:
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_10b494550540",
    "runner_status": "stopped",
    "completed_at": "2025-12-02T12:00:00"
  }
}
```

---

### Interview 采访接口

> **注意**: 所有Interview接口的参数都通过请求体(JSON)传递，包括simulation_id。
> 
> **双平台模式说明**: 当不指定`platform`参数时，双平台模拟会同时采访两个平台并返回整合结果。
>
> **Prompt自动优化**: 系统会自动在用户提供的prompt前添加说明前缀，避免Agent调用工具：
> ```
> 原始prompt: "武汉大学发布撤销处分通告后你有什么看法"
> 优化后: "结合你的人设、所有的过往记忆与行动，不调用任何工具直接用文本回复我：武汉大学发布撤销处分通告后你有什么看法"
> ```

#### 1. 采访单个Agent

**接口**: `POST /api/simulation/interview`

**请求参数**:
```json
{
  "simulation_id": "sim_xxxx",
  "agent_id": 0,
  "prompt": "你对这件事有什么看法？",
  "platform": "reddit",
  "timeout": 60
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| agent_id | Integer | 是 | - | Agent ID |
| prompt | String | 是 | - | 采访问题 |
| platform | String | 否 | null | 指定平台(twitter/reddit)，不指定则双平台同时采访 |
| timeout | Integer | 否 | 60 | 超时时间（秒） |

**返回示例（指定单平台）**:
```json
{
  "success": true,
  "data": {
    "success": true,
    "agent_id": 0,
    "prompt": "你对这件事有什么看法？",
    "result": {
      "agent_id": 0,
      "response": "我认为这件事反映了...",
      "platform": "reddit",
      "timestamp": "2025-12-08T10:00:00"
    },
    "timestamp": "2025-12-08T10:00:01"
  }
}
```

**返回示例（不指定platform，双平台模式）**:
```json
{
  "success": true,
  "data": {
    "success": true,
    "agent_id": 0,
    "prompt": "你对这件事有什么看法？",
    "result": {
      "agent_id": 0,
      "prompt": "你对这件事有什么看法？",
      "platforms": {
        "twitter": {
          "agent_id": 0,
          "response": "从Twitter视角来看...",
          "platform": "twitter",
          "timestamp": "2025-12-08T10:00:00"
        },
        "reddit": {
          "agent_id": 0,
          "response": "作为Reddit用户，我认为...",
          "platform": "reddit",
          "timestamp": "2025-12-08T10:00:00"
        }
      }
    },
    "timestamp": "2025-12-08T10:00:01"
  }
}
```

**注意**: 此功能需要模拟环境处于运行状态（完成模拟循环后进入等待命令模式）

---

#### 2. 批量采访多个Agent

**接口**: `POST /api/simulation/interview/batch`

**请求参数**:
```json
{
  "simulation_id": "sim_xxxx",
  "interviews": [
    {"agent_id": 0, "prompt": "你对A有什么看法？", "platform": "twitter"},
    {"agent_id": 1, "prompt": "你对B有什么看法？", "platform": "reddit"},
    {"agent_id": 2, "prompt": "你对C有什么看法？"}
  ],
  "platform": "reddit",
  "timeout": 120
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| interviews | Array | 是 | - | 采访列表，每项包含agent_id、prompt和可选的platform |
| platform | String | 否 | null | 默认平台(被每项的platform覆盖)，不指定则双平台同时采访 |
| timeout | Integer | 否 | 120 | 超时时间（秒） |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "success": true,
    "interviews_count": 3,
    "result": {
      "interviews_count": 6,
      "results": {
        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
        "reddit_1": {"agent_id": 1, "response": "...", "platform": "reddit"},
        "twitter_2": {"agent_id": 2, "response": "...", "platform": "twitter"},
        "reddit_2": {"agent_id": 2, "response": "...", "platform": "reddit"}
      }
    },
    "timestamp": "2025-12-08T10:00:01"
  }
}
```

---

#### 3. 全局采访（采访所有Agent）

**接口**: `POST /api/simulation/interview/all`

**请求参数**:
```json
{
  "simulation_id": "sim_xxxx",
  "prompt": "你对这件事整体有什么看法？",
  "platform": "reddit",
  "timeout": 180
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| prompt | String | 是 | - | 采访问题（所有Agent使用相同问题） |
| platform | String | 否 | null | 指定平台(twitter/reddit)，不指定则双平台同时采访 |
| timeout | Integer | 否 | 180 | 超时时间（秒） |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "success": true,
    "interviews_count": 50,
    "result": {
      "interviews_count": 100,
      "results": {
        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
        "twitter_1": {"agent_id": 1, "response": "...", "platform": "twitter"},
        "reddit_1": {"agent_id": 1, "response": "...", "platform": "reddit"},
        ...
      }
    },
    "timestamp": "2025-12-08T10:00:01"
  }
}
```

---

#### 4. 获取Interview历史

**接口**: `POST /api/simulation/interview/history`

**请求参数**:
```json
{
  "simulation_id": "sim_xxxx",
  "platform": "reddit",
  "agent_id": 0,
  "limit": 100
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| platform | String | 否 | null | 平台类型（reddit/twitter），不指定则返回两个平台的所有历史 |
| agent_id | Integer | 否 | - | 只获取该Agent的采访历史 |
| limit | Integer | 否 | 100 | 返回数量限制 |

**返回示例（不指定platform，返回双平台历史）**:
```json
{
  "success": true,
  "data": {
    "count": 10,
    "history": [
      {
        "agent_id": 0,
        "response": "我认为...",
        "prompt": "你对这件事有什么看法？",
        "timestamp": "2025-12-08T10:00:02",
        "platform": "twitter"
      },
      {
        "agent_id": 0,
        "response": "从Reddit角度来看...",
        "prompt": "你对这件事有什么看法？",
        "timestamp": "2025-12-08T10:00:01",
        "platform": "reddit"
      },
      ...
    ]
  }
}
```

---

#### 5. 获取模拟环境状态

**接口**: `POST /api/simulation/env-status`

**请求参数**:
```json
{
  "simulation_id": "sim_xxxx"
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_xxxx",
    "env_alive": true,
    "twitter_available": true,
    "reddit_available": true,
    "message": "环境正在运行，可以接收Interview命令"
  }
}
```

---

#### 6. 关闭模拟环境

**接口**: `POST /api/simulation/close-env`

**请求参数**:
```json
{
  "simulation_id": "sim_10b494550540",
  "timeout": 30
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| timeout | Integer | 否 | 30 | 超时时间（秒） |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "success": true,
    "message": "环境关闭命令已发送",
    "result": {"message": "环境即将关闭"},
    "timestamp": "2025-12-08T10:00:01"
  }
}
```

**注意**: 此接口与 `/stop` 不同：
- `/stop`: 强制终止模拟进程
- `/close-env`: 优雅地关闭环境，让模拟进程正常退出

---

### Report 报告接口

> **说明**: 报告生成完成后才能解锁Interview功能。Report Agent使用ReACT模式，可以在对话中自主调用Zep检索工具。

#### 1. 生成报告

**接口**: `POST /api/report/generate`

**请求参数**:
```json
{
  "simulation_id": "sim_xxxx",
  "force_regenerate": false
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| force_regenerate | Boolean | 否 | false | 强制重新生成 |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_xxxx",
    "task_id": "task_xxxx",
    "status": "generating",
    "message": "报告生成任务已启动",
    "already_generated": false
  }
}
```

**如果报告已存在**:
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_xxxx",
    "report_id": "report_xxxx",
    "status": "completed",
    "message": "报告已存在",
    "already_generated": true
  }
}
```

---

#### 2. 查询生成进度

**接口**: `POST /api/report/generate/status`

**请求参数**:
```json
{
  "task_id": "task_xxxx",
  "simulation_id": "sim_xxxx"
}
```

**返回示例**:
```json
{
  "success": true,
  "data": {
    "task_id": "task_xxxx",
    "status": "processing",
    "progress": 45,
    "message": "[generating] 正在生成章节: 关键发现 (3/5)"
  }
}
```

---

#### 3. 获取报告

**接口**: `GET /api/report/{report_id}`

**返回示例**:
```json
{
  "success": true,
  "data": {
    "report_id": "report_xxxx",
    "simulation_id": "sim_xxxx",
    "graph_id": "mirofish_xxxx",
    "simulation_requirement": "模拟武汉大学撤销处分后的舆情走向",
    "status": "completed",
    "outline": {
      "title": "武汉大学撤销处分事件舆情分析报告",
      "summary": "基于模拟结果的全面舆情分析",
      "sections": [
        {"title": "执行摘要", "content": "..."},
        {"title": "模拟背景", "content": "..."},
        {"title": "关键发现", "content": "..."},
        {"title": "舆情分析", "content": "..."},
        {"title": "建议与展望", "content": "..."}
      ]
    },
    "markdown_content": "# 武汉大学撤销处分事件舆情分析报告\n\n...",
    "created_at": "2025-12-09T10:00:00",
    "completed_at": "2025-12-09T10:05:00"
  }
}
```

---

#### 4. 根据模拟ID获取报告

**接口**: `GET /api/report/by-simulation/{simulation_id}`

**返回示例**:
```json
{
  "success": true,
  "data": {...},
  "has_report": true
}
```

---

#### 5. 下载报告

**接口**: `GET /api/report/{report_id}/download`

**返回**: Markdown文件下载

---

#### 6. 与Report Agent对话

**接口**: `POST /api/report/chat`

**请求参数**:
```json
{
  "simulation_id": "sim_xxxx",
  "message": "请详细解释一下舆情的主要趋势",
  "chat_history": [
    {"role": "user", "content": "报告提到了哪些关键人物？"},
    {"role": "assistant", "content": "根据分析，关键人物包括..."}
  ]
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| message | String | 是 | - | 用户消息 |
| chat_history | Array | 否 | [] | 对话历史（用于上下文） |

**返回示例**:
```json
{
  "success": true,
  "data": {
    "response": "根据模拟数据分析，舆情的主要趋势表现为...\n\n1. **初期阶段**：...\n2. **发酵阶段**：...\n3. **高峰阶段**：...",
    "tool_calls": [
      {"name": "search_graph", "parameters": {"query": "舆情趋势"}},
      {"name": "get_graph_statistics", "parameters": {}}
    ],
    "sources": []
  }
}
```

---

#### 7. 检查报告状态

**接口**: `GET /api/report/check/{simulation_id}`

**用途**: 判断是否解锁Interview功能

**返回示例**:
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_xxxx",
    "has_report": true,
    "report_status": "completed",
    "report_id": "report_xxxx",
    "interview_unlocked": true
  }
}
```

---

#### 8. 列出所有报告

**接口**: `GET /api/report/list?simulation_id=sim_xxxx&limit=50`

**返回示例**:
```json
{
  "success": true,
  "data": [...],
  "count": 5
}
```

---

#### 9. 删除报告

**接口**: `DELETE /api/report/{report_id}`

---

#### 10. 工具调试接口

**图谱搜索**: `POST /api/report/tools/search`
```json
{
  "graph_id": "mirofish_xxxx",
  "query": "舆情走向",
  "limit": 10
}
```

**图谱统计**: `POST /api/report/tools/statistics`
```json
{
  "graph_id": "mirofish_xxxx"
}
```

---

#### 6. 获取运行状态

**接口**: `GET /api/simulation/{simulation_id}/run-status`

**返回示例**:
```json
{
  "success": true,
  "data": {
    "simulation_id": "sim_10b494550540",
    "runner_status": "running",
    "current_round": 5,
    "total_rounds": 144,
    "progress_percent": 3.5,
    "simulated_hours": 2,
    "total_simulation_hours": 72,
    "twitter_running": true,
    "reddit_running": true,
    "twitter_actions_count": 150,
    "reddit_actions_count": 200,
    "total_actions_count": 350,
    "started_at": "2025-12-02T11:00:00",
    "updated_at": "2025-12-02T11:30:00"
  }
}
```

---

#### 7. 获取详细状态(含最近动作)

**接口**: `GET /api/simulation/{simulation_id}/run-status/detail`

**返回示例**:
```json
{
  "success": true,
  "data": {
    ... (基本状态同上) ...,
    "recent_actions": [
      {
        "round_num": 5,
        "timestamp": "2025-12-02T11:30:15",
        "platform": "twitter",
        "agent_id": 3,
        "agent_name": "张三_123",
        "action_type": "CREATE_POST",
        "action_args": {
          "content": "对学术不端事件的看法..."
        },
        "result": "post_id_123",
        "success": true
      },
      ...
    ]
  }
}
```

---

#### 8. 其他接口

**获取实体列表**: `GET /api/simulation/entities/{graph_id}`

**获取模拟配置**: `GET /api/simulation/{simulation_id}/config`

**获取Agent人设**: `GET /api/simulation/{simulation_id}/profiles?platform=reddit`

**获取动作历史**: `GET /api/simulation/{simulation_id}/actions?limit=100&platform=twitter`

**获取时间线**: `GET /api/simulation/{simulation_id}/timeline?start_round=0&end_round=10`

**获取Agent统计**: `GET /api/simulation/{simulation_id}/agent-stats`

**获取帖子**: `GET /api/simulation/{simulation_id}/posts?platform=reddit&limit=50`

**获取评论**: `GET /api/simulation/{simulation_id}/comments?post_id=123`

---

## 数据模型

### 1. Project (项目模型)

**文件**: `app/models/project.py`

**字段**:
```python
project_id: str              # 项目ID (proj_xxx)
name: str                    # 项目名称
status: ProjectStatus        # 状态
created_at: str              # 创建时间
updated_at: str              # 更新时间

# 文件信息
files: List[Dict]            # 上传的文件列表
total_text_length: int       # 文本总长度

# 本体信息
ontology: Dict               # 实体类型和关系类型
analysis_summary: str        # 分析摘要

# 图谱信息
graph_id: str                # Zep图谱ID
graph_build_task_id: str     # 构建任务ID

# 配置
simulation_requirement: str  # 模拟需求
chunk_size: int              # 文本块大小
chunk_overlap: int           # 块重叠大小

# 错误信息
error: str                   # 错误描述
```

**状态枚举**:
```python
CREATED = "created"                      # 已创建
ONTOLOGY_GENERATED = "ontology_generated"  # 本体已生成
GRAPH_BUILDING = "graph_building"        # 图谱构建中
GRAPH_COMPLETED = "graph_completed"      # 图谱已完成
FAILED = "failed"                        # 失败
```

---

### 2. Task (任务模型)

**文件**: `app/models/task.py`

**字段**:
```python
task_id: str                 # 任务ID (UUID)
task_type: str               # 任务类型
status: TaskStatus           # 状态
created_at: datetime         # 创建时间
updated_at: datetime         # 更新时间
progress: int                # 进度 (0-100)
message: str                 # 状态消息
result: Dict                 # 任务结果
error: str                   # 错误信息
metadata: Dict               # 元数据
progress_detail: Dict        # 详细进度
```

**状态枚举**:
```python
PENDING = "pending"          # 等待中
PROCESSING = "processing"    # 处理中
COMPLETED = "completed"      # 已完成
FAILED = "failed"            # 失败
```

---

### 3. SimulationState (模拟状态)

**文件**: `app/services/simulation_manager.py`

**字段**:
```python
simulation_id: str           # 模拟ID (sim_xxx)
project_id: str              # 项目ID
graph_id: str                # 图谱ID
enable_twitter: bool         # 启用Twitter
enable_reddit: bool          # 启用Reddit
status: SimulationStatus     # 状态
entities_count: int          # 实体数量
profiles_count: int          # 人设数量
entity_types: List[str]      # 实体类型列表
config_generated: bool       # 配置已生成
config_reasoning: str        # 配置推理说明
current_round: int           # 当前轮次
twitter_status: str          # Twitter状态
reddit_status: str           # Reddit状态
created_at: str              # 创建时间
updated_at: str              # 更新时间
error: str                   # 错误信息
```

---

### 4. EntityNode (实体节点)

**文件**: `app/services/zep_entity_reader.py`

**字段**:
```python
uuid: str                    # 实体UUID
name: str                    # 实体名称
labels: List[str]            # 标签列表
summary: str                 # 摘要
attributes: Dict             # 属性字典
related_edges: List[Dict]    # 相关边信息
related_nodes: List[Dict]    # 关联节点信息
```

---

### 5. OasisAgentProfile (Agent人设)

**文件**: `app/services/oasis_profile_generator.py`

**字段**:
```python
user_id: int                 # 用户ID
user_name: str               # 用户名
name: str                    # 真实姓名
bio: str                     # 简介 (200字)
persona: str                 # 详细人设 (2000字)
karma: int                   # Reddit积分
friend_count: int            # Twitter好友数
follower_count: int          # 粉丝数
statuses_count: int          # 发帖数
age: int                     # 年龄
gender: str                  # 性别 (male/female/other)
mbti: str                    # MBTI类型
country: str                 # 国家
profession: str              # 职业
interested_topics: List[str] # 兴趣话题
source_entity_uuid: str      # 来源实体UUID
source_entity_type: str      # 来源实体类型
created_at: str              # 创建时间
```

---

### 6. Report (报告模型)

**文件**: `app/services/report_agent.py`

**字段**:
```python
report_id: str                  # 报告ID (report_xxx)
simulation_id: str              # 模拟ID
graph_id: str                   # 图谱ID
simulation_requirement: str     # 模拟需求
status: ReportStatus            # 状态
outline: ReportOutline          # 报告大纲
markdown_content: str           # Markdown内容
created_at: str                 # 创建时间
completed_at: str               # 完成时间
error: str                      # 错误信息
```

**状态枚举**:
```python
PENDING = "pending"             # 等待中
PLANNING = "planning"           # 规划大纲中
GENERATING = "generating"       # 生成内容中
COMPLETED = "completed"         # 已完成
FAILED = "failed"               # 失败
```

**ReportOutline字段**:
```python
title: str                      # 报告标题
summary: str                    # 报告摘要
sections: List[ReportSection]   # 章节列表
```

**ReportSection字段**:
```python
title: str                      # 章节标题
content: str                    # 章节内容
subsections: List[ReportSection] # 子章节
```

---

### 7. SimulationParameters (模拟参数)

**文件**: `app/services/simulation_config_generator.py`

**字段**:
```python
simulation_id: str           # 模拟ID
project_id: str              # 项目ID
graph_id: str                # 图谱ID
simulation_requirement: str  # 模拟需求

# 时间配置
time_config: TimeSimulationConfig
  ├── total_simulation_hours: int        # 总时长(小时)
  ├── minutes_per_round: int             # 每轮分钟数
  ├── agents_per_hour_min: int           # 每小时最少激活Agent数
  ├── agents_per_hour_max: int           # 每小时最多激活Agent数
  ├── peak_hours: List[int]              # 高峰时段 [19,20,21,22]
  ├── off_peak_hours: List[int]          # 低谷时段 [0,1,2,3,4,5]
  ├── morning_hours: List[int]           # 早间时段 [6,7,8]
  ├── work_hours: List[int]              # 工作时段 [9-18]
  ├── peak_activity_multiplier: float    # 高峰活跃度系数 1.5
  ├── off_peak_activity_multiplier: float # 低谷活跃度系数 0.05
  ├── morning_activity_multiplier: float # 早间活跃度系数 0.4
  └── work_activity_multiplier: float    # 工作时段活跃度系数 0.7

# Agent配置列表
agent_configs: List[AgentActivityConfig]
  ├── agent_id: int              # Agent ID
  ├── entity_uuid: str           # 实体UUID
  ├── entity_name: str           # 实体名称
  ├── entity_type: str           # 实体类型
  ├── activity_level: float      # 活跃度 (0.0-1.0)
  ├── posts_per_hour: float      # 每小时发帖数
  ├── comments_per_hour: float   # 每小时评论数
  ├── active_hours: List[int]    # 活跃时间段
  ├── response_delay_min: int    # 最小响应延迟(分钟)
  ├── response_delay_max: int    # 最大响应延迟(分钟)
  ├── sentiment_bias: float      # 情感倾向 (-1.0到1.0)
  ├── stance: str                # 立场 (supportive/opposing/neutral/observer)
  └── influence_weight: float    # 影响力权重

# 事件配置
event_config: EventConfig
  ├── initial_posts: List[Dict]  # 初始帖子
  ├── scheduled_events: List[Dict] # 定时事件
  ├── hot_topics: List[str]      # 热点话题
  └── narrative_direction: str   # 舆论方向

# 平台配置
twitter_config: PlatformConfig
reddit_config: PlatformConfig
  ├── platform: str              # 平台名称
  ├── recency_weight: float      # 时间新鲜度权重
  ├── popularity_weight: float   # 热度权重
  ├── relevance_weight: float    # 相关性权重
  ├── viral_threshold: int       # 病毒传播阈值
  └── echo_chamber_strength: float # 回声室效应强度

# LLM配置
llm_model: str               # LLM模型名称
llm_base_url: str            # LLM API地址
generated_at: str            # 生成时间
generation_reasoning: str    # LLM推理说明
```

---

## 服务层详解

### 1. OntologyGenerator (本体生成器)

**文件**: `app/services/ontology_generator.py`

**功能**: 使用LLM分析文档内容,生成适合舆论模拟的实体类型和关系类型

**核心方法**:
```python
def generate(
    document_texts: List[str],
    simulation_requirement: str,
    additional_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    生成本体定义
    
    Returns:
        {
            "entity_types": [...],  # 10个实体类型(最后2个为Person和Organization)
            "edge_types": [...],     # 6-10个关系类型
            "analysis_summary": "..." # 分析摘要
        }
    """
```

**设计原则**:
- 必须返回**10个实体类型**,最后2个为兜底类型
- 实体必须是现实中可以发声的主体(人/组织)
- 属性名不能使用Zep保留字
- 关系类型要反映社交媒体互动

**LLM提示词要点**:
- 系统角色: 知识图谱本体设计专家
- 任务背景: 社交媒体舆论模拟
- 输出格式: 严格的JSON结构
- 实体类型层次: 具体类型(8个) + 兜底类型(2个)

---

### 2. GraphBuilderService (图谱构建服务)

**文件**: `app/services/graph_builder.py`

**功能**: 调用Zep API构建知识图谱

**核心方法**:
```python
def create_graph(name: str) -> str:
    """创建Zep图谱"""

def set_ontology(graph_id: str, ontology: Dict):
    """设置图谱本体(动态创建Pydantic类)"""

def add_text_batches(
    graph_id: str, 
    chunks: List[str], 
    batch_size: int = 3,
    progress_callback: Optional[Callable] = None
) -> List[str]:
    """分批添加文本,返回episode UUIDs"""

def _wait_for_episodes(
    episode_uuids: List[str],
    progress_callback: Optional[Callable] = None,
    timeout: int = 600
):
    """等待所有episode处理完成"""

def get_graph_data(graph_id: str) -> Dict:
    """获取完整图谱数据(节点和边)"""
```

**关键技术点**:
1. **动态类创建**: 根据本体定义动态创建Pydantic类
2. **批量上传**: 避免一次性提交大量数据
3. **异步等待**: 轮询episode的`processed`状态
4. **容错重试**: 所有API调用带重试机制

---

### 3. ZepEntityReader (实体读取器)

**文件**: `app/services/zep_entity_reader.py`

**功能**: 从Zep图谱读取并过滤实体

**核心方法**:
```python
def get_all_nodes(graph_id: str) -> List[Dict]:
    """获取所有节点(带重试)"""

def get_all_edges(graph_id: str) -> List[Dict]:
    """获取所有边(带重试)"""

def filter_defined_entities(
    graph_id: str,
    defined_entity_types: Optional[List[str]] = None,
    enrich_with_edges: bool = True
) -> FilteredEntities:
    """
    筛选符合预定义类型的实体
    
    筛选逻辑:
    - 只保留Labels中包含除"Entity"和"Node"外的自定义标签的节点
    - 如果指定了entity_types,只保留匹配的类型
    - 可选:获取每个实体的相关边和关联节点
    """

def get_entity_with_context(
    graph_id: str, 
    entity_uuid: str
) -> Optional[EntityNode]:
    """获取单个实体及其完整上下文"""
```

**容错机制**:
- 所有Zep API调用带**3次重试**
- 使用指数退避策略
- 详细的日志记录

---

### 4. OasisProfileGenerator (人设生成器)

**文件**: `app/services/oasis_profile_generator.py`

**功能**: 将图谱实体转换为OASIS Agent Profile

**核心方法**:
```python
def generate_profile_from_entity(
    entity: EntityNode, 
    user_id: int,
    use_llm: bool = True
) -> OasisAgentProfile:
    """
    从实体生成Agent人设
    
    步骤:
    1. 构建实体上下文(属性+边+关联节点+Zep检索)
    2. 使用LLM生成详细人设(2000字persona)
    3. 返回OasisAgentProfile对象
    """

def generate_profiles_from_entities(
    entities: List[EntityNode],
    use_llm: bool = True,
    progress_callback: Optional[callable] = None,
    graph_id: Optional[str] = None,
    parallel_count: int = 5
) -> List[OasisAgentProfile]:
    """
    批量生成人设(支持并行)
    
    特性:
    - 并行生成(默认5个并发)
    - Zep混合检索增强上下文
    - 区分个人实体和机构实体
    - 容错处理(失败则使用规则生成)
    """
```

**LLM提示词设计**:
- **个人实体**: 生成2000字详细人设(基本信息+背景+性格+社交行为+立场观点+个人记忆)
- **机构实体**: 生成官方账号设定(机构信息+账号定位+发言风格+发布内容+立场态度+机构记忆)
- **输出格式**: JSON (bio, persona, age, gender, mbti, country, profession, interested_topics)

**容错措施**:
1. LLM调用失败:最多重试3次
2. JSON解析失败:尝试修复JSON
3. 完全失败:使用规则生成基础人设

---

### 5. SimulationConfigGenerator (配置生成器)

**文件**: `app/services/simulation_config_generator.py`

**功能**: 使用LLM智能生成模拟配置参数

**核心方法**:
```python
def generate_config(
    simulation_id: str,
    project_id: str,
    graph_id: str,
    simulation_requirement: str,
    document_text: str,
    entities: List[EntityNode],
    enable_twitter: bool = True,
    enable_reddit: bool = True,
    progress_callback: Optional[Callable] = None,
) -> SimulationParameters:
    """
    智能生成完整模拟配置
    
    分步生成策略(避免一次性生成过长):
    1. 生成时间配置(符合中国人作息)
    2. 生成事件配置(热点话题+初始帖子)
    3. 分批生成Agent配置(每批15个)
    4. 生成平台配置
    """
```

**时间配置特点**:
- **高峰时段**: 19-22点(活跃度系数1.5)
- **低谷时段**: 0-5点(活跃度系数0.05)
- **早间时段**: 6-8点(活跃度系数0.4)
- **工作时段**: 9-18点(活跃度系数0.7)

**Agent配置规则**:
- **官方机构**: 活跃度低(0.1-0.3),工作时间活动,响应慢,影响力高(2.5-3.0)
- **媒体**: 活跃度中(0.4-0.6),全天活动,响应快,影响力高(2.0-2.5)
- **个人/学生**: 活跃度高(0.6-0.9),晚间活动,响应快,影响力低(0.8-1.2)
- **专家/教授**: 活跃度中(0.4-0.6),工作+晚间,影响力中高(1.5-2.0)

---

### 6. SimulationManager (模拟管理器)

**文件**: `app/services/simulation_manager.py`

**功能**: 管理模拟的完整生命周期

**核心方法**:
```python
def create_simulation(
    project_id: str,
    graph_id: str,
    enable_twitter: bool = True,
    enable_reddit: bool = True,
) -> SimulationState:
    """创建新模拟"""

def prepare_simulation(
    simulation_id: str,
    simulation_requirement: str,
    document_text: str,
    defined_entity_types: Optional[List[str]] = None,
    use_llm_for_profiles: bool = True,
    progress_callback: Optional[callable] = None,
    parallel_profile_count: int = 3
) -> SimulationState:
    """
    准备模拟环境(全程自动化)
    
    步骤:
    1. 读取并过滤图谱实体
    2. 并行生成Agent人设(带Zep检索增强)
    3. LLM智能生成模拟配置
    4. 保存配置和人设文件
    """

def get_simulation(simulation_id: str) -> Optional[SimulationState]:
    """获取模拟状态"""

def list_simulations(project_id: Optional[str] = None) -> List[SimulationState]:
    """列出所有模拟"""
```

**数据存储**:
```
uploads/simulations/sim_xxx/
├── state.json                  # 模拟状态
├── simulation_config.json      # 模拟配置(LLM生成)
├── reddit_profiles.json        # Reddit人设(JSON格式)
├── twitter_profiles.csv        # Twitter人设(CSV格式)
├── run_state.json              # 运行状态
├── simulation.log              # 主日志
├── twitter/
│   ├── actions.jsonl           # Twitter动作日志
│   └── twitter_simulation.db   # Twitter数据库
└── reddit/
    ├── actions.jsonl           # Reddit动作日志
    └── reddit_simulation.db    # Reddit数据库
```

---

### 7. SimulationRunner (模拟运行器)

**文件**: `app/services/simulation_runner.py`

**功能**: 在后台运行OASIS模拟并实时监控

**核心方法**:
```python
@classmethod
def start_simulation(
    cls,
    simulation_id: str,
    platform: str = "parallel"
) -> SimulationRunState:
    """
    启动模拟
    
    步骤:
    1. 启动模拟进程(subprocess)
    2. 创建监控线程
    3. 解析动作日志
    4. 实时更新状态
    """

@classmethod
def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
    """
    停止模拟
    
    使用进程组终止(确保子进程也被终止)
    """

@classmethod
def get_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
    """获取运行状态"""

@classmethod
def get_actions(
    cls,
    simulation_id: str,
    limit: int = 100,
    offset: int = 0,
    platform: Optional[str] = None,
    agent_id: Optional[int] = None,
    round_num: Optional[int] = None
) -> List[AgentAction]:
    """获取动作历史(支持过滤)"""

@classmethod
def cleanup_all_simulations(cls):
    """清理所有运行中的模拟进程(服务器关闭时调用)"""
```

**进程管理**:
- 使用`subprocess.Popen`启动模拟脚本
- 使用`start_new_session=True`创建新进程组
- 使用`os.killpg`终止整个进程组
- 支持优雅关闭(SIGTERM)和强制终止(SIGKILL)

**日志解析**:
- 实时读取`twitter/actions.jsonl`和`reddit/actions.jsonl`
- 解析每个Agent的动作记录
- 更新运行状态和进度
- 保存最近50个动作用于前端展示

---

### 8. ZepGraphMemoryUpdater (图谱记忆更新器)

**文件**: `app/services/zep_graph_memory_updater.py`

**功能**: 将模拟中的Agent活动动态更新到Zep图谱

**核心类**:

```python
class AgentActivity:
    """Agent活动记录"""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict
    round_num: int
    timestamp: str
    
    def to_episode_text(self) -> str:
        """
        将活动转换为自然语言描述(不添加模拟前缀)
        
        示例输出:
        - "张三: 发布了一条帖子：「官方声明：...」"
        - "李四: 在帖子#5下评论道：「我认为...」"
        - "王五: 引用帖子#3并评论：「同意！」"
        """
```

```python
class ZepGraphMemoryUpdater:
    """
    图谱记忆更新器
    
    特性:
    - 逐条发送活动到Zep,确保图谱正确解析
    - 后台线程异步处理,不阻塞主模拟流程
    - 带重试的API调用(MAX_RETRIES=3)
    - 自动跳过DO_NOTHING类型的活动
    - 发送间隔控制(SEND_INTERVAL=0.5秒)
    """
    
    def start(self):
        """启动后台工作线程"""
    
    def stop(self):
        """停止并发送剩余活动"""
    
    def add_activity(self, activity: AgentActivity):
        """添加活动到队列"""
    
    def add_activity_from_dict(self, data: Dict, platform: str):
        """从动作日志字典添加活动"""
    
    def get_stats(self) -> Dict:
        """获取统计信息(total_activities, total_sent, failed_count等)"""
```

```python
class ZepGraphMemoryManager:
    """
    管理多个模拟的更新器实例
    """
    
    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> ZepGraphMemoryUpdater:
        """为模拟创建并启动更新器"""
    
    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        """获取模拟的更新器"""
    
    @classmethod
    def stop_updater(cls, simulation_id: str):
        """停止并移除模拟的更新器"""
    
    @classmethod
    def stop_all(cls):
        """停止所有更新器(服务器关闭时调用)"""
```

**活动类型转换**:

| action_type | 转换后的描述 |
|-------------|-------------|
| CREATE_POST | 发布了一条帖子：「{content}」 |
| LIKE_POST | 点赞了帖子#{post_id} |
| DISLIKE_POST | 踩了帖子#{post_id} |
| REPOST | 转发了帖子#{post_id} |
| QUOTE_POST | 引用帖子#{quoted_id}并评论：「{content}」 |
| FOLLOW | 关注了用户#{user_id} |
| CREATE_COMMENT | 在帖子#{post_id}下评论道：「{content}」 |
| LIKE_COMMENT | 点赞了评论#{comment_id} |
| SEARCH_POSTS | 搜索了「{query}」 |
| MUTE | 屏蔽了用户#{user_id} |

**使用示例**:

```python
# 在启动模拟时启用图谱记忆更新
POST /api/simulation/start
{
    "simulation_id": "sim_xxx",
    "enable_graph_memory_update": true
}
```

启用后,模拟中的活动会被逐条转换为自然语言描述并发送到Zep:

```
上级: 发布了一条帖子：「官方声明：经复核并结合司法判决，校方决定撤销对肖某某的处分。学校向当事人致以正式歉意...」
全国顶尖新闻传播学院的大学: 发布了一条帖子：「武汉大学官方发布：学校已决定撤销此前对当事人的处分...」
全国考生: 引用帖子#5并评论
教师代表: 在帖子#2下评论道：「此事暴露出高校在程序正义上的问题...」
```

每条活动单独发送,确保Zep能正确从文本中提取实体(如人名、机构名)和关系,丰富图谱知识。

---

### 9. SimulationIPCClient/Server (IPC通信模块)

**文件**: `app/services/simulation_ipc.py`

**功能**: 实现Flask后端与模拟脚本之间的进程间通信

**核心类**:

```python
class SimulationIPCClient:
    """IPC客户端（Flask端使用）"""
    
    def send_interview(agent_id: int, prompt: str, timeout: float) -> IPCResponse:
        """发送单个Agent采访命令"""
    
    def send_batch_interview(interviews: List[Dict], timeout: float) -> IPCResponse:
        """发送批量采访命令"""
    
    def send_close_env(timeout: float) -> IPCResponse:
        """发送关闭环境命令"""
    
    def check_env_alive() -> bool:
        """检查模拟环境是否存活"""
```

```python
class SimulationIPCServer:
    """IPC服务器（模拟脚本端使用）"""
    
    def poll_commands() -> Optional[IPCCommand]:
        """轮询获取待处理命令"""
    
    def send_response(response: IPCResponse):
        """发送响应"""
```

**命令类型**:

| 命令类型 | 说明 |
|----------|------|
| interview | 单个Agent采访 |
| batch_interview | 批量采访 |
| close_env | 关闭环境 |

**文件结构**:

```
uploads/simulations/sim_xxx/
├── ipc_commands/           # 命令文件目录
│   └── {command_id}.json   # 待处理命令
├── ipc_responses/          # 响应文件目录
│   └── {command_id}.json   # 命令响应
└── env_status.json         # 环境状态文件
```

**使用示例**:

```python
# Flask端发送Interview命令
from app.services import SimulationRunner

# 单个采访
result = SimulationRunner.interview_agent(
    simulation_id="sim_xxx",
    agent_id=0,
    prompt="你对这件事有什么看法？"
)

# 批量采访
result = SimulationRunner.interview_agents_batch(
    simulation_id="sim_xxx",
    interviews=[
        {"agent_id": 0, "prompt": "问题A"},
        {"agent_id": 1, "prompt": "问题B"}
    ]
)

# 全局采访
result = SimulationRunner.interview_all_agents(
    simulation_id="sim_xxx",
    prompt="你认为事件会如何发展？"
)
```

---

### 10. ZepToolsService (Zep检索工具服务)

**文件**: `app/services/zep_tools.py`

**功能**: 封装多种Zep图谱检索工具，供Report Agent调用

**核心方法**:

```python
def search_graph(
    graph_id: str, 
    query: str, 
    limit: int = 10
) -> SearchResult:
    """
    图谱语义搜索
    
    使用混合搜索（语义+BM25）查找相关信息
    返回: facts列表、edges列表、nodes列表
    """

def get_all_nodes(graph_id: str) -> List[NodeInfo]:
    """获取图谱所有节点"""

def get_all_edges(graph_id: str) -> List[EdgeInfo]:
    """获取图谱所有边"""

def get_node_detail(node_uuid: str) -> Optional[NodeInfo]:
    """获取单个节点详情"""

def get_node_edges(node_uuid: str) -> List[EdgeInfo]:
    """获取节点相关的边"""

def get_entities_by_type(
    graph_id: str, 
    entity_type: str
) -> List[NodeInfo]:
    """按类型获取实体"""

def get_entity_summary(
    graph_id: str, 
    entity_name: str
) -> Dict[str, Any]:
    """获取实体关系摘要"""

def get_graph_statistics(graph_id: str) -> Dict[str, Any]:
    """
    获取图谱统计信息
    
    返回:
    - total_nodes: 节点总数
    - total_edges: 边总数
    - entity_types: 实体类型分布
    - relation_types: 关系类型分布
    """

def get_simulation_context(
    graph_id: str,
    simulation_requirement: str,
    limit: int = 30
) -> Dict[str, Any]:
    """
    获取模拟相关上下文
    
    综合搜索与模拟需求相关的所有信息
    """
```

**容错机制**:
- 所有API调用带3次重试
- 指数退避策略
- 搜索失败返回空结果而非抛出异常

---

### 11. ReportAgent (报告生成Agent)

**文件**: `app/services/report_agent.py`

**功能**: 使用ReACT模式生成模拟分析报告

**核心类**:

```python
class ReportAgent:
    """
    Report Agent - 模拟报告生成Agent
    
    采用ReACT（Reasoning + Acting）模式：
    1. 规划阶段：分析模拟需求，规划报告目录结构
    2. 生成阶段：逐章节生成内容，每章节可多次调用工具获取信息
    3. 对话阶段：支持与用户对话，自主调用检索工具
    """
    
    # 配置
    MAX_TOOL_CALLS_PER_SECTION = 5  # 每章节最大工具调用次数
    MAX_REFLECTION_ROUNDS = 2       # 最大反思轮数
```

**核心方法**:

```python
def plan_outline(
    progress_callback: Optional[Callable] = None
) -> ReportOutline:
    """
    规划报告大纲
    
    步骤:
    1. 获取模拟上下文（图谱统计、相关事实）
    2. 使用LLM分析并生成大纲结构
    3. 返回包含章节列表的大纲对象
    """

def _generate_section_react(
    section: ReportSection,
    outline: ReportOutline,
    previous_sections: List[str],
    progress_callback: Optional[Callable] = None
) -> str:
    """
    使用ReACT模式生成单个章节
    
    ReACT循环:
    1. Thought（思考）- 分析需要什么信息
    2. Action（行动）- 调用工具获取信息
    3. Observation（观察）- 分析工具返回结果
    4. 重复直到信息足够或达到最大次数
    5. Final Answer（最终回答）- 生成章节内容
    """

def generate_report(
    progress_callback: Optional[Callable] = None
) -> Report:
    """
    生成完整报告
    
    步骤:
    1. 规划大纲
    2. 逐章节生成（ReACT模式）
    3. 组装Markdown报告
    4. 保存报告文件
    """

def chat(
    message: str,
    chat_history: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    与Report Agent对话
    
    在对话中Agent可以自主调用检索工具来回答问题
    
    Returns:
        {
            "response": "Agent回复",
            "tool_calls": [调用的工具列表],
            "sources": [信息来源]
        }
    """
```

**工具调用格式**:

Agent使用以下格式调用工具：

```
<tool_call>
{"name": "search_graph", "parameters": {"query": "舆情走向", "limit": 10}}
</tool_call>
```

或者：

```
[TOOL_CALL] search_graph(query="舆情走向", limit="10")
```

**报告大纲结构示例**:

```json
{
  "title": "武汉大学撤销处分事件舆情分析报告",
  "summary": "基于模拟结果的全面舆情分析",
  "sections": [
    {
      "title": "执行摘要",
      "description": "简要总结模拟结果和关键发现"
    },
    {
      "title": "模拟背景",
      "description": "描述模拟的初始条件和场景设定"
    },
    {
      "title": "关键发现",
      "description": "分析模拟中的重要发现和趋势"
    },
    {
      "title": "舆情分析",
      "description": "分析舆论走向、情绪变化、关键意见领袖"
    },
    {
      "title": "影响评估",
      "description": "评估事件的影响范围和程度"
    },
    {
      "title": "建议与展望",
      "description": "基于分析结果提出建议"
    }
  ]
}
```

---

### 12. ReportManager (报告管理器)

**文件**: `app/services/report_agent.py`

**功能**: 报告的持久化存储和检索

**核心方法**:

```python
@classmethod
def save_report(cls, report: Report) -> None:
    """
    保存报告
    
    同时保存:
    - JSON文件（报告元数据）
    - Markdown文件（报告内容）
    """

@classmethod
def get_report(cls, report_id: str) -> Optional[Report]:
    """获取报告"""

@classmethod
def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
    """根据模拟ID获取报告"""

@classmethod
def list_reports(
    cls, 
    simulation_id: Optional[str] = None, 
    limit: int = 50
) -> List[Report]:
    """列出报告"""

@classmethod
def delete_report(cls, report_id: str) -> bool:
    """删除报告"""
```

**存储结构**:
```
uploads/reports/
├── report_abc123.json    # 报告元数据
└── report_abc123.md      # Markdown报告
```

---

## 工具类

### 1. FileParser (文件解析器)

**文件**: `app/utils/file_parser.py`

**功能**: 从PDF/MD/TXT文件提取文本

**支持格式**:
- PDF: 使用PyMuPDF
- Markdown: 直接读取
- TXT: 直接读取

**核心方法**:
```python
@classmethod
def extract_text(cls, file_path: str) -> str:
    """从文件提取文本"""

@classmethod
def extract_from_multiple(cls, file_paths: List[str]) -> str:
    """从多个文件提取并合并文本"""

def split_text_into_chunks(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[str]:
    """
    文本分块
    
    特点:
    - 尝试在句子边界分割
    - 支持中英文句子结束符
    - 块之间有重叠(overlap)
    """
```

---

### 2. LLMClient (LLM客户端)

**文件**: `app/utils/llm_client.py`

**功能**: 统一的LLM调用封装(OpenAI格式)

**核心方法**:
```python
def chat(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    response_format: Optional[Dict] = None
) -> str:
    """发送聊天请求"""

def chat_json(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 4096
) -> Dict[str, Any]:
    """发送聊天请求并返回JSON"""
```

**配置**:
- 从`Config.LLM_API_KEY`读取API密钥
- 从`Config.LLM_BASE_URL`读取API地址
- 从`Config.LLM_MODEL_NAME`读取模型名称

---

### 3. Logger (日志管理)

**文件**: `app/utils/logger.py`

**功能**: 统一的日志配置

**特点**:
- 双输出:控制台(INFO+) + 文件(DEBUG+)
- 按日期命名日志文件
- 日志轮转(10MB,保留5个备份)
- 详细格式(文件) + 简洁格式(控制台)

**使用方法**:
```python
from app.utils.logger import get_logger

logger = get_logger('mirofish.mymodule')
logger.debug("调试信息")
logger.info("普通信息")
logger.warning("警告")
logger.error("错误")
```

---

### 4. Retry (重试机制)

**文件**: `app/utils/retry.py`

**功能**: API调用重试装饰器

**核心方法**:
```python
@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    exceptions=(ConnectionError, TimeoutError)
)
def call_api():
    ...
```

**特点**:
- 指数退避
- 随机抖动(避免雷击)
- 自定义异常类型
- 重试回调

---

## 配置说明

### 环境变量配置

在项目根目录创建`.env`文件:

```bash
# Flask配置
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
SECRET_KEY=your-secret-key

# LLM配置(OpenAI兼容接口)
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

# Zep配置
ZEP_API_KEY=z_xxx

# OASIS模拟配置
OASIS_DEFAULT_MAX_ROUNDS=10

# Report Agent配置（可选）
REPORT_AGENT_MAX_TOOL_CALLS=5
REPORT_AGENT_MAX_REFLECTION_ROUNDS=2
REPORT_AGENT_TEMPERATURE=0.5
```

### 配置项说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| FLASK_DEBUG | Boolean | True | 调试模式 |
| FLASK_HOST | String | 0.0.0.0 | 监听地址 |
| FLASK_PORT | Integer | 5001 | 监听端口 |
| SECRET_KEY | String | - | Flask密钥 |
| LLM_API_KEY | String | - | LLM API密钥(必填) |
| LLM_BASE_URL | String | https://api.openai.com/v1 | LLM API地址 |
| LLM_MODEL_NAME | String | gpt-4o-mini | LLM模型名称 |
| ZEP_API_KEY | String | - | Zep API密钥(必填) |
| OASIS_DEFAULT_MAX_ROUNDS | Integer | 10 | 默认模拟轮数 |
| REPORT_AGENT_MAX_TOOL_CALLS | Integer | 5 | 每章节最大工具调用次数 |
| REPORT_AGENT_MAX_REFLECTION_ROUNDS | Integer | 2 | 最大反思轮数 |
| REPORT_AGENT_TEMPERATURE | Float | 0.5 | 报告生成温度参数 |

---

## 运行指南

### 1. 环境准备

```bash
# 1. 激活conda环境
conda activate MiroFish

# 2. 安装依赖
cd backend
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑.env文件,填入API密钥
```

### 2. 启动服务

```bash
# 启动Flask服务
python run.py
```

服务启动后访问:
- 主页: http://localhost:5001
- 健康检查: http://localhost:5001/health
- API文档: (见上文API接口文档)

### 3. 使用流程

**完整流程示例**:

```bash
# Step 1: 上传文档并生成本体
curl -X POST http://localhost:5001/api/graph/ontology/generate \
  -F "files=@document.pdf" \
  -F "simulation_requirement=模拟学术不端事件的舆论发展" \
  -F "project_name=学术不端事件"

# 返回: project_id, ontology

# Step 2: 构建图谱
curl -X POST http://localhost:5001/api/graph/build \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "proj_xxx",
    "graph_name": "学术不端事件图谱"
  }'

# 返回: task_id

# Step 3: 查询构建进度
curl http://localhost:5001/api/graph/task/{task_id}

# 等待status=completed, 获取graph_id

# Step 4: 创建模拟
curl -X POST http://localhost:5001/api/simulation/create \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "proj_xxx",
    "graph_id": "mirofish_xxx"
  }'

# 返回: simulation_id

# Step 5: 准备模拟
curl -X POST http://localhost:5001/api/simulation/prepare \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "sim_xxx",
    "use_llm_for_profiles": true,
    "parallel_profile_count": 5
  }'

# 返回: task_id

# Step 6: 查询准备进度
curl -X POST http://localhost:5001/api/simulation/prepare/status \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_xxx",
    "simulation_id": "sim_xxx"
  }'

# 等待status=completed

# Step 7: 启动模拟（可选参数：max_rounds限制轮数，enable_graph_memory_update启用图谱记忆更新）
curl -X POST http://localhost:5001/api/simulation/start \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "sim_xxx",
    "platform": "parallel",
    "max_rounds": 50,
    "enable_graph_memory_update": true
  }'

# Step 8: 实时查询运行状态
curl http://localhost:5001/api/simulation/{sim_xxx}/run-status

# Step 9: 检查环境状态（模拟完成后环境会进入等待命令模式）
curl http://localhost:5001/api/simulation/{sim_xxx}/env-status

# Step 10: 采访单个Agent
curl -X POST http://localhost:5001/api/simulation/{sim_xxx}/interview \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": 0,
    "prompt": "你对这件事有什么看法？"
  }'

# Step 11: 全局采访（采访所有Agent）
curl -X POST http://localhost:5001/api/simulation/{sim_xxx}/interview/all \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你认为事件的后续发展会如何？"
  }'

# Step 12: 获取Interview历史
curl http://localhost:5001/api/simulation/{sim_xxx}/interview/history

# Step 13: 生成模拟分析报告
curl -X POST http://localhost:5001/api/report/generate \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "sim_xxx"
  }'

# 返回: task_id

# Step 14: 查询报告生成进度
curl -X POST http://localhost:5001/api/report/generate/status \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_xxx",
    "simulation_id": "sim_xxx"
  }'

# 等待status=completed

# Step 15: 获取报告
curl http://localhost:5001/api/report/by-simulation/sim_xxx

# Step 16: 与Report Agent对话
curl -X POST http://localhost:5001/api/report/chat \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "sim_xxx",
    "message": "请解释一下舆情的主要趋势"
  }'

# Step 17: 下载Markdown报告
curl -O http://localhost:5001/api/report/{report_id}/download

# Step 18: 关闭模拟环境（优雅退出）
curl -X POST http://localhost:5001/api/simulation/close-env \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "sim_xxx"
  }'

# 或者强制停止模拟
curl -X POST http://localhost:5001/api/simulation/stop \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "sim_xxx"
  }'
```

---

## 开发指南

### 添加新的实体类型

1. 修改本体生成提示词(`app/services/ontology_generator.py`)
2. 更新实体类型参考列表
3. 测试本体生成

### 添加新的平台支持

1. 在`app/services/oasis_profile_generator.py`添加平台格式转换方法
2. 在`app/services/simulation_manager.py`更新文件保存逻辑
3. 在`scripts/`目录添加平台模拟脚本
4. 更新`SimulationRunner`的平台检测逻辑

### 自定义LLM提示词

主要提示词文件:
- 本体生成: `app/services/ontology_generator.py` → `ONTOLOGY_SYSTEM_PROMPT`
- 人设生成: `app/services/oasis_profile_generator.py` → `_build_individual_persona_prompt`
- 配置生成: `app/services/simulation_config_generator.py` → `_generate_time_config`

### 调试技巧

1. **查看日志**:
   ```bash
   tail -f logs/$(date +%Y-%m-%d).log
   ```

2. **测试API**:
   ```bash
   # 使用httpie
   http POST localhost:5001/api/graph/ontology/generate \
     files@document.pdf \
     simulation_requirement="测试需求"
   ```

3. **调试模式**:
   ```python
   # 在代码中添加断点
   import pdb; pdb.set_trace()
   ```

---

## 常见问题

### Q1: Zep API调用失败

**原因**: API密钥错误或网络问题

**解决**:
1. 检查`.env`中的`ZEP_API_KEY`
2. 测试Zep连接:
   ```python
   from zep_cloud.client import Zep
   client = Zep(api_key="your-key")
   client.graph.list()
   ```
3. 查看日志中的详细错误信息

### Q2: LLM生成的JSON解析失败

**原因**: LLM输出被截断或格式不正确

**解决**:
- 系统已实现JSON修复逻辑
- 如仍失败,会自动回退到规则生成
- 可调整`temperature`参数降低随机性

### Q3: 模拟进程启动失败

**原因**: conda环境未激活或依赖缺失

**解决**:
```bash
# 确保在MiroFish环境中
conda activate MiroFish

# 检查OASIS依赖
pip install oasis-ai camel-ai
```

### Q4: 内存不足

**原因**: 大型文档或大量实体

**解决**:
1. 减小chunk_size
2. 限制entity_types数量
3. 使用更小的LLM模型
4. 增加系统内存

### Q5: 文件上传失败

**原因**: 文件大小超过限制或格式不支持

**解决**:
- 检查`Config.MAX_CONTENT_LENGTH`(默认50MB)
- 支持格式:PDF/MD/TXT
- 确保文件编码为UTF-8

---

## 性能优化建议

1. **并行处理**:
   - 人设生成并行数:`parallel_profile_count=5`
   - Zep批量上传:`batch_size=3`

2. **缓存策略**:
   - 项目状态已持久化到文件
   - 任务状态使用内存缓存

3. **容错重试**:
   - Zep API调用:3次重试
   - LLM API调用:3次重试

4. **日志管理**:
   - 日志文件自动轮转
   - 控制台只显示INFO+

---

## 贡献指南

### 代码规范

1. 遵循PEP 8
2. 使用类型注解
3. 添加docstring
4. 编写单元测试

### 提交规范

```
feat: 添加新功能
fix: 修复bug
docs: 更新文档
refactor: 重构代码
test: 添加测试
```

---

## 许可证

MIT License

---

## 联系方式

- 项目地址: [GitHub链接]
- 问题反馈: [Issues链接]
- 技术文档: 见本README

---

**最后更新**: 2025-12-09
**版本**: v1.3.0

### 更新日志

**v1.3.0 (2025-12-09)**:
- 新增 Report Agent 模拟报告生成功能
  - 使用 LangChain + Zep 实现 ReACT 模式
  - 自动规划报告大纲，分段生成内容
  - 每章节可多次调用Zep检索工具获取信息
  - 生成专业的Markdown格式报告
- 新增 Report Agent 对话功能
  - 报告完成后可与Agent对话
  - Agent自主调用检索工具回答问题
- 新增 Zep 检索工具服务
  - 封装图谱搜索、节点读取、边查询等工具
  - 支持语义搜索、统计分析、上下文获取
- 新增报告管理接口
  - 报告生成、查询、下载、删除
  - 报告状态检查（解锁Interview功能）
- 依赖更新
  - 新增 langchain>=0.2.0
  - 新增 langchain-core>=0.2.0
  - 新增 langchain-openai>=0.1.0

**v1.2.0 (2025-12-08)**:
- 新增 Interview 采访功能
  - 支持单个Agent采访
  - 支持批量采访多个Agent
  - 支持全局采访（所有Agent使用相同问题）
  - 支持获取Interview历史记录
- 新增模拟状态持久化
  - 模拟完成后环境不立即关闭，进入等待命令模式
  - 支持优雅关闭环境命令
- 新增 IPC 通信机制
  - Flask后端与模拟脚本之间的进程间通信
  - 基于文件系统的命令/响应模式

**v1.1.0 (2025-12-05)**:
- 新增图谱记忆动态更新功能
- 支持 max_rounds 参数限制模拟轮数

**v1.0.0**:
- 初始版本发布
- 支持知识图谱构建
- 支持Agent人设生成
- 支持双平台模拟

