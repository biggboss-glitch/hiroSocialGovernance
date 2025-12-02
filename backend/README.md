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
│  └────────────────┘  │  - 实体读取  │  └─────────────────┘ │
│                      │  - 人设生成  │                        │
│                      │  - 配置生成  │                        │
│                      │  - 模拟运行  │                        │
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
   启动模拟 → 运行OASIS脚本 → 实时监控 → 记录动作 → 状态查询
   ```

---

## 技术栈

### 核心框架
- **Flask 3.0+**: Web 框架
- **Flask-CORS**: 跨域支持

### AI & 知识图谱
- **Zep Cloud SDK 2.0+**: 知识图谱构建与管理
- **OpenAI SDK 1.0+**: LLM 调用(支持 OpenAI 兼容接口)
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
    │   └── simulation.py      # 模拟相关接口
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
    │   ├── oasis_profile_generator.py     # 人设生成
    │   ├── simulation_config_generator.py # 配置生成
    │   ├── simulation_manager.py          # 模拟管理
    │   └── simulation_runner.py           # 模拟运行
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
5. 实时更新运行状态
6. 支持停止/暂停/恢复

**核心服务**:
- `SimulationRunner`: 模拟运行器

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
  "platform": "parallel"
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| simulation_id | String | 是 | - | 模拟ID |
| platform | String | 否 | parallel | 运行平台: twitter/reddit/parallel |

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
    "started_at": "2025-12-02T11:00:00"
  }
}
```

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

### 6. SimulationParameters (模拟参数)

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

# Step 7: 启动模拟
curl -X POST http://localhost:5001/api/simulation/start \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "sim_xxx",
    "platform": "parallel"
  }'

# Step 8: 实时查询运行状态
curl http://localhost:5001/api/simulation/{sim_xxx}/run-status

# Step 9: 停止模拟
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

**最后更新**: 2025-12-02
**版本**: v1.0.0

