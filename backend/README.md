# MiroFish Backend

社会舆论模拟系统后端服务，基于Flask框架。

## 项目结构

```
backend/
├── app/
│   ├── __init__.py              # Flask应用工厂
│   ├── config.py                # 配置管理
│   ├── api/                     # API路由
│   │   ├── __init__.py          # Blueprint注册
│   │   ├── graph.py             # Step1: 图谱相关接口
│   │   └── simulation.py        # Step2: 模拟相关接口
│   ├── services/                # 业务逻辑层
│   │   ├── __init__.py          # 服务模块导出
│   │   ├── ontology_generator.py         # 本体生成服务
│   │   ├── graph_builder.py              # 图谱构建服务
│   │   ├── text_processor.py             # 文本处理服务
│   │   ├── zep_entity_reader.py          # Zep实体读取与过滤
│   │   ├── oasis_profile_generator.py    # Agent Profile生成器
│   │   ├── simulation_config_generator.py # LLM智能配置生成器（核心）
│   │   └── simulation_manager.py         # 模拟管理器
│   ├── models/                  # 数据模型
│   │   ├── task.py              # 任务状态管理
│   │   └── project.py           # 项目上下文管理
│   └── utils/                   # 工具模块
│       ├── file_parser.py       # 文件解析
│       ├── llm_client.py        # LLM客户端
│       └── logger.py            # 日志工具
├── scripts/                     # 预设模拟脚本
│   ├── run_twitter_simulation.py    # Twitter模拟脚本
│   ├── run_reddit_simulation.py     # Reddit模拟脚本
│   └── run_parallel_simulation.py   # 双平台并行脚本
├── uploads/                     # 上传文件存储
│   ├── projects/                # 项目文件
│   └── simulations/             # 模拟数据（含配置和脚本副本）
├── requirements.txt
└── run.py                       # 启动入口
```

## 安装

```bash
conda activate MiroFish
cd backend
pip install -r requirements.txt
```

## 配置

在项目根目录 `MiroFish/.env` 中配置：

```bash
# LLM配置（统一使用OpenAI格式）
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=gpt-4o-mini

# Zep配置
ZEP_API_KEY=your-zep-api-key

# OASIS模拟配置（可选）
OASIS_DEFAULT_MAX_ROUNDS=10
```

## 启动服务

```bash
python run.py
```

服务默认运行在 http://localhost:5001

---

# 系统架构

## 完整工作流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Step 1: 图谱构建                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   上传文档 ──→ 生成本体定义 ──→ 构建Zep图谱 ──→ 图谱数据               │
│   (PDF/MD/TXT)  (LLM分析)      (异步任务)      (节点/边)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Step 2: 实体读取与模拟准备                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   读取图谱节点 ──→ 过滤符合条件实体 ──→ 生成Agent Profile ──→ 生成脚本  │
│   (Zep API)       (按Labels筛选)      (LLM生成人设)      (OASIS启动)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Step 3: 双平台并行模拟                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐          ┌─────────────────┐                     │
│   │  Twitter模拟    │          │   Reddit模拟    │                     │
│   │  (短平快交互)   │ 并行运行 │  (深度话题讨论)  │                     │
│   └─────────────────┘          └─────────────────┘                     │
│                        │                                                │
│                        ▼                                                │
│               同一批智能体，模拟真实社交环境                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# Step 1: 图谱构建 API

## 核心工作流程

```
1. 上传文件 + 生成本体
   POST /api/graph/ontology/generate
   → 返回 project_id
   
2. 构建图谱
   POST /api/graph/build
   → 返回 task_id
   
3. 查询任务进度
   GET /api/graph/task/{task_id}
   
4. 获取图谱数据
   GET /api/graph/data/{graph_id}
```

---

### 接口1：生成本体定义

**POST** `/api/graph/ontology/generate`

上传文档，分析生成适合社会模拟的实体和关系类型定义。

**请求（form-data）：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File | 是 | PDF/MD/TXT文件，可多个 |
| `simulation_requirement` | Text | 是 | 模拟需求描述 |
| `project_name` | Text | 否 | 项目名称 |
| `additional_context` | Text | 否 | 额外说明 |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "project_id": "proj_abc123def456",
        "project_name": "武汉大学舆情分析",
        "ontology": {
            "entity_types": [
                {
                    "name": "Student",
                    "description": "Students enrolled in educational institutions",
                    "attributes": [
                        {"name": "student_id", "type": "text", "description": "Unique identifier"},
                        {"name": "major", "type": "text", "description": "Field of study"}
                    ]
                }
            ],
            "edge_types": [
                {
                    "name": "AFFILIATED_WITH",
                    "description": "Indicates affiliation between entities",
                    "source_targets": [
                        {"source": "Student", "target": "University"}
                    ]
                }
            ]
        },
        "analysis_summary": "分析说明...",
        "files": [{"filename": "报告.pdf", "size": 123456}],
        "total_text_length": 20833
    }
}
```

---

### 接口2：构建图谱

**POST** `/api/graph/build`

根据 `project_id` 构建Zep知识图谱（异步任务）。

**请求（JSON）：**
```json
{
    "project_id": "proj_abc123def456",
    "graph_name": "图谱名称",
    "chunk_size": 500,
    "chunk_overlap": 50
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `project_id` | string | 是 | 来自接口1的返回 |
| `graph_name` | string | 否 | 图谱名称 |
| `chunk_size` | int | 否 | 文本块大小，默认500 |
| `chunk_overlap` | int | 否 | 块重叠字符，默认50 |

**响应：**
```json
{
    "success": true,
    "data": {
        "project_id": "proj_abc123def456",
        "task_id": "task_xyz789",
        "message": "图谱构建任务已启动"
    }
}
```

---

### 任务状态查询

**GET** `/api/graph/task/{task_id}`

```json
{
    "success": true,
    "data": {
        "task_id": "task_xyz789",
        "status": "processing",
        "progress": 45,
        "message": "Zep处理中... 15/30 完成",
        "result": null
    }
}
```

**状态值：**
- `pending` - 等待中
- `processing` - 处理中
- `completed` - 已完成
- `failed` - 失败

---

### 项目管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/graph/project/{project_id}` | 获取项目详情 |
| GET | `/api/graph/project/list` | 列出所有项目 |
| DELETE | `/api/graph/project/{project_id}` | 删除项目 |

---

### 图谱数据接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/graph/data/{graph_id}` | 获取图谱节点和边 |
| DELETE | `/api/graph/delete/{graph_id}` | 删除Zep图谱 |

---

# Step 2: 实体读取与模拟运行 API

## 核心设计理念

**全程自动化，无需人工设置参数：**
- 脚本是**预设的**，不是动态生成
- 所有模拟参数由**LLM智能生成**
- LLM读取模拟需求+文档+图谱信息，自动设置最佳参数
- **通过API接口启动和监控模拟**，前端可实时展示

## 核心工作流程

```
1. 创建模拟
   POST /api/simulation/create
   → 返回 simulation_id
   
2. 准备模拟环境（异步任务）
   POST /api/simulation/prepare
   Body: { "simulation_id": "sim_xxxx" }
   → 返回 task_id（立即响应）
   
   查询进度:
   POST /api/simulation/prepare/status
   Body: { "task_id": "task_xxxx" }
   → 返回 status, progress, result
   
3. 开始模拟
   POST /api/simulation/start
   Body: { "simulation_id": "sim_xxxx", "platform": "parallel" }
   → 在后台启动OASIS模拟进程
   → 返回运行状态

4. 实时监控（前端轮询）
   GET /api/simulation/{simulation_id}/run-status/detail
   → 返回当前进度、最近Agent动作
   
5. 停止模拟（可选）
   POST /api/simulation/stop
   Body: { "simulation_id": "sim_xxxx" }
```

---

## 实体读取接口

### 获取图谱实体（已过滤）

**GET** `/api/simulation/entities/{graph_id}`

获取图谱中符合预定义实体类型的节点。

**实体过滤逻辑：**
- Zep对符合预定义类型的实体，Labels为 `["Entity", "Student"]`
- 对不符合预定义类型的实体，Labels仅为 `["Entity"]`
- **筛选规则**：只保留Labels中包含除"Entity"和"Node"之外标签的节点

**Query参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `entity_types` | string | 否 | 逗号分隔的实体类型，用于进一步过滤 |
| `enrich` | boolean | 否 | 是否获取相关边信息，默认true |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "entities": [
            {
                "uuid": "node_uuid_123",
                "name": "杨景媛",
                "labels": ["Entity", "Student"],
                "summary": "武汉大学学生，图书馆事件当事人",
                "attributes": {
                    "student_id": "2021001",
                    "major": "计算机科学"
                },
                "related_edges": [
                    {
                        "direction": "outgoing",
                        "edge_name": "AFFILIATED_WITH",
                        "fact": "杨景媛是武汉大学的学生",
                        "target_node_uuid": "node_uuid_456"
                    }
                ],
                "related_nodes": [
                    {
                        "uuid": "node_uuid_456",
                        "name": "武汉大学",
                        "labels": ["Entity", "University"],
                        "summary": "中国著名高等学府"
                    }
                ]
            }
        ],
        "entity_types": ["Student", "University", "PublicFigure"],
        "total_count": 100,
        "filtered_count": 45
    }
}
```

---

### 获取单个实体详情

**GET** `/api/simulation/entities/{graph_id}/{entity_uuid}`

获取单个实体的完整信息，包含所有相关边和关联节点。

---

### 按类型获取实体

**GET** `/api/simulation/entities/{graph_id}/by-type/{entity_type}`

获取指定类型（如Student、PublicFigure）的所有实体。

---

## 模拟管理接口

### 创建模拟

**POST** `/api/simulation/create`

**请求（JSON）：**
```json
{
    "project_id": "proj_abc123def456",
    "graph_id": "mirofish_xxxx",
    "enable_twitter": true,
    "enable_reddit": true,
    "max_rounds": 10,
    "agents_per_round": -1
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `project_id` | string | 是 | 项目ID |
| `graph_id` | string | 否 | 图谱ID，不提供则从project获取 |
| `enable_twitter` | boolean | 否 | 启用Twitter模拟，默认true |
| `enable_reddit` | boolean | 否 | 启用Reddit模拟，默认true |
| `max_rounds` | int | 否 | 最大模拟轮数，默认10 |
| `agents_per_round` | int | 否 | 每轮激活智能体数，-1表示全部 |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "simulation_id": "sim_abc123def456",
        "config": {
            "project_id": "proj_xxxx",
            "graph_id": "mirofish_xxxx",
            "enable_twitter": true,
            "enable_reddit": true,
            "max_rounds": 10
        },
        "status": "created",
        "created_at": "2025-12-01T10:00:00"
    }
}
```

---

### 准备模拟环境（异步任务）

**POST** `/api/simulation/prepare`

**异步接口**：这是一个耗时操作，接口会立即返回`task_id`，通过`/prepare/status`查询进度。

执行模拟准备流程（LLM智能生成所有参数，带自动重试机制）：
1. 从Zep图谱读取并过滤实体
2. 为每个实体生成OASIS Agent Profile（带重试）
3. LLM智能生成模拟配置（带重试）
4. 保存配置文件和复制预设脚本

**请求（JSON）：**
```json
{
    "simulation_id": "sim_xxxx",
    "entity_types": ["Student", "PublicFigure"],
    "use_llm_for_profiles": true
}
```

**响应示例：**
```json
{
    "success": true,
    "data": {
        "simulation_id": "sim_xxxx",
        "task_id": "task_xxxx",
        "status": "preparing",
        "message": "准备任务已启动，请通过 /api/simulation/prepare/status 查询进度"
    }
}
```

---

### 查询准备进度

**POST** `/api/simulation/prepare/status`

查询准备任务的执行进度。

**请求（JSON）：**
```json
{
    "task_id": "task_xxxx"
}
```

**响应示例：**
```json
{
    "success": true,
    "data": {
        "task_id": "task_xxxx",
        "task_type": "simulation_prepare",
        "status": "processing",
        "progress": 45,
        "message": "[2/4] 生成Agent人设: 35/93 - 生成 教授张三 的人设...",
        "progress_detail": {
            "current_stage": "generating_profiles",
            "current_stage_name": "生成Agent人设",
            "stage_index": 2,
            "total_stages": 4,
            "stage_progress": 38,
            "current_item": 35,
            "total_items": 93,
            "item_description": "生成 教授张三 的人设..."
        },
        "result": null,
        "error": null,
        "metadata": {
            "project_id": "proj_xxxx",
            "simulation_id": "sim_xxxx"
        }
    }
}
```

**进度详情字段（progress_detail）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_stage` | string | 当前阶段标识 (reading/generating_profiles/generating_config/copying_scripts) |
| `current_stage_name` | string | 当前阶段中文名称 |
| `stage_index` | int | 当前阶段序号 (1-4) |
| `total_stages` | int | 总阶段数 (4) |
| `stage_progress` | int | 当前阶段内进度 (0-100) |
| `current_item` | int | 当前处理的项目序号 |
| `total_items` | int | 当前阶段总项目数 |
| `item_description` | string | 当前项目描述 |

**阶段说明：**

| 阶段 | 名称 | 权重 | 说明 |
|------|------|------|------|
| 1 | 读取图谱实体 | 0-20% | 从Zep读取并过滤实体 |
| 2 | 生成Agent人设 | 20-70% | 为每个实体生成OASIS Profile |
| 3 | 生成模拟配置 | 70-90% | LLM智能生成模拟参数 |
| 4 | 准备模拟脚本 | 90-100% | 复制预设脚本到模拟目录 |

**状态值（status）：**
- `pending` - 等待中
- `processing` - 处理中
- `completed` - 已完成（此时result包含结果）
- `failed` - 失败（此时error包含错误信息）

**完成后的响应：**
```json
{
    "success": true,
    "data": {
        "task_id": "task_xxxx",
        "status": "completed",
        "progress": 100,
        "message": "任务完成",
        "result": {
            "simulation_id": "sim_xxxx",
            "project_id": "proj_xxxx",
            "graph_id": "mirofish_xxxx",
            "status": "ready",
            "entities_count": 93,
            "profiles_count": 93,
            "entity_types": ["University", "Student", ...],
            "config_generated": true,
            "error": null
        }
    }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `entity_types` | array | 否 | 指定实体类型进行过滤 |
| `use_llm_for_profiles` | boolean | 否 | 是否使用LLM生成人设，默认true |

**注意**：`simulation_requirement`和`document_text`自动从项目中获取

**响应示例：**
```json
{
    "success": true,
    "data": {
        "simulation_id": "sim_abc123def456",
        "status": "ready",
        "entities_count": 45,
        "profiles_count": 45,
        "entity_types": ["Student", "PublicFigure", "University"],
        "config_generated": true,
        "config_reasoning": "根据武汉大学图书馆事件的特点，设置72小时模拟时长...",
        "run_instructions": {
            "simulation_dir": "/path/to/sim_xxx",
            "commands": {...},
            "instructions": "..."
        }
    }
}
```

---

### 获取模拟状态

**GET** `/api/simulation/{simulation_id}`

**响应示例：**
```json
{
    "success": true,
    "data": {
        "simulation_id": "sim_abc123def456",
        "status": "ready",
        "entities_count": 45,
        "profiles_count": 45,
        "entity_types": ["Student", "PublicFigure"],
        "current_round": 0,
        "twitter_status": "not_started",
        "reddit_status": "not_started"
    }
}
```

---

### 列出所有模拟

**GET** `/api/simulation/list`

| Query参数 | 类型 | 说明 |
|-----------|------|------|
| `project_id` | string | 按项目ID过滤（可选） |

---

### 获取Agent Profile

**GET** `/api/simulation/{simulation_id}/profiles`

| Query参数 | 类型 | 说明 |
|-----------|------|------|
| `platform` | string | 平台类型：reddit 或 twitter |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "platform": "reddit",
        "count": 45,
        "profiles": [
            {
                "user_id": 0,
                "user_name": "yangjingyuan_123",
                "name": "杨景媛",
                "bio": "武汉大学学生，关注教育公平与学生权益",
                "persona": "杨景媛是一名积极参与社会讨论的大学生，性格内敛但观点鲜明...",
                "karma": 1500,
                "age": 22,
                "gender": "female",
                "mbti": "INFJ",
                "country": "China",
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues"]
            }
        ]
    }
}
```

---

### 获取模拟配置

**GET** `/api/simulation/{simulation_id}/config`

获取LLM智能生成的完整配置，包含：
- `time_config`: 时间配置
- `agent_configs`: 每个Agent的活动配置
- `event_config`: 事件配置
- `generation_reasoning`: LLM的配置推理说明

---

### 下载文件

| 接口 | 说明 |
|------|------|
| GET `/api/simulation/{id}/config/download` | 下载配置文件 |
| GET `/api/simulation/{id}/script/{script_name}/download` | 下载脚本文件 |

**脚本名称：** 
- `run_twitter_simulation.py`
- `run_reddit_simulation.py`
- `run_parallel_simulation.py`

---

### 直接生成Profile

**POST** `/api/simulation/generate-profiles`

不创建模拟，直接从图谱生成Agent Profile。

```json
{
    "graph_id": "mirofish_xxxx",
    "entity_types": ["Student", "PublicFigure"],
    "use_llm": true,
    "platform": "reddit"
}
```

---

## 模拟运行控制接口

### 开始模拟

**POST** `/api/simulation/start`

启动OASIS模拟，在后台运行。

**请求（JSON）：**
```json
{
    "simulation_id": "sim_xxxx",  // 必填
    "platform": "parallel"         // 可选: twitter / reddit / parallel (默认)
}
```

**响应示例：**
```json
{
    "success": true,
    "data": {
        "simulation_id": "sim_xxxx",
        "runner_status": "running",
        "process_pid": 12345,
        "twitter_running": true,
        "reddit_running": true,
        "total_rounds": 144,
        "total_simulation_hours": 72,
        "started_at": "2025-12-01T10:00:00"
    }
}
```

---

### 停止模拟

**POST** `/api/simulation/stop`

停止正在运行的模拟。

**请求（JSON）：**
```json
{
    "simulation_id": "sim_xxxx"  // 必填
}
```

**响应示例：**
```json
{
    "success": true,
    "data": {
        "simulation_id": "sim_xxxx",
        "runner_status": "stopped",
        "completed_at": "2025-12-01T12:00:00",
        "twitter_actions_count": 500,
        "reddit_actions_count": 650
    }
}
```

---

## 实时状态监控接口

### 获取运行状态（基础）

**GET** `/api/simulation/{simulation_id}/run-status`

获取模拟运行的实时状态，用于前端轮询。

**响应示例：**
```json
{
    "success": true,
    "data": {
        "simulation_id": "sim_xxxx",
        "runner_status": "running",
        "current_round": 25,
        "total_rounds": 144,
        "progress_percent": 17.4,
        "simulated_hours": 12,
        "total_simulation_hours": 72,
        "twitter_running": true,
        "reddit_running": true,
        "twitter_actions_count": 150,
        "reddit_actions_count": 200,
        "total_actions_count": 350,
        "started_at": "2025-12-01T10:00:00",
        "updated_at": "2025-12-01T10:30:00"
    }
}
```

**运行状态值（runner_status）：**
- `idle` - 未运行
- `starting` - 启动中
- `running` - 运行中
- `paused` - 已暂停
- `stopping` - 停止中
- `stopped` - 已停止
- `completed` - 已完成
- `failed` - 失败

---

### 获取运行状态（详细，含最近动作）

**GET** `/api/simulation/{simulation_id}/run-status/detail`

获取详细运行状态，包含最近的Agent动作列表，**用于前端实时展示动态**。

**响应示例：**
```json
{
    "success": true,
    "data": {
        "simulation_id": "sim_xxxx",
        "runner_status": "running",
        "current_round": 25,
        "progress_percent": 17.4,
        "recent_actions": [
            {
                "round_num": 25,
                "timestamp": "2025-12-01T10:30:00",
                "platform": "twitter",
                "agent_id": 3,
                "agent_name": "Entity Name",
                "action_type": "CREATE_POST",
                "action_args": {"content": "Post content..."},
                "result": null,
                "success": true
            },
            {
                "round_num": 25,
                "timestamp": "2025-12-01T10:29:55",
                "platform": "reddit",
                "agent_id": 7,
                "agent_name": "Another Entity",
                "action_type": "LIKE_POST",
                "action_args": {"post_id": 5},
                "success": true
            }
        ]
    }
}
```

---

### 获取动作历史

**GET** `/api/simulation/{simulation_id}/actions`

获取完整的Agent动作历史记录。

**Query参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `limit` | int | 返回数量（默认100） |
| `offset` | int | 偏移量（默认0） |
| `platform` | string | 过滤平台（twitter/reddit） |
| `agent_id` | int | 过滤Agent ID |
| `round_num` | int | 过滤轮次 |

---

### 获取时间线

**GET** `/api/simulation/{simulation_id}/timeline`

获取按轮次汇总的时间线，用于前端展示进度条和时间线视图。

**Query参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `start_round` | int | 起始轮次（默认0） |
| `end_round` | int | 结束轮次（默认全部） |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "rounds_count": 25,
        "timeline": [
            {
                "round_num": 1,
                "twitter_actions": 10,
                "reddit_actions": 15,
                "total_actions": 25,
                "active_agents_count": 8,
                "active_agents": [0, 1, 3, 5, 7, 10, 12, 15],
                "action_types": {"CREATE_POST": 5, "LIKE_POST": 10, "LLM_ACTION": 10},
                "first_action_time": "2025-12-01T10:00:00",
                "last_action_time": "2025-12-01T10:05:00"
            }
        ]
    }
}
```

---

### 获取Agent统计

**GET** `/api/simulation/{simulation_id}/agent-stats`

获取每个Agent的活跃度统计，用于展示排行榜。

**响应示例：**
```json
{
    "success": true,
    "data": {
        "agents_count": 45,
        "stats": [
            {
                "agent_id": 3,
                "agent_name": "Active Agent",
                "total_actions": 50,
                "twitter_actions": 30,
                "reddit_actions": 20,
                "action_types": {"CREATE_POST": 10, "LIKE_POST": 25, "REPOST": 15},
                "first_action_time": "2025-12-01T10:00:00",
                "last_action_time": "2025-12-01T12:30:00"
            }
        ]
    }
}
```

---

## 数据库查询接口

### 获取帖子

**GET** `/api/simulation/{simulation_id}/posts`

从模拟数据库获取帖子列表。

**Query参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `platform` | string | 平台类型（twitter/reddit，默认reddit） |
| `limit` | int | 返回数量（默认50） |
| `offset` | int | 偏移量 |

---

### 获取评论

**GET** `/api/simulation/{simulation_id}/comments`

从Reddit模拟数据库获取评论列表。

**Query参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `post_id` | string | 过滤帖子ID（可选） |
| `limit` | int | 返回数量（默认50） |
| `offset` | int | 偏移量 |

---

# 服务层实现细节

## 1. ZepEntityReader（Zep实体读取服务）

**文件：** `app/services/zep_entity_reader.py`

### 核心功能

| 方法 | 说明 |
|------|------|
| `get_all_nodes(graph_id)` | 获取图谱所有节点 |
| `get_all_edges(graph_id)` | 获取图谱所有边 |
| `filter_defined_entities(graph_id, ...)` | 筛选符合条件的实体 |
| `get_entity_with_context(graph_id, uuid)` | 获取实体完整上下文 |
| `get_entities_by_type(graph_id, type)` | 按类型获取实体 |

### 数据结构

```python
@dataclass
class EntityNode:
    uuid: str                    # 节点UUID
    name: str                    # 实体名称
    labels: List[str]            # 标签列表 ["Entity", "Student"]
    summary: str                 # 实体摘要
    attributes: Dict[str, Any]   # 属性字典
    related_edges: List[Dict]    # 相关边信息
    related_nodes: List[Dict]    # 关联节点信息
    
    def get_entity_type(self) -> Optional[str]:
        """获取实体类型（排除默认Entity标签）"""

@dataclass
class FilteredEntities:
    entities: List[EntityNode]   # 实体列表
    entity_types: Set[str]       # 发现的实体类型
    total_count: int             # 总节点数
    filtered_count: int          # 过滤后数量
```

### 过滤逻辑示例

```python
# Zep返回的节点Labels示例：
# 符合预定义类型: ["Entity", "Student"]
# 不符合预定义类型: ["Entity"]

for node in all_nodes:
    labels = node.get("labels", [])
    custom_labels = [l for l in labels if l not in ["Entity", "Node"]]
    
    if not custom_labels:
        # 只有默认标签，跳过
        continue
    
    # 保留符合条件的实体
    entity_type = custom_labels[0]
    filtered_entities.append(node)
```

---

## 2. OasisProfileGenerator（Agent Profile生成器）

**文件：** `app/services/oasis_profile_generator.py`

### 核心功能

| 方法 | 说明 |
|------|------|
| `generate_profile_from_entity(entity, user_id)` | 从实体生成单个Profile |
| `generate_profiles_from_entities(entities)` | 批量生成Profile |
| `save_profiles_to_json(profiles, path, platform)` | 保存到JSON文件 |

### Profile数据结构

```python
@dataclass
class OasisAgentProfile:
    # 基础字段
    user_id: int              # 用户ID
    user_name: str            # 用户名
    name: str                 # 显示名称
    bio: str                  # 简介（max 150字符）
    persona: str              # 详细人设描述
    
    # Reddit字段
    karma: int = 1000
    
    # Twitter字段
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500
    
    # 人设详情
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None       # INTJ, ENFP等
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = []
    
    # 来源信息
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None
```

### Profile生成策略

**1. LLM生成（默认）**

使用LLM根据实体信息生成详细人设：

```python
prompt = f"""
Entity: {entity_name} ({entity_type})
Summary: {entity_summary}
Context: {related_edges_and_nodes}

Generate a social media user profile with:
- bio (max 150 chars)
- persona (detailed description)
- age, gender, mbti, country
- profession, interested_topics
"""
```

**2. 规则生成（Fallback）**

根据实体类型使用预定义模板：

| 实体类型 | 生成策略 |
|----------|----------|
| Student/Alumni | 年龄18-30，学生身份，关注教育话题 |
| PublicFigure/Expert | 年龄35-60，专业人士，政治经济话题 |
| MediaOutlet | 媒体官方账号，新闻时事话题 |
| University/GovernmentAgency | 机构官方账号，政策公告话题 |

---

## 3. SimulationConfigGenerator（模拟配置智能生成器）

**文件：** `app/services/simulation_config_generator.py`

### 核心功能

使用LLM分析模拟需求、文档内容、图谱实体信息，自动生成最佳的模拟参数配置。

| 方法 | 说明 |
|------|------|
| `generate_config(...)` | 智能生成完整模拟配置 |
| `_build_context(...)` | 构建LLM上下文（最大5万字） |
| `_generate_config_with_llm(...)` | 调用LLM生成配置 |
| `_generate_default_config(...)` | 默认配置（LLM失败时） |

### LLM智能生成的配置内容

**1. TimeSimulationConfig（时间配置）**
```python
@dataclass
class TimeSimulationConfig:
    total_simulation_hours: int = 72      # 模拟总时长（小时）
    minutes_per_round: int = 30           # 每轮代表的时间（分钟）
    agents_per_hour_min: int = 5          # 每小时激活Agent数量（最小）
    agents_per_hour_max: int = 20         # 每小时激活Agent数量（最大）
    peak_hours: List[int]                 # 高峰时段 [9,10,11,14,15,20,21,22]
    off_peak_hours: List[int]             # 低谷时段 [0,1,2,3,4,5]
    peak_activity_multiplier: float = 1.5 # 高峰活跃度乘数
    off_peak_activity_multiplier: float = 0.3  # 低谷活跃度乘数
```

**2. AgentActivityConfig（每个Agent的活动配置）**
```python
@dataclass
class AgentActivityConfig:
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    activity_level: float = 0.5           # 整体活跃度 (0.0-1.0)
    posts_per_hour: float = 1.0           # 每小时发帖频率
    comments_per_hour: float = 2.0        # 每小时评论频率
    active_hours: List[int]               # 活跃时间段 (0-23)
    response_delay_min: int = 5           # 响应延迟最小值（分钟）
    response_delay_max: int = 60          # 响应延迟最大值（分钟）
    sentiment_bias: float = 0.0           # 情感倾向 (-1到1)
    stance: str = "neutral"               # 立场 (supportive/opposing/neutral/observer)
    influence_weight: float = 1.0         # 影响力权重
```

**3. 不同实体类型的默认参数差异**

| 实体类型 | 活跃度 | 发帖频率 | 响应延迟 | 影响力 |
|----------|--------|----------|----------|--------|
| University/GovernmentAgency | 0.2 | 0.1/小时 | 60-240分钟 | 3.0 |
| MediaOutlet | 0.6 | 1.0/小时 | 5-30分钟 | 2.5 |
| PublicFigure/Expert | 0.5 | 0.3/小时 | 10-60分钟 | 2.0 |
| Student/Person | 0.7 | 0.5/小时 | 1-20分钟 | 1.0 |

---

## 4. SimulationManager（模拟管理器）

**文件：** `app/services/simulation_manager.py`

### 核心功能

| 方法 | 说明 |
|------|------|
| `create_simulation(project_id, graph_id, ...)` | 创建模拟 |
| `prepare_simulation(simulation_id, ...)` | 准备模拟环境（调用配置生成器） |
| `get_simulation(simulation_id)` | 获取模拟状态 |
| `get_profiles(simulation_id, platform)` | 获取Profile |
| `get_simulation_config(simulation_id)` | 获取模拟配置 |
| `get_run_instructions(simulation_id)` | 获取运行说明 |

### 模拟状态流转

```
created → preparing → ready → running → completed
                ↓              ↓
             failed         paused
```

### 生成的文件结构

```
uploads/simulations/sim_xxxx/
├── state.json                      # 模拟状态
├── simulation_config.json          # LLM生成的模拟配置（核心文件）
├── reddit_profiles.json            # Reddit Agent Profile（JSON格式）
├── twitter_profiles.csv            # Twitter Agent Profile（CSV格式）
├── run_reddit_simulation.py        # 预设Reddit模拟脚本
├── run_twitter_simulation.py       # 预设Twitter模拟脚本
├── run_parallel_simulation.py      # 预设双平台并行脚本
├── reddit_simulation.db            # Reddit数据库（运行后生成）
└── twitter_simulation.db           # Twitter数据库（运行后生成）
```

**重要：OASIS平台的Profile格式要求不同：**
- **Twitter**: 使用CSV格式，字段：`user_id,user_name,name,bio,friend_count,follower_count,statuses_count,created_at`
- **Reddit**: 使用JSON格式，支持详细人设字段：`realname,username,bio,persona,age,gender,mbti,country,profession,interested_topics`

### 配置文件示例 (simulation_config.json)

```json
{
  "simulation_id": "sim_abc123",
  "project_id": "proj_xxx",
  "graph_id": "mirofish_xxx",
  "simulation_requirement": "分析武汉大学图书馆事件舆论传播",
  
  "time_config": {
    "total_simulation_hours": 72,
    "minutes_per_round": 30,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 15,
    "peak_hours": [9, 10, 11, 14, 15, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "peak_activity_multiplier": 1.5,
    "off_peak_activity_multiplier": 0.3
  },
  
  "agent_configs": [
    {
      "agent_id": 0,
      "entity_name": "武汉大学",
      "entity_type": "University",
      "activity_level": 0.15,
      "posts_per_hour": 0.08,
      "comments_per_hour": 0.02,
      "active_hours": [9, 10, 11, 14, 15, 16, 17],
      "response_delay_min": 120,
      "response_delay_max": 360,
      "sentiment_bias": 0.1,
      "stance": "neutral",
      "influence_weight": 4.0
    },
    {
      "agent_id": 1,
      "entity_name": "杨景媛",
      "entity_type": "Student",
      "activity_level": 0.8,
      "posts_per_hour": 0.5,
      "comments_per_hour": 2.0,
      "active_hours": [7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
      "response_delay_min": 1,
      "response_delay_max": 15,
      "sentiment_bias": -0.3,
      "stance": "opposing",
      "influence_weight": 1.5
    }
  ],
  
  "event_config": {
    "initial_posts": [
      {
        "poster_agent_id": 1,
        "content": "今天在图书馆发生的事情让我非常失望..."
      }
    ],
    "hot_topics": ["图书馆事件", "学生权益", "校方回应"],
    "narrative_direction": "事件发酵后各方反应的模拟"
  },
  
  "generation_reasoning": "根据武汉大学图书馆事件的特点：1)涉及学生与校方的冲突，设置学生高活跃度、校方低频但高影响力；2)事件性质属于短期热点，设置72小时模拟时长；3)主要当事人杨景媛设置为高活跃度且持opposing立场..."
}

---

## 5. 预设模拟脚本

**目录：** `backend/scripts/`

脚本是**预设的**，不是动态生成。每次准备模拟时，脚本会被复制到模拟目录。

### 脚本说明

| 脚本 | 说明 |
|------|------|
| `run_twitter_simulation.py` | Twitter单平台模拟 |
| `run_reddit_simulation.py` | Reddit单平台模拟 |
| `run_parallel_simulation.py` | 双平台并行模拟（推荐） |

### 脚本工作原理

```python
# 脚本读取配置文件，自动设置所有参数
class TwitterSimulationRunner:
    def __init__(self, config_path: str):
        self.config = self._load_config()  # 读取simulation_config.json
    
    def _get_active_agents_for_round(self, env, current_hour, round_num):
        """根据时间和配置决定本轮激活哪些Agent"""
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])
        
        # 1. 检查是否高峰/低谷时段，调整激活数量
        # 2. 遍历每个Agent配置，检查是否在活跃时间
        # 3. 根据activity_level计算激活概率
        # 4. 返回本轮应激活的Agent列表
        ...
    
    async def run(self):
        # 1. 创建LLM模型
        # 2. 加载Agent图
        # 3. 执行初始事件（从event_config读取）
        # 4. 主循环：根据配置激活不同Agent
        ...
```

### 使用方式

```bash
# 进入模拟目录
cd backend/uploads/simulations/sim_xxxx/

# 运行模拟
python run_parallel_simulation.py --config simulation_config.json

# 其他选项
python run_parallel_simulation.py --config simulation_config.json --twitter-only
python run_parallel_simulation.py --config simulation_config.json --reddit-only
```

---

## 6. Profile文件格式说明

**OASIS对两个平台的Profile格式有不同要求：**

### Twitter Profile (CSV格式)

```csv
user_id,user_name,name,bio,friend_count,follower_count,statuses_count,created_at
0,user0,User Zero,I am user zero with interests in technology.,100,150,500,2023-01-01
1,user1,User One,Tech enthusiast and coffee lover.,200,250,1000,2023-01-02
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | int | 用户ID |
| `user_name` | string | 用户名 |
| `name` | string | 显示名称 |
| `bio` | string | 简介 |
| `friend_count` | int | 关注数 |
| `follower_count` | int | 粉丝数 |
| `statuses_count` | int | 发帖数 |
| `created_at` | string | 创建日期 |

### Reddit Profile (JSON详细格式)

```json
[
  {
    "realname": "Test User",
    "username": "test_user_123",
    "bio": "A test user for validation",
    "persona": "Test User is an enthusiastic participant in social discussions.",
    "age": 25,
    "gender": "male",
    "mbti": "INTJ",
    "country": "China",
    "profession": "Student",
    "interested_topics": ["Technology", "Education"]
  }
]
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `realname` | string | 是 | 真实姓名 |
| `username` | string | 是 | 用户名 |
| `bio` | string | 是 | 简介（最大150字符） |
| `persona` | string | 是 | 详细人设描述 |
| `age` | int | 否 | 年龄 |
| `gender` | string | 否 | 性别 |
| `mbti` | string | 否 | MBTI人格类型 |
| `country` | string | 否 | 国家 |
| `profession` | string | 否 | 职业 |
| `interested_topics` | array | 否 | 感兴趣话题列表 |

---

## 7. OASIS平台动作类型

### Twitter可用动作

| 动作 | 说明 |
|------|------|
| `CREATE_POST` | 发布推文 |
| `LIKE_POST` | 点赞推文 |
| `REPOST` | 转发推文 |
| `FOLLOW` | 关注用户 |
| `QUOTE_POST` | 引用转发 |
| `DO_NOTHING` | 不执行动作 |

### Reddit可用动作

| 动作 | 说明 |
|------|------|
| `CREATE_POST` | 发布帖子 |
| `CREATE_COMMENT` | 发表评论 |
| `LIKE_POST` | 点赞帖子 |
| `DISLIKE_POST` | 踩帖子 |
| `LIKE_COMMENT` | 点赞评论 |
| `DISLIKE_COMMENT` | 踩评论 |
| `SEARCH_POSTS` | 搜索帖子 |
| `SEARCH_USER` | 搜索用户 |
| `TREND` | 查看热门 |
| `REFRESH` | 刷新推荐 |
| `FOLLOW` | 关注用户 |
| `MUTE` | 屏蔽用户 |
| `DO_NOTHING` | 不执行动作 |

---

# 实体设计原则

本系统专为社会舆论模拟设计，实体必须是：

**可以是：**
- 具体的个人（有名有姓）
- 注册的公司、组织、机构
- 媒体机构
- 政府部门
- 高校、NGO等

**不可以是：**
- 抽象概念（如"技术"、"创新"）
- 情绪、观点、趋势
- 泛指的群体（如"用户"、"消费者"）

这是因为后续需要模拟各实体对舆论的反应和传播，抽象概念无法参与这种模拟。

---

# 项目状态流转

```
created → ontology_generated → graph_building → graph_completed
                                     ↓
                                  failed
```

---

# 运行模拟

准备完成后，进入模拟数据目录运行预设脚本：

```bash
# 激活conda环境
conda activate MiroFish

# 进入模拟目录
cd backend/uploads/simulations/sim_xxxx/

# 运行单平台模拟
python run_reddit_simulation.py --config simulation_config.json
# 或
python run_twitter_simulation.py --config simulation_config.json

# 运行双平台并行模拟（推荐）
python run_parallel_simulation.py --config simulation_config.json
```

### 脚本参数

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（必填） |
| `--twitter-only` | 只运行Twitter模拟（仅parallel脚本） |
| `--reddit-only` | 只运行Reddit模拟（仅parallel脚本） |

### 输出文件

模拟运行后会生成：
- `twitter_simulation.db` - Twitter模拟数据库
- `reddit_simulation.db` - Reddit模拟数据库

可使用SQLite工具查看模拟结果（帖子、评论、点赞等）

---

# API调用重试机制

**文件：** `app/utils/retry.py`

为LLM等外部API调用提供自动重试功能，提高系统稳定性。

## 重试策略

- **最大重试次数**：3次
- **退避策略**：指数退避（1s → 2s → 4s）
- **最大延迟**：30秒
- **随机抖动**：避免请求堆积

## 使用方式

**装饰器方式：**
```python
from app.utils.retry import retry_with_backoff

@retry_with_backoff(max_retries=3)
def call_llm_api():
    return client.chat.completions.create(...)
```

**客户端方式：**
```python
from app.utils.retry import RetryableAPIClient

retry_client = RetryableAPIClient(max_retries=3)
result = retry_client.call_with_retry(some_function, arg1, arg2)
```

**批量处理（单项失败不影响其他）：**
```python
results, failures = retry_client.call_batch_with_retry(
    items=entities,
    process_func=generate_profile,
    continue_on_failure=True
)
```

## 已应用重试机制的模块

| 模块 | 说明 |
|------|------|
| `OasisProfileGenerator` | LLM生成Agent人设 |
| `SimulationConfigGenerator` | LLM生成模拟配置 |

---

# 依赖说明

```
# Flask框架
flask>=3.0.0
flask-cors>=4.0.0

# Zep Cloud SDK
zep-cloud>=2.0.0

# OpenAI SDK（LLM调用）
openai>=1.0.0

# PDF处理
PyMuPDF>=1.24.0

# 环境变量
python-dotenv>=1.0.0

# 数据验证
pydantic>=2.0.0

# OASIS社交媒体模拟
oasis-ai>=0.1.0
camel-ai>=0.2.0
```
