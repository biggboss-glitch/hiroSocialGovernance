# MiroFish Backend

社会舆论模拟系统后端服务，基于Flask框架。

## 项目结构

```
backend/
├── app/
│   ├── __init__.py          # Flask应用工厂
│   ├── config.py             # 配置管理
│   ├── api/                  # API路由
│   │   ├── __init__.py
│   │   └── graph.py          # 图谱相关接口
│   ├── services/             # 业务逻辑层
│   │   ├── ontology_generator.py  # 本体生成服务
│   │   ├── graph_builder.py       # 图谱构建服务
│   │   └── text_processor.py      # 文本处理服务
│   ├── models/               # 数据模型
│   │   ├── task.py           # 任务状态管理
│   │   └── project.py        # 项目上下文管理
│   └── utils/                # 工具模块
│       ├── file_parser.py    # 文件解析
│       └── llm_client.py     # LLM客户端
├── requirements.txt
└── run.py                    # 启动入口
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
# LLM配置
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=gpt-4o-mini

# Zep配置
ZEP_API_KEY=your-zep-api-key
```

## 启动服务

```bash
python run.py
```

服务默认运行在 http://localhost:5000

---

## API接口

### 核心工作流程

```
1. 创建项目（可选）
   POST /api/graph/project/create
   
2. 上传文件 + 生成本体（接口1）
   POST /api/graph/ontology/generate
   → 返回 project_id
   
3. 构建图谱（接口2）
   POST /api/graph/build
   → 传入 project_id
   → 返回 task_id
   
4. 查询任务进度
   GET /api/graph/task/{task_id}
   
5. 获取图谱数据
   GET /api/graph/data/{graph_id}
```

---

### 接口1：生成本体定义

**POST** `/api/graph/ontology/generate`

上传文档，分析生成适合社会模拟的实体和关系类型定义。

**请求（form-data）：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File | ✅ | PDF/MD/TXT文件，可多个 |
| `simulation_requirement` | Text | ✅ | 模拟需求描述 |
| `project_name` | Text | ❌ | 项目名称 |
| `additional_context` | Text | ❌ | 额外说明 |

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
                    "name": "Person",
                    "description": "Individuals who can express opinions",
                    "attributes": [...]
                }
            ],
            "edge_types": [
                {
                    "name": "AFFILIATED_WITH",
                    "description": "Indicates affiliation",
                    "source_targets": [...]
                }
            ]
        },
        "analysis_summary": "分析说明...",
        "files": [
            {"filename": "报告.pdf", "size": 123456}
        ],
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
| `project_id` | string | ✅ | 来自接口1的返回 |
| `graph_name` | string | ❌ | 图谱名称 |
| `chunk_size` | int | ❌ | 文本块大小，默认500 |
| `chunk_overlap` | int | ❌ | 块重叠字符，默认50 |

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
        "message": "添加文本块 (15/30)...",
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
| POST | `/api/graph/project/create` | 创建项目 |
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

## 实体设计原则

本系统专为社会舆论模拟设计，实体必须是：

**✅ 可以是：**
- 具体的个人（有名有姓）
- 注册的公司、组织、机构
- 媒体机构
- 政府部门

**❌ 不可以是：**
- 抽象概念（如"技术"、"创新"）
- 情绪、观点、趋势
- 泛指的群体（如"用户"、"消费者"）

这是因为后续需要模拟各实体对舆论的反应和传播，抽象概念无法参与这种模拟。

---

## 项目状态流转

```
created → ontology_generated → graph_building → graph_completed
                                     ↓
                                  failed
```
