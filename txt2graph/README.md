# txt2graph

将文本文件（PDF/Markdown/TXT）转换为知识图谱的工具。

## 功能特点

- 支持多种文件格式：PDF、Markdown、TXT
- 基于 Zep Cloud 的知识图谱构建
- 自动提取真实存在的实体（人物、公司、组织、地点、产品、事件、媒体）
- 交互式图谱可视化界面
- 实体和关系的详细展示

## 实体类型

本工具只提取现实生活中真实存在的、可以有行动的实体：

| 类型 | 说明 | 示例 |
|------|------|------|
| Person | 真实的人物 | 马化腾、Elon Musk |
| Company | 注册的公司 | 腾讯、Apple Inc. |
| Organization | 组织机构 | 武汉大学、联合国 |
| Location | 地理位置 | 北京、硅谷 |
| Product | 具体产品/服务 | iPhone、微信 |
| Event | 真实事件 | 2024年巴黎奥运会 |
| Media | 媒体机构 | 人民日报、CNN |

## 安装

### 1. 激活conda环境

```bash
conda activate MiroFish
```

### 2. 安装依赖

```bash
cd txt2graph
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 Zep API Key：

```bash
cp .env.example .env
# 编辑 .env 文件，填入 ZEP_API_KEY
```

获取 API Key: https://app.getzep.com

## 使用方法

### 方式1: Web界面（推荐）

启动 Streamlit 应用：

```bash
streamlit run app.py
```

然后在浏览器中打开显示的URL（通常是 http://localhost:8501）

### 方式2: 命令行

```python
from text_extractor import extract_text
from graph_builder import build_graph_from_text

# 从文件提取文本
text = extract_text("your_document.pdf")

# 构建知识图谱
graph_data = build_graph_from_text(
    text=text,
    graph_name="我的知识图谱",
    progress_callback=print
)

# 查看结果
print(f"节点数: {len(graph_data.nodes)}")
print(f"边数: {len(graph_data.edges)}")
```

## 项目结构

```
txt2graph/
├── app.py              # Streamlit Web应用
├── text_extractor.py   # 文本提取模块
├── graph_builder.py    # 图谱构建模块
├── ontology.py         # 实体类型定义
├── requirements.txt    # 依赖列表
├── .env.example        # 环境变量示例
└── README.md           # 说明文档
```

## 注意事项

1. **处理时间**：知识图谱构建可能需要几分钟，取决于文本长度
2. **API限制**：Zep Cloud 有API调用限制，大文件建议分批处理
3. **文本质量**：输入文本的质量直接影响实体提取效果
4. **费用**：Zep Cloud 可能会产生API调用费用，请查看其定价

## 技术栈

- [Zep Cloud](https://www.getzep.com/) - 知识图谱服务
- [Streamlit](https://streamlit.io/) - Web界面框架
- [PyVis](https://pyvis.readthedocs.io/) - 图可视化
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF处理



