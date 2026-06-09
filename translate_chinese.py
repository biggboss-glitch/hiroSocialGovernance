"""
Bulk replace Chinese text with English across all backend Python files.
Also replaces MiroFish -> Hiro and mirofish -> hiro in Python files.
"""
import os
import re

ROOT = r"c:\Users\win11\Downloads\MetaHackathonAgent\hiro-social-governance\MiroFish"

# Chinese -> English translations for common patterns in the codebase
TRANSLATIONS = {
    # graph.py
    '图谱相关API路由': 'Graph-related API routes',
    '采用项目上下文机制，服务端持久化状态': 'Uses project context mechanism with server-side persistent state',
    '获取日志器': 'Get logger',
    '检查文件扩展名是否允许': 'Check if file extension is allowed',
    '项目管理接口': 'Project Management APIs',
    '获取项目详情': 'Get project details',
    '列出所有项目': 'List all projects',
    '删除项目': 'Delete project',
    '重建项目（保留文件和本体，清除图谱）': 'Reset project (keep files and ontology, clear graph)',
    '重置到本体已生成状态': 'Reset to ontology-generated state',
    '接口1：上传文件并生成本体': 'API 1: Upload files and generate ontology',
    '接口1：上传文件，分析生成本体定义': 'API 1: Upload files, analyze and generate ontology definition',
    '请求方式：multipart/form-data': 'Request format: multipart/form-data',
    '参数：': 'Parameters:',
    '上传的文件（PDF/MD/TXT），可多个': 'Uploaded files (PDF/MD/TXT), multiple allowed',
    '模拟需求描述（必填）': 'Simulation requirement description (required)',
    '项目名称（可选）': 'Project name (optional)',
    '额外说明（可选）': 'Additional context (optional)',
    '返回：': 'Returns:',
    '开始生成本体定义': 'Starting ontology generation',
    '获取参数': 'Get parameters',
    '获取上传的文件': 'Get uploaded files',
    '创建项目': 'Creating project',
    '保存文件并提取文本': 'Save files and extract text',
    '保存文件到项目目录': 'Save file to project directory',
    '提取文本': 'Extract text',
    '如果没有上传文档，或者上传的文档提取出的文本内容过少（小于500字），则利用 LLM 智能生成背景剧本': 'If no documents uploaded or extracted text is too short (<500 chars), use LLM to generate background scenario',
    '文档内容较少或未上传，将利用 LLM 智能生成详尽的背景剧本作为建图材料...': 'Document content too short or not uploaded, using LLM to generate detailed background scenario for graph building...',
    '成功生成背景剧本，共': 'Successfully generated background scenario,',
    '字符': 'characters',
    '智能生成背景剧本失败，退回到基本文本': 'Failed to generate background scenario, falling back to basic text',
    '如果有之前提取的少量文本，也保留': 'Keep previously extracted short text if available',
    '保存提取的文本': 'Save extracted text',
    '文本提取完成，共': 'Text extraction complete, total',
    '生成本体': 'Generate ontology',
    '调用 LLM 生成本体定义...': 'Calling LLM to generate ontology definition...',
    '生成本体定义失败': 'Failed to generate ontology definition',
    '保存本体到项目': 'Save ontology to project',
    '本体生成完成': 'Ontology generation complete',
    '个实体类型': 'entity types',
    '个关系类型': 'relation types',
    '本体生成完成 === 项目ID': 'Ontology generation complete === Project ID',
    '处理异常': 'Processing exception',
    '接口2: 构建GraphRAG图谱': 'API 2: Build GraphRAG graph',
    '接口3: 获取图谱数据': 'API 3: Get graph data',
    '获取图谱数据（带节点和边）': 'Get graph data (with nodes and edges)',
    '获取构建任务状态': 'Get build task status',
    '删除图谱': 'Delete graph',
    '搜索图谱': 'Search graph',
    '获取图谱统计信息': 'Get graph statistics',
    '获取实体详情': 'Get entity details',
    '获取实体的关系（边）': 'Get entity relationships (edges)',
    '按类型查询实体列表': 'Query entity list by type',
    '获取实体摘要': 'Get entity summary',
    '构建GraphRAG图谱': 'Build GraphRAG graph',
    '请求参数': 'Request parameters',
    '创建图谱构建任务': 'Creating graph build task',
    '后台执行图谱构建': 'Background graph build execution',
    '开始构建图谱': 'Starting graph build',
    '图谱构建完成': 'Graph build complete',
    '节点': 'nodes',
    '边': 'edges',
    '图谱构建失败': 'Graph build failed',
    '图谱构建过程中发生错误': 'Error during graph build process',
    '图谱构建线程启动': 'Graph build thread started',
    '错误': 'Error',
    
    # config.py
    '配置管理': 'Configuration Management',
    '统一从项目根目录的 .env 文件加载配置': 'Load configuration from project root .env file',
    '加载项目根目录的 .env 文件': 'Load project root .env file',
    '如果根目录没有 .env，尝试加载环境变量（用于生产环境）': 'If no .env in root, try loading environment variables (for production)',
    'Flask配置类': 'Flask Configuration Class',
    'Flask配置': 'Flask Config',
    'JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \\uXXXX 格式）': 'JSON config - disable ASCII escaping',
    'LLM配置（统一使用OpenAI格式）': 'LLM Config (unified OpenAI format)',
    'Mem0配置': 'Mem0 Config',
    '文件上传配置': 'File upload config',
    '文本处理配置': 'Text processing config',
    '默认切块大小': 'Default chunk size',
    '默认重叠大小': 'Default overlap size',
    'OASIS模拟配置': 'OASIS simulation config',
    'OASIS平台可用动作配置': 'OASIS platform available actions config',
    'Report Agent配置': 'Report Agent config',
    '验证必要配置': 'Validate required config',
    '未配置': 'not configured',
    
    # __init__.py
    'MiroFish Backend 启动中...': 'Hiro Backend starting...',
    'MiroFish Backend 启动完成': 'Hiro Backend startup complete',
    '已注册模拟进程清理函数': 'Registered simulation process cleanup function',
    '注册全局清理函数，在应用退出时清理所有模拟进程': 'Register global cleanup function to clean up all simulation processes on app exit',
    '应用退出，正在清理模拟进程...': 'App exiting, cleaning up simulation processes...',
    '清理完成': 'Cleanup complete',
    '创建Flask应用': 'Create Flask application',
    '请求日志': 'Request logging',
    '请求': 'Request',
    '请求体': 'Request body',
    '响应': 'Response',
    '注册蓝图': 'Register blueprints',
    '注册API路由': 'Register API routes',
    
    # LLM prompts in graph.py
    '你是一个资深的社会舆论与预测场景设计师。用户希望进行一个社会舆论模拟，预测主题是：': 'You are an expert social opinion and prediction scenario designer. The user wants to run a social opinion simulation. The prediction topic is:',
    '补充背景信息: ': 'Additional background: ',
    '用户提供的参考资料较少（如下）：': 'The user provided limited reference material (as follows):',
    '请你扮演一个"世界构建者"，结合上述仅有的资料，为这个模拟场景撰写一篇详尽的背景描述文档和多角色剧本设定（大约 1500-2500 字）。': 'Please act as a "world builder" and, combining the above limited materials, write a detailed background description document and multi-character script setting (approximately 1500-2500 words) for this simulation scenario.',
    '为了使后续构建的知识图谱足够丰富，你的文档必须包含以下内容：': 'To ensure the subsequent knowledge graph is sufficiently rich, your document must include:',
    '1. 详尽的事件起因、经过、发展，以及当前面临的核心争议点。': '1. Detailed event causes, progression, development, and current core controversies.',
    '2. 引入至少 10-15 个具体的利益相关实体（包括具体的个人和机构/组织），并详细描述他们的背景、立场和相互关系：': '2. Introduce at least 10-15 specific stakeholder entities (including specific individuals and institutions/organizations), with detailed backgrounds, positions, and relationships:',
    '- 至少 2-3 个具体的机构名称（例如：具体的学校、学校行政部门、政府监管部门、媒体机构等）': '   - At least 2-3 specific institution names (e.g., specific schools, administrative departments, government regulators, media organizations)',
    '- 至少 6-8 个具体的个人代表（包含具有不同立场、不同社会关系的学生、教职工、记者、家长代表、公众领袖等，提供具体的姓名、职务与性格）': '   - At least 6-8 specific individual representatives (students, faculty, journalists, parent representatives, public leaders with different positions and social relationships, providing specific names, titles, and personalities)',
    '3. 描述上述主体之间的具体互动事件和既往关系（例如：张三是某校学生，李四是该校校长并与教育部某官员相识，某媒体报道了关于该校的传闻等）。': '3. Describe specific interactions and historical relationships between these entities (e.g., John is a student at the school, Jane is the principal who knows a government official, a media outlet reported rumors about the school, etc.).',
    '4. 包含各方立场（支持派、反对派、中立派）的具体观点表述，以便模拟中生成不同的言论。': '4. Include specific viewpoints from all sides (supporters, opponents, neutrals) to enable different opinions in the simulation.',
    '请直接以详尽背景文档的形式输出，使用中文，不需要写任何多余的代码或解释性前言，直接开始输出文档内容。': 'Please output directly as a detailed background document in English, without any extra code or explanatory preamble. Start the document content directly.',
    '你是一个高超的世界构建与背景文档生成助手。请直接输出文档，不要有多余的客套话。': 'You are an expert world-building and background document generation assistant. Output the document directly, no pleasantries.',
    '补充背景': 'Additional background',
    '参考文档': 'Reference documents',
    
    # locale.py
    '设置当前线程的语言': 'Set current thread language',
    '获取当前线程的语言': 'Get current thread language',
    '翻译函数': 'Translation function',
    
    # logger.py
    '日志系统': 'Logging system',
    '支持控制台和文件输出，使用中文格式化': 'Supports console and file output with formatting',
    '获取带名称的日志器': 'Get named logger',
    '确保日志目录存在': 'Ensure log directory exists',
    '日志输出格式': 'Log output format',
    '文件处理器': 'File handler',
    '日志目录': 'Log directory',
    '控制台处理器': 'Console handler',
    
    # local_graph_store.py
    '本地图谱存储': 'Local graph store',
    '初始化存储目录': 'Initialize storage directory',
    '保存图谱数据': 'Save graph data',
    '加载图谱数据': 'Load graph data',
    '获取图谱的节点列表': 'Get graph node list',
    '获取图谱的边列表': 'Get graph edge list',
    '添加节点': 'Add node',
    '添加边': 'Add edge',
    '搜索节点': 'Search nodes',
    '删除图谱': 'Delete graph',
    
    # Common patterns in services
    '开始': 'Starting',
    '完成': 'Complete',
    '失败': 'Failed',
    '成功': 'Success',
    '配置': 'Config',
    '模拟': 'Simulation',
    '生成': 'Generate',
}

def replace_chinese_in_file(filepath):
    """Replace Chinese text with English in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return False
    
    original = content
    
    # Replace MiroFish -> Hiro (case sensitive)
    content = content.replace('MiroFish', 'Hiro')
    
    # Sort translations by length (longest first to avoid partial matches)
    sorted_translations = sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for chinese, english in sorted_translations:
        content = content.replace(chinese, english)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {os.path.basename(filepath)}")
        return True
    return False

# Process all Python files in backend
backend_dir = os.path.join(ROOT, "backend")
count = 0
for dirpath, dirnames, filenames in os.walk(backend_dir):
    # Skip __pycache__
    dirnames[:] = [d for d in dirnames if d != '__pycache__']
    for fname in filenames:
        if fname.endswith('.py'):
            fpath = os.path.join(dirpath, fname)
            if replace_chinese_in_file(fpath):
                count += 1

# Also process .env files
for fname in ['.env', '.env.example']:
    fpath = os.path.join(ROOT, fname)
    if os.path.exists(fpath):
        if replace_chinese_in_file(fpath):
            count += 1

print(f"\nTotal files updated: {count}")
