# MiroFish 项目状态报告

## 项目概览
MiroFish是一个社交媒体舆论模拟系统,采用前后端分离架构。

## 完成情况

### ✅ 后端 (Flask)
- [x] RESTful API架构
- [x] 知识图谱构建(Zep)
- [x] LLM集成(OpenAI兼容接口)
- [x] 本体生成服务
- [x] 图谱构建服务
- [x] 模拟管理服务
- [x] 容错重试机制
- [x] 详细的API文档

**后端地址**: http://localhost:5001

### ✅ 前端 (Vue 3)
- [x] Vue 3 + Vite脚手架搭建
- [x] 项目目录结构规划
- [x] 首页设计完成(极简黑白线条风)
- [x] 文件上传功能(支持拖拽)
- [x] API接口封装(含重试机制)
- [x] 路由管理(Vue Router 4)
- [x] 第二页基础框架
- [x] 自动轮询任务状态
- [x] 响应式设计

**前端地址**: http://localhost:3000

## 目录结构

```
MiroFish/
├── backend/                    # 后端服务
│   ├── app/
│   │   ├── api/               # API路由
│   │   ├── services/          # 业务逻辑
│   │   ├── models/            # 数据模型
│   │   └── utils/             # 工具类
│   ├── scripts/               # 模拟脚本
│   ├── uploads/               # 数据存储
│   ├── run.py                 # 启动入口
│   └── requirements.txt       # Python依赖
│
├── frontend/                   # 前端项目
│   ├── src/
│   │   ├── api/               # API接口封装
│   │   ├── assets/            # 静态资源
│   │   ├── views/             # 页面组件
│   │   ├── router/            # 路由配置
│   │   ├── App.vue            # 根组件
│   │   └── main.js            # 入口文件
│   ├── .env.development       # 开发环境配置
│   ├── vite.config.js         # Vite配置
│   └── package.json           # 依赖配置
│
├── static/                     # 共享静态资源
│   └── image/
│       └── MiroFish_logo_compressed.jpeg
│
├── mydoc/                      # 项目文档
│   ├── MiroFish文档.md
│   └── 前端设计思路.md
│
└── .env                        # 环境配置(后端)
```

## 首页功能展示

### 页面元素
1. **Logo**: MiroFish品牌标识(居中展示)
2. **标语**: "上传任意报告,模拟世界即刻开始"
3. **模拟需求输入框**: 用户描述模拟需求
4. **项目名称**: 可选的项目命名
5. **文件上传区域**: 
   - 支持拖拽上传
   - 支持点击选择
   - 支持多文件
   - 格式: PDF、Markdown、TXT
6. **额外说明**: 可选的补充信息
7. **开始模拟按钮**: 提交表单,调用API

### 设计风格
- **极简黑白线条风**
- 纯黑(#000000) + 纯白(#FFFFFF)
- 1-2px实线边框
- 清晰的排版和间距
- 简洁的交互动画

### 交互流程
1. 用户填写模拟需求
2. 上传相关文档
3. 点击"开始模拟"
4. 系统调用 `/api/graph/ontology/generate` 接口
5. 显示加载状态
6. 成功后跳转到处理页面 `/process/:projectId`

## 第二页功能

### 布局
- **左侧**: 实时图谱可视化
- **右侧**: Step 1 - 现实种子构建流程

### 自动化流程
1. 自动加载项目信息
2. 自动开始图谱构建
3. 轮询任务状态(每2秒)
4. 实时更新进度
5. 完成后加载图谱数据

### 流程步骤
- [x] 文档分析
- [x] 本体生成
- [x] 图谱构建
- [x] 完成

## API接口实现

### 已实现的接口调用
1. **生成本体**: `POST /api/graph/ontology/generate`
   - multipart/form-data格式
   - 包含文件和表单数据
   - 自动重试机制

2. **构建图谱**: `POST /api/graph/build`
   - 自动触发
   - 返回任务ID

3. **查询任务**: `GET /api/graph/task/{taskId}`
   - 轮询机制
   - 每2秒查询一次

4. **获取图谱**: `GET /api/graph/data/{graphId}`
   - 构建完成后加载

5. **获取项目**: `GET /api/graph/project/{projectId}`
   - 页面加载时获取

## 容错机制

### 前端
- API请求自动重试(最多3次)
- 指数退避策略(1s -> 2s -> 4s)
- 超时处理(5分钟)
- 友好的错误提示
- 网络错误处理

### 后端
- Zep API调用重试
- LLM API调用重试
- 详细的日志记录
- 异常捕获和处理

## 启动步骤

### 1. 启动后端
```bash
cd /Users/guohangjiang/Desktop/MiroFish/backend
conda activate MiroFish
python run.py
```

### 2. 启动前端
```bash
cd /Users/guohangjiang/Desktop/MiroFish/frontend
npm run dev
```

### 3. 访问系统
- 前端: http://localhost:3000
- 后端: http://localhost:5001

## 技术栈

### 后端
- Flask 3.0+
- Zep Cloud SDK 2.0+
- OpenAI SDK 1.0+
- PyMuPDF (文本提取)
- Python 3.8+

### 前端
- Vue 3 (Composition API)
- Vite 7
- Vue Router 4
- Axios

## 开发环境

- **操作系统**: macOS (M系列芯片)
- **Python环境**: conda环境 MiroFish
- **Node.js**: >= 16.0.0
- **浏览器**: Chrome (推荐)

## 配置文件

### 后端 (.env)
```bash
FLASK_PORT=5001
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini
ZEP_API_KEY=your-key
```

### 前端 (.env.development)
```bash
VITE_API_BASE_URL=http://localhost:5001
```

## 已实现的规则

✅ **API容错重试机制**: 所有远程API调用都有重试机制
✅ **RESTful风格**: 后端API采用RESTful设计
✅ **统一配置**: LLM和Zep密钥统一存储在.env文件
✅ **OpenAI格式**: LLM调用统一使用OpenAI库格式
✅ **Conda环境**: 程序运行在MiroFish环境中
✅ **前后端分离**: Vue前端 + Flask后端

## 测试建议

### 功能测试
1. [ ] 文件上传(拖拽、点击)
2. [ ] 表单验证
3. [ ] API调用
4. [ ] 页面跳转
5. [ ] 错误处理
6. [ ] 响应式布局

### 端到端测试
1. [ ] 上传PDF文档
2. [ ] 填写模拟需求
3. [ ] 提交表单
4. [ ] 观察处理流程
5. [ ] 等待图谱构建完成

## 后续开发

### 优先级高
- [ ] 图谱可视化实现(D3.js或ECharts)
- [ ] 模拟运行页面
- [ ] 结果展示页面
- [ ] Agent对话功能

### 优先级中
- [ ] 历史记录管理
- [ ] 项目列表页面
- [ ] 用户设置
- [ ] 导出功能

### 优先级低
- [ ] 数据统计分析
- [ ] 多语言支持
- [ ] 深色模式
- [ ] 移动端优化

## 项目亮点

1. **极简黑白线条风格**: 独特的视觉设计
2. **全自动化流程**: 用户只需上传文档和需求
3. **容错机制完善**: 多重重试和错误处理
4. **前后端分离**: 清晰的架构设计
5. **实时反馈**: 轮询机制实时更新状态
6. **响应式设计**: 支持多种设备

## 当前状态

🟢 **开发服务器运行中**
- 后端: 需要手动启动
- 前端: 已启动在 http://localhost:3000

✅ **首页功能完整**
- 文件上传 ✓
- 表单验证 ✓
- API调用 ✓
- 错误处理 ✓

⚠️ **待完善功能**
- 图谱可视化(第二页)
- 更多页面开发
- 完整的端到端测试

## 总结

MiroFish项目的前端脚手架已搭建完成,首页设计符合极简黑白线条风格要求,核心功能已实现:

1. ✅ 文件上传(支持拖拽)
2. ✅ 表单验证和提交
3. ✅ API接口调用
4. ✅ 路由管理
5. ✅ 错误处理
6. ✅ 响应式设计

用户可以通过首页上传文档和填写模拟需求,点击"开始模拟"后系统会调用后端的 `/api/graph/ontology/generate` 接口,成功后跳转到处理页面查看图谱构建进度。

---

**更新时间**: 2025-12-10
**版本**: v1.0.0
**状态**: 开发中 🚧
