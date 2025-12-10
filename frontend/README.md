# MiroFish Frontend

MiroFish项目的前端部分,采用Vue 3 + Vite构建,极简黑白线条设计风格。

## 技术栈

- **Vue 3** - 渐进式JavaScript框架
- **Vite** - 下一代前端构建工具
- **Vue Router 4** - 官方路由管理器
- **Axios** - HTTP客户端

## 项目结构

```
frontend/
├── src/
│   ├── api/              # API接口封装
│   │   ├── index.js      # axios配置和重试机制
│   │   └── graph.js      # 图谱相关接口
│   ├── assets/           # 静态资源
│   │   └── logo/         # Logo图片
│   ├── components/       # 通用组件
│   ├── views/            # 页面组件
│   │   ├── Home.vue      # 首页
│   │   └── Process.vue   # 处理页面
│   ├── router/           # 路由配置
│   │   └── index.js
│   ├── utils/            # 工具函数
│   ├── App.vue           # 根组件
│   └── main.js           # 入口文件
├── .env.development      # 开发环境配置
├── .env.production       # 生产环境配置
└── vite.config.js        # Vite配置
```

## 开发指南

### 环境要求

- Node.js >= 16.0.0
- npm >= 8.0.0

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run dev
```

服务将在 http://localhost:3000 启动。

### 构建生产版本

```bash
npm run build
```

### 预览生产构建

```bash
npm run preview
```

## 功能说明

### 首页 (/)

- Logo展示
- 模拟需求输入
- 文档上传(支持PDF、Markdown、TXT格式)
- 拖拽上传支持
- 点击"开始模拟"调用后端接口 `/api/graph/ontology/generate`

### 处理页面 (/process/:projectId)

- 左侧:实时图谱可视化展示
- 右侧:Step 1 - 现实种子构建流程展示
- 自动轮询任务状态
- 实时更新构建进度

## 设计风格

采用极简黑白线条风格:

- 主色调:黑色 (#000000) 和白色 (#FFFFFF)
- 线条粗细:1px - 2px
- 字体:系统默认无衬线字体
- 交互:简洁的hover效果和过渡动画
- 布局:清晰的分栏和间距

## API集成

### 容错机制

- 自动重试:API请求失败时最多重试3次
- 指数退避:重试间隔逐步增加(1s, 2s, 4s)
- 超时控制:默认5分钟超时时间
- 错误提示:友好的用户错误提示

### 接口说明

参见 `src/api/graph.js` 中的接口定义和注释。

## 开发规范

- 使用 Vue 3 Composition API
- 组件采用 `<script setup>` 语法
- 样式使用 scoped CSS
- 遵循 Vue 官方风格指南

## 部署说明

1. 修改 `.env.production` 中的 `VITE_API_BASE_URL` 为实际后端地址
2. 运行 `npm run build` 构建生产版本
3. 将 `dist` 目录部署到Web服务器

## 注意事项

- 确保后端服务运行在 http://localhost:5001 (开发环境)
- 上传文件大小限制由后端配置决定(默认50MB)
- 支持的文件格式:PDF、Markdown(.md)、TXT

## License

MIT
