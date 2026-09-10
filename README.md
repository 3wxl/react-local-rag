# react-local-rag

> 基于 React18 + TypeScript + Transformers.js 实现的**浏览器端离线 RAG 知识库问答系统**，全程无后端服务，文档解析、文本分块、向量 Embedding、检索、大模型推理全部在客户端浏览器内完成。

## ✨ 项目亮点

- 完全离线运行，文档数据不上传任何服务器，隐私友好，不依赖后端接口
- WebWorker 隔离 AI 计算任务，WebGPU 硬件加速，避免主线程阻塞页面卡顿
- IndexedDB 持久化存储文档块、向量与对话记录，突破 localStorage 存储容量限制
- 自研滑动窗口重叠分块算法 + 手写余弦相似度向量检索，RAG 核心逻辑自主实现
- 技术栈覆盖 React + TS + Vite，搭配 pdfjs-dist、@huggingface/transformers，工程结构分层清晰

## 🧩 核心模块

### 模块 1：本地文件解析（pdfParse.ts）

前端直接读取 PDF / TXT 文件，**无需后端中转**

- 逐页解析 PDF 文本内容
- 文本清洗：去除多余换行、空白、无效特殊字符
- 大文件分片预处理，适配大容量文档

### 模块 2：文本分块 Chunk 算法（chunk.ts）

不可直接全文 Embedding，是 RAG 必考知识点

- 固定长度滑动窗口分块 + 重叠切片策略
- 解决 LLM 上下文窗口溢出问题，提升检索匹配精准度

### 模块 3：前端本地 Embedding + 向量检索（embedding.ts / similarity.ts / search.ts）

项目核心亮点，面试加分项

- Transformers.js 在浏览器内加载轻量 Embedding 模型
- 对文档块、用户提问生成向量
- **手写余弦相似度算法**，计算向量相似度，筛选 Top-K 相关上下文片段

### 模块 4：本地大模型推理问答（llm.worker.ts / generateAnswer.ts）

- WebWorker 子线程加载模型，不阻塞 UI 主线程
- WebGPU 硬件加速，降低本地推理耗时
- 检索片段 + 用户问题拼接构造 Prompt，本地模型流式输出答案

## 🛠️ 技术栈

- 基础框架：React18 + TypeScript + Vite
- 前端离线 AI：@huggingface/transformers（Embedding + LLM）
- PDF 解析：pdfjs-dist
- 性能优化：Web Worker、WebGPU
- 本地持久化：IndexedDB
- UI：TailwindCSS
- 工程规范：Feature 分支迭代、Conventional Commits

## 📂 项目目录结构

react-local-rag
├── public/
│ ├── models/
│ │ ├── Xenova/
│ │ │ └── bge-small-zh-v1.5/ # Embedding 向量模型
│ │ │ ├── config.json
│ │ │ ├── tokenizer.json
│ │ │ ├── tokenizer_config.json
│ │ │ ├── special_tokens_map.json
│ │ │ ├── vocab.txt
│ │ │ └── onnx/
│ │ │ └── model_quantized.onnx
│ │ └── onnx-community/
│ │ └── Qwen2.5-0.5B-Instruct/ # LLM 对话模型
│ │ ├── config.json
│ │ ├── generation_config.json
│ │ ├── tokenizer.json
│ │ ├── tokenizer_config.json
│ │ ├── special_tokens_map.json
│ │ ├── merges.txt
│ │ ├── vocab.json
│ │ └── onnx/
│ │ └── model_quantized.onnx
│ ├── ort-wasm.wasm # ONNX Runtime WASM
│ └── ort-wasm-simd.wasm # ONNX Runtime WASM SIMD 优化版本
├── src/
│ ├── assets/ # 静态资源
│ ├── components/
│ │ └── FileUpload/
│ │ └── FileUpload.tsx # 文件上传组件
│ ├── hooks/ # 自定义 Hooks（规划中）
│ ├── types/
│ │ ├── doc.ts # 文档类型定义
│ │ └── index.ts # 类型统一导出
│ ├── utils/ # 纯工具函数
│ │ ├── chunk.ts # 滑动窗口文本分块算法
│ │ ├── embedding.ts # Embedding 向量化封装
│ │ ├── generateAnswer.ts # LLM 回答生成封装
│ │ ├── pdfParse.ts # PDF 文本解析封装
│ │ ├── search.ts # 向量检索 Top-K
│ │ └── similarity.ts # 手写余弦相似度
│ ├── worker/ # WebWorker 子线程目录
│ │ └── llm.worker.ts
│ ├── App.css
│ ├── App.tsx
│ ├── index.css
│ └── main.tsx
├── .gitignore
├── eslint.config.js
├── index.html
├── package-lock.json
├── package.json
├── postcss.config.js
├── README.md
├── tailwind.config.js
├── tsconfig.app.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts

````

## 🚀 快速启动
```bash
# 创建项目
npm create vite@latest react-local-rag -- --template react-ts
cd react-local-rag

# 核心AI依赖
npm install @huggingface/transformers

# PDF解析依赖
npm install pdfjs-dist

# 工具库
npm install lodash
npm install -D @types/lodash

# UI样式
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# 启动开发环境
npm run dev

# 打包构建
npm run build
````

## 📦 模型加载说明

本项目采用完全离线加载方案，**模型文件手动下载放置至 public/models/ 目录，放置完成后无需任何网络请求即可运行**。

> ⚠️ 重要提示：模型权重体积较大，已写入`.gitignore`，不会提交到 Git 仓库。
> 其他人克隆仓库后，**需要自行下载对应模型放到 public/models 目录，否则模型加载失败**。

| 用途           | 模型                  | 存放路径                                            | 说明                               |
| -------------- | --------------------- | --------------------------------------------------- | ---------------------------------- |
| Embedding 向量 | bge-small-zh-v1.5     | public/models/Xenova/bge-small-zh-v1.5/             | 中文语义向量模型，输出 512 维向量  |
| LLM 对话推理   | Qwen2.5-0.5B-Instruct | public/models/onnx-community/Qwen2.5-0.5B-Instruct/ | 轻量中文对话大模型，量化后约 500MB |

## 📝 Git 开发规范

工作流：Feature 分支迭代开发，main 分支保持稳定可运行版本

Commit 规范：遵循 Conventional Commits

- `feat`: 新增功能
- `fix`: 修复 bug
- `refactor`: 代码重构
- `docs`: 文档更新
- `chore`: 工程配置修改

````

## ✅ 提交命令
```powershell
git add README.md
git commit -m "docs: 修正README目录结构、模型说明与markdown排版"
````

```

```
