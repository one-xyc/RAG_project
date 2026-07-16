# 本地知识库 RAG 检索增强问答系统

## 项目简介

项目面向私有文档问答场景，搭建轻量化本地 RAG 应用，实现文档知识库构建与检索增强对话。

基于 `Streamlit` 搭建 Web 交互前端，支持网页端上传 TXT 文档；利用 `LangChain` 完成文本自动切分、向量化，持久化存入 `Chroma` 向量库。

实现 RAG 检索问答链路：用户输入问题后执行相似度检索，结合检索文档与提示词构建上下文，调用 **Ollama 本地部署的 `Qwen` 大模型**生成回答。

实现流式输出效果，还原模型思维链展示；持久化存储会话历史，支持历史对话查询与回溯。

设计文件 `MD5` 校验机制，避免重复文档向量化入库，减少冗余向量数据，优化知识库构建效率。


### 技术栈：
Python、Streamlit、LangChain、Chroma 向量数据库、通义千问 Qwen、Embedding 向量模型

### 🚀 系统核心功能模块

#### 1. 知识库文档入库模块
Streamlit 可视化页面上传文件，预览文档基础信息
自动读取解析文档文本
配置 RecursiveCharacterTextSplitter 完成文本分段
分片向量持久化存储至 Chroma 本地向量库
MD5 哈希内容校验，自动过滤重复片段

#### 2. 智能问答交互模块（RAG）
Streamlit 对话聊天交互界面
session_state 缓存会话，展示历史聊天记录
LangChain 串联完整检索增强生成流程：检索→Prompt 构造→大模型调用→结果返回
支持流式响应输出
对话历史本地文件持久化保存


### 项目结构
```RAG项目案例
├─ app_upload.py              # 知识库上传服务（Streamlit）
├─ app_qa.py                # 智能客服问答（Streamlit）
├─ knowledge_base.py          # 知识库处理：读取、切分、写库、去重
├─ rag.py                     # RAG 链组装
├─ vector_stores.py           # 向量库检索封装（持久化）
├─ file_history_store.py      # 会话历史存储
├─ config_data.py             # 模型、路径、chunk 等参数配置
└─ data                   # README 演示图片与示例素材文本所在
```

### 项目环境
本项目采用虚拟环境，python版本为3.12

