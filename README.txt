Git仓库地址: https://github.com/hello54053/coding-agent

运行方式:
  1. 安装依赖: pip install openai
  2. 配置环境变量:
     export OPENAI_API_KEY="your-api-key"
     export OPENAI_BASE_URL="your-api-base-url"
     export AGENT_MODEL="model-name"
  3. 运行: python coding_agent.py

特色功能:
  - 支持文件读写、精确编辑、目录浏览
  - 支持执行 Shell 命令
  - 智能 view 命令自动识别文件/目录
  - 自动上下文管理与多轮对话
  - 完善的错误处理与重试机制

设计思路:
  采用 ReAct 模式实现 Agent 核心循环：
  1. 用户输入追加到对话历史
  2. 调用 LLM，模型决定是否调用工具
  3. 如有工具调用，在本地执行并返回结果
  4. 循环直至任务完成或达到最大步数
  5. 工具独立封装于 TOOLS 字典，便于扩展

工具列表:
  - view: 智能查看文件或目录
  - read_file: 读取文件（支持 offset/limit）
  - write_file: 写入文件
  - edit_file: 精确替换文本
  - list_files: 列出目录内容
  - execute_command: 执行 shell 命令
