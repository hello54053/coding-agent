Git仓库地址: https://github.com/hello54053/coding-agent

运行方式:
  1. 安装依赖: pip install openai
     可选:  pip install tiktoken   # 用于精确 token 计数（缺失时退化为字符估算）
  2. 配置环境变量:
     export OPENAI_API_KEY="your-api-key"
     export OPENAI_BASE_URL="your-api-base-url"
     export AGENT_MODEL="model-name"
  3. 运行:
     python coding_agent.py                              # 默认模式
     python coding_agent.py --workdir ./project         # 启用工作目录沙箱
     python coding_agent.py --workdir ./project --confirm-edits   # 沙箱 + 编辑二次确认
     python coding_agent.py --help                       # 查看全部命令行参数

  命令行参数:
     --workdir DIR          工作目录沙箱根；所有文件操作被限制在该目录下
     --confirm-edits        编辑文件（edit_file/write_file）前需要人工 y/N 二次确认
     --no-stats             禁用每轮任务结束时的统计输出
     --no-stream            禁用流式输出
     --no-parallel          禁用多工具并行执行
     --max-steps N          单轮任务最大步数（默认 30）

  交互式命令（运行中输入）:
     undo     撤销最近一次 edit_file/write_file，把文件恢复到修改前
     backups  查看当前备份栈（栈顶=最近一次编辑）
     reset    清空对话历史
     history  查看对话历史
     exit     退出

  自检脚本（不消耗 API 额度）:
     python verify_improvements.py

特色功能:
  - 支持文件读写、精确编辑、目录浏览
  - 支持执行 Shell 命令
  - 智能 view 命令自动识别文件/目录
  - 自动上下文管理与多轮对话
  - 流式输出（逐 token 打印，体验更佳）
  - 多工具并行执行（OpenAI tool calling 单轮多 call 时并行调度）
  - API 失败指数退避重试（限流/超时/网络错误自动重试）
  - 上下文窗口管理（接近 token 上限时自动裁剪旧消息，保留 system 与最新一轮）
  - Token 计数与每轮统计（步数/工具调用数/prompt+completion tokens/耗时）
  - 工作目录沙箱（--workdir）：相对路径基于沙箱根解析，../ 越界被拒绝
  - 编辑类工具的 unified diff 输出（edit_file/write_file 返回值含 diff）
  - edit_file 支持 replace_all（批量替换）与 fuzzy match 提示（找不到时返回最相似行）
  - 编辑二次确认模式（--confirm-edits）：dry-run 生成 diff 预览后人工 y/N
  - 编辑前自动备份到栈，支持 undo 命令一键回滚（大胆尝试：放心让 Agent 重构复杂函数）

设计思路:
  采用 ReAct 模式实现 Agent 核心循环：
  1. 用户输入追加到对话历史
  2. 调用 LLM，模型决定是否调用工具
  3. 如有工具调用，在本地执行并返回结果
  4. 循环直至任务完成或达到最大步数
  5. 工具独立封装于 TOOLS 字典，便于扩展

  沙箱机制：通过隐藏参数 _workdir 把沙箱根目录注入到工具函数（不暴露给模型），
  所有文件工具调用 _resolve_path 解析路径；execute_command 的 cwd 切到沙箱根。
  注：shell 命令内容本身无法被沙箱限制（如 cd /），仅在 cwd 层面做隔离。

工具列表:
  - view:          智能查看文件或目录
  - read_file:     读取文件（支持 offset/limit）
  - write_file:    写入文件（返回值含 unified diff）
  - edit_file:     精确替换文本（支持 replace_all、fuzzy match 提示、diff 输出）
  - list_files:    列出目录内容
  - execute_command: 执行 shell 命令（cwd 切到沙箱根）
