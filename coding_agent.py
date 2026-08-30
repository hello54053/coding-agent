#!/usr/bin/env python3
# coding_agent.py
"""
编程智能体 - 最小可行实现
不依赖任何 Agent 框架，纯 Python 实现核心循环。
"""

import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

from openai import OpenAI


# =============================================================================
# 1. 工具定义与实现
# =============================================================================

class ToolError(Exception):
    """工具执行错误，会被捕获并返回给模型"""
    pass


def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """读取文件内容。offset 为起始行(1-based)，limit 为读取行数(0表示全部)。"""
    if not os.path.isfile(path):
        raise ToolError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if offset > 0:
        lines = lines[offset - 1:]
    if limit > 0:
        lines = lines[:limit]
    content = "".join(lines)
    # 对超大文件做截断保护
    if len(content) > 20000:
        content = content[:20000] + "\n... [内容已截断，共 {} 字符]".format(len(content))
    return content


def write_file(path: str, content: str) -> str:
    """写入文件（覆盖）。自动创建父目录。"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入文件: {path}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    """
    精确替换文件中的 old_str 为 new_str。
    old_str 必须精确匹配（包括缩进和换行）。
    如果 old_str 为空且文件不存在，则创建新文件。
    """
    if not os.path.isfile(path):
        if old_str == "":
            return write_file(path, new_str)
        raise ToolError(f"文件不存在: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    
    if old_str not in original:
        raise ToolError(f"在文件 {path} 中未找到要替换的文本")
    
    new_content = original.replace(old_str, new_str, 1)  # 只替换第一次出现
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    return f"已编辑文件: {path}"


def list_files(path: str = ".", depth: int = 2) -> str:
    """列出目录内容。depth 控制递归深度。"""
    if not os.path.isdir(path):
        raise ToolError(f"目录不存在: {path}")
    
    result = []
    prefix = "  "
    
    for root, dirs, files in os.walk(path):
        # 控制深度
        current_depth = root.count(os.sep) - path.count(os.sep)
        if current_depth > depth:
            del dirs[:]
            continue
        
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        indent = prefix * current_depth
        rel_root = os.path.relpath(root, path) if current_depth > 0 else "."
        if current_depth > 0:
            result.append(f"{indent}{os.path.basename(root)}/")
        
        file_indent = prefix * (current_depth + 1)
        for f in sorted(files):
            if f.startswith("."):
                continue
            result.append(f"{file_indent}{f}")
    
    return "\n".join(result) if result else "(空目录)"


def execute_command(command: str, timeout: int = 30) -> str:
    """
    执行 shell 命令。
    注意：在不受信任的环境中应限制命令范围，此处为编程任务简化处理。
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += f"[stdout]\n{result.stdout}\n"
        if result.stderr:
            output += f"[stderr]\n{result.stderr}\n"
        output += f"[返回码] {result.returncode}"
        return output
    except subprocess.TimeoutExpired:
        raise ToolError(f"命令执行超时（>{timeout}秒）")
    except Exception as e:
        raise ToolError(f"命令执行失败: {str(e)}")


def view(path: str) -> str:
    """智能查看：文件则读取，目录则列出。"""
    if os.path.isfile(path):
        return read_file(path)
    elif os.path.isdir(path):
        return list_files(path)
    else:
        raise ToolError(f"路径不存在: {path}")


# 工具注册表
TOOLS: Dict[str, Callable] = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_files": list_files,
    "execute_command": execute_command,
    "view": view,
}


# 工具 schema（发送给模型的定义）
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "view",
            "description": "查看指定路径的内容。如果是文件则读取，如果是目录则列出文件列表。这是探索项目结构的首选工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容。适合查看代码细节。支持 offset/limit 控制阅读范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "起始行号(1-based)，默认1"},
                    "limit": {"type": "integer", "description": "读取行数，0表示全部"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件（覆盖模式）。如果文件已存在会被完全覆盖。自动创建父目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "精确编辑文件：将 old_str 替换为 new_str。old_str 必须精确匹配原文（含缩进）。如果文件不存在且 old_str 为空，则创建新文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string", "description": "要替换的原文，必须精确匹配"},
                    "new_str": {"type": "string", "description": "新内容"}
                },
                "required": ["path", "old_str", "new_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录下的文件和子目录。depth 控制递归深度（默认2）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "depth": {"type": "integer", "default": 2}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "执行 shell 命令。用于运行代码、安装依赖、执行测试、git 操作等。timeout 默认30秒。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                    "timeout": {"type": "integer", "default": 30}
                },
                "required": ["command"]
            }
        }
    },
]


# =============================================================================
# 2. Agent 核心类
# =============================================================================

@dataclass
class AgentConfig:
    """Agent 配置"""
    model: str = "qwen3.8-27b"          # 模型名称
    base_url: Optional[str] = None        # API 基础地址
    api_key: Optional[str] = None         # API 密钥
    max_steps: int = 30                   # 单轮任务最大步数
    max_tokens: int = 8192                # 模型最大输出 token


class CodingAgent:
    """
    编程智能体核心实现。
    
    核心循环：
        用户输入 → 追加历史 → 调用模型 → 解析响应 
        → [如有工具调用] 执行工具 → 追加结果 → 继续循环
        → [如无工具调用] 输出结果 → 结束
    """
    
    SYSTEM_PROMPT = """你是一个编程智能体，通过调用工具完成用户交给你的编程任务。

你的工作流程：
1. 首先使用 view 或 list_files 了解项目结构
2. 读取相关文件理解现有代码
3. 使用 write_file 或 edit_file 修改代码
4. 使用 execute_command 运行测试或验证结果
5. 如果出错，分析错误信息并修复

重要规则：
- 每次编辑文件前，先读取文件确认内容
- 使用 edit_file 时，old_str 必须精确匹配原文（包括缩进和换行）
- 执行命令后，根据返回码和输出判断成功或失败
- 如果任务已完成，直接回复用户，不要继续调用工具
- 保持简洁，不要输出与任务无关的内容
- 回答用户关于当前目录的问题前，先用 list_files 或 view 确认实际文件状态
- 不要依赖对话历史中的文件列表，因为用户可能在外部手动修改了文件
"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.client = OpenAI(
            api_key=self.config.api_key or os.getenv("DASHSCOPE_API_KEY"), 
            base_url=self.config.base_url or os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.messages: List[Dict[str, Any]] = []
    
    def _call_model(self) -> Any:
        """调用模型，返回响应消息对象。"""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=self.messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=self.config.max_tokens,
            temperature=0.2,  # 低温度使工具调用更稳定
        )
        return response.choices[0].message
    
    def _execute_tool(self, tool_call: Any) -> Dict[str, Any]:
        """
        执行单个工具调用。
        返回符合 OpenAI tool 角色的消息字典。
        """
        tool_id = tool_call.id
        function_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"参数解析错误: {e}",
            }
        
        print(f"  🔧 调用工具: {function_name}({json.dumps(arguments, ensure_ascii=False)})")
        
        func = TOOLS.get(function_name)
        if not func:
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"错误: 未知工具 '{function_name}'",
            }
        
        try:
            result = func(**arguments)
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(result),
            }
        except ToolError as e:
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"工具执行错误: {e}",
            }
        except Exception as e:
            # 捕获所有未预期异常，防止 Agent 崩溃
            err_msg = f"内部错误: {str(e)}\n{traceback.format_exc()}"
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": err_msg,
            }
    
    def run(self, user_input: str) -> str:
        """
        执行一轮用户任务。
        返回最终回复文本。
        """
        # 初始化对话（仅首次）
        if not self.messages:
            self.messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
        
        # 追加用户输入
        self.messages.append({"role": "user", "content": user_input})
        print(f"\n👤 用户: {user_input}")
        
        for step in range(self.config.max_steps):
            # 调用模型
            response_msg = self._call_model()
            
            # 将模型响应加入历史
            self.messages.append({
                "role": "assistant",
                "content": response_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    } for tc in (response_msg.tool_calls or [])
                ] if response_msg.tool_calls else None,
            })
            
            # 如果没有工具调用，直接返回结果
            if not response_msg.tool_calls:
                print(f"\n🤖 Agent: {response_msg.content}")
                return response_msg.content or ""
            
            # 执行所有工具调用
            tool_results = []
            for tc in response_msg.tool_calls:
                result = self._execute_tool(tc)
                tool_results.append(result)
                # 打印工具结果摘要（前200字符）
                summary = result["content"][:200].replace("\n", " ")
                if len(result["content"]) > 200:
                    summary += "..."
                print(f"  📤 工具结果: {summary}")
            
            # 将工具结果追加到历史
            self.messages.extend(tool_results)
        
        # 达到最大步数限制
        msg = "达到最大步数限制，任务未完成。"
        print(f"\n🤖 Agent: {msg}")
        return msg
    
    def reset(self):
        """清空对话历史，开始新会话。"""
        self.messages = []
        print("🔄 已重置对话历史")

    def show_history(self):
        """打印对话历史（供用户查看）"""
        print("\n" + "=" * 60)
        print("📜 对话历史")
        print("=" * 60)
        
        # 统计消息数量
        msg_count = 0
        for msg in self.messages:
            role = msg.get("role")
            
            # 跳过 system 消息（不展示给用户）
            if role == "system":
                continue
            
            msg_count += 1
            content = msg.get("content", "")
            
            if role == "user":
                print(f"\n👤 用户 [{msg_count}]:")
                print(f"   {content}")
                
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    tools_str = ", ".join([tc["function"]["name"] for tc in tool_calls])
                    print(f"\n🤖 Agent [{msg_count}]: [调用工具: {tools_str}]")
                if content:
                    # 截断过长的内容
                    display_content = content[:500] + ("..." if len(content) > 500 else "")
                    print(f"   {display_content}")
                    
            elif role == "tool":
                tool_name = msg.get("tool_call_id", "unknown")
                # 只显示前 200 字符
                display_content = content[:200] + ("..." if len(content) > 200 else "")
                print(f"   📤 工具结果 ({tool_name[:8]}): {display_content}")
        
        if msg_count == 0:
            print("   (暂无对话历史)")
        
        print("\n" + "=" * 60)


# =============================================================================
# 3. 命令行交互入口
# =============================================================================

def main():
    print("=" * 50)
    print("  编程智能体 (Coding Agent)")
    print("  输入 'exit' 退出 | 'reset' 重置对话 | 'history' 查看历史")
    print("=" * 50)
    
    # 从环境变量读取配置，也可硬编码测试
    config = AgentConfig(
        model=os.getenv("AGENT_MODEL", "qwen3.8-27b"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = CodingAgent(config)
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue
        if user_input.lower() == "history":
            agent.show_history()
            continue
        
        try:
            agent.run(user_input)
        except Exception as e:
            print(f"\n❌ Agent 异常: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()