#!/usr/bin/env python3
# coding_agent.py
"""
编程智能体
不依赖任何 Agent 框架，纯 Python 实现核心循环。
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Any

from openai import (
    OpenAI,
    APIError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
)


# -----------------------------------------------------------------------------
# Token 计数：优先用 tiktoken；不可用时退化为字符估算
# -----------------------------------------------------------------------------
try:
    import tiktoken
    _ENCODER_CACHE: Dict[str, Any] = {}

    def count_tokens(text: str, model: str = "gpt-4") -> int:
        if model not in _ENCODER_CACHE:
            try:
                _ENCODER_CACHE[model] = tiktoken.encoding_for_model(model)
            except Exception:
                _ENCODER_CACHE[model] = tiktoken.get_encoding("cl100k_base")
        return len(_ENCODER_CACHE[model].encode(text))
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

    def count_tokens(text: str, model: str = "gpt-4") -> int:
        # 粗略估算：英文 ~4 字符/token，中文 ~1.5 字符/token，取折中值
        return max(1, len(text) // 3)


def count_message_tokens(messages: List[Dict[str, Any]], model: str = "gpt-4") -> int:
    """统计消息列表的 token 数（含消息结构开销，粗略值）。"""
    total = 0
    for msg in messages:
        total += 4  # 每条消息的角色/分隔开销
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += count_tokens(content, model)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                args = fn.get("arguments", "") if isinstance(fn, dict) else getattr(fn, "arguments", "")
                total += count_tokens(name, model) + count_tokens(args, model)
    return total


# =============================================================================
# 1. 工具定义与实现
# =============================================================================

class ToolError(Exception):
    """工具执行错误，会被捕获并返回给模型"""
    pass


# -----------------------------------------------------------------------------
# 沙箱与 diff helpers
# -----------------------------------------------------------------------------
def _resolve_path(path: str, workdir: Optional[str]) -> str:
    """
    将 path 解析到 workdir 沙箱内的绝对路径。
    - workdir 为 None 时不沙箱化（保持向后兼容）。
    - 相对路径以 workdir 为基；绝对路径必须落在 workdir 内。
    - 任何试图通过 ../ 逃逸沙箱的行为都会抛 ToolError。
    """
    if not workdir:
        return path
    workdir_abs = os.path.normpath(os.path.abspath(workdir))
    if os.path.isabs(path):
        target = os.path.normpath(os.path.abspath(path))
    else:
        target = os.path.normpath(os.path.join(workdir_abs, path))
    if target == workdir_abs or target.startswith(workdir_abs + os.sep):
        return target
    raise ToolError(f"路径越界（沙箱外）: {path}（工作目录: {workdir_abs}）")


def _make_diff(old: str, new: str, path: str) -> str:
    """生成 unified diff 字符串。"""
    diff_lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff_lines)


def _find_similar(original: str, old_str: str, n: int = 3) -> str:
    """
    在 original 中查找与 old_str 最相似的若干行（fuzzy match 提示）。
    用 difflib 计算行级相似度，返回 top-N 候选行。
    """
    original_lines = original.splitlines()
    old_lines = old_str.splitlines()
    needle = old_lines[0] if old_lines else old_str
    matches = difflib.get_close_matches(needle, original_lines, n=n, cutoff=0.3)
    if not matches:
        return ""
    return "\n".join(f"  → {m}" for m in matches)


def _write_to_disk(path: str, content: str) -> None:
    """写入文件，自动创建父目录。"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_file(path: str, offset: int = 0, limit: int = 0,
              _workdir: Optional[str] = None) -> str:
    """读取文件内容。offset 为起始行(1-based)，limit 为读取行数(0表示全部)。"""
    actual_path = _resolve_path(path, _workdir)
    if not os.path.isfile(actual_path):
        raise ToolError(f"文件不存在: {path}")
    with open(actual_path, "r", encoding="utf-8") as f:
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


def write_file(path: str, content: str,
               _workdir: Optional[str] = None,
               _dry_run: bool = False) -> str:
    """
    写入文件（覆盖模式）。自动创建父目录。
    返回值包含 unified diff，便于评审查看改动。
    _dry_run=True 时仅计算 diff 不真正写入（供 confirm_edits 预览使用）。
    """
    actual_path = _resolve_path(path, _workdir)
    old = ""
    if os.path.isfile(actual_path):
        with open(actual_path, "r", encoding="utf-8") as f:
            old = f.read()

    if not _dry_run:
        _write_to_disk(actual_path, content)

    diff = _make_diff(old, content, path)
    summary = f"已写入文件: {path}"
    if diff.strip():
        return f"{summary}\n{diff}"
    if old == content:
        return f"{summary}（内容未变化）"
    return summary


def edit_file(path: str, old_str: str, new_str: str,
              replace_all: bool = False,
              _workdir: Optional[str] = None,
              _dry_run: bool = False) -> str:
    """
    精确替换文件中的 old_str 为 new_str。
    - old_str 必须精确匹配（含缩进与换行）。
    - replace_all=True 时替换所有出现，否则只替换第一处。
    - 文件不存在且 old_str 为空时创建新文件。
    - 找不到 old_str 时返回最相似的若干行作为提示（fuzzy match）。
    - 返回值包含 unified diff。
    - _dry_run=True 时仅计算 diff 不真正写入（供 confirm_edits 预览使用）。
    """
    actual_path = _resolve_path(path, _workdir)

    # 文件不存在 + old_str 为空 -> 新建
    if not os.path.isfile(actual_path):
        if old_str == "":
            if not _dry_run:
                _write_to_disk(actual_path, new_str)
            diff = _make_diff("", new_str, path)
            summary = f"已创建文件: {path}"
            return f"{summary}\n{diff}" if diff.strip() else summary
        raise ToolError(f"文件不存在: {path}")

    with open(actual_path, "r", encoding="utf-8") as f:
        original = f.read()

    if old_str not in original:
        suggestions = _find_similar(original, old_str)
        hint = f"\n\n相似行提示（精确匹配失败）：\n{suggestions}" if suggestions else ""
        raise ToolError(f"在文件 {path} 中未找到要替换的文本{hint}")

    count = original.count(old_str)
    if replace_all:
        new_content = original.replace(old_str, new_str)
    else:
        new_content = original.replace(old_str, new_str, 1)
        count = 1

    if not _dry_run:
        _write_to_disk(actual_path, new_content)

    diff = _make_diff(original, new_content, path)
    summary = f"已编辑文件: {path}（替换 {count} 处）"
    return f"{summary}\n{diff}" if diff.strip() else f"{summary}（内容未变化）"


def list_files(path: str = ".", depth: int = 2,
               _workdir: Optional[str] = None) -> str:
    """列出目录内容。depth 控制递归深度。"""
    actual_path = _resolve_path(path, _workdir)
    if not os.path.isdir(actual_path):
        raise ToolError(f"目录不存在: {path}")

    result = []
    prefix = "  "

    for root, dirs, files in os.walk(actual_path):
        # 控制深度
        current_depth = root.count(os.sep) - actual_path.count(os.sep)
        if current_depth > depth:
            del dirs[:]
            continue

        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        indent = prefix * current_depth
        rel_root = os.path.relpath(root, actual_path) if current_depth > 0 else "."
        if current_depth > 0:
            result.append(f"{indent}{os.path.basename(root)}/")

        file_indent = prefix * (current_depth + 1)
        for f in sorted(files):
            if f.startswith("."):
                continue
            result.append(f"{file_indent}{f}")

    return "\n".join(result) if result else "(空目录)"


def execute_command(command: str, timeout: int = 30,
                    _workdir: Optional[str] = None) -> str:
    """
    执行 shell 命令。
    - cwd 默认为当前工作目录；若启用了沙箱（_workdir），则 cwd 改为沙箱根目录。
    - 注意：shell 命令本身的内容无法被沙箱限制（如 cd /、绝对路径等），
      这里仅在 cwd 层面做隔离，主要用于"在项目根目录执行"的语义。
    """
    cwd = _workdir if _workdir else os.getcwd()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
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


def view(path: str, _workdir: Optional[str] = None) -> str:
    """智能查看：文件则读取，目录则列出。"""
    actual_path = _resolve_path(path, _workdir)
    if os.path.isfile(actual_path):
        return read_file(path, _workdir=_workdir)
    elif os.path.isdir(actual_path):
        return list_files(path, _workdir=_workdir)
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


# 工具 schema（发送给模型的定义；_workdir/_dry_run 等隐藏参数不暴露给模型）
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
            "description": "写入文件（覆盖模式）。如果文件已存在会被完全覆盖。自动创建父目录。返回值含 unified diff。",
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
            "description": (
                "精确编辑文件：将 old_str 替换为 new_str。old_str 必须精确匹配原文（含缩进与换行）。"
                "replace_all=True 时替换所有出现，否则只替换第一处。"
                "如果文件不存在且 old_str 为空，则创建新文件。"
                "找不到 old_str 时会返回最相似的若干行作为提示。返回值含 unified diff。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string", "description": "要替换的原文，必须精确匹配"},
                    "new_str": {"type": "string", "description": "新内容"},
                    "replace_all": {"type": "boolean", "description": "是否替换所有匹配（默认 false，仅替换第一处）"}
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
            "description": "执行 shell 命令。用于运行代码、安装依赖、执行测试、git 操作等。timeout 默认30秒。命令在沙箱根目录（若启用）下执行。",
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
    context_window: int = 32768           # 上下文窗口大小（token）
    stream: bool = True                   # 是否流式输出
    parallel_tools: bool = True           # 是否并行执行多个工具调用
    max_retries: int = 3                  # API 调用失败最大重试次数
    retry_base_delay: float = 1.0         # 重试退避基础延迟（秒）
    workdir: Optional[str] = None         # 工作目录沙箱根（None 表示不沙箱化）
    confirm_edits: bool = False           # 编辑文件前是否需要人工二次确认
    stats_enabled: bool = True            # 是否在每轮任务结束时打印统计


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
- 使用 edit_file 时，old_str 必须精确匹配原文（包括缩进和换行）；如果相似但不完全匹配，工具会返回最相似的几行作为提示
- edit_file 默认只替换第一处；若需要替换全部相同文本，传 replace_all=true
- 文件路径默认相对于工作目录（沙箱根），不要尝试用 ../ 访问沙箱外文件
- 执行命令后，根据返回码和输出判断成功或失败
- 如果任务已完成，直接回复用户，不要继续调用工具
- 保持简洁，不要输出与任务无关的内容
- 回答用户关于当前目录的问题前，先用 list_files 或 view 确认实际文件状态
- 不要依赖对话历史中的文件列表，因为用户可能在外部手动修改了文件
- 大胆尝试：每次 edit_file/write_file 都会自动保存修改前的状态到备份栈，
  用户可以用 undo 命令随时回滚你的修改，所以可以放心重构复杂函数
- 当前运行环境是 Windows，执行命令时请使用 Windows CMD 语法：
  - 查看当前目录：用 `cd`（不带参数）而不是 `pwd`
  - 列出文件：用 `dir` 而不是 `ls`
  - 路径分隔符用 `\` 或 `/`（两者都支持）
"""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.client = OpenAI(
            api_key=self.config.api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.config.base_url or os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.messages: List[Dict[str, Any]] = []
        # 沙箱根目录：解析为绝对路径并确保存在
        self.workdir: Optional[str] = self._init_workdir(self.config.workdir)
        # 本轮统计（每次 run() 开始时重置）
        self.stats: Dict[str, Any] = self._fresh_stats()
        # 编辑历史备份栈：每次 edit_file/write_file 执行前 push (actual_path, old_content)
        # 撤销时弹出栈顶并恢复；old_content="" 表示原文件不存在（撤销时删除新建文件）
        self.backups: List[tuple] = []

    @staticmethod
    def _init_workdir(workdir: Optional[str]) -> Optional[str]:
        """初始化沙箱根目录；返回绝对路径或 None。"""
        if not workdir:
            return None
        wd = os.path.abspath(workdir)
        os.makedirs(wd, exist_ok=True)
        return wd

    @staticmethod
    def _fresh_stats() -> Dict[str, Any]:
        return {
            "steps": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "start_time": 0.0,
            "estimated": False,  # 是否为本地估算（提供方未返回 usage）
        }
    
    def _call_model_with_retry(self, stream: bool = False):
        """
        带指数退避重试的模型调用。
        对限流、超时、网络错误进行重试；其它 API 错误也尝试重试。
        流式模式下请求 stream_options 让服务端在最后 chunk 返回 usage。
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.config.model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    max_tokens=self.config.max_tokens,
                    temperature=0.2,  # 低温度使工具调用更稳定
                    stream=stream,
                )
                if stream:
                    # 请求服务端在流末尾返回 usage 统计
                    kwargs["stream_options"] = {"include_usage": True}
                return self.client.chat.completions.create(**kwargs)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_err = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    print(f"\n⚠️ API 调用失败 ({type(e).__name__})，"
                          f"{delay:.1f}s 后重试 ({attempt + 1}/{self.config.max_retries})...")
                    time.sleep(delay)
                    continue
                raise
            except APIError as e:
                last_err = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    print(f"\n⚠️ API 错误 ({type(e).__name__}: {e})，"
                          f"{delay:.1f}s 后重试 ({attempt + 1}/{self.config.max_retries})...")
                    time.sleep(delay)
                    continue
                raise
        assert last_err is not None
        raise last_err

    def _accumulate_usage(self, usage: Any) -> None:
        """把 response.usage / chunk.usage 累积到 self.stats。"""
        if not usage:
            return
        try:
            self.stats["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.stats["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        except Exception:
            pass

    def _estimate_usage_fallback(self, completion_text: str = "") -> None:
        """
        提供方未返回 usage 时的 fallback 估算：
        - prompt_tokens 用 count_message_tokens(self.messages) 估算本次输入
        - completion_tokens 用 count_tokens(completion_text) 估算本次输出
        注：self.messages 是当前累积的全部历史，每步会重复计算，
        所以估算出的 prompt_tokens 偏大，仅作为 token=0 时的兜底。
        """
        self.stats["estimated"] = True
        self.stats["prompt_tokens"] += count_message_tokens(
            self.messages, self.config.model)
        if completion_text:
            self.stats["completion_tokens"] += count_tokens(
                completion_text, self.config.model)

    def _call_model(self) -> Any:
        """
        调用模型，返回响应消息对象（SimpleNamespace 或原 message）。
        支持流式输出（逐 token 打印文本）与流式 tool_calls 累积。
        同时累积 usage（prompt/completion tokens）到 self.stats；
        若提供方未返回 usage，则用本地 token 计数估算。
        """
        if not self.config.stream:
            response = self._call_model_with_retry(stream=False)
            usage = getattr(response, "usage", None)
            self._accumulate_usage(usage)
            msg = response.choices[0].message
            # 提供 fallback 估算（仅当本次未拿到 usage）
            if not usage:
                self._estimate_usage_fallback(msg.content or "")
            return msg

        # 流式：边接收边打印文本，并累积 tool_calls
        response = self._call_model_with_retry(stream=True)
        content_buf = ""
        tool_calls_buf: Dict[int, Dict[str, Any]] = {}
        printed_prefix = False
        got_usage = False

        for chunk in response:
            # 流末尾的 usage chunk（部分提供方在最后单独发一个 chunk）
            if hasattr(chunk, "usage") and chunk.usage:
                self._accumulate_usage(chunk.usage)
                got_usage = True
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if not printed_prefix:
                    print("\n🤖 Agent: ", end="", flush=True)
                    printed_prefix = True
                print(delta.content, end="", flush=True)
                content_buf += delta.content
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index if tc_delta.index is not None else 0
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_calls_buf[idx]["id"] = tc_delta.id
                    fn = tc_delta.function
                    if fn:
                        if fn.name:
                            tool_calls_buf[idx]["function"]["name"] += fn.name
                        if fn.arguments:
                            tool_calls_buf[idx]["function"]["arguments"] += fn.arguments
            if chunk.choices[0].finish_reason:
                break

        # 若输出了文本，补一个换行（便于后续工具日志另起一行）
        if printed_prefix and content_buf and not content_buf.endswith("\n"):
            print()

        # 提供 fallback 估算（仅当流式未返回 usage）
        if not got_usage:
            self._estimate_usage_fallback(content_buf)

        tool_calls_list = [
            SimpleNamespace(
                id=tc["id"] or f"call_{i}",
                type="function",
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for i, tc in sorted(tool_calls_buf.items())
        ]
        return SimpleNamespace(
            content=content_buf or None,
            tool_calls=tool_calls_list or None,
        )
    
    def _execute_tool(self, tool_call: Any) -> Dict[str, Any]:
        """
        执行单个工具调用。
        返回符合 OpenAI tool 角色的消息字典。
        注意：
        - 本方法不做调用/结果日志（日志由 run() 统一控制），
          以便并行执行时输出顺序清晰。
        - 通过隐藏参数 _workdir 把沙箱根目录注入到工具函数（不暴露给模型）。
        - 若 config.confirm_edits=True，对 edit_file/write_file 做二次确认：
          先 dry-run 拿到 diff 预览，询问 y/N，确认后才真正写入。
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

        func = TOOLS.get(function_name)
        if not func:
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"错误: 未知工具 '{function_name}'",
            }

        # 注入沙箱根目录（隐藏参数，模型看不到）
        arguments["_workdir"] = self.workdir

        # 编辑类工具的二次确认
        if self.config.confirm_edits and function_name in ("edit_file", "write_file"):
            try:
                preview_args = dict(arguments)
                preview_args["_dry_run"] = True
                preview = func(**preview_args)
                print(f"  🔍 即将修改 {arguments.get('path')}，预览：")
                # 只打印前 30 行 diff，避免输出过长
                preview_lines = preview.splitlines()
                print("\n".join(preview_lines[:30])
                      + ("..." if len(preview_lines) > 30 else ""))
                try:
                    ans = input("  ❓ 确认修改? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = ""
                if ans != "y":
                    return {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": "用户取消了修改",
                    }
            except ToolError as e:
                # dry-run 阶段就出错（如文件不存在），直接返回错误
                return {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"工具执行错误: {e}",
                }

        # 编辑类工具：执行前先把原文件内容压入备份栈，便于 undo 回滚
        # 备份的是 actual_path 与原内容（文件不存在则记为空串，撤销时删除）
        backed_up = False
        if function_name in ("edit_file", "write_file"):
            try:
                actual_path = _resolve_path(arguments.get("path", ""), self.workdir)
                if os.path.isfile(actual_path):
                    with open(actual_path, "r", encoding="utf-8") as f:
                        old_content = f.read()
                    self.backups.append((actual_path, old_content))
                else:
                    # 原文件不存在（新建场景），撤销时删除该文件
                    self.backups.append((actual_path, ""))
                backed_up = True
            except Exception:
                # 备份失败不阻塞工具执行，只是无法撤销
                pass

        try:
            result = func(**arguments)
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(result),
            }
        except ToolError as e:
            # 工具失败，刚才的备份没意义，弹出
            if backed_up:
                self.backups.pop()
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"工具执行错误: {e}",
            }
        except Exception as e:
            # 捕获所有未预期异常，防止 Agent 崩溃
            if backed_up:
                self.backups.pop()
            err_msg = f"内部错误: {str(e)}\n{traceback.format_exc()}"
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": err_msg,
            }

    def undo_last_edit(self) -> str:
        """
        撤销最近一次 edit_file/write_file 修改，把文件恢复到修改前的状态。
        - 若原文件不存在（old_content 为空）则删除新创建的文件
        - 若原文件存在则用备份内容覆盖回去
        - 备份栈为空时返回提示
        """
        if not self.backups:
            return "没有可撤销的修改（备份栈为空）"
        actual_path, old_content = self.backups.pop()
        if old_content == "":
            # 原文件不存在，撤销 = 删除新建的文件
            if os.path.isfile(actual_path):
                try:
                    os.remove(actual_path)
                    return f"✅ 已撤销：删除了新建文件 {actual_path}"
                except Exception as e:
                    return f"⚠️ 撤销失败（删除 {actual_path} 时出错: {e}）"
            return f"⚠️ 无可撤销内容：{actual_path} 已不存在"
        # 恢复原内容
        try:
            _write_to_disk(actual_path, old_content)
            return f"✅ 已撤销：{actual_path} 已恢复到上次编辑前"
        except Exception as e:
            return f"⚠️ 撤销失败（恢复 {actual_path} 时出错: {e}）"

    def show_backups(self) -> str:
        """查看当前备份栈。"""
        if not self.backups:
            return "备份栈为空"
        lines = ["备份栈（栈顶在上，最近修改在前）："]
        for i, (path, _) in enumerate(reversed(self.backups), 1):
            lines.append(f"  {i}. {path}")
        return "\n".join(lines)
    
    def _trim_context_if_needed(self) -> None:
        """
        当上下文接近窗口上限时，自动裁剪旧消息。
        保留 system 消息和最后一轮 user 之后的对话；按"块"删除（user + 后续非 user），
        以保证 assistant 的 tool_calls 与对应 tool 结果不被割裂。
        """
        if len(self.messages) < 4:
            return
        threshold = int(self.config.context_window * 0.8)
        current = count_message_tokens(self.messages, self.config.model)
        if current <= threshold:
            return

        # 找到 system 段结尾
        sys_end = 0
        while sys_end < len(self.messages) and self.messages[sys_end].get("role") == "system":
            sys_end += 1

        # 找最后一个 user 消息索引
        last_user_idx = -1
        for i in range(len(self.messages) - 1, sys_end - 1, -1):
            if self.messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx <= sys_end:
            return  # 没有可裁剪的空间

        print(f"\n⚠️ 上下文接近上限 ({current} tokens)，自动裁剪历史...")

        while count_message_tokens(self.messages, self.config.model) > threshold:
            # 已只剩最后一轮 user 时停止
            if sys_end >= last_user_idx:
                break
            # 块边界：end = 下一个 user 的位置（包含 last_user_idx）
            end = sys_end + 1
            while end < last_user_idx and self.messages[end].get("role") != "user":
                end += 1
            # 删除 sys_end..end-1（保留 end 处的 user）
            del self.messages[sys_end:end]
            last_user_idx -= (end - sys_end)

        after = count_message_tokens(self.messages, self.config.model)
        print(f"   裁剪完成: {after} tokens (保留 {len(self.messages)} 条消息)")

    def run(self, user_input: str) -> str:
        """
        执行一轮用户任务。
        返回最终回复文本。无论正常返回或异常，结束都打印统计。
        """
        # 重置本轮统计
        self.stats = self._fresh_stats()
        self.stats["start_time"] = time.perf_counter()

        result = ""
        try:
            result = self._run_loop(user_input)
        finally:
            self._print_stats()
        return result

    def _run_loop(self, user_input: str) -> str:
        """实际的核心循环。"""
        # 初始化对话（仅首次）
        if not self.messages:
            self.messages.append({"role": "system", "content": self.SYSTEM_PROMPT})

        # 追加用户输入
        self.messages.append({"role": "user", "content": user_input})
        print(f"\n👤 用户: {user_input}")

        for step in range(self.config.max_steps):
            self.stats["steps"] += 1
            # 调用模型前裁剪上下文
            self._trim_context_if_needed()
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
                # 流式模式下内容已边输出边打印；非流式则在此打印
                if not self.config.stream:
                    print(f"\n🤖 Agent: {response_msg.content}")
                return response_msg.content or ""

            # 先打印所有工具调用（保持顺序清晰，便于审查）
            for tc in response_msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    args_str = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args_str = tc.function.arguments
                print(f"  🔧 调用工具: {tc.function.name}({args_str})")

            # 执行工具调用（多工具时并行）
            tool_calls = response_msg.tool_calls
            n = len(tool_calls)
            self.stats["tool_calls"] += n
            tool_results: List[Dict[str, Any]] = [None] * n  # type: ignore[list-item]
            if self.config.parallel_tools and n > 1:
                with ThreadPoolExecutor(max_workers=min(4, n)) as ex:
                    future_to_idx = {ex.submit(self._execute_tool, tc): i
                                      for i, tc in enumerate(tool_calls)}
                    for fut in as_completed(future_to_idx):
                        i = future_to_idx[fut]
                        tool_results[i] = fut.result()
            else:
                for i, tc in enumerate(tool_calls):
                    tool_results[i] = self._execute_tool(tc)

            # 打印结果摘要并追加到历史
            for result in tool_results:
                summary = result["content"][:200].replace("\n", " ")
                if len(result["content"]) > 200:
                    summary += "..."
                print(f"  📤 工具结果: {summary}")

            self.messages.extend(tool_results)

        # 达到最大步数限制
        msg = "达到最大步数限制，任务未完成。"
        print(f"\n🤖 Agent: {msg}")
        return msg

    def _print_stats(self) -> None:
        """打印本轮统计信息。"""
        if not self.config.stats_enabled:
            return
        elapsed = time.perf_counter() - self.stats["start_time"]
        pt = self.stats["prompt_tokens"]
        ct = self.stats["completion_tokens"]
        estimated = self.stats["estimated"]
        print("\n" + "=" * 60)
        print("📊 本轮统计")
        print("=" * 60)
        print(f"  推理步数:          {self.stats['steps']}")
        print(f"  工具调用:          {self.stats['tool_calls']} 次")
        suffix = " (本地估算，prompt 偏大)" if estimated else ""
        print(f"  prompt tokens:     {pt}{suffix}")
        print(f"  completion tokens: {ct}{suffix}")
        print(f"  总 tokens:          {pt + ct}")
        print(f"  耗时:              {elapsed:.2f}s")
        if pt + ct == 0:
            print("  （token 数为 0：可能本地计数器也失效）")
        print("=" * 60)
    
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
    parser = argparse.ArgumentParser(
        description="编程智能体 (Coding Agent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "环境变量:\n"
            "  OPENAI_API_KEY   API 密钥（也可设 DASHSCOPE_API_KEY 作为兜底）\n"
            "  OPENAI_BASE_URL  API 网关地址\n"
            "  AGENT_MODEL      模型名称（默认 qwen3.8-27b）\n"
            "\n"
            "示例:\n"
            "  python coding_agent.py --workdir ./project\n"
            "  python coding_agent.py --workdir ./project --confirm-edits\n"
            "  python coding_agent.py --no-stats --no-stream\n"
        ),
    )
    parser.add_argument("--workdir", default=None,
                        help="工作目录沙箱根；Agent 的所有文件操作被限制在该目录下，"
                             "execute_command 的 cwd 也会切到该目录")
    parser.add_argument("--confirm-edits", action="store_true",
                        help="编辑文件（edit_file/write_file）前需要人工 y/N 二次确认")
    parser.add_argument("--no-stats", action="store_true",
                        help="禁用每轮任务结束时的统计输出")
    parser.add_argument("--no-stream", action="store_true",
                        help="禁用流式输出（默认开启）")
    parser.add_argument("--no-parallel", action="store_true",
                        help="禁用多工具并行执行（默认开启）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="单轮任务最大步数（默认 30）")
    args = parser.parse_args()

    print("=" * 60)
    print("  编程智能体 (Coding Agent)")
    print("  输入 'exit' 退出 | 'reset' 重置对话 | 'history' 查看历史")
    print("  输入 'undo' 撤销上次编辑 | 'backups' 查看备份栈")
    if args.workdir:
        print(f"  工作目录沙箱: {os.path.abspath(args.workdir)}")
    if args.confirm_edits:
        print("  编辑二次确认: 已启用")
    print("=" * 60)

    # 从环境变量 + 命令行参数共同构造配置
    config = AgentConfig(
        model=os.getenv("AGENT_MODEL", "qwen3.8-27b"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        workdir=args.workdir,
        confirm_edits=args.confirm_edits,
        stats_enabled=not args.no_stats,
        stream=not args.no_stream,
        parallel_tools=not args.no_parallel,
    )
    if args.max_steps is not None:
        config.max_steps = args.max_steps

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
        if user_input.lower() == "undo":
            print(agent.undo_last_edit())
            continue
        if user_input.lower() in ("backups", "backup"):
            print(agent.show_backups())
            continue

        try:
            agent.run(user_input)
        except Exception as e:
            print(f"\n❌ Agent 异常: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()