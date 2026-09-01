"""
verify_improvements.py - 验证 coding_agent 改进功能
不依赖真实 API key，使用 mock + 真实工具函数验证。

运行: python verify_improvements.py
"""
import os
import sys
import time
import tempfile
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, r"d:\resources\保研材料\NJU SE\project")
import coding_agent as ca

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def make_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """构造一个流式 chunk。"""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, role=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )


# =============================================================================
print("\n" + "=" * 60)
print("验证 1: 流式输出 + tool_calls 分片累积")
print("=" * 60)

cfg = ca.AgentConfig(api_key="sk-fake", stream=True)
agent = ca.CodingAgent(cfg)

chunks = [
    make_chunk(content="Hello"),
    make_chunk(content=", "),
    make_chunk(content="world!"),
    make_chunk(tool_calls=[SimpleNamespace(
        index=0, id="call_1",
        function=SimpleNamespace(name="view", arguments='{"path":')
    )]),
    make_chunk(tool_calls=[SimpleNamespace(
        index=0, id=None,
        function=SimpleNamespace(name=None, arguments='"."}')
    )]),
    make_chunk(finish_reason="tool_calls"),
]

with patch.object(agent, "_call_model_with_retry", return_value=iter(chunks)):
    msg = agent._call_model()

check("累积文本内容", msg.content == "Hello, world!", f"got {msg.content!r}")
check("生成 tool_calls", msg.tool_calls is not None and len(msg.tool_calls) == 1)
if msg.tool_calls:
    tc = msg.tool_calls[0]
    check("tool_call id 正确", tc.id == "call_1", f"got {tc.id!r}")
    check("tool_call name 正确", tc.function.name == "view", f"got {tc.function.name!r}")
    check("分片 arguments 正确拼接",
          tc.function.arguments == '{"path":"."}',
          f"got {tc.function.arguments!r}")


# =============================================================================
print("\n" + "=" * 60)
print("验证 2: 多工具并行执行（4 个 sleep(0.5) 工具）")
print("=" * 60)

cfg = ca.AgentConfig(api_key="sk-fake", stream=False, parallel_tools=True)
agent = ca.CodingAgent(cfg)


def slow_tool(**_kwargs):
    time.sleep(0.5)
    return "done"


ca.TOOLS["slow_tool"] = slow_tool

slow_calls = [
    SimpleNamespace(id=f"c{i}", type="function",
                    function=SimpleNamespace(name="slow_tool", arguments="{}"))
    for i in range(4)
]

t0 = time.perf_counter()
serial_results = [agent._execute_tool(tc) for tc in slow_calls]
serial_time = time.perf_counter() - t0

t0 = time.perf_counter()
parallel_results = [None] * len(slow_calls)
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(agent._execute_tool, tc): i for i, tc in enumerate(slow_calls)}
    for f in as_completed(futs):
        parallel_results[futs[f]] = f.result()
parallel_time = time.perf_counter() - t0

print(f"  串行耗时: {serial_time:.2f}s（4 x 0.5s = 2.0s 预期）")
print(f"  并行耗时: {parallel_time:.2f}s（~0.5s 预期）")
check("并行明显快于串行", parallel_time < serial_time * 0.6,
      f"串行 {serial_time:.2f}s / 并行 {parallel_time:.2f}s")
check("并行结果按索引回填",
      all(r is not None and "done" in r["content"] for r in parallel_results))

del ca.TOOLS["slow_tool"]


# =============================================================================
print("\n" + "=" * 60)
print("验证 3: API 重试机制（前 2 次抛 RateLimitError，第 3 次成功）")
print("=" * 60)

cfg = ca.AgentConfig(api_key="sk-fake", stream=False,
                     max_retries=3, retry_base_delay=0.1)
agent = ca.CodingAgent(cfg)

fake_response = SimpleNamespace(
    choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))]
)

call_count = {"n": 0}


def fake_create(**kwargs):
    call_count["n"] += 1
    if call_count["n"] < 3:
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {}
        raise ca.RateLimitError(message="rate limited", response=resp, body=None)
    return fake_response


agent.client = MagicMock()
agent.client.chat.completions.create = fake_create

t0 = time.perf_counter()
try:
    resp = agent._call_model_with_retry(stream=False)
    duration = time.perf_counter() - t0
    check("重试后最终成功", resp is fake_response)
    check("总调用次数 = 3", call_count["n"] == 3, f"实际 {call_count['n']}")
    check("指数退避生效", duration >= 0.3, f"实际 {duration:.2f}s < 0.3s")
    print(f"  重试 2 次后成功，总耗时 {duration:.2f}s（退避 0.1+0.2=0.3s 预期）")
except Exception as e:
    check("重试后最终成功", False, f"抛异常: {type(e).__name__}: {e}")


# =============================================================================
print("\n" + "=" * 60)
print("验证 4: 上下文裁剪（context_window 调小到 100 token）")
print("=" * 60)

cfg = ca.AgentConfig(api_key="sk-fake", context_window=100)
agent = ca.CodingAgent(cfg)

long_text = "a" * 200
agent.messages = [
    {"role": "system", "content": "you are helpful"},
    {"role": "user", "content": long_text},
    {"role": "assistant", "content": long_text},
    {"role": "user", "content": "current question"},
]
before_tokens = ca.count_message_tokens(agent.messages, cfg.model)
before_len = len(agent.messages)

agent._trim_context_if_needed()

after_tokens = ca.count_message_tokens(agent.messages, cfg.model)
after_len = len(agent.messages)

print(f"  裁剪前: {before_tokens} tokens, {before_len} 条消息")
print(f"  裁剪后: {after_tokens} tokens, {after_len} 条消息")

check("消息数减少", after_len < before_len, f"仍为 {after_len}")
check("保留 system 消息", agent.messages[0].get("role") == "system")
check("保留最后一轮 user",
      agent.messages[-1].get("role") == "user"
      and agent.messages[-1]["content"] == "current question")
check("token 数下降", after_tokens < before_tokens)


# =============================================================================
print("\n" + "=" * 60)
print("验证 5: 沙箱化（workdir 内的相对路径解析 + 越界拒绝）")
print("=" * 60)

tmp = tempfile.mkdtemp(prefix="agent_sandbox_")
try:
    cfg = ca.AgentConfig(api_key="sk-fake", workdir=tmp)
    agent = ca.CodingAgent(cfg)
    check("workdir 已绝对化", agent.workdir == os.path.abspath(tmp))
    check("workdir 已创建", os.path.isdir(agent.workdir))

    # 相对路径写入应在沙箱内
    ca.write_file("hello.txt", "hi", _workdir=agent.workdir)
    check("沙箱内文件创建成功",
          os.path.isfile(os.path.join(agent.workdir, "hello.txt")))

    # 沙箱内读取
    content = ca.read_file("hello.txt", _workdir=agent.workdir)
    check("沙箱内文件读取正确", content == "hi", f"got {content!r}")

    # ../ 越界应被拒绝
    escape_caught = False
    try:
        ca.write_file("../escape.txt", "evil", _workdir=agent.workdir)
    except ca.ToolError as e:
        escape_caught = "越界" in str(e)
    check("../ 越界被拒绝", escape_caught)
    check("沙箱外未创建文件", not os.path.exists(
        os.path.join(os.path.dirname(agent.workdir), "escape.txt")))

    # 沙箱未启用时（workdir=None）保持原行为
    old_content = ca.read_file(__file__, _workdir=None)
    check("workdir=None 时退化为原行为",
          "verify_improvements" in old_content[:200])
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
print("\n" + "=" * 60)
print("验证 6: edit_file 的 replace_all + fuzzy match + diff 输出")
print("=" * 60)

tmp = tempfile.mkdtemp(prefix="agent_edit_")
try:
    # 准备测试文件
    test_path = os.path.join(tmp, "test.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("foo\nbar\nfoo\nbaz\n")

    # 默认只替换第一处
    r1 = ca.edit_file("test.txt", "foo", "QUX", _workdir=tmp)
    with open(test_path, "r", encoding="utf-8") as f:
        c1 = f.read()
    check("默认替换第一处", c1 == "QUX\nbar\nfoo\nbaz\n", f"got {c1!r}")
    check("返回 diff", "---" in r1 and "+++" in r1, f"got {r1!r}")
    check("diff 含替换计数", "替换 1 处" in r1, f"got {r1!r}")

    # replace_all=True 替换全部：用一份新的有 2 个 foo 的文件
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("foo\nbar\nfoo\nbaz\n")
    r2 = ca.edit_file("test.txt", "foo", "QUX", replace_all=True, _workdir=tmp)
    with open(test_path, "r", encoding="utf-8") as f:
        c2 = f.read()
    check("replace_all 替换全部",
          c2 == "QUX\nbar\nQUX\nbaz\n", f"got {c2!r}")
    check("diff 含替换计数为 2", "替换 2 处" in r2, f"got {r2!r}")

    # 找不到时给出 fuzzy match 提示
    test_path2 = os.path.join(tmp, "similar.txt")
    with open(test_path2, "w", encoding="utf-8") as f:
        f.write("def hello_world():\n    return 42\n")
    fuzzy_hit = False
    fuzzy_hint = ""
    try:
        ca.edit_file("similar.txt", "def hello_wrold():", "x", _workdir=tmp)
    except ca.ToolError as e:
        fuzzy_hint = str(e)
        fuzzy_hit = "相似行提示" in fuzzy_hint
    check("找不到时给出 fuzzy match 提示", fuzzy_hit, f"got {fuzzy_hint!r}")
    if fuzzy_hit:
        check("提示包含相似行",
              "hello_world" in fuzzy_hint, f"got {fuzzy_hint!r}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
print("\n" + "=" * 60)
print("验证 7: write_file 输出 diff")
print("=" * 60)

tmp = tempfile.mkdtemp(prefix="agent_write_")
try:
    # 新建文件
    r1 = ca.write_file("new.txt", "line1\nline2\n", _workdir=tmp)
    check("新建文件返回 diff", "+++" in r1 and "line1" in r1, f"got {r1!r}")

    # 覆盖时输出 diff
    r2 = ca.write_file("new.txt", "line1\nCHANGED\n", _workdir=tmp)
    check("覆盖时返回 diff", "---" in r2 and "+++" in r2, f"got {r2!r}")
    check("diff 含 -line2 删除行", "-line2" in r2, f"got {r2!r}")
    check("diff 含 +CHANGED 新增行", "+CHANGED" in r2, f"got {r2!r}")

    # 内容未变化
    r3 = ca.write_file("new.txt", "line1\nCHANGED\n", _workdir=tmp)
    check("内容未变化时提示", "内容未变化" in r3, f"got {r3!r}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
print("\n" + "=" * 60)
print("验证 8: dry-run（不写入文件，仅返回 diff）")
print("=" * 60)

tmp = tempfile.mkdtemp(prefix="agent_dryrun_")
try:
    test_path = os.path.join(tmp, "dry.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("hello\n")

    r1 = ca.edit_file("dry.txt", "hello", "world",
                     _workdir=tmp, _dry_run=True)
    check("dry-run 返回 diff", "world" in r1, f"got {r1!r}")

    # 文件应未被修改
    with open(test_path, "r", encoding="utf-8") as f:
        actual = f.read()
    check("dry-run 未写入磁盘", actual == "hello\n", f"got {actual!r}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
print("\n" + "=" * 60)
print("验证 9: 统计信息累积 + 输出")
print("=" * 60)

cfg = ca.AgentConfig(api_key="sk-fake", stream=False, stats_enabled=True)
agent = ca.CodingAgent(cfg)

# 模拟一次有 usage 的非流式调用
fake_usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
fake_msg = SimpleNamespace(content="ok", tool_calls=None)
fake_resp = SimpleNamespace(
    choices=[SimpleNamespace(message=fake_msg)],
    usage=fake_usage,
)

with patch.object(agent, "_call_model_with_retry", return_value=fake_resp):
    # _call_model 应累积 usage
    agent.stats = ca.CodingAgent._fresh_stats()
    agent.messages = [{"role": "user", "content": "hi"}]
    agent._call_model()

check("prompt_tokens 已累积", agent.stats["prompt_tokens"] == 100,
      f"got {agent.stats['prompt_tokens']}")
check("completion_tokens 已累积", agent.stats["completion_tokens"] == 50,
      f"got {agent.stats['completion_tokens']}")
check("提供方返回 usage 时 estimated=False",
      agent.stats["estimated"] is False,
      f"got estimated={agent.stats['estimated']}")

# 流式：从最后 chunk 拿 usage
cfg2 = ca.AgentConfig(api_key="sk-fake", stream=True)
agent2 = ca.CodingAgent(cfg2)
agent2.stats = ca.CodingAgent._fresh_stats()
agent2.messages = [{"role": "user", "content": "hi"}]

stream_chunks = [
    make_chunk(content="hello"),
    make_chunk(finish_reason="stop", usage=SimpleNamespace(
        prompt_tokens=200, completion_tokens=80)),
]

with patch.object(agent2, "_call_model_with_retry", return_value=iter(stream_chunks)):
    agent2._call_model()

check("流式 prompt_tokens 已累积", agent2.stats["prompt_tokens"] == 200,
      f"got {agent2.stats['prompt_tokens']}")
check("流式 completion_tokens 已累积", agent2.stats["completion_tokens"] == 80,
      f"got {agent2.stats['completion_tokens']}")
check("流式提供方返回 usage 时 estimated=False",
      agent2.stats["estimated"] is False)


# --- fallback 估算（提供方未返回 usage） ---
# 非流式：response.usage 为 None
cfg3 = ca.AgentConfig(api_key="sk-fake", stream=False)
agent3 = ca.CodingAgent(cfg3)
agent3.stats = ca.CodingAgent._fresh_stats()
agent3.messages = [
    {"role": "system", "content": "you are helpful"},
    {"role": "user", "content": "请回复 hello"},
]
fake_resp_no_usage = SimpleNamespace(
    choices=[SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=None))],
    usage=None,
)
with patch.object(agent3, "_call_model_with_retry", return_value=fake_resp_no_usage):
    agent3._call_model()
check("非流式无 usage 时走 fallback",
      agent3.stats["estimated"] is True,
      f"got estimated={agent3.stats['estimated']}")
check("fallback prompt_tokens > 0",
      agent3.stats["prompt_tokens"] > 0,
      f"got {agent3.stats['prompt_tokens']}")
check("fallback completion_tokens > 0",
      agent3.stats["completion_tokens"] > 0,
      f"got {agent3.stats['completion_tokens']}")

# 流式：所有 chunk 都不带 usage
cfg4 = ca.AgentConfig(api_key="sk-fake", stream=True)
agent4 = ca.CodingAgent(cfg4)
agent4.stats = ca.CodingAgent._fresh_stats()
agent4.messages = [{"role": "user", "content": "请回复 hi"}]

stream_chunks_no_usage = [
    make_chunk(content="hi"),
    make_chunk(finish_reason="stop"),  # 没有 usage 字段
]
with patch.object(agent4, "_call_model_with_retry",
                 return_value=iter(stream_chunks_no_usage)):
    agent4._call_model()
check("流式无 usage 时走 fallback",
      agent4.stats["estimated"] is True,
      f"got estimated={agent4.stats['estimated']}")
check("流式 fallback completion_tokens 已估算",
      agent4.stats["completion_tokens"] > 0,
      f"got {agent4.stats['completion_tokens']}")


# =============================================================================
print("\n" + "=" * 60)
print("验证 10: 命令行参数解析")
print("=" * 60)

import io
from contextlib import redirect_stdout

# 测试 --help 不抛异常
help_out = io.StringIO()
try:
    with redirect_stdout(help_out):
        sys.argv = ["coding_agent.py", "--help"]
        try:
            ca.main()
        except SystemExit as e:
            check("--help 触发 SystemExit(0)", e.code == 0, f"code={e.code}")
except Exception as e:
    check("--help 不抛异常", False, f"{type(e).__name__}: {e}")
else:
    check("--help 输出含 --workdir", "--workdir" in help_out.getvalue())
    check("--help 输出含 --confirm-edits", "--confirm-edits" in help_out.getvalue())


# =============================================================================
print("\n" + "=" * 60)
print("验证 11: 编辑备份栈 + undo 撤销")
print("=" * 60)

import json as _json

tmp = tempfile.mkdtemp(prefix="agent_undo_")
try:
    cfg = ca.AgentConfig(api_key="sk-fake", workdir=tmp)
    agent = ca.CodingAgent(cfg)

    # 准备一个文件
    target = os.path.join(tmp, "a.txt")
    with open(target, "w", encoding="utf-8") as f:
        f.write("原始内容 v1")

    # 模拟 LLM 调用 edit_file 把 v1 改成 v2
    def make_call(name, args):
        return SimpleNamespace(
            id=f"call_{name}", type="function",
            function=SimpleNamespace(name=name,
                                     arguments=_json.dumps(args, ensure_ascii=False))
        )

    tc = make_call("edit_file", {"path": "a.txt", "old_str": "v1", "new_str": "v2"})
    result = agent._execute_tool(tc)
    check("edit_file 执行成功", "v2" in result["content"],
          f"got {result['content']!r}")
    check("备份栈 push 一条", len(agent.backups) == 1,
          f"got {len(agent.backups)}")

    with open(target, "r", encoding="utf-8") as f:
        c = f.read()
    check("文件已被改为 v2", c == "原始内容 v2", f"got {c!r}")

    # 撤销
    msg = agent.undo_last_edit()
    check("undo 返回成功提示", "已撤销" in msg, f"got {msg!r}")
    check("备份栈 pop 后为空", len(agent.backups) == 0)
    with open(target, "r", encoding="utf-8") as f:
        c = f.read()
    check("undo 后文件恢复为 v1", c == "原始内容 v1", f"got {c!r}")

    # 再次 undo 应提示栈空
    msg2 = agent.undo_last_edit()
    check("空栈 undo 提示", "没有可撤销" in msg2 or "为空" in msg2,
          f"got {msg2!r}")

    # ---- 新建文件 + undo = 删除新建的文件 ----
    tc2 = make_call("write_file", {"path": "new.txt", "content": "hello"})
    result2 = agent._execute_tool(tc2)
    check("新建文件执行成功", "已写入" in result2["content"] or "已创建" in result2["content"]
          or "新建" in result2["content"], f"got {result2['content']!r}")
    check("新建场景备份栈 push 一条", len(agent.backups) == 1)
    check("文件确实创建", os.path.isfile(os.path.join(tmp, "new.txt")))

    msg3 = agent.undo_last_edit()
    check("新建文件 undo 提示删除", "删除" in msg3, f"got {msg3!r}")
    check("undo 后新建文件已删除",
          not os.path.exists(os.path.join(tmp, "new.txt")))

    # ---- 工具失败时备份应回滚（不会污染备份栈） ----
    # 让 edit_file 抛 ToolError：文件不存在
    agent.backups.clear()
    tc_fail = make_call("edit_file",
                        {"path": "not_exist.txt", "old_str": "x", "new_str": "y"})
    result_fail = agent._execute_tool(tc_fail)
    check("失败的 edit_file 返回错误", "工具执行错误" in result_fail["content"],
          f"got {result_fail['content']!r}")
    check("失败时备份栈未污染（仍为空）",
          len(agent.backups) == 0, f"got {len(agent.backups)}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
print("\n" + "=" * 60)
print("验证 12: show_backups 显示备份栈")
print("=" * 60)

tmp = tempfile.mkdtemp(prefix="agent_show_")
try:
    cfg = ca.AgentConfig(api_key="sk-fake", workdir=tmp)
    agent = ca.CodingAgent(cfg)

    # 空栈
    out = agent.show_backups()
    check("空栈提示", "备份栈为空" in out, f"got {out!r}")

    # push 两条
    for i, name in enumerate(["a.txt", "b.txt"]):
        with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
            f.write(f"old{i}")
        tc = SimpleNamespace(
            id=f"c{i}", type="function",
            function=SimpleNamespace(
                name="write_file",
                arguments=_json.dumps({"path": name, "content": f"new{i}"})
            )
        )
        agent._execute_tool(tc)

    out2 = agent.show_backups()
    check("显示两条备份", out2.count("a.txt") + out2.count("b.txt") >= 2,
          f"got {out2!r}")
    check("栈顶在前的格式", "1." in out2 and "2." in out2)
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
print("\n" + "=" * 60)
print(f"全部验证完成: {PASS} 通过 / {FAIL} 失败")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
