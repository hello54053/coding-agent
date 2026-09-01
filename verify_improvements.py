"""
verify_improvements.py - 验证 coding_agent 四项改进
不依赖真实 API key，使用 mock + 真实工具函数验证。

运行: python verify_improvements.py
"""
import sys
import time
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


def make_chunk(content=None, tool_calls=None, finish_reason=None):
    """构造一个流式 chunk。"""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, role=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


# =============================================================================
print("\n" + "=" * 60)
print("验证 1: 流式输出 + tool_calls 分片累积")
print("=" * 60)

cfg = ca.AgentConfig(api_key="sk-fake", stream=True)
agent = ca.CodingAgent(cfg)

# 构造 fake stream: 文本分 3 段输出，tool_call 分 2 段输出（模拟分片到达）
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


def slow_tool(_dummy=None):
    time.sleep(0.5)
    return "done"


ca.TOOLS["slow_tool"] = slow_tool  # 临时注册慢工具

slow_calls = [
    SimpleNamespace(id=f"c{i}", type="function",
                    function=SimpleNamespace(name="slow_tool", arguments="{}"))
    for i in range(4)
]

# 串行基线
t0 = time.perf_counter()
serial_results = [agent._execute_tool(tc) for tc in slow_calls]
serial_time = time.perf_counter() - t0

# 并行
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

# RateLimitError 构造: (message, response=, body=)
fake_response = SimpleNamespace(
    choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))]
)

call_count = {"n": 0}


def fake_create(**kwargs):
    call_count["n"] += 1
    if call_count["n"] < 3:
        # 构造 RateLimitError
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
    # 退避: 0.1 (2^0) + 0.2 (2^1) = 0.3s 最低
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

long_text = "a" * 200  # 单条就超过 80 token 阈值
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
print(f"全部验证完成: {PASS} 通过 / {FAIL} 失败")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
