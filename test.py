from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import DashScopeChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message._message_base import Msg
from agentscope.plan import PlanNotebook, SubTask
from agentscope.pipeline import sequential_pipeline
import asyncio

import os, dotenv

dotenv.load_dotenv()

ak: str | None = os.getenv("DASHSCOPE_API_KEY")
if not ak:
    raise ValueError("DASHSCOPE_API_KEY is not set")

plan_notebook_test = PlanNotebook()

async def manual_plan_specification() -> None:
    """手动计划规范示例。"""
    await plan_notebook_test.create_plan(
        name="智能体研究",
        description="对基于LLM的智能体进行全面研究",
        expected_outcome="一份Markdown格式的报告，回答三个问题：1. 什么是智能体？2. 智能体的当前技术水平是什么？3. 智能体的未来趋势是什么？",
        subtasks=[
            SubTask(
                name="搜索智能体相关调研论文",
                description=(
                    "在多个来源搜索调研论文，包括"
                    "Google Scholar、arXiv和Semantic Scholar。必须"
                    "在2021年后发表且引用数超过50。"
                ),
                expected_outcome="Markdown格式的论文列表",
            ),
            SubTask(
                name="阅读和总结论文",
                description="阅读前一步找到的论文，并总结关键点，包括定义、分类、挑战和关键方向。",
                expected_outcome="Markdown格式的关键点总结",
            ),
            SubTask(
                name="研究大公司的最新进展",
                description=(
                    "研究大公司的最新进展，包括但不限于Google、Microsoft、OpenAI、"
                    "Anthropic、阿里巴巴和Meta。查找官方博客或新闻文章。"
                ),
                expected_outcome="大公司的最新进展",
            ),
            SubTask(
                name="撰写报告",
                description="基于前面的步骤撰写报告，并回答预期结果中的三个问题。",
                expected_outcome=(
                    "一份Markdown格式的报告，回答三个问题：1. "
                    "什么是智能体？2. 智能体的当前技术水平"
                    "是什么？3. 智能体的未来趋势是什么？"
                ),
            ),
        ],
    )

    print("当前提示消息：\n")
    msg: Msg | None = await plan_notebook_test.get_current_hint()
    if msg:
        print(f"{msg.name}: {msg.content}")

asyncio.run(manual_plan_specification())

agent = ReActAgent(
    name="Friday",
    sys_prompt="You are a helpful assistant named Friday.",
    model=DashScopeChatModel(
        model_name="qwen-max",
        api_key=ak,
        stream=True,
    ),
    formatter=DashScopeChatFormatter(),
    memory=InMemoryMemory(),
    plan_notebook=plan_notebook_test,
    )

user = UserAgent(name="User")

async def main():
    msg = None
    while True:
        msg = await agent(msg)
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break

if __name__ == "__main__":
    asyncio.run(main())
