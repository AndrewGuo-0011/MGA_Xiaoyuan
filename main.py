import json
from agentscope.agent import ReActAgent, UserAgent
from agentscope.embedding import DashScopeTextEmbedding
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import Mem0LongTermMemory
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import Toolkit, execute_python_code, execute_shell_command, ToolResponse

from workflow_debate import start_debate
from mcp_gaode import generate_travel_plan, stateless_client_gaode

from dotenv import load_dotenv
import os, asyncio, agentscope

agentscope.init(studio_url="http://localhost:3000")
load_dotenv()

async def register_tools() -> Toolkit:
    tk = Toolkit()
    tk.register_tool_function(execute_python_code)
    tk.register_tool_function(execute_shell_command)

    # 工作流
    tk.register_tool_function(tool_func=start_debate, func_description="只有当用户请求中包含'辩论'相关的字词时被调用")

    # 计划书
    tk.register_tool_function(tool_func=generate_travel_plan, func_description="为用户生成一份详尽的旅游计划")

    # 高德 mcp - 天气查询
    get_weather_by_city = await stateless_client_gaode.get_callable_function(
        func_name="maps_weather",
        wrap_tool_result=True,  # 确保返回 ToolResponse 而不是 CallToolResult
    )
    tk.register_tool_function(get_weather_by_city)  # type: ignore

    return tk

async def main():
    # 注册（工具）智能体
    toolkit = await register_tools()

    # 加载 AK
    api_key: str | None = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置或为空")

    # # 注册长期记忆
    # long_term_memory = Mem0LongTermMemory(
    #     agent_name="小元",
    #     user_name="User",
    #     model=DashScopeChatModel(
    #         model_name="qwen-plus",
    #         api_key=api_key,
    #         stream=False,
    #     ),
    #     embedding_model=DashScopeTextEmbedding(
    #         api_key=api_key,
    #         model_name="text-embedding-v4"
    #     ),
    #     on_disk=False,
    # )

    xiao_yuan = ReActAgent(
        name="小元",
        sys_prompt="你是一个多智能体助手，你的名字是小元。",
        model=DashScopeChatModel(
            model_name="qwen-plus",
            api_key=api_key,
            stream=True,
        ),
        memory=InMemoryMemory(),
        # long_term_memory=long_term_memory,
        # long_term_memory_mode="static_control",
        formatter=DashScopeMultiAgentFormatter(),
        toolkit=toolkit,
    )

    print(json.dumps(xiao_yuan.toolkit.get_json_schemas(), indent=4, ensure_ascii=False))

    user = UserAgent(name="User")

    msg = None
    while True:
        msg = await xiao_yuan(msg)
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break


if __name__ == "__main__":
    asyncio.run(main())
