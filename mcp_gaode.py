from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import TextBlock
from agentscope.model import DashScopeChatModel
from agentscope.mcp import HttpStatelessClient, HttpStatefulClient
from agentscope.plan import Plan, PlanNotebook, SubTask
from agentscope.agent import ReActAgent, UserAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_shell_command, execute_python_code
from agentscope.tool import ToolResponse

from dotenv import load_dotenv
import os, asyncio, json

load_dotenv()

amap_maps_api_key: str | None = os.getenv("AMAP_MAPS_API_KEY")
if not amap_maps_api_key:
    raise ValueError("请检查AMAP_MAPS_API_KEY配置")

dashscope_api_key_temp: str | None = os.getenv("DASHSCOPE_API_KEY")
if not dashscope_api_key_temp:
    raise ValueError("请检查DASHSCOPE_API_KEY配置")
dashscope_api_key = dashscope_api_key_temp

stateful_client_gaode = HttpStatefulClient(
    name="map_service_stateless",
    transport="streamable_http",
    url=f"https://mcp.amap.com/mcp?key={amap_maps_api_key}"
)
stateless_client_gaode = HttpStatelessClient(
    name="map_service_stateless",
    transport="streamable_http",
    url=f"https://mcp.amap.com/mcp?key={amap_maps_api_key}"
)

# await tk.register_mcp_client(stateless_client_gaode, group_name="map_services")

class GaodePlans:
    mcp = Toolkit()
    mcp.create_tool_group(
        group_name="map_services",
        description="高德地图服务",
        active=True,
    )
    travel_plannotebook = PlanNotebook()

    async def create_plan_travlel_plan(self):
        await self.travel_plannotebook.create_plan(
            name="生成旅游计划",
            description="根据用户的需求，生成一份详细的旅游计划",
            expected_outcome="一份Markdown格式的详细旅游计划",
            subtasks=[
                SubTask(
                    name="获取用户需求",
                    description="获取用户输入的旅行需求，如目的地、预算、时间等",
                    expected_outcome="用户输入的旅行需求",
                ),
                SubTask(
                    name="规划旅行路线",
                    description="根据用户需求，规划旅行路线，并选择合适的酒店、 Transport、 sightseeing spots",
                    expected_outcome="",
                ),
            ]
        )

async def generate_travel_plan(query: str):
    """生成一份旅游计划

    Args:
        query (`str`):
            用户输入的旅行需求，如目的地、预算、时间等
    """
    
    # 计划书模板
    plan_notebook = PlanNotebook()
    await plan_notebook.create_plan(
            name="生成旅游计划",
            description=f"根据用户的需求“{query}”，生成一份详细的旅游计划",
            expected_outcome="一份Markdown格式的详细旅游计划",
            subtasks=[
                SubTask(
                    name="明确旅行基本信息",
                    description="""
                    获取用户旅行的核心参数，作为后续规划的基础。包括：
                    目的地
                    旅行天数
                    出行时间（具体的日期或季节，会影响天气、人流、景点开放状态等）
                    出行人数及人员构成（单人/情侣/团队/家庭）
                    出发地""",
                    expected_outcome="Markdown格式的旅行基本信息",
                ),
                SubTask(
                    name="了解用户偏好与限制",
                    description="""
                    收集用户个性化需求。包括：
                    旅行风格偏好（美食探索/文化历史/自然风光/娱乐购物）
                    是否有必去景点或活动（如：一定要去欢乐海岸、吃火锅）
                    饮食偏好或禁忌（素食/辣食/过敏源）
                    住宿偏好（星级酒店/民宿）
                    预算范围（经济型/中高端/不限）
                    """,
                    expected_outcome="Markdown格式的用户偏好与限制",
                ),
                SubTask(
                    name="生成每日行程框架",
                    description="""
                    构建构建初步的时间结构，合理分配每天的活动区域和节奏。内容包括：
                    按天划分大致安排（如：第1天：xx区西部景点；第2天：东部+返程；）
                    每日主题（如：科技文化日/海滨休闲日/娱乐购物日）

                    要注意区域集中度优化（减少跨区奔波）和时间节奏的控制（避免过紧或过松）
                    """,
                    expected_outcome="Markdown格式的每日行程框架",
                ),
                SubTask(
                    name="填充具体景点与活动",
                    description="""
                    根据前三步的信息，选择匹配度高的具体景点和体验项目。内容包括：
                    推荐每个时间段的具体地点（含具体时间段、名称、简介、亮点）
                    标注是否需预约、门票价格、开放时间
                    加入特色体验（如：南头古城Citywalk、成都川剧变脸演出）
                    """,
                    expected_outcome="Markdown表格形式的每日具体景点活动安排",
                ),
                SubTask(
                    name="规划交通与动线",
                    description="""
                    规划可行的移动路径，降低通勤成本与时间。内容包括：
                    各点之间的交通方式建议（地铁/公交/打车/步行）
                    预估通勤时间和费用
                    提供路线图或顺序优化建议
                    若跨城，加入大交通建议（高铁/航班/自驾）
                    """,
                    expected_outcome="简洁的语言表述各地点之间的线路和通行方式",
                ),
                SubTask(
                    name="安排餐饮与住宿",
                    description="""
                    提供实用的生活配套建议。内容包括：
                    每日三餐推荐（结合口味偏好、地理位置、餐厅评价）
                    住宿推荐（2-3个选项，包含价格区间、位置优势）
                    """,
                    expected_outcome="Markdown表格形式的食宿推荐信息",
                ),
                SubTask(
                    name="输出完整的旅游计划并保存",
                    description="""
                    基于前面的步骤，整合所有信息，生成一份Markdow格式的结构化的详细旅游计划，并在用户确认后保存到本地指定路径
                    """,
                    expected_outcome="一份Markdown格式的详细旅游计划，包含出行基本信息、用户偏好、每日详细景点活动安排和通勤方式、每日食宿安排、一些额外的注意事项和建议、整个旅游计划的总预估成本",
                ),
            ]
        )

    # 注册tools和mcp
    tk = Toolkit()
    tk.register_tool_function(execute_shell_command)
    tk.register_tool_function(execute_python_code)
    
    # 连接有状态MCP客户端
    # await stateful_client_gaode.connect()
    await tk.register_mcp_client(stateless_client_gaode)

    agent = ReActAgent(
        name="小徳",
        sys_prompt="""你是高德地图智能助手，你的名字叫小徳。你要根据用户的要求生成一个旅游计划。
        并将最终的Markdown格式的旅游计划保存到本地用户的桌面路径下，具体路径根据操作系统自动确定：
            - macOS: `/Users/{username}/Desktop/`
            - Windows: `C:\\Users\\{username}\\Desktop\\`
            - Linux: `/home/{username}/Desktop/`
        请确保有写入权限，并以 Markdown 格式保存为 `travel_plan.md`。

        文件保存完后，请在结尾处附上`Travel plan generation done!`
        """,
        model=DashScopeChatModel(
            model_name="qwen-plus",
            api_key=dashscope_api_key,
            stream=True,
            enable_thinking=False,
        ),
        formatter=DashScopeChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=tk,
        plan_notebook=plan_notebook,
    )

    user = UserAgent(name="User")
    msg = None
    while True:
        msg = await agent(msg)
        assistant_msg = msg.get_text_content()
        if assistant_msg and "Travel plan generation done!" in assistant_msg:
            return ToolResponse(content=[TextBlock(type="text", text="Travel plan generation done!")])
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            return ToolResponse(content=[TextBlock(type="text", text="Task stoped by user.")])
    

