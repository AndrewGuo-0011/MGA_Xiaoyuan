from ast import arguments
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.message import Msg, TextBlock
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeMultiAgentFormatter, DashScopeChatFormatter
from agentscope.memory import InMemoryMemory, MemoryBase
from agentscope.tool import ToolResponse
from agentscope.pipeline import sequential_pipeline

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os, asyncio, json

load_dotenv()
_dashscope_api_key_temp = os.getenv("DASHSCOPE_API_KEY")
if not _dashscope_api_key_temp:
    raise ValueError("请检查DASHSCOPE_API_KEY环境变量是否设置正确")
# 确保类型安全，此时已经验证不为None
_dashscope_api_key: str = _dashscope_api_key_temp

class DebateConfig:
    """辩论智能体配置类
    
    负责管理辩论智能体的所有配置参数
    """

    # 自由辩论轮数
    DEBATE_ROUNDS = 4

    # 模型配置
    HOST_MODEL = "qwen-plus-latest"
    JUDGE_MODEL = "qwen-plus-latest"
    TEACHER_MODEL = "qwen-plus-latest"
    DEBATER_MODEL = "qwen-max-latest"
    
class TemplateGetPOVs(BaseModel):
    """获取正反两方的POVs"""

    pov_positive: str= Field(
        description="正方的立场观点，根据辩论主题提取。",
        )
    pov_negative:str= Field(
        description="反方的立场观点，根据辩论主题提取。",
        )

class TemplateTeacherSuggestion(BaseModel):
    """指导老师建议"""

    suggestion_positive: str= Field(
        description="给正方辩手的建议，包含：【核心论证维度】、【关键论点】、【表达建议】和【可能出现的核心交锋点以及攻防建议】",
        )
    suggestion_negative: str= Field(
        description="给反方辩手的建议，包含：【核心论证维度】、【关键论点】、【表达建议】和【可能出现的核心交锋点以及攻防建议】",
        )

class TemplateDebateRusult(BaseModel):
    """辩论结果"""

    scores: str = Field(
        description="双方辩手的分项得分和总分",
        )
    winner: str = Field(
        description="获胜方（总分更高者）",
        )
    pov_winner: str = Field(
        description="获胜方的辩论立场",
        )
    key_arguments: str = Field(
        description="获胜方的主要论点",
    )
    score_details: str = Field(
        description="获胜关键原因",
        )

class AgentFactory:
    """智能体工厂类
    
    负责创建和配置各种智能体
    """

    # 智能体创建
    def create_agent_host(self):
        return ReActAgent(
            name="主持人",
            sys_prompt=self._get_host_prompt(),
            model=DashScopeChatModel(
                model_name=DebateConfig.HOST_MODEL,
                api_key=_dashscope_api_key,
                stream=True,
                enable_thinking=False,
            ),
            formatter=DashScopeMultiAgentFormatter(),
            memory=InMemoryMemory(),
        )

    def create_agent_judge(self):
        return ReActAgent(
            name="评委",
            sys_prompt=self._get_judge_prompt(),
            model=DashScopeChatModel(
                model_name=DebateConfig.JUDGE_MODEL,
                api_key=_dashscope_api_key,
                stream=True,
                enable_thinking=True,
            ),
            formatter=DashScopeMultiAgentFormatter(),
            memory=InMemoryMemory(),
        )

    def create_agent_teacher(self):
        return ReActAgent(
            name="辩论教练",
            sys_prompt=self._get_teacher_prompt(),
            model=DashScopeChatModel(
                model_name=DebateConfig.TEACHER_MODEL,
                api_key=_dashscope_api_key,
                stream=True,
                enable_thinking=True,
            ),
            formatter=DashScopeChatFormatter(),
            memory=InMemoryMemory(),
        )

    def create_agent_debater_positive(self) -> ReActAgent:
        return ReActAgent(
            name="正方辩手",
            sys_prompt=self._get_debater_prompt_positive(),
            model=DashScopeChatModel(
                model_name=DebateConfig.DEBATER_MODEL,
                api_key=_dashscope_api_key,
                stream=True,
                enable_thinking=True,
            ),
            formatter=DashScopeMultiAgentFormatter(),
            memory=InMemoryMemory(),
        )

    def create_agent_debater_negative(self) -> ReActAgent:
        return ReActAgent(
            name="反方辩手",
            sys_prompt=self._get_debater_prompt_negative(),
            model=DashScopeChatModel(
                model_name=DebateConfig.DEBATER_MODEL,
                api_key=_dashscope_api_key,
                stream=True,
                enable_thinking=True,
            ),
            formatter=DashScopeChatFormatter(),
            memory=InMemoryMemory(),
        )

    # 智能体配置
    debate_subject: str=""
    pov_positive: str=""
    pov_negative: str=""
    suggestion_positive: str=""
    suggestion_negative: str=""

    def _get_host_prompt(self) -> str:
        """获取主持人智能体的系统提示词。"""
        return f"""
        你是一名专业、中立且具备丰富辩论经验的主持人，负责主持一场关于 **"{self.debate_subject}"** 的正式AI辩论赛。

        本场辩论共有四个角色：
        - 主持人（你）
        - 评委（独立智能体）
        - 辩论教练（独立智能体）
        - 正方辩手（独立智能体）
        - 反方辩手（独立智能体）

        你的职责仅限于**流程主持与环节引导**，不得参与辩论内容、表达立场或进行评判。所有评分与胜负判定将由独立的“评委智能体”完成。

        ---

        ### 【辩论流程指令】

        1. **开场与立场分配**
        - 首先，清晰宣布辩论题目：“本次辩论的题目是：{self.debate_subject}。”
        - 为双方分配明确且对立的立场（无需解释原因）

        2. **邀请辩论教练提供建议**
        - 宣布：“现在请辩论教练为双方辩手提供准备建议。”
        - 等待教练发言结束后，进入正式辩论环节。

        3. **正式辩论流程**（按顺序逐轮推进）

        **第一轮：立论环节**
        - 宣布：“第一轮开始，立论环节。请正方辩手首先陈述立场与主要论点。”
        - 正方发言后：“请反方辩手陈述立场与主要论点。”

        **第二轮：攻辩环节**
        - 宣布：“第二轮开始，攻辩环节。”
        - “请正方辩手向反方提出3-5个问题，要求反方正面回答，不得反问。”
        - 待反方回答后：“请反方辩手向正方提出3-5个问题，要求正方正面回答，不得反问。”

        **第三轮：自由辩论环节**
        - 宣布：“第三轮开始，自由辩论环节。双方将交替发言共 {DebateConfig.DEBATE_ROUNDS} 轮，由正方率先发言。”
        - 每轮依次引导：
            > “请正方发言。” → “请反方发言。” （重复至满轮次）

        **第四轮：总结陈词**
        - 宣布：“自由辩论结束，现在进入总结陈词环节，请双方辩手做最后陈述”

        ---

        ### 【主持规范】

        - 在每个环节开始前，必须明确提示当前环节名称（如“第二轮开始，攻辩环节”）。
        - 使用简洁、权威、中立的语言，不添加个人评论、解释或情感色彩。
        - 不对发言内容进行评价、纠正或补充。
        - 不介入辩论逻辑或内容质量判断——这些由后续的“评委智能体”独立完成。
        - 确保流程完整、节奏清晰，推动辩论有序进行。

        ---

        **示例开场语：**
        “各位好，欢迎来到本场辩论。本次辩论的题目是：{self.debate_subject}。
        现在请辩论教练为双方辩手提供准备建议。”
        """

    def _get_judge_prompt(self) -> str:
        """获取评委智能体的系统提示词。"""
        return f"""
        你是一位专业、中立的辩论赛评委智能体，具备深度语义理解、逻辑推理与多维度评估能力。
        你的任务是：在一场由两个AI辩手参与的标准辩论对战结束后，全面分析双方发言内容，从以下四个客观维度进行公正评分（每项满分10分，总分40分），并据此判定胜者。
        本次辩论的题目为：“{self.debate_subject}”

        评分维度如下：

        1. **立论清晰度**：立场是否明确，论证结构是否完整，核心观点是否条理清晰、易于理解。  
        2. **论据质量与充分性**：是否提供可靠、相关且有力的事实、数据、案例或逻辑推理来支撑主张；证据来源是否具说服力。  
        3. **反驳精准度与有效性**：是否准确识别对方论点中的漏洞或前提假设，并进行有逻辑、有针对性的驳斥；能否有效削弱对方立场。  
        4. **语言表达与逻辑连贯性**：语言是否准确、严谨，推理过程是否无矛盾、无跳跃，整体论述是否具有说服力和思维深度。

        请执行以下步骤：

        1. 分别为正方和反方在上述四个维度上独立评分，并为每项评分附上简要依据（一句话说明）。  
        2. 计算双方总分，宣布得分更高的一方为胜者（若平分，则指出“表现相当，难分胜负”）。  
        3. 用一段简洁文字总结胜方的：
        - 辩论立场（支持/反对辩题）
        - 核心主张
        - 3-5个最具说服力的主要论点
        - 获胜关键原因（如更强的逻辑链条、更有效的反驳、更高质量的证据等）

        要求：  
        - 评判基于内容本身，不考虑语气、情感或表演性因素。  
        - 所有判断必须紧扣发言内容，避免主观臆断。  
        - 语言简洁、专业、条理清晰。
        """

    def _get_teacher_prompt(self) -> str:
        """获取辩论教练智能体的系统提示词。"""
        return f"""
        你是一位资深辩论教练，擅长逻辑分析、论点构建和策略指导。现在有一场正式辩论即将开始，你需要根据给定的辩题，为正方和反方辩手分别提供专业的辩论指导。

        #辩论题目
        辩论主题：{self.debate_subject}
        官方给出的正方立场是：{self.pov_positive}，反方立场是：{self.pov_negative}

        请你完成以下任务：
        1. **为正方提供辩论策略指导**
        - 【核心论证维度】：列出3–5个正方可重点展开的价值、原则或事实维度（如效率、公平性、可行性、长期影响等）；
        - 【关键论点】：提供3–5条具体、有说服力的论点，要求结合事实依据、统计数据、典型案例或逻辑推理，避免空泛陈述；
        - 【表达建议】：建议适合正方的发言风格（如理性论证 / 情感共鸣 / 制度批判），并提示可引用的权威来源类型。

        2. **为反方提供对等策略指导**
        - 同样输出【核心论证维度】、【关键论点】和【表达建议】；
        - 确保反方论点能形成有效反驳，而非自说自话；
        - 鼓励从价值观冲突、现实障碍、 unintended consequences（意外后果）等角度切入。

        3. **预判核心交锋点并给出攻防建议**
        - 列出2–3个最可能成为辩论焦点的争议点（如“自由 vs 安全”、“短期成本 vs 长期收益”）；
        - 对每个交锋点：
            - 给出正方应如何辩护；
            - 给出反方应如何质疑或反击；
            - 建议使用何种论证方式（类比、归谬、数据反驳等）。

        【输出要求】
        - 所有论点必须基于事实或合理推论，禁止虚构数据；
        - 保持中立客观，不偏向任何一方；
        - 语言简洁专业，适合直接传递给AI辩手作为策略输入。
        - 结构化输出，格式如下:'''
        ##核心论证纬度
        纬度1、纬度2、纬度3、纬度4……

        ##关键论点
        论点1：……
        论点2：……
        ……

        ##表达建议
        ……

        ##可能出现的核心交锋点以及攻防建议
        交锋点1：……
        建议：……
        交锋点2：……
        建议：……
        ……'''
        """
    
    def _get_debater_prompt_positive(self) -> str:
        """获取正方辩手智能体的系统提示词。"""
        return f"""
        你是一名逻辑严谨、富有说服力的辩论专家，作为正方参与本次辩论。

        【辩题】
        {self.debate_subject}

        【你的立场】
        {self.pov_positive}

        【辩论教练的建议】
        辩论教练给出的参考建议如下：
        {self.suggestion_positive}

        【辩论流程】
        1. 一共进行4轮辩论，第一轮双方立论，以“总-分”形式表明各自的立场和主要观点，从正方开始；
        2. 第二轮为攻辩环节，由你开始针对反方的立场观点进行攻辩提问3-5个问题，反方作出正面回答，不得反问；
        3. 反方回答完你的提问后，针对你的立场观点进行攻辩提问3-5个问题，你作出正面回答，不得反问；
        4. 第三轮为自由辩论环节，你和对手自由交流{DebateConfig.DEBATE_ROUNDS}轮，你可以自由地提问、回答、反问或者对之前的发言做补充。在这一轮中，每次发言不超过200字；
        5. 自由辩论结束后，最后进行一轮总结陈词环节，按反方->正方的顺序做最后陈述。

        【你的任务】
        1. 综合分析辩题、立场和辩论教练的建议，确定你在本场辩论中的辩论方案
        2. 主动提出清晰有力的观点；
        3. 引用真实数据、评测结果或实际应用场景支撑论点；
        4. 精准回应反方质疑，体现交锋性；
        5. 辩论流程和秩序由主持人负责，不要在发言中谈及辩论流程。
        """

    def _get_debater_prompt_negative(self) -> str:
        """获取反方辩手智能体的系统提示词。"""
        return f"""
        你是一名批判性强、思维敏捷的辩论专家，作为反方参与本次辩论。

        【辩题】
        {self.debate_subject}

        【你的立场】
        {self.pov_negative}

        【辩论教练的建议】
        辩论教练给出的参考建议如下：
        {self.suggestion_negative}

        【辩论流程】
        1. 一共进行4轮辩论，第一轮双方立论，以“总-分”形式表明各自的立场和主要观点，从正方开始；
        2. 第二轮为攻辩环节，由正方开始针对你的立场观点进行攻辩提问3-5个问题，你作出正面回答，不得反问；
        3. 你回答完正方的提问后，针对正方的立场观点进行攻辩提问3-5个问题，正方作出正面回答，不得反问；
        4. 第三轮为自由辩论环节，你和对手自由交流{DebateConfig.DEBATE_ROUNDS}轮，你可以自由地提问、回答、反问或者对之前的发言做补充。在这一轮中，每次发言不超过200字；
        5. 自由辩论结束后，最后进行一轮总结陈词环节，按反方->正方的顺序做最后陈述。

        【你的任务】
        1. 综合分析辩题、立场和辩论教练的建议，确定你在本场辩论中的辩论方案
        2. 明确指出正方观点的局限或偏差；
        3. 结合权威评测、技术特性或用户实践提供反证；
        4. 回应时紧扣对方逻辑漏洞，避免泛泛而谈；
        5. 辩论流程和秩序由主持人负责，不要在发言中谈及辩论流程。
        """

async def start_debate(
    debate_subject: str,
) -> ToolResponse:
    """开始一场辩论
    
    Args:
        debate_subject (``str``):
            辩论的主题
    """

    # 引导词
    msg = Msg(
        name="小元",
        content=f"请开始一场关于“{debate_subject}”的辩论",
        role="user",
    )

    # 创建辩论需要的所有智能体，包括1个主持人、1个指导老师、2位辩手
    factory = AgentFactory()

    # 主持人分析辩论主题并提取正反双方的主观点
    factory.debate_subject = debate_subject
    host = factory.create_agent_host()
    msg = await host(msg, structured_model=TemplateGetPOVs)
    print(json.dumps(msg.metadata, indent=4, ensure_ascii=False))
    factory.pov_positive = str((msg.metadata or {}).get("pov_positive", ""))
    factory.pov_negative = str((msg.metadata or {}).get("pov_negative", ""))

    # 指导老师为双方辩手分析各自的主观点以及可供参考的主要论点
    teacher = factory.create_agent_teacher()
    msg = await teacher(msg, structured_model=TemplateTeacherSuggestion)
    print(json.dumps(msg.metadata, indent=4, ensure_ascii=False))
    factory.suggestion_positive = str((msg.metadata or {}).get("suggestion_positive", ""))
    factory.suggestion_negative = str((msg.metadata or {}).get("suggestion_negative", ""))

    # 双方辩手入场
    judge = factory.create_agent_judge()
    debater_positive = factory.create_agent_debater_positive()
    debater_nagative = factory.create_agent_debater_negative()
    
    # 正式开始辩论
    # 立论
    # print(f"辩论比赛正式开始！辩论的主题是：{debate_subject}")
    print("="*100)
    print("🛎️ 主持人发言：")
    msg = await host(msg)
    print(f"🔵正方辩手立论：")
    msg = await debater_positive(msg)
    # await judge.observe(msg)
    print("🛎️ 主持人发言：")
    msg = await host(msg)
    print(f"🔴反方辩手立论：")
    msg = await debater_nagative(msg)
    # await judge.observe(msg)
    # # Pipeline 语法糖
    # msg = await sequential_pipeline(
    #     agents=[host, debater_positive, host, debater_nagative]
    # )
    print("\n\n")

    # 攻辩
    print("="*100)
    print("🛎️ 主持人发言：")
    msg = await host(msg)
    print(f"🔵正方辩手攻辩：")
    msg = await debater_positive(msg)
    # await judge.observe(msg)
    print(f"🔴反方辩手回应：")
    msg = await debater_nagative(msg)
    # await judge.observe(msg)
    print("🛎️ 主持人发言：")
    msg = await host(msg)
    print(f"🔴反方辩手攻辩：")
    msg = await debater_nagative(msg)
    # await judge.observe(msg)
    print(f"🔵正方辩手回应：")
    msg = await debater_positive(msg)
    # await judge.observe(msg)
    # msg = await sequential_pipeline(
    #     agents=[host, debater_positive, debater_nagative, host, debater_nagative, debater_positive]
    # )
    print("\n\n")

    # 自由辩论
    print("="*100)
    print("🛎️ 主持人发言：")
    msg = await host(msg)
    for _ in range(DebateConfig.DEBATE_ROUNDS):
        print(f"🔵正方发言：")
        msg = await debater_positive(msg)
        # await judge.observe(msg)
        print(f"🔴反方发言：")
        msg = await debater_nagative(msg)
        # await judge.observe(msg)
        # msg = await sequential_pipeline(
        #     agents=[debater_positive, debater_nagative]
        # )
    print("\n\n")
    

    # 总结陈词
    print("="*100)
    # msg = Msg(
    #     name="主持人",
    #     content="请主持人引导双方进行总结陈词",
    #     role="user",
    # )
    print("🛎️ 主持人发言：")
    msg =  await host(msg)
    print(f"🔴反方总结：")
    msg = await debater_nagative(msg)
    # await judge.observe(msg)
    print(f"🔵正方总结：")
    msg = await debater_positive(msg)
    # await judge.observe(msg)
    # msg = await sequential_pipeline(
    #     agents = [host, debater_nagative, debater_positive],
    #     msg = Msg("主持人", "请主持人引导双方进行总结陈词", "user")
    # )
    print("\n\n")
    
    #评委评定结果
    print("="*100)
    msg = await host(msg)
    # print(await host.memory.get_memory())
    debater_history = host.memory.get_memory()
    print(debater_history)
    # judge.memory.load_state_dict(debater_history)
    
    print("⚖️ 评委宣布结果：")
    msg = await judge(msg, structured_model=TemplateDebateRusult)

    # 返回辩论结果
    msg_res=msg.get_content_blocks("text")[0]
    return ToolResponse(
        content=[msg_res]
    )
