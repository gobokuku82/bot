import os
from typing import TypedDict, List
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.agent import RunnableAgent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from tools.filter_tool import parse_conditions, load_jsonl, filter_jsonl_by_condition
from tools.llm_tool import summarize_results
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 1. LLM 세팅
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# ✅ 2. Tool 정의

@tool
def filter_coupon_data(query: str) -> List[dict]:
    """질문에서 조건을 추출하고, JSONL 데이터를 조건에 맞게 필터링합니다."""
    cond = parse_conditions(query)
    data = load_jsonl("data/지역사랑상품권_긍정_부정전처리_cleaned.jsonl")
    results = filter_jsonl_by_condition(data, cond)
    return results[:30]  # 너무 많으면 일부만 리턴

@tool
def summarize_coupon_results(results: List[dict]) -> str:
    """필터링된 결과 리스트를 요약하여 설명해줍니다."""
    return summarize_results(results)

# ✅ 3. Tool 목록 정의
tools = [filter_coupon_data]

# ✅ 4. 멀티턴 메모리 설정
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ✅ 5. Runnable Agent 구성 # ✅ 기본 system prompt 명시 (필수)
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="당신은 지역사랑상품권에 대해 질문을 분석하고 도구를 사용해 응답하는 AI입니다."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),  # ✅ 문자열 input을 메시지로 변환
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


# ✅ 수정된 agent 구성
agent: Runnable = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# ✅ 6. AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
