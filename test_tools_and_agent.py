import os
from dotenv import load_dotenv

# tools
from tools.filter_tool import parse_conditions, load_jsonl, filter_jsonl_by_condition
from tools.llm_tool import summarize_results
from agents.agent_executor import route_query
# agent
from agents.agent_executor import agent_executor

# OpenAI API Key 확인
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "❌ OPENAI_API_KEY is not set in .env"

# ✅ 1. 질문 → 조건 파싱 테스트
def test_parse_conditions():
    q = "모바일 되는 경상도 지역 알려줘"
    cond = parse_conditions(q)
    print("✅ 조건 파싱 결과:", cond)

# ✅ 2. JSONL 로딩 테스트
def test_load_jsonl():
    data = load_jsonl("data/지역사랑상품권_긍정_부정전처리_cleaned.jsonl")
    print(f"✅ JSONL 로드 완료: {len(data)}개 항목")

# ✅ 3. 조건 필터링 테스트
def test_filter_by_condition():
    data = load_jsonl("data/지역사랑상품권_긍정_부정전처리_cleaned.jsonl")
    cond = {"지원방식": ["모바일"], "지역1": ["경북", "경남"]}
    filtered = filter_jsonl_by_condition(data, cond)
    print(f"✅ 필터링 결과: {len(filtered)}개")
    if filtered:
        print("샘플:", filtered[0])

# ✅ 4. 요약 테스트
def test_summarize():
    data = load_jsonl("data/지역사랑상품권_긍정_부정전처리_cleaned.jsonl")
    cond = {"지원방식": ["모바일"], "지역1": ["경북", "경남"]}
    filtered = filter_jsonl_by_condition(data, cond)
    summary = summarize_results(filtered[:10])
    print("✅ 요약 결과:", summary)

# ✅ 5. AgentExecutor 테스트
from langchain_core.messages import HumanMessage

def test_agent_executor():
    response = agent_executor.invoke({
    "input": "모바일 되는 충청도 지역상품권 알려줘"
})

    print("✅ 에이전트 응답:", response["output"])

if __name__ == "__main__":
    print("===== Tool/Agent 단위 테스트 시작 =====")
    test_parse_conditions()
    test_load_jsonl()
    test_filter_by_condition()
    test_summarize()
    test_agent_executor()
    print("===== ✅ 테스트 완료 =====")

def test_parse_gyeonggi():
    q = "경기도 상품권 종류 전부 알려줘"
    cond = parse_conditions(q)
    print("경기도 파싱 결과:", cond)

def test_router():
    print("===== Hybrid 라우터 테스트 =====")
    query = "모바일 되는 경기도 상품권 알려줘"
    result = route_query(query)
    print(result)
