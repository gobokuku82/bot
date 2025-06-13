from tools.query_classifier import classify_query
from agents.agent_executor import agent_executor
# 추후 외부검색용 에이전트, 계산기 에이전트 등도 여기에 import 예정

def route_query(user_input: str):
    classification = classify_query(user_input)
    # print(f"[📌 분류 결과] {classification.type}")
    print(f"[📌 분류 결과] {classification.category}")


    # if classification.type == "internal_search":
    if classification.category == "internal_search":
        return internal_agent_executor.invoke({"input": user_input})
    
    # elif classification.type == "external_search":
    elif classification.category == "external_search":
        return "[🔍 외부 검색 기능] 아직 구현되지 않았습니다."
    
    # elif classification.type == "calculator":
    elif classification.category == "calculator":    
        return "[🧮 계산기 기능] 아직 구현되지 않았습니다."
    
    else:
        return "[❓ 기타 질문] 해당 질문은 처리할 수 없습니다."
