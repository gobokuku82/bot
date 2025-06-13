from agents.router_agent import route_query

if __name__ == "__main__":
    query = input("💬 사용자 질문: ")
    response = route_query(query)
    print("\n[💡 응답 결과]")
    print(response)
    
if __name__ == "__main__":
    query = "충청도 지역 맛집 추천해줘"
    print(f"💬 사용자 질문: {query}")

    domain = route_query(query)
    print(f"[📌 분류 결과] {domain}")

    result = handle_query_with_routing(query)
    print("\n[💡 응답 결과]")
    print(result)
