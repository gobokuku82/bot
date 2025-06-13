# test_router_chain.py
from agents.router_chain import router_chain  # 방금 만든 router_chain
from dotenv import load_dotenv
import os

load_dotenv()

def test_router():
    print("===== 하이브리드 라우팅 테스트 시작 =====")
    
    while True:
        user_input = input("💬 질문: ")
        if user_input.lower() in ["exit", "quit", "종료"]:
            break

        result = router_chain.invoke({"input": user_input})
        print("🧠 응답:", result)
        print("-" * 60)

if __name__ == "__main__":
    test_router()
