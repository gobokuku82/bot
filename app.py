import streamlit as st
from agents.agent_executor import agent_executor
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="대 동 여 지 갑", layout="wide")
st.title("💬 대 동 여 지 갑")

# ✅ 세션 상태로 멀티턴 대화 유지
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ 사용자 입력 받기
user_input = st.chat_input("무엇이 궁금한가요? 예: '모바일 되는 충청도 지역상품권 알려줘'")

if user_input:
    # 대화 기록 저장
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Agent 실행
    with st.spinner("🤖 답변 생성 중..."):
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history
        })

    # 응답 저장
    st.session_state.chat_history.append(AIMessage(content=response["output"]))

# ✅ 대화 내용 출력
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)
