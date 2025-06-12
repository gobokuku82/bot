import streamlit as st
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, Any, List

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 환경 변수 로드
load_dotenv()

# 커스텀 모듈 임포트
from utils.vector_db import VectorDBManager
from agents.local_currency_agent import LocalCurrencyAgent


def init_session_state():
    """세션 상태 초기화"""
    if 'vector_db_manager' not in st.session_state:
        st.session_state.vector_db_manager = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def load_models():
    """모델 및 에이전트 로드"""
    if st.session_state.vector_db_manager is None:
        with st.spinner("벡터 DB를 로드하고 있습니다..."):
            try:
                # 벡터 DB 관리자 초기화
                model_name = os.getenv('MODEL_NAME', 'nlpai-lab/KURE-v1')
                vector_db_path = os.getenv('VECTOR_DB_PATH', 'data/faiss_coupon_db')
                
                st.session_state.vector_db_manager = VectorDBManager(
                    model_name=model_name,
                    vector_db_path=vector_db_path
                )
                st.success("벡터 DB 로드 완료!")
                
                # 에이전트 초기화
                openai_api_key = os.getenv('OPENAI_API_KEY')
                if not openai_api_key:
                    st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
                    st.stop()
                
                st.session_state.agent = LocalCurrencyAgent(
                    vector_db_manager=st.session_state.vector_db_manager,
                    openai_api_key=openai_api_key
                )
                st.success("에이전트 초기화 완료!")
                
            except Exception as e:
                st.error(f"초기화 중 오류가 발생했습니다: {str(e)}")
                st.stop()


def display_statistics():
    """통계 정보 표시"""
    if st.session_state.vector_db_manager:
        stats = st.session_state.vector_db_manager.get_statistics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("전체 지역화폐", f"{stats.get('total_count', 0)}개")
        
        with col2:
            region_count = len(stats.get('region_stats', {}))
            st.metric("지역 수", f"{region_count}개")
        
        with col3:
            support_count = len(stats.get('support_stats', {}))
            st.metric("지원방식 종류", f"{support_count}개")
        
        # 지역별 통계 차트
        if stats.get('region_stats'):
            st.subheader("지역별 지역화폐 현황")
            region_df = pd.DataFrame(
                list(stats['region_stats'].items()),
                columns=['지역', '개수']
            )
            st.bar_chart(region_df.set_index('지역'))
        
        # 지원방식별 통계 차트
        if stats.get('support_stats'):
            st.subheader("지원방식별 현황")
            support_df = pd.DataFrame(
                list(stats['support_stats'].items()),
                columns=['지원방식', '개수']
            )
            st.bar_chart(support_df.set_index('지원방식'))


def display_chat_interface():
    """채팅 인터페이스 표시"""
    st.subheader("💬 지역화폐 검색 채팅")
    
    # 채팅 히스토리 표시
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # 사용자 입력
    user_input = st.chat_input("지역화폐에 대해 궁금한 것을 물어보세요!")
    
    if user_input:
        # 사용자 메시지 추가
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.write(user_input)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("검색 중..."):
                try:
                    response = st.session_state.agent.process_query(user_input)
                    st.write(response)
                    
                    # AI 응답 저장
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def display_example_queries():
    """예시 쿼리 표시"""
    st.subheader("💡 예시 질문")
    
    examples = [
        "모든 지역화폐 알려줘",
        "경기도의 지역화폐는?",
        "모바일만 지원되는 지역화폐는?",
        "충청도의 카드형 지역화폐 알려줘",
        "모바일 되는 것 중에서 경상도에서 쓸 수 있는 것은?",
        "지류형은 지원하지 않는 지역화폐는?",
        "전체 지역화폐 통계 보여줘",
        "부산의 지역화폐 정보",
        "카드형과 모바일 둘 다 지원하는 곳은?",
        "제주도 지역화폐 정보"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, example in enumerate(examples):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(example, key=f"example_{i}"):
                # 예시 질문을 채팅에 추가
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": example
                })
                
                # AI 응답 생성
                try:
                    response = st.session_state.agent.process_query(example)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.rerun()
                except Exception as e:
                    error_msg = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.rerun()


def main():
    """메인 함수"""
    st.set_page_config(
        page_title="지역화폐 검색 시스템",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 지역화폐 검색 시스템")
    st.markdown("---")
    
    # 세션 상태 초기화
    init_session_state()
    
    # 사이드바 - 설정 및 정보
    with st.sidebar:
        st.header("🔧 시스템 정보")
        
        # API 키 확인
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and len(openai_key) > 10:
            st.success("✅ OpenAI API 키 설정됨")
        else:
            st.error("❌ OpenAI API 키가 설정되지 않음")
            st.info("`.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.")
        
        st.markdown("---")
        
        # 모델 정보
        st.subheader("🤖 모델 정보")
        st.write(f"**임베딩 모델**: {os.getenv('MODEL_NAME', 'nlpai-lab/KURE-v1')}")
        st.write("**LLM**: GPT-4o-mini")
        st.write("**벡터 DB**: Faiss")
        
        st.markdown("---")
        
        # 채팅 히스토리 클리어 버튼
        if st.button("🗑️ 채팅 히스토리 삭제"):
            st.session_state.chat_history = []
            st.rerun()
        
        # 시스템 재시작 버튼
        if st.button("🔄 시스템 재시작"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # 모델 로드
    load_models()
    
    # 메인 컨텐츠
    if st.session_state.vector_db_manager and st.session_state.agent:
        # 탭 생성
        tab1, tab2, tab3 = st.tabs(["💬 채팅", "📊 통계", "💡 예시"])
        
        with tab1:
            display_chat_interface()
        
        with tab2:
            display_statistics()
        
        with tab3:
            display_example_queries()
    
    else:
        st.error("시스템 초기화가 완료되지 않았습니다.")


if __name__ == "__main__":
    main() 