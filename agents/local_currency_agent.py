import os
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import json
import re

# 도구 임포트
from tools.search_tools import (
    VectorSearchTool, MetadataFilterTool, RegionSearchTool, 
    SupportTypeSearchTool, AllDataTool, StatisticsTool
)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    search_results: Optional[str]
    query_analysis: Optional[Dict[str, Any]]


class LocalCurrencyAgent:
    def __init__(self, vector_db_manager, openai_api_key: str):
        """
        지역화폐 검색 에이전트 초기화
        
        Args:
            vector_db_manager: 벡터 DB 관리자
            openai_api_key: OpenAI API 키
        """
        self.vector_db_manager = vector_db_manager
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0
        )
        
        # 도구 초기화
        self.tools = [
            VectorSearchTool(vector_db_manager),
            MetadataFilterTool(vector_db_manager),
            RegionSearchTool(vector_db_manager),
            SupportTypeSearchTool(vector_db_manager),
            AllDataTool(vector_db_manager),
            StatisticsTool(vector_db_manager)
        ]
        
        # 도구 매핑
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # 그래프 구성
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """LangGraph 그래프 생성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("search", self._search)
        workflow.add_node("generate_response", self._generate_response)
        
        # 엣지 추가
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "search")
        workflow.add_edge("search", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """쿼리 분석 노드"""
        last_message = state["messages"][-1]
        query = last_message.content
        
        # 쿼리 분석 프롬프트
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""당신은 지역화폐 검색 쿼리를 분석하는 전문가입니다.
사용자의 질문을 분석하여 적절한 검색 전략을 결정하세요.

분석해야 할 요소:
1. 검색 의도 (구체적 검색, 전체 목록, 조건부 검색, 통계)
2. 지역 정보 (특정 지역명이 있는지)
3. 지원 방식 (지류형, 모바일, 카드형)
4. 제외 조건 (지원하지 않는 방식)
5. 복합 조건 (여러 조건의 조합)

가능한 검색 유형:
- vector_search: 자연어 검색
- region_search: 지역별 검색
- support_type_search: 지원방식별 검색
- metadata_filter: 복합 조건 검색
- get_all_data: 전체 목록
- get_statistics: 통계 정보

JSON 형태로 분석 결과를 반환하세요."""),
            HumanMessage(content=f"질문: {query}")
        ])
        
        response = self.llm.invoke(analysis_prompt.format_messages())
        
        try:
            # JSON 파싱 시도
            analysis = json.loads(response.content)
        except:
            # JSON 파싱 실패시 기본 벡터 검색으로 설정
            analysis = {
                "search_type": "vector_search",
                "query": query,
                "parameters": {"k": 10}
            }
        
        state["query_analysis"] = analysis
        return state
    
    def _search(self, state: AgentState) -> AgentState:
        """검색 실행 노드"""
        analysis = state["query_analysis"]
        search_type = analysis.get("search_type", "vector_search")
        
        # 키워드 기반 검색 로직
        last_message = state["messages"][-1]
        query = last_message.content.lower()
        
        try:
            if "모든" in query or "전체" in query or "모든" in query:
                # 전체 목록 요청
                if "통계" in query or "현황" in query:
                    tool = self.tool_map["get_statistics"]
                    results = tool._run()
                else:
                    tool = self.tool_map["get_all_data"]
                    results = tool._run()
            
            elif any(region in query for region in ["경기", "경북", "경남", "전북", "전남", "충북", "충남", "강원", "제주", "울산", "세종", "광주", "대전", "대구", "인천", "부산"]):
                # 지역별 검색
                region = None
                support_type = None
                
                # 지역 추출
                for r in ["경기", "경북", "경남", "전북", "전남", "충북", "충남", "강원", "제주", "울산", "세종", "광주", "대전", "대구", "인천", "부산"]:
                    if r in query:
                        region = r
                        break
                
                # 지원 방식 추출
                if "모바일" in query:
                    support_type = "모바일"
                elif "카드" in query:
                    support_type = "카드형"
                elif "지류" in query:
                    support_type = "지류형"
                
                tool = self.tool_map["region_search"]
                results = tool._run(region=region, support_type=support_type)
            
            elif any(support in query for support in ["모바일", "카드", "지류"]):
                # 지원 방식별 검색
                support_types = []
                exclude_types = []
                
                if "모바일" in query:
                    if "안" in query or "제외" in query or "빼고" in query:
                        exclude_types.append("모바일")
                    else:
                        support_types.append("모바일")
                
                if "카드" in query:
                    if "안" in query or "제외" in query or "빼고" in query:
                        exclude_types.append("카드형")
                    else:
                        support_types.append("카드형")
                
                if "지류" in query:
                    if "안" in query or "제외" in query or "빼고" in query:
                        exclude_types.append("지류형")
                    else:
                        support_types.append("지류형")
                
                if support_types:
                    tool = self.tool_map["support_type_search"]
                    results = tool._run(support_types=support_types, exclude_types=exclude_types if exclude_types else None)
                else:
                    # 기본 벡터 검색
                    tool = self.tool_map["vector_search"]
                    results = tool._run(query=query, k=10)
            
            elif "통계" in query or "현황" in query or "얼마나" in query:
                # 통계 정보
                tool = self.tool_map["get_statistics"]
                results = tool._run()
            
            else:
                # 기본 벡터 검색
                tool = self.tool_map["vector_search"]
                results = tool._run(query=query, k=10)
            
            state["search_results"] = results
        
        except Exception as e:
            state["search_results"] = f"검색 중 오류가 발생했습니다: {str(e)}"
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """응답 생성 노드"""
        last_message = state["messages"][-1]
        query = last_message.content
        search_results = state["search_results"]
        
        # 응답 생성 프롬프트
        response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""당신은 지역화폐 정보를 제공하는 친절한 도우미입니다.
검색 결과를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요.

응답 가이드라인:
1. 검색 결과를 요약하여 핵심 정보를 먼저 제시
2. 구체적인 지역화폐 정보를 명확하게 정리
3. 필요시 추가 질문을 유도
4. 한국어로 자연스럽게 응답
5. 검색 결과가 많은 경우 주요 항목만 선별하여 제시

응답 형식:
- 요약
- 주요 지역화폐 목록 (최대 10개)
- 필요시 추가 정보나 추천"""),
            HumanMessage(content=f"""사용자 질문: {query}

검색 결과:
{search_results}

위 정보를 바탕으로 사용자에게 도움이 되는 응답을 생성해주세요.""")
        ])
        
        response = self.llm.invoke(response_prompt.format_messages())
        
        # 응답 메시지 추가
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    def process_query(self, query: str) -> str:
        """쿼리 처리"""
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            search_results=None,
            query_analysis=None
        )
        
        final_state = self.graph.invoke(initial_state)
        
        # 마지막 AI 메시지 반환
        for message in reversed(final_state["messages"]):
            if isinstance(message, AIMessage):
                return message.content
        
        return "죄송합니다. 처리 중 오류가 발생했습니다." 