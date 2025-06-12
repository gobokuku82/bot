#!/usr/bin/env python3
"""
지역화폐 검색 시스템 테스트 스크립트
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.vector_db import VectorDBManager
from agents.local_currency_agent import LocalCurrencyAgent


def test_vector_db():
    """벡터 DB 테스트"""
    print("=" * 50)
    print("벡터 DB 테스트 시작")
    print("=" * 50)
    
    try:
        # 벡터 DB 초기화
        vector_db = VectorDBManager()
        print(f"✅ 벡터 DB 로드 성공: {len(vector_db.metadata)}개 문서")
        
        # 통계 정보 테스트
        stats = vector_db.get_statistics()
        print(f"📊 통계 정보:")
        print(f"   - 전체 문서 수: {stats['total_count']}")
        print(f"   - 지역 수: {len(stats['region_stats'])}")
        print(f"   - 지원방식 종류: {len(stats['support_stats'])}")
        
        # 기본 검색 테스트
        print("\n🔍 벡터 검색 테스트:")
        results = vector_db.search("경기도 지역화폐", k=3)
        for i, result in enumerate(results[:3], 1):
            print(f"   {i}. {result['metadata']['이름']} ({result['score']:.3f})")
        
        # 지역별 필터링 테스트
        print("\n🏛️ 지역별 필터링 테스트:")
        results = vector_db.filter_by_metadata({"지역1": "제주"})
        print(f"   제주도 지역화폐: {len(results)}개")
        for result in results:
            print(f"   - {result['metadata']['이름']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 벡터 DB 테스트 실패: {str(e)}")
        return False


def test_agent():
    """에이전트 테스트"""
    print("\n" + "=" * 50)
    print("에이전트 테스트 시작")
    print("=" * 50)
    
    try:
        # API 키 확인
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("❌ OpenAI API 키가 설정되지 않았습니다.")
            return False
        
        # 벡터 DB 초기화
        vector_db = VectorDBManager()
        
        # 에이전트 초기화
        agent = LocalCurrencyAgent(vector_db, openai_api_key)
        print("✅ 에이전트 초기화 성공")
        
        # 테스트 쿼리들
        test_queries = [
            "제주도 지역화폐 알려줘",
            "모바일 지원되는 지역화폐는?",
            "경기도의 카드형 지역화폐는?",
            "전체 지역화폐 통계 보여줘"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🤖 테스트 {i}: {query}")
            print("-" * 30)
            
            try:
                response = agent.process_query(query)
                print(f"응답: {response[:200]}...")  # 응답의 첫 200자만 출력
                print("✅ 성공")
            except Exception as e:
                print(f"❌ 실패: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 에이전트 테스트 실패: {str(e)}")
        return False


def test_search_tools():
    """검색 도구 테스트"""
    print("\n" + "=" * 50)
    print("검색 도구 테스트 시작")
    print("=" * 50)
    
    try:
        from tools.search_tools import (
            VectorSearchTool, RegionSearchTool, SupportTypeSearchTool,
            AllDataTool, StatisticsTool
        )
        
        # 벡터 DB 초기화
        vector_db = VectorDBManager()
        
        # 도구들 초기화
        vector_tool = VectorSearchTool(vector_db)
        region_tool = RegionSearchTool(vector_db)
        support_tool = SupportTypeSearchTool(vector_db)
        all_data_tool = AllDataTool(vector_db)
        stats_tool = StatisticsTool(vector_db)
        
        print("✅ 모든 도구 초기화 성공")
        
        # VectorSearchTool 테스트
        print("\n🔍 벡터 검색 도구 테스트:")
        result = vector_tool._run("부산 지역화폐", k=2)
        print(f"   결과 길이: {len(result)} 문자")
        
        # RegionSearchTool 테스트
        print("\n🏛️ 지역 검색 도구 테스트:")
        result = region_tool._run("강원", "카드형")
        print(f"   결과 길이: {len(result)} 문자")
        
        # SupportTypeSearchTool 테스트
        print("\n💳 지원방식 검색 도구 테스트:")
        result = support_tool._run(["모바일"])
        print(f"   결과 길이: {len(result)} 문자")
        
        # StatisticsTool 테스트
        print("\n📊 통계 도구 테스트:")
        result = stats_tool._run()
        print(f"   결과 길이: {len(result)} 문자")
        
        return True
        
    except Exception as e:
        print(f"❌ 검색 도구 테스트 실패: {str(e)}")
        return False


def main():
    """메인 테스트 함수"""
    print("🧪 지역화폐 검색 시스템 테스트")
    print("=" * 60)
    
    # 환경 변수 확인
    print("📋 환경 변수 확인:")
    required_vars = ['OPENAI_API_KEY', 'MODEL_NAME', 'VECTOR_DB_PATH']
    all_set = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            display_value = f"{value[:10]}..." if len(value) > 10 else value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ❌ {var}: 설정되지 않음")
            all_set = False
    
    if not all_set:
        print("\n❌ 필수 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일을 확인해주세요.")
        return
    
    # 테스트 실행
    tests = [
        ("벡터 DB", test_vector_db),
        ("검색 도구", test_search_tools),
        ("에이전트", test_agent)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {str(e)}")
            results[test_name] = False
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("   이제 'streamlit run app.py' 명령어로 애플리케이션을 실행할 수 있습니다.")
    else:
        print("⚠️  일부 테스트가 실패했습니다.")
        print("   README.md 파일의 문제 해결 섹션을 참조해주세요.")


if __name__ == "__main__":
    main() 