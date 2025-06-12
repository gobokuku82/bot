from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import json
import re


class VectorSearchInput(BaseModel):
    query: str = Field(description="검색할 쿼리 문자열")
    k: int = Field(default=10, description="반환할 결과 수")


class MetadataFilterInput(BaseModel):
    filters: Dict[str, Any] = Field(description="필터링 조건 딕셔너리")


class RegionFilterInput(BaseModel):
    region: str = Field(description="검색할 지역명 (예: 경기, 경북, 충남 등)")
    support_type: Optional[str] = Field(default=None, description="지원 방식 (지류형, 모바일, 카드형)")


class SupportTypeFilterInput(BaseModel):
    support_types: List[str] = Field(description="검색할 지원 방식 리스트 (지류형, 모바일, 카드형)")
    exclude_types: Optional[List[str]] = Field(default=None, description="제외할 지원 방식 리스트")


class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = "벡터 검색을 통해 지역화폐 정보를 검색합니다. 자연어 쿼리를 입력하면 관련된 지역화폐를 찾아줍니다."
    args_schema: type = VectorSearchInput
    vector_db_manager: object = None
    
    def __init__(self, vector_db_manager):
        super().__init__()
        self.vector_db_manager = vector_db_manager
    
    def _run(self, query: str, k: int = 10) -> str:
        """벡터 검색 실행"""
        try:
            results = self.vector_db_manager.search(query, k)
            if not results:
                return "검색 결과가 없습니다."
            
            output = f"'{query}' 검색 결과 ({len(results)}개):\n\n"
            for result in results:
                # 메타데이터 구조 확인 후 추출
                metadata = result['metadata']
                if 'metadata' in metadata:  # 중첩된 구조인 경우
                    actual_metadata = metadata['metadata']
                else:
                    actual_metadata = metadata
                
                output += f"순위 {result['rank']}: {actual_metadata['이름']}\n"
                output += f"  - 지역: {actual_metadata['지역1']} {actual_metadata['지역2']}\n"
                output += f"  - 지원방식: {', '.join(actual_metadata['지원방식'])}\n"
                if actual_metadata['비지원방식']:
                    output += f"  - 비지원방식: {', '.join(actual_metadata['비지원방식'])}\n"
                output += f"  - 유사도: {result['score']:.3f}\n"
                output += f"  - 상세정보: {actual_metadata['링크']}\n\n"
            
            return output
            
        except Exception as e:
            return f"검색 중 오류가 발생했습니다: {str(e)}"


class MetadataFilterTool(BaseTool):
    name: str = "metadata_filter"
    description: str = "메타데이터 조건으로 지역화폐를 필터링합니다. 정확한 조건 매칭에 사용됩니다."
    args_schema: type = MetadataFilterInput
    vector_db_manager: object = None
    
    def __init__(self, vector_db_manager):
        super().__init__()
        self.vector_db_manager = vector_db_manager
    
    def _run(self, filters: Dict[str, Any]) -> str:
        """메타데이터 필터링 실행"""
        try:
            results = self.vector_db_manager.filter_by_metadata(filters)
            if not results:
                return f"필터 조건 '{filters}'에 맞는 결과가 없습니다."
            
            output = f"필터 결과 ({len(results)}개):\n\n"
            for result in results:
                # 메타데이터 구조 확인 후 추출
                metadata = result['metadata']
                if 'metadata' in metadata:  # 중첩된 구조인 경우
                    actual_metadata = metadata['metadata']
                else:
                    actual_metadata = metadata
                
                output += f"{result['rank']}. {actual_metadata['이름']}\n"
                output += f"  - 지역: {actual_metadata['지역1']} {actual_metadata['지역2']}\n"
                output += f"  - 지원방식: {', '.join(actual_metadata['지원방식'])}\n"
                if actual_metadata['비지원방식']:
                    output += f"  - 비지원방식: {', '.join(actual_metadata['비지원방식'])}\n"
                output += f"  - 상세정보: {actual_metadata['링크']}\n\n"
            
            return output
            
        except Exception as e:
            return f"필터링 중 오류가 발생했습니다: {str(e)}"


class RegionSearchTool(BaseTool):
    name: str = "region_search"
    description: str = "특정 지역의 지역화폐를 검색합니다. 지역명과 선택적으로 지원 방식을 지정할 수 있습니다."
    args_schema: type = RegionFilterInput
    vector_db_manager: object = None
    
    def __init__(self, vector_db_manager):
        super().__init__()
        self.vector_db_manager = vector_db_manager
    
    def _run(self, region: str, support_type: Optional[str] = None) -> str:
        """지역별 검색 실행"""
        try:
            filters = {}
            
            # 지역 필터 추가
            if region:
                filters['지역1'] = region
            
            # 지원 방식 필터 추가
            if support_type:
                filters['지원방식'] = [support_type]
            
            results = self.vector_db_manager.filter_by_metadata(filters)
            
            if not results:
                condition = f"지역: {region}"
                if support_type:
                    condition += f", 지원방식: {support_type}"
                return f"조건 '{condition}'에 맞는 결과가 없습니다."
            
            condition = f"{region}"
            if support_type:
                condition += f"의 {support_type} 지원"
            
            output = f"{condition} 지역화폐 ({len(results)}개):\n\n"
            for result in results:
                # 메타데이터 구조 확인 후 추출
                metadata = result['metadata']
                if 'metadata' in metadata:  # 중첩된 구조인 경우
                    actual_metadata = metadata['metadata']
                else:
                    actual_metadata = metadata
                
                output += f"{result['rank']}. {actual_metadata['이름']}\n"
                output += f"  - 지역: {actual_metadata['지역1']} {actual_metadata['지역2']}\n"
                output += f"  - 지원방식: {', '.join(actual_metadata['지원방식'])}\n"
                if actual_metadata['비지원방식']:
                    output += f"  - 비지원방식: {', '.join(actual_metadata['비지원방식'])}\n"
                output += f"  - 상세정보: {actual_metadata['링크']}\n\n"
            
            return output
            
        except Exception as e:
            return f"지역 검색 중 오류가 발생했습니다: {str(e)}"


class SupportTypeSearchTool(BaseTool):
    name: str = "support_type_search"
    description: str = "특정 지원 방식의 지역화폐를 검색합니다. 지원되는 방식과 지원되지 않는 방식을 모두 지정할 수 있습니다."
    args_schema: type = SupportTypeFilterInput
    vector_db_manager: object = None
    
    def __init__(self, vector_db_manager):
        super().__init__()
        self.vector_db_manager = vector_db_manager
    
    def _run(self, support_types: List[str], exclude_types: Optional[List[str]] = None) -> str:
        """지원 방식별 검색 실행"""
        try:
            results = []
            all_data = self.vector_db_manager.get_all_data()
            
            for data in all_data:
                # 메타데이터 구조 확인 후 추출
                if 'metadata' in data:
                    metadata = data['metadata']
                else:
                    metadata = data
                
                # 지원 방식 확인
                supported = metadata.get('지원방식', [])
                not_supported = metadata.get('비지원방식', [])
                
                # 요구사항 확인
                has_required = True
                for support_type in support_types:
                    if support_type not in supported:
                        has_required = False
                        break
                
                # 제외 조건 확인
                has_excluded = False
                if exclude_types:
                    for exclude_type in exclude_types:
                        if exclude_type in supported:
                            has_excluded = True
                            break
                
                if has_required and not has_excluded:
                    results.append({
                        'rank': len(results) + 1,
                        'metadata': data  # 전체 데이터 구조 보존
                    })
            
            if not results:
                condition = f"지원방식: {', '.join(support_types)}"
                if exclude_types:
                    condition += f", 제외: {', '.join(exclude_types)}"
                return f"조건 '{condition}'에 맞는 결과가 없습니다."
            
            condition = f"{', '.join(support_types)} 지원"
            if exclude_types:
                condition += f" (단, {', '.join(exclude_types)} 제외)"
            
            output = f"{condition} 지역화폐 ({len(results)}개):\n\n"
            for result in results:
                # 메타데이터 구조 확인 후 추출
                metadata = result['metadata']
                if 'metadata' in metadata:  # 중첩된 구조인 경우
                    actual_metadata = metadata['metadata']
                else:
                    actual_metadata = metadata
                
                output += f"{result['rank']}. {actual_metadata['이름']}\n"
                output += f"  - 지역: {actual_metadata['지역1']} {actual_metadata['지역2']}\n"
                output += f"  - 지원방식: {', '.join(actual_metadata['지원방식'])}\n"
                if actual_metadata['비지원방식']:
                    output += f"  - 비지원방식: {', '.join(actual_metadata['비지원방식'])}\n"
                output += f"  - 상세정보: {actual_metadata['링크']}\n\n"
            
            return output
            
        except Exception as e:
            return f"지원 방식 검색 중 오류가 발생했습니다: {str(e)}"


class AllDataTool(BaseTool):
    name: str = "get_all_data"
    description: str = "모든 지역화폐 정보를 조회합니다. 전체 목록을 확인할 때 사용됩니다."
    vector_db_manager: object = None
    
    def __init__(self, vector_db_manager):
        super().__init__()
        self.vector_db_manager = vector_db_manager
    
    def _run(self) -> str:
        """모든 데이터 조회 실행"""
        try:
            results = self.vector_db_manager.get_all_data()
            
            if not results:
                return "데이터가 없습니다."
            
            output = f"전체 지역화폐 목록 ({len(results)}개):\n\n"
            for i, result in enumerate(results, 1):
                # 메타데이터 구조 확인 후 추출
                metadata = result['metadata']
                if 'metadata' in metadata:  # 중첩된 구조인 경우
                    actual_metadata = metadata['metadata']
                else:
                    actual_metadata = metadata
                
                output += f"{i}. {actual_metadata['이름']}\n"
                output += f"  - 지역: {actual_metadata['지역1']} {actual_metadata['지역2']}\n"
                output += f"  - 지원방식: {', '.join(actual_metadata['지원방식'])}\n"
                if actual_metadata['비지원방식']:
                    output += f"  - 비지원방식: {', '.join(actual_metadata['비지원방식'])}\n"
                output += f"  - 상세정보: {actual_metadata['링크']}\n\n"
                
                # 출력이 너무 길어지는 것을 방지
                if i > 50:
                    output += f"... 및 {len(results) - 50}개 더\n"
                    break
            
            return output
            
        except Exception as e:
            return f"전체 데이터 조회 중 오류가 발생했습니다: {str(e)}"


class StatisticsTool(BaseTool):
    name: str = "get_statistics"
    description: str = "지역화폐 데이터의 통계 정보를 제공합니다. 지역별, 지원방식별 현황을 확인할 수 있습니다."
    vector_db_manager: object = None
    
    def __init__(self, vector_db_manager):
        super().__init__()
        self.vector_db_manager = vector_db_manager
    
    def _run(self) -> str:
        """통계 정보 조회 실행"""
        try:
            stats = self.vector_db_manager.get_statistics()
            
            if not stats:
                return "통계 데이터가 없습니다."
            
            output = f"지역화폐 통계 정보:\n\n"
            output += f"전체 지역화폐 수: {stats['total_count']}개\n\n"
            
            # 지역별 통계
            output += "지역별 현황:\n"
            for region, count in sorted(stats['region_stats'].items()):
                output += f"  - {region}: {count}개\n"
            
            output += "\n지원방식별 현황:\n"
            for support, count in sorted(stats['support_stats'].items()):
                output += f"  - {support}: {count}개\n"
            
            return output
            
        except Exception as e:
            return f"통계 조회 중 오류가 발생했습니다: {str(e)}" 