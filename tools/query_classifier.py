from typing import Literal
from langchain_core.pydantic_v1 import BaseModel

class QueryType(str):
    INTERNAL = "internal_search"
    EXTERNAL = "external_search"
    CALC = "calculator"
    ETC = "etc"

class QueryClassification(BaseModel):
    query: str
    query_type: Literal["internal_search", "external_search", "calculator", "etc"]

def classify_query(query: str) -> QueryClassification:
    query = query.strip().lower()

    # 내부 DB 질의 판단 (ex. 상품권, 지역, 모바일, 카드형 등)
    if any(keyword in query for keyword in ["상품권", "화폐", "카드형", "모바일", "지역"]):
        return QueryClassification(query=query, query_type=QueryType.INTERNAL)

    # 계산 요청 (향후 대응)
    if any(keyword in query for keyword in ["계산", "얼마", "합", "나누기", "곱하기"]):
        return QueryClassification(query=query, query_type=QueryType.CALC)

    # 외부 검색 (네이버 등)
    if any(keyword in query for keyword in ["맛집", "관광", "추천", "검색"]):
        return QueryClassification(query=query, query_type=QueryType.EXTERNAL)

    # 그 외 기타
    return QueryClassification(query=query, query_type=QueryType.ETC)
