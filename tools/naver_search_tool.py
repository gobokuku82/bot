# tools/naver_search_tool.py
import requests
from langchain_core.tools import tool
import os

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

@tool
def naver_local_search(query: str) -> str:
    """지역 기반 정보(맛집, 관광 등)를 검색합니다. 예: '대전 맛집', '충북 관광지'"""
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": query,
        "display": 5,
        "start": 1,
        "sort": "random"
    }

    response = requests.get(url, headers=headers, params=params)
    items = response.json().get("items", [])

    if not items:
        return "검색 결과가 없습니다."

    result = ""
    for i, item in enumerate(items, 1):
        result += f"{i}. {item['title'].replace('<b>', '').replace('</b>', '')}\n"
        result += f"   📍 {item['address']}\n"
        result += f"   🔗 {item['link']}\n\n"

    return result
