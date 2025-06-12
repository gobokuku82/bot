import json
import re
from typing import List, Dict

# ✅ 규칙 기반 질문 파서
from typing import Dict, List

def parse_conditions(query: str) -> Dict[str, List[str]]:
    support_types = ["모바일", "카드형", "지류형"]

    provinces = {
        "경상도": ["경북", "경남", "대구", "부산", "울산"],
        "충청도": ["충북", "충남", "대전", "세종"],
        "전라도": ["전북", "전남", "광주"],
        "경기도": ["경기", "인천"],
        "강원도": ["강원"]
    }

    # 전체 시도 (provinces의 값 + 서울, 제주 등 누락된 광역시 보완)
    all_regions = list(set(sum(provinces.values(), []))) + ["서울", "제주"]

    cond = {"지원방식": [], "지역1": []}

    # ✅ 지원방식 추출
    for stype in support_types:
        if stype in query:
            cond["지원방식"].append(stype)

    # ✅ 도 단위 명칭이 포함된 경우
    for pname, subs in provinces.items():
        if pname in query:
            cond["지역1"].extend(subs)

    # ✅ 시도 직접 언급된 경우
    for r in all_regions:
        if r in query:
            cond["지역1"].append(r)

    # ✅ 중복 제거
    cond["지역1"] = list(set(cond["지역1"]))

    return cond

# ✅ JSONL 불러오기
def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ✅ 조건 기반 필터링
def filter_jsonl_by_condition(data: List[Dict], cond: Dict[str, List[str]]) -> List[Dict]:
    result = []
    for row in data:
        meta = row["metadata"]
        if all(t in meta.get("지원방식", []) for t in cond.get("지원방식", [])) and \
           meta.get("지역1") in cond.get("지역1", []):
            result.append({
                "이름": meta["이름"],
                "지역": f"{meta['지역1']} {meta['지역2']}",
                "지원방식": ", ".join(meta["지원방식"]),
                "링크": meta["링크"]
            })
    return result
