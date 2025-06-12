from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ 결과 요약
def summarize_results(results: list) -> str:
    content = "\n".join([f"{r['지역']} - {r['이름']} ({r['지원방식']})" for r in results[:20]])
    prompt = f"""다음은 지역상품권 리스트입니다. 중요한 지역/지원방식 특성을 요약해줘. 100자 이내 요약:
{content}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 지역화폐 정보를 요약하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()
