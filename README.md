# 지역화폐 검색 시스템 🏦💰

한국의 지역화폐 정보를 검색할 수 있는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 📋 주요 기능

- **벡터 검색**: 자연어로 지역화폐 정보 검색
- **지역별 검색**: 특정 지역의 지역화폐 조회
- **지원방식 필터링**: 모바일, 카드형, 지류형 등 지원방식별 검색
- **복합 조건 검색**: 여러 조건을 조합한 검색
- **실시간 채팅**: 대화형 인터페이스로 편리한 검색
- **통계 대시보드**: 지역화폐 현황 통계 시각화

## 🏗️ 시스템 구조

```
.
├── app.py                          # Streamlit 메인 애플리케이션
├── requirements.txt                # 필요 패키지 목록
├── .env                           # 환경변수 (생성 필요)
├── data/
│   └── faiss_coupon_db/
│       ├── index.faiss            # Faiss 벡터 인덱스
│       ├── index.pkl              # 메타데이터
│       └── 지역사랑상품권_긍정_부정전처리_cleaned.jsonl
├── utils/
│   └── vector_db.py               # 벡터 DB 관리 유틸리티
├── tools/
│   └── search_tools.py            # 검색 도구들
└── agents/
    └── local_currency_agent.py    # LangGraph 에이전트
```

## 🔧 기술 스택

- **임베딩 모델**: nlpai-lab/KURE-v1 (한국어 특화)
- **벡터 DB**: Faiss (Facebook AI Similarity Search)
- **LLM**: OpenAI GPT-4o-mini
- **에이전트 프레임워크**: LangGraph
- **UI**: Streamlit
- **기타**: LangChain, Sentence Transformers

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt

# 환경변수 파일 생성
# .env 파일을 생성하고 다음 내용을 추가:
```

**.env 파일 내용:**
```env
OPENAI_API_KEY=sk-your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
MODEL_NAME=nlpai-lab/KURE-v1
VECTOR_DB_PATH=data/faiss_coupon_db
DATA_PATH=data/faiss_coupon_db/지역사랑상품권_긍정_부정전처리_cleaned.jsonl
```

### 2. OpenAI API 키 발급

1. [OpenAI Platform](https://platform.openai.com/)에 가입
2. API Keys 페이지에서 새 API 키 생성
3. `.env` 파일에 API 키 입력

### 3. 애플리케이션 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`으로 접속

## 📖 사용법

### 기본 검색
- "경기도 지역화폐 알려줘"
- "모바일 지원되는 상품권 찾아줘"
- "부산 동백전 정보"

### 조건부 검색
- "모바일만 지원되는 지역화폐는?"
- "충청도의 카드형 지역화폐 알려줘"
- "지류형은 지원하지 않는 곳은?"

### 복합 검색
- "모바일 되는 것 중에서 경상도에서 쓸 수 있는 것은?"
- "카드형과 모바일 둘 다 지원하는 곳은?"

### 전체 조회
- "모든 지역화폐 알려줘"
- "전체 지역화폐 통계 보여줘"

## 🎯 검색 도구 설명

1. **VectorSearchTool**: 자연어 벡터 검색
2. **RegionSearchTool**: 지역별 검색
3. **SupportTypeSearchTool**: 지원방식별 검색
4. **MetadataFilterTool**: 메타데이터 조건 검색
5. **AllDataTool**: 전체 데이터 조회
6. **StatisticsTool**: 통계 정보 조회

## 📊 데이터 형식

```json
{
  "content": "제주 제주시에서는 \"[제주특별자치도]탐나는전\"이 제공되며, 지류형, 모바일, 카드형은 지원되며.",
  "metadata": {
    "지역1": "제주",
    "지역2": "제주시", 
    "이름": "[제주특별자치도]탐나는전",
    "지원방식": ["지류형", "모바일", "카드형"],
    "비지원방식": [],
    "링크": "http://tamna.jeju.go.kr/mainView.do"
  }
}
```

## 🔄 벡터 DB 재생성

기존 벡터 DB를 삭제하고 새로 생성하려면:

```python
import os
from utils.vector_db import VectorDBManager

# 기존 인덱스 파일 삭제
os.remove("data/faiss_coupon_db/index.faiss")
os.remove("data/faiss_coupon_db/index.pkl")

# 새로 생성
vector_db = VectorDBManager()
```

## 🧪 테스트

```python
from utils.vector_db import VectorDBManager
from agents.local_currency_agent import LocalCurrencyAgent

# 벡터 DB 테스트
vector_db = VectorDBManager()
results = vector_db.search("경기도 지역화폐", k=5)
print(results)

# 에이전트 테스트
agent = LocalCurrencyAgent(vector_db, "your_openai_key")
response = agent.process_query("모든 지역화폐 알려줘")
print(response)
```

## 📈 성능 최적화

- **캐싱**: Streamlit의 `@st.cache_data` 활용
- **배치 처리**: 대량 검색시 배치 단위로 처리
- **메모리 관리**: 대용량 모델 로딩시 메모리 최적화

## 🔍 문제 해결

### 일반적인 문제

1. **모델 로딩 오류**
   - 인터넷 연결 확인
   - Hugging Face 토큰 설정

2. **OpenAI API 오류**
   - API 키 유효성 확인
   - 요금제 한도 확인

3. **메모리 부족**
   - 작은 배치 크기 사용
   - GPU 메모리 확인

## 📝 라이선스

MIT License

## 🤝 기여

1. Fork 프로젝트
2. Feature 브랜치 생성
3. 변경사항 커밋
4. 브랜치에 Push
5. Pull Request 생성

## 📞 지원

문제가 있거나 기능 요청이 있으시면 Issue를 생성해주세요. 