import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import warnings

# PyTorch 관련 경고 및 설정
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Faiss 임포트 시도 (실패시 None으로 설정)
try:
    import faiss
    FAISS_AVAILABLE = True
    print("faiss가 성공적으로 로드되었습니다.")
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    print("Warning: faiss-cpu가 설치되지 않았습니다. 기본 벡터 검색을 사용합니다.")

class VectorDBManager:
    def __init__(self, vector_db_path: str = "data/faiss_coupon_db", model_name: str = "nlpai-lab/KURE-v1"):
        """
        벡터 DB 관리자 초기화
        
        Args:
            vector_db_path: 벡터 DB 저장 경로 (기존 벡터DB 폴더 경로)
            model_name: 사용할 임베딩 모델명
        """
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        self.index_path = os.path.join(vector_db_path, "index.faiss")
        self.metadata_path = os.path.join(vector_db_path, "index.pkl")
        
        # 모델 로드 (PyTorch 호환성 문제 해결)
        try:
            # 기본 CPU 디바이스로 모델 로드
            self.model = SentenceTransformer(model_name, device='cpu')
        except Exception as e:
            print(f"모델 로드 중 오류 발생, 재시도 중: {e}")
            # 대안적 방법으로 모델 로드
            import torch as torch_lib  # os 충돌을 피하기 위해 별명 사용
            # PyTorch 백엔드 설정
            torch_lib.set_default_dtype(torch_lib.float32)
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            self.model = SentenceTransformer(model_name, device='cpu')
        
        # 벡터 DB 및 메타데이터 로드
        self.index = None
        self.metadata = []
        self.load_vector_db()
        
    def load_vector_db(self):
        """기존 벡터 DB가 있으면 로드"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("기존 벡터 DB를 로드합니다...")
            if FAISS_AVAILABLE:
                self.index = faiss.read_index(self.index_path)
            else:
                # Faiss가 없으면 numpy 배열로 로드
                self.index = np.load(self.index_path.replace('.faiss', '.npy'))
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"벡터 DB 로드 완료: {len(self.metadata)}개 문서")
        else:
            print("벡터 DB가 없습니다. add_documents 메서드를 사용하여 생성하세요.")
            self.index = None
            self.metadata = []
    
    def create_vector_db(self):
        """JSONL 파일로부터 벡터 DB 생성"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
        
        # JSONL 파일 읽기
        documents = []
        metadata = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                documents.append(data['content'])
                metadata.append(data['metadata'])
        
        print(f"문서 수: {len(documents)}")
        
        # 임베딩 생성
        print("임베딩 생성 중...")
        embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Faiss 인덱스 생성
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 내적(코사인 유사도) 사용
        
        # 임베딩 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(embeddings)
        
        # 인덱스에 추가
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata
        
        # 저장
        os.makedirs(self.vector_db_path, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"벡터 DB 생성 완료: {len(metadata)}개 문서")
    
    def search(self, query: str, k: int = 10, filter_func: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        벡터 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter_func: 메타데이터 필터링 함수 (Optional)
        
        Returns:
            검색 결과 리스트
        """
        if self.index is None:
            return []
        
        # 쿼리 임베딩
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 검색 수행
        if filter_func is None:
            # 필터링 없는 검색
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # 유효한 인덱스인지 확인
                    results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'metadata': self.metadata[idx]
                    })
            return results
        else:
            # 필터링을 위해 더 많은 결과 검색
            search_k = min(len(self.metadata), k * 5)  # 필터링을 고려해 더 많이 검색
            scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
            
            filtered_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and len(filtered_results) < k:
                    metadata = self.metadata[idx]
                    if filter_func(metadata):
                        filtered_results.append({
                            'rank': len(filtered_results) + 1,
                            'score': float(score),
                            'metadata': metadata
                        })
            
            return filtered_results
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """모든 데이터 반환 (기존 벡터DB 구조에 맞게 조정)"""
        # 기존 벡터DB의 메타데이터가 이미 content + metadata 구조인 경우
        if self.metadata and isinstance(self.metadata[0], dict) and 'content' in self.metadata[0] and 'metadata' in self.metadata[0]:
            return self.metadata  # 그대로 반환
        else:
            # 단순 메타데이터만 있는 경우
            return [{'metadata': meta} for meta in self.metadata]
    
    def filter_by_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        메타데이터로 필터링
        
        Args:
            filters: 필터링 조건 딕셔너리
                예: {'지역1': '경기', '지원방식': ['카드형']}
        
        Returns:
            필터링된 결과 리스트
        """
        results = []
        for i, item in enumerate(self.metadata):
            # 메타데이터 구조 확인
            if 'metadata' in item:  # 중첩된 구조인 경우
                metadata = item['metadata']
                full_item = item
            else:
                metadata = item
                full_item = item
                
            match = True
            for key, value in filters.items():
                if key not in metadata:
                    match = False
                    break
                
                if isinstance(value, list):
                    # 리스트의 경우 교집합 확인
                    if isinstance(metadata[key], list):
                        if not set(value) & set(metadata[key]):  # 교집합이 없으면
                            match = False
                            break
                    else:
                        if metadata[key] not in value:
                            match = False
                            break
                elif isinstance(value, str):
                    # 문자열의 경우 포함 여부 확인
                    if isinstance(metadata[key], list):
                        if value not in metadata[key]:
                            match = False
                            break
                    else:
                        if value not in str(metadata[key]):
                            match = False
                            break
                else:
                    # 정확한 매치
                    if metadata[key] != value:
                        match = False
                        break
            
            if match:
                results.append({
                    'rank': len(results) + 1,
                    'score': 1.0,  # 메타데이터 필터링은 완전 매치
                    'metadata': full_item
                })
        
        return results
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """문서와 메타데이터를 벡터DB에 추가"""
        print(f"문서 수: {len(documents)}")
        
        # 임베딩 생성
        print("임베딩 생성 중...")
        embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Faiss 인덱스 생성
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 내적(코사인 유사도) 사용
        
        # 임베딩 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(embeddings)
        
        # 인덱스에 추가
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadatas
        
        # 저장
        os.makedirs(self.vector_db_path, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"벡터 DB 생성 완료: {len(metadatas)}개 문서")

    def get_statistics(self) -> Dict[str, Any]:
        """데이터 통계 정보 반환"""
        if not self.metadata:
            return {}
        
        # 지역별 통계
        region_stats = {}
        support_stats = {}
        
        for item in self.metadata:
            # 메타데이터 구조 확인
            if 'metadata' in item:  # 중첩된 구조인 경우
                meta = item['metadata']
            else:
                meta = item
            
            # 지역1 통계
            region1 = meta.get('지역1', 'Unknown')
            if region1 not in region_stats:
                region_stats[region1] = 0
            region_stats[region1] += 1
            
            # 지원방식 통계
            for support in meta.get('지원방식', []):
                if support not in support_stats:
                    support_stats[support] = 0
                support_stats[support] += 1
        
        return {
            'total_count': len(self.metadata),
            'region_stats': region_stats,
            'support_stats': support_stats
        } 