import unittest
from agents.agent_executor import agent_executor
from tools.filter_tool import parse_conditions, filter_jsonl_by_condition
from tools.llm_tool import summarize_results
from tools.query_classifier import classify_query

class TestSystem(unittest.TestCase):
    def setUp(self):
        """테스트 환경 설정"""
        self.test_query = "경기도 모바일 지역화폐"
        self.test_data = [{
            "content": "경기도 수원시에서는 \"수원사랑상품권\"이 제공되며, 모바일 지원됩니다.",
            "metadata": {
                "지역1": "경기도",
                "지역2": "수원시",
                "이름": "수원사랑상품권",
                "지원방식": ["모바일"],
                "비지원방식": ["지류형", "카드형"],
                "링크": "http://example.com"
            }
        }]

    def test_query_classification(self):
        """쿼리 분류 테스트"""
        classification = classify_query(self.test_query)
        self.assertIn(classification["query_type"], 
                     ["internal_search", "external_search", "calculator"])
        self.assertGreaterEqual(classification["confidence"], 0.0)
        self.assertLessEqual(classification["confidence"], 1.0)

    def test_condition_parsing(self):
        """조건 파싱 테스트"""
        conditions = parse_conditions(self.test_query)
        self.assertEqual(conditions["region"], "경기도")
        self.assertEqual(conditions["support_type"], "모바일")

    def test_data_filtering(self):
        """데이터 필터링 테스트"""
        conditions = {
            "region": "경기도",
            "support_type": "모바일",
            "keywords": []
        }
        results = filter_jsonl_by_condition(self.test_data, conditions)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["지역1"], "경기도")
        self.assertIn("모바일", results[0]["metadata"]["지원방식"])

    def test_result_summarization(self):
        """결과 요약 테스트"""
        summary = summarize_results(self.test_data)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

    def test_agent_execution(self):
        """에이전트 실행 테스트"""
        response = agent_executor.invoke({
            "input": self.test_query,
            "chat_history": []
        })
        self.assertIn("output", response)
        self.assertIsInstance(response["output"], str)

if __name__ == '__main__':
    unittest.main() 