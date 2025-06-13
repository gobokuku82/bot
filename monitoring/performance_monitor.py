import time
import logging
import psutil
import json
from datetime import datetime
from functools import wraps
from typing import Dict, List, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            "response_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "errors": []
        }

    def log_metric(self, metric_type: str, value: Any, metadata: Dict = None):
        """메트릭 로깅"""
        timestamp = datetime.now().isoformat()
        metric_data = {
            "timestamp": timestamp,
            "value": value,
            "metadata": metadata or {}
        }
        self.metrics[metric_type].append(metric_data)
        logger.info(f"{metric_type}: {value}")

    def get_system_metrics(self) -> Dict[str, float]:
        """시스템 메트릭 수집"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }

    def monitor_performance(self, func):
        """함수 성능 모니터링 데코레이터"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss
                memory_used = end_memory - start_memory

                # 성능 메트릭 로깅
                self.log_metric("response_times", execution_time, {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                })
                self.log_metric("memory_usage", memory_used, {
                    "function": func.__name__
                })

                return result

            except Exception as e:
                self.log_metric("errors", str(e), {
                    "function": func.__name__,
                    "error_type": type(e).__name__
                })
                raise

        return wrapper

    def save_metrics(self, filepath: str = "performance_metrics.json"):
        """메트릭 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

    def analyze_performance(self) -> Dict[str, Any]:
        """성능 분석"""
        analysis = {
            "average_response_time": 0,
            "max_memory_usage": 0,
            "error_rate": 0,
            "total_requests": 0
        }

        if self.metrics["response_times"]:
            analysis["average_response_time"] = sum(
                m["value"] for m in self.metrics["response_times"]
            ) / len(self.metrics["response_times"])

        if self.metrics["memory_usage"]:
            analysis["max_memory_usage"] = max(
                m["value"] for m in self.metrics["memory_usage"]
            )

        total_requests = len(self.metrics["response_times"])
        total_errors = len(self.metrics["errors"])
        analysis["total_requests"] = total_requests
        analysis["error_rate"] = total_errors / total_requests if total_requests > 0 else 0

        return analysis

# 싱글톤 인스턴스
performance_monitor = PerformanceMonitor()

# 사용 예시
@performance_monitor.monitor_performance
def example_function():
    """테스트용 함수"""
    time.sleep(1)  # 작업 시뮬레이션
    return "완료"

if __name__ == "__main__":
    # 테스트 실행
    for _ in range(5):
        example_function()
        time.sleep(0.5)

    # 메트릭 저장
    performance_monitor.save_metrics()

    # 성능 분석 결과 출력
    analysis = performance_monitor.analyze_performance()
    print("\n성능 분석 결과:")
    print(f"평균 응답 시간: {analysis['average_response_time']:.2f}초")
    print(f"최대 메모리 사용량: {analysis['max_memory_usage'] / 1024 / 1024:.2f}MB")
    print(f"에러율: {analysis['error_rate'] * 100:.2f}%")
    print(f"총 요청 수: {analysis['total_requests']}") 