import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from performance_monitor import performance_monitor
import time

st.set_page_config(page_title="시스템 성능 대시보드", layout="wide")
st.title("📊 시스템 성능 모니터링 대시보드")

# 사이드바 설정
st.sidebar.title("설정")
time_range = st.sidebar.selectbox(
    "시간 범위",
    ["최근 1시간", "최근 24시간", "최근 7일", "전체"]
)

# 메트릭 데이터 로드
try:
    with open("performance_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
except FileNotFoundError:
    st.error("성능 메트릭 데이터를 찾을 수 없습니다.")
    st.stop()

# 데이터 전처리
def process_metrics(metrics, time_range):
    """메트릭 데이터 전처리"""
    now = datetime.now()
    
    if time_range == "최근 1시간":
        start_time = now - timedelta(hours=1)
    elif time_range == "최근 24시간":
        start_time = now - timedelta(days=1)
    elif time_range == "최근 7일":
        start_time = now - timedelta(days=7)
    else:
        start_time = datetime.min

    filtered_metrics = {
        "response_times": [],
        "memory_usage": [],
        "cpu_usage": [],
        "errors": []
    }

    for metric_type in metrics:
        for metric in metrics[metric_type]:
            timestamp = datetime.fromisoformat(metric["timestamp"])
            if timestamp >= start_time:
                filtered_metrics[metric_type].append(metric)

    return filtered_metrics

filtered_metrics = process_metrics(metrics, time_range)

# 대시보드 레이아웃
col1, col2 = st.columns(2)

with col1:
    st.subheader("응답 시간")
    if filtered_metrics["response_times"]:
        df_response = pd.DataFrame(filtered_metrics["response_times"])
        df_response["timestamp"] = pd.to_datetime(df_response["timestamp"])
        fig_response = px.line(
            df_response,
            x="timestamp",
            y="value",
            title="응답 시간 추이"
        )
        st.plotly_chart(fig_response, use_container_width=True)
    else:
        st.info("응답 시간 데이터가 없습니다.")

with col2:
    st.subheader("메모리 사용량")
    if filtered_metrics["memory_usage"]:
        df_memory = pd.DataFrame(filtered_metrics["memory_usage"])
        df_memory["timestamp"] = pd.to_datetime(df_memory["timestamp"])
        df_memory["value"] = df_memory["value"] / (1024 * 1024)  # MB로 변환
        fig_memory = px.line(
            df_memory,
            x="timestamp",
            y="value",
            title="메모리 사용량 (MB)"
        )
        st.plotly_chart(fig_memory, use_container_width=True)
    else:
        st.info("메모리 사용량 데이터가 없습니다.")

# 시스템 상태
st.subheader("시스템 상태")
system_metrics = performance_monitor.get_system_metrics()

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("CPU 사용률", f"{system_metrics['cpu_percent']}%")

with col4:
    st.metric("메모리 사용률", f"{system_metrics['memory_percent']}%")

with col5:
    st.metric("디스크 사용률", f"{system_metrics['disk_usage']}%")

# 에러 로그
st.subheader("에러 로그")
if filtered_metrics["errors"]:
    df_errors = pd.DataFrame(filtered_metrics["errors"])
    df_errors["timestamp"] = pd.to_datetime(df_errors["timestamp"])
    st.dataframe(
        df_errors[["timestamp", "value", "metadata"]],
        use_container_width=True
    )
else:
    st.info("에러 로그가 없습니다.")

# 성능 분석
st.subheader("성능 분석")
analysis = performance_monitor.analyze_performance()

col6, col7, col8, col9 = st.columns(4)

with col6:
    st.metric(
        "평균 응답 시간",
        f"{analysis['average_response_time']:.2f}초"
    )

with col7:
    st.metric(
        "최대 메모리 사용량",
        f"{analysis['max_memory_usage'] / (1024 * 1024):.2f}MB"
    )

with col8:
    st.metric(
        "에러율",
        f"{analysis['error_rate'] * 100:.2f}%"
    )

with col9:
    st.metric(
        "총 요청 수",
        f"{analysis['total_requests']}"
    )

# 자동 새로고침
if st.button("새로고침"):
    st.experimental_rerun()

# 자동 새로고침 설정
auto_refresh = st.sidebar.checkbox("자동 새로고침 (5초)")
if auto_refresh:
    time.sleep(5)
    st.experimental_rerun() 