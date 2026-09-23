# benchmark.py

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 프로젝트 루트 등록
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# 파이프라인 불러오기
# ============================================================

from src.pipeline_v1 import run_pipeline_v1
from src.pipeline_v2 import run_pipeline_v2
from src.pipeline_v3 import run_pipeline_v3


# ============================================================
# 경로 설정
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config.json"

RESULT_DIR.mkdir(exist_ok=True)


# ============================================================
# 테스트 영상
# ============================================================

videos = {
    "640x480": DATA_DIR / "sample_640x480.mp4",
    "1280x720": DATA_DIR / "sample_1280x720.mp4",
    "1920x1080": DATA_DIR / "sample_1920x1080.mp4"
}


# ============================================================
# 테스트할 파이프라인
# ============================================================

pipelines = {
    "V1": run_pipeline_v1,
    "V2": run_pipeline_v2,
    "V3": run_pipeline_v3
}


# ============================================================
# 벤치마크 결과 저장
# ============================================================

results = []


# ============================================================
# Benchmark
# ============================================================

for resolution, video_path in videos.items():

    for version, pipeline_func in pipelines.items():

        for run in range(1, 4):

            print()
            print("=" * 60)
            print(
                f"{resolution} | "
                f"{version} | "
                f"Run {run}/3"
            )
            print("=" * 60)

            result = pipeline_func(
                str(video_path),
                config_path=str(CONFIG_PATH),
                show_window=False
            )

            results.append({
                "resolution": resolution,
                "version": version,
                "run": run,
                "fps": result["fps"]
            })


# ============================================================
# DataFrame 생성
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# CSV 저장
# ============================================================

csv_path = RESULT_DIR / "fps_comparison_table.csv"

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)

print()
print("CSV 저장 완료:")
print(csv_path)


# ============================================================
# 평균 FPS 계산
# ============================================================

avg_df = (
    df.groupby(
        ["resolution", "version"]
    )["fps"]
    .mean()
    .reset_index()
)

print()
print("========== 평균 FPS ==========")
print(avg_df)


# ============================================================
# 그래프 생성
# ============================================================

pivot_df = avg_df.pivot(
    index="resolution",
    columns="version",
    values="fps"
)

pivot_df.plot(
    kind="bar"
)

plt.title("FPS Benchmark")
plt.xlabel("Resolution")
plt.ylabel("Average FPS")
plt.xticks(rotation=0)
plt.tight_layout()


# ============================================================
# 그래프 저장
# ============================================================

graph_path = RESULT_DIR / "fps_benchmark_graph.png"

plt.savefig(
    graph_path,
    dpi=150
)

plt.close()

print()
print("그래프 저장 완료:")
print(graph_path)