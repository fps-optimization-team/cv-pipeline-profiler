# benchmark.py

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 프로젝트 루트 등록
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# 2. 파이프라인 불러오기
# ============================================================

from src.pipeline_v1 import run_pipeline_v1
from src.pipeline_v2 import run_pipeline_v2
from src.pipeline_v3 import run_pipeline_v3


# ============================================================
# 3. 경로 설정
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config.json"

RESULT_DIR.mkdir(exist_ok=True)


# ============================================================
# 4. 이번 벤치마크 실행 ID 생성
# ============================================================

# benchmark.py를 한 번 실행할 때 하나의 ID를 부여
# 예: 20260928_113025
BENCHMARK_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 5. 테스트 영상
# ============================================================

videos = {
    "640x480": DATA_DIR / "sample_640x480.mp4",
    "1280x720": DATA_DIR / "sample_1280x720.mp4",
    "1920x1080": DATA_DIR / "sample_1920x1080.mp4"
}


# ============================================================
# 6. 테스트할 파이프라인
# ============================================================

pipelines = {
    "V1": run_pipeline_v1,
    "V2": run_pipeline_v2,
    "V3": run_pipeline_v3
}


# ============================================================
# 7. 이번 벤치마크 결과 저장용 리스트
# ============================================================

results = []


# ============================================================
# 8. Benchmark
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

            try:
                # ------------------------------------------------
                # 파이프라인 실행
                # ------------------------------------------------

                result = pipeline_func(
                    str(video_path),
                    config_path=str(CONFIG_PATH),
                    show_window=False
                )

                # ------------------------------------------------
                # 반환값 검증
                # ------------------------------------------------

                if result is None:
                    raise ValueError(
                        f"{version} 파이프라인의 반환값이 None입니다."
                    )

                if "fps" not in result:
                    raise KeyError(
                        f"{version} 파이프라인 결과에 "
                        "'fps' 값이 없습니다."
                    )

                fps = result["fps"]

                # ------------------------------------------------
                # 성공 결과 저장
                # ------------------------------------------------

                results.append({
                    "benchmark_id": BENCHMARK_ID,
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "resolution": resolution,
                    "version": version,
                    "run": run,
                    "fps": fps,
                    "status": "SUCCESS",
                    "error": ""
                })

                print(
                    f"[SUCCESS] "
                    f"{resolution} | "
                    f"{version} | "
                    f"Run {run}/3 | "
                    f"FPS: {fps:.2f}"
                )

            # ====================================================
            # 사용자가 Ctrl + C로 중단한 경우
            # ====================================================

            except KeyboardInterrupt:

                print()
                print("[중단] 사용자가 벤치마크를 중단했습니다.")

                results.append({
                    "benchmark_id": BENCHMARK_ID,
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "resolution": resolution,
                    "version": version,
                    "run": run,
                    "fps": None,
                    "status": "INTERRUPTED",
                    "error": "KeyboardInterrupt"
                })

                # Ctrl + C는 전체 벤치마크 중단
                raise


            # ====================================================
            # 그 외 오류 발생
            # ====================================================

            except Exception as e:

                error_message = (
                    f"{type(e).__name__}: {e}"
                )

                print(
                    f"[ERROR] "
                    f"{resolution} | "
                    f"{version} | "
                    f"Run {run}/3"
                )

                print(
                    f"원인: {error_message}"
                )

                # 실패한 테스트도 결과에 저장
                results.append({
                    "benchmark_id": BENCHMARK_ID,
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "resolution": resolution,
                    "version": version,
                    "run": run,
                    "fps": None,
                    "status": "ERROR",
                    "error": error_message
                })

                # 여기서 프로그램을 종료하지 않고
                # 다음 테스트로 계속 진행
                continue


# ============================================================
# 9. 이번 실행 결과 DataFrame 생성
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# 10. CSV 누적 저장
# ============================================================

csv_path = RESULT_DIR / "fps_comparison_table.csv"


# 기존 CSV가 존재하면 기존 기록을 불러옴
if csv_path.exists():

    try:
        old_df = pd.read_csv(
            csv_path,
            encoding="utf-8-sig"
        )

        # 기존 기록 + 이번 실행 기록
        all_df = pd.concat(
            [old_df, df],
            ignore_index=True
        )

    except Exception as e:

        print()
        print(
            "[WARNING] 기존 CSV를 읽는 중 "
            "문제가 발생했습니다."
        )

        print(
            f"원인: {type(e).__name__}: {e}"
        )

        print(
            "이번 벤치마크 결과만 저장합니다."
        )

        all_df = df.copy()

else:

    # 처음 실행하는 경우
    all_df = df.copy()


# 누적된 전체 기록 저장
all_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("CSV 누적 저장 완료:")
print(csv_path)


# ============================================================
# 11. 이번 실행에서 성공한 데이터만 추출
# ============================================================

success_df = df[
    df["status"] == "SUCCESS"
].copy()


# ============================================================
# 12. 평균 FPS 계산
# ============================================================

if not success_df.empty:

    avg_df = (
        success_df
        .groupby(
            ["resolution", "version"]
        )["fps"]
        .mean()
        .reset_index()
    )

    print()
    print("========== 이번 실행 평균 FPS ==========")
    print(avg_df)


    # ========================================================
    # 13. 그래프 생성
    # ========================================================

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


    # ========================================================
    # 14. 그래프 저장
    # ========================================================

    graph_path = (
        RESULT_DIR
        / "fps_benchmark_graph.png"
    )

    plt.savefig(
        graph_path,
        dpi=150
    )

    plt.close()

    print()
    print("그래프 저장 완료:")
    print(graph_path)


else:

    print()
    print(
        "[WARNING] 이번 실행에서 성공한 테스트가 없어 "
        "FPS 그래프를 생성하지 않았습니다."
    )


# ============================================================
# 15. 최종 벤치마크 요약
# ============================================================

success_count = len(
    df[df["status"] == "SUCCESS"]
)

error_count = len(
    df[df["status"] == "ERROR"]
)

interrupted_count = len(
    df[df["status"] == "INTERRUPTED"]
)


print()
print("=" * 60)
print("벤치마크 완료")
print("=" * 60)

print(
    f"Benchmark ID : {BENCHMARK_ID}"
)

print(
    f"전체 테스트  : {len(df)}"
)

print(
    f"성공 테스트  : {success_count}"
)

print(
    f"실패 테스트  : {error_count}"
)

print(
    f"중단 테스트  : {interrupted_count}"
)

print("=" * 60)