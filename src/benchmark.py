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
# 4. 이번 벤치마크 ID
# ============================================================

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
# 6. 테스트 파이프라인
# ============================================================

pipelines = {
    "V1": run_pipeline_v1,
    "V2": run_pipeline_v2,
    "V3": run_pipeline_v3
}


# ============================================================
# 7. 이번 실행 결과 저장
# ============================================================

results = []


# ============================================================
# 8. Benchmark 실행
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
                        f"{version} 파이프라인 반환값이 None입니다."
                    )

                if "fps" not in result:
                    raise KeyError(
                        f"{version} 결과에 'fps'가 없습니다."
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
            # Ctrl + C
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

                raise


            # ====================================================
            # 일반 오류
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

                continue


# ============================================================
# 9. 이번 실행 결과 DataFrame
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# 10. 개별 Run CSV 누적 저장
# ============================================================

csv_path = RESULT_DIR / "fps_comparison_table.csv"

if csv_path.exists():

    try:

        old_df = pd.read_csv(
            csv_path,
            encoding="utf-8-sig"
        )

        all_df = pd.concat(
            [old_df, df],
            ignore_index=True
        )

    except Exception as e:

        print()
        print("[WARNING] 기존 CSV 읽기 실패")
        print(f"원인: {type(e).__name__}: {e}")

        all_df = df.copy()

else:

    all_df = df.copy()


all_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)

print()
print("개별 Run CSV 누적 저장 완료:")
print(csv_path)


# ============================================================
# 11. 성공한 데이터만 사용
# ============================================================

success_df = df[
    df["status"] == "SUCCESS"
].copy()


# ============================================================
# 12. 평균 / 표준편차 계산
# ============================================================

if not success_df.empty:

    summary_df = (
        success_df
        .groupby(
            ["resolution", "version"]
        )["fps"]
        .agg(
            mean_fps="mean",
            std_fps="std",
            run_count="count"
        )
        .reset_index()
    )


    # ========================================================
    # 13. V1 대비 성능 향상률 계산
    # ========================================================

    # 각 해상도의 V1 평균 FPS 가져오기
    v1_fps = (
        summary_df[
            summary_df["version"] == "V1"
        ]
        .set_index("resolution")["mean_fps"]
    )


    def calculate_improvement(row):

        resolution = row["resolution"]

        # 해당 해상도의 V1 결과가 없으면 계산 불가
        if resolution not in v1_fps.index:
            return None

        baseline = v1_fps.loc[resolution]

        if baseline == 0:
            return None

        return (
            (row["mean_fps"] - baseline)
            / baseline
            * 100
        )


    summary_df["improvement_vs_v1_pct"] = (
        summary_df.apply(
            calculate_improvement,
            axis=1
        )
    )


    # 이번 벤치마크 ID와 시간도 기록
    summary_df.insert(
        0,
        "benchmark_id",
        BENCHMARK_ID
    )

    summary_df.insert(
        1,
        "timestamp",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    # ========================================================
    # 14. 이번 실행 결과 출력
    # ========================================================

    print()
    print("=" * 85)
    print("이번 실행 FPS 요약")
    print("=" * 85)

    print(
        summary_df[
            [
                "resolution",
                "version",
                "mean_fps",
                "std_fps",
                "improvement_vs_v1_pct"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("=" * 85)


    # ========================================================
    # 15. 그래프용 요약 CSV 누적
    # ========================================================

    summary_csv_path = (
        RESULT_DIR
        / "fps_benchmark_summary.csv"
    )

    # 직전 데이터 확인을 위해
    # 새 데이터를 합치기 전에 기존 파일을 읽음
    previous_summary = None

    if summary_csv_path.exists():

        try:

            old_summary_df = pd.read_csv(
                summary_csv_path,
                encoding="utf-8-sig"
            )

            # -----------------------------------------------
            # 직전 benchmark_id 찾기
            # -----------------------------------------------

            if not old_summary_df.empty:

                previous_id = (
                    old_summary_df["benchmark_id"]
                    .astype(str)
                    .iloc[-1]
                )

                previous_summary = (
                    old_summary_df[
                        old_summary_df[
                            "benchmark_id"
                        ].astype(str) == previous_id
                    ].copy()
                )

            # 기존 + 현재 요약 데이터
            all_summary_df = pd.concat(
                [
                    old_summary_df,
                    summary_df
                ],
                ignore_index=True
            )

        except Exception as e:

            print()
            print(
                "[WARNING] 기존 Summary CSV "
                "읽기 실패"
            )

            print(
                f"원인: {type(e).__name__}: {e}"
            )

            all_summary_df = summary_df.copy()

    else:

        all_summary_df = summary_df.copy()


    # 누적 저장
    all_summary_df.to_csv(
        summary_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("벤치마크 요약 CSV 누적 저장 완료:")
    print(summary_csv_path)


    # ========================================================
    # 16. 현재 vs 직전 벤치마크 그래프 생성
    # ========================================================

    current_graph_df = (
        summary_df.pivot(
            index="resolution",
            columns="version",
            values="mean_fps"
        )
    )


    # --------------------------------------------------------
    # 이전 실행 결과가 있는 경우
    # --------------------------------------------------------

    if previous_summary is not None:

        previous_graph_df = (
            previous_summary.pivot(
                index="resolution",
                columns="version",
                values="mean_fps"
            )
        )

        # 그래프에서 현재/직전을 구분하기 위해 이름 변경
        previous_graph_df.columns = [
            f"{col}_Previous"
            for col in previous_graph_df.columns
        ]

        current_graph_df.columns = [
            f"{col}_Current"
            for col in current_graph_df.columns
        ]

        graph_df = previous_graph_df.join(
            current_graph_df,
            how="outer"
        )


    # --------------------------------------------------------
    # 첫 번째 벤치마크라 이전 기록이 없는 경우
    # --------------------------------------------------------

    else:

        current_graph_df.columns = [
            f"{col}_Current"
            for col in current_graph_df.columns
        ]

        graph_df = current_graph_df


    # ========================================================
    # 17. 그래프 그리기
    # ========================================================

    graph_df.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(
        "FPS Benchmark - Current vs Previous"
    )

    plt.xlabel("Resolution")
    plt.ylabel("Average FPS")

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()


    # ========================================================
    # 18. 그래프 저장
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
    print("비교 그래프 저장 완료:")
    print(graph_path)


else:

    print()
    print(
        "[WARNING] 성공한 테스트가 없어 "
        "평균/표준편차/그래프를 생성하지 않았습니다."
    )


# ============================================================
# 19. 최종 벤치마크 요약
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