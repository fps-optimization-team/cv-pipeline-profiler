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
# 4. 테스트 영상
# ============================================================

VIDEOS = {
    "640x480": DATA_DIR / "samples"/ "tennis_sample_640x480.mp4",
    "1280x720": DATA_DIR / "samples"/ "tennis_sample_1280x720.mp4",
    "1920x1080": DATA_DIR / "samples"/ "tennis_sample_1920x1080.mp4"
}


# ============================================================
# 5. 테스트 파이프라인
# ============================================================

PIPELINES = {
    "V1": run_pipeline_v1,
    "V2": run_pipeline_v2,
    "V3": run_pipeline_v3
}


# ============================================================
# 함수 1. Benchmark ID 생성
# ============================================================

def create_benchmark_id():
    """
    현재 시간을 이용해서
    이번 벤치마크의 고유 ID를 생성합니다.
    """

    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 함수 2. 개별 파이프라인 1회 실행
# ============================================================

def run_single_test(
    resolution,
    version,
    run,
    video_path,
    pipeline_func,
    benchmark_id
):
    """
    하나의 파이프라인을 한 번 실행하고
    결과를 딕셔너리 형태로 반환합니다.
    """

    print()
    print("=" * 60)

    print(
        f"{resolution} | "
        f"{version} | "
        f"Run {run}/3"
    )

    print("=" * 60)

    try:

        # ----------------------------------------------------
        # 파이프라인 실행
        # ----------------------------------------------------

        result = pipeline_func(
            str(video_path),
            config_path=str(CONFIG_PATH),
            show_window=False
        )


        # ----------------------------------------------------
        # 반환값 검증
        # ----------------------------------------------------

        if result is None:
            raise ValueError(
                f"{version} 파이프라인 반환값이 None입니다."
            )

        if "fps" not in result:
            raise KeyError(
                f"{version} 결과에 'fps'가 없습니다."
            )


        fps = result["fps"]


        # ----------------------------------------------------
        # 성공 결과
        # ----------------------------------------------------

        test_result = {
            "benchmark_id": benchmark_id,
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "resolution": resolution,
            "version": version,
            "run": run,
            "fps": fps,
            "status": "SUCCESS",
            "error": ""
        }


        print(
            f"[SUCCESS] "
            f"{resolution} | "
            f"{version} | "
            f"Run {run}/3 | "
            f"FPS: {fps:.2f}"
        )

        return test_result


    # ========================================================
    # Ctrl + C
    # ========================================================

    except KeyboardInterrupt:

        print()
        print("[중단] 사용자가 벤치마크를 중단했습니다.")

        raise


    # ========================================================
    # 일반 오류
    # ========================================================

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


        return {
            "benchmark_id": benchmark_id,
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "resolution": resolution,
            "version": version,
            "run": run,
            "fps": None,
            "status": "ERROR",
            "error": error_message
        }


# ============================================================
# 함수 3. 전체 벤치마크 실행
# ============================================================

def run_benchmark(benchmark_id):
    """
    모든 해상도 × 모든 파이프라인 × 3회
    벤치마크를 실행합니다.
    """

    results = []

    for resolution, video_path in VIDEOS.items():

        for version, pipeline_func in PIPELINES.items():

            for run in range(1, 4):

                result = run_single_test(
                    resolution=resolution,
                    version=version,
                    run=run,
                    video_path=video_path,
                    pipeline_func=pipeline_func,
                    benchmark_id=benchmark_id
                )

                results.append(result)


    return pd.DataFrame(results)


# ============================================================
# 함수 4. 개별 Run 결과 CSV 누적 저장
# ============================================================

def save_run_results(df):

    csv_path = (
        RESULT_DIR
        / "fps_comparison_table.csv"
    )


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

            print(
                f"원인: "
                f"{type(e).__name__}: {e}"
            )

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
# 함수 5. V1 대비 성능 향상률 계산
# ============================================================

def calculate_improvement(row, v1_fps):
    """
    각 해상도에서
    V1을 기준으로 성능 향상률을 계산합니다.
    """

    resolution = row["resolution"]


    # 해당 해상도의 V1이 없으면 계산 불가
    if resolution not in v1_fps.index:
        return None


    baseline = v1_fps.loc[resolution]


    if baseline == 0:
        return None


    improvement = (
        (row["mean_fps"] - baseline)
        / baseline
        * 100
    )


    return improvement


# ============================================================
# 함수 6. 평균 / 표준편차 / 향상률 계산
# ============================================================

def create_summary(df, benchmark_id):
    """
    성공한 벤치마크 결과만 사용해서
    평균 FPS, 표준편차, 실행 횟수,
    V1 대비 향상률을 계산합니다.
    """

    success_df = df[
        df["status"] == "SUCCESS"
    ].copy()


    if success_df.empty:

        return None


    # --------------------------------------------------------
    # 평균 / 표준편차
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 각 해상도의 V1 평균 FPS
    # --------------------------------------------------------

    v1_fps = (
        summary_df[
            summary_df["version"] == "V1"
        ]
        .set_index("resolution")["mean_fps"]
    )


    # --------------------------------------------------------
    # V1 대비 성능 향상률
    # --------------------------------------------------------

    summary_df["improvement_vs_v1_pct"] = (
        summary_df.apply(
            lambda row: calculate_improvement(
                row,
                v1_fps
            ),
            axis=1
        )
    )


    # --------------------------------------------------------
    # Benchmark ID / 시간 추가
    # --------------------------------------------------------

    summary_df.insert(
        0,
        "benchmark_id",
        benchmark_id
    )


    summary_df.insert(
        1,
        "timestamp",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    return summary_df


# ============================================================
# 함수 7. 이번 실행 결과 출력
# ============================================================

def print_summary(summary_df):

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


# ============================================================
# 함수 8. Summary CSV 저장 + 직전 결과 가져오기
# ============================================================

def save_summary(summary_df):

    summary_csv_path = (
        RESULT_DIR
        / "fps_benchmark_summary.csv"
    )


    previous_summary = None


    # ========================================================
    # 기존 Summary가 있는 경우
    # ========================================================

    if summary_csv_path.exists():

        try:

            old_summary_df = pd.read_csv(
                summary_csv_path,
                encoding="utf-8-sig"
            )


            # ------------------------------------------------
            # 직전 benchmark_id 찾기
            # ------------------------------------------------

            if not old_summary_df.empty:

                previous_id = (
                    old_summary_df[
                        "benchmark_id"
                    ]
                    .astype(str)
                    .iloc[-1]
                )


                previous_summary = (
                    old_summary_df[
                        old_summary_df[
                            "benchmark_id"
                        ].astype(str)
                        == previous_id
                    ]
                    .copy()
                )


            # ------------------------------------------------
            # 기존 + 현재 Summary
            # ------------------------------------------------

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
                f"원인: "
                f"{type(e).__name__}: {e}"
            )

            all_summary_df = (
                summary_df.copy()
            )


    else:

        all_summary_df = (
            summary_df.copy()
        )


    # ========================================================
    # 누적 저장
    # ========================================================

    all_summary_df.to_csv(
        summary_csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    print()
    print("벤치마크 요약 CSV 누적 저장 완료:")
    print(summary_csv_path)


    return previous_summary


# ============================================================
# 함수 9. 현재 vs 직전 그래프 데이터 생성
# ============================================================

def create_graph_dataframe(
    summary_df,
    previous_summary
):

    # --------------------------------------------------------
    # 현재 실행
    # --------------------------------------------------------

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
    # 첫 번째 실행인 경우
    # --------------------------------------------------------

    else:

        current_graph_df.columns = [
            f"{col}_Current"
            for col in current_graph_df.columns
        ]

        graph_df = current_graph_df


    return graph_df


# ============================================================
# 함수 10. 그래프 생성 및 저장
# ============================================================

def save_graph(graph_df):

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


# ============================================================
# 함수 11. 최종 벤치마크 결과 출력
# ============================================================

def print_final_result(
    df,
    benchmark_id
):

    success_count = len(
        df[
            df["status"] == "SUCCESS"
        ]
    )


    error_count = len(
        df[
            df["status"] == "ERROR"
        ]
    )


    interrupted_count = len(
        df[
            df["status"] == "INTERRUPTED"
        ]
    )


    print()
    print("=" * 60)

    print("벤치마크 완료")

    print("=" * 60)


    print(
        f"Benchmark ID : {benchmark_id}"
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


# ============================================================
# 함수 12. run_full_benchmark
# ============================================================

def run_full_benchmark():

    # --------------------------------------------------------
    # 1. Benchmark ID 생성
    # --------------------------------------------------------

    benchmark_id = create_benchmark_id()


    # --------------------------------------------------------
    # 2. V1 / V2 / V3 벤치마크 실행
    # --------------------------------------------------------

    df = run_benchmark(
        benchmark_id
    )


    # --------------------------------------------------------
    # 3. 개별 Run CSV 저장
    # --------------------------------------------------------

    save_run_results(
        df
    )


    # --------------------------------------------------------
    # 4. 평균 / 표준편차 / 향상률 계산
    # --------------------------------------------------------

    summary_df = create_summary(
        df,
        benchmark_id
    )


    # --------------------------------------------------------
    # 성공한 테스트가 있는 경우
    # --------------------------------------------------------

    if summary_df is not None:

        # 결과 출력
        print_summary(
            summary_df
        )


        # Summary 저장 및 직전 결과 가져오기
        previous_summary = save_summary(
            summary_df
        )


        # 그래프용 데이터 생성
        graph_df = create_graph_dataframe(
            summary_df,
            previous_summary
        )


        # 그래프 저장
        save_graph(
            graph_df
        )


    # --------------------------------------------------------
    # 성공한 테스트가 없는 경우
    # --------------------------------------------------------

    else:

        print()

        print(
            "[WARNING] 성공한 테스트가 없어 "
            "평균/표준편차/그래프를 "
            "생성하지 않았습니다."
        )


    # --------------------------------------------------------
    # 최종 결과 출력
    # --------------------------------------------------------

    print_final_result(
        df,
        benchmark_id
    )


# ============================================================
# 프로그램 시작점
# ============================================================

if __name__ == "__main__":

    run_full_benchmark()