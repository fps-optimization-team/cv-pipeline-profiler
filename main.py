# main.py — 파이프라인 전체 시연 (V1 -> V2 -> V3) 및 벤치마크 통합 진입점
import argparse
from pathlib import Path

from src.benchmark import run_full_benchmark
from src.pipeline_v1 import run_pipeline_v1
from src.pipeline_v2 import run_pipeline_v2
from src.pipeline_v3 import run_pipeline_v3

# 프로젝트 최상위 루트 경로 자동 인식 (team_project/)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE = str(PROJECT_ROOT / "data" / "samples" / "tennis_sample_1280x720.mp4")
DEFAULT_CONFIG = str(PROJECT_ROOT / "config.json")


def main():
    parser = argparse.ArgumentParser(description="Green Circle Detection Project Main Entry")
    parser.add_argument("--source", default=DEFAULT_SAMPLE, help="시연용 샘플 영상 경로")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="설정 파일 경로")
    parser.add_argument("--skip-demo", action="store_true", help="시연(V1~V3)을 건너뛰고 벤치마크만 바로 실행")
    args = parser.parse_args()

    print("==================================================")
    print("Green Circle Detection Project Main Execution")
    print("==================================================")

    # --skip-demo 옵션이 붙지 않은 경우 V1 -> V2 -> V3 시연 순차 진행
    if not args.skip_demo:
        # 1. V1 시연 (Baseline)
        print("\n[1/4] V1 파이프라인 실행 중 (Baseline - 단일 스레드)...")
        print(" -> 영상 창을 닫거나 'q'를 누르면 다음 버전으로 넘어갑니다.")
        run_pipeline_v1(video_source=args.source, config_path=args.config, show_window=True)

        # 2. V2 시연 (I/O Multi-threading)
        print("\n[2/4] V2 파이프라인 실행 중 (I/O Multi-threading Queue)...")
        print(" -> 영상 창을 닫거나 'q'를 누르면 다음 버전으로 넘어갑니다.")
        run_pipeline_v2(video_source=args.source, config_path=args.config, show_window=True)

        # 3. V3 시연 (Algorithm Optimization)
        print("\n[3/4] V3 파이프라인 실행 중 (Downsampling & ROI Optimization)...")
        print(" -> 영상 창을 닫거나 'q'를 누르면 벤치마크 측정을 시작합니다.")
        run_pipeline_v3(video_source=args.source, config_path=args.config, show_window=True)

    # 4. 전체 벤치마크 자동 측정 실행
    print("\n[4/4] 전체 벤치마크 측정 및 결과 리포트 생성 시작...")
    run_full_benchmark()

    print("\n==================================================")
    print(" 모든 파이프라인 시연 및 벤치마크 완료!")
    print("    결과물 저장 위치: ./results/")
    print("==================================================")


if __name__ == "__main__":
    main()