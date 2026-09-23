import sys
from pathlib import Path

# =============================================================================
# 프로젝트 루트 경로 등록
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import cv2
from src.utils import StepTimer, FPSMeter, load_config, process_green_circle_v3

def run_pipeline_v3(video_source=0, config_path="config.json", show_window=True):
    """
    [V3 Optimized 파이프라인 메인 실행 함수]
    
    :param video_source: 비디오 파일 경로 또는 카메라 디바이스 인덱스
    :param config_path: 설정 파일(config.json) 경로
    :param show_window: 화면 렌더링 출력 여부
    """
    # [최적화] OpenCV 내 CPU 멀티스레드 세팅 (불필요한 스레드 오버헤드 방지)
    cv2.setNumThreads(1)

    # 유틸리티 객체 및 설정 불러오기
    config = load_config(config_path)
    timer = StepTimer()
    fps_meter = FPSMeter()

    # 비디오 소스 오픈
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] 비디오 소스를 열 수 없습니다: {video_source}")
        return

    print("[Pipeline V3 - Optimized] 최적화 파이프라인을 시작합니다. ('q' 키로 종료)")

    frame_idx = 0       # 프레임 인덱스 카운터
    last_results = None # 이전 프레임의 검출 결과 재사용 변수

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Pipeline V3] 영상 종료 또는 프레임을 읽을 수 없습니다.")
                break

            # utils.py에 모듈화된 V3 최적화 처리 함수 호출
            output_frame, last_results = process_green_circle_v3(
                frame,
                config,
                timer=timer,
                frame_idx=frame_idx,
                last_results=last_results,
                scale_factor=0.5
            )

            # 실시간 FPS 갱신
            fps_meter.tick()
            current_fps = fps_meter.get_fps()

            if show_window:
                # 좌상단 실시간 V3 FPS 표기
                cv2.putText(
                    output_frame,
                    f"V3 FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("Green Circle Detection - V3 (Optimized)", output_frame)

                # 'q' 키 누를 시 안전하게 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1  # 프레임 번호 증가

    finally:
        # 비디오 및 출력 창 메모리 해제
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

        print("[Pipeline V3] 파이프라인이 안전하게 종료되었습니다.")

        # 단계별 평균 소요시간(ms) 가독성 있게 출력
        avg_times = {name: timer.average(name) for name in timer.start_times}
        print("\n" + "=" * 40)
        print("    [V3 Optimized - Step Execution Times]")
        print("=" * 40)
        for step, ms in avg_times.items():
            print(f"  • {step:<12}: {ms:6.2f} ms")
        print("=" * 40)
    return {
    "fps": fps_meter.get_fps(),
    "avg_times": avg_times
    }


if __name__ == "__main__":
    # 실행에 필요한 절대 경로 설정
    sample_path = str(PROJECT_ROOT / "data" / "sample_1920x1080.mp4")
    config_path = str(PROJECT_ROOT / "config.json")

    # V3 메인 파이프라인 단독 실행
    run_pipeline_v3(sample_path, config_path=config_path, show_window=True)