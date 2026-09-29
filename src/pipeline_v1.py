import sys
from pathlib import Path

# =============================================================================
# 프로젝트 루트 경로 등록
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import cv2
from src.utils import StepTimer, FPSMeter, load_config, process_green_circle

def run_pipeline_v1(video_source=0, config_path="config.json", show_window=True):
    """
    [V1 Baseline 파이프라인]
    단일 스레드로 프레임을 순차적으로 읽어와 전처리, 검출, 후처리, 렌더링을 수행합니다.
    (프로젝트 요구사항: 인위적 대기시간 없이 최대 연산 FPS 측정)
    """
    # 1. 설정 파일 로드 및 유틸리티 객체 초기화
    config = load_config(config_path)
    timer = StepTimer()
    fps_meter = FPSMeter()

    # 2. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[Error] 비디오 소스를 열 수 없습니다: {video_source}")
        return

    print("[Pipeline V1 Baseline] 시작합니다. ('q' 키로 종료)")

    try:
        while True:
            ret, frame = cap.read() # 프레임 읽기
            if not ret:
                print("[Pipeline V1] 프레임을 읽을 수 없거나 영상이 종료되었습니다.")
                break

            # 3. 비전 영상 수행 (7단계 세분화 타이머 전달)
            output_frame = process_green_circle(frame, config, timer=timer)

            # 4. FPS 업데이트
            fps_meter.tick()
            current_fps = fps_meter.get_fps()

            # 5. 실시간 화면 렌더링 (FPS 표기) 
            if show_window:
                cv2.putText(
                    output_frame,
                    f"V1 FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("Green Circle Detection - V1 (Baseline)", output_frame)

                # 최소 GUI 렌더링 딜레이만 부여 (최대 처리 속도 측정)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # 자원 해제 및 정리
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

        print("[Pipeline V1] 종료되었습니다.")
        avg_times = {name: timer.average(name) for name in timer.start_times}
        print("\n" + "="*40)
        print("    [V1 Baseline - Step Execution Times]")
        print("="*40)
        for step, ms in avg_times.items():
            print(f"  • {step:<12}: {ms:6.2f} ms")
        print("="*40)

    return {
        "fps": fps_meter.get_fps(),
        "avg_times": avg_times
    }

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    # samples/ 폴더 경로로 변경
    sample_path = str(project_root / "data" / "samples" / "tennis_sample_640x480.mp4")
    config_path = str(project_root / "config.json")
    run_pipeline_v1(sample_path, config_path=config_path, show_window=True)