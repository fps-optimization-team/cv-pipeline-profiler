import sys
from pathlib import Path

# team_project 루트 등록
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import time
from src.utils import StepTimer, FPSMeter, load_config, process_green_circle

def run_pipeline_v1(video_source=0, config_path="config.json", show_window=True):
    """
    [V1 Baseline 파이프라인]
    단일 스레드로 프레임을 순차적으로 읽어와 전처리, 검출, 후처리, 렌더링을 수행합니다.
    
    :param video_source: 카메라 인덱스(0, 1...) 또는 동영상 파일 경로
    :param config_path: config.json 경로
    :param show_window: 실시간 화면 출력 여부 (벤치마크 시 False로 설정 가능)
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

    print("[Pipeline V1] 시작합니다. (종료하려면 'q' 키를 누르세요)")

    try:
        while(True):
            #프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("[Pipeline V1] 프레임을 읽을 수 없거나 영상이 종료되었습니다.")
                break

            # 3. 비전 영상 수행 (7단계 세분화 타이머 전달)
            output_frame = process_green_circle(frame, config, timer=timer)

            # 4. FPS 및 프레임 드롭률 업데이트
            fps_meter.tick()
            current_fps = fps_meter.get_fps()

            # 5. 실시간 화면 렌더링 (FPS 및 연산 시간 표기) 
            if show_window:
                # 화면 상단에 실시간 FPS 표기
                cv2.putText(
                    output_frame,
                    f"V1 FPS: {current_fps:1f}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0 ,255 ,0),
                    2
            )

            # 6. 결과 출력
            cv2.imshow("Green Circle Detection - V1 (Baseline) ", output_frame)

            # 7.'q' 키 입력 시 루프 탈출
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 자원 해제 및 정리
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

        print("[Pipeline V1] 종료되었습니다.")
        # 연산 단계별 평균 소요시간(ms) 출력 (디버깅용)
        avg_times = {name: timer.average(name) for name in timer.start_times}
        print("\n" + "="*40)
        print("    [V1 Baseline - Step Execution Times]")
        print("="*40)
        for step, ms in avg_times.items():
            print(f"  • {step:<12}: {ms:6.2f} ms")
        print("="*40)

if __name__ == "__main__":
    # 단독 실행 테스트용 (웹캠 0번 기준)
    project_root = Path(__file__).resolve().parent.parent
    sample_path = str(project_root / "data" / "sample_640x480.mp4")
    config_path = str(project_root / "config.json")
    run_pipeline_v1(sample_path, config_path=config_path, show_window=True)