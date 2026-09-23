import sys
from pathlib import Path

# =============================================================================
# 프로젝트 루트 경로 등록
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import cv2
import threading
import queue

from src.utils import (
    StepTimer,
    FPSMeter,
    load_config,
    process_green_circle
)


# =============================================================================
# Producer
# =============================================================================

def frame_producer(cap, frame_queue, stop_event):
    """
    [Producer]
    별도 스레드에서 비디오 프레임을 읽어 Queue에 전달합니다.
    """

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:
            break

        # Queue가 가득 차면 공간이 생길 때까지 기다림
        # 벤치마크에서는 프레임을 버리지 않음
        frame_queue.put(frame)

    stop_event.set()




# =============================================================================
# V2 Pipeline
# =============================================================================

def run_pipeline_v2(
    video_source=0,
    config_path="config.json",
    show_window=True
):
    """
    [V2 Multithread 파이프라인]

    Producer Thread:
        비디오 프레임 읽기

    Consumer(Main Thread):
        Queue에서 프레임을 가져와 비전 연산 수행

    :param video_source: 비디오 파일 경로 또는 카메라 인덱스
    :param config_path: config.json 경로
    :param show_window: 실시간 화면 출력 여부
    """

    # 1. 설정 및 유틸리티 객체 초기화
    config = load_config(config_path)
    timer = StepTimer()
    fps_meter = FPSMeter()

    # 2. 비디오 소스 열기
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] 비디오 소스를 열 수 없습니다: {video_source}")
        return

    # 3. Producer / Consumer 공유 객체
    frame_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # 4. Producer Thread 생성
    producer_thread = threading.Thread(
        target=frame_producer,
        args=(cap, frame_queue, stop_event)
    )

    producer_thread.start()

    print(
        "[Pipeline V2 - Multithread] "
        "멀티스레드 파이프라인을 시작합니다. ('q' 키로 종료)"
    )

    try:
        # =============================================================
        # Consumer(Main Thread)
        # =============================================================

        while True:

            try:
                frame = frame_queue.get(timeout=0.1)

            except queue.Empty:

                if stop_event.is_set() and frame_queue.empty():
                    break

                continue

            # 비전 연산
            output_frame = process_green_circle(
                frame,
                config,
                timer=timer
            )

            # FPS 갱신
            fps_meter.tick()
            current_fps = fps_meter.get_fps()

            # 실시간 화면 출력
            if show_window:

                cv2.putText(
                    output_frame,
                    f"V2 FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    "Green Circle Detection - V2 (Multithread)",
                    output_frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

    finally:

        # Producer 종료 요청
        stop_event.set()

        # Producer가 완전히 종료될 때까지 대기
        producer_thread.join()

        # 비디오 자원 해제
        cap.release()

        if show_window:
            cv2.destroyAllWindows()

        print(
            "[Pipeline V2] "
            "파이프라인이 안전하게 종료되었습니다."
        )

        # 단계별 평균 처리시간 출력
        avg_times = {
            name: timer.average(name)
            for name in timer.start_times
        }

        print("\n" + "=" * 40)
        print("    [V2 Multithread - Step Execution Times]")
        print("=" * 40)

        for step, ms in avg_times.items():
            print(f"  • {step:<12}: {ms:6.2f} ms")

        print("=" * 40)
    return {
    "fps": fps_meter.get_fps(),
    "avg_times": avg_times
    }


# =============================================================================
# 단독 실행
# =============================================================================

if __name__ == "__main__":

    sample_path = str(
        PROJECT_ROOT
        / "data"
        / "sample_1920x1080.mp4"
    )

    config_path = str(
        PROJECT_ROOT
        / "config.json"
    )

    run_pipeline_v2(
        sample_path,
        config_path=config_path,
        show_window=True
    )

