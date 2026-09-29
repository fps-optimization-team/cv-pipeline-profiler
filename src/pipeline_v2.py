import sys
from pathlib import Path
import threading
import queue

import cv2

# 프로젝트 루트 경로 등록
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import (
    StepTimer,
    FPSMeter,
    load_config,
    process_green_circle,
)


# ============================================================
# Producer Thread
# ============================================================

def frame_producer(cap, frame_queue, stop_event):
    """
    비디오 프레임을 읽어 Queue에 전달하는 Producer Thread.

    Queue가 가득 찬 경우 0.1초 간격으로 재시도하며,
    stop_event가 설정되면 안전하게 종료한다.
    """

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                print("[Producer] 영상의 마지막 프레임에 도달했습니다.")
                break

            # Queue가 가득 찬 경우 stop_event를 확인하며 재시도
            while not stop_event.is_set():
                try:
                    frame_queue.put(frame, timeout=0.1)
                    break

                except queue.Full:
                    continue

    finally:
        stop_event.set()
        print("[Producer] 종료되었습니다.")


# ============================================================
# V2 Multithread Pipeline
# ============================================================

def run_pipeline_v2(
    video_source=0,
    config_path="config.json",
    show_window=True,
):
    """
    V2 멀티스레드 파이프라인.

    Producer Thread:
        VideoCapture → Queue

    Consumer (Main Thread):
        Queue → process_green_circle() → 화면 출력
    """

    # 1. 초기 설정
    config = load_config(config_path)
    timer = StepTimer()
    fps_meter = FPSMeter()

    # 2. 비디오 열기
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] 비디오 소스를 열 수 없습니다: {video_source}")
        return None

    # 3. Producer / Consumer 공유 객체
    frame_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # 4. Producer Thread 생성 및 시작
    producer_thread = threading.Thread(
        target=frame_producer,
        args=(cap, frame_queue, stop_event),
    )
    producer_thread.start()

    print(
        "[Pipeline V2 - Multithread] "
        "멀티스레드 파이프라인을 시작합니다. ('q' 키로 종료)"
    )

    avg_times = {}

    try:
        # ====================================================
        # Consumer (Main Thread)
        # ====================================================
        while True:
            try:
                frame = frame_queue.get(timeout=0.1)

            except queue.Empty:
                # Producer 종료 + Queue 비어 있음 → 전체 처리 완료
                if stop_event.is_set() and frame_queue.empty():
                    break

                continue

            # 비전 연산
            output_frame = process_green_circle(
                frame,
                config,
                timer=timer,
            )

            # FPS 측정
            fps_meter.tick()
            current_fps = fps_meter.get_fps()

            # 화면 출력
            if show_window:
                cv2.putText(
                    output_frame,
                    f"V2 FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(
                    "Green Circle Detection - V2 (Multithread)",
                    output_frame,
                )

                # q 입력 시 종료
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Pipeline V2] q 입력 → 종료 요청")
                    stop_event.set()
                    break

            frame_queue.task_done()

    except KeyboardInterrupt:
        print("\n[Pipeline V2] Ctrl+C 입력 → 종료 요청")
        stop_event.set()

    finally:
        # ====================================================
        # 안전한 종료
        # ====================================================
        print("[Pipeline V2] 종료 작업을 시작합니다.")

        stop_event.set()
        producer_thread.join()

        cap.release()

        if show_window:
            cv2.destroyAllWindows()

        print("[Pipeline V2] 파이프라인이 안전하게 종료되었습니다.")

        # 단계별 평균 처리 시간
        avg_times = {
            name: timer.average(name)
            for name in timer.start_times
        }

        print("\n" + "=" * 50)
        print("    [V2 Multithread - Step Execution Times]")
        print("=" * 50)

        for step, ms in avg_times.items():
            print(f"  • {step:<15}: {ms:6.2f} ms")

        print("=" * 50)

    # 벤치마크에서 사용할 결과 반환
    return {
        "fps": fps_meter.get_fps(),
        "avg_times": avg_times,
    }


# ============================================================
# 단독 실행
# ============================================================

if __name__ == "__main__":

    sample_path = str(
        PROJECT_ROOT
        / "data"
        / "tennis_sample_1280x720.mp4"
    )

    sample_path = str(PROJECT_ROOT / "data" / "samples" / "tennis_sample_1280x720.mp4")
    config_path = str(PROJECT_ROOT / "config.json")
    run_pipeline_v2(sample_path, config_path=config_path, show_window=True)

    config_path = str(
        PROJECT_ROOT
        / "config.json"
    )

    run_pipeline_v2(
        sample_path,
        config_path=config_path,
        show_window=True,
    )