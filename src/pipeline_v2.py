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
    [Producer Thread]

    비디오에서 프레임을 읽고
    Queue를 통해 Consumer(Main Thread)에게 전달합니다.

    중요:
    Queue가 꽉 찬 경우 put()에서 영원히 기다리지 않고
    0.1초마다 stop_event를 확인합니다.
    """

    try:

        while not stop_event.is_set():

            # ---------------------------------------------------------
            # 1. 비디오에서 프레임 읽기
            # ---------------------------------------------------------

            ret, frame = cap.read()

            if not ret:
                print("[Producer] 영상의 마지막 프레임에 도달했습니다.")
                break


            # ---------------------------------------------------------
            # 2. 읽은 프레임을 Queue에 넣기
            # ---------------------------------------------------------
            # Queue가 꽉 찬 경우에도
            # stop_event를 확인할 수 있도록 timeout 사용

            while not stop_event.is_set():

                try:

                    frame_queue.put(
                        frame,
                        timeout=0.1
                    )

                    # Queue에 정상적으로 넣었다면
                    # 내부 while 종료
                    break

                except queue.Full:

                    # Queue가 꽉 찼다면
                    # 다시 stop_event를 확인
                    continue

    finally:

        # Producer가 끝났음을 Consumer에게 알림
        stop_event.set()

        print("[Producer] 종료되었습니다.")


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

    Producer Thread
        ↓
    VideoCapture
        ↓
    Queue
        ↓
    Consumer(Main Thread)
        ↓
    OpenCV 처리
        ↓
    화면 출력


    Producer:
        비디오 프레임 읽기

    Consumer:
        Queue에서 프레임을 가져와
        process_green_circle() 실행
    """


    # =========================================================================
    # 1. 설정 및 유틸리티 객체 초기화
    # =========================================================================

    config = load_config(config_path)

    timer = StepTimer()

    fps_meter = FPSMeter()


    # =========================================================================
    # 2. 비디오 소스 열기
    # =========================================================================

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():

        print(
            f"[ERROR] 비디오 소스를 열 수 없습니다: "
            f"{video_source}"
        )

        return


    # =========================================================================
    # 3. Producer / Consumer 공유 객체
    # =========================================================================

    # 최대 5개의 프레임을 저장
    frame_queue = queue.Queue(
        maxsize=5
    )

    # Producer와 Consumer가 공유하는 종료 신호
    stop_event = threading.Event()


    # =========================================================================
    # 4. Producer Thread 생성
    # =========================================================================

    producer_thread = threading.Thread(

        target=frame_producer,

        args=(
            cap,
            frame_queue,
            stop_event
        )
    )


    # =========================================================================
    # 5. Producer Thread 시작
    # =========================================================================

    producer_thread.start()


    print(
        "[Pipeline V2 - Multithread] "
        "멀티스레드 파이프라인을 시작합니다. "
        "('q' 키로 종료)"
    )


    # avg_times를 미리 만들어 둠
    # 중간에 예외가 발생하더라도 return에서 참조 가능
    avg_times = {}


    try:

        # =====================================================================
        # Consumer (Main Thread)
        # =====================================================================

        while True:

            # -----------------------------------------------------------------
            # 6. Queue에서 프레임 가져오기
            # -----------------------------------------------------------------

            try:

                frame = frame_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                # Producer가 종료되었고
                # Queue까지 완전히 비었다면 종료
                if (
                    stop_event.is_set()
                    and frame_queue.empty()
                ):

                    break

                continue


            # -----------------------------------------------------------------
            # 7. 비전 연산
            # -----------------------------------------------------------------

            output_frame = process_green_circle(
                frame,
                config,
                timer=timer
            )


            # -----------------------------------------------------------------
            # 8. FPS 갱신
            # -----------------------------------------------------------------

            fps_meter.tick()

            current_fps = fps_meter.get_fps()


            # -----------------------------------------------------------------
            # 9. 실시간 화면 출력
            # -----------------------------------------------------------------

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


                # -------------------------------------------------------------
                # 10. q 입력 확인
                # -------------------------------------------------------------

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    print(
                        "[Pipeline V2] "
                        "q 입력 → 종료 요청"
                    )

                    # Producer에게 종료 요청
                    stop_event.set()

                    # Consumer 반복문 종료
                    break


            # -----------------------------------------------------------------
            # 11. 현재 프레임 처리 완료
            # -----------------------------------------------------------------

            frame_queue.task_done()


    except KeyboardInterrupt:

        # Ctrl + C 종료도 안전하게 처리

        print(
            "\n[Pipeline V2] "
            "KeyboardInterrupt → 종료 요청"
        )

        stop_event.set()


    finally:

        # =====================================================================
        # 12. 종료 처리
        # =====================================================================

        print(
            "[Pipeline V2] "
            "종료 작업을 시작합니다."
        )


        # ---------------------------------------------------------------------
        # Producer 종료 요청
        # ---------------------------------------------------------------------

        stop_event.set()


        # ---------------------------------------------------------------------
        # Producer Thread 종료 기다리기
        # ---------------------------------------------------------------------
        #
        # Producer의 put()에 timeout이 있기 때문에
        # Queue가 꽉 차 있어도 정상적으로 빠져나올 수 있음
        # ---------------------------------------------------------------------

        producer_thread.join()


        # ---------------------------------------------------------------------
        # VideoCapture 해제
        # ---------------------------------------------------------------------

        cap.release()


        # ---------------------------------------------------------------------
        # OpenCV Window 제거
        # ---------------------------------------------------------------------

        if show_window:

            cv2.destroyAllWindows()


        print(
            "[Pipeline V2] "
            "파이프라인이 안전하게 종료되었습니다."
        )


        # =====================================================================
        # 13. 단계별 평균 처리시간 계산
        # =====================================================================

        avg_times = {

            name: timer.average(name)

            for name in timer.start_times
        }


        # =====================================================================
        # 14. 결과 출력
        # =====================================================================

        print("\n" + "=" * 50)

        print(
            "    [V2 Multithread - Step Execution Times]"
        )

        print("=" * 50)


        for step, ms in avg_times.items():

            print(
                f"  • {step:<15}: "
                f"{ms:6.2f} ms"
            )


        print("=" * 50)


    # =========================================================================
    # 15. 결과 반환
    # =========================================================================

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
        / "tennis_sample_1280x720.mp4"
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