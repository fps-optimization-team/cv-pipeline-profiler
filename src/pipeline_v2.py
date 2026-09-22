# pipeline_v2.py
# 해상도별 파이프라인 단계 처리시간 측정

import os
import cv2

from utils import (
    open_source,
    load_config,
    process_green_circle,
    StepTimer
)


# ============================================================
# 1. 프로젝트 경로 설정
# ============================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")


# ============================================================
# 2. 테스트할 샘플 영상 목록
# ============================================================

videos = [
    "sample_640x480.mp4",
    "sample_1280x720.mp4",
    "sample_1920x1080.mp4"
]


# ============================================================
# 3. config.json 불러오기
# ============================================================

config = load_config()


# ============================================================
# 4. 영상별 결과 저장
# ============================================================

results = []


# ============================================================
# 5. 세 영상을 하나씩 테스트
# ============================================================

for video_name in videos:

    video_path = os.path.join(DATA_DIR, video_name)

    print()
    print("=" * 60)
    print(f"테스트 시작: {video_name}")
    print("=" * 60)


    # 영상 열기
    cap = open_source(video_path)


    # 이 영상 전용 StepTimer 생성
    timer = StepTimer()


    # 처리한 프레임 개수
    frame_count = 0


    # ========================================================
    # 6. 프레임 단위 처리
    # ========================================================
    MAX_FRAMES = 200
    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # 핵심 CV 파이프라인 실행
        # timer를 전달하면 utils.py 내부에서
        # preprocess / detection / postprocess / draw 시간이 측정됨
        result = process_green_circle(
            frame,
            config,
            timer
        )


        frame_count += 1


        # 처리 결과 확인
        cv2.imshow(
            "Pipeline V2 - Step Profiler",
            result
        )

        # 모든 영상에서 처음 200프레임만 정확히 측정하도록
        frame_count += 1
        if frame_count >= MAX_FRAMES:
            break


        # q를 누르면 현재 영상 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()


    # ========================================================
    # 7. 단계별 평균 처리시간 계산
    # ========================================================

    preprocess_ms = timer.average("preprocess")
    detection_ms = timer.average("detection")
    postprocess_ms = timer.average("postprocess")
    draw_ms = timer.average("draw")


    # 네 단계의 평균 시간 합계
    total_ms = (
        preprocess_ms
        + detection_ms
        + postprocess_ms
        + draw_ms
    )
    # ========================================================
    # Postprocess 내부 세부 처리시간
    # ========================================================

    blur_ms = timer.average("blur")
    hsv_convert_ms = timer.average("hsv_convert")
    in_range_ms = timer.average("in_range")

    morph_open_ms = timer.average("morph_open")
    morph_close_ms = timer.average("morph_close")
    find_contours_ms = timer.average("find_contours")


    # 세부 연산별 시간을 딕셔너리로 정리
    all_steps = {
        "blur": blur_ms,
        "hsv_convert": hsv_convert_ms,
        "in_range": in_range_ms,
        "morph_open": morph_open_ms,
        "morph_close": morph_close_ms,
        "find_contours": find_contours_ms,
        "draw": draw_ms
    }


    # 가장 오래 걸린 연산 찾기
    bottleneck = max(
        all_steps,
        key=all_steps.get
    )


    # 병목 후보 출력
    print(
        f"전체 연산 병목 후보: "
        f"{bottleneck} "
        f"({all_steps[bottleneck]:.3f} ms)"
    )

    # ========================================================
    # 8. 결과 저장
    # ========================================================

    results.append({
         "video": video_name,
        "frames": frame_count,

        # 큰 단계
        "preprocess": preprocess_ms,
        "detection": detection_ms,
        "postprocess": postprocess_ms,

        # 세부 연산 7개
        "blur": blur_ms,
        "hsv_convert": hsv_convert_ms,
        "in_range": in_range_ms,
        "morph_open": morph_open_ms,
        "morph_close": morph_close_ms,
        "find_contours": find_contours_ms,
        "draw": draw_ms,

        # 전체 연산 중 가장 오래 걸린 연산
        "bottleneck": bottleneck
    })


# ============================================================
# 9. 모든 OpenCV 창 닫기
# ============================================================

cv2.destroyAllWindows()


# ============================================================
# 10. 최종 결과 출력
# ============================================================

print()
print("=" * 145)
print("Pipeline V2 - 단계별 성능 비교")
print("=" * 145)

print(
    f"{'Video':25s}"
    f"{'Frames':>10s}"
    f"{'Pre(ms)':>12s}"
    f"{'Detect(ms)':>12s}"
    f"{'Post(ms)':>12s}"
    f"{'Blur':>9s}"
    f"{'HSV':>9s}"
    f"{'InRange':>9s}"
    f"{'Draw(ms)':>12s}"
    f"{'Open':>9s}"
    f"{'Close':>9s}"
    f"{'Contour':>9s}"
    f"{'Bottleneck':>16s}"
)

print("-" * 145)


for r in results:

    print(
        f"{r['video']:25s}"
        f"{r['frames']:10d}"
        f"{r['preprocess']:12.3f}"
        f"{r['detection']:12.3f}"
        f"{r['postprocess']:12.3f}"
        f"{r['blur']:9.3f}"
        f"{r['hsv_convert']:9.3f}"
        f"{r['in_range']:9.3f}"
        f"{r['draw']:12.3f}"
        f"{r['morph_open']:9.3f}"
        f"{r['morph_close']:9.3f}"
        f"{r['find_contours']:9.3f}"
        f"{r['bottleneck']:>16s}"
    )


print("=" * 145)