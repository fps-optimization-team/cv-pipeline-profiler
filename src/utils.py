# utils.py — 모든 팀이 공통으로 쓰는 도우미 함수
import json
import os
import time
from collections import deque
import cv2
import numpy as np


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """한글이 들어간 경로도 읽을 수 있는 imread. 실패하면 None."""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img, params=None):
    """한글이 들어간 경로에도 저장할 수 있는 imwrite. 성공하면 True."""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img, params or [])
    if ok:
        buf.tofile(path)
    return ok


def open_source(source):
    """'0','1' 같은 숫자면 웹캠, 그 외에는 동영상 파일로 연다."""
    if str(source).isdigit():
        # Windows에서는 DirectShow(CAP_DSHOW)로 열면 카메라가 빨리 켜집니다.
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] 입력을 열 수 없습니다: {source}")
    return cap


class FPSMeter:
    """최근 N프레임 이동평균으로 FPS를 계산한다."""

    def __init__(self, window=30):
        self.intervals = deque(maxlen=window)
        self.last = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self.intervals.append(now - self.last)
        self.last = now
        total = sum(self.intervals)
        return len(self.intervals) / total if total > 0 else 0.0


def load_config(path=None):
    """조정할 값들을 config.json에서 읽어 딕셔너리로 돌려준다.
    path를 주지 않으면 프로젝트 폴더(src의 상위)의 config.json을 읽는다."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)



# =========================================================
# 1. 전처리 (Pre-processing)
# =========================================================
def preprocess(frame):
    """
    입력 프레임을 검출하기 좋은 형태로 전처리한다.

    - 원본 해상도 유지
    - 노이즈 감소
    - BGR → HSV 변환
    """

    # 노이즈 감소
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # BGR → HSV 변환
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    return hsv


# =========================================================
# 2. 검출 (Detection)
# =========================================================
def detect(hsv):
    """
    HSV 이미지에서 초록색 영역을 검출한다.
    검출된 영역은 흰색, 나머지는 검은색인 mask를 반환한다.
    """

    # 초록색 HSV 범위
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # 초록색 영역만 추출
    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    return mask


# =========================================================
# 3. 후처리 (Post-processing)
# =========================================================
def postprocess(mask):
    """
    검출된 mask를 정리하고
    초록색 물체의 외곽선을 찾는다.
    """

    # 모폴로지 연산에 사용할 커널
    kernel = np.ones((5, 5), np.uint8)

    # 작은 노이즈 제거
    cleaned_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # 끊어진 영역 연결
    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 외곽선 검출
    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


# =========================================================
# 4. 결과 그리기 (Visualization)
# =========================================================
def draw_results(frame, contours, fps=None):
    """
    검출된 물체에 Bounding Box를 그리고
    FPS를 화면에 표시한다.
    """

    output_frame = frame.copy()

    for contour in contours:

        # 너무 작은 영역은 노이즈로 판단하여 제외
        area = cv2.contourArea(contour)

        if area < 100:
            continue

        # Bounding Box 좌표 계산
        x, y, w, h = cv2.boundingRect(contour)

        # 사각형 그리기
        cv2.rectangle(
            output_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # FPS 표시
    if fps is not None:
        cv2.putText(
            output_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output_frame




# =========================================================
# 5. 공통 프레임 처리 파이프라인
# =========================================================
def process_frame(frame, draw=True):
    """
    공통 영상 처리 파이프라인

    전처리
        ↓
    초록색 검출
        ↓
    후처리
        ↓
    결과 그리기

    각 단계의 처리 시간과 FPS도 함께 측정한다.
    """

    total_start = time.perf_counter()


    # -----------------------------------------------------
    # 1. 전처리
    # -----------------------------------------------------
    start = time.perf_counter()

    hsv = preprocess(frame)

    preprocess_ms = (time.perf_counter() - start) * 1000


    # -----------------------------------------------------
    # 2. 검출
    # -----------------------------------------------------
    start = time.perf_counter()

    mask = detect(hsv)

    detect_ms = (time.perf_counter() - start) * 1000


    # -----------------------------------------------------
    # 3. 후처리
    # -----------------------------------------------------
    start = time.perf_counter()

    contours = postprocess(mask)

    postprocess_ms = (time.perf_counter() - start) * 1000


    # -----------------------------------------------------
    # 전체 처리 시간 및 FPS 계산
    # -----------------------------------------------------
    total_time = time.perf_counter() - total_start

    if total_time > 0:
        fps = 1.0 / total_time
    else:
        fps = 0.0


    # -----------------------------------------------------
    # 4. 결과 그리기
    # -----------------------------------------------------
    if draw:
        output_frame = draw_results(
            frame,
            contours,
            fps
        )
    else:
        output_frame = frame.copy()


    # -----------------------------------------------------
    # 성능 측정 결과
    # -----------------------------------------------------
    metrics = {
        "preprocess_ms": preprocess_ms,
        "detect_ms": detect_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_time * 1000,
        "fps": fps
    }


    return output_frame, contours, metrics