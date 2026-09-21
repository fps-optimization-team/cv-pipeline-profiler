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
    입력 프레임의 원본 해상도를 유지하면서
    모델 입력에 필요한 전처리를 수행한다.
    """

    # 원본 해상도 확인
    height, width = frame.shape[:2]

    # 노이즈 감소
    blurred = cv2.GaussianBlur(
        frame,
        (5, 5),
        0
    )

    # OpenCV BGR → RGB 변환
    rgb = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2RGB
    )

    # 0~255 → 0~1 정규화
    normalized = rgb.astype(np.float32) / 255.0

    return normalized


# =========================================================
# 2. 후처리 (Post-processing)
# =========================================================
def postprocess(raw_preds, orig_shape, conf_thresh=0.5):
    """
    모델의 원본 출력값을 실제 검출 결과로 변환한다.

    추후 모델이 정해지면
    - Confidence Threshold
    - NMS
    - 좌표 변환
    등을 구현한다.
    """

    # TODO: 실제 모델에 맞게 구현
    detections = raw_preds

    return detections


# =========================================================
# 3. 결과 그리기 (Visualization)
# =========================================================
def draw_results(frame, detections, fps=None):
    """
    원본 프레임에 검출 결과와 FPS를 표시한다.
    """

    output = frame.copy()

    # TODO:
    # 모델의 detections 구조가 결정되면
    # Bounding Box 등을 그리는 코드 추가

    if fps is not None:
        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output


# =========================================================
# 4. 공통 프레임 처리 파이프라인
# =========================================================
def process_frame(frame, model, conf_threshold=0.5, draw=True):
    """
    공통 프레임 처리 파이프라인

    원본 해상도를 유지한 상태에서

    전처리
      ↓
    모델 추론
      ↓
    후처리
      ↓
    결과 그리기

    순서로 처리한다.
    """

    total_start = time.perf_counter()

    # -----------------------------------------------------
    # 1. 전처리
    # -----------------------------------------------------
    pre_start = time.perf_counter()

    input_tensor = preprocess(frame)

    pre_end = time.perf_counter()


    # -----------------------------------------------------
    # 2. 검출 / AI 추론
    # -----------------------------------------------------
    detect_start = time.perf_counter()

    raw_preds = model(input_tensor)

    detect_end = time.perf_counter()


    # -----------------------------------------------------
    # 3. 후처리
    # -----------------------------------------------------
    post_start = time.perf_counter()

    detections = postprocess(
        raw_preds,
        orig_shape=frame.shape,
        conf_thresh=conf_threshold
    )

    post_end = time.perf_counter()


    # -----------------------------------------------------
    # 단계별 처리 시간 계산 (ms)
    # -----------------------------------------------------
    preprocess_ms = (pre_end - pre_start) * 1000
    detect_ms = (detect_end - detect_start) * 1000
    postprocess_ms = (post_end - post_start) * 1000


    # -----------------------------------------------------
    # 4. 결과 그리기
    # -----------------------------------------------------
    output_frame = frame.copy()

    total_end = time.perf_counter()

    total_time = total_end - total_start

    if total_time > 0:
        fps = 1.0 / total_time
    else:
        fps = 0.0


    if draw:
        output_frame = draw_results(
            output_frame,
            detections,
            fps
        )


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


    return output_frame, detections, metrics