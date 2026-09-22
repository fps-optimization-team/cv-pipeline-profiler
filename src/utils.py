# utils.py — 모든 팀이 공통으로 쓰는 도우미 함수
import json
import os
import time
from collections import deque
import cv2
import numpy as np
import logging
from datetime import datetime


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
        """프레임 시간을 새로 측정하고 FPS를 반환한다."""
        now = time.perf_counter()
        self.intervals.append(now - self.last)
        self.last = now

        return self.get_fps()

    def get_fps(self):
        """시간을 새로 측정하지 않고 현재 FPS 값만 반환한다."""
        total = sum(self.intervals)
        return len(self.intervals) / total if total > 0 else 0.0

class StepTimer:
    """파이프라인의 각 단계별 처리 시간을 ms 단위로 측정하는 클래스"""

    def __init__(self):
        # 각 단계의 시작 시간을 저장
        self.start_times = {}

        # 각 단계에서 측정된 시간들을 저장
        self.times = {}

    def start(self, name):
        """name 단계의 시간 측정을 시작한다."""
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        """name 단계의 시간 측정을 종료하고 걸린 시간을 ms로 저장한다."""

        # 시작 시간이 없는 단계라면 측정하지 않음
        if name not in self.start_times:
            return

        end_time = time.perf_counter()

        elapsed_ms = (
            end_time - self.start_times[name]
        ) * 1000

        # 처음 측정하는 단계라면 빈 리스트 생성
        if name not in self.times:
            self.times[name] = []

        # 측정된 시간을 리스트에 추가
        self.times[name].append(elapsed_ms)

    def average(self, name):
        """해당 단계의 평균 처리 시간을 ms 단위로 반환한다."""

        values = self.times.get(name, [])

        if len(values) == 0:
            return 0.0

        return sum(values) / len(values)

    def reset(self):
        """저장된 모든 측정값을 초기화한다."""
        self.start_times.clear()
        self.times.clear()


def load_config(path=None):
    """조정할 값들을 config.json에서 읽어 딕셔너리로 돌려준다.
    path를 주지 않으면 프로젝트 폴더(src의 상위)의 config.json을 읽는다."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def process_green_circle(frame, config, timer=None):
    """
    [비전 연산 핵심 파이프라인]
    입력 프레임에서 초록색 원을 검출하고 Bounding Box 및 텍스트를 렌더링합니다.
    
    :param frame: 입력 BGR 이미지 (numpy ndarray)
    :param config: 설정값 사전 (config.json 데이터)
    :param timer: StepTimer 객체 (단계별 ms 측정용, 선택 사항)
    :return: 검출 결과가 그려진 출력 이미지
    """
    
    # =========================================================================
    # 1. 전처리 (Pre-processing) Phase
    # =========================================================================
    if timer:
        timer.start("preprocess")  # [타이머] 전처리 시작 시간 기록
        
    # config.json에서 가우시안 블러 커널 크기 가져오기 (기본값: (5, 5))
    blur_kernel = tuple(config.get("blur_kernel", [5, 5]))
    
    # 노이즈 제거를 위한 Gaussian Blur 적용
    blurred = cv2.GaussianBlur(frame, blur_kernel, 0)
    
    # 색상 공간 변환: BGR -> HSV (색상 추적이 용이한 HSV 공간 사용)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    if timer:
        timer.stop("preprocess")  # [타이머] 전처리 종료 시간 기록

    # =========================================================================
    # 2. 검출 (Detection) Phase
    # =========================================================================
    if timer:
        timer.start("detection")  # [타이머] 검출 시작 시간 기록
        
    # HSV 하한값/상한값 설정 (config.json에서 불러오기)
    lower_green = np.array(config.get("lower_green", [35, 100, 100]))
    upper_green = np.array(config.get("upper_green", [85, 255, 255]))
    
    # 초록색 범위에 해당하는 영역만 흰색(255), 나머지는 검은색(0) 마스크 생성
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    if timer:
        timer.stop("detection")  # [타이머] 검출 종료 시간 기록

    # =========================================================================
    # 3. 후처리 (Post-processing) Phase
    # =========================================================================
    if timer:
        timer.start("postprocess")  # [타이머] 후처리 시작 시간 기록
        
    # 모폴로지 연산에 사용할 사각형 커널 생성
    morph_kernel_size = tuple(config.get("morph_kernel", [3, 3]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    
    # Morphology Opening: 작은 흰색 점(노이즈) 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Morphology Closing: 객체 내부의 검은 구멍 채우기
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 이진화 마스크 이미지에서 외곽선(Contour) 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if timer:
        timer.stop("postprocess")  # [타이머] 후처리 종료 시간 기록

    # =========================================================================
    # 4. 결과 그리기 (Rendering Phase)
    # =========================================================================
    if timer:
        timer.start("draw")  # [타이머] 그리기 시작 시간 기록
        
    # 원본 이미지 훼손 방지를 위해 복사본 생성
    output_frame = frame.copy()
    
    # 판단 기준 최소 반지름 (기본값: 10픽셀)
    min_radius = config.get("min_radius", 10)
    
    # 검출된 모든 외곽선(Contour) 후보군 순회
    for cnt in contours:
        # 외곽선 둘레 길이 계산
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        # 외곽선 내부 면적 계산
        area = cv2.contourArea(cnt)
        
        # 원형도(Circularity) 계산 공식: 4 * pi * 면적 / (둘레^2) -> 완벽한 원일 때 1.0
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # 최소 외접원 계산 (중심 좌표 x, y 및 반지름 radius)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        
        # 조건 검증: 반지름 기준값 이상 + 원형도 0.6 이상인 경우만 '원'으로 인식
        if radius > min_radius and circularity > 0.6:
            center = (int(x), int(y))
            radius = int(radius)
            
            # 1) 검출된 원 테두리 그리기 (초록색, 두께 2)
            cv2.circle(output_frame, center, radius, (0, 255, 0), 2)
            
            # 2) Bounding Box 좌표 계산 및 직사각형 그리기 (노란색, 두께 2)
            x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(output_frame, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 255), 2)
            
            # 3) 상단에 라벨 텍스트 표기
            cv2.putText(output_frame, "Green Circle", (x_box, y_box - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    if timer:
        timer.stop("draw")  # [타이머] 그리기 종료 시간 기록

    # 최종 결과 프레임 반환
    return output_frame

def process_green_circle_v3(
    frame,
    config,
    timer=None,
    frame_idx=0,
    last_results=None,
    # ---------------------------------------------------------
    # [최적화 기법 4가지 온/오프 스위치 및 파라미터]
    # ---------------------------------------------------------
    use_downscale=True,     # [기법 1] Downscaling 적용 여부
    scale_factor=0.5,       # 축소 비율 (0.5 = 50% 축소)
    use_fast_interp=True,   # [기법 2] cv2.INTER_NEAREST 고속 보간법 사용 여부
    use_frame_skip=True,    # [기법 3] N-Frame Skip 사용 여부
    skip_interval=2,        # 스킵 주기 (2프레임당 1번 연산)
    use_roi=True            # [기법 4] ROI(관심 영역) 국한 연산 사용 여부
):
    """
    [V3 스위치형 최적화 비전 연산 파이프라인]
    4가지 최적화 기법을 매개변수 플래그로 온/오프하여 단독/조합 벤치마크 수행
    """
    
    # -------------------------------------------------------------------------
    # Phase 1. 전처리 (Downscaling & HSV 변환)
    # -------------------------------------------------------------------------
    if timer:
        timer.start("preprocess")

    # [최적화 1 & 2] Downscaling 및 보간법 선택
    if use_downscale and scale_factor < 1.0:
        interp = cv2.INTER_NEAREST if use_fast_interp else cv2.INTER_LINEAR
        proc_frame = cv2.resize(
            frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp
        )
        curr_scale = scale_factor
    else:
        proc_frame = frame
        curr_scale = 1.0

    blur_kernel = tuple(config.get("blur_kernel", [5, 5]))
    blurred = cv2.GaussianBlur(proc_frame, blur_kernel, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if timer:
        timer.stop("preprocess")

    # -------------------------------------------------------------------------
    # [최적화 3] N-Frame Skip 판별
    # -------------------------------------------------------------------------
    if use_frame_skip:
        should_detect = (frame_idx % skip_interval == 0) or (last_results is None)
    else:
        should_detect = True  # 매 프레임 검출 수행

    detected_circles = []

    if should_detect:
        # ---------------------------------------------------------------------
        # Phase 2. 검출 ([최적화 4] ROI 적용 및 inRange)
        # ---------------------------------------------------------------------
        if timer:
            timer.start("detection")

        lower_green = np.array(config.get("lower_green", [35, 100, 100]))
        upper_green = np.array(config.get("upper_green", [85, 255, 255]))

        # [최적화 4] ROI 국한 연산: 이전 프레임 결과가 존재하면 Bounding Box 주변만 검출
        if use_roi and last_results and len(last_results) > 0:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            xb, yb, wb, hb = last_results[0]["bbox_scaled"]
            margin = int(20 * curr_scale) # 축소 비율에 맞춰 마진 설정
            h_img, w_img = hsv.shape[:2]

            # 이미지 경계 초과 방지
            x1, y1 = max(0, xb - margin), max(0, yb - margin)
            x2, y2 = min(w_img, xb + wb + margin), min(h_img, yb + hb + margin)

            # 관심 영역(ROI)만 크롭하여 마스킹 수행
            roi_hsv = hsv[y1:y2, x1:x2]
            mask[y1:y2, x1:x2] = cv2.inRange(roi_hsv, lower_green, upper_green)
        else:
            # ROI 미사용 시 전체 화면 마스킹
            mask = cv2.inRange(hsv, lower_green, upper_green)

        if timer:
            timer.stop("detection")

        # ---------------------------------------------------------------------
        # Phase 3. 후처리 (Morphology & Contour 추출)
        # ---------------------------------------------------------------------
        if timer:
            timer.start("postprocess")

        morph_kernel_size = tuple(config.get("morph_kernel", [3, 3]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_radius = config.get("min_radius", 10) * curr_scale

        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            area = cv2.contourArea(cnt)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            (x, y), radius = cv2.minEnclosingCircle(cnt)

            if radius > min_radius and circularity > 0.6:
                x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
                detected_circles.append({
                    "center": (x, y),
                    "radius": radius,
                    "bbox_scaled": (x_box, y_box, w_box, h_box),
                    "scale_used": curr_scale
                })

        if timer:
            timer.stop("postprocess")
    else:
        # N-Frame Skip: 무거운 연산 건너뛰고 이전 검출 결과 그대로 사용
        detected_circles = last_results

    # -------------------------------------------------------------------------
    # Phase 4. 시각화 (원본 크기 좌표 복원 및 렌더링)
    # -------------------------------------------------------------------------
    if timer:
        timer.start("draw")

    output_frame = frame.copy()

    for item in detected_circles:
        scale = item.get("scale_used", 1.0)
        inv_scale = 1.0 / scale

        cx = int(item["center"][0] * inv_scale)
        cy = int(item["center"][1] * inv_scale)
        r = int(item["radius"] * inv_scale)

        xb, yb, wb, hb = item["bbox_scaled"]
        xb, yb = int(xb * inv_scale), int(yb * inv_scale)
        wb, hb = int(wb * inv_scale), int(hb * inv_scale)

        cv2.circle(output_frame, (cx, cy), r, (0, 255, 0), 2)
        cv2.rectangle(output_frame, (xb, yb), (xb + wb, yb + hb), (0, 255, 255), 2)
        cv2.putText(
            output_frame, "Green Circle (V3 Opt)", (xb, yb - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    if timer:
        timer.stop("draw")

    return output_frame, detected_circles