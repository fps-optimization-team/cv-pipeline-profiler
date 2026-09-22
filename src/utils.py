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
    
    # 노이즈 제거를 위한 Gaussian Blur 적용 및 측정
    if timer:
        timer.start("blur")

    blurred = cv2.GaussianBlur(
        frame,
        blur_kernel,
        0
    )

    if timer:
        timer.stop("blur")


    # 색상 공간 변환 및 측정: BGR -> HSV (색상 추적이 용이한 HSV 공간 사용)
    if timer:
        timer.start("hsv_convert")

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    if timer:
        timer.stop("hsv_convert")
    
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
    if timer:
        timer.start("in_range")
    mask = cv2.inRange(hsv, lower_green, upper_green)
    if timer:
        timer.stop("in_range")
    
    if timer:
        timer.stop("detection")  # [타이머] 검출 종료 시간 기록

   
    # ============================================================
    # 3. 후처리 (Post-processing) Phase
    # ============================================================

    # Postprocess 전체 측정 시작
    if timer:
        timer.start("postprocess")


    # 모폴로지 연산에 사용할 커널 생성
    morph_kernel_size = tuple(
        config.get("morph_kernel", [3, 3])
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        morph_kernel_size
    )


    # ------------------------------------------------------------
    # 3-1. Morphology Opening 측정
    # ------------------------------------------------------------

    if timer:
        timer.start("morph_open")

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    if timer:
        timer.stop("morph_open")


    # ------------------------------------------------------------
    # 3-2. Morphology Closing 측정
    # ------------------------------------------------------------

    if timer:
        timer.start("morph_close")

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    if timer:
        timer.stop("morph_close")


    # ------------------------------------------------------------
    # 3-3. Contour 검출 측정
    # ------------------------------------------------------------

    if timer:
        timer.start("find_contours")

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if timer:
        timer.stop("find_contours")


    # Postprocess 전체 측정 종료
    if timer:
        timer.stop("postprocess")

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