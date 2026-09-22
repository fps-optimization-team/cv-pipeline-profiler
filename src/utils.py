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

def process_green_circle_v3(frame, config, timer=None, frame_idx=0, last_results=None, scale_factor=0.5):
    """
    [V3 최적화 비전 연산 파이프라인 함수]
    
    적용된 핵심 최적화 기법 4가지:
    1. Downscaling (다운스케일링): 입력 프레임 해상도를 축소(기본 50%)하여 연산량 절감
    2. Fast Interpolation (고속 보간법): cv2.INTER_NEAREST를 사용하여 크기 조절 속도 극대화
    3. N-Frame Skip (프레임 스킵/결과 재사용): N프레임마다 1번만 무거운 검출 연산을 수행하고 중간은 재사용
    4. Coordinate Rescaling (좌표 복원): 축소된 좌표계에서 검출한 결과를 원본 크기 좌표계로 정밀 변환
    
    :param frame: 원본 BGR 입력 이미지 (numpy ndarray)
    :param config: config.json에서 로드한 설정값 사전
    :param timer: 단계별 ms 소요 시간을 측정하는 StepTimer 객체 (선택)
    :param frame_idx: 현재 처리 중인 프레임 번호 (N-Frame Skip 판별용)
    :param last_results: 이전 프레임에서 검출해둔 객체 정보 리스트 (재사용 목적)
    :param scale_factor: 축소 비율 (0.5 = 가로/세로 50% 축소 -> 전체 픽셀 수 75% 감소)
    :return: (결과 라벨이 그려진 출력 이미지, 현재 프레임의 검출 결과 데이터)
    """
    
    # -------------------------------------------------------------------------
    # Phase 1. 전처리 (Pre-processing Phase)
    # -------------------------------------------------------------------------
    if timer:
        timer.start("preprocess")  # [타이머 측정] 전처리 단계 시작

    # [최적화 1 & 2] INTER_NEAREST 속도 우수 보간법을 사용하여 이미지 해상도 축소
    # (예: 640x480 -> 320x240 / 1920x1080 -> 960x540)
    small_frame = cv2.resize(
        frame, 
        (0, 0), 
        fx=scale_factor, #default : 0.5
        fy=scale_factor, #default : 0.5
        interpolation=cv2.INTER_NEAREST
    )

    # config.json에서 가우시안 블러 커널 크기 추출 (기본값: (5, 5))
    blur_kernel = tuple(config.get("blur_kernel", [5, 5]))
    
    # 축소된 프레임에 노이즈 제거 블러 적용
    blurred = cv2.GaussianBlur(small_frame, blur_kernel, 0)
    
    # BGR 색상 공간을 색상(H), 색상농도(S), 명도(V)를 분리할 수 있는 HSV 공간으로 변환
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if timer:
        timer.stop("preprocess")   # [타이머 측정] 전처리 단계 종료

    # -------------------------------------------------------------------------
    # N-Frame Skip 조건 검사
    # -------------------------------------------------------------------------
    # skip_interval 프레임 주기(예: 2프레임)마다 연산을 실행하거나, 이전 검출 결과가 없는 경우 연산 수행
    skip_interval = config.get("skip_interval", 2)
    should_detect = (frame_idx % skip_interval == 0) or (last_results is None)

    # 현재 프레임에서 검출된 객체 정보를 담을 리스트 초기화
    detected_circles = []

    if should_detect:
        # ---------------------------------------------------------------------
        # Phase 2. 검출 (Detection Phase)
        # ---------------------------------------------------------------------
        if timer:
            timer.start("detection")  # [타이머 측정] 검출 단계 시작

        # config.json에서 초록색 HSV 범위 threshold 값 로드
        lower_green = np.array(config.get("lower_green", [35, 100, 100]))
        upper_green = np.array(config.get("upper_green", [85, 255, 255]))
        
        # 설정한 HSV 범위에 포함되는 픽셀만 255(흰색), 나머지는 0(검은색)으로 이진화 마스크 생성
        mask = cv2.inRange(hsv, lower_green, upper_green)

        if timer:
            timer.stop("detection")   # [타이머 측정] 검출 단계 종료

        # ---------------------------------------------------------------------
        # Phase 3. 후처리 (Post-processing Phase)
        # ---------------------------------------------------------------------
        if timer:
            timer.start("postprocess")  # [타이머 측정] 후처리 단계 시작

        # 노이즈 제거를 위한 모폴로지 연산 커널 생성
        morph_kernel_size = tuple(config.get("morph_kernel", [3, 3]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
        
        # Opening: 점 노이즈 제거 / Closing: 객체 내부 빈 구멍 채우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 이진화 마스크에서 객체 외곽선(Contour) 좌표 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 이미지 해상도가 축소된 상태이므로 판단 최소 반지름도 축소 비율에 맞춰 조정
        min_radius = config.get("min_radius", 10) * scale_factor

        # 검출된 모든 외곽선 후보 순회
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            area = cv2.contourArea(cnt)
            # 원형도 계산 공식: (4 * pi * 면적) / (둘레^2) -> 완벽한 원일 경우 1.0
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            # 외곽선을 감싸는 최소 외접원 계산
            (x, y), radius = cv2.minEnclosingCircle(cnt)

            # 최소 반지름 및 원형도 조건 만족 시 정상 원으로 판별
            if radius > min_radius and circularity > 0.6:
                x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
                # 축소 좌표계 기준으로 원 중심, 반지름, 바운딩 박스 정보 보관
                detected_circles.append({
                    "center": (x, y),
                    "radius": radius,
                    "bbox": (x_box, y_box, w_box, h_box)
                })

        if timer:
            timer.stop("postprocess")  # [타이머 측정] 후처리 단계 종료
    else:
        # [최적화 3] N-Frame Skip 적용: 무거운 검출 연산 없이 이전 프레임의 결과 데이터를 그대로 재사용
        detected_circles = last_results

    # -------------------------------------------------------------------------
    # Phase 4. 결과 그리기 (Rendering Phase)
    # -------------------------------------------------------------------------
    if timer:
        timer.start("draw")  # [타이머 측정] 시각화 단계 시작

    # 원본 해상도 프레임을 복사하여 그리기 작업 진행
    output_frame = frame.copy()
    
    # 축소된 좌표를 원본 해상도 좌표로 되돌리기 위한 역비율 (0.5배 축소 시 2.0배 확대)
    inv_scale = 1.0 / scale_factor

    for item in detected_circles:
        # [최적화 4] 축소 좌표계의 결과를 원본 해상도 위치로 복원 (정수형 변환)
        cx = int(item["center"][0] * inv_scale)
        cy = int(item["center"][1] * inv_scale)
        r = int(item["radius"] * inv_scale)
        
        xb, yb, wb, hb = item["bbox"]
        xb = int(xb * inv_scale)
        yb = int(yb * inv_scale)
        wb = int(wb * inv_scale)
        hb = int(hb * inv_scale)

        # 1) 초록색 원 그리기 (BGR: (0, 255, 0), 두께 2)
        cv2.circle(output_frame, (cx, cy), r, (0, 255, 0), 2)
        
        # 2) 노란색 Bounding Box 그리기 (BGR: (0, 255, 255), 두께 2)
        cv2.rectangle(output_frame, (xb, yb), (xb + wb, yb + hb), (0, 255, 255), 2)
        
        # 3) 상단에 인식 라벨 표기
        cv2.putText(
            output_frame, 
            "Green Circle (V3 Opt)", 
            (xb, yb - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )

    if timer:
        timer.stop("draw")  # [타이머 측정] 시각화 단계 종료

    return output_frame, detected_circles