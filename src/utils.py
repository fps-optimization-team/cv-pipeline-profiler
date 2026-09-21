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
