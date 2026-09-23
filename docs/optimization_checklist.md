# 📋 실시간 비전 파이프라인 성능 최적화 체크리스트 (20/30 FPS 달성용)

> 본 문서는 교과 14/15 및 실시간 임베디드/PC 비전 프로젝트에서 **타겟 FPS(Jetson 20+ FPS / PC 30+ FPS)**를 달성하기 위한 단계별 가이드라인 및 핵심 체크리스트입니다.

---

## 1. 목표 성능 및 환경 기준 Check

| 항목 | 타겟 사양 / 목표 | 현재 달성 여부 | 비고 |
| :--- | :--- | :---: | :--- |
| **타겟 프레임 레이트** | Jetson Orin Super (20+ FPS) / PC (30+ FPS) | [ ] Pass / [ ] Fail | 1080p 해상도 기준 |
| **처리 제약 조건** | AI/DL 추론 모델 미사용 (Rule-based OpenCV) | [ ] Pass / [ ] Fail | Pure OpenCV 연산 적용 |
| **안전성 (Thread Safety)** | Thread Stop Event 및 Deadlock 방지 | [ ] Pass / [ ] Fail | `threading.Event()` 적용 |

---

## 2. 단계별 최적화 적용 체크리스트 (V1 → V2 → V3)

### Step 1. Structure Optimization (I/O & Architecture)
- [ ] **Producer-Consumer Threading**: 비디오 Frame Read(I/O)와 Vision Processing(CPU) 분리
- [ ] **Frame Drop Policy**: `queue.Queue(maxsize=1)` 적용 및 Overflow 시 이전 프레임 Discard 처리 (지연 누적 방지)
- [ ] **GUI Threading Rule**: `cv2.imshow` 및 `cv2.waitKey`는 반드시 Main Thread에서만 호출

### Step 2. Algorithmic Optimization (V3 Processing)
- [ ] **Downscaling**: 입력 프레임 $N\%$ 축소 처리 후 시각화 시 좌표 $1/N$ 복원
- [ ] **Fast Interpolation**: Downscaling 시 `cv2.INTER_NEAREST` 보간법 적용으로 연산 속도 확보
- [ ] **Dynamic ROI & Fail-Safe**:
  - [ ] 이전 프레임 Bounding Box 주변 Crop 후 연산
  - [ ] 객체 이탈 방지용 Margin 확보 (최소 50px 이상 권장)
  - [ ] 검출 실패(Miss) 시 자동 Full-Frame Reset 로직 적용
- [ ] **N-Frame Skip**: heavy 연산(HSV/Contour)은 $N$ 프레임마다 1회 수행, 중간 프레임은 좌표 재사용

---

## 3. 성능 개선 결과 요약 (Benchmark Verification)

| 버전 | 적용된 핵심 기술 | 1080p FPS | 병목 구간 (Worst Step) | 비고 |
| :--- | :--- | :---: | :--- | :--- |
| **V1 (Baseline)** | Single Thread 순차 처리 | `00.0` FPS | Blur / inRange | 베이스라인 |
| **V2 (Multi-Thread)**| Threading + Frame Drop Queue | `00.0` FPS | Processing Heavy | I/O 병목 제거 |
| **V3 (Optimized)** | Downscale + ROI + Frame Skip | `00.0` FPS | Draw / Rendering | **최종 목표 달성** |

---

## Quick Troubleshooting Guide
1. **화면이 끊기거나 프레임 지연(Lag)이 누적될 때**: Queue 크기를 `1`로 줄이고 `queue.Full` 발생 시 `get_nowait()`로 버리는 로직 확인
2. **공을 빠르게 움직일 때 추적을 놓칠 때**: ROI Margin을 확장(예: 20px → 50px)하거나 Fail-safe(전체 화면 재검출) 조건 확인
3. **스레드가 종료되지 않고 프로그램이 멈출 때**: 스레드 루프 내 `stop_event.is_set()` 체크 및 `join()` 호출 여부 확인