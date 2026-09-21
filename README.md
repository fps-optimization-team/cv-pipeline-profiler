# P12 실시간 처리 속도 최적화 — 프로젝트 종합 문서화 및 실행 가이드라인

---

## 1. 프로젝트 개요 및 배경

본 프로젝트는 OpenCV 기반 영상 처리 파이프라인의 **단계별 처리 시간(ms)과 프레임 처리 속도(FPS)**를 정밀하게 측정하고, 단계적 최적화 구조(V1 → V2 → V3)를 적용함에 따른 성능 변화를 **정량적 데이터(표 및 그래프)**로 증명하는 엔지니어링 프로젝트입니다.

### 핵심 프로젝트 규칙
* **딥러닝 모델 사용 금지:** YOLO, CNN 등 AI 추론 모델을 일절 사용하지 않고, 오직 규칙 기반(Color Space Transformation, Thresholding, Morphology, Contour Detection) OpenCV 연산으로만 파이프라인을 구축합니다.
* **동일 알고리즘 유지:** 버전 간 성능 비교의 신뢰성을 위해 V1, V2, V3의 영상 처리 논리(Logical Pipeline)는 동일하게 유지합니다.
* **실시간성 보장:** Windows + Anaconda Python 환경에서 GUI 이벤트(`cv2.imshow`, `cv2.waitKey`) 처리 규칙을 준수하며, Frame Drop 및 큐(Queue) 관리 정책을 적용합니다.

---

## 2. 요구사항 분석 (Requirements Analysis)

### 2.1 기능 요구사항 (Functional Requirements)

1. **공통 영상 처리 파이프라인 (Processing Pipeline)**
   * **전처리(Pre-processing):** 입력 프레임 노이즈 제거(GaussianBlur) 및 HSV 색상 공간 변환.
   * **검출(Detection):** 특정 색상 영역(초록색 원) 마스킹(`cv2.inRange`).
   * **후처리(Post-processing):** 잡영 제거를 위한 모폴로지 연산(Open/Close) 및 외곽선 추출(`cv2.findContours`).
   * **결과 표시(Rendering):** 검출 객체 Bounding Box 렌더링 및 개수 표기.

2. **버전별 실행 구조 (Architectural Versions)**
   * **V1 (순차 처리 - Single Thread):** 프레임 디코딩(I/O) → 연산 → 화면 출력을 단일 메인 루프에서 순차 실행하여 베이스라인 성능 및 병목(Bottleneck) 구간 측정.
   * **V2 (멀티스레드 - Multi-threaded with Queue):** Frame Producer(영상 읽기 스레드)와 Frame Consumer(영상 처리 스레드)를 분리. 큐(`queue.Queue(maxsize=N)`)를 도입하고 오버플로우 시 최신 프레임 유지/오래된 프레임 폐기(Frame Drop) 정책 적용.
   * **V3 (OpenCV 가속 및 알고리즘 최적화):**
     * Processing 해상도 Downsampling 후 원본 좌표 복원
     * 관심 영역(ROI) 국한 연산
     * `cv2.UMat`(OpenCL GPU 가속) 적용
     * OpenCV 내부 쓰레드 수 조절(`cv2.setNumThreads`)

3. **시간 측정 및 벤치마크 (Measurement & Benchmarking)**
   * **이동평균 FPS:** 최근 N프레임간의 간격을 기반으로 한 이동평균 FPS 계산.
   * **단계별 소요 시간(ms):** `전처리`, `검출`, `후처리`, `그리기` 각 단계의 소요 시간을 `time.perf_counter()` 기반으로 독립 측정 및 병목 단계(Worst Step) 탐지.
   * **벤치마크 자동화:** 해상도 3종(640x480, 1280x720, 1920x1080) x 버전 3종(V1, V2, V3) 조건별 60초 이상 3회 측정 평균값/표준편차 산출, CSV 및 Matplotlib 그래프 자동 출력.

### 2.2 비기능 요구사항 (Non-Functional Requirements)

* **안전한 종료 (Thread Safety):** `threading.Event` 객체를 통한 스레드 자원 해제 및 Deadlock 방지.
* **GUI 스레드 제약:** OpenCV GUI 메서드(`imshow`, `waitKey`)는 반드시 Main Thread에서 구동.
* **설정 관리:** 파라미터(커널 크기, Threshold, 큐 크기, 테스트 반복 횟수 등)는 `config.json`으로 일원화.

---

## 3. 시스템 아키텍처 및 폴더 구조

```text
team_project/
├── .gitignore               # Large media data, cache, and env configs
├── check_env.py             # OpenCV-contrib and python version checker
├── config.json              # Central parameter configuration
├── README.md                # Project documentation & execution guide
├── requirements.txt         # Dependencies specification
├── data/                    # Video samples repository
│   ├── sample_640x480.mp4
│   ├── sample_1280x720.mp4
│   └── sample_1920x1080.mp4
├── docs/                    # Team reports & checklists
│   └── optimization_checklist.md
├── results/                 # Benchmarking output repository
│   ├── fps_comparison_table.csv
│   └── fps_benchmark_graph.png
└── src/                     # Core codebase
    ├── utils.py             # Pipeline, timer, FPS meter, config helpers
    ├── pipeline_v1.py       # Single-thread pipeline implementation
    ├── pipeline_v2.py       # Multi-threaded queued pipeline
    ├── pipeline_v3.py       # Optimized pipeline (UMat, ROI, Downscaling)
    └── benchmark.py         # Automated matrix benchmark runner