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
   * **전처리(Pre-processing):** 입력 프레임 노이즈 제거(`cv2.GaussianBlur`) 및 HSV 색상 공간 변환(`cv2.cvtColor`)[cite: 10].
   * **검출(Detection):** 설정된 HSV 범위(`lower_green`, `upper_green`) 기반 초록색 영역 마스킹(`cv2.inRange`)[cite: 10].
   * **후처리(Post-processing):** 잡영 제거 및 형상 보정을 위한 모폴로지 연산(Opening/Closing) 및 외곽선 추출(`cv2.findContours`)[cite: 10].
   * **결과 표시(Rendering):** 검출 객체 최소 외접원, Bounding Box 및 식별 라벨("Green Circle") 렌더링[cite: 10].

2. **버전별 실행 구조 (Architectural Versions)**
   * **V1 (순차 처리 - Single Thread):** 프레임 디코딩(I/O) → 연산 → 화면 출력을 단일 루프에서 순차 실행하여 베이스라인 성능 및 병목(Bottleneck) 구간 측정.
   * **V2 (멀티스레드 - Multi-threaded with Queue): Frame Producer(영상 읽기 스레드)와 Frame Consumer(영상 처리 스레드)를 분리하여 I/O 병목을 제거. 스레드 간 데이터 전달을 위해 큐(queue.Queue(maxsize=5))를 도입하고, 큐가 가득 찰 경우 Producer가 대기 및 재시도하는 동기화 메커니즘 적용.
   * **V3 (알고리즘 및 연산 최적화):**
    * Processing 해상도 Downsampling (`scale_factor=0.5`) 후 원본 좌표 역복원
    * 이전 검출 영역 기반 동적 관심 영역(ROI) 국한 연산 (`use_roi=True`)
    * 빠른 보간법 (`cv2.INTER_NEAREST`) 및 N-Frame Skip (`skip_interval=2`) 적용
    * 불필요한 스레드 오버헤드 방지를 위한 OpenCV 내부 스레드 제어 (`cv2.setNumThreads(1)`)
    * (`cv2.UMat` 기반 OpenCL 가속 옵션 보유, 기본 실행 시 `use_umat=False` 설정)

3. **시간 측정 및 벤치마크 (Measurement & Benchmarking)**
   * 이동평균 FPS: 최근 30프레임 간의 연산 시차를 기반으로 한 이동평균 FPS 계산.   
   * 단계별 소요 시간(ms): StepTimer를 활용해 전처리, 검출, 후처리, 그리기 각 단계의 소요 시간을 time.perf_counter() 기반으로 측정하여 병목 구간 탐지.   
   * 벤치마크 자동화: 70초 분량 샘플 영상 기준 해상도 3종(640x480, 1280x720, 1920x1080) × 버전 3종(V1, V2, V3) 조건별 3회 반복 실행.   
   * 결과 처리 및 직관적 비교: 3회 측정 평균/표준편차 및 V1 대비 성능 향상률(%) 산출 후 fps_comparison_table.csv 및 fps_benchmark_summary.csv 누적 저장,
   * 직전 vs 현재 측정 결과를 직관적으로 비교하는 Matplotlib 막대그래프(fps_benchmark_graph.png) 자동 생성.

### 2.2 비기능 요구사항 (Non-Functional Requirements)

* **안전한 종료 (Thread Safety):** `threading.Event` 객체를 통한 스레드 자원 해제 및 Deadlock 방지.
* **GUI 스레드 제약:** OpenCV GUI 메서드(`imshow`, `waitKey`)는 반드시 Main Thread에서 구동.
* **설정 관리:** 파라미터(커널 크기, Threshold, 큐 크기, 테스트 반복 횟수 등)는 `config.json`으로 일원화.
---

## 3. 시스템 아키텍처 및 폴더 구조

```text
team_project/
│
├── data/                       # Video samples repository
│   └── samples/
│       ├── tennis_sample_640x480.mp4
│       ├── tennis_sample_1280x720.mp4
│       └── tennis_sample_1920x1080.mp4
│
├── docs/                       # Team reports & checklists
│   ├── 실험기록.md
│   └── 최종보고서.md
│
├── results/                    # Benchmarking output repository
│   ├── fps_comparison_table.csv
│   │   # 개별 Benchmark Run 결과 누적 저장
│   │   # benchmark_id, timestamp, resolution, version, run,
│   │   # fps, status, error 기록
│   │   # 성공/실패/중단 기록을 모두 보존
│   │
│   ├── fps_benchmark_summary.csv
│   │   # 벤치마크별 통계 및 그래프용 데이터 누적 저장
│   │   # 평균 FPS(mean_fps), 표준편차(std_fps),
│   │   # 실행 횟수(run_count), V1 대비 성능 향상률(%) 기록
│   │
│   └── fps_benchmark_graph.png
│       # 현재 Benchmark와 직전 Benchmark의
│       # 평균 FPS 성능 비교 그래프
├── scripts
│    ├── check_env.py                 # 환경 설정 확인 함수   
│    ├── generate_sample_videos.py    # 테스트용 샘플비디오 3개 생성 (해상도별)
│    ├── tenis_sample_videos.py       # 벤치마크용 테니스공 객체 동영상 3개 생성 (해상도별)
├── src/                        # 핵심 소스 코드
│    ├── utils.py               # 파이프라인, 타이머, FPS 측정, 설정(config) 도우미 함수
│    ├── pipeline_v1.py         # 단일 스레드 기반 베이스라인 파이프라인
│    ├── pipeline_v2.py         # 생산자-큐-소비자 구조의 멀티스레드 파이프라인
│    ├── pipeline_v3.py         # 최적화 적용 파이프라인
│    └── benchmark.py           # 자동화 성능 벤치마크 실행기
├── .gitignore                  # 대용량 미디어 데이터, 캐시 및 환경 설정 제외 목록
├── check_env.py                # OpenCV-contrib 및 파이썬 버전 점검 스크립트
├── config.json                 # 중앙 매개변수 설정 파일
├── README.md                   # 프로젝트 문서 및 실행 가이드
└── requirements.txt            # 의존성 라이브러리 명세서
```

### Benchmark 결과 파일 설명
`results/` 폴더에는 벤치마크 실행 결과가 자동으로 저장되며, 이전 실행 결과를 삭제하지 않고 누적하여 성능 변화를 추적할 수 있도록 구성합니다.

- **`fps_comparison_table.csv`**
  - 각 파이프라인의 개별 실행(Run) 결과를 저장합니다.
  - 각 Benchmark 실행마다 고유한 `benchmark_id`와 `timestamp`를 기록합니다.
  - 해상도(`resolution`), 파이프라인 버전(`version`), 실행 번호(`run`), FPS를 저장합니다.
  - 정상 실행뿐만 아니라 `SUCCESS`, `ERROR`, `INTERRUPTED` 상태와 오류 원인도 함께 기록하여 실패 이력을 보존합니다.

- **`fps_benchmark_summary.csv`**
  - 개별 실행 데이터를 기반으로 계산한 벤치마크 통계 결과를 저장합니다.
  - 해상도 및 파이프라인 버전별 평균 FPS(`mean_fps`)와 표준편차(`std_fps`)를 기록합니다.
  - 실제 통계 계산에 사용된 실행 횟수(`run_count`)를 기록합니다.
  - V1을 Baseline으로 하여 V2와 V3의 FPS 성능 향상률(`improvement_vs_v1_pct`)을 계산합니다.
  - 새로운 Benchmark 실행 결과를 기존 데이터에 누적하여 과거 성능 변화를 추적할 수 있도록 합니다.

- **`fps_benchmark_graph.png`**
  - `fps_benchmark_summary.csv`의 데이터를 이용하여 생성합니다.
  - 현재 Benchmark의 평균 FPS와 직전 Benchmark의 평균 FPS를 비교합니다.
  - 해상도별 V1, V2, V3의 성능 변화를 시각적으로 확인할 수 있도록 합니다.