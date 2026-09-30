# P12 실시간 처리 속도 최적화 — 프로젝트 종합 문서화 및 실행 가이드라인

---

## 1. 프로젝트 개요 및 배경

본 프로젝트는 OpenCV 기반 영상 처리 파이프라인의 **단계별 처리 시간(ms)과 프레임 처리 속도(FPS)**를 정밀하게 측정하고, 단계적 최적화 구조(V1 → V2 → V3)를 적용함에 따른 성능 변화를 **정량적 데이터(표 및 그래프)**로 증명하는 엔지니어링 프로젝트입니다.

### 핵심 프로젝트 규칙

- **딥러닝 모델 사용 금지:** YOLO, CNN 등 AI 추론 모델을 사용하지 않고, 규칙 기반(Color Space Transformation, Thresholding, Morphology, Contour Detection) OpenCV 연산으로만 파이프라인을 구축합니다.
- **동일 알고리즘 유지:** 버전 간 성능 비교의 신뢰성을 위해 V1, V2, V3의 영상 처리 논리(Logical Pipeline)는 동일하게 유지합니다.
- **실시간성 보장:** Windows + Anaconda Python 환경에서 GUI 이벤트(`cv2.imshow`, `cv2.waitKey`) 처리 규칙을 준수하며, Frame Drop 및 Queue 관리 정책을 적용합니다.

### 팀 구성 및 역할

| 팀원 | 담당 역할 |
|---|---|
| 김태양 | 멀티스레드 Queue 구조, FPS 및 단계별 시간 측정, Benchmark 자동화 및 결과 시각화 |
| 팀원명 | 담당 역할 입력 |

김태양 담당 주요 파일은 `src/utils.py`, `src/pipeline_v2.py`, `src/benchmark.py` 및 `results/`입니다.

### 형상관리 사용 여부

**Git / GitHub 사용**

팀원별 Branch에서 기능을 개발한 후 Push 및 Pull Request를 통해 코드를 통합했습니다.

### 프로젝트 평가

- **완성도:** [상 / 중 / 하]
- **과제 난이도:** [상 / 중 / 하]

### 개인별 참여도

| 팀원 | 참여도 |
|---|---:|
| 김태양 | XX% |
| 팀원명 | XX% |
| **합계** | **100%** |

### 한계와 개선 방향

- Python 멀티스레드 적용이 항상 FPS 향상으로 이어지지는 않으므로 실제 Benchmark를 통한 검증이 필요합니다.
- Downsampling과 Frame Skip은 처리 속도를 높일 수 있지만 객체 검출 정확도가 낮아질 가능성이 있습니다.
- 향후 `multiprocessing`, 동적 ROI, 다양한 영상 및 실제 Webcam 환경을 이용한 추가 성능 비교가 가능합니다.

---

## 2. 요구사항 분석 (Requirements Analysis)

### 2.1 기능 요구사항 (Functional Requirements)

1. **공통 영상 처리 파이프라인 (Processing Pipeline)**
   - **전처리(Pre-processing):** 입력 프레임 노이즈 제거(`cv2.GaussianBlur`) 및 HSV 색상 공간 변환(`cv2.cvtColor`).
   - **검출(Detection):** 설정된 HSV 범위(`lower_green`, `upper_green`) 기반 초록색 영역 마스킹(`cv2.inRange`).
   - **후처리(Post-processing):** 잡영 제거 및 형상 보정을 위한 모폴로지 연산(Opening/Closing) 및 외곽선 추출(`cv2.findContours`).
   - **결과 표시(Rendering):** 검출 객체 최소 외접원, Bounding Box 및 식별 라벨("Green Circle") 렌더링.

2. **버전별 실행 구조 (Architectural Versions)**
   - **V1 (순차 처리 - Single Thread):** 프레임 디코딩(I/O) → 연산 → 화면 출력을 단일 루프에서 순차 실행하여 베이스라인 성능 및 병목(Bottleneck) 구간 측정.
   - **V2 (멀티스레드 - Multi-threaded with Queue):** Frame Producer와 Frame Consumer를 분리하여 영상 입력과 처리 작업을 병렬화합니다. `queue.Queue(maxsize=5)`를 이용해 프레임을 전달하고 Queue 적체 시 Frame Drop 정책을 적용합니다.
   - **V3 (알고리즘 및 연산 최적화):**
     - Processing 해상도 Downsampling (`scale_factor=0.5`) 후 원본 좌표 역복원
     - 이전 검출 영역 기반 동적 관심 영역 ROI (`use_roi=True`)
     - 빠른 보간법 (`cv2.INTER_NEAREST`) 및 N-Frame Skip (`skip_interval=2`) 적용
     - OpenCV 내부 스레드 제어 (`cv2.setNumThreads(1)`)
     - `cv2.UMat` 기반 OpenCL 가속 옵션 보유, 기본 실행 시 `use_umat=False`

3. **시간 측정 및 벤치마크 (Measurement & Benchmarking)**
   - 이동평균 FPS: 최근 프레임의 연산 시차를 기반으로 이동평균 FPS 계산.
   - 단계별 소요 시간(ms): `StepTimer`를 활용하여 전처리, 검출, 후처리, 그리기 단계의 시간을 `time.perf_counter()` 기반으로 측정.
   - 벤치마크 자동화: 해상도 3종(640x480, 1280x720, 1920x1080) × 버전 3종(V1, V2, V3)을 조건별 3회 반복 실행.
   - 결과 처리: 평균/표준편차 및 V1 대비 성능 향상률(%) 산출 후 `fps_comparison_table.csv`, `fps_benchmark_summary.csv` 저장.
   - Matplotlib을 이용하여 `fps_benchmark_graph.png` 자동 생성.

### 2.2 비기능 요구사항 (Non-Functional Requirements)

- **안전한 종료 (Thread Safety):** `threading.Event` 객체를 통한 스레드 자원 해제 및 Deadlock 방지.
- **GUI 스레드 제약:** OpenCV GUI 메서드(`imshow`, `waitKey`)는 Main Thread에서 구동.
- **설정 관리:** 파라미터(커널 크기, Threshold, Queue 크기, 테스트 반복 횟수 등)는 `config.json`으로 관리.

---

## 3. 시스템 아키텍처 및 폴더 구조

```text
cv-pipeline-profiler/
│
├── data/
│   └── samples/
│       ├── tennis_sample_640x480.mp4
│       ├── tennis_sample_1280x720.mp4
│       └── tennis_sample_1920x1080.mp4
│
├── docs/
│   ├── 실험기록.md
│   └── 최종보고서.md
│
├── results/
│   ├── fps_comparison_table.csv
│   ├── fps_benchmark_summary.csv
│   └── fps_benchmark_graph.png
│
├── scripts/
│   ├── check_env.py
│   ├── generate_sample_videos.py
│   └── tenis_sample_videos.py
│
├── src/
│   ├── utils.py
│   ├── pipeline_v1.py
│   ├── pipeline_v2.py
│   ├── pipeline_v3.py
│   └── benchmark.py
│
├── .gitignore
├── config.json
├── main.py
├── README.md
└── requirements.txt
```

### Benchmark 결과 파일 설명

`results/` 폴더에는 벤치마크 실행 결과가 자동으로 저장되며, 이전 실행 결과를 삭제하지 않고 누적하여 성능 변화를 추적할 수 있도록 구성합니다.

- **`fps_comparison_table.csv`**
  - 각 파이프라인의 개별 실행(Run) 결과를 저장합니다.
  - 각 Benchmark 실행마다 고유한 `benchmark_id`와 `timestamp`를 기록합니다.
  - 해상도(`resolution`), 파이프라인 버전(`version`), 실행 번호(`run`), FPS를 저장합니다.
  - `SUCCESS`, `ERROR`, `INTERRUPTED` 상태와 오류 원인도 함께 기록합니다.

- **`fps_benchmark_summary.csv`**
  - 개별 실행 데이터를 기반으로 계산한 벤치마크 통계 결과를 저장합니다.
  - 해상도 및 파이프라인 버전별 평균 FPS(`mean_fps`)와 표준편차(`std_fps`)를 기록합니다.
  - 실제 통계 계산에 사용된 실행 횟수(`run_count`)를 기록합니다.
  - V1을 Baseline으로 하여 V2와 V3의 FPS 성능 향상률(`improvement_vs_v1_pct`)을 계산합니다.

- **`fps_benchmark_graph.png`**
  - `fps_benchmark_summary.csv`의 데이터를 이용하여 생성합니다.
  - 해상도별 V1, V2, V3의 평균 FPS 성능을 그래프로 비교합니다.

---

## 4. 개발 환경 및 실행 방법

### 개발 환경

- OS: Windows
- 개발 도구: Visual Studio Code
- Python 환경: Anaconda
- Python Version: [최종 확인 후 입력]
- OpenCV: `opencv-contrib-python==4.10.0.84`
- NumPy: `1.26.4`
- Pandas: `2.2.2`
- Matplotlib: `3.8.4`
- 형상관리: Git / GitHub
- 사용 LLM: ChatGPT
- 딥러닝 모델: 사용하지 않음

### 설치

```bash
pip install -r requirements.txt
```

### 전체 실행

```bash
python main.py
```

### 특정 영상 실행

```bash
python main.py --source data/samples/tennis_sample_1280x720.mp4 --config config.json
```

### Benchmark만 실행

```bash
python src/benchmark.py
```

---

## 5. 성능 측정 결과

최종 Benchmark 완료 후 `fps_benchmark_summary.csv`의 결과를 아래 표에 입력합니다.

| Resolution | V1 평균 FPS | V2 평균 FPS | V3 평균 FPS | V2 vs V1 | V3 vs V1 |
|---|---:|---:|---:|---:|---:|
| 640×480 | - | - | - | - | - |
| 1280×720 | - | - | - | - | - |
| 1920×1080 | - | - | - | - | - |

성능이 감소한 결과 역시 실패로 제외하지 않고, 해당 최적화 기법이 현재 실행 환경에서는 효과적이지 않았다는 실험 결과로 활용합니다.

### Benchmark Graph

![FPS Benchmark Result](results/fps_benchmark_graph.png)

---

## 6. 프로젝트 결과 요약

본 프로젝트에서는 동일한 OpenCV 객체 검출 알고리즘을 기준으로 세 가지 구조의 성능을 비교했습니다.

- **V1:** Single Thread 기반 Baseline
- **V2:** Producer / Consumer + Queue 기반 Multi-threading
- **V3:** Downsampling, ROI, Frame Skip 등의 연산 최적화

단순히 FPS가 높은 구조를 만드는 것이 아니라, 동일 조건에서 반복 Benchmark를 수행하여 **어떤 최적화 방법이 실제 실행 환경에서 효과가 있는지를 정량적으로 검증하는 것**을 목표로 합니다.