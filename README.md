# P12 실시간 처리 속도 최적화
## 멀티스레드와 성능 측정을 활용한 OpenCV 영상 처리 파이프라인 최적화

---

## 1. 프로젝트 개요 및 배경

본 프로젝트는 OpenCV 기반 영상 처리 파이프라인의 **단계별 처리 시간(ms)과 프레임 처리 속도(FPS)**를 정밀하게 측정하고, 단계적 최적화 구조인 **V1 → V2 → V3**를 적용함에 따른 성능 변화를 **정량적인 데이터(표 및 그래프)**로 비교·분석하는 프로젝트입니다.

단순히 영상을 처리하는 프로그램을 구현하는 것에 그치지 않고,

- 단일 스레드 구조
- 멀티스레드 구조
- 알고리즘 최적화 구조

를 동일한 영상과 동일한 검출 알고리즘을 기준으로 비교하여 **어떤 최적화 방식이 실제 처리 속도 향상에 효과적인지 측정하는 것**을 목표로 합니다.

### 핵심 프로젝트 규칙

- **딥러닝 모델 사용 금지**
  - YOLO, CNN 등의 딥러닝 기반 객체 검출 모델을 사용하지 않습니다.
  - HSV 색상 변환, Thresholding, Morphology, Contour Detection 등 규칙 기반 OpenCV 영상 처리 기법만 사용합니다.

- **동일 알고리즘 유지**
  - V1, V2, V3의 성능 비교 신뢰성을 확보하기 위해 기본 객체 검출 알고리즘은 동일하게 유지합니다.

- **정량적 성능 측정**
  - FPS뿐만 아니라 전처리, 검출, 후처리, 그리기 단계의 처리 시간을 ms 단위로 측정합니다.

- **실시간 처리 구조 고려**
  - Queue, Thread, Frame Drop, ROI, Downsampling 등의 방법을 적용하여 실시간 영상 처리 구조를 실험합니다.

---

# 2. 요구사항 분석

## 2.1 기능 요구사항

### 2.1.1 공통 영상 처리 파이프라인

모든 버전에서 기본적으로 동일한 초록색 객체 검출 알고리즘을 사용합니다.

#### ① 전처리 (Pre-processing)

입력 프레임의 노이즈를 감소시키고 HSV 색상 공간으로 변환합니다.

주요 OpenCV 함수:

```python
cv2.GaussianBlur()
cv2.cvtColor()
```

처리 과정:

```text
Original Frame
      ↓
Gaussian Blur
      ↓
BGR → HSV
```

---

#### ② 검출 (Detection)

설정된 HSV 범위를 이용하여 초록색 영역을 마스크로 생성합니다.

주요 함수:

```python
cv2.inRange()
```

설정값:

```text
lower_green
upper_green
```

을 기준으로 초록색 영역을 분리합니다.

---

#### ③ 후처리 (Post-processing)

마스크 내부의 작은 노이즈를 제거하고 검출 영역의 형태를 보정합니다.

주요 처리:

- Morphological Opening
- Morphological Closing
- Contour Detection

주요 함수:

```python
cv2.morphologyEx()
cv2.findContours()
```

처리 과정:

```text
Mask
 ↓
Opening
 ↓
Closing
 ↓
findContours
```

---

#### ④ 결과 표시 (Rendering)

검출된 객체에 다음 정보를 표시합니다.

- 최소 외접원
- Bounding Box
- 객체 식별 Label
- FPS 정보

---

## 2.2 버전별 실행 구조

### V1 — Single Thread Baseline

V1은 최적화 전 성능을 측정하기 위한 **Baseline 파이프라인**입니다.

모든 작업을 하나의 Main Thread에서 순차적으로 수행합니다.

```text
Frame Read
    ↓
Pre-processing
    ↓
Detection
    ↓
Post-processing
    ↓
Drawing
    ↓
Display
```

특징:

- 단일 스레드 구조
- 구현이 단순함
- I/O와 영상 처리가 순차적으로 수행됨
- 이후 V2/V3 성능 비교의 기준점 역할

---

### V2 — Multi-threading + Queue

V2에서는 영상 프레임을 읽는 작업과 영상 처리 작업을 분리합니다.

```text
Producer Thread
     │
     │ Frame
     ↓
 queue.Queue
     │
     ↓
Consumer / Main Thread
     │
     ↓
Image Processing
```

#### Producer

별도의 Thread에서 다음 작업을 수행합니다.

```python
cap.read()
```

읽어온 프레임을 Queue에 전달합니다.

#### Consumer

Main Thread에서는 Queue에서 프레임을 가져와 실제 OpenCV 영상 처리를 수행합니다.

#### Queue

```python
queue.Queue(maxsize=5)
```

를 사용하여 Producer와 Consumer 사이에서 프레임을 전달합니다.

Queue가 가득 찬 경우 일정 시간 대기한 후에도 공간이 확보되지 않으면 해당 프레임을 건너뛰는 방식으로 과도한 대기 및 Queue 적체를 방지합니다.

#### Thread 종료

```python
threading.Event
```

를 사용하여 Producer와 Consumer가 안전하게 종료될 수 있도록 구성합니다.

---

### V3 — Algorithm Optimization

V3에서는 영상 처리 연산 자체의 계산량을 감소시키는 방향으로 최적화를 적용합니다.

주요 최적화 기능은 다음과 같습니다.

#### ① Downsampling

```python
scale_factor = 0.5
```

원본 프레임보다 작은 해상도에서 객체 검출을 수행하여 전체 연산량을 감소시킵니다.

검출이 완료되면 결과 좌표를 다시 원본 크기에 맞게 복원합니다.

---

#### ② ROI 기반 처리

```python
use_roi = True
```

이전에 검출된 객체 주변의 관심 영역(ROI)을 중심으로 연산하도록 하여 전체 화면을 계속 탐색하는 비용을 줄입니다.

---

#### ③ Frame Skip

```python
skip_interval = 2
```

일부 프레임에서는 이전 검출 결과를 재사용하여 모든 프레임에서 전체 검출 알고리즘을 실행하지 않도록 합니다.

---

#### ④ 빠른 보간법

```python
cv2.INTER_NEAREST
```

을 사용하여 Resize 연산 비용을 감소시킵니다.

---

#### ⑤ OpenCV 내부 Thread 제한

```python
cv2.setNumThreads(1)
```

을 통해 Python Thread와 OpenCV 내부 Thread가 동시에 과도하게 생성되는 상황을 방지하도록 실험합니다.

---

#### ⑥ OpenCL / UMat

OpenCL 가속 성능을 실험하기 위해

```python
cv2.UMat
```

사용 옵션을 구성했습니다.

기본 실행에서는:

```python
use_umat = False
```

로 설정하며, 필요 시 `True`로 변경하여 별도의 성능 실험을 진행할 수 있습니다.

OpenCL 사용이 항상 FPS를 향상시키는 것은 아니며, CPU ↔ GPU 데이터 전달 비용 등의 이유로 오히려 FPS가 감소할 수도 있으므로 실험 결과를 통해 효과를 판단합니다.

---

# 3. 시간 측정 및 벤치마크

## 3.1 StepTimer

`StepTimer`를 이용하여 영상 처리 단계별 실행 시간을 측정합니다.

측정 대상:

```text
preprocess
detection
postprocess
draw
```

후처리 과정은 추가적으로 다음 세부 연산도 측정할 수 있습니다.

```text
morph_open
morph_close
find_contours
```

시간 측정에는:

```python
time.perf_counter()
```

를 사용합니다.

결과는 ms 단위로 계산합니다.

이를 통해 단순히 전체 FPS만 확인하는 것이 아니라 **어떤 영상 처리 단계가 가장 큰 병목(Bottleneck)을 발생시키는지 확인할 수 있습니다.**

---

## 3.2 FPSMeter

최근 프레임의 시간 정보를 이용하여 이동평균 FPS를 계산합니다.

이를 통해 순간적인 FPS 변동보다 실제 영상 처리 성능을 안정적으로 확인할 수 있도록 구성합니다.

---

## 3.3 Benchmark 자동화

Benchmark는 다음 조건을 자동으로 반복 실행합니다.

### 해상도

```text
640 × 480
1280 × 720
1920 × 1080
```

### Pipeline

```text
V1
V2
V3
```

### 반복 횟수

각 조건별:

```text
3회
```

따라서 전체 테스트 수는:

```text
3개 해상도 × 3개 Pipeline × 3회
= 총 27회
```

입니다.

---

## 3.4 Benchmark 결과 처리

각 실행 결과에서 다음 정보를 저장합니다.

- Benchmark ID
- 실행 시간
- Resolution
- Pipeline Version
- Run Number
- FPS
- 실행 상태
- Error Message

실행 상태는 다음과 같이 구분합니다.

```text
SUCCESS
ERROR
INTERRUPTED
```

성공한 테스트를 기준으로 다음 통계도 계산합니다.

- 평균 FPS
- FPS 표준편차
- 실제 성공 Run 수
- V1 대비 성능 변화율(%)

---

# 4. 비기능 요구사항

## 4.1 안전한 Thread 종료

`threading.Event`를 사용하여 프로그램 종료 시 Producer Thread가 정상적으로 종료될 수 있도록 구성합니다.

이를 통해 Thread가 남아 프로그램이 종료되지 않는 문제나 Deadlock 발생 가능성을 줄입니다.

---

## 4.2 GUI Thread 제약

OpenCV GUI 관련 함수:

```python
cv2.imshow()
cv2.waitKey()
cv2.destroyAllWindows()
```

는 Main Thread에서 실행하도록 구성합니다.

---

## 4.3 설정 관리

영상 처리 및 Benchmark에 사용되는 주요 설정값은:

```text
config.json
```

에서 관리합니다.

예:

- HSV Threshold
- Morphology Kernel Size
- Queue Size
- Benchmark 반복 횟수
- V3 최적화 옵션

---

# 5. 시스템 아키텍처 및 폴더 구조

```text
cv-pipeline-profiler/
│
├── data/
│   └── samples/
│       ├── tennis_sample_640x480.mp4
│       ├── tennis_sample_1280x720.mp4
│       ├── tennis_sample_1920x1080.mp4
│       └── origin_tenis_sample_1920x1080.mp4
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
├── OpenCL_test.py
├── README.md
└── requirements.txt
```

---

# 6. 주요 파일 설명

## `src/utils.py`

프로젝트의 공통 기능을 담당합니다.

주요 기능:

- 영상 처리 공통 함수
- StepTimer
- FPSMeter
- `config.json` Loader
- OpenCV 영상 입출력 관련 Utility

---

## `src/pipeline_v1.py`

Single Thread 기반 Baseline Pipeline입니다.

```text
Read → Process → Draw → Display
```

순서로 하나의 Thread에서 모든 작업을 수행합니다.

---

## `src/pipeline_v2.py`

Producer / Consumer 구조를 적용한 멀티스레드 Pipeline입니다.

주요 기능:

- Frame Producer Thread
- Main Thread Consumer
- `queue.Queue`
- Frame Drop 처리
- `threading.Event` 기반 종료

---

## `src/pipeline_v3.py`

영상 처리 계산량을 감소시키기 위한 최적화 Pipeline입니다.

주요 기능:

- Downsampling
- ROI
- Frame Skip
- Interpolation 최적화
- OpenCV Thread 제어
- 선택적 OpenCL / UMat 사용

---

## `src/benchmark.py`

V1, V2, V3의 성능을 동일한 조건에서 자동으로 측정합니다.

주요 기능:

- 3개 해상도 자동 Benchmark
- V1 / V2 / V3 자동 실행
- 조건별 반복 실행
- FPS 기록
- 성공 / 실패 / 중단 기록
- 평균 / 표준편차 계산
- V1 대비 성능 변화 계산
- 결과 CSV 저장
- Benchmark Graph 자동 생성

---

# 7. Benchmark 결과 파일

`results/` 폴더에는 Benchmark 결과가 자동으로 저장됩니다.

기존 결과를 삭제하지 않고 누적하여 여러 실험의 결과를 비교할 수 있도록 구성합니다.

---

## 7.1 `fps_comparison_table.csv`

각 Pipeline의 **개별 Run 결과**를 저장합니다.

저장 정보:

```text
benchmark_id
timestamp
resolution
version
run
fps
status
error
```

정상 실행뿐만 아니라 실패 및 사용자가 중단한 테스트까지 기록하여 실험 이력을 보존합니다.

---

## 7.2 `fps_benchmark_summary.csv`

개별 실행 데이터를 이용해 계산한 통계 결과입니다.

저장 정보:

- 평균 FPS (`mean_fps`)
- FPS 표준편차 (`std_fps`)
- 실제 실행 횟수 (`run_count`)
- V1 대비 성능 변화율 (`improvement_vs_v1_pct`)

새로운 Benchmark를 실행하면 이전 결과에 추가되어 과거 실험과 성능 변화를 비교할 수 있습니다.

---

## 7.3 `fps_benchmark_graph.png`

`fps_benchmark_summary.csv`를 기반으로 Matplotlib을 이용하여 자동 생성하는 성능 비교 그래프입니다.

해상도별:

```text
V1
V2
V3
```

평균 FPS를 시각적으로 비교할 수 있습니다.

필요한 경우 현재 Benchmark와 이전 Benchmark의 성능 변화도 함께 비교할 수 있습니다.

---

# 8. 설치 및 실행 방법

## 8.1 GitHub Repository Clone

프로젝트를 저장할 위치에서 다음 명령어를 실행합니다.

```bash
git clone https://github.com/fps-optimization-team/cv-pipeline-profiler.git
```

프로젝트 폴더로 이동합니다.

```bash
cd cv-pipeline-profiler
```

---

## 8.2 Python 가상환경

본 프로젝트는 Anaconda 가상환경을 기준으로 개발했습니다.

현재 사용 중인 Python 버전은 제출 전에 다음 명령어로 확인합니다.

```bash
python --version
```

최종 제출 시 아래 항목에 실제 버전을 입력합니다.

```text
Python Version : [최종 확인 후 입력]
```

---

## 8.3 패키지 설치

프로젝트에 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

주요 패키지는 다음과 같습니다.

```text
opencv-contrib-python==4.10.0.84
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
```

---

## 8.4 전체 프로젝트 실행

프로젝트 Root 폴더에서:

```bash
python main.py
```

를 실행합니다.

전체 실행 순서는 다음과 같습니다.

```text
V1 시연
 ↓
V2 시연
 ↓
V3 시연
 ↓
전체 Benchmark
```

---

## 8.5 영상 및 설정 파일 직접 지정

```bash
python main.py --source data/samples/tennis_sample_1280x720.mp4 --config config.json
```

처럼 실행할 수 있습니다.

---

## 8.6 Benchmark만 실행

Pipeline 시연 없이 Benchmark만 실행하려면:

```bash
python src/benchmark.py
```

를 실행합니다.

Benchmark가 완료되면 결과는:

```text
results/
```

폴더에 자동으로 저장됩니다.

---

# 9. 팀 구성 및 역할

## 9.1 김태양

### 담당 분야

**멀티스레드 Queue 구조 및 측정 / Benchmark / 시각화**

---

### `src/utils.py` — 측정 파트

#### StepTimer

다음 영상 처리 단계의 실행 시간을 ms 단위로 측정합니다.

```text
전처리
검출
후처리
그리기
```

세부 병목 측정을 위해:

```text
Morph Open
Morph Close
findContours
```

시간도 추가로 측정할 수 있도록 구성했습니다.

---

#### FPSMeter

- 이동평균 FPS 계산
- Frame 처리 속도 측정

기능을 담당합니다.

---

#### Config Loader

`config.json`에 정의된 설정값을 프로그램에서 사용할 수 있도록 불러오는 기능을 구성했습니다.

---

### `src/pipeline_v2.py`

Producer / Consumer 기반 멀티스레드 구조 구현을 담당했습니다.

주요 작업:

- Frame Producer Thread 구현
- Producer / Consumer 구조 분리
- `queue.Queue`를 이용한 Frame 전달
- Queue 적체 시 Frame Drop 처리
- `threading.Event` 기반 Thread 종료 처리
- Main Thread 기반 OpenCV GUI 처리

---

### `src/benchmark.py`

자동 성능 측정 및 결과 저장 기능을 담당했습니다.

주요 작업:

- 해상도 3종 자동 반복 실행
- Pipeline V1 / V2 / V3 자동 실행
- 조건별 반복 Benchmark
- 성공 / 실패 / 중단 상태 기록
- FPS 평균 계산
- FPS 표준편차 계산
- V1 대비 성능 변화 계산
- Benchmark ID를 이용한 실험 결과 구분

---

### `results/`

Benchmark 결과의 자동 저장 및 시각화 기능을 구성했습니다.

주요 결과물:

```text
fps_comparison_table.csv
fps_benchmark_summary.csv
fps_benchmark_graph.png
```

---

## 9.2 전체 팀 역할

| 팀원 | 담당 분야 | 주요 작업 |
|---|---|---|
| 김태양 | 멀티스레드 Queue / 성능 측정 / Benchmark / 시각화 | `utils.py`, `pipeline_v2.py`, `benchmark.py`, `results/` |
| 팀원명 입력 | 실제 담당 분야 입력 | 실제 구현 내용 입력 |

※ 최종 제출 전 다른 팀원의 이름 및 실제 담당 내용을 추가합니다.

---

# 10. 형상관리

본 프로젝트는 **Git과 GitHub를 사용하여 형상관리 및 팀 협업**을 진행했습니다.

주요 협업 과정은 다음과 같습니다.

1. GitHub Repository를 공용 저장소로 사용
2. Repository Clone을 통한 개발 환경 구성
3. 팀원별 Branch에서 기능 개발
4. 작업 완료 후 Commit
5. 원격 Repository로 Push
6. Pull Request를 통한 코드 통합
7. `pull`, `fetch` 등을 이용한 최신 코드 동기화
8. `.gitignore`를 이용한 불필요한 파일 제외

예시:

```bash
git clone https://github.com/fps-optimization-team/cv-pipeline-profiler.git
```

새 Branch 생성:

```bash
git checkout -b feature/taeyang
```

작업 내용 저장:

```bash
git add .
git commit -m "Add multithread benchmark"
```

원격 Branch에 Push:

```bash
git push -u origin feature/taeyang
```

최신 Main Branch 동기화:

```bash
git pull origin main
```

필요한 경우:

```bash
git fetch origin
```

을 이용하여 원격 Repository의 최신 정보를 확인했습니다.

---

# 11. 개발 환경

| 항목 | 내용 |
|---|---|
| OS | Windows |
| 개발 도구 | Visual Studio Code |
| Python 환경 | Anaconda Virtual Environment |
| Python Version | 최종 확인 후 입력 |
| OpenCV | opencv-contrib-python 4.10.0.84 |
| NumPy | 1.26.4 |
| Pandas | 2.2.2 |
| Matplotlib | 3.8.4 |
| 설정 관리 | JSON (`config.json`) |
| 형상관리 | Git / GitHub |
| 사용 LLM | ChatGPT |
| 딥러닝 모델 | 사용하지 않음 |

---

# 12. LLM 활용 범위

프로젝트 진행 과정에서 LLM은 다음과 같은 보조 목적으로 활용했습니다.

- Python / OpenCV 문법 확인
- 코드 구조 검토
- 오류 메시지 분석
- 멀티스레드 구조 검토
- Queue 구조 이해 및 디버깅
- Benchmark 구조 검토
- 코드 리팩터링 보조
- README 및 프로젝트 문서화 보조

LLM이 제안한 코드는 그대로 사용하는 것이 아니라 실제 개발 환경에서 직접 실행하고, 결과를 확인하면서 프로젝트 구조에 맞도록 수정했습니다.

성능 측정 결과 역시 실제 로컬 개발환경에서 실행한 Benchmark 결과를 기준으로 사용합니다.

---

# 13. 딥러닝 사용 여부

본 프로젝트에서는 딥러닝 모델을 사용하지 않았습니다.

사용하지 않은 모델 예:

```text
YOLO
CNN
R-CNN
SSD
```

객체 검출에는 다음과 같은 전통적인 OpenCV 영상 처리 기법을 사용했습니다.

```text
Gaussian Blur
HSV Color Space
Thresholding
Morphology
Contour Detection
```

따라서 별도의 학습 데이터나 딥러닝 모델 Weight를 사용하지 않습니다.

---

# 14. 프로젝트 완성도

## 완성도

**[상 / 중 / 하 중 최종 선택]**

### 판단 근거

본 프로젝트에서는 다음 기능을 구현했습니다.

- V1 Baseline Pipeline 구현
- V2 Multi-thread Pipeline 구현
- V3 Optimization Pipeline 구현
- 단계별 처리시간 측정
- FPS 측정
- 3개 해상도 Benchmark
- 조건별 반복 측정
- 평균 FPS 계산
- 표준편차 계산
- 성공 / 실패 / 중단 기록
- CSV 결과 자동 저장
- Benchmark Graph 자동 생성
- Git / GitHub 기반 협업

최종적으로 실제 시연 영상과 Benchmark 결과를 확인한 뒤 완성도를 상·중·하 중 하나로 판단합니다.

---

# 15. 과제 난이도

## 난이도

**[상 / 중 / 하 중 최종 선택]**

### 판단 근거

본 프로젝트는 단순한 OpenCV 객체 검출뿐만 아니라 다음 개념을 동시에 다뤄야 했습니다.

- 영상 처리 알고리즘
- FPS 측정
- 단계별 성능 Profiling
- Python Thread
- Producer / Consumer 구조
- Queue
- Frame Drop
- Thread Synchronization
- ROI
- Downsampling
- Frame Skip
- 자동 Benchmark
- Pandas 기반 결과 처리
- Matplotlib 시각화
- Git / GitHub 협업

특히 최적화 기능을 적용하는 것뿐 아니라 **실제 FPS 향상 여부를 반복 실험을 통해 검증해야 한다는 점**에서 난이도가 높은 부분이 있었습니다.

---

# 16. 팀원 개인별 참여도

실제 작업량 및 역할 분담을 기준으로 작성합니다.

| 팀원 | 참여도 |
|---|---:|
| 김태양 | XX% |
| 팀원명 입력 | XX% |
| **합계** | **100%** |

참여도는 단순한 코드 작성량뿐만 아니라 다음 요소를 함께 고려하여 산정합니다.

- 기능 설계
- 코드 구현
- 테스트
- 디버깅
- 성능 실험
- Git 협업
- 결과 분석
- 문서화

※ 최종 제출 전 팀원 간 협의를 통해 실제 참여도 수치를 입력합니다.

---

# 17. 프로젝트 한계 및 개선 방향

## 17.1 Python Multi-threading의 한계

V2에서는 Producer와 Consumer를 분리하여 영상 입력과 영상 처리를 동시에 수행하도록 구성했습니다.

그러나 Python Thread 관리 비용 및 OpenCV 연산의 특성에 따라 멀티스레드를 적용했다고 해서 반드시 V1보다 FPS가 향상되는 것은 아닙니다.

따라서 본 프로젝트에서는:

> 멀티스레드 적용 = 무조건 성능 향상

으로 판단하지 않고 실제 Benchmark 결과를 통해 효과를 검증합니다.

### 개선 방향

향후 다음 방식과 비교할 수 있습니다.

- `multiprocessing`
- C++ OpenCV
- Queue Size별 비교
- Frame Drop 정책별 비교
- Producer / Consumer 개수 변화

---

## 17.2 Downsampling에 따른 정확도 저하

V3에서는 연산량을 감소시키기 위해 입력 프레임의 크기를 줄여 처리합니다.

이는 FPS 향상에는 도움이 될 수 있지만 작은 객체의 정보가 손실될 가능성이 있습니다.

### 개선 방향

- 객체 크기에 따른 동적 Scale Factor
- ROI 내부만 고해상도 처리
- FPS뿐 아니라 검출 정확도도 함께 측정

---

## 17.3 Frame Skip의 한계

Frame Skip을 적용하면 전체 영상 처리 연산 횟수를 감소시킬 수 있습니다.

하지만 객체가 빠르게 이동할 경우 이전 Frame의 검출 결과와 현재 객체 위치 사이에 오차가 발생할 수 있습니다.

### 개선 방향

객체의 이동 속도에 따라:

```text
skip_interval
```

을 자동으로 변경하는 Adaptive Frame Skip 방식을 적용할 수 있습니다.

---

## 17.4 ROI 방식의 한계

이전 객체 위치를 기반으로 다음 Frame의 탐색 범위를 제한하면 처리량을 줄일 수 있습니다.

하지만 객체가 빠르게 이동하여 ROI 밖으로 벗어나면 객체를 놓칠 가능성이 있습니다.

### 개선 방향

- 일정 Frame마다 전체 화면 재탐색
- ROI 영역 동적 확장
- 객체 이동 방향을 이용한 ROI 예측

등을 적용할 수 있습니다.

---

## 17.5 OpenCL / UMat의 환경 의존성

GPU 또는 OpenCL을 사용한다고 해서 항상 FPS가 증가하는 것은 아닙니다.

작은 연산에서는 CPU와 GPU 사이의 데이터 전달 비용이 실제 연산 시간보다 커질 수 있습니다.

따라서 기본 Benchmark에서는:

```python
use_umat = False
```

를 사용하며 OpenCL은 별도의 실험 조건으로 비교합니다.

성능이 감소한 결과 역시 실패 데이터로 제거하지 않고 **현재 환경에서는 해당 최적화 방법이 효과적이지 않았다는 실험 결과**로 활용합니다.

---

## 17.6 Hardware에 따른 FPS 차이

FPS는 다음과 같은 환경에 영향을 받을 수 있습니다.

- CPU 성능
- Memory
- OpenCV Version
- Python Version
- 영상 Codec
- OS
- 실행 중인 Background Process

따라서 서로 다른 PC에서 측정된 절대 FPS를 직접 비교하기보다는 **동일한 환경에서 V1 → V2 → V3의 상대적인 성능 차이를 비교하는 것**을 중요하게 봅니다.

---

## 17.7 테스트 영상의 한계

현재 Benchmark는 초록색 테니스공이 포함된 샘플 영상을 중심으로 측정합니다.

따라서 실제 모든 영상 처리 환경을 대표하지는 않습니다.

### 개선 방향

향후 다음과 같은 영상 데이터를 추가할 수 있습니다.

- 빠르게 움직이는 객체
- 여러 개의 객체
- 밝기가 변화하는 영상
- 복잡한 배경
- 카메라 흔들림이 있는 영상
- 실시간 Webcam 입력

---

# 18. 성능 측정 결과

Benchmark는 다음 세 가지 해상도를 대상으로 수행합니다.

```text
640 × 480
1280 × 720
1920 × 1080
```

각 해상도에서:

```text
V1
V2
V3
```

를 각각 반복 실행하여 평균 FPS와 표준편차를 계산합니다.

---

## 18.1 최종 성능 비교표

최종 Benchmark 실행 후 `fps_benchmark_summary.csv`의 결과를 기준으로 아래 표를 작성합니다.

| Resolution | V1 평균 FPS | V2 평균 FPS | V3 평균 FPS | V2 vs V1 | V3 vs V1 |
|---|---:|---:|---:|---:|---:|
| 640×480 | - | - | - | - | - |
| 1280×720 | - | - | - | - | - |
| 1920×1080 | - | - | - | - | - |

실제 측정 완료 후 `-` 부분을 최종 결과값으로 변경합니다.

---

## 18.2 결과 해석 기준

본 프로젝트에서는:

> V2 또는 V3의 FPS가 반드시 V1보다 높아야 성공한 실험이다.

라고 판단하지 않습니다.

예를 들어 특정 최적화 기능을 적용한 결과 FPS가 오히려 낮아졌다면 이는:

> 해당 최적화 방법이 현재 Hardware / OpenCV / 영상 조건에서는 성능 향상에 효과적이지 않았다.

는 것을 보여주는 의미 있는 실험 결과입니다.

따라서 성공한 결과뿐 아니라 **성능이 감소한 결과 역시 분석 자료로 활용합니다.**

---

# 19. Benchmark Graph

최종 Benchmark 실행 시 다음 파일이 자동 생성됩니다.

```text
results/fps_benchmark_graph.png
```

README에서는 다음과 같이 표시합니다.

![FPS Benchmark Result](results/fps_benchmark_graph.png)

그래프를 통해 해상도별:

- V1
- V2
- V3

의 평균 FPS 차이를 시각적으로 비교할 수 있습니다.

---

# 20. 시연 영상

프로젝트 최종 제출 시 실제 프로그램 실행 과정을 확인할 수 있는 시연 영상을 함께 제출합니다.

시연 영상에는 다음 내용을 포함합니다.

1. 프로젝트 실행
2. V1 Pipeline 동작
3. V2 Multi-thread Pipeline 동작
4. V3 Optimization Pipeline 동작
5. Benchmark 실행
6. 결과 CSV 생성
7. Benchmark Graph 생성

시연 영상 파일 또는 제출 링크는 최종 제출 방식에 맞춰 추가합니다.

---

# 21. 프로젝트 결과 요약

본 프로젝트에서는 동일한 OpenCV 객체 검출 알고리즘을 기준으로 세 가지 실행 구조를 구현하고 성능을 비교했습니다.

### V1

```text
Single Thread Baseline
```

가장 기본적인 순차 처리 구조입니다.

---

### V2

```text
Producer
    ↓
Queue
    ↓
Consumer
```

구조를 사용하여 영상 입력과 영상 처리를 분리했습니다.

---

### V3

다음과 같은 방법으로 영상 처리 연산량 자체를 감소시키는 최적화를 적용했습니다.

```text
Downsampling
ROI
Frame Skip
Interpolation Optimization
OpenCV Thread Control
OpenCL Option
```

---

본 프로젝트의 핵심은 단순히 가장 높은 FPS를 만드는 것이 아니라,

> **어떤 최적화 방법이 실제 실행 환경에서 효과가 있는지를 동일 조건의 반복 Benchmark를 통해 정량적으로 확인하는 것**

입니다.

이를 위해 FPS뿐만 아니라 단계별 처리시간, 평균, 표준편차 및 V1 대비 성능 변화율을 함께 측정하고 CSV와 Graph 형태로 저장하도록 구성했습니다.

결과적으로 영상 처리 Pipeline의 성능을 감각적으로 판단하는 것이 아니라 **실제 측정 데이터를 기반으로 병목 구간과 최적화 효과를 분석할 수 있는 구조를 구현**하는 것을 프로젝트의 최종 목표로 했습니다.