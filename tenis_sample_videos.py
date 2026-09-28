import os
import cv2
from pathlib import Path

def generate_samples(input_filename: str = "origin_tenis_sample_1920x1080.mp4") -> None:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    input_path = DATA_DIR / input_filename

    if not input_path.exists():
        print(f"❌ [오류] 원본 파일이 없습니다: {input_path}")
        return

    # 1. 원본 영상을 읽어 메모리(리스트)에 사전 저장
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ [오류] 영상을 열 수 없습니다: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("❌ [오류] 영상에서 프레임을 읽어오지 못했습니다.")
        return

    print(f"🎬 원본 로드 완료: 총 {len(frames)}프레임 | FPS: {fps:.1f}")

    repeat_count = 10
    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    
    # 별도 DLL 설치가 필요 없는 기본 mp4v 코덱 사용
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for target_w, target_h in resolutions:
        output_filename = f"tennis_sample_{target_w}x{target_h}.mp4"
        output_path = DATA_DIR / output_filename

        print(f"🎥 [{output_filename}] 변환 진행 중...")

        out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_w, target_h))
        total_written_frames = 0
        
        try:
            # 2. 메모리 프레임을 10회 반복 작성 (일정한 FPS 유지)
            for r in range(repeat_count):
                for frame in frames:
                    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    out.write(resized)
                    total_written_frames += 1
        finally:
            out.release()  # 파일 닫기 및 저장 확정

        if output_path.exists() and output_path.stat().st_size > 0:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  └ ✅ 저장 완료! (총 {total_written_frames} 프레임 | 용량: {file_size_mb:.2f} MB)")

    print("\n🎉 모든 샘플 영상이 정상적으로 생성되었습니다!")

if __name__ == "__main__":
    generate_samples(input_filename="origin_tenis_sample_1920x1080.mp4")