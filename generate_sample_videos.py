# generate_sample_videos.py
"""
[벤치마크용 테스트 영상 자동 생성 스크립트]
- 프로젝트 초기 설정 시 1회 실행하여 640x480, 1280x720, 1920x1080 해상도의
  테스트 영상(.mp4) 3종을 data/ 폴더에 자동 생성합니다.
- 생성된 영상 파일은 .gitignore에 등록되어 깃허브에는 커밋되지 않습니다.
"""

import os
import cv2
import numpy as np

def create_sample_videos():
    # 1. 영상 저장 폴더 생성
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # 2. 테스트할 3가지 해상도 규격 정의 (가로, 세로)
    resolutions = [
        (640, 480),    # SD 해상도
        (1280, 720),   # HD 해상도
        (1920, 1080)   # FHD 해상도
    ]

    # 3. 비디오 인코더 설정 (MP4V 코덱, 30 FPS, 10초 분량 = 300 프레임)
    fps = 30.0
    duration_sec = 10
    total_frames = int(fps * duration_sec)

    print("벤치마크용 샘플 영상 생성을 시작합니다.")

    for width, height in resolutions:
        file_path = os.path.join(data_dir, f"sample_{width}x{height}.mp4")
        
        # MP4 파일 저장을 위한 VideoWriter 객체 생성
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

        # 프레임 단위 영상 그리기 및 작성
        for i in range(total_frames):
            # 검은색 배경 프레임 생성 (Height, Width, Channel=3)
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # 좌우로 왕복 이동하는 움직이는 원 객체 계산 (영상 처리 연산 테스트용)
            center_x = int((width / 2) + np.sin(i / 10.0) * (width / 4))
            center_y = height // 2
            radius = min(width, height) // 8

            # 프레임에 도형 그리기 (초록색 원)
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), -1)

            # 프레임 정보 텍스트 삽입 (프레임 번호 및 해상도)
            text = f"Frame {i+1}/{total_frames} | {width}x{height}"
            cv2.putText(
                frame, 
                text, 
                (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8 if width <= 640 else 1.2, 
                (255, 255, 255), 
                2
            )

            # 생성된 프레임을 비디오 파일에 쓰기
            out.write(frame)

        # 파일 닫기 및 리소스 해제
        out.release()
        print(f"✅ 생성 완료: {file_path}")

    print("\n모든 샘플 영상이 성공적으로 생성되었습니다. (data/ 폴더)")


if __name__ == "__main__":
    create_sample_videos()