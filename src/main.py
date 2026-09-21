# main.py — 영상 입력 주제 공통 뼈대
import argparse

import cv2

from utils import FPSMeter, open_source


def process(frame):
    """팀이 구현할 처리 함수. 지금은 그레이스케일 변환만 한다."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=웹캠, 또는 동영상 파일 경로")
    args = ap.parse_args()

    cap = open_source(args.source)
    fps = FPSMeter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] 영상이 끝났거나 프레임을 읽지 못했습니다.")
            break

        out = process(frame)
        cv2.putText(out, f"FPS {fps.tick():.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("result", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
