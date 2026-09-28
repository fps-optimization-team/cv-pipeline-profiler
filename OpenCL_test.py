#use_umat적용가능여부 테스트
import cv2

print("OpenCL available:", cv2.ocl.haveOpenCL())
print("OpenCL enabled:", cv2.ocl.useOpenCL())