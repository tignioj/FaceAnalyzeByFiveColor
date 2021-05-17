from core.FaceLandMark import faceDetection
from service.ReportService import ReportService
import cv2,time

def _testROIDraw():
    img = cv2.imread("../../faces/white.jpg")
    f = faceDetection
    pt = time.time()
    faces = f.faceDetectByImg(img)
    et = time.time()
    print("detect time usage:", et - pt)
    reports = ReportService.generateReports(faces)

    et2 = time.time()
    print("report time usage:", et2 - et)


if __name__ == '__main__':
    _testROIDraw()