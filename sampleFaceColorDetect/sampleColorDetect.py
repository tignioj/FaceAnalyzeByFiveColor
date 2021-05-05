from core.const_var import *
from core.FaceLandMark import faceDetection
import cv2

# blackImg = faceDetection.faceDetectByImg(cv2.imread(("../four_color_face/black.png")))[0]
blackImg = faceDetection.faceDetectByImg(cv2.imread(("../faces/7.jpeg")))[0]


# chiImg = faceDetection.faceDetectByImg(cv2.imread(("../four_color_face/white.png")))[0]
# whiteImg = faceDetection.faceDetectByImg(cv2.imread(("../four_color_face/black.png")))[0]
# yellowImg = faceDetection.faceDetectByImg(cv2.imread(("../four_color_face/yellow.png")))[0]

def cs(img, title=None):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print(blackImg.landMarkROI)
cv2.imwrite(OUTPUT_PATH + "\\" + "drawed.jpg", blackImg.drawImg)
# rd = blackImg.landMarkROI

# for (name, roi) in rd.items():
