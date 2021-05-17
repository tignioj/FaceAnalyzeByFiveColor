import cv2

from utils.ImageUtils import ImgUtils
import imutils

def getImg(path):
    img = cv2.imread(path)
    return imutils.resize(img, width=500)

if __name__ == '__main__':
    chi = getImg("../../four_color_face_sample/chi.png")
    chi = ImgUtils.putTextCN(chi, "赤（红）", color=(255, 0, 0), fontSize=95)
    yellow = getImg("../../four_color_face_sample/yellow.png")
    yellow = ImgUtils.putTextCN(yellow, "黄", color=(255, 0, 0), fontSize=95)
    black = getImg("../../four_color_face_sample/black.png")
    black = ImgUtils.putTextCN(black, "黑", color=(255, 0, 0), fontSize=95)
    white = getImg("../../four_color_face_sample/white.png")
    white = ImgUtils.putTextCN(white, "白", color=(255, 0, 0), fontSize=95)

    cv2.imshow("chi", chi)
    cv2.imshow("yellow", yellow)
    cv2.imshow("black", black)
    cv2.imshow("white", white)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
