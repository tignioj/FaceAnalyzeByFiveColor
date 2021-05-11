import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# mpl.use('Qt5Agg')
#
from core.const_var import OUTPUT_PATH


def cs(titile="0x123", img=None):
    cv2.imshow(titile, img)
    cv2.waitKey(0)
    cv2.destroyWindow(titile)


chi_sample = cv2.imread(OUTPUT_PATH + "/chi/ting.jpg")
black_sample = cv2.imread(OUTPUT_PATH + "/black/ting.jpg")
white_sample = cv2.imread(OUTPUT_PATH + "/white/ting.jpg")
yellow_sample = cv2.imread(OUTPUT_PATH + "/yellow/ting.jpg")
predict_white = cv2.imread(OUTPUT_PATH + "/predict_white/ting.jpg")
predict_white_trim = cv2.imread(OUTPUT_PATH + "/predict_white/ting_trim.jpg")
predict_dark = cv2.imread(OUTPUT_PATH + "/predict_dark/ting.jpg")
predict_dark_trim = cv2.imread(OUTPUT_PATH + "/predict_dark/ting.jpg")


# HSV
def sp(img):
    img = cv2.resize(img, (50,50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten())

def sprgb(img):
    img = cv2.resize(img, (50,50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten())

def p(sample, c):
    (x,y,z) = sp(sample)
    (r,g,b) = sprgb(sample)
    ax.scatter(x,y,z, alpha=.7, c=[(r[0] / 255., r[1] / 255., r[2] / 255.,) for r in zip(r,g,b)])

    #ax.set_xlim([-0.5, 1.5])
    #ax.set_ylim([-0.5, 1.5])
    #ax.set_zlim([-1.5, 1.5])



fig= plt.figure()
ax  = Axes3D(fig)
ax.set_xlabel('Cr')
ax.set_ylabel('Cb')
ax.set_zlabel('Y')
p(chi_sample, 'red')
# p(white_sample, 'lightblue')
# p(yellow_sample, 'yellow')
# p(black_sample, 'purple')
#p(predict_white, 'green')
#p(predict_dark, 'blue')

plt.show()

plt.show()
