import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np


def lines(axis):
    '''return a list of lines for a give axis'''
    # equation(3)
    line1 = 1.5862 * axis + 20
    # equation(4)
    line2 = 0.3448 * axis + 76.2069
    # equation(5)
    # the slope of this equation is not correct Cr ≥ -4.5652 × Cb + 234.5652
    # it should be around -1
    line3 = -1.005 * axis + 234.5652
    # equation(6)
    line4 = -1.15 * axis + 301.75
    # equation(7)
    line5 = -2.2857 * axis + 432.85
    return [line1, line2, line3, line4, line5]


# The five bounding rules of Cr-Cb
def Rule_B(YCrCb_Frame, plot=False):
    '''this function implements the five bounding rules of Cr-Cb components
    --inputs:
    YCrCb_Frame: YCrCb components of an image
    plot: Bool type variable,if set to True draw the output of the algorithm
    --return a anumpy array of type bool like the following:
    [[False False False True]
    [False False False True]
    .
    .
    .
    [False False False True]]
    2d numpy array
    So in order to plot this matrix, we need to convert it to numbers like:
    255 for True values(white)
    0 for False(black)
    '''
    Y_Frame, Cr_Frame, Cb_Frame = [YCrCb_Frame[..., YCrCb] for YCrCb in range(3)]
    line1, line2, line3, line4, line5 = lines(Cb_Frame)
    YCrCb_Rule = np.logical_and.reduce([line1 - Cr_Frame >= 0,
                                        line2 - Cr_Frame <= 0,
                                        line3 - Cr_Frame <= 0,
                                        line4 - Cr_Frame >= 0,
                                        line5 - Cr_Frame >= 0])
    # Create a plot
    if plot:
        fig1 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.scatter(Cb_Frame, Cr_Frame, alpha=0.8, c='black', edgecolors='none', s=10, label="Cr")
        ax1.set_xlim([0, 255])
        ax1.set_ylim([0, 255])
        ax1.set_xlabel('Cb')
        ax1.set_ylabel('Cr')
        ax1.xaxis.set_label_coords(0.5, -0.025)
        # draw a line
        x_axis = np.linspace(0, 255, 100)
        line1, line2, line3, line4, line5 = lines(x_axis)
        ax1.plot(x_axis, line1, alpha=0.5, c='b', label="line1")
        ax1.plot(x_axis, line2, alpha=0.5, c='g', label="line2")
        ax1.plot(x_axis, line3, alpha=0.5, c='r', label="line3")
        ax1.plot(x_axis, line4, alpha=0.5, c='m', label="line4")
        ax1.plot(x_axis, line5, alpha=0.5, c='y', label="line5")
        plt.title('Bounding Rule for Cb-Cr space')
        plt.legend(loc=(1, 0.7))
        # plot the Y Cr and Cb components on a different figure
        fig2 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
        fig2.suptitle('Y-Cr-Cb components', fontsize=16)
        # Y components
        ax2 = fig2.add_subplot(3, 1, 1)
        ax2.set_title('Distribution of Y')
        ax2.title.set_position([0.9, 0.95])
        ax2.set_xlabel('pixel intensity')
        ax2.xaxis.set_label_coords(0.5, -0.025)
        ax2.set_ylabel('number of pixels')
        ax2.hist_rgb(Y_Frame.ravel(), bins=256, range=(0, 256), fc='b', ec='b')
        # Cb components
        ax3 = fig2.add_subplot(3, 1, 2)
        ax3.set_title('Distribution of Cb')
        ax3.title.set_position([0.9, 0.95])
        ax3.set_xlabel('pixel intensity')
        ax3.xaxis.set_label_coords(0.5, -0.025)
        ax3.set_ylabel('number of pixels')
        ax3.hist_rgb(Cb_Frame.ravel(), bins=256, range=(0, 256), fc='b', ec='b')
        # Cr components
        ax4 = fig2.add_subplot(3, 1, 3)
        ax4.set_title('Distribution of Cr')
        ax4.title.set_position([0.9, 0.95])
        ax4.set_xlabel('pixel intensity')
        ax4.xaxis.set_label_coords(0.5, -0.025)
        ax4.set_ylabel('number of pixels')
        ax4.hist_rgb(Cr_Frame.ravel(), bins=256, range=(0, 256), fc='b', ec='b')
        # show the effect of the bounding rules of Cr-Cb
        # black and white image after the mask
        img_bw = YCrCb_Rule.astype(np.uint8)
        img_bw *= 255
        fig3 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
        ax1 = fig3.add_subplot(1, 1, 1)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.set_title('CrCb-Mask')
        # plot as a Grayscale image
        ax1.imshow(img_bw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')

    return YCrCb_Rule


# original_img = cv2.imread("../../faces/7.jpeg")
# original_img = cv2.imread("../../result/chi/ming_tang.jpg")
original_img = cv2.imread("../../result/chi/ting.jpg")
# img = cv2.resize(original_img)
Rule_B(original_img, True)
plt.show()


def testcamera():
    videoCapture = cv2.VideoCapture(1)
    while videoCapture.isOpened():
        # image = cv2.imread("../..")
        flag, frame = videoCapture.read()
        if not flag:
            break
        # img = hsv(frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()
