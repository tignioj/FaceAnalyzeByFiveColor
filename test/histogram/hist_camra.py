import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def grab_frame(cap):
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# Initiate the two cameras
# cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# create two subplots
# ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# create two image plots
# im1 = ax1.imshow(grab_frame(cap1))
im2 = ax2.imshow(grab_frame(cap2))


def update(i):
    # im1.set_data(grab_frame(cap1))
    im2.set_data(grab_frame(cap2))


ani = FuncAnimation(plt.gcf(), update, interval=200)
plt.show()