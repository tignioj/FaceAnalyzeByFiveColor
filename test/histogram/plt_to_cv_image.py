# https://www.codegrepper.com/code-examples/python/convert+matplotlib+figure+to+cv2+image
import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import cv2
import matplotlib.pyplot as plt

fig = plt.figure()
cap = cv2.VideoCapture(1)

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

line1, = plt.plot(x1, y1, 'ko-')  # so that we can update data later

for i in range(1000):
    # update data
    line1.set_ydata(np.cos(2 * np.pi * (x1 + i * 3.14 / 2)) * np.exp(-x1))



    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # display image with opencv or any operation you like
    cv2.imshow("plot", img)

    # display camera feed
    ret, frame = cap.read()
    cv2.imshow("cam", frame)

    k = cv2.waitKey(33) & 0xFF
    if k == 27:
        break
