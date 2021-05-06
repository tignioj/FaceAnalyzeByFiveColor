import matplotlib.pyplot as plt
import numpy

fig, ax = plt.subplots(1, 1)
image = numpy.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
im = ax.imshow(image)

while True:
    image = numpy.multiply(1.1, image)
    im.set_data(image)
    fig.canvas.draw_idle()
    plt.pause(1)
