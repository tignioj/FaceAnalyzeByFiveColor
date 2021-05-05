import cv2

image = cv2.imread('../faces/7.jpeg')
y = 0
x = 0
h = 1300
w = 800
crop = image[y:y + h, x:x + w]
cv2.imshow('Image', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()