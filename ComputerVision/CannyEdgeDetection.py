import matplotlib.pyplot as plt
import matplotlib.image as mping
import cv2

image = mping.imread('../images/canny.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


kernel_size = 3
gray_blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

low_threshold = 40
high_threshold = 225
edges = cv2.Canny(gray_blur, low_threshold, high_threshold)
plt.imshow(edges, cmap='Greys_r')
plt.show()
