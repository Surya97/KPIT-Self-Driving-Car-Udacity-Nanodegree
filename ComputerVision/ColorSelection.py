import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np


image = mping.imread('../images/road.jpg')

plt.imshow(image)

print("Input image is:", type(image), "\nDimensions are:", image.shape)

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)


# Define color thresholds
red_threshold = 180
blue_threshold = 180
green_threshold = 180
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# identify pixels below threshold
thresholds = (image[:, :, 0] < red_threshold)\
              | (image[:, :, 1] < green_threshold)\
              | (image[:, :, 2] < blue_threshold)

print("Thresholds", thresholds)
color_select[thresholds] = [0, 0, 0]

plt.imshow(color_select)
plt.show()
