import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np

image = mping.imread('../images/image.jpg')
print('Image is', type(image), 'with dimensions', image.shape)

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)
line_image = np.copy(image)

# define color thresholds
red_threshold = 200
blue_threshold = 200
green_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# identify pixels below threshold
color_thresholds = (image[:, :, 0] < rgb_threshold[0])\
              | (image[:, :, 1] < rgb_threshold[1])\
              | (image[:, :, 2] < rgb_threshold[2])

color_select[color_thresholds] = [0, 0, 0]

# set triangular region of interest
'''
    For the image in image processing upper left is (0,0) and y values
    increase from there . 
'''

left_bottom = [100, 540]
right_bottom = [900, 540]
apex = [475, 320]


# fit lines using y = Ax+B
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# find region
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

region_thresholds = (YY > (XX*fit_left[0]+fit_left[1])) & \
                    (YY > (XX*fit_right[0]+fit_right[1])) & \
                    (YY < (XX*fit_bottom[0]+fit_bottom[1]))


line_image[~color_thresholds & region_thresholds] = [255, 0, 0]
plt.imshow(color_select)
plt.show()
plt.imshow(line_image)
plt.show()
