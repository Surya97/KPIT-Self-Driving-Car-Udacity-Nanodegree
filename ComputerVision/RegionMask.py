import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np

image = mping.imread('../images/road.jpg')
print('Image is:', type(image), 'with dimensions:', image.shape)

xsize = image.shape[1]
ysize = image.shape[0]

region_select = np.copy(image)

# set triangular region of interest
'''
    For the image in image processing upper left is (0,0) and y values
    increase from there . 
'''

left_bottom = [0, 1400]
right_bottom = [2800, 1400]
apex = [1400, 600]


# fit line susing y = Ax+B
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# find region
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

region_thresholds = (YY > (XX*fit_left[0]+fit_left[1])) & \
                    (YY > (XX*fit_right[0]+fit_right[1])) & \
                    (YY < (XX*fit_bottom[0]+fit_bottom[1]))

# color the pixels inside the region to red
region_select[region_thresholds] = [255, 0, 0]

plt.imshow(region_select)
plt.show()