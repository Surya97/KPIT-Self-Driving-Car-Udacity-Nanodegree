# **Finding Lane Lines on the Road** 


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images/solidYelloCurve.jpg
[image2]: ./test_images/solidYellowCurve_final.jpg


---

### Reflection

### 1. Pipeline.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied GaussianBlur to 
smoothen the image using a kernel size of 5. Later to detect the edges used "Canny Edge Detection" algorithm, where the lower threshold and upper thresholds are set to 10 and 150 respectively.

Now, we have the edges, we need the region of interest and based on the region of interest we mask the edges so that we have edges corresponding to only the region of interest.

After masking the region of interest, now want to detect lines in that particular region. So, used Hough transform method to detect lines. After Hough transformation, lines must me drawn on the image. Initially lines are drawn based on the co-ordinates given by the Hough transformation but there isn't a continuous line formation. So, used extrapolation technique to complete the lines.

##### Extrapolation technique
First calculate slope of the line if the x co-ordinates of the line vary. Then if the slope is negative it is a left lane line as y increases from top to bottom in an image, and similarly if slope is positive it is a right lane line. Collect the points of lines and for each lane, calculate average slope and b from all the points. Then, if it is a set of left lane points,
plot line from x1=(y_max-b)/m, y1=y_max to x2=x_max, y2=m*x_max + b. And if it is a set of right lane points, plot line from x1=x_min, y1=(x_min*m)+b to x2=(y_max-b)/m, y2=y_max


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![Initial image][image1] ![Final Image][image2]

After applying for images, implemented the same pipeline for videos also.


### 2. Shortcomings

I could see that whenver there is a curve in the video, the lines kind of behave in a wierd way. Sometimes they become horizontal.


### 3. Future improvements

A possible improvement would be to change the way the co-ordinates are calculated for extrapolation. And also need to figure out how to enable the lane lines complete as shown in the example output video.
