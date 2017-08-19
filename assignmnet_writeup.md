## Advanced Lane Finding Write Up 

### By Nicholas Johnson --- Following the Template so I don't forget anything this time. 

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Undistorted-1.jpg "Undistorted-1"
[image2]: ./output_images/Undistorted-2.jpg "Undistorted-2"
[image3]: ./output_images/Undistorted-Road.jpg "Undistorted-Road"
[image4]: ./output_images/binary_threshold_output.jpg "Binary Output"
[image5]: ./output_images/corrections-images.jpg "RGB-HSL-HLV"
[image6]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image7]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image8]: ./output_images/example_output.jpg "Output"
[video1]: ./challenge_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

--

### Camera Calibration

####1. Step one of this assignment is to calibrate the camera being used on the vehicle. 

Because different lens distort images, and depending on where an object falls in the image, it can become distorted. We learned in the lectures about  ‘’’findChessboardCorners’’’ function in OpenCV that will look for the corners where a black and white corner meet. This doesn’t happen on the edges so you have to count the number of intersection corners one square n from all sides. Udacity really wants to make sure you don't get confused about this and tells you a like 4 times that the calibration photos provided in this lab are different than the ones used in lectures.
This meant that for this lab the number of corners was 9X6, but more important that that was the number of images in the calibation_cal folder. The photos are taken at different angles and organizations, this helps the calibration get more detail about how a variety of objects are distorted by the specific camera used for testing. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Once you have the corners of all 20  images found, well I had three images break on me, I can't figure out why they won’t show up, but more importantly is that they will not be calibrated for. Meaning if you try and undistort one that wasn’t properly identified you end up with a even more distorted image. 

![alt text][image1]

The code for this step is contained in the first code cell of the IPython notebook located in "https://github.com/NicholasJohnson9149/Udacity-P4-Advanced-Lane-Lines/blob/master/Advanced-Lane-Lines-Notebook%20.ipynb`). Once the points are found through distortion correction in the previous step I drew lines on them to see how well the function worked, ‘’’drawChessboardCorners’’’ did this without much added effort. I really like openCV.  Next I used another openCV function to generate a file that would help undistorted images. OpenCV ‘’’cv2.calibrateCamera’’ created a binary file with the calibration data from our previous results. 

Using this file I undistorted two calibration images, the first one become flatter and more square but somehow the side seemed to blur. So I tried another image and found that it became more square without blurring. I have been playing with getting all the calibration images to work better, but decided I want to work with my own camera and calibrate it for use on my own dash videos. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]


### Image Pipeline 

#### 1. An example of a distortion corrected image side by side. 

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images to remove any camera artifacts. 

![alt text][image3]

It’s difficult to distinguish any change between the two images, but the hood of car seems to be level across the lower frame in the undistorted image more than the original image. To undistorted the image I created a function called undistorted which accepted an image and returned the result by applying the cv2.undistort. 

The function `cv2.unsidtort()` revives an image and the matrix from the wrap files created earlier, from these values it  applies The functions in this section use a so-called pinhole camera model. In this model, a scene view is formed by projecting 3D points into the image plane using a perspective transformation. This allows the image to be  transformed  to account for lens distortion.


#### 2. Color and gradient corrections to extract lane markings from images. I identify where in my code I did color transforms, gradients and other methods to create a thresholded binary images.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 117 through 244 in `lane_line_detection.py`).  Here's an example of my output showing the final binary image after playing with the HSL color spectrum to pull out yellow and while lane lines form the original test image. 

![alt text][image4]

I started the processes by applying a Sobel function in the x direction. The Sobel operator is at the heart of the Canny edge detection algorithm. Applying the Sobel operator to an image is a way of taking the derivative of the image in the x or y direction and therefore away to find changes between colors. because lane lines normally move in y direction we want the edges with this function. From this information we want to use magnitude and gradient to pull important features from the image to find the lane lines. Gradient magnitude is at the heart of Canny edge detection, and is why Canny works well for picking up all edges. In the case of lane lines, we're interested only in edges of a particular orientation. So now we will explore the direction, or orientation, of the gradient. 


Now consider how you can use various aspects of your gradient measurements (x, y, magnitude, direction) to isolate lane-line pixels. Specifically, think about how you can use thresholds of the x and y gradients, the overall gradient magnitude, and the gradient direction to focus on pixels that are likely to be part of the lane lines. Once 


![alt text][image5]

The first row shows the RGB colors in gray scale broken out into individual filters. Because each image is made up of 3 layers and we want to filter out whit and yellow lane lines from the images in all lighting conditions. When you convert to gray scale the white and yellow colors appear almost identical.

To deal with loosing the yellow lane line in the blue channel, I converted the image to HSV, (hue, saturation, and value), and HLS space (hue, lightness, and saturation). These are some of the most commonly used color spaces in image analysis, digital cameras and art. The images in the second row appear bury and distorted in the H and S channels. The bury images are a result of color bleed across; converting the images to binary, allowed for the deltas to become more extreme and noticeable.

Now you can see that, the S channel is still doing a fairly robust job of picking up the lines under very different color and contrast conditions, while the other selections look messy. You could tweak the thresholds and get closer in the other channels, but the S channel is preferable because it is more robust to changing conditions.

Color and Gradient
You can clearly see which parts of the lane lines were detected by the gradient threshold and which parts were detected by the color threshold by stacking the channels and seeing the individual components. You can create a binary combination of these two images to map out where either the color or gradient thresholds were met.


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The next part of the pipeline is to warp the roadway as if the camera was located from.  To archive this I used `cv2.getPerspectiveTransform`
to  Calculates a perspective transform from four pairs of the corresponding points. 

```python
def unwarp(img, src, dst):
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv
```

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcore the source and destination points in the following manner:

```python
#Define source and destination points for transform
src = np.float32([(575,464),
                  (707,464), 
                  (258,684), 
                  (1049,684)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

exampleImg_unwarp, M, Minv = unwarp(exampleImg_undistort, src, dst)
```



This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 464      | 450, 0        | 
| 707, 464      | w-450, 0      |
| 258, 684      | 450, 960      |
| 1049, 684     | w-450, 960    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.


