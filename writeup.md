## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)


[image1]: ./output_images/calibration.png 
[image2]: ./output_images/perspective_transformation.png
[image3]: ./output_images/perspective_transformation_example.png
[image3_1]: ./test_images/straight_lines2.jpg

[image4]: ./output_images/testimages.png
[image5]: ./output_images/warped_images.png

[image6]: ./output_images/binary.png
[image6_1]: ./output_images/binary_example.png

[image7]: ./output_images/nn_predicted.png
[image7_1]: ./output_images/nn_predicted_example.png

[image8]: ./output_images/adaptive_thresholded.png
[image8_1]: ./output_images/adaptive_thresholded_example.png

[image9]: ./output_images/lines_drawn.png
[image9_1]: ./output_images/lines_drawn_example.png


[video1]: ./project_video.mp4 "Video"

  

---


**Camera Calibration**

The code for camera calibration is contained 
[here](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/blob/d0411ee60ab75e915df3ba5fca301ab3f9e08bb8/solution.py#L45). 
The goal with camera calebration is to find parameters that correspond to rotation and translation vectors which translate the 
coordinates of a 3D point to a 2D coordinate system. First, for each chessboard image in [camera_cal/](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/tree/master/camera_cal) 
we try to find the corners using ```cv2.findChessboardCorners(gray, (columns, rows), None)```. Once the corner coordinates are collected,
we use them to find the needed matrix and vectors. Later, we use those to remove the distortion effect from the images (by calling `cv2.undistort(image, camera_matrix, distortion_coefs, None, None)`). Here's an example of
original image made by camera and the one with removed distortion:

![image1]

**Perspective transformation**

As the end goal of the project is to detect lane lines and measure the curvature, we need to get a bird's-eye view of the road.
This can be done by selecting 4 points on an image and defining where we want them to be mapped. We use `cv2.getPerspectiveTransform(src, dst)`
to find this transformation matrix needed for this process and `src` and `dst` are arrays with 4 image and target points respectively. To perform
the actual transformation we use `cv2.warpPerspective()`  [function](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/blob/d0411ee60ab75e915df3ba5fca301ab3f9e08bb8/solution.py#L67).
 For checking the transformation it is better to use an image where we have straight lines, so that we can test the correctness. For example, 
the warped copy of this image
 
 ![image3_1]

will look like this

![image2]

**Creating a thresholded binary image**

To create a binary image from the warped frames of the video, where lane lines are easily detectable, I manually picked up threshold values for YUV and HLS channels. Given an RGB
`image` the following code would get a binary image, where lane lines would have value 1 and the rest of the image would have 0 value.

```python
array = numpy.array

yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
yellow_color = cv2.inRange(yuv[:, :, 1], array([0]), array([115]))
yellow_color[yellow_color != 0] = 1

hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
white_color = cv2.inRange(hls, array([0, 200, 0]), array([255, 255, 255]))
white_color[white_color != 0] = 1

out = numpy.zeros_like(white_color)
out[(white_color != 0) | (yellow_color != 0)] = 1
```

This procedure applied to image

![image3]

would result in 

![image6_1]

However, manually picking up thresholds on different color channels or on results of applying gradient operators to the image
didn't look attractive to me. Below you can see the images on which it would be difficult to detect the lane lines. I gathered additional 
test images from [challenge_video.mp4](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/blob/master/challenge_video.mp4) and
[harder_challenge_video.mp4](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/blob/master/harder_challenge_video.mp4).

Original 

![image4]

Undistorted and warped

![image5]

Binary

![image6]

As we can see, it might be very difficult to come up with thresholds that would work for all the images. Thus, I decided to gather
data and make pixel wise classification to spot the lane lines. For creating input, I stored the frames from [project_video.mp4](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4)
 and as target, I stored the images with binary thresholding strategy applied to the input. I selected this particular video, because
  it was rather easy compared to the others and the defined thresholds worked relatively well on it. Only few images were manually edited to
  remove outliers. The created dataset is located at [line_detection_data](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/tree/master/line_detection_data) and there are 1260 instances.
  
 I trained a neural network for the segmentation and the code for training a model is in [`line_model.py`](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/blob/master/line_model.py)
Architecture I used is given below:
 
 |Layer (type) |Output Shape|Param #|
 --------------|:-----------:|------ |
 |input_1 (InputLayer)| (None, 720, 1280, 3)|0|
 lambda_1 (Lambda) (normalization) |(None, 720, 1280, 3)       |0|         
 conv2d_1 (Conv2D), kernel_size=3x3 |(None, 718, 1278, 32)     | 896  |    
 max_pooling2d_1 (MaxPooling2) kernel_size=3x3, strides=3x3|(None, 239, 426, 32)      | 0     |    
 conv2d_2 (Conv2D) ,kernel_size=3x3|(None, 237, 424, 64)      | 18496 |    
 max_pooling2d_2 (MaxPooling2) kernel_size=2x2, strides=2x2|(None, 118, 212, 64)       | 0     |
 dropout_1 (Dropout)          |(None, 118, 212, 64)              | 0     |    
 conv2d_3 (Conv2D), kernel_size=3x3|(None, 116, 210, 128)       | 73856 |    
 max_pooling2d_3 (MaxPooling2) kernel_size=2x2, strides=2x2|(None, 58, 105, 128)       | 0     |
 dropout_1 (Dropout)          |(None, 58, 105, 128)              | 0     |
 conv2d_4 (Conv2D), kernel_size=5x5|(None, 56, 103, 1)       | 1153|    
 upsampling_1 (Upsampling)              |(None, 720, 1280, 1)                | 0   |    
 
 All the hidden layers use rectified linear units as non-linearity and output uses sigmoid function for doing binary classification per pixel. I trained the model using adaptive momentum optimizer with the batch size of 8 images.
 To make the classifier robust to shady and bright regions in the images, I randomly lower or increase the brightness in randomly picked regions on them before feeding to the training process, by varying the luminosity(lightness) channel of HLS color channels.
 
 After the classifier trains, we can check how it performs on the previous images. Here's the single test image
 
 ![image7_1]
 
 and the difficult ones
 
 ![image7]
 
 As we can see, now the lane lines can be spotted way much better on the difficult images. However, since the sigmoid output might not be
 exactly 1, I also apply adaptive thresholding to the results:
 
 ![image8_1]
 
 and 
 
 ![image8]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
