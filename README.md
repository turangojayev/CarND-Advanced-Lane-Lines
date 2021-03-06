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

[image10]: ./output_images/plotted_back.png


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

As we can see, it might be very difficult to come up with thresholds that would work for all the images. Rather than applying manually picked thresholds, I decided to let 
convolutional layers apply different gradient operators. Thus, I gathered data and made pixel wise classification to spot the lane lines. For creating input, I stored the frames from [project_video.mp4](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4)
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
 exactly 1, I also apply adaptive thresholding (`cv2.adaptiveThreshold(nn_output * 255, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0) // 255`) to the result:
 
 ![image8_1]
 
**Detect lane lines and fit polynomial**

Detection of lines is implemented in class [`Lines`](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/blob/9280aded388ed4769041a3e55baf43f3d0af257d/solution.py#L177).
Here my goal was to keep the procedure as simple as possible, without tuning thresholds on differences between fit lanes from frame to frame. At first frame of the video
I use sliding window approach to detect the lines and try to fit a second order polynomial to it. In all the following frames, I look at
 certain margin around the previously fit curve to spot the lines. Once I fit a new curve to the current image, I update the previous coefficients
 by using low-pass filter over all history of found coefficients. It means, I calculate exponentially weighted moving average with update weight of 0.2. The update mechanism is written in 
   [`_fit_and_update`](https://github.com/turangojayev/CarND-Advanced-Lane-Lines/blob/9280aded388ed4769041a3e55baf43f3d0af257d/solution.py#L201) function.
   
   One point to mention is the following code piece
 ```python
...
closest_point_difference = right_fit[-1] - left_fit[-1]
if closest_point_difference > 0:
    differences = right_fit - left_fit
    acceptable = differences > 0.7 * closest_point_difference
    start = numpy.argmax(acceptable)
    self._start += int(0.1 * (start - self._start))
    y = y[self._start:]
    if right_fit[2] - left_fit[2] > 350:
        self._left_coeffs += self._alpha * (left_coeffs - self._left_coeffs)
        self._right_coeffs += self._alpha * (right_coeffs - self._right_coeffs)
        self._left_coeffs_m += self._alpha * (left_coeffs_m - self._left_coeffs_m)
        self._right_coeffs_m += self._alpha * (right_coeffs_m - self._right_coeffs_m)
...
```

The goal of piece is to update the coefficients only if the difference of coefficients of the term with 0th power from right
and left lines is more than 350 pixels (and of course, right line should be at the right: `if closest_point_difference > 0:`).
Besides, I try to keep the further end of the lines moving back and forth, depending on how good the fit is. As a check,
 I use 0.7 portion of the distance between right and left lane lines at the bottom of the image.
   
  Here's an example of detected lane lines:
  
  ![image9_1]
  
  
**Radius of the curvature and position of the vehicle**

The curvature of the left and right lines are calculated as

```
left_curverad = (1 + (2 * left_coeffs_m[0] * 720 * ym_per_pix + left_coeffs_m[1]) ** 2) ** 1.5 / \
                    numpy.absolute(2 * left_coeffs_m[0])

right_curverad = (1 + (2 * right_coeffs_m[0] * 720 * ym_per_pix + right_coeffs_m[1]) ** 2) ** 1.5 / \
                     numpy.absolute(2 * right_coeffs_m[0])
```

where `left_coeffs_m` and `right_coeffs_m` are coefficients found by fitting a polynomial to the pixel coordinates converted to meters.
Assuming that camera is located at the center of the image, the distance to the lane center is calculated as
 
```
((left_fit[-1] + right_fit[-1]) / 2 - columns / 2) * xm_per_pix
```

where `left_fit[-1]` and `right_fit[-1]` are the x coordinates of the points located at the bottom of the lines fitted to the left and right lanes in pixel coordinates.
The values for `ym_per_pix` and `xm_per_pix` are `3/72` and  `3.7/700` respectively.

**Plot back onto the road**

Once we find lines and fill the polygon between them, we can unwarp the image and check the result of overall processing.

![image10]


---

## Pipeline

The steps listed above are all gathered together as pieces of class `Pipeline`. Here are the links to the resulting videos
 
 [Project video](https://youtu.be/B3e4ZyqtqTE)
 
 [Challenge video](https://youtu.be/CHDEO_GBlV4)
 
 [Harder challenge video](https://youtu.be/yISj5LafxJU)

---

### Discussion and further improvements

Since I used neural networks for segmentation, thresholding is quite robust against sunlight in harder challenge video. 
The middle frame at the top of the videos proves this point. However, fitted lines are not perfect and require further improvement. 
Moreover, when the fitted lines are not robust for several frames in a row, it would be better to drop the coefficients of 
the polynomials and start fitting from scratch. In current implementation I tried not to complicate this problem by introducing 
a lot of thresholds and manual checks. One can also use convolutional neural networks to directly predict the polynomial coefficients or
use recurrent neural networks to consider the context from the previous frames.

  
