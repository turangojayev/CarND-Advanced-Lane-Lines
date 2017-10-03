import glob
import os
from functools import partial

import cv2
import keras
import numpy
import tensorflow as tf
from keras import Input
from keras.engine import Model
from keras.layers import *
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

COLUMNS = 9
ROWS = 6
rows = 720
columns = 1280
# src = numpy.float32([[0, rows], [568, 453], [710, 453], [columns, rows]])
# src = numpy.float32([[0, rows], [568, 453], [715, 453], [columns, rows]])
# src = numpy.float32([[0, rows], [568, 453], [715, 453], [columns, rows]])
# src = numpy.float32([[0, rows], [568, 453], [720, 453], [columns, rows]])
# dst = numpy.float32([[0, rows], [0, 0], [columns, 0], [columns, rows]])
#
src = numpy.float32([[0, 700],
                     [515, 472],
                     [764, 472.],
                     [1280, 700.]])

dst = numpy.float32([[100, 710],
                     [100, 10],
                     [1180, 10],
                     [1180, 710]])

perspective_tr_matrix = cv2.getPerspectiveTransform(src, dst)
inverse_perspective_tr_matrix = cv2.getPerspectiveTransform(dst, src)
RGB = ['Red', 'Green', 'Blue']
array = numpy.array
undistort = cv2.undistort


def makeGaussian(size, sigma=2):
    x = numpy.arange(0, size, 1, float)
    y = x[:, numpy.newaxis]
    x0 = y0 = size // 2
    gaussian = numpy.exp(-4 * numpy.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
    gaussian /= numpy.sum(gaussian)
    return gaussian


gaussian = makeGaussian(21, 50)


def contrast_normalization(img):
    # img1 = img.reshape(*img.shape[:2])
    # img1 = img#.reshape(*img.shape[:2])
    subtractive_normalized = img - cv2.filter2D(img, cv2.CV_64F, gaussian)
    image = subtractive_normalized ** 2
    height, width = image.shape[:2]

    output = numpy.sqrt(cv2.filter2D(image, cv2.CV_64F, gaussian))
    mean_sigma = numpy.mean(output)
    indices = output < mean_sigma
    output[indices] = mean_sigma

    output = subtractive_normalized / output
    return output


def get_calibration_results(rows=ROWS, columns=COLUMNS):
    image_paths = glob.glob('camera_cal/*.jpg')
    per_image_object_points = numpy.zeros((columns * rows, 3), numpy.float32)
    per_image_object_points[:, :2] = numpy.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    object_points = []
    image_points = []
    for path in image_paths:
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        return_value, corners = cv2.findChessboardCorners(gray, (columns, rows), None)
        if return_value is True:
            object_points.append(per_image_object_points)
            image_points.append(corners)

    # test_image = cv2.imread('test_images/test5.jpg')
    test_image = cv2.cvtColor(cv2.imread(image_paths[2]), cv2.COLOR_BGR2GRAY)

    return_value, camera_matrix, distortion_coefs, rotation_vectors, translation_vectors = \
        cv2.calibrateCamera(object_points, image_points, test_image.shape[:2], None, None)
    # cv2.calibrateCamera(object_points, image_points, test_image.shape[::-1][:2], None, None)
    # TODO:or the one below?
    #     cv2.calibrateCamera(object_points, image_points, test_image.shape[:2], None, None)

    return camera_matrix, distortion_coefs


def process_and_save_video(input, output, pipeline):
    clip = VideoFileClip(input)
    white_clip = clip.fl_image(pipeline)
    white_clip.write_videofile(output, audio=False)


def plot(images, columns=3, channel=None, cmap=None, title=None, directory=None):
    rows = len(images) / columns
    subplot = partial(plt.subplot, rows, columns)
    plt.figure(figsize=(15, 10))
    for i, image in enumerate(images, 1):
        subplot(i)
        # plt.imshow(image[:,:,0], cmap='gray' if len(image.shape) == 2 else cmap)
        plt.imshow(image if channel is None else image[:, :, channel], cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    if title is not None:
        if directory:
            title = os.path.join(directory, title)
        plt.savefig(title)
    plt.show()


def convert_if_needed(image):
    if image.dtype == numpy.float32:
        image = numpy.uint8(image * 255)
    return image


def plot_for_line(images,
                  cmap=None,
                  title=None,
                  line_loc_as_float=0.8,
                  directory=None,
                  colors=RGB):
    rows = len(images)
    if len(images[0].shape) == 2:
        columns = len(images[0].shape)
    else:
        columns = len(images[0].shape) + 1

    subplot = partial(plt.subplot, rows, columns)
    plt.figure(figsize=(20, 10))

    for image, i in zip(images, range(1, columns * rows, columns)):
        image = convert_if_needed(image)
        subplot(i)
        plt.xticks([])
        plt.yticks([])
        line_number = int(line_loc_as_float * image.shape[0])
        plt.axhline(line_number, 0, color='red')
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else cmap)
        line = image[line_number, :] if columns == 2 else image[line_number, :, :]

        def plot_subplot(idx):
            subplot(idx)
            plt.xticks([])
            if columns == 2:
                plt.plot(range(line.shape[0]), line)
            else:
                plt.plot(range(line.shape[0]), line[:, idx - i - 1])
                plt.title(colors[idx - i - 1])

        for channel in range(columns - 1):
            plot_subplot(i + 1 + channel)

    if title is not None:
        if directory:
            title = os.path.join(directory, title)
        plt.savefig(title)
    plt.show()


def gradient(image, low, high):
    x_gradient = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=15)
    absolute_x_gradient = numpy.absolute(x_gradient)
    scaled_abs_x_grad = numpy.uint8(255 * absolute_x_gradient / numpy.max(absolute_x_gradient))
    binary = cv2.inRange(scaled_abs_x_grad, low, high)
    return binary


def build_model(input_shape=None):
    img_input = Input(shape=input_shape)
    x = Lambda(lambda x: (x - 127.) / 128)(img_input)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((3, 3), (3, 3))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)
    x = Dropout(0.7)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)
    x = Dropout(0.7)(x)
    x = Conv2D(1, (5, 5), activation='sigmoid')(x)
    x = BilinearUpSampling2D(target_size=(720, 1280))(x)
    model = Model(img_input, x)
    # model.compile(optimizer=optimizers.adam(0.001), loss='binary_crossentropy')
    return model


def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None,
                           data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(numpy.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(numpy.array([height_factor, width_factor]).astype('int32'))
        X = K.permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = K.permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(numpy.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(numpy.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)


class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1],
                                          data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1],
                                          data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Pipeline(object):
    def __init__(self):
        self.camera_matrix, self.distortion_coefs = get_calibration_results()
        # self._binary_model = build_model((720, 1280, 3))
        self._binary_model = keras.models.load_model('model.h5',
                                                     custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D})

    def __call__(self, image, **kwargs):
        undistorted = undistort(image, self.camera_matrix, self.distortion_coefs, None, None)
        warped = perspective_transform(undistorted, perspective_tr_matrix)
        warped_binary = self._get_thresholded(warped)

        # return self._cnn(warped_binary)
        if hasattr(self, 'left_fit'):
            lines_drawn = self._continue_from_last(warped_binary)
        else:
            lines_drawn = self._detect_lines(warped_binary)
        unwarped_lines = perspective_transform(lines_drawn, inverse_perspective_tr_matrix)

        ym_per_pix = 30 / 720  # meters per pixel in y dimension

        left_curverad = (1 + (2 * self.left_fit_cr[0] * 720 * ym_per_pix + self.left_fit_cr[1]) ** 2) ** 1.5 / \
                        numpy.absolute(2 * self.left_fit_cr[0])

        right_curverad = (1 + (2 * self.right_fit_cr[0] * 720 * ym_per_pix + self.right_fit_cr[1]) ** 2) ** 1.5 / \
                         numpy.absolute(2 * self.right_fit_cr[0])

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(undistorted, str(left_curverad) + ', ' + str(right_curverad),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        return cv2.addWeighted(undistorted, 1, unwarped_lines, 0.5, 0)

        # return perspective_transform(undistorted, perspective_tr_matrix), self.left_fit, self.right_fit

        # return numpy.dstack((warped_binary, warped_binary, warped_binary)) * 255

    def _get_thresholded(self, image):
        # yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        # yellow_color = cv2.inRange(yuv[:, :, 1], array([0]), array([115]))
        # yellow_color[yellow_color != 0] = 1
        #
        # hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # white_color = cv2.inRange(hls, array([0, 200, 0]), array([255, 255, 255]))
        # white_color[white_color != 0] = 1
        #
        # out = numpy.zeros_like(white_color)
        # out[(white_color != 0) | (yellow_color != 0)] = 1
        # return out
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        result = self._binary_model.predict(image.reshape(1, *image.shape)).squeeze() * 255
        result = result.astype(numpy.uint8)
        result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
        return result / 255

    def _detect_lines(self, warped_binary):
        out = numpy.dstack((warped_binary, warped_binary, warped_binary)) * 255
        histogram = numpy.sum(warped_binary[warped_binary.shape[0] // 2:, :], axis=0)

        midpoint = numpy.int(histogram.shape[0] / 2)
        leftx_base = numpy.argmax(histogram[:midpoint])
        rightx_base = numpy.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = numpy.int(warped_binary.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_binary.nonzero()
        nonzeroy = array(nonzero[0])
        nonzerox = array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 60
        # Set minimum number of pixels found to recenter window
        minpix = 30
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped_binary.shape[0] - (window + 1) * window_height
            win_y_high = warped_binary.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            # cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            # cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = numpy.int(numpy.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = numpy.int(numpy.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = numpy.concatenate(left_lane_inds)
        right_lane_inds = numpy.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = numpy.polyfit(lefty, leftx, 2)
        right_fit = numpy.polyfit(righty, rightx, 2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700

        self.left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        self.right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        ploty = numpy.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.closest_line_difference = right_fitx[-1] - left_fitx[-1]
        self.start = 0

        # out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        #
        # leftfitx = left_fitx.astype(numpy.int32)
        # rightfix = right_fitx.astype(numpy.int32)
        # ycoord = ploty.astype(numpy.int32)
        #
        # output = numpy.zeros_like(out)
        #
        # cv2.polylines(output, array(list(zip(leftfitx, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=25)
        # cv2.polylines(output, array(list(zip(rightfix, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=25)
        # return output

        warp_zero = numpy.zeros_like(warped_binary).astype(numpy.uint8)
        color_warp = numpy.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = numpy.array([numpy.transpose(numpy.vstack([left_fitx, ploty]))])
        pts_right = numpy.array([numpy.flipud(numpy.transpose(numpy.vstack([right_fitx, ploty])))])
        pts = numpy.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, numpy.int_([pts]), (0, 255, 0))
        return color_warp

    def _continue_from_last(self, warped_binary):
        nonzero = warped_binary.nonzero()
        nonzeroy = numpy.array(nonzero[0])
        nonzerox = numpy.array(nonzero[1])
        margin = 60

        old_leftx = self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2]
        left_lane_inds = ((nonzerox > (old_leftx - margin)) & (nonzerox < (old_leftx + margin)))

        old_rightx = self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2]
        right_lane_inds = ((nonzerox > (old_rightx - margin)) & (nonzerox < (old_rightx + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = numpy.polyfit(lefty, leftx, 2) if len(leftx) > 0 else self.left_fit
        right_fit = numpy.polyfit(righty, rightx, 2) if len(rightx) > 0 else self.right_fit

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700

        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2) if len(leftx) > 0 else self.left_fit_cr
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2) if len(rightx) > 0 else self.right_fit_cr

        ploty = numpy.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
        # left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        # right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        closest_point_difference = right_fitx[-1] - left_fitx[-1]
        if closest_point_difference > 0:
            differences = right_fitx - left_fitx
            acceptable = differences > 0.7 * closest_point_difference
            start = numpy.argmax(acceptable)
            self.start += int(0.1 * (start - self.start))
            if right_fit[2] - left_fit[2] > 350:
                self._update_curves(left_fit, right_fit)
                self.left_fit_cr += 0.2 * (left_fit_cr - self.left_fit_cr)
                self.right_fit_cr += 0.2 * (right_fit_cr - self.right_fit_cr)

            ploty = ploty[self.start:]

        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # out = numpy.dstack((warped_binary, warped_binary, warped_binary)) * 255
        # out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        #
        # leftfitx = left_fitx.astype(numpy.int32)
        # rightfix = right_fitx.astype(numpy.int32)
        # ycoord = ploty.astype(numpy.int32)
        #
        # output = numpy.zeros_like(out)
        #
        # cv2.polylines(output, array(list(zip(leftfitx, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=25)
        # cv2.polylines(output, array(list(zip(rightfix, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=25)
        # return output

        warp_zero = numpy.zeros_like(warped_binary).astype(numpy.uint8)
        color_warp = numpy.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = numpy.array([numpy.transpose(numpy.vstack([left_fitx, ploty]))])
        pts_right = numpy.array([numpy.flipud(numpy.transpose(numpy.vstack([right_fitx, ploty])))])
        pts = numpy.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, numpy.int_([pts]), (0, 255, 0))
        return color_warp

    def _cnn(self, warped):
        window_width = 50
        window_height = 120  # Break image into 9 vertical layers since image height is 720
        margin = 60  # How much to slide left and right for searching
        warped *= 255

        def window_mask(width, height, img_ref, center, level):
            output = numpy.zeros_like(img_ref)
            output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
            max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
            return output

        def find_window_centroids(window_width, window_height, margin):
            window_centroids = []  # Store the (left,right) window centroid positions per level
            window = numpy.ones(window_width)  # Create our window template that we will use for convolutions

            # First find the two starting positions for the left and right lane by using numpy.sum to get the vertical image slice
            # and then numpy.convolve the vertical image slice with the window template

            # Sum quarter left? bottom of image to get slice, could use a different ratio
            l_sum = numpy.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
            # l_center = numpy.argmax(numpy.convolve(window, l_sum)) - window_width / 2
            l_center = numpy.argmax(
                cv2.filter2D(l_sum, cv2.CV_64F, window, borderType=cv2.BORDER_CONSTANT)) - window_width / 2

            r_sum = numpy.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
            r_center = numpy.argmax(numpy.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

            # Add what we found for the first layer
            window_centroids.append((l_center, r_center))

            # Go through each layer looking for max pixel locations
            for level in range(1, int(warped.shape[0] / window_height)):
                # convolve the window into the vertical slice of the image
                image_layer = numpy.sum(
                    warped[int(warped.shape[0] - (level + 1) * window_height):
                    int(warped.shape[0] - level * window_height), :], axis=0)

                conv_signal = numpy.convolve(window, image_layer)
                # Find the best left centroid by using past left center as a reference
                # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
                offset = window_width / 2
                l_min_index = int(max(l_center + offset - margin, 0))
                l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
                l_center = numpy.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
                # Find the best right centroid by using past right center as a reference
                r_min_index = int(max(r_center + offset - margin, 0))
                r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
                r_center = numpy.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
                # Add what we found for that layer
                window_centroids.append((l_center, r_center))

            return window_centroids

        # If we found any window centers
        window_centroids = find_window_centroids(window_width, window_height, margin)

        if len(window_centroids) > 0:
            # Points used to draw all the left and right windows
            l_points = numpy.zeros_like(warped)
            r_points = numpy.zeros_like(warped)

            # Go through each level and draw the windows
            for level in range(0, len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
                r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

            # Draw the results
            template = array(r_points + l_points, numpy.uint8)  # add both left and right window pixels together
            zero_channel = numpy.zeros_like(template)  # create a zero color channel

            # make window pixels green
            template = array(cv2.merge((zero_channel, template, zero_channel)), numpy.uint8)

            # making the original road pixels 3 color channels
            warpage = array(cv2.merge((warped, warped, warped)), numpy.uint8)
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0)  # overlay the orignal road image with window results

        # If no window centers found, just display orginal road image
        else:
            output = numpy.array(cv2.merge((warped, warped, warped)), numpy.uint8)

        # Display the final results
        # plt.imshow(output)
        # plt.title('window fitting results')
        # plt.show()
        return output

    def _update_curves(self, left_fit, right_fit):
        alpha = 0.2
        self.left_fit += alpha * (left_fit - self.left_fit)
        self.right_fit += alpha * (right_fit - self.right_fit)


def perspective_transform(image, matrix):
    return cv2.warpPerspective(image, matrix, (columns, rows), flags=cv2.INTER_LINEAR)


if __name__ == '__main__':
    video_files = ['project_video.mp4', 'harder_challenge_video.mp4', 'challenge_video.mp4']
    for video in video_files:
        process_and_save_video(video, os.path.join('output_videos', 'cnn-' + video), Pipeline())
