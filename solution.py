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
polyfit = numpy.polyfit


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


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


class Lines:
    def __init__(self, margin=60, minpix=30, num_windows=9):
        '''
        :param margin:
        :param minpix: minimum number of pixels found to recenter window
        :param num_windows:
        '''
        self._margin = margin
        self._minpix = minpix
        self._num_windows = num_windows
        self._image_shape = None

    def get_lines(self, binary_image):
        nonzeroy, nonzerox = binary_image.nonzero()
        if self._image_shape is None:
            self._image_shape = binary_image.shape

        left_indices, right_indices = self._find_indices(binary_image, nonzerox, nonzeroy)
        y, left_fit, right_fit = self._fit_and_update(nonzerox[left_indices], nonzeroy[left_indices],
                                                      nonzerox[right_indices], nonzeroy[right_indices])

        return self._prepare_output(y, left_fit, right_fit)

    def _fit_and_update(self, leftx, lefty, rightx, righty):
        # TODO: actually not correct, if nothing found at first it will fail
        left_coeffs = polyfit(lefty, leftx, 2) if len(leftx) > 0 else self._left_coeffs
        right_coeffs = polyfit(righty, rightx, 2) if len(rightx) > 0 else self._right_coeffs

        y = numpy.linspace(0, self._image_shape[0] - 1, self._image_shape[0])
        left_fit = left_coeffs[0] * y ** 2 + left_coeffs[1] * y + left_coeffs[2]
        right_fit = right_coeffs[0] * y ** 2 + right_coeffs[1] * y + right_coeffs[2]

        if not hasattr(self, '_left_coeffs'):  # should actually check both, this is just current impl.
            self._left_coeffs = left_coeffs
            self._right_coeffs = right_coeffs
            self.closest_line_difference = right_fit[-1] - left_fit[-1]
            self._start = 0

        closest_point_difference = right_fit[-1] - left_fit[-1]
        if closest_point_difference > 0:
            differences = right_fit - left_fit
            acceptable = differences > 0.7 * closest_point_difference
            start = numpy.argmax(acceptable)
            self._start += int(0.1 * (start - self._start))
            if right_fit[2] - left_fit[2] > 350:
                alpha = 0.2
                self._left_coeffs += alpha * (left_coeffs - self._left_coeffs)
                self._right_coeffs += alpha * (right_coeffs - self._right_coeffs)

            y = y[self._start:]

        left_fit = self._left_coeffs[0] * y ** 2 + self._left_coeffs[1] * y + self._left_coeffs[2]
        right_fit = self._right_coeffs[0] * y ** 2 + self._right_coeffs[1] * y + self._right_coeffs[2]
        return y, left_fit, right_fit

    def _prepare_output(self, y, left_x, right_x):
        warp_zero = numpy.zeros(self._image_shape).astype(numpy.uint8)
        color_warp = numpy.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        left_points = numpy.array([numpy.transpose(numpy.vstack([left_x, y]))])
        right_points = numpy.array([numpy.flipud(numpy.transpose(numpy.vstack([right_x, y])))])
        pts = numpy.hstack((left_points, right_points))

        # Draw the lane onto the warped blank image
        return cv2.fillPoly(color_warp, numpy.int_([pts]), (0, 255, 0))

    def _find_indices(self, binary_image, nonzerox, nonzeroy):
        if not hasattr(self, '_left_coeffs'):
            print(self._image_shape)
            histogram = numpy.sum(binary_image[self._image_shape[0] // 2:, :], axis=0)
            midpoint = numpy.int(histogram.shape[0] / 2)
            self._leftx_current = numpy.argmax(histogram[:midpoint])
            self._rightx_current = numpy.argmax(histogram[midpoint:]) + midpoint

            window_height = numpy.int(self._image_shape[0] / self._num_windows)

            left_indices, right_indices = [], []

            for window in range(self._num_windows):
                left_within_window_indices, right_within_window_indices = self._process_window(nonzerox,
                                                                                               nonzeroy,
                                                                                               window,
                                                                                               window_height)
                left_indices.append(left_within_window_indices)
                right_indices.append(right_within_window_indices)

            left_indices = numpy.concatenate(left_indices)
            right_indices = numpy.concatenate(right_indices)
        else:
            old_leftx = self._left_coeffs[0] * (nonzeroy ** 2) + self._left_coeffs[1] * nonzeroy + self._left_coeffs[2]
            left_indices = ((nonzerox > (old_leftx - self._margin)) & (nonzerox < (old_leftx + self._margin)))

            old_rightx = self._right_coeffs[0] * (nonzeroy ** 2) + self._right_coeffs[1] * nonzeroy + \
                         self._right_coeffs[2]

            right_indices = ((nonzerox > (old_rightx - self._margin)) & (nonzerox < (old_rightx + self._margin)))
        return left_indices, right_indices

    def _process_window(self, nonzerox, nonzeroy, window, window_height):
        # lower and upper part of screen, origin is at top left of screen
        window_bottom = self._image_shape[0] - window * window_height
        window_top = window_bottom - window_height
        left_window_left = self._leftx_current - self._margin
        left_window_right = self._leftx_current + self._margin
        right_window_left = self._rightx_current - self._margin
        right_window_right = self._rightx_current + self._margin

        # Draw the windows on the visualization image
        # cv2.rectangle(out, (left_window_left, window_top), (left_window_right, window_bottom), (0, 255, 0), 2)
        # cv2.rectangle(out, (right_window_left, window_top), (right_window_right, window_bottom), (0, 255, 0), 2)
        within_vertical_boundaries = (nonzeroy >= window_top) & (nonzeroy < window_bottom)
        within_left_window_horizontal_boundaries = (nonzerox >= left_window_left) & (nonzerox < left_window_right)
        within_right_window_horizontal_boundaries = (nonzerox >= right_window_left) & (nonzerox < right_window_right)

        left_within_window_indices = (within_vertical_boundaries &
                                      within_left_window_horizontal_boundaries).nonzero()[0]

        right_within_window_indices = (within_vertical_boundaries &
                                       within_right_window_horizontal_boundaries).nonzero()[0]
        # Append these indices to the lists

        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_within_window_indices) > self._minpix:
            self._leftx_current = numpy.int(numpy.mean(nonzerox[left_within_window_indices]))
        if len(right_within_window_indices) > self._minpix:
            self._rightx_current = numpy.int(numpy.mean(nonzerox[right_within_window_indices]))

        return left_within_window_indices, right_within_window_indices


class Pipeline(object):
    def __init__(self):
        self.camera_matrix, self.distortion_coefs = get_calibration_results()
        self._binary_model = keras.models.load_model(
            'model.h5', custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D})
        self._lines = Lines()

    def __call__(self, image, **kwargs):
        undistorted = undistort(image, self.camera_matrix, self.distortion_coefs, None, None)
        warped = perspective_transform(undistorted, perspective_tr_matrix)
        warped_binary = self._get_thresholded(warped)
        lines_drawn = self._lines.get_lines(warped_binary)
        unwarped_lines = perspective_transform(lines_drawn, inverse_perspective_tr_matrix)
        return cv2.addWeighted(undistorted, 1, unwarped_lines, 0.5, 0)

    def _get_lines_drawn(self, warped_binary):
        if hasattr(self, 'left_fit'):
            return self._continue_from_last(warped_binary)
        else:
            return self._detect_lines(warped_binary)

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

        #
        result = self._binary_model.predict(image.reshape(1, *image.shape)).squeeze() * 255
        result = result.astype(numpy.uint8)
        # result[result > 128] = 255
        # result[result <= 128] = 0
        result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
        return result / 255

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


def perspective_transform(image, matrix):
    return cv2.warpPerspective(image, matrix, (columns, rows), flags=cv2.INTER_LINEAR)


if __name__ == '__main__':
    # video_files = ['project_video.mp4', 'harder_challenge_video.mp4', 'challenge_video.mp4']
    video_files = ['harder_challenge_video.mp4', 'challenge_video.mp4']
    for video in video_files:
        process_and_save_video(video, os.path.join('output_videos', 'cnn-' + video), Pipeline())
