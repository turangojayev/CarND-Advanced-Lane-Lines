import glob
from functools import partial

import cv2
import os
import numpy
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
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


class Pipeline(object):
    def __init__(self):
        self.camera_matrix, self.distortion_coefs = get_calibration_results()

    def __call__(self, image, **kwargs):
        undistorted = undistort(image, self.camera_matrix, self.distortion_coefs, None, None)
        yuv = cv2.cvtColor(undistorted, cv2.COLOR_RGB2YUV)
        yellow_color = cv2.inRange(yuv[:, :, 1], array([0]), array([115]))
        yellow_color[yellow_color != 0] = 1

        hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
        white_color = cv2.inRange(hls, array([0, 200, 0]), array([255, 255, 255]))
        white_color[white_color != 0] = 1

        out = numpy.zeros_like(white_color)
        out[(white_color != 0) | (yellow_color != 0)] = 1

        warped = perspective_transform(out, perspective_tr_matrix)

        if hasattr(self, 'left_fit'):
            lines_drawn = self._continue_from_last(warped)
        else:
            lines_drawn = self._detect_lines(warped)
        unwarped_lines = perspective_transform(lines_drawn, inverse_perspective_tr_matrix)
        return cv2.addWeighted(undistorted, 1, unwarped_lines, 0.5, 0)
        # return perspective_transform(undistorted, perspective_tr_matrix)

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
        margin = 100
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

        ploty = numpy.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        leftfitx = left_fitx.astype(numpy.int32)
        rightfix = right_fitx.astype(numpy.int32)
        ycoord = ploty.astype(numpy.int32)

        # cv2.polylines(out, array(list(zip(leftfitx, ycoord))).reshape(-1, 1, 2), True, (200, 200, 200), thickness=5)
        # cv2.polylines(out, array(list(zip(rightfix, ycoord))).reshape(-1, 1, 2), True, (200, 200, 200), thickness=5)

        output = numpy.zeros_like(out)

        cv2.polylines(output, array(list(zip(leftfitx, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=25)
        cv2.polylines(output, array(list(zip(rightfix, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=25)
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
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy +
                                       self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0] * (nonzeroy ** 2) +
                                                                                  self.left_fit[1] * nonzeroy +
                                                                                  self.left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy +
                                        self.right_fit[2] - margin)) & (
                               nonzerox < (self.right_fit[0] * (nonzeroy ** 2) +
                                           self.right_fit[1] * nonzeroy +
                                           self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = numpy.polyfit(lefty, leftx, 2)
        right_fit = numpy.polyfit(righty, rightx, 2)
        self._update(left_fit, right_fit)

        # Generate x and y values for plotting
        ploty = numpy.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        out = numpy.dstack((warped_binary, warped_binary, warped_binary)) * 255
        out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        leftfitx = left_fitx.astype(numpy.int32)
        rightfix = right_fitx.astype(numpy.int32)
        ycoord = ploty.astype(numpy.int32)

        # cv2.polylines(out, array(list(zip(leftfitx, ycoord))).reshape(-1, 1, 2), True, (200, 200, 200), thickness=5)
        # cv2.polylines(out, array(list(zip(rightfix, ycoord))).reshape(-1, 1, 2), True, (200, 200, 200), thickness=5)
        output = numpy.zeros_like(out)

        cv2.polylines(output, array(list(zip(leftfitx, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=25)
        cv2.polylines(output, array(list(zip(rightfix, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=25)
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

    def _update(self, left_fit, right_fit):
        alpha = 0.4
        self.left_fit += alpha * (left_fit - self.left_fit)
        self.right_fit += alpha * (right_fit - self.right_fit)


def perspective_transform(image, matrix):
    return cv2.warpPerspective(image, matrix, (columns, rows), flags=cv2.INTER_LINEAR)


if __name__ == '__main__':
    video_files = ['project_video.mp4', 'harder_challenge_video.mp4', 'challenge_video.mp4']
    # video_files = ['challenge_video.mp4']
    # video_files = ['challenge_video.mp4']

    for video in video_files:
        process_and_save_video(video, os.path.join('output_videos', video), Pipeline())


        # image_names = glob.glob('test_images/straight*')
        # # image_names.extend(glob.glob('test_images/test*'))
        # images = list(map(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), image_names))
        # images = list(map(Pipeline(), images))
        # plt.imshow(images[1])
        # plt.show()

        # plot(images)
        # trapes = array([[205, 720], [603, 446], [677, 446], [1100, 720]])
        #
        # undistorted = undistort(images[6], camera_matrix, distortion_coefs, None, None)
        # plt.imshow(undistorted)
        # plt.plot(trapes[:, 0], trapes[:, 1], marker='o')
        # plt.show()
        # # plt.imshow(perspective_transform(undistorted))
        # # plt.imshow(perspective_transform(undistorted))
        # plt.imshow(cv2.warpPerspective(undistorted, setCurrentImage(undistorted), (columns, rows), flags=cv2.INTER_LINEAR))
        # # transofmed = perspective_transform(trapes)
        # # plt.plot(transofmed[:, 0], transofmed[:, 1], marker='o')
        # plt.show()
        # images = list(map(lambda x: cv2.imread(x), image_names))
        # plot(images, columns=4, channel=1, cmap='gray')
