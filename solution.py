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
RGB = ['Red', 'Green', 'Blue']
array = numpy.array


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
    height, width, *_ = image.shape

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


def apply_hls_mask(image):
    hlsed = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    mask = cv2.inRange(hlsed, array([10, 0, 100]), array([40, 255, 255]))
    return cv2.bitwise_and(hlsed, hlsed, mask=mask)


undistort = cv2.undistort


# partial(
#     cameraMatrix=camera_matrix,
#     distCoeffs=distortion_coefs,
#     dst=None,
#     newCameraMatrix=None)


# def pipeline(image):
#     undistorted = undistort(image, camera_matrix, distortion_coefs, None, None)
#     color_masked = apply_hls_mask(undistorted)
#     return color_masked

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


def pipeline(image):
    # print(image.shape)
    image = undistort(image, camera_matrix, distortion_coefs, None, None)
    # image = contrast_normalization(image)
    # plt.imshow(image, cmap='gray')
    # plt.show()
    s_channel_threshold = (170, 255)
    x_gradient_threshold = (20, 100)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # binary_s_channel = numpy.zeros_like(s_channel)
    # binary_s_channel[(s_channel >= s_channel_threshold[0]) & (s_channel <= s_channel_threshold[1])] = 1
    # binary_s_channel = cv2.inRange(hls, array([10, 0, 100]), array([40, 255, 255]))
    binary_s_channel = cv2.inRange(hls, array([10, 0, 40]), array([45, 255, 255]))
    binary_s_channel[binary_s_channel != 0] = 1

    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    x_gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    absolute_x_gradient = numpy.absolute(x_gradient)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_abs_x_grad = numpy.uint8(255 * absolute_x_gradient / numpy.max(absolute_x_gradient))

    # Threshold x gradient
    binary_gradient = numpy.zeros_like(scaled_abs_x_grad)
    binary_gradient[(scaled_abs_x_grad >= x_gradient_threshold[0]) & (scaled_abs_x_grad <= x_gradient_threshold[1])] = 1

    # color_binary = numpy.dstack((numpy.zeros_like(binary_gradient), binary_gradient, binary_s_channel)) * 255
    color_binary = numpy.dstack(
        (numpy.zeros_like(binary_gradient), numpy.zeros_like(binary_gradient), binary_s_channel)) * 255

    combined_binary = numpy.zeros_like(binary_gradient)
    # combined_binary[(binary_s_channel == 1) | (binary_gradient == 1)] = 1
    combined_binary[(binary_s_channel == 1)] = 1

    # Plotting thresholded images
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.set_title('Stacked thresholds')
    # ax1.imshow(color_binary)
    #
    # ax2.set_title('Combined S channel and gradient thresholds')
    # ax2.imshow(combined_binary, cmap='gray')
    # plt.show()
    return color_binary


KIRSCH_K1 = array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=numpy.float32) / 15
KIRSCH_K2 = array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=numpy.float32) / 15
KIRSCH_K3 = array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=numpy.float32) / 15
KIRSCH_K4 = array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=numpy.float32) / 15
KIRSCH_K5 = array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=numpy.float32) / 15
KIRSCH_K6 = array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=numpy.float32) / 15
KIRSCH_K7 = array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=numpy.float32) / 15
KIRSCH_K8 = array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=numpy.float32) / 15


def kirsch_filter(img):
    fimg = numpy.maximum(
        cv2.filter2D(img, cv2.CV_64F, KIRSCH_K1),
        numpy.maximum(
            cv2.filter2D(img, cv2.CV_64F, KIRSCH_K2),
            numpy.maximum(
                cv2.filter2D(img, cv2.CV_64F, KIRSCH_K3),
                numpy.maximum(
                    cv2.filter2D(img, cv2.CV_64F, KIRSCH_K4),
                    numpy.maximum(
                        cv2.filter2D(img, cv2.CV_64F, KIRSCH_K5),
                        numpy.maximum(
                            cv2.filter2D(img, cv2.CV_64F, KIRSCH_K6),
                            numpy.maximum(
                                cv2.filter2D(img, cv2.CV_64F, KIRSCH_K7),
                                cv2.filter2D(img, cv2.CV_64F, KIRSCH_K8),
                            )))))))

    return (fimg)


def kirsch(image):
    scaled = kirsch_filter(image)
    return numpy.uint8(255 * scaled / numpy.max(scaled))


def gradient(image, low, high):
    x_gradient = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=15)
    absolute_x_gradient = numpy.absolute(x_gradient)
    scaled_abs_x_grad = numpy.uint8(255 * absolute_x_gradient / numpy.max(absolute_x_gradient))
    binary = cv2.inRange(scaled_abs_x_grad, low, high)
    return binary


def pipeline2(image):
    image = undistort(image, camera_matrix, distortion_coefs, None, None)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yellow_line = cv2.inRange(yuv[:, :, 1], array([0]), array([115]))
    yellow_line[yellow_line != 0] = 1

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    result = cv2.inRange(image, array([0, 200, 0]), array([255, 255, 255]))
    result[result != 0] = 1
    #
    # gradients = gradient(image[:, :, 1], 70, 180)
    # result = cv2.blur(gradients, (3, 9))
    # result = cv2.inRange(kirsch(result), 30, 100)
    # result[result != 0] = 1
    # # return numpy.dstack((numpy.zeros_like(result), yellow_line, result)) * 255
    # lines = numpy.dstack((numpy.zeros_like(result), yellow_line, result))
    # warped = perspective_transform(lines)

    #######

    out = numpy.zeros_like(result)
    out[(result != 0) | (yellow_line != 0)] = 1

    warped = perspective_transform(out)
    out = numpy.dstack((warped, warped, warped)) * 255
    # histogram = numpy.sum(warped[warped.shape[0] // 2:, :], axis=0)
    # # histogram = numpy.sum(warped, axis=0)
    #
    # midpoint = numpy.int(histogram.shape[0] / 2)
    # leftx_base = numpy.argmax(histogram[:midpoint])
    # rightx_base = numpy.argmax(histogram[midpoint:]) + midpoint
    #
    # # Choose the number of sliding windows
    # nwindows = 9
    # # Set height of windows
    # window_height = numpy.int(warped.shape[0] / nwindows)
    # # Identify the x and y positions of all nonzero pixels in the image
    # nonzero = warped.nonzero()
    # nonzeroy = array(nonzero[0])
    # nonzerox = array(nonzero[1])
    # # Current positions to be updated for each window
    # leftx_current = leftx_base
    # rightx_current = rightx_base
    # # Set the width of the windows +/- margin
    # margin = 60
    # # Set minimum number of pixels found to recenter window
    # minpix = 30
    # # Create empty lists to receive left and right lane pixel indices
    # left_lane_inds = []
    # right_lane_inds = []
    #
    # # Step through the windows one by one
    # for window in range(nwindows):
    #     # Identify window boundaries in x and y (and right and left)
    #     win_y_low = warped.shape[0] - (window + 1) * window_height
    #     win_y_high = warped.shape[0] - window * window_height
    #     win_xleft_low = leftx_current - margin
    #     win_xleft_high = leftx_current + margin
    #     win_xright_low = rightx_current - margin
    #     win_xright_high = rightx_current + margin
    #     # Draw the windows on the visualization image
    #     cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    #     cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    #     # Identify the nonzero pixels in x and y within the window
    #     good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
    #                       (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    #     good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
    #                        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    #     # Append these indices to the lists
    #     left_lane_inds.append(good_left_inds)
    #     right_lane_inds.append(good_right_inds)
    #     # If you found > minpix pixels, recenter next window on their mean position
    #     if len(good_left_inds) > minpix:
    #         leftx_current = numpy.int(numpy.mean(nonzerox[good_left_inds]))
    #     if len(good_right_inds) > minpix:
    #         rightx_current = numpy.int(numpy.mean(nonzerox[good_right_inds]))
    #
    # # Concatenate the arrays of indices
    # left_lane_inds = numpy.concatenate(left_lane_inds)
    # right_lane_inds = numpy.concatenate(right_lane_inds)
    #
    # # Extract left and right line pixel positions
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds]
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds]
    #
    # # Fit a second order polynomial to each
    # left_fit = numpy.polyfit(lefty, leftx, 2)
    # right_fit = numpy.polyfit(righty, rightx, 2)
    #
    # ploty = numpy.linspace(0, warped.shape[0] - 1, warped.shape[0])
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #
    # out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()


    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 60  # How much to slide left and right for searching
    warped *= 255

    def window_mask(width, height, img_ref, center, level):
        output = numpy.zeros_like(img_ref)
        output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
        max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
        return output

    def find_window_centroids(image, window_width, window_height, margin):
        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = numpy.ones(window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using numpy.sum to get the vertical image slice
        # and then numpy.convolve the vertical image slice with the window template

        # Sum quarter left? bottom of image to get slice, could use a different ratio
        l_sum = numpy.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = numpy.argmax(numpy.convolve(window, l_sum)) - window_width / 2
        # print(l_center)
        # l_center = numpy.argmax(cv2.filter2D(l_sum, cv2.CV_64F, window, borderType=cv2.BORDER_CONSTANT)) - window_width / 2

        r_sum = numpy.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = numpy.argmax(numpy.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, int(warped.shape[0] / window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = numpy.sum(
                warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
                :], axis=0)
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

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
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
        template = array(cv2.merge((zero_channel, template, zero_channel)),
                         numpy.uint8)  # make window pixels green
        warpage = array(cv2.merge((warped, warped, warped)),
                        numpy.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = numpy.array(cv2.merge((warped, warped, warped)), numpy.uint8)

    # Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()

    return out


def continue_from_last():
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = numpy.array(nonzero[0])
    nonzerox = numpy.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = numpy.polyfit(lefty, leftx, 2)
    right_fit = numpy.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = numpy.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


rows = 720
columns = 1280
# src = numpy.float32([[100, 720], [609, 437], [666, 437], [1200, 720]])
# src = numpy.float32([[0, rows], [570, 453], [710, 453], [columns, rows]])
src = numpy.float32([[0, rows], [568, 453], [710, 453], [columns, rows]])
# src = numpy.float32([[0, 700], [568, 453], [710, 453], [columns, 700]])
dst = numpy.float32([[0, rows], [0, 0], [columns, 0], [columns, rows]])
# dst = numpy.float32([[100, 710], [100, 10], [1180, 10], [columns - 100, rows - 10]])

src = numpy.float32([[0, 700],
                     [515, 472],
                     [764, 472.],
                     [1280, 700.]])

dst = numpy.float32([[100, 710],
                     [100, 10],
                     [1180, 10],
                     [1180, 710]])

perspective_tr_matrix = cv2.getPerspectiveTransform(src, dst)


def perspective_transform(image):
    return cv2.warpPerspective(image, perspective_tr_matrix, (columns, rows), flags=cv2.INTER_LINEAR)


def setCurrentImage(image):
    w, h = image.shape[1], image.shape[0]
    bottomW = w
    topW = 249  # 235 ((1180+100)*(180/1180))
    bottomH = h - 20
    topH = bottomH - 228  # h//2 + 100
    deltaW = 0  #

    region_vertices = array([[((w - bottomW) // 2 + deltaW, bottomH),
                              ((w - topW) // 2 + deltaW, topH),
                              ((w + topW) // 2 + deltaW, topH),
                              ((w + bottomW) // 2 + deltaW, bottomH)]], dtype=numpy.float32)
    print(region_vertices)

    offsetH = 10
    offsetW = 100
    dest_vertices = array([[(offsetW, h - offsetH),
                            (offsetW, offsetH),
                            (w - offsetW, offsetH),
                            (w - offsetW, h - offsetH)]], dtype=numpy.float32)
    print(dest_vertices)

    return cv2.getPerspectiveTransform(region_vertices, dest_vertices)


if __name__ == '__main__':
    # video_files = ['harder_challenge_video.mp4', 'challenge_video.mp4', 'project_video.mp4']
    video_files = ['project_video.mp4', 'harder_challenge_video.mp4', 'challenge_video.mp4']
    camera_matrix, distortion_coefs = get_calibration_results()

    # for video in video_files:
    #     process_and_save_video(video, os.path.join('output_videos', video), pipeline2)


    image_names = glob.glob('test_images/straight*')
    image_names.extend(glob.glob('test_images/test*'))
    image_names.extend(glob.glob('test_images/challenge*'))
    image_names.extend(glob.glob('test_images/harder*'))

    images = list(map(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), image_names))
    images = list(map(pipeline2, images))
    plot(images)

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
