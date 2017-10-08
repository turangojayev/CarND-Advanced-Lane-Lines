from operator import itemgetter

import cv2
import numpy
import tensorflow as tf
from keras import losses
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import *
from sklearn.utils import shuffle

from solution import *


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
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
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
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)


class Upsampling(Layer):
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
        super(Upsampling, self).__init__(**kwargs)

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
        base_config = super(Upsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def model1(input_shape=None):
    img_input = Input(shape=input_shape)
    x = Lambda(lambda x: (x - 127.) / 128)(img_input)
    # image_size = input_shape[0:2]
    x = Conv2D(32, (3, 3), activation='elu')(x)
    x = MaxPooling2D((3, 3), (3, 3))(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Dropout(0.7)(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)
    # x = Conv2D(192, (3, 3), activation='relu')(x)
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = Upsampling(target_size=(720, 1280))(x)
    model = Model(img_input, x)

    return model


def model2(input_shape=None):
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
    # x = Conv2D(192, (3, 3), activation='relu')(x)
    x = Conv2D(1, (3, 3), activation='sigmoid')(x)
    x = Upsampling(target_size=(720, 1280))(x)
    model = Model(img_input, x)

    return model


def pipeline3(image):
    image = cv2.undistort(image, camera_matrix, distortion_coefs, None, None)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yellow_line = cv2.inRange(yuv[:, :, 1], array([0]), array([115]))
    yellow_line[yellow_line != 0] = 1

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    result = cv2.inRange(image, array([0, 200, 0]), array([255, 255, 255]))
    result[result != 0] = 1

    out = numpy.zeros_like(result)
    out[(result != 0) | (yellow_line != 0)] = 1

    warped = perspective_transform(out)
    return warped.reshape(720, 1280, 1)


def get_data():
    pass


uniform = numpy.random.uniform



def shadow(image):
    rows = image.shape[0]
    cols = image.shape[1]

    y1, y2 = cols * uniform(size=2)
    x1, x2 = rows * uniform(size=2)
    hlsed = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = numpy.zeros(shape=(rows, cols))
    x_mesh, y_mesh = numpy.mgrid[0:rows, 0:cols]

    shadow_mask[((x_mesh - x1) * (y2 - y1) - (x2 - x1) * (y_mesh - y1) >= 0)] = 1

    to_be_shadowed = shadow_mask == 1
    # hlsed[:, :, 1][to_be_shadowed] = hlsed[:, :, 1][to_be_shadowed] * numpy.random.uniform(0.2, 0.6)
    hlsed[:, :, 1][to_be_shadowed] = hlsed[:, :, 1][to_be_shadowed] * numpy.random.uniform(0.2, 2.0)
    hlsed[:, :, 1][hlsed[:, :, 1] > 255] = 255
    return cv2.cvtColor(hlsed, cv2.COLOR_HLS2RGB)


def yes():
    return uniform() > 0.5


def generate_from(inputs, outputs, half_batch_size=32):
    # half_batch_size = int(batch_size / 2)
    while True:
        inputs, outputs = shuffle(inputs, outputs)
        for offset in range(0, len(inputs), half_batch_size):
            batch_inputs, batch_outputs = inputs[offset:offset + half_batch_size], \
                                          outputs[offset:offset + half_batch_size]

            images = array([cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in batch_inputs])
            output_images = array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255 for path in batch_outputs])
            output_images = array([image.reshape(*image.shape, 1) for image in output_images])

            # image2angle = [translate(img, angle, 40) for img, angle in zip(images, batch_outputs)]
            # images = list(map(itemgetter(0), image2angle))
            # batch_outputs = numpy.array(list(map(itemgetter(1), image2angle)))
            images = array([shadow(image) if yes() else image for image in images])
            # images = numpy.vstack((images, numpy.array(list(map(partial(cv2.flip, flipCode=1), images)))))
            # batch_outputs = numpy.hstack((batch_outputs, -batch_outputs))
            yield shuffle(images, output_images)


if __name__ == '__main__':
    camera_matrix, distortion_coefs = get_calibration_results()
    model = model2((720, 1280, 3))
    model.compile(optimizer=optimizers.adam(0.001), loss='binary_crossentropy')
    model.summary()

    input_image_names = glob.glob('line_detection_data/input/*.jpg')
    output_image_names = glob.glob('line_detection_data/output2/*.jpg')

    model_name = 'model4.h5'

    # checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)

    batch_size = 8
    validation_size = 200
    train_generator = generate_from(input_image_names[:-validation_size], output_image_names[:-validation_size],
                                    batch_size)

    valid_generator = generate_from(input_image_names[-validation_size:], output_image_names[-validation_size:],
                                    batch_size)
    model.fit_generator(train_generator,
                        steps_per_epoch=len(input_image_names[:-validation_size]) / batch_size,
                        epochs=20,
                        callbacks=[checkpoint],
                        validation_data=valid_generator,
                        validation_steps=validation_size / batch_size)

    model = keras.models.load_model(model_name, custom_objects={'Upsampling': Upsampling})

    test_images = glob.glob('test_images/challenge*')
    test_images.extend(glob.glob('test_images/harder*'))
    print(test_images[1])
    test_images = list(map(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), test_images))
    undistorted = list(map(lambda x: undistort(x, camera_matrix, distortion_coefs, None, None), test_images))
    test_images = list(map(lambda x: perspective_transform(x, perspective_tr_matrix), undistorted))

    # plt.imshow(test_images[1], cmap='gray')
    # plt.show()

    results = model.predict(array(test_images))
    # results = model.predict(next(train_generator)[0])
    # results = model.predict(array(X))

    for i in range(len(test_images)):
        out = results[i].reshape(720, 1280)
        # out[out != 0] = 255
        plt.imshow(out, cmap='gray')
        plt.show()
