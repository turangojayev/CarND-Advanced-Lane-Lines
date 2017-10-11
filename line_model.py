from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from sklearn.utils import shuffle

from solution import *

uniform = numpy.random.uniform


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
    x = Conv2D(1, (3, 3), activation='sigmoid')(x)
    x = Upsampling(target_size=(720, 1280))(x)
    model = Model(img_input, x)
    return model


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
    hlsed[:, :, 1][to_be_shadowed] = hlsed[:, :, 1][to_be_shadowed] * numpy.random.uniform(0.2, 2.0)
    hlsed[:, :, 1][hlsed[:, :, 1] > 255] = 255
    return cv2.cvtColor(hlsed, cv2.COLOR_HLS2RGB)


def yes():
    return uniform() > 0.5


def generate_from(inputs, outputs, half_batch_size=32):
    while True:
        inputs, outputs = shuffle(inputs, outputs)
        for offset in range(0, len(inputs), half_batch_size):
            batch_inputs, batch_outputs = inputs[offset:offset + half_batch_size], \
                                          outputs[offset:offset + half_batch_size]

            images = array([cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in batch_inputs])
            output_images = array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255 for path in batch_outputs])
            output_images = array([image.reshape(*image.shape, 1) for image in output_images])
            images = array([shadow(image) if yes() else image for image in images])
            yield shuffle(images, output_images)


if __name__ == '__main__':
    camera_matrix, distortion_coefs = get_calibration_results()
    model = build_model((720, 1280, 3))
    model.compile(optimizer=optimizers.adam(0.001), loss='binary_crossentropy')
    model.summary()

    input_image_names = glob.glob('line_detection_data/input/*.jpg')
    output_image_names = glob.glob('line_detection_data/output/*.jpg')

    model_name = 'model.h5'

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