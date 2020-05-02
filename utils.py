"""
This is utility file containing all the helper functions required to run the 
model.
The idea for this file was somewhat inspired by the previous work done on BehaviouralCloning
https://github.com/ManajitPal/BehavioralCloning
"""

# First Let's import all the necessary libraries
import cv2
import os
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
# Input Shape of our keras model
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def preprocess(image):
    # Crop the Image to remove Sky
    image = image[60:-25, :, :]
    # Resize Image to Match our Input Shape
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    return image


def load_image(data_dir, image_file):
    return cv2.imread(os.path.join(data_dir, image_file.strip()))


def choose_image(data_dir, center, left, right, steering_angle):
    """
    We have Three sets of Images in our dataset. For Training purpose we are goin
    to randomly choose one type of image and will adjust the steering angle according to that.
    """
    rand = np.random.choice(3)
    if rand == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif rand == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    else:
        return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    if np.random.rand() < 0.6:
        image = cv2.flip(image, 1)
        steering_angle = - steering_angle
    return image, steering_angle


def augment(data_dir, center, left, right, steering_angle):
    image, steering_angle = choose_image(
        data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    return image, steering_angle

# Now it's time to write the final function for this file A.K.A. Our batch generator
# function that will generate training data parallely using CPU while our model is
# training on GPU


def batch_generator(data_dir, images_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(images_paths.shape[0]):
            center, left, right = images_paths[index]
            steering_angle = steering_angles[index]

            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(
                    data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
